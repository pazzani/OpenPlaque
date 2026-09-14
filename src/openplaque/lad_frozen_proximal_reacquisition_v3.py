from __future__ import annotations

"""Post-hoc final-path-tangent validation for frozen-LAD proximal reacquisition.

This is a same-experiment correction. The target-free proposal search is inherited
unchanged from v1.0 and the JPEG-lossless-safe loader from v1.1. After the raw
track is generated, the completed trajectory is resampled densely and re-evaluated
in source CCTA on planes perpendicular to the *actual local final-path tangent*.
The extension is automatically truncated at the first sustained rolling failure.
"""

import copy
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lad_frozen_proximal_reacquisition import (
    _json_write,
    arc_mm,
    lumen_metrics,
    orthogonal_plane,
    resample_path,
    score_metrics,
    source_zyx_to_lps,
    _unit,
)
from .lad_frozen_proximal_reacquisition_v2 import (
    LADFrozenProximalReacquisitionWorkflowV2,
)

ALGORITHM_VERSION = "lad-frozen-proximal-reacquisition-v1.2-final-tangent-validation"
POSTHOC_STEP_MM = 0.30
POSTHOC_RECENTER_LIMIT_MM = 0.80
ROLLING_WINDOW = 3
ROLLING_FAILS_REQUIRED = 2


def _sustained_failure_index(pass_flags, window=ROLLING_WINDOW, fails_required=ROLLING_FAILS_REQUIRED):
    """First failing index starting a short window with sustained failure.

    A point is considered the onset only when it itself fails and at least
    `fails_required` planes fail within the following `window` samples.
    """
    f = np.asarray(pass_flags, dtype=bool)
    n = len(f)
    for i in range(n):
        if f[i]:
            continue
        j = min(n, i + int(window))
        if int(np.sum(~f[i:j])) >= int(fails_required):
            return i
    # At the terminal edge, two consecutive failures are also treated as sustained.
    if n >= 2 and (not f[-2]) and (not f[-1]):
        return n - 2
    return None


def synthetic_posthoc_tangent_validation_self_test():
    flags = [True, True, True, False, True, False, False]
    idx = _sustained_failure_index(flags)
    flags2 = [True, True, False, True, True, True]
    idx2 = _sustained_failure_index(flags2)
    return {
        "passed": bool(idx == 3 and idx2 is None),
        "sustained_failure_index": idx,
        "isolated_failure_result": idx2,
    }


class LADFrozenProximalReacquisitionWorkflowV3(LADFrozenProximalReacquisitionWorkflowV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_track = None
        self.raw_track_qc = None
        self.raw_summary = None
        self.posthoc_qc = None
        self.posthoc_failure_index = None

    def _local_tangent(self, path_zyx, idx, half_window=3):
        p = np.asarray(path_zyx, float)
        a = max(0, int(idx) - int(half_window))
        b = min(len(p) - 1, int(idx) + int(half_window))
        if b <= a:
            return np.array([1.0, 0.0, 0.0])
        return _unit((p[b] - p[a]) * self.spacing)

    def _posthoc_plane_qc(self, point_zyx, tangent_phys_zyx):
        """Evaluate one final-path-tangent plane with fixed tangent and limited recenter."""
        p = np.asarray(point_zyx, float)
        t = _unit(tangent_phys_zyx)
        im, g, u, v = orthogonal_plane(self.ct, p, t, self.spacing)
        m0 = lumen_metrics(im, g)

        shift0 = float(m0.get("centroid_shift_mm", np.nan))
        recenter = 0.0
        p_eval = p.copy()
        m = m0

        if np.isfinite(shift0) and shift0 <= POSTHOC_RECENTER_LIMIT_MM:
            cu = float(m0.get("centroid_u_mm", np.nan))
            cv = float(m0.get("centroid_v_mm", np.nan))
            if np.isfinite(cu) and np.isfinite(cv):
                recenter = float(math.hypot(cu, cv))
                phys = p * self.spacing + cu * u + cv * v
                p_eval = phys / self.spacing
                im2, g2, _, _ = orthogonal_plane(self.ct, p_eval, t, self.spacing)
                m = lumen_metrics(im2, g2)

        score, hard = score_metrics(m, self.ref)
        # The final tangent is fixed. Recenter is allowed only inside the local plane.
        hard = bool(hard and np.isfinite(shift0) and shift0 <= POSTHOC_RECENTER_LIMIT_MM)
        return p_eval, m, float(score), hard, float(recenter), float(shift0)

    def _validate_final_path_tangent(self):
        raw = self.raw_track[["z", "y", "x"]].to_numpy(float)
        dense = resample_path(raw, self.spacing, POSTHOC_STEP_MM)
        s = arc_mm(dense, self.spacing)
        rows = []

        for i, p in enumerate(dense):
            t = self._local_tangent(dense, i, half_window=3)
            p_eval, m, score, hard, recenter, initial_shift = self._posthoc_plane_qc(p, t)
            ad, inside = self._aorta_state(p_eval)
            rows.append({
                "sample_index": int(i),
                "track_length_mm": float(s[i]),
                "z": float(p[0]), "y": float(p[1]), "x": float(p[2]),
                "eval_z": float(p_eval[0]), "eval_y": float(p_eval[1]), "eval_x": float(p_eval[2]),
                "radius_mm": float(m["radius_mm"]) if np.isfinite(m["radius_mm"]) else np.nan,
                "centroid_shift_mm": float(m["centroid_shift_mm"]) if np.isfinite(m["centroid_shift_mm"]) else np.nan,
                "initial_centroid_shift_mm": initial_shift,
                "circularity": float(m["circularity"]) if np.isfinite(m["circularity"]) else np.nan,
                "component_median_hu": float(m["component_median_hu"]) if np.isfinite(m["component_median_hu"]) else np.nan,
                "plane_score": float(score),
                "plane_pass": bool(hard),
                "recenter_mm": float(recenter),
                "aorta_distance_mm": float(ad),
                "inside_aorta": bool(inside),
                "tangent_z": float(t[0]), "tangent_y": float(t[1]), "tangent_x": float(t[2]),
            })

        q = pd.DataFrame(rows)
        flags = q.plane_pass.to_numpy(bool)
        fail_idx = _sustained_failure_index(flags)
        self.posthoc_failure_index = fail_idx

        # Trailing rolling pass fraction, useful for QC display but not itself the truncation rule.
        roll = []
        for i in range(len(q)):
            a = max(0, i - ROLLING_WINDOW + 1)
            roll.append(float(np.mean(flags[a:i + 1])))
        q["rolling_pass_fraction_3"] = roll
        q["sustained_failure_onset"] = False
        if fail_idx is not None:
            q.loc[fail_idx, "sustained_failure_onset"] = True

        if fail_idx is None:
            accepted_last = len(q) - 1
        else:
            accepted_last = max(0, fail_idx - 1)
        q["within_posthoc_validated_extent"] = np.arange(len(q)) <= accepted_last

        self.posthoc_qc = q
        q.to_csv(self.cache / "posthoc_final_tangent_qc.csv", index=False)

        accepted_dense = dense[:accepted_last + 1]
        accepted_s = s[:accepted_last + 1]
        validated_len = float(accepted_s[-1]) if len(accepted_s) else 0.0

        # Preserve the complete raw proposal-time outputs before replacing the public track.
        self.raw_track.to_csv(self.cache / "proposal_time_raw_track_centerline.csv", index=False)
        self.raw_track_qc.to_csv(self.cache / "proposal_time_raw_track_qc.csv", index=False)
        _json_write(self.raw_summary, self.cache / "proposal_time_raw_summary.json")

        # Public/final track is now the post-hoc validated truncated trajectory.
        self.track = pd.DataFrame({
            "track_length_mm": accepted_s,
            "z": accepted_dense[:, 0],
            "y": accepted_dense[:, 1],
            "x": accepted_dense[:, 2],
        })
        self.track_qc = q[q.within_posthoc_validated_extent].copy()

        first_failure_mm = None if fail_idx is None else float(q.track_length_mm.iloc[fail_idx])
        accepted_ad = float(self.track_qc.aorta_distance_mm.min()) if len(self.track_qc) else float("inf")
        accepted_terminal_ad = float(self.track_qc.aorta_distance_mm.iloc[-1]) if len(self.track_qc) else float("inf")
        accepted_inside = bool(self.track_qc.inside_aorta.any()) if len(self.track_qc) else False

        if accepted_inside or accepted_ad <= 0.75:
            status = "POSTHOC_TANGENT_VALIDATED_PROXIMAL_EXTENSION_REACHES_AORTA"
            accepted = validated_len >= 1.0
        elif accepted_ad <= 2.0:
            status = "POSTHOC_TANGENT_VALIDATED_EXTENSION_REACHES_OSTIUM_NEIGHBORHOOD"
            accepted = validated_len >= 1.0
        elif validated_len >= 1.0:
            status = "POSTHOC_TANGENT_VALIDATED_PROXIMAL_EXTENSION_AORTA_NOT_REACHED"
            accepted = True
        else:
            status = "NO_ROBUST_POSTHOC_TANGENT_VALIDATED_PROXIMAL_EXTENSION"
            accepted = False

        combined = np.vstack([accepted_dense[::-1][:-1], self.frozen]) if len(accepted_dense) else self.frozen.copy()
        cs = arc_mm(combined, self.spacing)
        lps = source_zyx_to_lps(self.meta, combined)

        self.summary = {
            "algorithm": ALGORITHM_VERSION,
            "status": status,
            "accepted_proximal_extension": bool(accepted),
            "target_free": True,
            "used_rca_or_prior_trunk_as_target": False,
            "aorta_used_only_as_stop_constraint": True,
            "frozen_lad_length_mm": float(arc_mm(self.frozen, self.spacing)[-1]),
            "proposal_time_raw_extension_mm": float(self.raw_summary.get("new_proximal_track_length_mm", 0.0)),
            "posthoc_validated_extension_mm": validated_len,
            "combined_lad_length_mm": float(cs[-1]),
            "first_sustained_posthoc_failure_mm": first_failure_mm,
            "posthoc_sample_step_mm": POSTHOC_STEP_MM,
            "posthoc_failure_rule": "first failing sample with >=2 failures among the next 3 dense final-tangent planes",
            "posthoc_plane_pass_fraction_all": float(q.plane_pass.mean()) if len(q) else 0.0,
            "posthoc_plane_pass_fraction_validated_extent": float(self.track_qc.plane_pass.mean()) if len(self.track_qc) else 0.0,
            "median_posthoc_score_validated_extent": float(self.track_qc.plane_score.median()) if len(self.track_qc) else float("nan"),
            "median_posthoc_radius_validated_extent_mm": float(self.track_qc.radius_mm.median()) if len(self.track_qc) else float("nan"),
            "terminal_aorta_distance_mm": accepted_terminal_ad,
            "minimum_aorta_distance_mm": accepted_ad,
            "aorta_reached": bool(accepted_inside or accepted_ad <= 0.75),
            "raw_proposal_status": self.raw_summary.get("status"),
            "raw_terminal_reason": self.raw_summary.get("terminal_reason"),
            "calibration": self.ref,
            "interpretation": "Only the dense final-path-tangent validated extent is eligible for anatomy freezing; proposal-time length is retained for provenance only.",
        }

        self.track.to_csv(self.cache / "proximal_track_centerline.csv", index=False)
        self.track_qc.to_csv(self.cache / "proximal_track_qc.csv", index=False)
        pd.DataFrame({
            "arc_mm": cs,
            "z": combined[:, 0], "y": combined[:, 1], "x": combined[:, 2],
            "lps_x_mm": lps[:, 0], "lps_y_mm": lps[:, 1], "lps_z_mm": lps[:, 2],
        }).to_csv(self.cache / "combined_lad_centerline.csv", index=False)
        _json_write(self.summary, self.cache / "tracking_summary.json")
        self._record("tracking", "posthoc_final_tangent_validation", self.cache / "tracking_summary.json", status)
        return self.summary

    def run_tracking(self, *args, **kwargs):
        raw_summary = super().run_tracking(*args, **kwargs)
        self.raw_track = self.track.copy()
        self.raw_track_qc = self.track_qc.copy()
        self.raw_summary = copy.deepcopy(raw_summary)
        return self._validate_final_path_tangent()

    def make_figures(self):
        if self.summary is None:
            self.run_tracking()
        names = super().make_figures()

        q = self.posthoc_qc
        if q is not None and len(q):
            fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
            x = q.track_length_mm.to_numpy(float)
            accepted_end = float(self.summary["posthoc_validated_extension_mm"])
            failure = self.summary.get("first_sustained_posthoc_failure_mm")

            axs[0].plot(x, q.radius_mm, marker="o", ms=3)
            axs[0].axhline(self.ref["median_radius_mm"], ls="--")
            axs[0].set_ylabel("Radius mm")

            axs[1].plot(x, q.plane_score, marker="o", ms=3)
            axs[1].axhline(0.60, ls="--")
            axs[1].set_ylabel("Final-tangent score")

            axs[2].step(x, q.plane_pass.astype(int), where="mid", label="plane pass")
            axs[2].plot(x, q.rolling_pass_fraction_3, marker="o", ms=3, label="rolling 3-plane pass fraction")
            axs[2].set_ylim(-0.05, 1.05)
            axs[2].set_ylabel("Serial QC")
            axs[2].legend(fontsize=8)

            axs[3].plot(x, q.aorta_distance_mm, marker="o", ms=3)
            axs[3].axhline(2.0, ls="--")
            axs[3].axhline(0.75, ls=":")
            axs[3].set_ylabel("Aorta distance mm")
            axs[3].set_xlabel("Raw proposed proximal path mm")

            for ax in axs:
                ax.axvline(accepted_end, ls="--", alpha=0.8)
                if failure is not None:
                    ax.axvline(float(failure), ls=":", alpha=0.8)
            fig.suptitle(
                f"Post-hoc final-path-tangent validation | raw {self.summary['proposal_time_raw_extension_mm']:.2f} mm -> validated {accepted_end:.2f} mm"
            )
            fp = self.out / "05_posthoc_final_tangent_validation.png"
            fig.tight_layout()
            fig.savefig(fp, dpi=180)
            plt.close(fig)
            names = list(names) + [fp.name]
            _json_write(names, self.out / "figure_manifest.json")
        return names

    def build_report(self):
        if self.summary is None:
            self.run_tracking()
        if not (self.cache / "figures.done").exists() or not (self.out / "05_posthoc_final_tangent_validation.png").exists():
            self.make_figures()

        manifest = json.loads((self.out / "figure_manifest.json").read_text(encoding="utf-8"))
        s = self.summary
        summary_table = pd.DataFrame([{
            "status": s["status"],
            "frozen_lad_mm": s["frozen_lad_length_mm"],
            "raw_proposal_mm": s["proposal_time_raw_extension_mm"],
            "posthoc_validated_mm": s["posthoc_validated_extension_mm"],
            "combined_validated_mm": s["combined_lad_length_mm"],
            "first_sustained_failure_mm": s["first_sustained_posthoc_failure_mm"],
            "min_aorta_mm": s["minimum_aorta_distance_mm"],
            "aorta_reached": s["aorta_reached"],
        }]).to_html(index=False, float_format=lambda x: f"{x:.3f}")

        qcols = [
            "track_length_mm", "radius_mm", "centroid_shift_mm", "circularity",
            "component_median_hu", "plane_score", "plane_pass",
            "rolling_pass_fraction_3", "recenter_mm", "aorta_distance_mm",
            "sustained_failure_onset", "within_posthoc_validated_extent",
        ]
        qtab = self.posthoc_qc[qcols].to_html(index=False, float_format=lambda x: f"{x:.3f}")

        html = f"""<html><head><meta charset='utf-8'><title>OpenPlaque Frozen LAD Proximal Reacquisition — Final Tangent Validation</title>
<style>body{{font-family:Arial;max-width:1450px;margin:24px auto;padding:0 18px}}img{{max-width:100%;margin:8px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:5px;border-bottom:1px solid #ddd}}.warn{{background:#fff3cd;padding:12px;border-left:4px solid #c99600}}.good{{background:#eef7ee;padding:12px;border-left:4px solid #4b8f4b}}</style></head><body>
<h1>OpenPlaque — Frozen LAD Proximal Reacquisition</h1>
<div class='good'><b>Final status: {s['status']}</b><br>The original target-free proposal search is preserved, but anatomy acceptance is based only on dense source-CCTA planes perpendicular to the completed local path tangent.</div>
<div class='warn'><b>Correction:</b> raw proposal-time extension {s['proposal_time_raw_extension_mm']:.3f} mm; post-hoc final-tangent validated extent {s['posthoc_validated_extension_mm']:.3f} mm. The raw proposal length is not eligible for anatomy freezing.</div>
<h2>Summary</h2>{summary_table}
<h2>Post-hoc dense final-tangent QC</h2>{qtab}
<h2>Figures</h2>{''.join(f"<h3>{n}</h3><img src='{n}'>" for n in manifest)}
</body></html>"""
        fp = self.out / "OPENPLAQUE_LAD_FROZEN_PROXIMAL_REACQUISITION_REPORT.html"
        fp.write_text(html, encoding="utf-8")
        (self.cache / "report.done").write_text("done")
        self._record("report", "generated_posthoc_corrected", fp)
        return fp

    def package(self):
        zp = super().package()
        extras = [
            self.cache / "posthoc_final_tangent_qc.csv",
            self.cache / "proposal_time_raw_track_centerline.csv",
            self.cache / "proposal_time_raw_track_qc.csv",
            self.cache / "proposal_time_raw_summary.json",
            self.out / "05_posthoc_final_tangent_validation.png",
        ]
        with zipfile.ZipFile(zp, "a", compression=zipfile.ZIP_DEFLATED) as z:
            present = set(z.namelist())
            for fp in extras:
                fp = Path(fp)
                if fp.exists() and fp.name not in present:
                    z.write(fp, arcname=fp.name)
        return zp
