from __future__ import annotations

"""Instrumented rejection-reason diagnostic for target-free secondary-branch continuation.

This does not try to improve tracking. It repeats the target-free local search while
recording why every attempted next step is rejected. Research use only.
"""
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .lcx_gap_bridge import (
    _angle,
    _component_metrics,
    _nearest_dist,
    _orth_basis,
    _unit,
    _write_json,
    arc_mm,
    orthogonal_plane,
    resample_path,
    score_component,
)
from .secondary_target_free_continuation import SecondaryTargetFreeContinuationWorkflow

ALGORITHM_VERSION = "secondary-rejection-diagnostic-v1.0"


class SecondaryRejectionDiagnosticWorkflow(SecondaryTargetFreeContinuationWorkflow):
    COMPONENTS = ("source_ct", "frozen_geometry", "diagnostic_search")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse=reuse)
        self.cache = self.root / "Cache" / "Secondary_Rejection_Diagnostic_v1"
        self.out = self.root / "Secondary_Rejection_Diagnostic_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.attempts = self.rejection_summary = self.survivors = None
        self.summary = None
        self.best_survivor_path = None

    def cache_status(self):
        names = {
            "source_ct": "series7_int16.npy",
            "frozen_geometry": "frozen_geometry.json",
            "diagnostic_search": "diagnostic_summary.json",
        }
        return pd.DataFrame([
            {"component": k, "reuse": self.reuse[k], "cache_exists": (self.cache / v).exists()}
            for k, v in names.items()
        ])

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        om = self.cache / "series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and om.exists():
            self.ct = np.load(own, mmap_mode="r")
            self.meta = json.loads(om.read_text())
            self.spacing = np.asarray(self.meta["spacing_zyx"], float)
            self._record("source_ct", "reused", own)
            return self.ct
        sources = [
            self.root / "Cache" / "Secondary_Target_Free_Continuation_v1" / "series7_int16.npy",
            self.root / "Cache" / "LCX_Gap_Bridge_v1" / "series7_int16.npy",
            self.root / "Cache" / "LCX_Distal_Relaunch_v1" / "series7_int16.npy",
        ]
        src = next((p for p in sources if p.exists() and p.with_suffix(".json").exists()), None)
        if src is None:
            raise FileNotFoundError("No source-CCTA cache found for rejection diagnostic")
        shutil.copyfile(src, own)
        shutil.copyfile(src.with_suffix(".json"), om)
        self.ct = np.load(own, mmap_mode="r")
        self.meta = json.loads(om.read_text())
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", "imported_prior_cache", src)
        return self.ct

    def load_frozen_geometry(self):
        snap = super().load_frozen_geometry()
        snap["algorithm"] = ALGORITHM_VERSION
        snap["diagnostic_purpose"] = "Record every local and geometric rejection reason; do not loosen thresholds."
        _write_json(snap, self.cache / "frozen_geometry.json")
        return snap

    def _component_probe(self, im, c, search_mm=1.30):
        rr = float(self.rca["median_radius_mm"])
        rh = float(self.rca["median_center_hu"])
        lo_hu = max(170.0, min(260.0, 0.38 * rh))
        bright = (im >= lo_hu) & (im <= 1200.0)
        lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
        pix = float(abs(c[1] - c[0]))
        raw = []
        for k in range(1, int(nlab) + 1):
            comp = lab == k
            area = int(comp.sum())
            if area < 7:
                continue
            yy, xx = np.nonzero(comp)
            cu, cv = float(np.mean(c[xx])), float(np.mean(c[yy]))
            shift = float(math.hypot(cu, cv))
            radius = math.sqrt(area * pix * pix / math.pi)
            raw.append((shift, radius, comp, cu, cv, area))

        base = {
            "bright_threshold_hu": float(lo_hu),
            "n_labeled_bright_components": int(nlab),
            "n_components_area_ge7": int(len(raw)),
        }
        if not raw:
            return "no_bright_component", None, base

        within = [r for r in raw if r[0] <= float(search_mm)]
        if not within:
            nearest = min(raw, key=lambda r: r[0])
            base.update({
                "nearest_component_shift_mm": float(nearest[0]),
                "nearest_component_radius_mm": float(nearest[1]),
            })
            return "component_outside_search", None, base

        sized = [r for r in within if 0.65 <= r[1] <= 2.65]
        if not sized:
            target_r = 1.10 * rr
            nearest_r = min(within, key=lambda r: abs(r[1] - target_r))
            base.update({
                "nearest_component_shift_mm": float(nearest_r[0]),
                "nearest_component_radius_mm": float(nearest_r[1]),
            })
            reason = "radius_too_small" if nearest_r[1] < 0.65 else "radius_too_large"
            return reason, None, base

        best = None
        for shift, radius, comp, cu, cv, area in sized:
            m = _component_metrics(im, c, comp, cu, cv)
            rscore = math.exp(-0.5 * ((m["radius_mm"] - 1.10 * rr) / max(0.62 * rr, 0.72)) ** 2)
            pscore = math.exp(-0.5 * (m["recenter_shift_mm"] / 0.72) ** 2)
            cscore = float(np.clip(m["circularity"] / 0.50, 0, 1))
            hscore = math.exp(-0.5 * ((m["center_hu"] - rh) / 330.0) ** 2)
            xscore = 1.0 / (1.0 + math.exp(-(m["core_minus_ring_hu"] - 10.0) / 80.0))
            choose = 0.33 * rscore + 0.30 * pscore + 0.16 * cscore + 0.11 * hscore + 0.10 * xscore
            m["choose_score"] = float(choose)
            if best is None or choose > best[0]:
                best = (choose, m)
        return "component_found", best[1], base

    def _probe_local(self, cur, direction, step_mm):
        row = {
            "rejection_stage": "local",
            "rejection_reason": "",
            "requested_step_mm": float(step_mm),
        }
        guess = np.asarray(cur, float) + (float(step_mm) * _unit(direction)) / self.spacing
        if np.any(guess < 2) or np.any(guess >= np.asarray(self.ct.shape) - 3):
            row["rejection_reason"] = "volume_bounds"
            return None, row

        im, c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        status, m, detail = self._component_probe(im, c, 1.30)
        row.update(detail)
        if m is None:
            row["rejection_reason"] = status
            return None, row

        row.update({
            "radius_mm": float(m["radius_mm"]),
            "circularity": float(m["circularity"]),
            "center_hu": float(m["center_hu"]),
            "core_minus_ring_hu": float(m["core_minus_ring_hu"]),
            "recenter_shift_mm": float(m["recenter_shift_mm"]),
            "choose_score": float(m["choose_score"]),
        })
        u, v = _orth_basis(direction)
        p = (guess * self.spacing + m["centroid_u_mm"] * u + m["centroid_v_mm"] * v) / self.spacing
        vec = (p - cur) * self.spacing
        L = float(np.linalg.norm(vec))
        row["actual_step_mm"] = L
        if not (0.18 <= L <= 1.15):
            row["rejection_reason"] = "recenter_step_length"
            return None, row

        sc, hp, sp = score_component(m, self.rca)
        row.update({"plane_score": float(sc), "hard": bool(hp), "soft": bool(sp)})
        if not sp:
            row["rejection_reason"] = "soft_score_fail"
            return None, row

        refsep = _nearest_dist(p, np.vstack([self.reference, self.trunk]), self.spacing)
        row["reference_separation_mm"] = float(refsep)
        if refsep < 1.75:
            row["rejection_reason"] = "reference_too_close"
            return None, row

        row["rejection_reason"] = "local_pass"
        result = {
            "point": p,
            "direction": _unit(vec),
            "score": sc,
            "hard": hp,
            "soft": sp,
            "refsep": refsep,
            "step": L,
            **m,
        }
        return result, row

    def run_diagnostic(self, max_new_mm=4.0, beam_width=50):
        sf = self.cache / "diagnostic_summary.json"
        af = self.cache / "attempt_diagnostics.csv"
        rf = self.cache / "rejection_summary.csv"
        vf = self.cache / "beam_survivors.csv"
        bf = self.cache / "best_survivor_centerline.csv"

        if self.source_point is None:
            self.load_frozen_geometry()

        if self.reuse["diagnostic_search"] and all(p.exists() for p in (sf, af, rf, vf, bf)):
            self.summary = json.loads(sf.read_text())
            self.attempts = pd.read_csv(af)
            self.rejection_summary = pd.read_csv(rf)
            self.survivors = pd.read_csv(vf)
            self.best_survivor_path = pd.read_csv(bf)[["z", "y", "x"]].to_numpy(float)
            self._record("diagnostic_search", "reused", sf)
            return self.summary

        beams = [{
            "point": self.source_point.copy(),
            "direction": d,
            "path": [self.source_point.copy()],
            "scores": [],
            "hard": [],
            "turns": [],
            "oldsep": [],
        } for d in self._source_direction_hypotheses()]

        pool = []
        attempt_rows = []
        survivor_rows = []
        first_dead_step = None

        for step_i in range(int(math.ceil(max_new_mm / 0.28))):
            props = []
            step_mm = 0.26 if step_i < 8 else 0.30
            for bi, st in enumerate(beams):
                base = st["direction"]
                dirs = [(base, 0.0)]
                u, v = self._basis(base)
                for nominal in (8, 16, 24, 32, 40):
                    aa = math.radians(nominal)
                    for ph in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                        dirs.append((_unit(
                            math.cos(aa) * base +
                            math.sin(aa) * (math.cos(ph) * u + math.sin(ph) * v)
                        ), float(nominal)))

                for direction_index, (d, nominal) in enumerate(dirs):
                    r, row = self._probe_local(st["point"], d, step_mm)
                    row.update({
                        "step_index": int(step_i),
                        "beam_index": int(bi),
                        "direction_index": int(direction_index),
                        "nominal_turn_deg": float(nominal),
                    })
                    if r is None:
                        attempt_rows.append(row)
                        continue

                    turn = _angle(st["direction"], r["direction"])
                    old_sep = self._old_sep(r["point"])
                    self_sep = self._self_sep(r["point"], st["path"])
                    path = st["path"] + [r["point"].copy()]
                    met = self._metrics(path)
                    row.update({
                        "actual_turn_deg": float(turn),
                        "new_length_mm": float(met["length_mm"]),
                        "endpoint_displacement_mm": float(met["endpoint_displacement_mm"]),
                        "forward_projection_mm": float(met["forward_projection_mm"]),
                        "old_branch_separation_mm": float(old_sep),
                        "self_separation_mm": float(self_sep),
                        "tortuosity": float(met["tortuosity"]),
                    })

                    reason = None
                    if turn > 58:
                        reason = "turn_gt_58"
                    elif met["length_mm"] >= 1.2 and old_sep < 0.80:
                        reason = "old_branch_loopback"
                    elif met["length_mm"] >= 1.8 and self_sep < 0.62:
                        reason = "self_approach"
                    elif met["length_mm"] >= 2.5 and met["tortuosity"] > 2.15:
                        reason = "tortuosity_gt_2p15"
                    elif met["length_mm"] >= 3.0 and met["forward_projection_mm"] < 0.55:
                        reason = "insufficient_forward_progress"

                    if reason is not None:
                        row["rejection_stage"] = "geometry"
                        row["rejection_reason"] = reason
                        attempt_rows.append(row)
                        continue

                    smooth = float(np.clip(np.dot(st["direction"], r["direction"]), -1, 1))
                    obj = (
                        0.48 * r["score"] +
                        0.17 * float(r["hard"]) +
                        0.16 * ((smooth + 1) / 2) +
                        0.11 * min(old_sep / 2.2, 1) +
                        0.08 * min(max(met["forward_projection_mm"], 0) / 4, 1)
                    )
                    ns = {
                        "point": r["point"],
                        "direction": r["direction"],
                        "path": path,
                        "scores": st["scores"] + [r["score"]],
                        "hard": st["hard"] + [r["hard"]],
                        "turns": st["turns"] + [turn],
                        "oldsep": st["oldsep"] + [old_sep],
                    }
                    row["rejection_stage"] = "accepted"
                    row["rejection_reason"] = "accepted_to_beam_pool"
                    row["objective"] = float(obj)
                    attempt_rows.append(row)
                    props.append((obj, ns))

            if not props:
                first_dead_step = int(step_i)
                break

            def rank(item):
                _, st = item
                met = self._metrics(st["path"])
                return (
                    0.36 * np.mean(st["scores"]) +
                    0.16 * np.mean(st["hard"]) +
                    0.20 * min(met["length_mm"] / 7, 1) +
                    0.12 * min(met["endpoint_displacement_mm"] / 5, 1) +
                    0.09 * min(st["oldsep"][-1] / 2.2, 1) +
                    0.07 * min(max(met["forward_projection_mm"], 0) / 4, 1)
                )

            props.sort(key=rank, reverse=True)
            keep = []
            for _, st in props:
                if all(np.linalg.norm((st["point"] - q["point"]) * self.spacing) >= 0.28 for q in keep):
                    keep.append(st)
                if len(keep) >= beam_width:
                    break

            for rank_i, st in enumerate(keep):
                met = self._metrics(st["path"])
                survivor_rows.append({
                    "step_index": int(step_i),
                    "beam_rank": int(rank_i),
                    **{k: float(v) for k, v in met.items()},
                    "mean_plane_score": float(np.mean(st["scores"])),
                    "hard_fraction": float(np.mean(st["hard"])),
                    "max_turn_deg": float(max(st["turns"]) if st["turns"] else 0.0),
                })
            pool.extend(keep)
            beams = keep

        self.attempts = pd.DataFrame(attempt_rows)
        if len(self.attempts):
            self.rejection_summary = (
                self.attempts.groupby(["step_index", "rejection_reason"], dropna=False)
                .size().rename("count").reset_index()
                .sort_values(["step_index", "count"], ascending=[True, False])
            )
        else:
            self.rejection_summary = pd.DataFrame(columns=["step_index", "rejection_reason", "count"])
        self.survivors = pd.DataFrame(survivor_rows)

        if pool:
            best = max(pool, key=lambda st: self._metrics(st["path"])["length_mm"])
            self.best_survivor_path = resample_path(np.asarray(best["path"], float), self.spacing, 0.20)
            max_len = float(self._metrics(best["path"])["length_mm"])
        else:
            self.best_survivor_path = np.asarray([self.source_point])
            max_len = 0.0

        dead = first_dead_step
        terminal_counts = {}
        terminal_attempts = 0
        dominant = None
        if dead is not None and len(self.attempts):
            t = self.attempts[self.attempts.step_index == dead]
            terminal_attempts = int(len(t))
            terminal_counts = {str(k): int(v) for k, v in t.rejection_reason.value_counts().to_dict().items()}
            if terminal_counts:
                dominant, dom_n = max(terminal_counts.items(), key=lambda kv: kv[1])
                if dom_n / max(terminal_attempts, 1) < 0.50:
                    dominant = "mixed"

        if dominant in {"radius_too_large", "radius_too_small"}:
            interpretation = "Terminal failure is dominated by coronary-size gating."
        elif dominant in {"no_bright_component", "component_outside_search"}:
            interpretation = "Terminal failure is dominated by inability to find a local contrast-filled component in the allowed recenter neighborhood."
        elif dominant in {"soft_score_fail", "recenter_step_length", "reference_too_close"}:
            interpretation = "Terminal failure is dominated by local component/QC gating after a bright structure is detected."
        elif dominant in {"turn_gt_58", "old_branch_loopback", "self_approach", "tortuosity_gt_2p15", "insufficient_forward_progress"}:
            interpretation = "Terminal failure is dominated by geometry/continuity gating rather than component detection."
        else:
            interpretation = "Terminal failure is mixed; inspect rejection_summary.csv and terminal attempts."

        self.summary = {
            "algorithm": ALGORITHM_VERSION,
            "status": "DIAGNOSTIC_COMPLETE",
            "first_dead_step": dead,
            "max_surviving_length_mm": max_len,
            "terminal_attempts": terminal_attempts,
            "terminal_rejection_counts": terminal_counts,
            "dominant_terminal_failure": dominant,
            "interpretation": interpretation,
        }

        self.attempts.to_csv(af, index=False)
        self.rejection_summary.to_csv(rf, index=False)
        self.survivors.to_csv(vf, index=False)
        pd.DataFrame({
            "arc_mm": arc_mm(self.best_survivor_path, self.spacing),
            "z": self.best_survivor_path[:, 0],
            "y": self.best_survivor_path[:, 1],
            "x": self.best_survivor_path[:, 2],
        }).to_csv(bf, index=False)
        _write_json(self.summary, sf)
        self._record("diagnostic_search", "recomputed_and_cached", sf, f"dead_step={dead}; dominant={dominant}")
        return self.summary

    def make_figures(self):
        if self.summary is None:
            self.run_diagnostic()
        names = ["01_rejection_counts.png", "02_terminal_attempts.png", "03_best_survivor_geometry.png"]

        fig, ax = plt.subplots(figsize=(11, 5))
        if self.rejection_summary is not None and len(self.rejection_summary):
            piv = self.rejection_summary.pivot(index="step_index", columns="rejection_reason", values="count").fillna(0)
            piv.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("All proposal outcomes by search step")
        ax.set_xlabel("search step")
        ax.set_ylabel("attempt count")
        fig.tight_layout()
        fig.savefig(self.out / names[0], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        dead = self.summary.get("first_dead_step")
        if dead is not None and self.attempts is not None and len(self.attempts):
            t = self.attempts[self.attempts.step_index == dead]
            vc = t.rejection_reason.value_counts()
            if len(vc):
                vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Terminal attempted expansions — first dead step {dead}")
        ax.set_xlabel("rejection reason")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(self.out / names[1], dpi=150)
        plt.close(fig)

        ref = np.vstack([self.reference, self.trunk])
        p = self.best_survivor_path
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(ref[:, 2], ref[:, 1], lw=1, label="LAD/trunk")
        ax.plot(self.seed[:, 2], self.seed[:, 1], lw=1.5, label="accepted secondary branch")
        if self.prior_course is not None:
            ax.plot(self.prior_course[:, 2], self.prior_course[:, 1], lw=1, label="prior relaunch")
        if p is not None:
            ax.plot(p[:, 2], p[:, 1], lw=2, label="best diagnostic survivor")
        ax.scatter([self.source_point[2]], [self.source_point[1]], s=35, label="fixed source")
        ax.set_title("Target-free rejection diagnostic geometry")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.out / names[2], dpi=150)
        plt.close(fig)
        return [self.out / n for n in names]

    def make_report(self):
        if self.summary is None:
            self.run_diagnostic()
        self.make_figures()
        rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in self.summary.items())
        html = (
            "<html><body><h1>OpenPlaque Secondary Branch — Rejection Diagnostic</h1>"
            f"<p><b>Status: {self.summary['status']}</b></p><table border='1' cellpadding='4'>{rows}</table>"
            "<h2>Proposal outcomes by step</h2><img src='01_rejection_counts.png' width='95%'>"
            "<h2>Terminal failure reasons</h2><img src='02_terminal_attempts.png' width='90%'>"
            "<h2>Geometry</h2><img src='03_best_survivor_geometry.png' width='90%'>"
            "<p>Research use only. Thresholds are unchanged from the target-free continuation experiment.</p>"
            "</body></html>"
        )
        path = self.out / "OPENPLAQUE_SECONDARY_REJECTION_DIAGNOSTIC_REPORT.html"
        path.write_text(html)
        return path

    def package(self):
        self.make_report()
        z = self.out / "OPENPLAQUE_SECONDARY_REJECTION_DIAGNOSTIC_REPORT_BACK.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
            for p in sorted(self.out.iterdir()):
                if p.is_file() and p != z:
                    zz.write(p, arcname=p.name)
            for name in [
                "diagnostic_summary.json",
                "attempt_diagnostics.csv",
                "rejection_summary.csv",
                "beam_survivors.csv",
                "best_survivor_centerline.csv",
                "frozen_geometry.json",
            ]:
                p = self.cache / name
                if p.exists():
                    zz.write(p, arcname=f"cache/{name}")
        return z
