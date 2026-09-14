from __future__ import annotations

"""Fine source-CCTA characterization of the secondary-branch terminal transition.

This experiment does not search for a new distal vessel. It samples the accepted secondary
branch at fine spacing, then continues a short tangent extrapolation beyond the geometric
endpoint, to identify where a compact coronary-sized lumen stops being robustly separable
from the adjacent broad contrast-filled structure.

Research use only. Vessel identity is never assigned automatically.
"""

import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ALGORITHM_VERSION = "secondary-terminal-transition-v1.0"


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    d = np.linalg.norm(np.diff(p, axis=0) * np.asarray(spacing, float), axis=1)
    return np.r_[0.0, np.cumsum(d)]


def resample_path(path, spacing, step_mm=0.05):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return p.copy()
    s = arc_mm(p, spacing)
    x = np.arange(0.0, s[-1] + 1e-9, float(step_mm))
    if len(x) == 0 or x[-1] < s[-1] - 1e-6:
        x = np.r_[x, s[-1]]
    return np.column_stack([np.interp(x, s, p[:, j]) for j in range(3)])


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def orthogonal_plane_memmap(ct, point_zyx, tangent_zyx_mm, spacing, half_mm=4.2, pix_mm=0.16):
    t = _unit(tangent_zyx_mm)
    u, v = _orth_basis(t)
    grid = np.arange(-half_mm, half_mm + 1e-9, pix_mm)
    vv, uu = np.meshgrid(grid, grid, indexing="ij")
    center_mm = np.asarray(point_zyx, float) * np.asarray(spacing, float)
    zyx_mm = center_mm[None, None, :] + uu[..., None] * u + vv[..., None] * v
    zyx = zyx_mm / np.asarray(spacing, float)
    im = ndi.map_coordinates(
        ct,
        [zyx[..., 0], zyx[..., 1], zyx[..., 2]],
        output=np.float32,
        order=1,
        mode="nearest",
    )
    return im, grid


def _component_metrics(im, grid, comp):
    yy, xx = np.nonzero(comp)
    pix = float(abs(grid[1] - grid[0]))
    area = int(comp.sum()) * pix * pix
    radius = math.sqrt(area / math.pi)
    cu, cv = float(np.mean(grid[xx])), float(np.mean(grid[yy]))
    shift = float(math.hypot(cu, cv))
    er = ndi.binary_erosion(comp)
    perimeter = max(float(np.logical_and(comp, ~er).sum()) * pix, pix)
    circ = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0, 1))
    return {
        "radius_mm": float(radius),
        "centroid_shift_mm": shift,
        "circularity": circ,
        "component_median_hu": float(np.median(im[comp])),
        "offset_u_mm": cu,
        "offset_v_mm": cv,
    }


def _near_center_component(im, grid, threshold_hu, max_shift_mm=0.95):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    candidates = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        if m["centroid_shift_mm"] <= max_shift_mm or dmin <= 0.45:
            candidates.append((m["centroid_shift_mm"] + 0.30 * dmin, m))
    return None if not candidates else min(candidates, key=lambda x: x[0])[1]


def _run_start(mask, run_len=3):
    a = np.asarray(mask, bool)
    for i in range(0, max(0, len(a) - int(run_len) + 1)):
        if bool(np.all(a[i : i + int(run_len)])):
            return i
    return None


def synthetic_transition_self_test():
    arcs = np.arange(11.5, 14.51, 0.10)
    radius = np.where(arcs < 12.9, 1.9, 1.9 + 1.8 * (arcs - 12.9))
    passed = radius <= 2.65
    df = pd.DataFrame({"arc_mm": arcs, "radius_mm": radius, "plane_pass": passed, "plane_score": np.where(passed, 0.9, 0.5), "centroid_shift_mm": 0.25})
    win = 5
    df["rolling_pass_fraction"] = df["plane_pass"].rolling(win, center=True, min_periods=3).mean()
    df["rolling_median_radius_mm"] = df["radius_mm"].rolling(win, center=True, min_periods=3).median()
    gate = (df["rolling_pass_fraction"] >= 0.80) & (df["rolling_median_radius_mm"] <= 2.65)
    last = float(df.loc[gate, "arc_mm"].max())
    broad_i = _run_start((df["radius_mm"] > 2.65).to_numpy(), 3)
    broad = None if broad_i is None else float(df.iloc[broad_i].arc_mm)
    return {"passed": bool(12.9 <= last <= 13.4 and broad is not None and broad > last), "last_supported_arc_mm": last, "sustained_broadening_onset_arc_mm": broad}


class SecondaryTerminalTransitionWorkflow:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_Terminal_Transition_v1"
        self.out = self.root / "Secondary_Terminal_Transition_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {"inputs": True, "profile": True, "figures": True, "report": True}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.spacing = None
        self.ct = None
        self.seed = None
        self.seed_length = None
        self.calibration = None
        self.profile = None
        self.summary = None

    def cache_status(self):
        names = {"inputs": "input_snapshot.json", "profile": "transition_profile.csv", "figures": "figures.done", "report": "report.done"}
        return pd.DataFrame([{"component": k, "reuse": self.reuse[k], "cache_exists": (self.cache / v).exists()} for k, v in names.items()])

    def load_inputs(self):
        src = self.root / "Cache" / "Secondary_3D_Vesselness_Topology_v1"
        ctf = src / "series7_int16.npy"
        metaf = src / "series7_int16.json"
        seedf = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv"
        missing = [p for p in (ctf, metaf, seedf) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing prior source/accepted-branch outputs: " + "; ".join(map(str, missing)))
        meta = json.loads(metaf.read_text())
        self.spacing = np.asarray(meta["spacing_zyx"], float)
        self.ct = np.load(ctf, mmap_mode="r")
        self.seed = pd.read_csv(seedf)[["z", "y", "x"]].to_numpy(float)
        self.seed_length = float(arc_mm(self.seed, self.spacing)[-1])
        self._calibrate_lumen()
        snap = {
            "algorithm": ALGORITHM_VERSION,
            "source_representation": "source-CCTA int16 memmap sampled directly into float32 planes",
            "accepted_seed_length_mm": self.seed_length,
            "profile_arc_range_mm": [11.5, 14.5],
            "profile_step_mm": 0.10,
            "extrapolation_after_seed_endpoint": True,
            "rolling_window_planes": 5,
            "rolling_window_span_mm": 0.50,
            "sustained_failure_run_planes": 3,
            "search_target": None,
            "note": "Characterization only: no distal reacquisition and no vessel identity assignment."
        }
        (self.cache / "input_snapshot.json").write_text(json.dumps(snap, indent=2))
        return snap

    def _calibrate_lumen(self):
        sp = resample_path(self.seed, self.spacing, 0.10)
        s = arc_mm(sp, self.spacing)
        ref = sp[(s >= 10.0) & (s <= 11.8)]
        center_hu = ndi.map_coordinates(self.ct, [ref[:, 0], ref[:, 1], ref[:, 2]], order=1, mode="nearest")
        refhu = float(np.median(center_hu))
        thr = float(max(170.0, min(300.0, 0.38 * refhu)))
        rows = []
        for i, pt in enumerate(ref):
            i0, i1 = max(0, i - 2), min(len(ref) - 1, i + 2)
            tangent = (ref[i1] - ref[i0]) * self.spacing
            im, grid = orthogonal_plane_memmap(self.ct, pt, tangent, self.spacing)
            m = _near_center_component(im, grid, thr, 0.80)
            if m is not None:
                rows.append(m)
        if not rows:
            raise RuntimeError("Could not calibrate compact lumen on the accepted branch.")
        df = pd.DataFrame(rows)
        good = df[(df.radius_mm >= 0.65) & (df.radius_mm <= 2.65) & (df.centroid_shift_mm <= 0.80)]
        if len(good) < 3:
            good = df
        self.calibration = {
            "reference_center_hu": refhu,
            "bright_threshold_hu": thr,
            "median_radius_mm": float(good.radius_mm.median()),
            "p90_radius_mm": float(good.radius_mm.quantile(0.90)),
            "median_circularity": float(good.circularity.median()),
            "n_planes": int(len(good)),
        }
        (self.cache / "lumen_calibration.json").write_text(json.dumps(self.calibration, indent=2))

    def _sample_path_and_tangent(self, arc_value):
        fine = resample_path(self.seed, self.spacing, 0.025)
        ss = arc_mm(fine, self.spacing)
        if arc_value <= ss[-1]:
            p = np.array([np.interp(arc_value, ss, fine[:, j]) for j in range(3)], float)
            a0 = max(0.0, arc_value - 0.25)
            a1 = min(float(ss[-1]), arc_value + 0.25)
            p0 = np.array([np.interp(a0, ss, fine[:, j]) for j in range(3)], float)
            p1 = np.array([np.interp(a1, ss, fine[:, j]) for j in range(3)], float)
            tangent = _unit((p1 - p0) * self.spacing)
            return p, tangent, False
        back = max(0.0, float(ss[-1]) - 0.80)
        p0 = np.array([np.interp(back, ss, fine[:, j]) for j in range(3)], float)
        pend = fine[-1]
        tangent = _unit((pend - p0) * self.spacing)
        delta = float(arc_value - ss[-1])
        p = pend + (delta * tangent) / self.spacing
        return p, tangent, True

    def _measure_plane(self, arc_value):
        p, tangent, extrap = self._sample_path_and_tangent(float(arc_value))
        im, grid = orthogonal_plane_memmap(self.ct, p, tangent, self.spacing, half_mm=4.2, pix_mm=0.16)
        m = _near_center_component(im, grid, self.calibration["bright_threshold_hu"], max_shift_mm=0.95)
        row = {"arc_mm": float(arc_value), "z": p[0], "y": p[1], "x": p[2], "extrapolated": bool(extrap), "component_found": m is not None}
        if m is None:
            row.update({"radius_mm": np.nan, "centroid_shift_mm": np.nan, "circularity": np.nan, "component_median_hu": np.nan, "plane_score": 0.0, "plane_pass": False})
            return row
        rref = float(self.calibration["median_radius_mm"])
        passed = bool(
            0.65 <= m["radius_mm"] <= 2.65
            and m["centroid_shift_mm"] <= 0.75
            and m["circularity"] >= 0.30
            and m["component_median_hu"] >= max(220.0, 0.45 * self.calibration["reference_center_hu"])
        )
        rscore = math.exp(-0.5 * ((m["radius_mm"] - rref) / max(0.55 * rref, 0.70)) ** 2)
        sscore = math.exp(-0.5 * (m["centroid_shift_mm"] / 0.55) ** 2)
        cscore = float(np.clip(m["circularity"] / max(self.calibration["median_circularity"], 0.35), 0, 1))
        hscore = float(np.clip(m["component_median_hu"] / max(self.calibration["reference_center_hu"], 1.0), 0, 1.2) / 1.2)
        score = float(0.35 * rscore + 0.35 * sscore + 0.18 * cscore + 0.12 * hscore)
        row.update({**m, "plane_score": score, "plane_pass": passed})
        return row

    def run_profile(self, start_arc_mm=11.5, end_arc_mm=14.5, step_mm=0.10):
        if self.ct is None:
            self.load_inputs()
        pf = self.cache / "transition_profile.csv"
        sf = self.cache / "summary.json"
        if self.reuse["profile"] and pf.exists() and sf.exists():
            self.profile = pd.read_csv(pf)
            self.summary = json.loads(sf.read_text())
            return self.summary

        arcs = np.arange(float(start_arc_mm), float(end_arc_mm) + 1e-9, float(step_mm))
        df = pd.DataFrame([self._measure_plane(a) for a in arcs])
        win = 5
        df["rolling_pass_fraction"] = df["plane_pass"].astype(float).rolling(win, center=True, min_periods=3).mean()
        df["rolling_median_radius_mm"] = df["radius_mm"].rolling(win, center=True, min_periods=3).median()
        df["rolling_median_shift_mm"] = df["centroid_shift_mm"].rolling(win, center=True, min_periods=3).median()
        df["rolling_median_score"] = df["plane_score"].rolling(win, center=True, min_periods=3).median()
        df["rolling_compact_gate"] = (
            (df["rolling_pass_fraction"] >= 0.80)
            & (df["rolling_median_radius_mm"] <= 2.65)
            & (df["rolling_median_shift_mm"] <= 0.65)
            & (df["rolling_median_score"] >= 0.62)
        )

        known = df[~df.extrapolated.astype(bool)].copy()
        robust = known[known.rolling_compact_gate.astype(bool)]
        last_robust = float(robust.arc_mm.max()) if len(robust) else None

        fail_i = _run_start((~known.plane_pass.astype(bool)).to_numpy(), 3)
        broad_i = _run_start((known.radius_mm > 2.65).fillna(True).to_numpy(), 3)
        fail_onset = None if fail_i is None else float(known.iloc[fail_i].arc_mm)
        broad_onset = None if broad_i is None else float(known.iloc[broad_i].arc_mm)
        onset_vals = [x for x in (fail_onset, broad_onset) if x is not None]
        transition_onset = min(onset_vals) if onset_vals else None

        endpoint_i = int(np.argmin(np.abs(df.arc_mm.to_numpy(float) - float(self.seed_length))))
        endpoint = df.iloc[endpoint_i]
        first_extrap = df[df.extrapolated.astype(bool)].iloc[0] if bool(df.extrapolated.any()) else None

        summary = {
            "algorithm": ALGORITHM_VERSION,
            "status": "TERMINAL_TRANSITION_CHARACTERIZED",
            "accepted_seed_length_mm": float(self.seed_length),
            "last_robust_compact_arc_mm": last_robust,
            "first_sustained_failure_arc_mm": fail_onset,
            "first_sustained_broadening_arc_mm": broad_onset,
            "transition_onset_arc_mm": transition_onset,
            "endpoint_sample_arc_mm": float(endpoint.arc_mm),
            "endpoint_plane_pass": bool(endpoint.plane_pass),
            "endpoint_radius_mm": None if not np.isfinite(endpoint.radius_mm) else float(endpoint.radius_mm),
            "endpoint_centroid_shift_mm": None if not np.isfinite(endpoint.centroid_shift_mm) else float(endpoint.centroid_shift_mm),
            "endpoint_plane_score": float(endpoint.plane_score),
            "first_extrapolated_arc_mm": None if first_extrap is None else float(first_extrap.arc_mm),
            "first_extrapolated_plane_pass": None if first_extrap is None else bool(first_extrap.plane_pass),
            "first_extrapolated_radius_mm": None if first_extrap is None or not np.isfinite(first_extrap.radius_mm) else float(first_extrap.radius_mm),
            "n_profile_planes": int(len(df)),
            "n_known_path_planes": int((~df.extrapolated.astype(bool)).sum()),
            "n_extrapolated_planes": int(df.extrapolated.astype(bool).sum()),
            "interpretation": (
                "Fine source-CCTA sampling estimates the last robustly compact segment and the onset of sustained transition into a non-separable/broader contrast structure. "
                "This characterizes support length; it does not prove anatomical vessel termination."
            ),
        }
        self.profile = df
        self.summary = summary
        df.to_csv(pf, index=False)
        sf.write_text(json.dumps(summary, indent=2))
        return summary

    def _plane_at_arc(self, arc_value):
        p, tangent, _ = self._sample_path_and_tangent(float(arc_value))
        return orthogonal_plane_memmap(self.ct, p, tangent, self.spacing, half_mm=4.2, pix_mm=0.16)[0]

    def make_figures(self):
        if self.summary is None or self.profile is None:
            self.run_profile()
        names = [
            "01_transition_profiles.png",
            "02_transition_cross_sections.png",
            "03_transition_geometry.png",
        ]
        df = self.profile

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.arc_mm, df.radius_mm, marker=".", label="radius")
        ax.axhline(2.65, linestyle="--", label="2.65 mm radius limit")
        if self.summary["last_robust_compact_arc_mm"] is not None:
            ax.axvline(self.summary["last_robust_compact_arc_mm"], linestyle="--", label="last robust compact")
        if self.summary["transition_onset_arc_mm"] is not None:
            ax.axvline(self.summary["transition_onset_arc_mm"], linestyle=":", label="transition onset")
        ax.axvline(self.seed_length, linestyle="-.", label="accepted geometric endpoint")
        ax.set_xlabel("Arc / extrapolated distance coordinate (mm)")
        ax.set_ylabel("Apparent bright-component radius (mm)")
        ax.legend()
        ax.set_title("Secondary branch terminal transition")
        fig.tight_layout()
        fig.savefig(self.out / names[0], dpi=160)
        plt.close(fig)

        keys = [11.5, 12.0]
        if self.summary["last_robust_compact_arc_mm"] is not None:
            keys.append(float(self.summary["last_robust_compact_arc_mm"]))
        if self.summary["transition_onset_arc_mm"] is not None:
            keys.append(float(self.summary["transition_onset_arc_mm"]))
        keys += [float(self.seed_length), 14.2, 14.5]
        keys = sorted(set(round(float(x), 2) for x in keys))
        fig, axes = plt.subplots(2, int(math.ceil(len(keys) / 2)), figsize=(3.4 * int(math.ceil(len(keys) / 2)), 6.5))
        axes = np.asarray(axes).ravel()
        for ax, a in zip(axes, keys):
            im = self._plane_at_arc(a)
            row = df.iloc[int(np.argmin(np.abs(df.arc_mm.to_numpy(float) - a)))]
            ax.imshow(im, cmap="gray", vmin=0, vmax=900)
            ax.set_title(f"{a:.2f} mm\nR={row.radius_mm:.2f} pass={bool(row.plane_pass)}")
            ax.axis("off")
        for ax in axes[len(keys):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.out / names[1], dpi=160)
        plt.close(fig)

        fine = resample_path(self.seed, self.spacing, 0.05)
        sample = df[["z", "y", "x"]].to_numpy(float)
        fig = plt.figure(figsize=(12, 4))
        for k, (a, b, title) in enumerate(((2, 1, "X-Y"), (2, 0, "X-Z"), (1, 0, "Y-Z")), 1):
            ax = fig.add_subplot(1, 3, k)
            ax.plot(fine[:, a] * self.spacing[a], fine[:, b] * self.spacing[b], label="accepted geometry")
            good = df.plane_pass.astype(bool).to_numpy()
            ax.scatter(sample[good, a] * self.spacing[a], sample[good, b] * self.spacing[b], s=12, label="compact pass")
            ax.scatter(sample[~good, a] * self.spacing[a], sample[~good, b] * self.spacing[b], s=12, marker="x", label="fail")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="datalim")
            if k == 1:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.out / names[2], dpi=160)
        plt.close(fig)

        (self.cache / "figures.done").write_text("done")
        return names

    def package(self):
        if self.summary is None:
            self.run_profile()
        names = self.make_figures()
        html = self.out / "OPENPLAQUE_SECONDARY_TERMINAL_TRANSITION_REPORT.html"
        s = self.summary
        html.write_text(
            "<html><body><h1>OpenPlaque Secondary Terminal Transition</h1>"
            f"<p><b>Status:</b> {s['status']}</p>"
            f"<p><b>Last robust compact arc:</b> {s['last_robust_compact_arc_mm']}</p>"
            f"<p><b>Transition onset:</b> {s['transition_onset_arc_mm']}</p>"
            f"<p><b>Accepted geometric endpoint:</b> {s['accepted_seed_length_mm']}</p>"
            f"<p>{s['interpretation']}</p>"
            + "".join(f"<p><img src='{n}' style='max-width:100%'></p>" for n in names)
            + "</body></html>"
        )
        for src in (self.cache / "summary.json", self.cache / "transition_profile.csv", self.cache / "lumen_calibration.json", self.cache / "input_snapshot.json"):
            if src.exists():
                (self.out / src.name).write_bytes(src.read_bytes())
        zip_path = self.out / "OPENPLAQUE_SECONDARY_TERMINAL_TRANSITION_REPORT_BACK.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in self.out.iterdir():
                if p.is_file() and p.name != zip_path.name:
                    z.write(p, arcname=p.name)
        (self.cache / "report.done").write_text("done")
        return zip_path
