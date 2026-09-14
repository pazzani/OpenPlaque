from __future__ import annotations

"""Target-free distal compact-lumen reacquisition for the accepted secondary branch.

This experiment does not require a visible connected lumen through the ambiguous distal
zone. It starts from the last unequivocally compact upstream segment, independently scans
a forward 3-D volume for reappearing coronary-sized lumen tracklets, and only after a
tracklet is validated does it assess whether a short bridge back to the upstream branch is
geometrically/lumen-plausible.

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

from . import secondary_3d_vesselness_topology as base
from .secondary_anchored_lumen_validation import (
    SecondaryAnchoredLumenValidationWorkflow,
    _component_metrics,
    _max_consecutive_false,
)

ALGORITHM_VERSION = "secondary-distal-reacquisition-v1.0"


def _compact_component_with_offset(im, grid, threshold_hu, max_shift_mm=0.85):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    out = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        cu = float(np.mean(grid[xx]))
        cv = float(np.mean(grid[yy]))
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        if m["centroid_shift_mm"] <= max_shift_mm or dmin <= 0.40:
            out.append((m["centroid_shift_mm"] + 0.30 * dmin, cu, cv, m))
    if not out:
        return None
    _, cu, cv, m = min(out, key=lambda x: x[0])
    return {**m, "offset_u_mm": cu, "offset_v_mm": cv}


def _angle_deg(a, b):
    a = base._unit(a)
    b = base._unit(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _direction_fan(base_dir, shell_deg=(0, 18, 32), n_azimuth=8):
    t = base._unit(base_dir)
    u, v = base._orth_basis(t)
    out = []
    for ang in shell_deg:
        if float(ang) == 0.0:
            out.append(t.copy())
            continue
        th = math.radians(float(ang))
        for j in range(int(n_azimuth)):
            ph = 2.0 * math.pi * j / float(n_azimuth)
            d = (
                math.cos(th) * t
                + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v)
            )
            out.append(base._unit(d))
    return out


def synthetic_reacquisition_self_test():
    grid = np.arange(-4.2, 4.2 + 1e-9, 0.16)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    tube = 100.0 + 650.0 * (((xx - 0.30) ** 2 + (yy + 0.20) ** 2) <= 1.75 ** 2)
    broad = 100.0 + 650.0 * (xx >= -0.10)
    mt = _compact_component_with_offset(tube, grid, 220.0)
    mb = _compact_component_with_offset(broad, grid, 220.0)
    dirs = _direction_fan([0, 1, 0])
    passed = bool(
        mt
        and mb
        and 1.45 <= mt["radius_mm"] <= 2.05
        and mb["radius_mm"] > 2.65
        and len(dirs) == 17
    )
    return {
        "passed": passed,
        "tube_radius_mm": None if mt is None else mt["radius_mm"],
        "broad_radius_mm": None if mb is None else mb["radius_mm"],
        "n_orientation_hypotheses": len(dirs),
    }


class SecondaryDistalReacquisitionWorkflow(SecondaryAnchoredLumenValidationWorkflow):
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse={})
        self.cache = self.root / "Cache" / "Secondary_Distal_Reacquisition_v1"
        self.out = self.root / "Secondary_Distal_Reacquisition_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {
            "inputs": True,
            "proposals": True,
            "reacquisition": True,
            "bridge": True,
            "figures": True,
            "report": True,
        }
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.proposal_points = None
        self.tracklet_candidates = None
        self.best_tracklet_qc = None
        self.best_tracklet_path = None
        self.bridge_candidates = None
        self.best_bridge_qc = None
        self.best_bridge_path = None
        self.summary = None
        self.reacquisition_origin = None
        self.reacquisition_tangent = None
        self.reacquisition_origin_arc = None
        self.broad_dt = None

    def cache_status(self):
        names = {
            "inputs": "input_snapshot.json",
            "proposals": "proposal_points.csv",
            "reacquisition": "tracklet_candidates.csv",
            "bridge": "bridge_candidates.csv",
            "figures": "figures.done",
            "report": "report.done",
        }
        return pd.DataFrame(
            [
                {
                    "component": k,
                    "reuse": self.reuse[k],
                    "cache_exists": (self.cache / v).exists(),
                }
                for k, v in names.items()
            ]
        )

    def load_inputs(self, reacquisition_origin_arc_mm=12.20):
        snap = super().load_inputs(control_arc_mm=12.7)
        sp = base.resample_path(self.seed, self.spacing, 0.10)
        ss = base.arc_mm(sp, self.spacing)
        self.reacquisition_origin_arc = float(reacquisition_origin_arc_mm)
        i = int(np.argmin(np.abs(ss - self.reacquisition_origin_arc)))
        self.reacquisition_origin = sp[i].copy()
        i0 = max(0, i - 8)
        self.reacquisition_tangent = base._unit(
            (sp[i] - sp[i0]) * self.spacing
        )
        bright = np.asarray(self.roi >= self.calibration["bright_threshold_hu"], bool)
        self.broad_dt = ndi.distance_transform_edt(bright, sampling=self.spacing).astype(
            np.float32
        )
        snap = dict(snap)
        snap.update(
            {
                "algorithm": ALGORITHM_VERSION,
                "reacquisition_origin_arc_mm": self.reacquisition_origin_arc,
                "reacquisition_origin_zyx": self.reacquisition_origin.tolist(),
                "search_shell_mm": [1.2, 5.5],
                "minimum_forward_projection_mm": 0.50,
                "maximum_lateral_offset_mm": 4.0,
                "tracklet_half_length_mm": 0.75,
                "tracklet_plane_step_mm": 0.25,
                "search_target": None,
                "note": (
                    "No connected lumen through the ambiguous zone is required. "
                    "Vesselness is proposal-only; reacquisition requires serial compact "
                    "source-CCTA lumen planes."
                ),
            }
        )
        base._write_json(snap, self.cache / "input_snapshot.json")
        return snap

    def _proposal_candidates(self, max_points=80, nms_mm=0.70):
        if self.ct is None:
            self.load_inputs()
        pf = self.cache / "proposal_points.csv"
        if self.reuse["proposals"] and pf.exists():
            self.proposal_points = pd.read_csv(pf)
            return self.proposal_points

        v = np.asarray(self.vesselness, float)
        hu = np.asarray(self.roi, float)
        dt = np.asarray(self.broad_dt, float)
        zz, yy, xx = np.indices(v.shape)
        g = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
        glob = g + self.roi_lo[None, :]
        phys = glob * self.spacing[None, :]
        origin_phys = self.reacquisition_origin * self.spacing
        d = phys - origin_phys[None, :]
        dist = np.linalg.norm(d, axis=1)
        proj = d @ self.reacquisition_tangent
        lateral = np.sqrt(np.maximum(dist * dist - proj * proj, 0.0))
        vf = v.ravel()
        huf = hu.ravel()
        dtf = dt.ravel()

        localmax = v >= ndi.maximum_filter(v, size=3, mode="nearest")
        keep = (
            localmax.ravel()
            & (dist >= 1.2)
            & (dist <= 5.5)
            & (proj >= 0.50)
            & (lateral <= 4.0)
            & (huf >= 180.0)
            & (huf <= 1000.0)
            & (vf >= 0.55 * float(self.vessel_threshold))
            & (dtf <= 3.20)
        )
        ids = np.flatnonzero(keep)
        if not len(ids):
            self.proposal_points = pd.DataFrame(
                columns=[
                    "proposal_id","z","y","x","distance_mm","forward_projection_mm",
                    "lateral_offset_mm","vesselness","best_scale_mm","center_hu",
                    "broad_dt_mm","proposal_score"
                ]
            )
            self.proposal_points.to_csv(pf, index=False)
            return self.proposal_points

        scale = np.asarray(self.best_scale, float).ravel()
        vn = vf[ids] / max(float(np.percentile(vf[vf > 0], 90)) if np.any(vf > 0) else 1.0, 1e-6)
        broad_pen = np.clip((dtf[ids] - 2.0) / 1.2, 0, 1)
        score = (
            0.50 * np.clip(vn, 0, 1.5)
            + 0.18 * np.clip(proj[ids] / 5.0, 0, 1)
            + 0.12 * np.exp(-lateral[ids] / 2.5)
            + 0.10 * np.exp(-np.maximum(scale[ids] - 1.1, 0) / 0.4)
            + 0.10 * (1.0 - broad_pen)
        )
        order_idx = np.argsort(score)[::-1]
        chosen = []
        rows = []
        for pos in order_idx:
            flat = ids[pos]
            p = glob[flat].astype(float)
            if any(np.linalg.norm((p - q) * self.spacing) < nms_mm for q in chosen):
                continue
            chosen.append(p.copy())
            delta = (p - self.reacquisition_origin) * self.spacing
            dd = float(np.linalg.norm(delta))
            pr = float(np.dot(delta, self.reacquisition_tangent))
            lat = float(math.sqrt(max(dd * dd - pr * pr, 0.0)))
            li = tuple(g[flat].astype(int))
            rows.append(
                {
                    "proposal_id": len(rows) + 1,
                    "z": p[0], "y": p[1], "x": p[2],
                    "distance_mm": dd,
                    "forward_projection_mm": pr,
                    "lateral_offset_mm": lat,
                    "vesselness": float(v[li]),
                    "best_scale_mm": float(self.best_scale[li]),
                    "center_hu": float(self.roi[li]),
                    "broad_dt_mm": float(self.broad_dt[li]),
                    "proposal_score": float(score[pos]),
                }
            )
            if len(rows) >= int(max_points):
                break

        self.proposal_points = pd.DataFrame(rows)
        self.proposal_points.to_csv(pf, index=False)
        return self.proposal_points

    def _measure_plane(self, point, direction, max_shift=0.80):
        im, grid = base.orthogonal_plane(self.ct, point, direction, self.spacing, half_mm=4.2, pix_mm=0.16)
        m = _compact_component_with_offset(im, grid, self.calibration["bright_threshold_hu"], max_shift_mm=max_shift)
        if m is None:
            return None, False, 0.0
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
        return m, passed, float(0.35 * rscore + 0.35 * sscore + 0.18 * cscore + 0.12 * hscore)

    def _evaluate_tracklet(self, center, direction, half_mm=0.75, step_mm=0.25):
        offsets = np.arange(-half_mm, half_mm + 1e-9, step_mm)
        rows = []
        corrected = []
        u, v = base._orth_basis(direction)
        for s in offsets:
            p = center + (float(s) * direction) / self.spacing
            m, passed, score = self._measure_plane(p, direction, 0.90)
            row = {"tracklet_offset_mm": float(s), "z": p[0], "y": p[1], "x": p[2], "component_found": m is not None, "plane_pass": bool(passed), "plane_score": float(score)}
            if m is not None:
                corr = p + (m["offset_u_mm"] * u + m["offset_v_mm"] * v) / self.spacing
                row.update({"radius_mm": m["radius_mm"], "centroid_shift_mm": m["centroid_shift_mm"], "circularity": m["circularity"], "component_median_hu": m["component_median_hu"], "corrected_z": corr[0], "corrected_y": corr[1], "corrected_x": corr[2]})
                corrected.append(corr)
            rows.append(row)

        q = pd.DataFrame(rows)
        found = q[q.component_found]
        center_row = q.iloc[int(np.argmin(np.abs(q.tracklet_offset_mm.to_numpy())))]
        metrics = {
            "tracklet_pass_fraction": float(q.plane_pass.mean()),
            "tracklet_center_pass": bool(center_row.plane_pass),
            "tracklet_median_score": float(q.plane_score.median()),
            "tracklet_min_score": float(q.plane_score.min()),
            "tracklet_max_consecutive_failures": _max_consecutive_false(q.plane_pass.tolist()),
            "tracklet_median_radius_mm": float(found.radius_mm.median()) if len(found) else np.nan,
            "tracklet_p90_radius_mm": float(found.radius_mm.quantile(0.9)) if len(found) else np.nan,
            "tracklet_median_shift_mm": float(found.centroid_shift_mm.median()) if len(found) else np.nan,
        }
        metrics["tracklet_gate"] = bool(
            metrics["tracklet_pass_fraction"] >= 6.0 / 7.0
            and metrics["tracklet_center_pass"]
            and metrics["tracklet_median_score"] >= 0.62
            and metrics["tracklet_max_consecutive_failures"] <= 1
            and np.isfinite(metrics["tracklet_p90_radius_mm"])
            and metrics["tracklet_p90_radius_mm"] <= 2.65
            and np.isfinite(metrics["tracklet_median_shift_mm"])
            and metrics["tracklet_median_shift_mm"] <= 0.60
        )
        path = np.asarray(corrected, float) if len(corrected) >= 2 else np.asarray([center - half_mm * direction / self.spacing, center + half_mm * direction / self.spacing], float)
        path = base.resample_path(path, self.spacing, 0.20)
        return q, metrics, path

    def search_reacquisition(self, top_proposals=60):
        cf = self.cache / "tracklet_candidates.csv"
        qf = self.cache / "best_tracklet_qc.csv"
        pf = self.cache / "best_tracklet_path.csv"
        if self.proposal_points is None:
            self._proposal_candidates()
        if self.reuse["reacquisition"] and cf.exists() and qf.exists() and pf.exists():
            self.tracklet_candidates = pd.read_csv(cf)
            self.best_tracklet_qc = pd.read_csv(qf)
            self.best_tracklet_path = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            return self.tracklet_candidates

        rows, qcs, paths = [], [], []
        for _, pr in self.proposal_points.head(int(top_proposals)).iterrows():
            center = np.array([pr.z, pr.y, pr.x], float)
            radial = base._unit((center - self.reacquisition_origin) * self.spacing)
            seed_dir = base._unit(0.65 * radial + 0.35 * self.reacquisition_tangent)
            best = None
            for d in _direction_fan(seed_dir):
                qc, m, path = self._evaluate_tracklet(center, d)
                continuity = _angle_deg(d, self.reacquisition_tangent)
                radial_angle = _angle_deg(d, radial)
                rank = (
                    0.34 * m["tracklet_pass_fraction"] + 0.28 * m["tracklet_median_score"]
                    + 0.13 * math.exp(-continuity / 45.0) + 0.08 * math.exp(-radial_angle / 35.0)
                    + 0.07 * min(float(pr.forward_projection_mm) / 5.0, 1.0)
                    + 0.06 * min(float(pr.vesselness) / max(self.vessel_threshold, 1e-6), 1.0)
                    + 0.04 * math.exp(-max(float(pr.broad_dt_mm) - 2.0, 0.0))
                )
                rec = {"proposal_id": int(pr.proposal_id), "z": center[0], "y": center[1], "x": center[2], "distance_from_origin_mm": float(pr.distance_mm), "forward_projection_mm": float(pr.forward_projection_mm), "lateral_offset_mm": float(pr.lateral_offset_mm), "proposal_vesselness": float(pr.vesselness), "proposal_best_scale_mm": float(pr.best_scale_mm), "proposal_broad_dt_mm": float(pr.broad_dt_mm), "orientation_continuity_deg": continuity, "orientation_radial_deg": radial_angle, **m, "rank_score": float(rank)}
                if best is None or (bool(rec["tracklet_gate"]), rec["rank_score"]) > (bool(best[0]["tracklet_gate"]), best[0]["rank_score"]):
                    best = (rec, qc, path, d.copy())
            if best is not None:
                rec, qc, path, d = best
                rec["direction_z_mm"] = d[0]; rec["direction_y_mm"] = d[1]; rec["direction_x_mm"] = d[2]
                rows.append(rec); qcs.append(qc); paths.append(path)

        self.tracklet_candidates = pd.DataFrame(rows)
        if len(self.tracklet_candidates):
            self.tracklet_candidates = self.tracklet_candidates.sort_values(["tracklet_gate", "rank_score"], ascending=[False, False]).reset_index(drop=True)
            best_prop = int(self.tracklet_candidates.iloc[0].proposal_id)
            orig_idx = next(i for i, r in enumerate(rows) if int(r["proposal_id"]) == best_prop)
            self.best_tracklet_qc = qcs[orig_idx]
            self.best_tracklet_path = paths[orig_idx]
        else:
            self.best_tracklet_qc = pd.DataFrame(columns=["tracklet_offset_mm", "plane_pass", "plane_score"])
            self.best_tracklet_path = np.asarray([self.reacquisition_origin], float)

        self.tracklet_candidates.to_csv(cf, index=False)
        self.best_tracklet_qc.to_csv(qf, index=False)
        bp = np.asarray(self.best_tracklet_path, float)
        pd.DataFrame({"arc_mm": base.arc_mm(bp, self.spacing), "z": bp[:, 0], "y": bp[:, 1], "x": bp[:, 2]}).to_csv(pf, index=False)
        return self.tracklet_candidates

    def _bezier_bridge(self, p0, t0, p1, t1, scale_frac):
        p0m = p0 * self.spacing; p1m = p1 * self.spacing
        gap = float(np.linalg.norm(p1m - p0m))
        c1 = p0m + scale_frac * gap * base._unit(t0)
        c2 = p1m - scale_frac * gap * base._unit(t1)
        u = np.linspace(0.0, 1.0, max(8, int(math.ceil(gap / 0.20)) + 1))
        q = (((1-u)**3)[:, None] * p0m + (3*(1-u)**2*u)[:, None] * c1 + (3*(1-u)*u**2)[:, None] * c2 + (u**3)[:, None] * p1m)
        return q / self.spacing[None, :]

    def _bridge_qc(self, path):
        p = base.resample_path(path, self.spacing, 0.30); s = base.arc_mm(p, self.spacing); rows = []
        for i, pt in enumerate(p):
            i0 = max(0, i - 2); i1 = min(len(p) - 1, i + 2); t = (p[i1] - p[i0]) * self.spacing
            m, passed, score = self._measure_plane(pt, t, 1.10)
            row = {"arc_mm": float(s[i]), "plane_pass": bool(passed), "plane_score": float(score), "component_found": m is not None}
            if m is not None:
                row.update({"radius_mm": m["radius_mm"], "centroid_shift_mm": m["centroid_shift_mm"], "circularity": m["circularity"], "component_median_hu": m["component_median_hu"], "broad_plane": bool(m["radius_mm"] > 2.65)})
            else:
                row["broad_plane"] = False
            rows.append(row)
        q = pd.DataFrame(rows); found = q[q.component_found]; turns = base._path_turns_deg(p, self.spacing)
        length = float(s[-1]) if len(s) else 0.0; disp = float(np.linalg.norm((p[-1] - p[0]) * self.spacing)) if len(p) > 1 else 0.0
        m = {"bridge_length_mm": length, "bridge_tortuosity": length / max(disp, 1e-6), "bridge_pass_fraction": float(q.plane_pass.mean()) if len(q) else 0.0, "bridge_median_plane_score": float(q.plane_score.median()) if len(q) else 0.0, "bridge_broad_fraction": float(q.broad_plane.mean()) if len(q) else 1.0, "bridge_max_consecutive_failures": _max_consecutive_false(q.plane_pass.tolist()) if len(q) else 999, "bridge_p90_radius_mm": float(found.radius_mm.quantile(0.9)) if len(found) else np.nan, "bridge_max_turn_deg": float(np.max(turns)) if len(turns) else 0.0}
        m["bridge_plausible"] = bool(m["bridge_pass_fraction"] >= 0.60 and m["bridge_median_plane_score"] >= 0.55 and m["bridge_broad_fraction"] <= 0.40 and m["bridge_max_consecutive_failures"] <= 2 and m["bridge_tortuosity"] <= 1.70 and m["bridge_max_turn_deg"] <= 60.0)
        return q, m

    def assess_bridge(self):
        bf = self.cache / "bridge_candidates.csv"; qf = self.cache / "best_bridge_qc.csv"; pf = self.cache / "best_bridge_path.csv"
        if self.tracklet_candidates is None: self.search_reacquisition()
        if self.reuse["bridge"] and bf.exists() and qf.exists() and pf.exists():
            self.bridge_candidates = pd.read_csv(bf); self.best_bridge_qc = pd.read_csv(qf); self.best_bridge_path = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float); return self.bridge_candidates
        passing = self.tracklet_candidates[self.tracklet_candidates.tracklet_gate.astype(bool)] if len(self.tracklet_candidates) else pd.DataFrame()
        rows, qcs, paths = [], [], []
        if len(passing):
            best = passing.iloc[0]
            p1 = np.asarray(self.best_tracklet_path[0], float)
            t1 = np.array([best.direction_z_mm, best.direction_y_mm, best.direction_x_mm], float)
            for scale in (0.25, 0.40, 0.60, 0.80):
                path = self._bezier_bridge(self.reacquisition_origin, self.reacquisition_tangent, p1, t1, scale)
                qc, m = self._bridge_qc(path)
                rank = 0.40*m["bridge_pass_fraction"] + 0.25*m["bridge_median_plane_score"] + 0.15*(1.0-min(m["bridge_broad_fraction"],1.0)) + 0.10/max(m["bridge_tortuosity"],1.0) + 0.10*math.exp(-m["bridge_max_turn_deg"]/45.0)
                rows.append({"control_scale_fraction": float(scale), **m, "rank_score": float(rank)}); qcs.append(qc); paths.append(path)
        self.bridge_candidates = pd.DataFrame(rows)
        if len(rows):
            self.bridge_candidates = self.bridge_candidates.sort_values(["bridge_plausible", "rank_score"], ascending=[False, False]).reset_index(drop=True)
            best_scale = float(self.bridge_candidates.iloc[0].control_scale_fraction)
            idx = next(i for i, r in enumerate(rows) if float(r["control_scale_fraction"]) == best_scale)
            self.best_bridge_qc = qcs[idx]; self.best_bridge_path = paths[idx]
        else:
            self.best_bridge_qc = pd.DataFrame(columns=["arc_mm", "plane_pass", "plane_score"]); self.best_bridge_path = np.asarray([self.reacquisition_origin], float)
        self.bridge_candidates.to_csv(bf, index=False); self.best_bridge_qc.to_csv(qf, index=False)
        bp = np.asarray(self.best_bridge_path, float); pd.DataFrame({"arc_mm": base.arc_mm(bp, self.spacing), "z": bp[:, 0], "y": bp[:, 1], "x": bp[:, 2]}).to_csv(pf, index=False)
        return self.bridge_candidates

    def run(self):
        sf = self.cache / "summary.json"
        if self.ct is None: self.load_inputs()
        self._proposal_candidates(); self.search_reacquisition(); self.assess_bridge()
        nprop = int(len(self.proposal_points)); ntrack = int(len(self.tracklet_candidates)); npass = int(self.tracklet_candidates.tracklet_gate.sum()) if ntrack else 0
        if npass == 0:
            summary = {"algorithm": ALGORITHM_VERSION, "status": "NO_DISTAL_COMPACT_LUMEN_REACQUIRED", "accepted_continuation": False, "reacquired_segment_supported": False, "bridge_plausible": False, "proposal_points_tested": nprop, "tracklet_candidates_tested": ntrack, "supported_tracklets": 0, "interpretation": "No independently sampled distal proposal produced a serial coronary-sized compact lumen tracklet. No bridge inference is warranted."}
        else:
            best = self.tracklet_candidates.iloc[0].to_dict(); bridge_ok = bool(len(self.bridge_candidates) and bool(self.bridge_candidates.iloc[0].bridge_plausible))
            status = "DISTAL_COMPACT_LUMEN_REACQUIRED_BRIDGE_PLAUSIBLE" if bridge_ok else "DISTAL_COMPACT_LUMEN_REACQUIRED_BRIDGE_UNRESOLVED"
            interp = "A compact coronary-sized distal segment was independently reacquired. " + ("A short bridge is geometrically/lumen-plausible, but vessel identity remains unassigned." if bridge_ok else "The intervening bridge remains unresolved; the distal segment is not yet proven to be the same vessel.")
            summary = {"algorithm": ALGORITHM_VERSION, "status": status, "accepted_continuation": False, "reacquired_segment_supported": True, "bridge_plausible": bridge_ok, "proposal_points_tested": nprop, "tracklet_candidates_tested": ntrack, "supported_tracklets": npass, **{f"best_{k}": v for k, v in best.items()}, "interpretation": interp}
            if len(self.bridge_candidates):
                for k, v in self.bridge_candidates.iloc[0].to_dict().items(): summary[f"best_bridge_{k}"] = v
        self.summary = summary; base._write_json(summary, sf); return summary

    def make_figures(self):
        if self.summary is None: self.run()
        names = ["01_reacquisition_geometry.png", "02_proposal_landscape.png", "03_best_tracklet_planes.png", "04_bridge_validation.png"]
        seed = base.resample_path(self.seed, self.spacing, 0.15)
        fig = plt.figure(figsize=(13, 4))
        for k, (a, b, title) in enumerate([(2, 1, "X-Y"), (2, 0, "X-Z"), (1, 0, "Y-Z")], 1):
            ax = fig.add_subplot(1, 3, k); ax.plot(seed[:, a]*self.spacing[a], seed[:, b]*self.spacing[b], label="accepted branch")
            ax.scatter([self.reacquisition_origin[a]*self.spacing[a]], [self.reacquisition_origin[b]*self.spacing[b]], marker="x", s=80, label="reacquisition origin")
            if self.proposal_points is not None and len(self.proposal_points):
                pp = self.proposal_points.iloc[:40][["z","y","x"]].to_numpy(); ax.scatter(pp[:,a]*self.spacing[a], pp[:,b]*self.spacing[b], s=10, alpha=.35, label="proposals")
            if self.best_tracklet_path is not None and len(self.best_tracklet_path)>1: ax.plot(self.best_tracklet_path[:,a]*self.spacing[a], self.best_tracklet_path[:,b]*self.spacing[b], lw=3, label="best reacquired tracklet")
            if self.best_bridge_path is not None and len(self.best_bridge_path)>1: ax.plot(self.best_bridge_path[:,a]*self.spacing[a], self.best_bridge_path[:,b]*self.spacing[b], lw=2, ls="--", label="best bridge")
            ax.set_title(title); ax.set_aspect("equal", adjustable="box")
            if k == 1: ax.legend(fontsize=7)
        fig.suptitle(f"Distal compact-lumen reacquisition — {self.summary['status']}"); fig.tight_layout(); fig.savefig(self.out/names[0], dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5))
        if self.proposal_points is not None and len(self.proposal_points):
            p=self.proposal_points; sc=ax.scatter(p.forward_projection_mm,p.lateral_offset_mm,c=p.proposal_score,s=35); fig.colorbar(sc,ax=ax,label="proposal score")
        if self.tracklet_candidates is not None and len(self.tracklet_candidates):
            s=self.tracklet_candidates[self.tracklet_candidates.tracklet_gate.astype(bool)]
            if len(s): ax.scatter(s.forward_projection_mm,s.lateral_offset_mm,marker="x",s=90,label="validated tracklets")
        ax.set_xlabel("Forward projection from 12.2-mm origin (mm)"); ax.set_ylabel("Lateral offset (mm)"); ax.set_title("Independent distal proposal landscape")
        if ax.get_legend_handles_labels()[0]: ax.legend()
        fig.tight_layout(); fig.savefig(self.out/names[1], dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2,4,figsize=(11,5.5)); axes=axes.ravel()
        if self.best_tracklet_path is not None and len(self.best_tracklet_path)>1:
            p=base.resample_path(self.best_tracklet_path,self.spacing,.25); ids=np.linspace(0,len(p)-1,min(8,len(p))).astype(int)
            for ax,i in zip(axes,ids):
                i0=max(0,i-1); i1=min(len(p)-1,i+1); im,_=base.orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing,half_mm=4.2,pix_mm=.16); ax.imshow(im,cmap="gray",vmin=100,vmax=900); ax.set_title(f"tracklet {i}"); ax.axis("off")
        for ax in axes:
            if not ax.has_data(): ax.axis("off")
        fig.suptitle("Best independently reacquired compact-lumen tracklet"); fig.tight_layout(); fig.savefig(self.out/names[2],dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.8)); q=self.best_bridge_qc
        if q is not None and len(q):
            ax.plot(q.arc_mm,q.plane_score,label="plane score")
            if "radius_mm" in q: ax.plot(q.arc_mm,q.radius_mm,label="radius (mm)"); ax.axhline(2.65,ls="--",label="compact radius limit")
            ax.set_xlabel("Bridge arc (mm)"); ax.set_title("Best short-bridge validation"); ax.legend(fontsize=8)
        else: ax.text(.5,.5,"No bridge assessed",ha="center",va="center"); ax.set_axis_off()
        fig.tight_layout(); fig.savefig(self.out/names[3],dpi=160); plt.close(fig); (self.cache/"figures.done").write_text("done"); return names

    def make_report(self):
        if self.summary is None: self.run()
        self.make_figures(); html=self.out/"OPENPLAQUE_SECONDARY_DISTAL_REACQUISITION_REPORT.html"; rows="".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k,v in self.summary.items() if k!="interpretation")
        html.write_text("<html><body><h1>OpenPlaque Secondary Branch — Distal Compact-Lumen Reacquisition</h1>" + f"<p><b>Status: {self.summary['status']}</b></p><p>{self.summary.get('interpretation','')}</p><table border='1' cellpadding='4'>{rows}</table>" + "<h2>Reacquisition geometry</h2><img src='01_reacquisition_geometry.png' width='95%'><h2>Proposal landscape</h2><img src='02_proposal_landscape.png' width='85%'><h2>Best tracklet planes</h2><img src='03_best_tracklet_planes.png' width='95%'><h2>Bridge validation</h2><img src='04_bridge_validation.png' width='90%'><p>Research use only. Distal tracklets are independently detected; bridge plausibility does not assign vessel identity.</p></body></html>")
        (self.cache/"report.done").write_text("done"); return html

    def package(self):
        report=self.make_report(); zp=self.out/"OPENPLAQUE_SECONDARY_DISTAL_REACQUISITION_REPORT_BACK.zip"
        include=[report,self.cache/"input_snapshot.json",self.cache/"lumen_calibration.json",self.cache/"proposal_points.csv",self.cache/"tracklet_candidates.csv",self.cache/"best_tracklet_qc.csv",self.cache/"best_tracklet_path.csv",self.cache/"bridge_candidates.csv",self.cache/"best_bridge_qc.csv",self.cache/"best_bridge_path.csv",self.cache/"summary.json"]+[self.out/n for n in ["01_reacquisition_geometry.png","02_proposal_landscape.png","03_best_tracklet_planes.png","04_bridge_validation.png"]]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p in include:
                if p.exists(): z.write(p,arcname=p.name)
        return zp
