from __future__ import annotations

"""Dense source-CCTA distal compact-lumen reacquisition.

This experiment removes vesselness from proposal generation entirely. Starting from the
last unequivocally compact segment of the accepted secondary branch, it densely samples
a forward 3-D search volume on the source CCTA, tests many local plane orientations for
a centered coronary-sized bright component, then performs serial 7-plane validation on
the strongest source-only hits. A short bridge is assessed only after an independently
supported distal tracklet is found.

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
from scipy.spatial import cKDTree

from . import secondary_3d_vesselness_topology as base

ALGORITHM_VERSION = "secondary-dense-source-reacquisition-v1.0"


def _component_metrics(im, grid, comp):
    yy, xx = np.nonzero(comp)
    pix = float(abs(grid[1] - grid[0]))
    area = int(comp.sum()) * pix * pix
    radius = math.sqrt(area / math.pi)
    cu, cv = float(np.mean(grid[xx])), float(np.mean(grid[yy]))
    shift = float(math.hypot(cu, cv))
    er = ndi.binary_erosion(comp)
    perimeter = max(float(np.logical_and(comp, ~er).sum()) * pix, pix)
    circularity = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0, 1))
    return {
        "radius_mm": float(radius),
        "centroid_shift_mm": shift,
        "circularity": circularity,
        "component_median_hu": float(np.median(im[comp])),
        "offset_u_mm": cu,
        "offset_v_mm": cv,
    }


def _compact_component(im, grid, threshold_hu, max_shift_mm=0.85):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    out = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        if m["centroid_shift_mm"] <= max_shift_mm or dmin <= 0.40:
            out.append((m["centroid_shift_mm"] + 0.30 * dmin, m))
    return None if not out else min(out, key=lambda x: x[0])[1]


def _max_consecutive_false(values):
    best = cur = 0
    for x in values:
        if bool(x):
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return int(best)


def _angle_deg(a, b):
    a = base._unit(a)
    b = base._unit(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _direction_fan(base_dir, shell_deg=(0, 24, 42), n_azimuth=8):
    t = base._unit(np.asarray(base_dir, float))
    u, v = base._orth_basis(t)
    out = []
    for ang in shell_deg:
        if float(ang) == 0.0:
            out.append(t.copy())
            continue
        th = math.radians(float(ang))
        for j in range(int(n_azimuth)):
            ph = 2.0 * math.pi * j / float(n_azimuth)
            d = math.cos(th) * t + math.sin(th) * (
                math.cos(ph) * u + math.sin(ph) * v
            )
            out.append(base._unit(d))
    return out


def _refine_fan(base_dir, shell_deg=(0, 10, 20), n_azimuth=8):
    return _direction_fan(base_dir, shell_deg=shell_deg, n_azimuth=n_azimuth)


def synthetic_dense_source_self_test():
    grid = np.arange(-4.0, 4.0 + 1e-9, 0.16)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    tube = 100.0 + 650.0 * (((xx - 0.25) ** 2 + (yy + 0.15) ** 2) <= 1.75**2)
    broad = 100.0 + 650.0 * (xx >= -0.10)
    mt = _compact_component(tube, grid, 220.0)
    mb = _compact_component(broad, grid, 220.0)
    dirs = _direction_fan([0, 1, 0])
    passed = bool(
        mt
        and mb
        and 1.45 <= mt["radius_mm"] <= 2.05
        and mt["centroid_shift_mm"] < 0.50
        and mb["radius_mm"] > 2.65
        and len(dirs) == 17
    )
    return {
        "passed": passed,
        "tube_radius_mm": None if mt is None else mt["radius_mm"],
        "broad_radius_mm": None if mb is None else mb["radius_mm"],
        "n_coarse_orientations": len(dirs),
    }


class SecondaryDenseSourceReacquisitionWorkflow:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_Dense_Source_Reacquisition_v1"
        self.out = self.root / "Secondary_Dense_Source_Reacquisition_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {
            "inputs": True,
            "dense_scan": True,
            "reacquisition": True,
            "bridge": True,
            "figures": True,
            "report": True,
        }
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.spacing = None
        self.ct = None
        self.seed = None
        self.reference = None
        self.trunk = None
        self.calibration = None
        self.reacquisition_origin = None
        self.reacquisition_tangent = None
        self.reacquisition_origin_arc = None
        self.center_hits = None
        self.scan_summary = None
        self.tracklet_candidates = None
        self.best_tracklet_qc = None
        self.best_tracklet_path = None
        self.bridge_candidates = None
        self.best_bridge_qc = None
        self.best_bridge_path = None
        self.summary = None

    def cache_status(self):
        names = {
            "inputs": "input_snapshot.json",
            "dense_scan": "dense_center_hits.csv",
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
        src = self.root / "Cache" / "Secondary_3D_Vesselness_Topology_v1"
        need = [src / "series7_int16.npy", src / "series7_int16.json"]
        seedf = (
            self.root
            / "Secondary_Branch_Lateral_Divergence_Report"
            / "branch_centerline.csv"
        )
        reff = (
            self.root
            / "LAD_Takeoff_Root_Alternatives_Report"
            / "alternative_01_centerline.csv"
        )
        trunkf = (
            self.root
            / "LAD_Takeoff_Confirmation_Report"
            / "trunk_centerline.csv"
        )
        need += [seedf, reff, trunkf]
        if not all(p.exists() for p in need):
            raise FileNotFoundError(
                "Missing source-CCTA/accepted-anatomy outputs: "
                + "; ".join(str(p) for p in need if not p.exists())
            )
        meta = json.loads((src / "series7_int16.json").read_text())
        self.spacing = np.asarray(meta["spacing_zyx"], float)
        self.ct = np.load(src / "series7_int16.npy", mmap_mode="r")
        self.seed = pd.read_csv(seedf)[["z", "y", "x"]].to_numpy(float)
        self.reference = pd.read_csv(reff)[["z", "y", "x"]].to_numpy(float)
        self.trunk = pd.read_csv(trunkf)[["z", "y", "x"]].to_numpy(float)

        sp = base.resample_path(self.seed, self.spacing, 0.10)
        ss = base.arc_mm(sp, self.spacing)
        self.reacquisition_origin_arc = float(reacquisition_origin_arc_mm)
        i = int(np.argmin(np.abs(ss - self.reacquisition_origin_arc)))
        self.reacquisition_origin = sp[i].copy()
        i0 = max(0, i - 8)
        self.reacquisition_tangent = base._unit(
            (sp[i] - sp[i0]) * self.spacing
        )
        self._calibrate_lumen()

        snap = {
            "algorithm": ALGORITHM_VERSION,
            "source_representation": "source-CCTA only; no vesselness proposals or scores",
            "reacquisition_origin_arc_mm": self.reacquisition_origin_arc,
            "reacquisition_origin_zyx": self.reacquisition_origin.tolist(),
            "search_longitudinal_mm": [1.0, 5.6],
            "search_lateral_mm": 4.0,
            "dense_lattice_step_mm": {
                "longitudinal": 0.45,
                "lateral_u": 0.55,
                "lateral_v": 0.55,
            },
            "minimum_distance_from_accepted_seed_mm": 0.65,
            "minimum_distance_from_lad_trunk_mm": 1.50,
            "coarse_orientation_hypotheses": 17,
            "serial_tracklet_half_length_mm": 0.75,
            "serial_tracklet_plane_step_mm": 0.25,
            "search_target": None,
            "note": (
                "Candidate positions and orientations are generated from source CCTA only. "
                "No vesselness field is loaded or used. A distal segment requires serial "
                "compact coronary-sized lumen validation before bridge assessment."
            ),
        }
        base._write_json(snap, self.cache / "input_snapshot.json")
        return snap

    def _calibrate_lumen(self):
        sp = base.resample_path(self.seed, self.spacing, 0.30)
        s = base.arc_mm(sp, self.spacing)
        p = sp[(s >= 10.0) & (s <= 12.7)]
        center = ndi.map_coordinates(
            np.asarray(self.ct),
            [p[:, 0], p[:, 1], p[:, 2]],
            order=1,
            mode="nearest",
        )
        refhu = float(np.median(center))
        thr = float(max(170.0, min(300.0, 0.38 * refhu)))
        rows = []
        for i, pt in enumerate(p):
            i0, i1 = max(0, i - 2), min(len(p) - 1, i + 2)
            t = (p[i1] - p[i0]) * self.spacing
            im, grid = base.orthogonal_plane(
                self.ct, pt, t, self.spacing, half_mm=4.2, pix_mm=0.16
            )
            m = _compact_component(im, grid, thr, max_shift_mm=0.80)
            if m is not None:
                rows.append(m)
        if not rows:
            raise RuntimeError("Could not calibrate lumen on accepted secondary branch")
        df = pd.DataFrame(rows)
        good = df[
            (df.radius_mm >= 0.65)
            & (df.radius_mm <= 2.65)
            & (df.centroid_shift_mm <= 0.80)
        ]
        if len(good) < 3:
            good = df
        self.calibration = {
            "reference_center_hu": refhu,
            "bright_threshold_hu": thr,
            "median_radius_mm": float(good.radius_mm.median()),
            "p90_radius_mm": float(good.radius_mm.quantile(0.9)),
            "median_circularity": float(good.circularity.median()),
            "n_planes": int(len(good)),
        }
        base._write_json(
            self.calibration, self.cache / "lumen_calibration.json"
        )

    def _plane_measure(self, point, direction, half_mm=4.0, pix_mm=0.18):
        im, grid = base.orthogonal_plane(
            self.ct,
            point,
            direction,
            self.spacing,
            half_mm=half_mm,
            pix_mm=pix_mm,
        )
        m = _compact_component(
            im,
            grid,
            self.calibration["bright_threshold_hu"],
            max_shift_mm=0.90,
        )
        if m is None:
            return im, None, False, 0.0
        rref = float(self.calibration["median_radius_mm"])
        passed = bool(
            0.65 <= m["radius_mm"] <= 2.65
            and m["centroid_shift_mm"] <= 0.75
            and m["circularity"] >= 0.30
            and m["component_median_hu"]
            >= max(220.0, 0.45 * self.calibration["reference_center_hu"])
        )
        rscore = math.exp(
            -0.5
            * (
                (m["radius_mm"] - rref)
                / max(0.55 * rref, 0.70)
            )
            ** 2
        )
        sscore = math.exp(
            -0.5 * (m["centroid_shift_mm"] / 0.55) ** 2
        )
        cscore = float(
            np.clip(
                m["circularity"]
                / max(self.calibration["median_circularity"], 0.35),
                0,
                1,
            )
        )
        hscore = float(
            np.clip(
                m["component_median_hu"]
                / max(self.calibration["reference_center_hu"], 1.0),
                0,
                1.2,
            )
            / 1.2
        )
        score = float(0.35 * rscore + 0.35 * sscore + 0.18 * cscore + 0.12 * hscore)
        return im, m, passed, score

    def dense_scan(
        self,
        longitudinal_step_mm=0.45,
        lateral_step_mm=0.55,
        max_center_hits=180,
        nms_mm=0.55,
    ):
        if self.ct is None:
            self.load_inputs()
        hf = self.cache / "dense_center_hits.csv"
        sf = self.cache / "dense_scan_summary.json"
        if self.reuse["dense_scan"] and hf.exists() and sf.exists():
            self.center_hits = pd.read_csv(hf)
            self.scan_summary = json.loads(sf.read_text())
            return self.center_hits

        t = self.reacquisition_tangent
        u, v = base._orth_basis(t)
        origin_phys = self.reacquisition_origin * self.spacing
        seed_phys = base.resample_path(self.seed, self.spacing, 0.15) * self.spacing
        ref_phys = np.vstack([self.reference, self.trunk]) * self.spacing
        seed_tree = cKDTree(seed_phys)
        ref_tree = cKDTree(ref_phys)

        svals = np.arange(1.0, 5.6 + 1e-9, float(longitudinal_step_mm))
        lat = np.arange(-4.0, 4.0 + 1e-9, float(lateral_step_mm))
        rows = []
        n_positions = 0
        n_hu = 0
        n_geometry = 0
        n_planes = 0

        for s in svals:
            for a in lat:
                for b in lat:
                    lateral = float(math.hypot(a, b))
                    if lateral > 4.0:
                        continue
                    phys = origin_phys + s * t + a * u + b * v
                    p = phys / self.spacing
                    n_positions += 1
                    if np.any(p < 2.0) or np.any(p >= np.asarray(self.ct.shape) - 3.0):
                        continue
                    seed_d = float(seed_tree.query(phys, k=1)[0])
                    ref_d = float(ref_tree.query(phys, k=1)[0])
                    if seed_d < 0.65 or ref_d < 1.50:
                        continue
                    n_geometry += 1
                    hu = float(
                        ndi.map_coordinates(
                            np.asarray(self.ct),
                            [[p[0]], [p[1]], [p[2]]],
                            order=1,
                            mode="nearest",
                        )[0]
                    )
                    if not (
                        self.calibration["bright_threshold_hu"]
                        <= hu
                        <= 1000.0
                    ):
                        continue
                    n_hu += 1
                    radial = base._unit(phys - origin_phys)
                    seed_dir = base._unit(0.55 * radial + 0.45 * t)
                    best = None
                    for d in _direction_fan(seed_dir):
                        n_planes += 1
                        _, m, passed, score = self._plane_measure(
                            p, d, half_mm=3.8, pix_mm=0.18
                        )
                        if m is None:
                            continue
                        continuity = _angle_deg(d, t)
                        radial_angle = _angle_deg(d, radial)
                        compact = bool(
                            passed
                            and m["radius_mm"] <= 2.65
                            and m["centroid_shift_mm"] <= 0.75
                        )
                        rank = (
                            0.62 * score
                            + 0.14 * math.exp(-continuity / 50.0)
                            + 0.10 * math.exp(-radial_angle / 35.0)
                            + 0.08 * min(s / 5.0, 1.0)
                            + 0.06 * math.exp(-lateral / 2.5)
                        )
                        rec = {
                            "z": float(p[0]),
                            "y": float(p[1]),
                            "x": float(p[2]),
                            "longitudinal_mm": float(s),
                            "lateral_u_mm": float(a),
                            "lateral_v_mm": float(b),
                            "lateral_offset_mm": lateral,
                            "center_hu": hu,
                            "distance_to_accepted_seed_mm": seed_d,
                            "distance_to_lad_trunk_mm": ref_d,
                            "center_plane_pass": compact,
                            "center_plane_score": float(score),
                            "center_radius_mm": float(m["radius_mm"]),
                            "center_shift_mm": float(m["centroid_shift_mm"]),
                            "center_circularity": float(m["circularity"]),
                            "center_component_median_hu": float(
                                m["component_median_hu"]
                            ),
                            "orientation_continuity_deg": continuity,
                            "orientation_radial_deg": radial_angle,
                            "direction_z_mm": float(d[0]),
                            "direction_y_mm": float(d[1]),
                            "direction_x_mm": float(d[2]),
                            "rank_score": float(rank),
                        }
                        if best is None or (
                            bool(rec["center_plane_pass"]),
                            rec["rank_score"],
                        ) > (
                            bool(best["center_plane_pass"]),
                            best["rank_score"],
                        ):
                            best = rec
                    if best is not None and best["center_plane_pass"]:
                        rows.append(best)

        raw = pd.DataFrame(rows)
        if len(raw):
            raw = raw.sort_values("rank_score", ascending=False).reset_index(drop=True)
            chosen = []
            kept = []
            for _, r in raw.iterrows():
                p = np.array([r.z, r.y, r.x], float)
                if any(
                    np.linalg.norm((p - q) * self.spacing) < float(nms_mm)
                    for q in chosen
                ):
                    continue
                chosen.append(p)
                kept.append(r.to_dict())
                if len(kept) >= int(max_center_hits):
                    break
            self.center_hits = pd.DataFrame(kept)
            if len(self.center_hits):
                self.center_hits.insert(
                    0, "hit_id", np.arange(1, len(self.center_hits) + 1)
                )
        else:
            self.center_hits = pd.DataFrame(
                columns=[
                    "hit_id",
                    "z",
                    "y",
                    "x",
                    "longitudinal_mm",
                    "lateral_offset_mm",
                    "center_hu",
                    "center_plane_score",
                    "center_radius_mm",
                    "center_shift_mm",
                    "direction_z_mm",
                    "direction_y_mm",
                    "direction_x_mm",
                    "rank_score",
                ]
            )

        self.scan_summary = {
            "algorithm": ALGORITHM_VERSION,
            "positions_considered": int(n_positions),
            "positions_after_geometry_exclusion": int(n_geometry),
            "positions_after_center_hu_prefilter": int(n_hu),
            "orientation_planes_tested": int(n_planes),
            "raw_compact_center_hits": int(len(raw)),
            "nms_compact_center_hits": int(len(self.center_hits)),
            "proposal_generation_used_vesselness": False,
        }
        self.center_hits.to_csv(hf, index=False)
        base._write_json(self.scan_summary, sf)
        return self.center_hits

    def _evaluate_tracklet(
        self,
        center,
        direction,
        half_mm=0.75,
        step_mm=0.25,
    ):
        offsets = np.arange(
            -half_mm, half_mm + 1e-9, float(step_mm)
        )
        rows = []
        corrected = []
        u, v = base._orth_basis(direction)
        for s in offsets:
            p = center + (float(s) * direction) / self.spacing
            _, m, passed, score = self._plane_measure(
                p, direction, half_mm=4.2, pix_mm=0.16
            )
            row = {
                "tracklet_offset_mm": float(s),
                "z": float(p[0]),
                "y": float(p[1]),
                "x": float(p[2]),
                "component_found": m is not None,
                "plane_pass": bool(passed),
                "plane_score": float(score),
            }
            if m is not None:
                corr = p + (
                    m["offset_u_mm"] * u + m["offset_v_mm"] * v
                ) / self.spacing
                row.update(
                    {
                        "radius_mm": m["radius_mm"],
                        "centroid_shift_mm": m["centroid_shift_mm"],
                        "circularity": m["circularity"],
                        "component_median_hu": m["component_median_hu"],
                        "corrected_z": float(corr[0]),
                        "corrected_y": float(corr[1]),
                        "corrected_x": float(corr[2]),
                    }
                )
                corrected.append(corr)
            rows.append(row)

        q = pd.DataFrame(rows)
        found = q[q.component_found]
        center_row = q.iloc[
            int(np.argmin(np.abs(q.tracklet_offset_mm.to_numpy())))
        ]
        metrics = {
            "tracklet_pass_fraction": float(q.plane_pass.mean()),
            "tracklet_center_pass": bool(center_row.plane_pass),
            "tracklet_median_score": float(q.plane_score.median()),
            "tracklet_min_score": float(q.plane_score.min()),
            "tracklet_max_consecutive_failures": _max_consecutive_false(
                q.plane_pass.tolist()
            ),
            "tracklet_median_radius_mm": (
                float(found.radius_mm.median()) if len(found) else np.nan
            ),
            "tracklet_p90_radius_mm": (
                float(found.radius_mm.quantile(0.9)) if len(found) else np.nan
            ),
            "tracklet_median_shift_mm": (
                float(found.centroid_shift_mm.median()) if len(found) else np.nan
            ),
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
        if len(corrected) >= 2:
            path = np.asarray(corrected, float)
        else:
            path = np.asarray(
                [
                    center - half_mm * direction / self.spacing,
                    center + half_mm * direction / self.spacing,
                ],
                float,
            )
        path = base.resample_path(path, self.spacing, 0.20)
        return q, metrics, path

    def search_reacquisition(self, top_center_hits=120):
        cf = self.cache / "tracklet_candidates.csv"
        qf = self.cache / "best_tracklet_qc.csv"
        pf = self.cache / "best_tracklet_path.csv"
        if self.center_hits is None:
            self.dense_scan()
        if (
            self.reuse["reacquisition"]
            and cf.exists()
            and qf.exists()
            and pf.exists()
        ):
            self.tracklet_candidates = pd.read_csv(cf)
            self.best_tracklet_qc = pd.read_csv(qf)
            self.best_tracklet_path = pd.read_csv(pf)[
                ["z", "y", "x"]
            ].to_numpy(float)
            return self.tracklet_candidates

        rows = []
        qcs = []
        paths = []
        for _, hit in self.center_hits.head(int(top_center_hits)).iterrows():
            center = np.array([hit.z, hit.y, hit.x], float)
            d0 = np.array(
                [
                    hit.direction_z_mm,
                    hit.direction_y_mm,
                    hit.direction_x_mm,
                ],
                float,
            )
            best = None
            for d in _refine_fan(d0):
                qc, metrics, path = self._evaluate_tracklet(center, d)
                continuity = _angle_deg(d, self.reacquisition_tangent)
                rank = (
                    0.42 * metrics["tracklet_pass_fraction"]
                    + 0.32 * metrics["tracklet_median_score"]
                    + 0.12 * math.exp(-continuity / 45.0)
                    + 0.08 * float(hit.center_plane_score)
                    + 0.06 * min(float(hit.longitudinal_mm) / 5.0, 1.0)
                )
                rec = {
                    "hit_id": int(hit.hit_id),
                    "z": float(center[0]),
                    "y": float(center[1]),
                    "x": float(center[2]),
                    "longitudinal_mm": float(hit.longitudinal_mm),
                    "lateral_offset_mm": float(hit.lateral_offset_mm),
                    "center_plane_score": float(hit.center_plane_score),
                    "center_radius_mm": float(hit.center_radius_mm),
                    "orientation_continuity_deg": continuity,
                    **metrics,
                    "rank_score": float(rank),
                    "direction_z_mm": float(d[0]),
                    "direction_y_mm": float(d[1]),
                    "direction_x_mm": float(d[2]),
                }
                if best is None or (
                    bool(rec["tracklet_gate"]),
                    rec["rank_score"],
                ) > (
                    bool(best[0]["tracklet_gate"]),
                    best[0]["rank_score"],
                ):
                    best = (rec, qc, path)
            if best is not None:
                rec, qc, path = best
                rows.append(rec)
                qcs.append(qc)
                paths.append(path)

        self.tracklet_candidates = pd.DataFrame(rows)
        if len(self.tracklet_candidates):
            self.tracklet_candidates = self.tracklet_candidates.sort_values(
                ["tracklet_gate", "rank_score"],
                ascending=[False, False],
            ).reset_index(drop=True)
            best_hit = int(self.tracklet_candidates.iloc[0].hit_id)
            idx = next(
                i for i, r in enumerate(rows) if int(r["hit_id"]) == best_hit
            )
            self.best_tracklet_qc = qcs[idx]
            self.best_tracklet_path = paths[idx]
        else:
            self.best_tracklet_qc = pd.DataFrame(
                columns=[
                    "tracklet_offset_mm",
                    "plane_pass",
                    "plane_score",
                    "radius_mm",
                    "centroid_shift_mm",
                ]
            )
            self.best_tracklet_path = np.asarray(
                [self.reacquisition_origin], float
            )

        self.tracklet_candidates.to_csv(cf, index=False)
        self.best_tracklet_qc.to_csv(qf, index=False)
        bp = np.asarray(self.best_tracklet_path, float)
        pd.DataFrame(
            {
                "arc_mm": base.arc_mm(bp, self.spacing),
                "z": bp[:, 0],
                "y": bp[:, 1],
                "x": bp[:, 2],
            }
        ).to_csv(pf, index=False)
        return self.tracklet_candidates

    def _bezier_bridge(self, p0, t0, p1, t1, scale_frac):
        p0m = p0 * self.spacing
        p1m = p1 * self.spacing
        gap = float(np.linalg.norm(p1m - p0m))
        c1 = p0m + scale_frac * gap * base._unit(t0)
        c2 = p1m - scale_frac * gap * base._unit(t1)
        uu = np.linspace(
            0.0, 1.0, max(8, int(math.ceil(gap / 0.20)) + 1)
        )
        q = (
            ((1 - uu) ** 3)[:, None] * p0m
            + (3 * (1 - uu) ** 2 * uu)[:, None] * c1
            + (3 * (1 - uu) * uu**2)[:, None] * c2
            + (uu**3)[:, None] * p1m
        )
        return q / self.spacing[None, :]

    def _bridge_qc(self, path):
        p = base.resample_path(path, self.spacing, 0.30)
        s = base.arc_mm(p, self.spacing)
        rows = []
        for i, pt in enumerate(p):
            i0, i1 = max(0, i - 2), min(len(p) - 1, i + 2)
            t = (p[i1] - p[i0]) * self.spacing
            _, m, passed, score = self._plane_measure(
                pt, t, half_mm=4.2, pix_mm=0.16
            )
            row = {
                "arc_mm": float(s[i]),
                "plane_pass": bool(passed),
                "plane_score": float(score),
                "component_found": m is not None,
            }
            if m is not None:
                row.update(
                    {
                        "radius_mm": m["radius_mm"],
                        "centroid_shift_mm": m["centroid_shift_mm"],
                        "circularity": m["circularity"],
                        "component_median_hu": m["component_median_hu"],
                        "broad_plane": bool(m["radius_mm"] > 2.65),
                    }
                )
            else:
                row["broad_plane"] = False
            rows.append(row)
        q = pd.DataFrame(rows)
        found = q[q.component_found]
        turns = base._path_turns_deg(p, self.spacing)
        length = float(s[-1]) if len(s) else 0.0
        disp = (
            float(np.linalg.norm((p[-1] - p[0]) * self.spacing))
            if len(p) > 1
            else 0.0
        )
        metrics = {
            "bridge_length_mm": length,
            "bridge_tortuosity": length / max(disp, 1e-6),
            "bridge_pass_fraction": (
                float(q.plane_pass.mean()) if len(q) else 0.0
            ),
            "bridge_median_plane_score": (
                float(q.plane_score.median()) if len(q) else 0.0
            ),
            "bridge_broad_fraction": (
                float(q.broad_plane.mean()) if len(q) else 1.0
            ),
            "bridge_max_consecutive_failures": (
                _max_consecutive_false(q.plane_pass.tolist()) if len(q) else 999
            ),
            "bridge_p90_radius_mm": (
                float(found.radius_mm.quantile(0.9)) if len(found) else np.nan
            ),
            "bridge_max_turn_deg": (
                float(np.max(turns)) if len(turns) else 0.0
            ),
        }
        metrics["bridge_plausible"] = bool(
            metrics["bridge_pass_fraction"] >= 0.60
            and metrics["bridge_median_plane_score"] >= 0.55
            and metrics["bridge_broad_fraction"] <= 0.40
            and metrics["bridge_max_consecutive_failures"] <= 2
            and metrics["bridge_tortuosity"] <= 1.70
            and metrics["bridge_max_turn_deg"] <= 60.0
        )
        return q, metrics

    def assess_bridge(self):
        bf = self.cache / "bridge_candidates.csv"
        qf = self.cache / "best_bridge_qc.csv"
        pf = self.cache / "best_bridge_path.csv"
        if self.tracklet_candidates is None:
            self.search_reacquisition()
        if (
            self.reuse["bridge"]
            and bf.exists()
            and qf.exists()
            and pf.exists()
        ):
            try:
                self.bridge_candidates = pd.read_csv(bf)
            except pd.errors.EmptyDataError:
                self.bridge_candidates = pd.DataFrame(
                    columns=["control_scale_fraction", "bridge_plausible"]
                )
            self.best_bridge_qc = pd.read_csv(qf)
            self.best_bridge_path = pd.read_csv(pf)[
                ["z", "y", "x"]
            ].to_numpy(float)
            return self.bridge_candidates

        passing = (
            self.tracklet_candidates[
                self.tracklet_candidates.tracklet_gate.astype(bool)
            ]
            if len(self.tracklet_candidates)
            else pd.DataFrame()
        )
        rows = []
        qcs = []
        paths = []
        if len(passing):
            best = passing.iloc[0]
            p1 = np.asarray(self.best_tracklet_path[0], float)
            t1 = np.array(
                [
                    best.direction_z_mm,
                    best.direction_y_mm,
                    best.direction_x_mm,
                ],
                float,
            )
            for scale in (0.25, 0.40, 0.60, 0.80):
                path = self._bezier_bridge(
                    self.reacquisition_origin,
                    self.reacquisition_tangent,
                    p1,
                    t1,
                    scale,
                )
                qc, metrics = self._bridge_qc(path)
                rank = (
                    0.40 * metrics["bridge_pass_fraction"]
                    + 0.25 * metrics["bridge_median_plane_score"]
                    + 0.15 * (1.0 - min(metrics["bridge_broad_fraction"], 1.0))
                    + 0.10 / max(metrics["bridge_tortuosity"], 1.0)
                    + 0.10 * math.exp(-metrics["bridge_max_turn_deg"] / 45.0)
                )
                rows.append(
                    {
                        "control_scale_fraction": float(scale),
                        **metrics,
                        "rank_score": float(rank),
                    }
                )
                qcs.append(qc)
                paths.append(path)

        self.bridge_candidates = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "control_scale_fraction",
                    "bridge_plausible",
                    "rank_score",
                ]
            )
        )
        if rows:
            self.bridge_candidates = self.bridge_candidates.sort_values(
                ["bridge_plausible", "rank_score"],
                ascending=[False, False],
            ).reset_index(drop=True)
            best_scale = float(
                self.bridge_candidates.iloc[0].control_scale_fraction
            )
            idx = next(
                i
                for i, r in enumerate(rows)
                if float(r["control_scale_fraction"]) == best_scale
            )
            self.best_bridge_qc = qcs[idx]
            self.best_bridge_path = paths[idx]
        else:
            self.best_bridge_qc = pd.DataFrame(
                columns=[
                    "arc_mm",
                    "plane_pass",
                    "plane_score",
                    "radius_mm",
                    "centroid_shift_mm",
                ]
            )
            self.best_bridge_path = np.asarray(
                [self.reacquisition_origin], float
            )

        self.bridge_candidates.to_csv(bf, index=False)
        self.best_bridge_qc.to_csv(qf, index=False)
        bp = np.asarray(self.best_bridge_path, float)
        pd.DataFrame(
            {
                "arc_mm": base.arc_mm(bp, self.spacing),
                "z": bp[:, 0],
                "y": bp[:, 1],
                "x": bp[:, 2],
            }
        ).to_csv(pf, index=False)
        return self.bridge_candidates

    def run(self):
        if self.ct is None:
            self.load_inputs()
        self.dense_scan()
        self.search_reacquisition()
        self.assess_bridge()

        nhit = int(len(self.center_hits))
        ntrack = int(len(self.tracklet_candidates))
        npass = (
            int(self.tracklet_candidates.tracklet_gate.sum())
            if ntrack
            else 0
        )
        if npass == 0:
            summary = {
                "algorithm": ALGORITHM_VERSION,
                "status": "NO_DISTAL_COMPACT_LUMEN_REACQUIRED_DENSE_SOURCE",
                "accepted_continuation": False,
                "reacquired_segment_supported": False,
                "bridge_plausible": False,
                "dense_compact_center_hits": nhit,
                "tracklet_candidates_tested": ntrack,
                "supported_tracklets": 0,
                **(
                    {
                        f"scan_{k}": v
                        for k, v in (self.scan_summary or {}).items()
                        if k != "algorithm"
                    }
                ),
                "interpretation": (
                    "A dense source-CCTA-only scan found no independently sampled distal "
                    "location/orientation with a serial coronary-sized compact lumen tracklet. "
                    "No bridge inference is warranted."
                ),
            }
            if ntrack:
                best = self.tracklet_candidates.iloc[0].to_dict()
                summary.update(
                    {f"best_{k}": v for k, v in best.items()}
                )
        else:
            best = self.tracklet_candidates.iloc[0].to_dict()
            bridge_ok = bool(
                len(self.bridge_candidates)
                and bool(self.bridge_candidates.iloc[0].bridge_plausible)
            )
            status = (
                "DISTAL_COMPACT_LUMEN_REACQUIRED_DENSE_SOURCE_BRIDGE_PLAUSIBLE"
                if bridge_ok
                else "DISTAL_COMPACT_LUMEN_REACQUIRED_DENSE_SOURCE_BRIDGE_UNRESOLVED"
            )
            summary = {
                "algorithm": ALGORITHM_VERSION,
                "status": status,
                "accepted_continuation": False,
                "reacquired_segment_supported": True,
                "bridge_plausible": bridge_ok,
                "dense_compact_center_hits": nhit,
                "tracklet_candidates_tested": ntrack,
                "supported_tracklets": npass,
                **{f"best_{k}": v for k, v in best.items()},
                **(
                    {
                        f"scan_{k}": v
                        for k, v in (self.scan_summary or {}).items()
                        if k != "algorithm"
                    }
                ),
                "interpretation": (
                    "A compact coronary-sized distal segment was independently reacquired "
                    "using source CCTA without vesselness proposals. "
                    + (
                        "A short bridge is geometrically/lumen-plausible, but vessel identity "
                        "remains unassigned."
                        if bridge_ok
                        else "The intervening bridge remains unresolved; the distal segment is "
                        "not yet proven to be the same vessel."
                    )
                ),
            }
            if len(self.bridge_candidates):
                for k, v in self.bridge_candidates.iloc[0].to_dict().items():
                    summary[f"best_bridge_{k}"] = v

        self.summary = summary
        base._write_json(summary, self.cache / "summary.json")
        return summary

    def make_figures(self):
        if self.summary is None:
            self.run()
        names = [
            "01_dense_reacquisition_geometry.png",
            "02_dense_scan_landscape.png",
            "03_best_tracklet_planes.png",
            "04_bridge_validation.png",
        ]
        seed = base.resample_path(self.seed, self.spacing, 0.15)
        fig = plt.figure(figsize=(13, 4))
        for k, (a, b, title) in enumerate(
            [(2, 1, "X-Y"), (2, 0, "X-Z"), (1, 0, "Y-Z")], 1
        ):
            ax = fig.add_subplot(1, 3, k)
            ax.plot(
                seed[:, a] * self.spacing[a],
                seed[:, b] * self.spacing[b],
                label="accepted branch",
            )
            ax.scatter(
                [self.reacquisition_origin[a] * self.spacing[a]],
                [self.reacquisition_origin[b] * self.spacing[b]],
                marker="x",
                s=70,
                label="12.2 mm origin",
            )
            if self.center_hits is not None and len(self.center_hits):
                hh = self.center_hits.head(80)[["z", "y", "x"]].to_numpy(float)
                ax.scatter(
                    hh[:, a] * self.spacing[a],
                    hh[:, b] * self.spacing[b],
                    s=10,
                    alpha=0.35,
                    label="source-only compact hits",
                )
            if self.best_tracklet_path is not None and len(self.best_tracklet_path) > 1:
                pp = np.asarray(self.best_tracklet_path, float)
                ax.plot(
                    pp[:, a] * self.spacing[a],
                    pp[:, b] * self.spacing[b],
                    lw=3,
                    label="best distal tracklet",
                )
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            if k == 1:
                ax.legend(fontsize=7)
        fig.suptitle(f"Dense source reacquisition — {self.summary['status']}")
        fig.tight_layout()
        fig.savefig(self.out / names[0], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        if self.center_hits is not None and len(self.center_hits):
            ax.scatter(
                self.center_hits.longitudinal_mm,
                self.center_hits.lateral_offset_mm,
                s=20,
            )
            ax.set_xlabel("Longitudinal distance from 12.2 mm origin (mm)")
            ax.set_ylabel("Lateral offset (mm)")
            ax.set_title("Source-only compact center-plane hits")
        else:
            ax.text(
                0.5,
                0.5,
                "No compact center-plane hits in dense scan",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(self.out / names[1], dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
        axes = axes.ravel()
        p = (
            np.asarray(self.best_tracklet_path, float)
            if self.best_tracklet_path is not None
            else np.empty((0, 3))
        )
        if len(p) >= 2:
            rp = base.resample_path(p, self.spacing, 0.22)
            ids = np.linspace(
                0, len(rp) - 1, min(8, len(rp))
            ).astype(int)
            for ax, i in zip(axes, ids):
                i0, i1 = max(0, i - 2), min(len(rp) - 1, i + 2)
                im, _ = base.orthogonal_plane(
                    self.ct,
                    rp[i],
                    (rp[i1] - rp[i0]) * self.spacing,
                    self.spacing,
                    half_mm=4.2,
                    pix_mm=0.16,
                )
                ax.imshow(im, cmap="gray", vmin=100, vmax=900)
                ax.axis("off")
        for ax in axes:
            if not ax.has_data():
                ax.axis("off")
        fig.suptitle("Best dense-source distal tracklet")
        fig.tight_layout()
        fig.savefig(self.out / names[2], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        if self.best_bridge_qc is not None and len(self.best_bridge_qc):
            q = self.best_bridge_qc
            ax.plot(q.arc_mm, q.plane_score, label="plane score")
            if "radius_mm" in q:
                ax.plot(q.arc_mm, q.radius_mm, label="radius (mm)")
            ax.set_xlabel("Bridge arc (mm)")
            ax.set_title("Best bridge validation")
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No bridge assessed without a supported distal tracklet",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(self.out / names[3], dpi=160)
        plt.close(fig)

        (self.cache / "figures.done").write_text("done")
        return names

    def make_report(self):
        if self.summary is None:
            self.run()
        self.make_figures()
        html = (
            self.out
            / "OPENPLAQUE_SECONDARY_DENSE_SOURCE_REACQUISITION_REPORT.html"
        )
        rows = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in self.summary.items()
            if k != "interpretation"
        )
        html.write_text(
            "<html><body>"
            "<h1>OpenPlaque Secondary Branch — Dense Source-CCTA Reacquisition</h1>"
            f"<p><b>Status: {self.summary['status']}</b></p>"
            f"<p>{self.summary.get('interpretation', '')}</p>"
            f"<table border='1' cellpadding='4'>{rows}</table>"
            "<h2>Geometry</h2><img src='01_dense_reacquisition_geometry.png' width='95%'>"
            "<h2>Dense scan landscape</h2><img src='02_dense_scan_landscape.png' width='85%'>"
            "<h2>Best distal tracklet</h2><img src='03_best_tracklet_planes.png' width='90%'>"
            "<h2>Bridge validation</h2><img src='04_bridge_validation.png' width='85%'>"
            "<p>Research use only. Proposal generation is source-CCTA only and uses no "
            "vesselness field. Vessel identity remains unassigned.</p>"
            "</body></html>"
        )
        (self.cache / "report.done").write_text("done")
        return html

    def package(self):
        report = self.make_report()
        zip_path = (
            self.out
            / "OPENPLAQUE_SECONDARY_DENSE_SOURCE_REACQUISITION_REPORT_BACK.zip"
        )
        include = [
            report,
            self.cache / "input_snapshot.json",
            self.cache / "lumen_calibration.json",
            self.cache / "dense_scan_summary.json",
            self.cache / "dense_center_hits.csv",
            self.cache / "tracklet_candidates.csv",
            self.cache / "best_tracklet_qc.csv",
            self.cache / "best_tracklet_path.csv",
            self.cache / "bridge_candidates.csv",
            self.cache / "best_bridge_qc.csv",
            self.cache / "best_bridge_path.csv",
            self.cache / "summary.json",
        ]
        include += [self.out / name for name in self.make_figures()]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in include:
                if p.exists():
                    z.write(p, arcname=p.name)
        return zip_path
