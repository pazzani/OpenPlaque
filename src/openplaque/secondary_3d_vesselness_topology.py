from __future__ import annotations

"""3-D multiscale vesselness/topology continuation for the accepted secondary branch.

This experiment changes representation from local 2-D connected-component tracking to a
3-D Hessian vesselness field plus topology-aware geodesic search. It starts inside the
known-good secondary branch and must first rediscover a known 2-3 mm segment before any
new distal continuation can be considered.

Research use only. No LCX identity is assigned automatically.
"""

import heapq
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "secondary-3d-vesselness-topology-v1.0"


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) == 0:
        return np.array([], float)
    if len(p) == 1:
        return np.array([0.0])
    d = np.linalg.norm(np.diff(p, axis=0) * np.asarray(spacing, float), axis=1)
    return np.r_[0.0, np.cumsum(d)]


def resample_path(path, spacing, step_mm=0.20):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return p.copy()
    s = arc_mm(p, spacing)
    if s[-1] <= step_mm:
        return p.copy()
    x = np.arange(0.0, s[-1] + 1e-9, step_mm)
    if x[-1] < s[-1] - 0.05:
        x = np.r_[x, s[-1]]
    return np.column_stack([np.interp(x, s, p[:, j]) for j in range(3)])


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def _nearest_dist_mm(point, ref, spacing):
    p = np.asarray(point, float) * np.asarray(spacing, float)
    r = np.asarray(ref, float) * np.asarray(spacing, float)
    return float(np.min(np.linalg.norm(r - p, axis=1)))


def _sample_trilinear(vol, pts_zyx_local):
    pts = np.asarray(pts_zyx_local, float)
    return ndi.map_coordinates(
        np.asarray(vol, float),
        [pts[:, 0], pts[:, 1], pts[:, 2]],
        order=1,
        mode="nearest",
    )


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing, half_mm=4.5, pix_mm=0.18):
    t = _unit(tangent_zyx_mm)
    u, v = _orth_basis(t)
    grid = np.arange(-half_mm, half_mm + 1e-9, pix_mm)
    vv, uu = np.meshgrid(grid, grid, indexing="ij")
    center_mm = np.asarray(point_zyx, float) * np.asarray(spacing, float)
    xyz_mm = center_mm[None, None, :] + uu[..., None] * u + vv[..., None] * v
    zyx = xyz_mm / np.asarray(spacing, float)
    im = ndi.map_coordinates(
        np.asarray(ct, float),
        [zyx[..., 0], zyx[..., 1], zyx[..., 2]],
        order=1,
        mode="nearest",
    )
    return im, grid


def _path_turns_deg(path, spacing):
    p = resample_path(path, spacing, 0.30)
    if len(p) < 4:
        return np.array([], float)
    q = p * np.asarray(spacing, float)
    a = np.diff(q, axis=0)
    a = np.asarray([_unit(x) for x in a])
    dots = np.clip(np.sum(a[:-1] * a[1:], axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _frangi_3d(volume_hu, spacing, scales_mm=(0.55, 0.80, 1.10, 1.45), alpha=0.5, beta=0.5):
    """Bright-tube Frangi-like vesselness in physical units."""
    x = np.clip(np.asarray(volume_hu, np.float32), 80.0, 1000.0)
    x = (x - 80.0) / 920.0
    spacing = np.asarray(spacing, float)
    best = np.zeros_like(x, dtype=np.float32)
    best_scale = np.zeros_like(x, dtype=np.float32)

    for sigma_mm in scales_mm:
        sig = np.maximum(float(sigma_mm) / spacing, 0.55)
        norm = float(sigma_mm) ** 2
        hzz = ndi.gaussian_filter(x, sig, order=(2, 0, 0), mode="nearest") * norm / (spacing[0] ** 2)
        hyy = ndi.gaussian_filter(x, sig, order=(0, 2, 0), mode="nearest") * norm / (spacing[1] ** 2)
        hxx = ndi.gaussian_filter(x, sig, order=(0, 0, 2), mode="nearest") * norm / (spacing[2] ** 2)
        hzy = ndi.gaussian_filter(x, sig, order=(1, 1, 0), mode="nearest") * norm / (spacing[0] * spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1, 0, 1), mode="nearest") * norm / (spacing[0] * spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0, 1, 1), mode="nearest") * norm / (spacing[1] * spacing[2])

        H = np.empty(x.shape + (3, 3), dtype=np.float32)
        H[..., 0, 0] = hzz
        H[..., 1, 1] = hyy
        H[..., 2, 2] = hxx
        H[..., 0, 1] = H[..., 1, 0] = hzy
        H[..., 0, 2] = H[..., 2, 0] = hzx
        H[..., 1, 2] = H[..., 2, 1] = hyx

        vals = np.linalg.eigvalsh(H)
        order = np.argsort(np.abs(vals), axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        l1, l2, l3 = vals[..., 0], vals[..., 1], vals[..., 2]
        eps = 1e-8
        ra = np.abs(l2) / (np.abs(l3) + eps)
        rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
        s = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        nz = s[s > 0]
        c = max(float(np.percentile(nz, 90)) * 0.45 if nz.size else 0.05, 1e-4)
        v = (1.0 - np.exp(-(ra * ra) / (2 * alpha * alpha)))
        v *= np.exp(-(rb * rb) / (2 * beta * beta))
        v *= (1.0 - np.exp(-(s * s) / (2 * c * c)))
        v[(l2 >= 0) | (l3 >= 0)] = 0.0
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        take = v > best
        best[take] = v[take]
        best_scale[take] = float(sigma_mm)

    return best, best_scale


def synthetic_vesselness_self_test():
    shape = (41, 41, 41)
    z, y, x = np.indices(shape)
    tube = np.exp(-((y - 20.0) ** 2 + (x - 20.0) ** 2) / (2 * 2.1 ** 2))
    vol = 80.0 + 650.0 * tube
    v, _ = _frangi_3d(vol, np.array([0.4, 0.4, 0.4]))
    center = float(np.median(v[8:33, 20, 20]))
    bg = float(np.median(v[:, :5, :5]))
    ok = center > max(0.02, 5.0 * bg)
    return {"passed": bool(ok), "center_vesselness": center, "background_vesselness": bg}


class Secondary3DVesselnessTopologyWorkflow:
    COMPONENTS = ("source_ct", "frozen_geometry", "vesselness", "topology_search", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_3D_Vesselness_Topology_v1"
        self.out = self.root / "Secondary_3D_Vesselness_Topology_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.seed = self.reference = self.trunk = self.prior_course = None
        self.start_point = self.control_point = self.seed_endpoint = None
        self.start_arc = self.control_arc = None
        self.terminal_tangent = None
        self.roi = self.roi_lo = self.roi_hi = None
        self.vesselness = self.best_scale = None
        self.vessel_threshold = None
        self.control_path = self.extension_path = self.full_path = None
        self.candidates = None
        self.summary = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({
            "component": component,
            "reuse_requested": self.reuse.get(component),
            "action": action,
            "path": str(path),
            "note": note,
        })
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {
            "source_ct": "series7_int16.npy",
            "frozen_geometry": "frozen_geometry.json",
            "vesselness": "vesselness.npy",
            "topology_search": "topology_summary.json",
            "figures": "figures.done",
            "report": "report.done",
        }
        return pd.DataFrame([
            {"component": k, "reuse": self.reuse[k], "cache_exists": (self.cache / v).exists()}
            for k, v in names.items()
        ])

    def _first(self, rels):
        for r in rels:
            p = self.root / r
            if p.exists():
                return p
        return None

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        om = self.cache / "series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and om.exists():
            self.ct = np.load(own, mmap_mode="r")
            self.meta = json.loads(om.read_text())
            self.spacing = np.asarray(self.meta["spacing_zyx"], float)
            self._record("source_ct", "reused", own)
            return self.ct
        src = self._first([
            "Cache/Secondary_Rejection_Diagnostic_v1/series7_int16.npy",
            "Cache/Secondary_Target_Free_Continuation_v1/series7_int16.npy",
            "Cache/LCX_Gap_Bridge_v1/series7_int16.npy",
            "Cache/LCX_Distal_Relaunch_v1/series7_int16.npy",
        ])
        if src is None or not src.with_suffix(".json").exists():
            raise FileNotFoundError("Disk-backed series-7 source CCTA cache not found")
        shutil.copyfile(src, own)
        shutil.copyfile(src.with_suffix(".json"), om)
        self.ct = np.load(own, mmap_mode="r")
        self.meta = json.loads(om.read_text())
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", "imported_prior_cache", src)
        return self.ct

    def load_frozen_geometry(self, start_arc_mm=10.6, control_arc_mm=12.7):
        if self.ct is None:
            self.load_source_ct()
        seed = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv"
        ss = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_summary.json"
        ref = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "alternative_01_centerline.csv"
        trunk = self.root / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv"
        prior = self.root / "LCX_Distal_Relaunch_Report" / "best_combined_centerline.csv"
        if not all(p.exists() for p in (seed, ss, ref, trunk)):
            raise FileNotFoundError("Required accepted secondary-branch and LAD/trunk outputs not found")
        if not bool(json.loads(ss.read_text()).get("accepted", False)):
            raise RuntimeError("Accepted lateral-divergence branch missing")

        self.seed = pd.read_csv(seed)[["z", "y", "x"]].to_numpy(float)
        self.reference = pd.read_csv(ref)[["z", "y", "x"]].to_numpy(float)
        self.trunk = pd.read_csv(trunk)[["z", "y", "x"]].to_numpy(float)
        self.prior_course = pd.read_csv(prior)[["z", "y", "x"]].to_numpy(float) if prior.exists() else None

        sp = resample_path(self.seed, self.spacing, 0.16)
        s = arc_mm(sp, self.spacing)
        self.start_arc = float(np.clip(start_arc_mm, 9.5, max(9.6, s[-1] - 2.0)))
        self.control_arc = float(np.clip(control_arc_mm, self.start_arc + 1.6, s[-1] - 0.6))
        i0 = int(np.argmin(np.abs(s - self.start_arc)))
        i1 = int(np.argmin(np.abs(s - self.control_arc)))
        self.start_point = sp[i0].copy()
        self.control_point = sp[i1].copy()
        self.seed_endpoint = sp[-1].copy()
        j0 = max(0, i1 - 8)
        self.terminal_tangent = _unit((sp[i1] - sp[j0]) * self.spacing)

        seed_phys = sp[s >= max(8.5, self.start_arc - 1.5)] * self.spacing
        forward_phys = self.control_point * self.spacing + np.outer(
            np.linspace(0.0, 12.0, 7), self.terminal_tangent
        )
        pts_phys = np.vstack([seed_phys, forward_phys])
        margin = 7.0
        lo_phys = np.min(pts_phys, axis=0) - margin
        hi_phys = np.max(pts_phys, axis=0) + margin
        lo = np.floor(lo_phys / self.spacing).astype(int)
        hi = np.ceil(hi_phys / self.spacing).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.ct.shape))
        self.roi_lo, self.roi_hi = lo, hi
        self.roi = np.asarray(self.ct[
            lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]
        ], dtype=np.float32)

        prior_return = None
        if self.prior_course is not None:
            pp = resample_path(self.prior_course, self.spacing, 0.18)
            ps = arc_mm(pp, self.spacing)
            k = int(np.argmin(np.abs(ps - 17.9)))
            prior_return = float(np.linalg.norm((pp[k] - self.control_point) * self.spacing))

        snap = {
            "algorithm": ALGORITHM_VERSION,
            "start_arc_mm": self.start_arc,
            "control_arc_mm": self.control_arc,
            "known_control_length_mm": self.control_arc - self.start_arc,
            "start_zyx": self.start_point.tolist(),
            "control_zyx": self.control_point.tolist(),
            "accepted_seed_endpoint_zyx": self.seed_endpoint.tolist(),
            "terminal_tangent_zyx_mm": self.terminal_tangent.tolist(),
            "roi_lo_zyx": self.roi_lo.tolist(),
            "roi_hi_zyx": self.roi_hi.tolist(),
            "roi_shape": list(map(int, self.roi.shape)),
            "prior_relaunch_17p9_distance_from_control_mm": prior_return,
            "search_target": None,
            "note": "Known 10.6-12.7 mm segment is positive control; distal search is target-free.",
        }
        _write_json(snap, self.cache / "frozen_geometry.json")
        self._record("frozen_geometry", "loaded_positive_control_and_target_free_roi", self.cache / "frozen_geometry.json")
        return snap

    def _global_to_local(self, pts):
        return np.asarray(pts, float) - self.roi_lo[None, :]

    def _local_to_global(self, pts):
        return np.asarray(pts, float) + self.roi_lo[None, :]

    def compute_vesselness(self, scales_mm=(0.55, 0.80, 1.10, 1.45)):
        vf = self.cache / "vesselness.npy"
        sf = self.cache / "best_scale.npy"
        mf = self.cache / "vesselness_meta.json"
        if self.roi is None:
            self.load_frozen_geometry()
        if self.reuse["vesselness"] and vf.exists() and sf.exists() and mf.exists():
            self.vesselness = np.load(vf, mmap_mode="r")
            self.best_scale = np.load(sf, mmap_mode="r")
            meta = json.loads(mf.read_text())
            self.vessel_threshold = float(meta["adaptive_threshold"])
            self._record("vesselness", "reused", vf)
            return meta

        v, bs = _frangi_3d(self.roi, self.spacing, scales_mm=scales_mm)
        self.vesselness = v
        self.best_scale = bs
        np.save(vf, v)
        np.save(sf, bs)

        sp = resample_path(self.seed, self.spacing, 0.12)
        s = arc_mm(sp, self.spacing)
        control = sp[(s >= self.start_arc) & (s <= self.control_arc)]
        vals = _sample_trilinear(v, self._global_to_local(control))
        vals = vals[np.isfinite(vals)]
        p25 = float(np.percentile(vals, 25)) if len(vals) else 0.02
        med = float(np.median(vals)) if len(vals) else 0.05
        thr = float(max(0.003, min(0.12, 0.35 * p25)))
        self.vessel_threshold = thr
        meta = {
            "algorithm": ALGORITHM_VERSION,
            "scales_mm": list(map(float, scales_mm)),
            "control_vesselness_p25": p25,
            "control_vesselness_median": med,
            "adaptive_threshold": thr,
            "roi_shape": list(map(int, self.roi.shape)),
        }
        _write_json(meta, mf)
        self._record("vesselness", "computed", vf)
        return meta

    def _build_mask_and_cost(self):
        hu = np.asarray(self.roi, float)
        v = np.asarray(self.vesselness, float)
        bs = np.asarray(self.best_scale, float)
        med_control = json.loads((self.cache / "vesselness_meta.json").read_text())["control_vesselness_median"]
        vn = np.clip(v / max(1.5 * float(med_control), 1e-4), 0.0, 1.0)
        mask = (hu >= 170.0) & (hu <= 1200.0) & (v >= float(self.vessel_threshold))
        cost = 0.30 + 4.5 * (1.0 - vn) ** 2
        cost += 0.30 * np.clip((bs - 1.15) / 0.50, 0.0, 1.0)
        cost += 0.20 * np.clip((hu - 900.0) / 300.0, 0.0, 1.0)
        return mask, cost.astype(np.float32)

    def _nearest_voxel(self, global_point):
        p = np.rint(np.asarray(global_point, float) - self.roi_lo).astype(int)
        return tuple(np.clip(p, 0, np.asarray(self.roi.shape) - 1))

    def _neighbors(self):
        out = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    step = math.sqrt(
                        (dz * self.spacing[0]) ** 2
                        + (dy * self.spacing[1]) ** 2
                        + (dx * self.spacing[2]) ** 2
                    )
                    out.append((dz, dy, dx, step))
        return out

    def _dijkstra(self, mask, cost, start, goal_mask=None, max_cost=140.0):
        shape = mask.shape
        n = int(np.prod(shape))
        dist = np.full(n, np.inf, np.float32)
        prev = np.full(n, -1, np.int32)
        start_flat = np.ravel_multi_index(start, shape)
        dist[start_flat] = 0.0
        heap = [(0.0, start_flat)]
        neigh = self._neighbors()
        reached_goal = -1

        while heap:
            dcur, flat = heapq.heappop(heap)
            if dcur > float(dist[flat]) + 1e-6:
                continue
            if dcur > max_cost:
                break
            z, y, x = np.unravel_index(flat, shape)
            if goal_mask is not None and goal_mask[z, y, x]:
                reached_goal = flat
                break
            for dz, dy, dx, step in neigh:
                zz, yy, xx = z + dz, y + dy, x + dx
                if zz < 0 or yy < 0 or xx < 0 or zz >= shape[0] or yy >= shape[1] or xx >= shape[2]:
                    continue
                if not mask[zz, yy, xx]:
                    continue
                nf = np.ravel_multi_index((zz, yy, xx), shape)
                nd = dcur + step * 0.5 * (float(cost[z, y, x]) + float(cost[zz, yy, xx]))
                if nd < float(dist[nf]):
                    dist[nf] = nd
                    prev[nf] = flat
                    heapq.heappush(heap, (nd, nf))
        return dist, prev, reached_goal

    @staticmethod
    def _reconstruct(prev, flat, shape):
        if flat < 0:
            return None
        seq = []
        seen = set()
        while flat >= 0 and flat not in seen:
            seen.add(flat)
            seq.append(np.array(np.unravel_index(flat, shape), float))
            flat = int(prev[flat])
        seq.reverse()
        return np.asarray(seq, float)

    def _sphere_mask(self, point_global, radius_mm):
        p = np.asarray(point_global, float) - self.roi_lo
        zz, yy, xx = np.indices(self.roi.shape)
        d2 = ((zz - p[0]) * self.spacing[0]) ** 2
        d2 += ((yy - p[1]) * self.spacing[1]) ** 2
        d2 += ((xx - p[2]) * self.spacing[2]) ** 2
        return d2 <= float(radius_mm) ** 2

    def _positive_control(self, mask, cost):
        start = self._nearest_voxel(self.start_point)
        target_mask = self._sphere_mask(self.control_point, 0.75)
        mask = mask.copy()
        mask[self._sphere_mask(self.start_point, 0.55)] = True
        mask[target_mask] = True
        dist, prev, reached = self._dijkstra(mask, cost, start, goal_mask=target_mask, max_cost=80.0)
        if reached < 0:
            return None, {
                "control_pass": False,
                "reason": "control_target_unreachable",
            }
        p_local = self._reconstruct(prev, reached, self.roi.shape)
        p_global = resample_path(self._local_to_global(p_local), self.spacing, 0.20)
        known = resample_path(self.seed, self.spacing, 0.15)
        ks = arc_mm(known, self.spacing)
        known = known[(ks >= self.start_arc - 0.2) & (ks <= self.control_arc + 0.2)]
        tree = cKDTree(known * self.spacing)
        dd, _ = tree.query(p_global * self.spacing)
        length = float(arc_mm(p_global, self.spacing)[-1])
        target_d = float(np.linalg.norm((p_global[-1] - self.control_point) * self.spacing))
        v = _sample_trilinear(self.vesselness, self._global_to_local(p_global))
        gate = bool(
            target_d <= 0.85
            and float(np.median(dd)) <= 0.75
            and float(np.percentile(dd, 90)) <= 1.15
            and length <= 1.85 * (self.control_arc - self.start_arc)
            and float(np.median(v)) >= 0.8 * float(self.vessel_threshold)
        )
        return p_global, {
            "control_pass": gate,
            "control_path_length_mm": length,
            "known_arc_length_mm": self.control_arc - self.start_arc,
            "control_target_distance_mm": target_d,
            "control_median_distance_to_seed_mm": float(np.median(dd)),
            "control_p90_distance_to_seed_mm": float(np.percentile(dd, 90)),
            "control_median_vesselness": float(np.median(v)),
            "control_vessel_threshold": float(self.vessel_threshold),
        }

    def _candidate_metrics(self, path_global):
        p = resample_path(path_global, self.spacing, 0.20)
        s = arc_mm(p, self.spacing)
        length = float(s[-1])
        disp = float(np.linalg.norm((p[-1] - self.control_point) * self.spacing))
        proj = float(np.dot((p[-1] - self.control_point) * self.spacing, self.terminal_tangent))
        turns = _path_turns_deg(p, self.spacing)
        v = _sample_trilinear(self.vesselness, self._global_to_local(p))
        bs = _sample_trilinear(self.best_scale, self._global_to_local(p))
        hu = _sample_trilinear(self.roi, self._global_to_local(p))
        old_seed = resample_path(self.seed, self.spacing, 0.15)
        os = arc_mm(old_seed, self.spacing)
        old = old_seed[os <= self.control_arc - 0.8]
        old_sep = _nearest_dist_mm(p[-1], old, self.spacing)
        refsep = _nearest_dist_mm(p[-1], np.vstack([self.reference, self.trunk]), self.spacing)
        return {
            "new_length_mm": length,
            "endpoint_displacement_mm": disp,
            "forward_projection_mm": proj,
            "tortuosity": length / max(disp, 1e-6),
            "max_turn_deg": float(np.max(turns)) if len(turns) else 0.0,
            "median_vesselness": float(np.median(v)),
            "p10_vesselness": float(np.percentile(v, 10)),
            "median_best_scale_mm": float(np.median(bs)),
            "p90_best_scale_mm": float(np.percentile(bs, 90)),
            "median_center_hu": float(np.median(hu)),
            "endpoint_old_seed_separation_mm": float(old_sep),
            "endpoint_reference_separation_mm": float(refsep),
        }

    def search_topology(self, min_candidate_projection_mm=1.5, max_cost=150.0):
        sf = self.cache / "topology_summary.json"
        cf = self.cache / "topology_candidates.csv"
        pf = self.cache / "control_path.csv"
        ef = self.cache / "extension_path.csv"
        ff = self.cache / "full_path.csv"
        if self.vesselness is None:
            self.compute_vesselness()
        if self.reuse["topology_search"] and all(p.exists() for p in (sf, cf, pf, ef, ff)):
            self.summary = json.loads(sf.read_text())
            self.candidates = pd.read_csv(cf)
            self.control_path = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.extension_path = pd.read_csv(ef)[["z", "y", "x"]].to_numpy(float)
            self.full_path = pd.read_csv(ff)[["z", "y", "x"]].to_numpy(float)
            self._record("topology_search", "reused", sf)
            return self.summary

        mask, cost = self._build_mask_and_cost()
        self.control_path, csum = self._positive_control(mask, cost)
        if not csum["control_pass"]:
            self.extension_path = np.asarray([self.control_point])
            self.full_path = self.control_path if self.control_path is not None else np.asarray([self.start_point])
            self.candidates = pd.DataFrame()
            self.summary = {
                "algorithm": ALGORITHM_VERSION,
                "status": "POSITIVE_CONTROL_FAILED",
                "accepted_continuation": False,
                **csum,
                "interpretation": "3-D vesselness topology search could not reliably rediscover the known branch segment; distal inference is invalid.",
            }
        else:
            start = self._nearest_voxel(self.control_point)
            ext_mask = mask.copy()
            ext_mask[self._sphere_mask(self.control_point, 0.60)] = True
            dist, prev, _ = self._dijkstra(ext_mask, cost, start, goal_mask=None, max_cost=max_cost)
            finite = np.isfinite(dist)
            coords = np.column_stack(np.nonzero(finite))
            if len(coords):
                phys = (coords + self.roi_lo) * self.spacing
                cp = self.control_point * self.spacing
                proj = (phys - cp) @ self.terminal_tangent
                disp = np.linalg.norm(phys - cp, axis=1)
                vals = self.vesselness[tuple(coords.T)]
                ok = (proj >= min_candidate_projection_mm) & (disp >= 1.8) & (vals >= self.vessel_threshold)
                coords = coords[ok]
                proj = proj[ok]
                vals = vals[ok]
            rows = []
            paths = []
            if len(coords):
                score = proj + 1.5 * vals
                order = np.argsort(score)[::-1]
                chosen = []
                for ii in order:
                    g = coords[ii] + self.roi_lo
                    if all(np.linalg.norm((g - q) * self.spacing) >= 1.0 for q in chosen):
                        chosen.append(g.copy())
                        flat = np.ravel_multi_index(tuple(coords[ii]), self.roi.shape)
                        loc = self._reconstruct(prev, flat, self.roi.shape)
                        if loc is None or len(loc) < 2:
                            continue
                        path = resample_path(self._local_to_global(loc), self.spacing, 0.20)
                        m = self._candidate_metrics(path)
                        continuous = bool(
                            m["new_length_mm"] >= 2.0
                            and m["endpoint_displacement_mm"] >= 1.8
                            and m["forward_projection_mm"] >= 1.5
                            and m["tortuosity"] <= 2.0
                            and m["max_turn_deg"] <= 75.0
                            and m["p10_vesselness"] >= 0.65 * self.vessel_threshold
                            and m["endpoint_reference_separation_mm"] >= 1.7
                        )
                        supported = bool(
                            continuous
                            and m["new_length_mm"] >= 4.0
                            and m["endpoint_displacement_mm"] >= 3.0
                            and m["forward_projection_mm"] >= 2.5
                            and m["endpoint_old_seed_separation_mm"] >= 1.0
                            and m["p90_best_scale_mm"] <= 1.50
                        )
                        rank = (
                            0.24 * min(m["new_length_mm"] / 6.0, 1.0)
                            + 0.22 * min(m["forward_projection_mm"] / 5.0, 1.0)
                            + 0.20 * min(m["median_vesselness"] / max(self.vessel_threshold * 2.0, 1e-4), 1.0)
                            + 0.12 * min(m["endpoint_displacement_mm"] / 5.0, 1.0)
                            + 0.12 * (1.0 / max(m["tortuosity"], 1.0))
                            + 0.10 * min(m["endpoint_old_seed_separation_mm"] / 2.0, 1.0)
                        )
                        row = {
                            "candidate": len(rows) + 1,
                            **m,
                            "continuous_gate": continuous,
                            "supported_continuation": supported,
                            "rank_score": float(rank),
                        }
                        rows.append(row)
                        paths.append((row, path))
                        if len(rows) >= 24:
                            break
            self.candidates = pd.DataFrame(rows)
            if len(self.candidates):
                self.candidates = self.candidates.sort_values(
                    ["supported_continuation", "continuous_gate", "rank_score"],
                    ascending=[False, False, False],
                )
                best_id = int(self.candidates.iloc[0]["candidate"])
                row, path = next((r, p) for r, p in paths if int(r["candidate"]) == best_id)
                self.extension_path = path
                self.full_path = resample_path(
                    np.vstack([self.control_path, self.extension_path[1:]]),
                    self.spacing,
                    0.20,
                )
                if bool(row["supported_continuation"]):
                    status = "SUPPORTED_3D_TOPOLOGIC_EXTENSION"
                    interp = "Positive control passed and a >=4 mm geometrically new 3-D vesselness-supported continuation was found; vessel identity remains unassigned."
                elif bool(row["continuous_gate"]):
                    status = "PARTIAL_3D_TOPOLOGIC_EXTENSION"
                    interp = "Positive control passed and a multi-mm 3-D tubular corridor was found, but it did not satisfy the full new-course support gate."
                else:
                    status = "NO_SUPPORTED_3D_TOPOLOGIC_EXTENSION"
                    interp = "Positive control passed, but reachable distal 3-D vesselness paths failed continuity/geometry gates."
                self.summary = {
                    "algorithm": ALGORITHM_VERSION,
                    "status": status,
                    "accepted_continuation": bool(row["supported_continuation"]),
                    **csum,
                    **row,
                    "interpretation": interp,
                }
            else:
                self.extension_path = np.asarray([self.control_point])
                self.full_path = self.control_path
                self.summary = {
                    "algorithm": ALGORITHM_VERSION,
                    "status": "NO_3D_TOPOLOGIC_EXTENSION",
                    "accepted_continuation": False,
                    **csum,
                    "interpretation": "Positive control passed, but no target-free distal node >=1.5 mm forward was reachable through the adaptive 3-D vesselness graph.",
                }

        self.candidates.to_csv(cf, index=False)
        for path, arr in [(pf, self.control_path), (ef, self.extension_path), (ff, self.full_path)]:
            arr = np.asarray(arr, float)
            pd.DataFrame({
                "arc_mm": arc_mm(arr, self.spacing),
                "z": arr[:, 0],
                "y": arr[:, 1],
                "x": arr[:, 2],
            }).to_csv(path, index=False)
        _write_json(self.summary, sf)
        self._record("topology_search", "computed", sf)
        return self.summary

    def make_figures(self):
        if self.summary is None:
            self.search_topology()
        names = [
            "01_vesselness_geometry.png",
            "02_positive_control.png",
            "03_distal_cross_sections.png",
            "04_path_profiles.png",
        ]
        done = self.cache / "figures.done"
        if self.reuse["figures"] and done.exists() and all((self.out / n).exists() for n in names):
            self._record("figures", "reused", done)
            return names

        sp = resample_path(self.seed, self.spacing, 0.15)
        pp = resample_path(self.prior_course, self.spacing, 0.18) if self.prior_course is not None else None

        fig = plt.figure(figsize=(13, 4))
        pairs = [(2, 1, "X-Y"), (2, 0, "X-Z"), (1, 0, "Y-Z")]
        for k, (a, b, title) in enumerate(pairs, 1):
            ax = fig.add_subplot(1, 3, k)
            ax.plot(sp[:, a] * self.spacing[a], sp[:, b] * self.spacing[b], label="accepted secondary branch")
            if pp is not None:
                ax.plot(pp[:, a] * self.spacing[a], pp[:, b] * self.spacing[b], alpha=0.45, label="prior relaunch")
            if self.control_path is not None:
                ax.plot(self.control_path[:, a] * self.spacing[a], self.control_path[:, b] * self.spacing[b], lw=3, label="3D positive control")
            if self.extension_path is not None and len(self.extension_path) > 1:
                ax.plot(self.extension_path[:, a] * self.spacing[a], self.extension_path[:, b] * self.spacing[b], lw=3, label="3D extension")
            ax.scatter([self.start_point[a] * self.spacing[a]], [self.start_point[b] * self.spacing[b]], s=35)
            ax.scatter([self.control_point[a] * self.spacing[a]], [self.control_point[b] * self.spacing[b]], s=35)
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            if k == 1:
                ax.legend(fontsize=7)
        fig.suptitle(f"3-D vesselness/topology geometry — {self.summary['status']}")
        fig.tight_layout()
        fig.savefig(self.out / names[0], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        known = resample_path(self.seed, self.spacing, 0.12)
        ks = arc_mm(known, self.spacing)
        ctl = known[(ks >= self.start_arc) & (ks <= self.control_arc)]
        ax.plot(ctl[:, 2] * self.spacing[2], ctl[:, 1] * self.spacing[1], label="known control segment")
        if self.control_path is not None:
            ax.plot(self.control_path[:, 2] * self.spacing[2], self.control_path[:, 1] * self.spacing[1], lw=3, label="recovered 3D control")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Positive control: pass={self.summary.get('control_pass')}  "
            f"median distance={self.summary.get('control_median_distance_to_seed_mm', float('nan')):.2f} mm"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.out / names[1], dpi=160)
        plt.close(fig)

        path = self.extension_path if self.extension_path is not None and len(self.extension_path) > 1 else self.control_path
        fig, axes = plt.subplots(2, 5, figsize=(13, 5.3))
        axes = axes.ravel()
        if path is not None and len(path) >= 3:
            p = resample_path(path, self.spacing, 0.22)
            s = arc_mm(p, self.spacing)
            xs = np.linspace(0, s[-1], min(10, max(2, len(p))))
            for ax, x in zip(axes, xs):
                i = int(np.argmin(np.abs(s - x)))
                i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
                t = (p[i1] - p[i0]) * self.spacing
                im, _ = orthogonal_plane(self.ct, p[i], t, self.spacing)
                ax.imshow(im, cmap="gray", vmin=100, vmax=900)
                ax.set_title(f"{s[i]:.1f} mm")
                ax.axis("off")
        for ax in axes:
            if not ax.has_data():
                ax.axis("off")
        fig.suptitle("Automatically sampled source-CCTA cross-sections along best 3-D path")
        fig.tight_layout()
        fig.savefig(self.out / names[2], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        if self.full_path is not None and len(self.full_path) > 1:
            p = resample_path(self.full_path, self.spacing, 0.18)
            s = arc_mm(p, self.spacing)
            v = _sample_trilinear(self.vesselness, self._global_to_local(p))
            bs = _sample_trilinear(self.best_scale, self._global_to_local(p))
            ax.plot(s, v, label="vesselness")
            ax.axhline(self.vessel_threshold, ls="--", label="adaptive threshold")
            ax2 = ax.twinx()
            ax2.plot(s, bs, alpha=0.6, label="best scale (mm)")
            ax.set_xlabel("Path arc length (mm)")
            ax.set_ylabel("3-D vesselness")
            ax2.set_ylabel("Best scale (mm)")
            ax.set_title("Vesselness and scale profile")
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(self.out / names[3], dpi=160)
        plt.close(fig)

        done.write_text("done")
        self._record("figures", "computed", done)
        return names

    def make_report(self):
        if self.summary is None:
            self.search_topology()
        self.make_figures()
        html = self.out / "OPENPLAQUE_SECONDARY_3D_VESSELNESS_TOPOLOGY_REPORT.html"
        rows = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>"
            for k, v in self.summary.items()
            if k not in {"interpretation"}
        )
        body = f"""<html><body>
<h1>OpenPlaque Secondary Branch — 3-D Vesselness / Topology</h1>
<p><b>Status: {self.summary['status']}</b></p>
<p>{self.summary.get('interpretation','')}</p>
<table border='1' cellpadding='4'>{rows}</table>
<h2>3-D geometry</h2><img src='01_vesselness_geometry.png' width='95%'>
<h2>Known-segment positive control</h2><img src='02_positive_control.png' width='85%'>
<h2>Source-CCTA cross-sections</h2><img src='03_distal_cross_sections.png' width='95%'>
<h2>Vesselness profile</h2><img src='04_path_profiles.png' width='85%'>
<p>Research use only. The 17.9-mm relaunch is comparison-only. No LCX identity is assigned automatically.</p>
</body></html>"""
        html.write_text(body)
        (self.cache / "report.done").write_text("done")
        self._record("report", "computed", html)
        return html

    def package(self):
        report = self.make_report()
        zip_path = self.out / "OPENPLAQUE_SECONDARY_3D_VESSELNESS_TOPOLOGY_REPORT_BACK.zip"
        include = [
            report,
            self.cache / "frozen_geometry.json",
            self.cache / "vesselness_meta.json",
            self.cache / "topology_summary.json",
            self.cache / "topology_candidates.csv",
            self.cache / "control_path.csv",
            self.cache / "extension_path.csv",
            self.cache / "full_path.csv",
            self.out / "cache_provenance.csv",
            self.out / "01_vesselness_geometry.png",
            self.out / "02_positive_control.png",
            self.out / "03_distal_cross_sections.png",
            self.out / "04_path_profiles.png",
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in include:
                if p.exists():
                    z.write(p, arcname=p.name)
        return zip_path
