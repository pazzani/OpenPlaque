from __future__ import annotations

"""Independent source-CCTA adjudication of the apparent distal LAD continuation.

The preceding label-neutral backbone scan found one strong additional source-supported path
near the *distal* endpoint of the frozen LAD. Its geometry is suspicious for endpoint
continuation rather than a true side branch. This experiment therefore starts exactly at the
frozen LAD distal endpoint and performs a new source-led trace using only the frozen LAD
terminal tangent plus source CCTA/vesselness. The previously discovered path is used only
after tracing, as a post-hoc reproducibility reference.

No clinical vessel labels are changed and the frozen master is never modified.
"""

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lad-distal-endpoint-continuation-v1.0"
OUTPUT_DIRNAME = "LAD_Distal_Endpoint_Continuation_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
DISCOVERY_SUMMARY = Path("Left_Coronary_Backbone_Branch_Discovery_v1/summary.json")
DISCOVERY_PATH = Path("Left_Coronary_Backbone_Branch_Discovery_v1/additional_branch_01_s00_d3_f0.csv")

STEP_MM = 0.40
TARGET_SEARCH_MM = 15.0
CONTROL_SEARCH_MM = 6.0
BEAM_WIDTH = 72
PLANE_STEP_MM = 0.40
SUSTAINED_FAIL_N = 3
VESSELNESS_MARGIN_MM = 22.0

HU_MIN = 100.0
HU_MAX = 1400.0
MIN_VESSELNESS_FACTOR = 0.20
MAX_STEP_TURN_DEG = 55.0
KNOWN_LAD_EXCLUSION_AFTER_MM = 1.6
KNOWN_LAD_EXCLUSION_MM = 0.80

MIN_CONTROL_OVERLAP_FRACTION_2MM = 0.80
MAX_CONTROL_MEDIAN_DISTANCE_MM = 1.0
MIN_CONTROL_ACCEPTED_ARC_MM = 5.0
MIN_ACCEPTED_EXTENSION_MM = 5.0
MIN_PLANE_PASS_FRACTION = 0.80
MIN_ENDPOINT_LAD_SEPARATION_MM = 3.0

MIN_REFERENCE_MATCH_FRACTION_2MM = 0.60
MAX_REFERENCE_MEDIAN_DISTANCE_MM = 1.5
MIN_REFERENCE_TANGENT_ALIGNMENT = 0.70

STATUS_PREREQ = "LAD_DISTAL_CONTINUATION_PREREQUISITE_FAILED"
STATUS_CONTROL = "LAD_DISTAL_CONTINUATION_KNOWN_LAD_CONTROL_FAILED"
STATUS_NONE = "LAD_DISTAL_CONTINUATION_NO_VALID_EXTENSION"
STATUS_REPRO = "LAD_DISTAL_ENDPOINT_EXTENSION_INDEPENDENTLY_REPRODUCED"
STATUS_OTHER = "LAD_DISTAL_ENDPOINT_EXTENSION_SOURCE_SUPPORTED_DIFFERENT_TRAJECTORY"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_path(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinate columns in {p}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=.25):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0.0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array(
            [[row[0], col[0], slc[0]],
             [row[1], col[1], slc[1]],
             [row[2], col[2], slc[2]]], float
        )
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz_index = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz_index[:, ::-1]

    def zyx_to_xyz(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz_index = pts[:, ::-1]
        return self.origin + (xyz_index * self.spacing_xyz) @ self.D.T


def _source(cache):
    cache = Path(cache)
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    return SourceGeometry(meta), src


def _sample_array(geom, arr, pts, cval=0.0, origin_zyx=None):
    z = geom.xyz_to_zyx(pts)
    if origin_zyx is not None:
        z = z - np.asarray(origin_zyx, float)[None, :]
    return map_coordinates(np.asarray(arr), z.T, order=1, mode="constant", cval=float(cval))


def _frangi_3d(vol, spacing_zyx, scales=(0.55, 0.80, 1.10, 1.45)):
    x = np.clip(np.asarray(vol, np.float32), 80.0, 1000.0)
    x = (x - 80.0) / 920.0
    spacing = np.asarray(spacing_zyx, float)
    best = np.zeros_like(x, dtype=np.float32)
    eps = 1e-8
    for sm in scales:
        sig = np.maximum(sm / spacing, 0.55)
        n = float(sm * sm)
        hzz = ndi.gaussian_filter(x, sig, order=(2, 0, 0), mode="nearest") * n / spacing[0] ** 2
        hyy = ndi.gaussian_filter(x, sig, order=(0, 2, 0), mode="nearest") * n / spacing[1] ** 2
        hxx = ndi.gaussian_filter(x, sig, order=(0, 0, 2), mode="nearest") * n / spacing[2] ** 2
        hzy = ndi.gaussian_filter(x, sig, order=(1, 1, 0), mode="nearest") * n / (spacing[0] * spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1, 0, 1), mode="nearest") * n / (spacing[0] * spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0, 1, 1), mode="nearest") * n / (spacing[1] * spacing[2])
        H = np.empty(x.shape + (3, 3), dtype=np.float32)
        H[..., 0, 0], H[..., 1, 1], H[..., 2, 2] = hzz, hyy, hxx
        H[..., 0, 1] = H[..., 1, 0] = hzy
        H[..., 0, 2] = H[..., 2, 0] = hzx
        H[..., 1, 2] = H[..., 2, 1] = hyx
        vals = np.linalg.eigvalsh(H)
        vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
        l1, l2, l3 = vals[..., 0], vals[..., 1], vals[..., 2]
        ra = np.abs(l2) / (np.abs(l3) + eps)
        rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
        ss = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        nz = ss[ss > 0]
        c = max(float(np.percentile(nz, 90)) * 0.45 if nz.size else 0.05, 1e-4)
        vv = (1.0 - np.exp(-(ra * ra) / 0.5)) * np.exp(-(rb * rb) / 0.5) * (
            1.0 - np.exp(-(ss * ss) / (2.0 * c * c))
        )
        vv[(l2 >= 0) | (l3 >= 0)] = 0.0
        best = np.maximum(best, np.nan_to_num(vv).astype(np.float32))
        del H, vals, hzz, hyy, hxx, hzy, hzx, hyx, vv
    return best


class LocalVesselness:
    def __init__(self, geom, src, anchor_pts, margin_mm=VESSELNESS_MARGIN_MM):
        z = geom.xyz_to_zyx(np.asarray(anchor_pts, float))
        margin_vox = float(margin_mm) / geom.spacing_zyx
        lo = np.maximum(np.floor(z.min(axis=0) - margin_vox).astype(int), 0)
        hi = np.minimum(np.ceil(z.max(axis=0) + margin_vox).astype(int) + 1, np.asarray(src.shape))
        if np.any(hi - lo < 7):
            raise RuntimeError(f"local vesselness ROI too small: lo={lo.tolist()} hi={hi.tolist()}")
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        roi = np.asarray(src[sl], dtype=np.float32)
        self.field = _frangi_3d(roi, geom.spacing_zyx)
        self.lo = lo.astype(float)
        self.shape = tuple(int(v) for v in roi.shape)

    def sample(self, geom, pts):
        z = geom.xyz_to_zyx(pts) - self.lo[None, :]
        return map_coordinates(self.field, z.T, order=1, mode="constant", cval=0.0)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _cone_dirs(t, degrees=(0, 10, 20, 30, 40), nphi=10):
    t = _unit(t)
    u, v = _orth_basis(t)
    out = [t]
    for deg in degrees:
        if deg == 0:
            continue
        a = np.deg2rad(deg)
        for phi in np.linspace(0, 2 * np.pi, nphi, endpoint=False):
            out.append(_unit(np.cos(a) * t + np.sin(a) * (np.cos(phi) * u + np.sin(phi) * v)))
    return out


def _initial_dirs(t):
    # Wider first-step sampling, but still centered on the terminal LAD tangent.
    return _cone_dirs(t, degrees=(0, 8, 16, 24, 32, 40), nphi=12)


def _rank_initial_dirs(geom, src, vesselness, start, tangent, vnorm, n=8):
    ranked = []
    for d in _initial_dirs(tangent):
        q = np.array([0.4, 0.8, 1.2, 1.6], float)
        pts = np.asarray(start, float)[None, :] + q[:, None] * d[None, :]
        hu = _sample_array(geom, src, pts, cval=-1024.0)
        vv = vesselness.sample(geom, pts)
        if float(np.mean((hu >= HU_MIN) & (hu <= HU_MAX))) < 0.75:
            continue
        sc = (
            1.8 * float(np.mean(np.clip(vv / max(vnorm, 1e-6), 0, 1.5)))
            + .30 * float(np.mean(np.clip((hu - HU_MIN) / 500.0, 0, 1)))
            + .20 * max(0.0, float(np.dot(_unit(d), _unit(tangent))))
        )
        ranked.append((sc, d))
    ranked.sort(key=lambda x: x[0], reverse=True)
    keep = []
    for sc, d in ranked:
        if all(_angle(d, kd) >= 10.0 for _, kd in keep):
            keep.append((sc, d))
        if len(keep) >= n:
            break
    return keep


def _beam(
    geom, src, vesselness, start, initial_dir, vthr, vnorm, max_mm,
    known_tree=None, exclude_known=True
):
    states = [(0.0, [np.asarray(start, float)], _unit(initial_dir))]
    best = states[0]
    rows = []
    nsteps = int(math.ceil(max_mm / STEP_MM))
    for si in range(nsteps):
        nxt = []
        cnt = {
            "step": si, "arc_budget_mm": float((si + 1) * STEP_MM), "states_in": len(states),
            "proposals": 0, "reject_turn": 0, "reject_loop": 0, "reject_known_lad": 0,
            "reject_hu": 0, "reject_vesselness": 0, "accepted": 0, "kept": 0
        }
        for score, pts, t in states:
            for d in _cone_dirs(t):
                cnt["proposals"] += 1
                if np.dot(d, t) < math.cos(math.radians(MAX_STEP_TURN_DEG)):
                    cnt["reject_turn"] += 1
                    continue
                q = np.asarray(pts[-1]) + STEP_MM * d
                if len(pts) > 5 and np.min(np.linalg.norm(np.asarray(pts[:-4]) - q, axis=1)) < .60 * STEP_MM:
                    cnt["reject_loop"] += 1
                    continue
                arc_new = len(pts) * STEP_MM
                if exclude_known and known_tree is not None and arc_new > KNOWN_LAD_EXCLUSION_AFTER_MM:
                    if float(known_tree.query(q)[0]) < KNOWN_LAD_EXCLUSION_MM:
                        cnt["reject_known_lad"] += 1
                        continue
                hu = float(_sample_array(geom, src, [q], cval=-1024.0)[0])
                vv = float(vesselness.sample(geom, [q])[0])
                if not (HU_MIN <= hu <= HU_MAX):
                    cnt["reject_hu"] += 1
                    continue
                if vv < vthr:
                    cnt["reject_vesselness"] += 1
                    continue
                vn = min(1.5, max(0.0, vv / max(vnorm, 1e-6)))
                align = max(0.0, float(np.dot(d, t)))
                ns = score + 1.8 * vn + .30 * min(1.0, max(0.0, hu - HU_MIN) / 500.0) + .35 * align
                nt = _unit(.72 * t + .28 * d)
                st = (ns, pts + [q], nt)
                nxt.append(st)
                cnt["accepted"] += 1
                if len(st[1]) > len(best[1]) or (len(st[1]) == len(best[1]) and ns > best[0]):
                    best = st
        if not nxt:
            rows.append(cnt)
            break
        nxt.sort(key=lambda x: x[0], reverse=True)
        keep, bins = [], set()
        for st in nxt:
            key = tuple(np.round(np.asarray(st[1][-1]) / .35).astype(int))
            if key in bins:
                continue
            bins.add(key)
            keep.append(st)
            if len(keep) >= BEAM_WIDTH:
                break
        states = keep
        cnt["kept"] = len(states)
        rows.append(cnt)

    finals = []
    for st in list(states) + [best]:
        ep = np.asarray(st[1][-1])
        if all(np.linalg.norm(ep - np.asarray(x[1][-1])) > 1.0 for x in finals):
            finals.append(st)
    finals.sort(key=lambda s: (len(s[1]), s[0]), reverse=True)
    return finals[:6], pd.DataFrame(rows)


def _plane(geom, src, c, t, half=5.5, step=.20):
    t = _unit(t)
    u, v = _orth_basis(t)
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[..., None] * u + yy[..., None] * v
    im = _sample_array(geom, src, P.reshape(-1, 3), cval=-1024.0).reshape(len(q), len(q))
    return im, q


def _component_metrics(im, q):
    iy = ix = int(np.argmin(np.abs(q)))
    center = float(im[iy, ix])
    yy, xx = np.meshgrid(q, q, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    thr = max(220.0, min(500.0, .55 * center))
    bw = (im >= thr) & (rr <= 3.5)
    lab, _ = ndi.label(bw, np.ones((3, 3), int))
    labels = []
    if lab[iy, ix] > 0:
        labels = [int(lab[iy, ix])]
    else:
        pts = np.argwhere(bw)
        if len(pts):
            dist = np.sqrt(q[pts[:, 1]] ** 2 + q[pts[:, 0]] ** 2)
            k = int(np.argmin(dist))
            if dist[k] <= 1.0:
                labels = [int(lab[tuple(pts[k])])]
    if not labels:
        return {
            "center_hu": center, "component_found": False, "radius_mm": np.nan,
            "centroid_offset_mm": np.inf, "axis_ratio": np.inf, "contrast_hu": -np.inf
        }
    mask = lab == labels[0]
    pts = np.argwhere(mask)
    xs, ys = q[pts[:, 1]], q[pts[:, 0]]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    off = float(np.hypot(cx, cy))
    area = float(len(pts) * .04)
    rad = float(np.sqrt(area / np.pi))
    if len(pts) >= 4:
        C = np.cov(np.column_stack([xs, ys]).T)
        ev = np.linalg.eigvalsh(C)
        axis = float(np.sqrt(max(ev[-1], 1e-6) / max(ev[0], 1e-6)))
    else:
        axis = np.inf
    ring = (rr >= 3.5) & (rr <= 5.0)
    contrast = float(np.median(im[mask]) - np.median(im[ring])) if np.any(ring) else np.nan
    return {
        "center_hu": center, "component_found": True, "radius_mm": rad,
        "centroid_offset_mm": off, "axis_ratio": axis, "contrast_hu": contrast
    }


def _plane_pass(m):
    return bool(
        m["component_found"] and m["center_hu"] >= 200 and .55 <= m["radius_mm"] <= 3.2
        and m["centroid_offset_mm"] <= 1.10 and m["axis_ratio"] <= 2.2
        and m["contrast_hu"] >= 40
    )


def _dense_qc(geom, src, path):
    p, q = _resample(path, PLANE_STEP_MM)
    if len(p) < 2:
        return p, q, pd.DataFrame([{
            "index": 0, "arc_mm": 0.0, "center_hu": np.nan, "component_found": False,
            "radius_mm": np.nan, "centroid_offset_mm": np.inf, "axis_ratio": np.inf,
            "contrast_hu": -np.inf, "plane_pass": False
        }])
    tt = np.gradient(p, axis=0)
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    rows = []
    for i, (c, t, a) in enumerate(zip(p, tt, q)):
        im, g = _plane(geom, src, c, t)
        m = _component_metrics(im, g)
        m.update(index=i, arc_mm=float(a), plane_pass=_plane_pass(m))
        rows.append(m)
    return p, q, pd.DataFrame(rows)


def _truncate_qc(df):
    ps = df.plane_pass.astype(bool).to_numpy()
    cut = len(df) - 1
    first = None
    for i in range(max(0, len(ps) - SUSTAINED_FAIL_N + 1)):
        if not np.any(ps[i:i + SUSTAINED_FAIL_N]):
            first = i
            cut = max(0, i - 1)
            break
    d = df.iloc[:cut + 1]
    arc = float(d.arc_mm.iloc[-1]) if len(d) else 0.0
    frac = float(d.plane_pass.mean()) if len(d) else 0.0
    return {
        "first_sustained_failure_index": first,
        "accepted_last_index": int(cut),
        "accepted_arc_mm": arc,
        "accepted_plane_pass_fraction": frac,
        "covers_full_path": bool(cut == len(df) - 1),
    }


def _safe_tangents(p):
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.zeros_like(p)
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return t


def _overlap_metrics(query, ref, step=.25):
    q, qa = _resample(query, step)
    r, ra = _resample(ref, step)
    if len(q) == 0 or len(r) == 0:
        return {
            "min_distance_mm": np.inf, "median_distance_mm": np.inf, "p90_distance_mm": np.inf,
            "endpoint_distance_mm": np.inf, "fraction_within_2mm": 0.0,
            "max_contiguous_span_within_2mm": 0.0,
            "median_tangent_alignment_within_2mm": 0.0
        }
    tree = cKDTree(r)
    d, ix = tree.query(q)
    tq, tr = _safe_tangents(q), _safe_tangents(r)
    al = np.abs(np.sum(tq * tr[ix], axis=1))
    within = d <= 2.0
    best = cur = 0.0
    for v in within:
        cur = cur + step if v else 0.0
        best = max(best, cur)
    return {
        "min_distance_mm": float(np.min(d)),
        "median_distance_mm": float(np.median(d)),
        "p90_distance_mm": float(np.percentile(d, 90)),
        "endpoint_distance_mm": float(d[-1]),
        "fraction_within_2mm": float(np.mean(within)),
        "max_contiguous_span_within_2mm": float(best),
        "median_tangent_alignment_within_2mm": float(np.median(al[within])) if np.any(within) else 0.0,
    }


def _orient_lad_distal_first(lad, discovered):
    """Use the prior blind finding only to identify which frozen LAD endpoint is the tested endpoint."""
    lad = np.asarray(lad, float)
    discovered = np.asarray(discovered, float)
    tree = cKDTree(discovered)
    d0 = float(tree.query(lad[0])[0])
    d1 = float(tree.query(lad[-1])[0])
    return (lad if d0 <= d1 else lad[::-1].copy()), min(d0, d1)


def _reference_tail(discovered, distal_endpoint, lad_tree):
    """Extract the previously discovered part extending beyond the frozen distal LAD endpoint."""
    p = np.asarray(discovered, float)
    k = int(np.argmin(np.linalg.norm(p - distal_endpoint[None, :], axis=1)))
    left = p[:k + 1][::-1].copy()
    right = p[k:].copy()
    def endpoint_sep(x):
        return float(lad_tree.query(x[-1])[0]) if len(x) else -np.inf
    return left if endpoint_sep(left) >= endpoint_sep(right) else right


def _evaluate_paths(geom, src, finals, lad_tree):
    rows, paths, qcs = [], {}, {}
    for i, st in enumerate(finals):
        raw = np.asarray(st[1], float)
        p, q, qc = _dense_qc(geom, src, raw)
        tr = _truncate_qc(qc)
        last = int(tr["accepted_last_index"])
        acc = p[:last + 1] if last >= 0 else p[:1]
        epsep = float(lad_tree.query(acc[-1])[0])
        accepted = bool(
            tr["accepted_arc_mm"] >= MIN_ACCEPTED_EXTENSION_MM
            and tr["accepted_plane_pass_fraction"] >= MIN_PLANE_PASS_FRACTION
            and epsep >= MIN_ENDPOINT_LAD_SEPARATION_MM
        )
        hid = f"target_{i:02d}"
        rows.append({
            "hypothesis_id": hid, "beam_score": float(st[0]), "path_arc_mm": float(q[-1]) if len(q) else 0.0,
            "endpoint_lad_separation_mm": epsep, **tr, "extension_gate_pass": accepted
        })
        paths[hid] = acc
        qcs[hid] = qc
    return pd.DataFrame(rows), paths, qcs


def _plot_geometry(out, lad, reference_tail, best):
    fig = plt.figure(figsize=(14, 4))
    pairs = [(0, 1, "LPS X", "LPS Y"), (0, 2, "LPS X", "LPS Z"), (1, 2, "LPS Y", "LPS Z")]
    for j, (a, b, xa, ya) in enumerate(pairs, start=1):
        ax = fig.add_subplot(1, 3, j)
        ax.plot(lad[:, a], lad[:, b], label="frozen LAD")
        ax.plot(reference_tail[:, a], reference_tail[:, b], label="prior blind path")
        if best is not None:
            ax.plot(best[:, a], best[:, b], label="independent endpoint trace")
        ax.set_xlabel(xa); ax.set_ylabel(ya); ax.axis("equal")
        if j == 1:
            ax.legend()
    fig.suptitle("Frozen LAD distal endpoint and independent source-led continuation")
    fig.tight_layout()
    fig.savefig(out / "01_distal_extension_geometry.png", dpi=160)
    plt.close(fig)


def _plot_qc(out, geom, src, path):
    if path is None or len(path) < 2:
        return
    p, q = _resample(path, .8)
    tt = _safe_tangents(p)
    pick = np.linspace(0, len(p) - 1, min(12, len(p))).round().astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, pick):
        im, g = _plane(geom, src, p[i], tt[i])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=900, origin="lower")
        ax.scatter([len(g)//2], [len(g)//2], s=8)
        ax.set_title(f"{q[i]:.1f} mm")
        ax.axis("off")
    fig.suptitle("Independent distal LAD endpoint continuation: orthogonal source-CCTA QC")
    fig.tight_layout()
    fig.savefig(out / "02_distal_extension_orthogonal_qc.png", dpi=160)
    plt.close(fig)


def synthetic_distal_continuation_self_test():
    x = np.linspace(0, 10, 41)
    lad = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    discovered = np.column_stack([np.linspace(2, -5, 29), np.zeros(29), np.zeros(29)])
    oriented, d = _orient_lad_distal_first(lad, discovered)
    assert np.allclose(oriented[0], [0, 0, 0])
    tree = cKDTree(oriented)
    tail = _reference_tail(discovered, oriented[0], tree)
    assert tail[-1, 0] < -4.5
    outgoing = _unit(oriented[0] - _interp(oriented, [4.0])[0])
    assert _angle(outgoing, [-1, 0, 0]) < 1e-6
    ov = _overlap_metrics(tail, tail)
    assert ov["fraction_within_2mm"] == 1.0
    return {"ok": True, "endpoint_match_mm": d, "tail_length_mm": float(_arc(tail)[-1])}


def _finalize(out, summary):
    report = out / "OPENPLAQUE_LAD_DISTAL_ENDPOINT_CONTINUATION_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque LAD Distal Endpoint Continuation v1</h1>"
        f"<p><b>Status:</b> {summary.get('status')}</p>"
        f"<p>Known-LAD trace control: {summary.get('control', {}).get('pass')}</p>"
        f"<p>Best independent accepted extension: {summary.get('best_target', {}).get('accepted_arc_mm')} mm.</p>"
        f"<p>Post-hoc agreement with prior blind path: {summary.get('reference_agreement', {}).get('pass')}</p>"
        "<p>Research-only source-CCTA endpoint-continuation adjudication. Frozen master unchanged.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    zpath = out / "OPENPLAQUE_LAD_DISTAL_ENDPOINT_CONTINUATION_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file():
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {
        "status": "STARTED", "algorithm": ALGORITHM, "baseline_commit": BASELINE
    })

    required = [
        root / SOURCE_CACHE / "series7_int16.npy",
        root / SOURCE_CACHE / "series7_int16.json",
        root / MASTER, root / LAD, root / DISCOVERY_SUMMARY, root / DISCOVERY_PATH
    ]
    for p in required:
        _req(p)

    master = _read_json(root / MASTER)
    discovery_summary = _read_json(root / DISCOVERY_SUMMARY)
    prereq = bool(
        master.get("status") == "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
        and discovery_summary.get("status") == "BACKBONE_BRANCH_DISCOVERY_ADDITIONAL_SOURCE_BRANCHES_FOUND"
        and discovery_summary.get("control", {}).get("pass") is True
        and discovery_summary.get("clustered_additional_branch_count", 0) >= 1
    )
    if not prereq:
        s = {
            "status": STATUS_PREREQ, "algorithm": ALGORITHM, "baseline_commit": BASELINE,
            "master_status": master.get("status"), "master_modified": False,
            "discovery_status": discovery_summary.get("status"),
            "discovery_control_pass": discovery_summary.get("control", {}).get("pass"),
        }
        _write_json(out / "summary.json", s)
        _write_json(out / "run_state.json", {"status": "COMPLETE", "scientific_status": STATUS_PREREQ})
        return _finalize(out, s)

    geom, src = _source(root / SOURCE_CACHE)
    lad = _load_path(root / LAD)
    discovered = _load_path(root / DISCOVERY_PATH)
    lad, endpoint_match = _orient_lad_distal_first(lad, discovered)
    distal = lad[0].copy()
    lad_arc = _arc(lad)
    if lad_arc[-1] < 12.0:
        raise RuntimeError(f"Frozen LAD unexpectedly short: {lad_arc[-1]:.3f} mm")

    terminal_anchor = _interp(lad, np.linspace(0, min(10.0, lad_arc[-1]), 41))
    vesselness = LocalVesselness(geom, src, terminal_anchor, margin_mm=VESSELNESS_MARGIN_MM)
    vv = np.asarray(vesselness.sample(geom, terminal_anchor), float)
    good = vv[np.isfinite(vv) & (vv > 0)]
    if not len(good):
        raise RuntimeError("No positive local vesselness on terminal frozen LAD")
    vthr = max(1e-5, MIN_VESSELNESS_FACTOR * float(np.percentile(good, 20)))
    vnorm = max(1e-4, float(np.median(good)))
    calibration = {
        "terminal_lad_vesselness_p20": float(np.percentile(good, 20)),
        "terminal_lad_vesselness_median": float(np.median(good)),
        "vesselness_threshold": vthr, "vesselness_norm": vnorm,
        "local_vesselness_roi_shape_zyx": list(vesselness.shape),
        "local_vesselness_margin_mm": VESSELNESS_MARGIN_MM,
    }
    _write_json(out / "vesselness_calibration.json", calibration)

    lad_dense, _ = _resample(lad, .20)
    lad_tree = cKDTree(lad_dense)

    # Positive control: source-led recovery of a known terminal LAD segment.
    control_start_arc = 8.0
    control_start = _interp(lad, [control_start_arc])[0]
    control_t = _unit(_interp(lad, [max(0.0, control_start_arc - 4.0)])[0] - control_start)
    control_finals_all, control_attrs = [], []
    control_dirs = _rank_initial_dirs(geom, src, vesselness, control_start, control_t, vnorm, n=8)
    for di, (preview_score, d) in enumerate(control_dirs):
        finals, attr = _beam(
            geom, src, vesselness, control_start, d, vthr, vnorm,
            CONTROL_SEARCH_MM, known_tree=None, exclude_known=False
        )
        if len(attr):
            attr["initial_dir_index"] = di
            attr["preview_score"] = preview_score
            control_attrs.append(attr)
        control_finals_all.extend(finals[:2])
    control_rows = []
    for i, st in enumerate(control_finals_all):
        p = np.asarray(st[1], float)
        _, _, qc = _dense_qc(geom, src, p)
        tr = _truncate_qc(qc)
        ov = _overlap_metrics(p, lad)
        passed = bool(
            tr["accepted_arc_mm"] >= MIN_CONTROL_ACCEPTED_ARC_MM
            and tr["accepted_plane_pass_fraction"] >= MIN_PLANE_PASS_FRACTION
            and ov["fraction_within_2mm"] >= MIN_CONTROL_OVERLAP_FRACTION_2MM
            and ov["median_distance_mm"] <= MAX_CONTROL_MEDIAN_DISTANCE_MM
        )
        control_rows.append({
            "control_id": f"control_{i:03d}", "beam_score": float(st[0]), **tr,
            **{f"lad_{k}": v for k, v in ov.items()}, "control_gate_pass": passed
        })
    cdf = pd.DataFrame(control_rows)
    cdf.to_csv(out / "known_lad_control_candidates.csv", index=False)
    if control_attrs:
        pd.concat(control_attrs, ignore_index=True).to_csv(out / "known_lad_control_attrition.csv", index=False)
    passing_control = cdf[cdf.control_gate_pass == True] if len(cdf) else cdf
    control_ok = bool(len(passing_control))
    control_best = (
        passing_control.sort_values(["accepted_arc_mm", "lad_median_distance_mm"], ascending=[False, True]).iloc[0].to_dict()
        if control_ok else (cdf.sort_values("lad_median_distance_mm").iloc[0].to_dict() if len(cdf) else {})
    )
    _write_json(out / "known_lad_control.json", {"pass": control_ok, "best": control_best})
    if not control_ok:
        s = {
            "status": STATUS_CONTROL, "algorithm": ALGORITHM, "baseline_commit": BASELINE,
            "master_status": master.get("status"), "master_modified": False,
            "frozen_lad_length_mm": float(lad_arc[-1]), "tested_distal_endpoint_lps_mm": distal.tolist(),
            "endpoint_selection_prior_path_distance_mm": endpoint_match,
            "calibration": calibration, "control": {"pass": False, "best": control_best},
        }
        _write_json(out / "summary.json", s)
        _write_json(out / "run_state.json", {"status": "COMPLETE", "scientific_status": STATUS_CONTROL})
        return _finalize(out, s)

    # Independent target trace: previous blind path is not used anywhere in tracing/scoring.
    outgoing = _unit(distal - _interp(lad, [4.0])[0])
    target_attrs, finalists = [], []
    target_dirs = _rank_initial_dirs(geom, src, vesselness, distal, outgoing, vnorm, n=8)
    for di, (preview_score, d) in enumerate(target_dirs):
        finals, attr = _beam(
            geom, src, vesselness, distal, d, vthr, vnorm,
            TARGET_SEARCH_MM, known_tree=lad_tree, exclude_known=True
        )
        if len(attr):
            attr["initial_dir_index"] = di
            attr["preview_score"] = preview_score
            target_attrs.append(attr)
        for st in finals[:2]:
            finalists.append(st)
    if target_attrs:
        pd.concat(target_attrs, ignore_index=True).to_csv(out / "target_search_attrition.csv", index=False)

    hyp, paths, qcs = _evaluate_paths(geom, src, finalists, lad_tree)
    hyp.to_csv(out / "target_hypotheses.csv", index=False)
    valid = hyp[hyp.extension_gate_pass == True].copy() if len(hyp) else hyp

    best_row, best_path, best_qc = {}, None, None
    if len(valid):
        best = valid.sort_values(
            ["accepted_arc_mm", "accepted_plane_pass_fraction", "beam_score"],
            ascending=[False, False, False]
        ).iloc[0]
        best_row = best.to_dict()
        hid = str(best.hypothesis_id)
        best_path = paths[hid]
        best_qc = qcs[hid]
        pd.DataFrame(best_path, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
            out / "best_independent_distal_extension.csv", index=False
        )
        best_qc.to_csv(out / "best_independent_distal_extension_dense_qc.csv", index=False)

    reference_tail = _reference_tail(discovered, distal, lad_tree)
    pd.DataFrame(reference_tail, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
        out / "prior_blind_path_beyond_distal_endpoint.csv", index=False
    )
    if best_path is not None:
        agreement = _overlap_metrics(best_path, reference_tail)
        agreement["pass"] = bool(
            agreement["fraction_within_2mm"] >= MIN_REFERENCE_MATCH_FRACTION_2MM
            and agreement["median_distance_mm"] <= MAX_REFERENCE_MEDIAN_DISTANCE_MM
            and agreement["median_tangent_alignment_within_2mm"] >= MIN_REFERENCE_TANGENT_ALIGNMENT
        )
    else:
        agreement = {
            "pass": False, "fraction_within_2mm": 0.0, "median_distance_mm": np.inf,
            "median_tangent_alignment_within_2mm": 0.0
        }
    _write_json(out / "prior_blind_path_agreement.json", agreement)

    if best_path is None:
        status = STATUS_NONE
    elif agreement["pass"]:
        status = STATUS_REPRO
    else:
        status = STATUS_OTHER

    _plot_geometry(out, lad, reference_tail, best_path)
    _plot_qc(out, geom, src, best_path)

    summary = {
        "status": status, "algorithm": ALGORITHM, "baseline_commit": BASELINE,
        "master_status": master.get("status"), "master_modified": False,
        "frozen_lad_length_mm": float(lad_arc[-1]),
        "tested_distal_endpoint_lps_mm": distal.tolist(),
        "endpoint_selection_prior_path_distance_mm": endpoint_match,
        "terminal_outgoing_tangent_lps": outgoing.tolist(),
        "calibration": calibration,
        "control": {"pass": True, "best": control_best},
        "target_hypothesis_count": int(len(hyp)),
        "valid_extension_hypothesis_count": int(len(valid)),
        "best_target": best_row,
        "reference_tail_length_mm": float(_arc(reference_tail)[-1]) if len(reference_tail) else 0.0,
        "reference_agreement": agreement,
        "scientific_boundary": (
            "Independent source-CCTA endpoint-continuation adjudication. The prior blind path is "
            "used only to select the tested frozen-LAD endpoint and for post-hoc reproducibility. "
            "It does not guide or score the target trace. Frozen anatomy and clinical labels remain unchanged."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "run_state.json", {
        "status": "COMPLETE", "scientific_status": status, "algorithm": ALGORITHM,
        "baseline_commit": BASELINE, "master_modified": False
    })
    return _finalize(out, summary)
