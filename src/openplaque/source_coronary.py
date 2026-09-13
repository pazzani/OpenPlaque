from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import frangi
from skimage.graph import MCP_Geometric

ALGORITHM_VERSION = "source-volume-global-graph-v1.0"


def arc_mm(path_zyx, spacing_zyx):
    p = np.asarray(path_zyx, float)
    if len(p) < 2:
        return np.array([0.0])
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx)[None, :]
    return np.r_[0.0, np.cumsum(np.sqrt((d * d).sum(axis=1)))]


def resample_path(path_zyx, spacing_zyx, step_mm=0.6):
    p = np.asarray(path_zyx, float)
    s = arc_mm(p, spacing_zyx)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < step_mm:
        return p
    su = np.arange(0.0, s[-1] + 1e-6, step_mm)
    out = np.column_stack([np.interp(su, s, p[:, j]) for j in range(3)])
    return out


def source_to_ds(path_source_zyx, lo_zyx, zoom_zyx):
    return (np.asarray(path_source_zyx, float) - np.asarray(lo_zyx, float)) * np.asarray(zoom_zyx, float)


def ds_to_source(path_ds_zyx, lo_zyx, zoom_zyx):
    return np.asarray(lo_zyx, float) + np.asarray(path_ds_zyx, float) / np.asarray(zoom_zyx, float)


def physical_lps_from_zyx(image, zyx):
    out = []
    for z, y, x in np.asarray(zyx, float):
        out.append(image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))
    return np.asarray(out, float)


def build_downsampled_evidence(ct, aorta, spacing_zyx, seed_source_zyx, half_mm=(82, 96, 96), target_mm=1.0):
    """Crop around the aortic root/heart and build a coronary-scale 3-D evidence volume."""
    ct = np.asarray(ct, np.float32)
    aorta = np.asarray(aorta, bool)
    spacing_zyx = np.asarray(spacing_zyx, float)
    seed = np.asarray(seed_source_zyx, float)
    half_vox = np.ceil(np.asarray(half_mm, float) / spacing_zyx).astype(int)
    o = np.rint(seed).astype(int)
    lo = np.maximum(0, o - half_vox)
    hi = np.minimum(np.asarray(ct.shape), o + half_vox + 1)
    crop = ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    acrop = aorta[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    zoom = spacing_zyx / float(target_mm)
    zoom = np.minimum(1.0, zoom)
    ds_spacing = spacing_zyx / zoom
    dct = ndi.zoom(crop, zoom=zoom, order=1, mode="nearest", prefilter=False).astype(np.float32)
    daorta = ndi.zoom(acrop.astype(np.uint8), zoom=zoom, order=0, mode="nearest", prefilter=False) > 0

    sigma = np.maximum(0.55 / ds_spacing, 0.45)
    sm = gaussian_filter(dct, sigma=sigma)
    intensity = np.clip((sm - 140.0) / 560.0, 0.0, 1.0)
    intensity[(sm < 120) | (sm > 1000)] = 0

    norm = np.clip((sm + 100.0) / 1050.0, 0, 1)
    vesselness = np.nan_to_num(frangi(norm, sigmas=(0.7, 1.0, 1.4, 1.8, 2.2), black_ridges=False))
    positive = vesselness[vesselness > 0]
    v99 = np.percentile(positive, 99.2) if positive.size else 1.0
    vesselness = np.clip(vesselness / max(v99, 1e-8), 0, 1).astype(np.float32)

    small = gaussian_filter(sm, sigma=np.maximum(0.8 / ds_spacing, 0.5))
    broad = gaussian_filter(sm, sigma=np.maximum(4.0 / ds_spacing, 1.5))
    dog = np.maximum(small - broad, 0)
    vals = dog[np.isfinite(dog)]
    p99 = np.percentile(vals, 99.2) if vals.size else 1.0
    dog = np.clip(dog / max(p99, 1e-6), 0, 1).astype(np.float32)

    support = (0.50 * intensity + 0.38 * vesselness + 0.12 * dog).astype(np.float32)
    bright = sm >= 150
    lumen_r = ndi.distance_transform_edt(bright, sampling=ds_spacing).astype(np.float32)
    dist_aorta = ndi.distance_transform_edt(~daorta, sampling=ds_spacing).astype(np.float32)
    large = np.clip((lumen_r - 3.8) / 3.0, 0, 1).astype(np.float32)

    cost = 1.0 / (0.04 + support) + 7.0 * large
    cost[sm < 120] += 16
    cost[daorta] = 1e5
    cost[(dist_aorta < 1.3) & (~daorta)] += 30
    cost = cost.astype(np.float32)

    return {
        "ct": dct,
        "aorta": daorta,
        "support": support,
        "vesselness": vesselness,
        "dog": dog,
        "lumen_radius_mm": lumen_r,
        "dist_aorta_mm": dist_aorta,
        "cost": cost,
        "lo_source_zyx": lo.astype(int),
        "hi_source_zyx": hi.astype(int),
        "zoom_zyx": zoom.astype(float),
        "spacing_zyx": ds_spacing.astype(float),
    }


def _aorta_center_at_z(aorta, z):
    z = int(np.clip(round(z), 0, aorta.shape[0] - 1))
    for dz in range(0, 8):
        for zz in {z - dz, z + dz}:
            if 0 <= zz < aorta.shape[0]:
                yy, xx = np.where(aorta[zz])
                if len(yy) >= 20:
                    return np.array([float(zz), float(np.mean(yy)), float(np.mean(xx))])
    raise RuntimeError("Could not estimate local aortic center")


def detect_left_ostium(evd, rca_seed_ds, topn=40):
    """Find a left-coronary ostium candidate on the aortic wall opposite the RCA seed."""
    support = evd["support"]
    ct = evd["ct"]
    aorta = evd["aorta"]
    da = evd["dist_aorta_mm"]
    lr = evd["lumen_radius_mm"]
    sp = evd["spacing_zyx"]
    rca_seed_ds = np.asarray(rca_seed_ds, float)
    ctr = _aorta_center_at_z(aorta, rca_seed_ds[0])
    rca_vec = (rca_seed_ds - ctr) * sp
    rca_vec[0] *= 0.35
    rca_vec /= max(np.linalg.norm(rca_vec), 1e-6)

    zz, yy, xx = np.indices(aorta.shape)
    dz_mm = np.abs((zz - rca_seed_ds[0]) * sp[0])
    shell = (~aorta) & (da >= 1.3) & (da <= 5.5) & (dz_mm <= 14.0) & (ct >= 150) & (ct <= 950) & (lr <= 4.8)
    if np.any(shell):
        sth = np.percentile(support[shell], 70)
        shell &= support >= sth
    coords = np.argwhere(shell)
    if len(coords) == 0:
        raise RuntimeError("No aortic-wall candidates for left ostium")

    scores = []
    for p in coords:
        c = _aorta_center_at_z(aorta, p[0])
        rv = (p.astype(float) - c) * sp
        rv[0] *= 0.35
        n = np.linalg.norm(rv)
        if n < 2:
            continue
        u = rv / n
        opposite = -float(np.dot(u, rca_vec))
        if opposite < -0.10:
            continue
        ray_support = []
        ray_hu = []
        for mm in np.linspace(2, 12, 8):
            q_mm = p * sp + u * mm
            q = q_mm / sp
            if np.any(q < 1) or np.any(q >= np.asarray(ct.shape) - 2):
                continue
            ray_support.append(float(map_coordinates(support, q[:, None], order=1, mode="nearest")[0]))
            ray_hu.append(float(map_coordinates(ct, q[:, None], order=1, mode="nearest")[0]))
        if len(ray_support) < 5:
            continue
        rs = np.asarray(ray_support)
        rh = np.asarray(ray_hu)
        plausible = float(np.mean((rh >= 120) & (rh <= 950)))
        score = 0.47 * float(np.mean(rs)) + 0.23 * float(np.percentile(rs, 25)) + 0.16 * opposite + 0.10 * plausible + 0.04 * float(support[tuple(p)])
        scores.append((score, p.astype(float), u, opposite, float(np.mean(rs)), plausible))
    if not scores:
        raise RuntimeError("No viable left ostium candidate after outward-support test")
    scores.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for rank, (score, p, u, opp, mean_s, plausible) in enumerate(scores[:topn], 1):
        rows.append({"rank": rank, "score": score, "z": p[0], "y": p[1], "x": p[2], "dir_z": u[0], "dir_y": u[1], "dir_x": u[2], "opposite_rca": opp, "outward_mean_support": mean_s, "outward_plausible_hu": plausible})
    best = rows[0]
    return np.array([best["z"], best["y"], best["x"]], float), np.array([best["dir_z"], best["dir_y"], best["dir_x"]], float), pd.DataFrame(rows)


def _mcp_paths(cost, start, endpoints):
    mcp = MCP_Geometric(np.asarray(cost, float), fully_connected=True)
    mcp.find_costs([tuple(np.rint(start).astype(int))])
    out = []
    for ep in endpoints:
        try:
            p = np.asarray(mcp.traceback(tuple(np.rint(ep).astype(int))), int)
        except Exception:
            continue
        if len(p) >= 2:
            out.append(p)
    return out


def _path_metrics(path, evd, heading=None):
    p = np.asarray(path, int)
    sp = evd["spacing_zyx"]
    s = arc_mm(p, sp)
    L = float(s[-1])
    su = evd["support"][tuple(p.T)]
    ct = evd["ct"][tuple(p.T)]
    da = evd["dist_aorta_mm"][tuple(p.T)]
    lr = evd["lumen_radius_mm"][tuple(p.T)]
    step = np.diff(p.astype(float), axis=0) * sp
    if len(step) >= 2:
        u = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-6)
        ang = np.arccos(np.clip(np.sum(u[:-1] * u[1:], axis=1), -1, 1))
        mean_turn = float(np.mean(ang))
        p95_turn = float(np.percentile(ang, 95))
    else:
        mean_turn = p95_turn = math.pi
    heading_angle = 0.0
    if heading is not None and len(p) > 4:
        first = (p[min(len(p)-1, 6)] - p[0]) * sp
        heading_angle = float(np.degrees(np.arccos(np.clip(np.dot(first, heading) / (np.linalg.norm(first) * max(np.linalg.norm(heading), 1e-6) + 1e-9), -1, 1))))
    return {
        "length_mm": L,
        "mean_support": float(np.mean(su)),
        "p10_support": float(np.percentile(su, 10)),
        "mean_hu": float(np.mean(ct)),
        "p10_hu": float(np.percentile(ct, 10)),
        "aorta_reentry_fraction": float(np.mean(da < 1.3)),
        "large_lumen_fraction": float(np.mean(lr > 5.0)),
        "mean_lumen_radius_mm": float(np.mean(lr)),
        "end_dist_aorta_mm": float(da[-1]),
        "mean_turn_rad": mean_turn,
        "p95_turn_rad": p95_turn,
        "heading_angle_deg": heading_angle,
    }


def _endpoint_candidates(evd, start, heading, min_r=28, max_r=115, min_forward=10, max_endpoints=36):
    support = evd["support"]
    da = evd["dist_aorta_mm"]
    lr = evd["lumen_radius_mm"]
    ct = evd["ct"]
    sp = evd["spacing_zyx"]
    G = np.indices(support.shape, dtype=np.float32).reshape(3, -1).T
    D = (G - np.asarray(start)[None, :]) * sp[None, :]
    rad = np.linalg.norm(D, axis=1)
    h = np.asarray(heading, float)
    h /= max(np.linalg.norm(h), 1e-6)
    forward = D @ h
    su = support.ravel(); dd = da.ravel(); rr = lr.ravel(); hv = ct.ravel()
    sth = np.percentile(support[support > 0], 82) if np.any(support > 0) else 0.2
    m = (rad >= min_r) & (rad <= max_r) & (forward >= min_forward) & (su >= sth) & (dd >= 2.0) & (rr <= 5.0) & (hv >= 120) & (hv <= 1000)
    cand = G[m].astype(int)
    if len(cand) == 0:
        return []
    val = su[m] + 0.025 * np.minimum(dd[m], 16) - 0.07 * np.maximum(rr[m] - 3.5, 0) + 0.002 * np.minimum(forward[m], 80)
    order = np.argsort(val)[::-1]
    eps = []
    for j in order:
        p = cand[j]
        if all(np.linalg.norm((p - q) * sp) >= 5.0 for q in eps):
            eps.append(p)
        if len(eps) >= max_endpoints:
            break
    return eps


def trace_routes(evd, start_ds, heading_ds_mm, min_len=28, max_len=145, max_routes=24):
    eps = _endpoint_candidates(evd, start_ds, heading_ds_mm, min_r=min_len, max_r=max_len, min_forward=8, max_endpoints=42)
    if not eps:
        raise RuntimeError("No distal coronary endpoint candidates")
    paths = _mcp_paths(evd["cost"], start_ds, eps)
    routes = []
    for p in paths:
        m = _path_metrics(p, evd, heading_ds_mm)
        if not (min_len <= m["length_mm"] <= max_len):
            continue
        if m["aorta_reentry_fraction"] > 0.02 or m["large_lumen_fraction"] > 0.12:
            continue
        score = (2.0*m["mean_support"] + 0.65*m["p10_support"] + 0.008*min(m["length_mm"], 110) + 0.018*min(m["end_dist_aorta_mm"], 18) - 0.35*m["mean_turn_rad"] - 0.12*m["p95_turn_rad"] - 0.004*m["heading_angle_deg"] - 1.2*m["large_lumen_fraction"])
        routes.append({"path": p, "score": float(score), **m})
    routes.sort(key=lambda r: r["score"], reverse=True)
    return routes[:max_routes]


def _sample_resampled_mm(path, sp, step=1.0):
    p = resample_path(path, sp, step_mm=step)
    return p * np.asarray(sp)[None, :]


def _pair_common_length(a, b, sp, tol_mm=4.0):
    am = _sample_resampled_mm(a, sp, 1.0)
    bm = _sample_resampled_mm(b, sp, 1.0)
    n = min(len(am), len(bm))
    if n == 0:
        return 0.0
    d = np.linalg.norm(am[:n] - bm[:n], axis=1)
    idx = np.where(d > tol_mm)[0]
    return float(idx[0]) if len(idx) else float(n - 1)


def select_left_branch_pair(routes, evd, image, lo_source_zyx, zoom_zyx):
    """Select two globally strong, diverging branches from a shared left-main seed."""
    if len(routes) < 2:
        raise RuntimeError("Need at least two left-coronary routes")
    sp = evd["spacing_zyx"]
    pairs = []
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            a, b = routes[i], routes[j]
            common = _pair_common_length(a["path"], b["path"], sp)
            enda = a["path"][-1] * sp
            endb = b["path"][-1] * sp
            sep = float(np.linalg.norm(enda - endb))
            if sep < 22 or common < 2 or common > 45:
                continue
            common_quality = math.exp(-0.5 * ((common - 12.0) / 15.0) ** 2)
            score = a["score"] + b["score"] + 0.020*min(sep, 70) + 0.22*common_quality
            pairs.append((score, common, sep, a, b))
    if not pairs:
        raise RuntimeError("Could not find a sufficiently diverging LAD/LCX route pair")
    pairs.sort(key=lambda x: x[0], reverse=True)
    _, common, sep, a, b = pairs[0]

    def src_path(r): return ds_to_source(r["path"], lo_source_zyx, zoom_zyx)
    pa, pb = src_path(a), src_path(b)
    pha = physical_lps_from_zyx(image, [pa[-1]])[0]
    phb = physical_lps_from_zyx(image, [pb[-1]])[0]
    start_ph = physical_lps_from_zyx(image, [pa[0]])[0]
    da, db = pha - start_ph, phb - start_ph
    lad_a = -1.00*da[2] - 0.45*da[1] + 0.10*da[0]
    lad_b = -1.00*db[2] - 0.45*db[1] + 0.10*db[0]
    if lad_a >= lad_b:
        lad, lcx = a, b
    else:
        lad, lcx = b, a
    return lad, lcx, {"pair_score": float(pairs[0][0]), "common_trunk_mm": common, "endpoint_separation_mm": sep, "lad_metric_a": float(lad_a), "lad_metric_b": float(lad_b)}


def trace_rca(evd, seed_ds, heading_ds_mm):
    routes = trace_routes(evd, seed_ds, heading_ds_mm, min_len=35, max_len=150, max_routes=16)
    if not routes:
        raise RuntimeError("No plausible RCA global-graph route")
    return routes[0], routes


def parallel_transport_frame(centerline_source_zyx, source_spacing_zyx):
    p = resample_path(centerline_source_zyx, source_spacing_zyx, step_mm=0.6)
    pm = p * np.asarray(source_spacing_zyx)[None, :]
    t = np.gradient(pm, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
    n = np.zeros_like(t)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, t[0])) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    n[0] = ref - np.dot(ref, t[0]) * t[0]
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(t)):
        v = n[i-1] - np.dot(n[i-1], t[i]) * t[i]
        if np.linalg.norm(v) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0]) if abs(t[i,0]) < 0.85 else np.array([0.0, 1.0, 0.0])
            v = ref - np.dot(ref, t[i]) * t[i]
        n[i] = v / np.linalg.norm(v)
    b = np.cross(t, n)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return p, t, n, b


def generate_rotating_cpr(ct, centerline_source_zyx, source_spacing_zyx, n_rot=24, half_width_mm=18.0, cross_step_mm=0.45):
    """Generate an OpenPlaque-owned rotating CPR stack plus reconstructible geometry."""
    ct = np.asarray(ct, float)
    sp = np.asarray(source_spacing_zyx, float)
    p, t, n, b = parallel_transport_frame(centerline_source_zyx, sp)
    pm = p * sp[None, :]
    s = arc_mm(p, sp)
    cross = np.arange(-half_width_mm, half_width_mm + 1e-6, cross_step_mm)
    stack = []
    for k in range(n_rot):
        theta = 2*np.pi*k/n_rot
        u = math.cos(theta)*n + math.sin(theta)*b
        coords_mm = pm[None, :, :] + cross[:, None, None] * u[None, :, :]
        coords = coords_mm / sp[None, None, :]
        vals = map_coordinates(ct, [coords[...,0], coords[...,1], coords[...,2]], order=1, mode="nearest")
        stack.append(vals.astype(np.float32))
    return {
        "stack": np.stack(stack, axis=0),
        "centerline_zyx": p.astype(np.float32),
        "tangent_zyx_mm": t.astype(np.float32),
        "normal_zyx_mm": n.astype(np.float32),
        "binormal_zyx_mm": b.astype(np.float32),
        "arc_mm": s.astype(np.float32),
        "cross_mm": cross.astype(np.float32),
        "rotation_radians": np.linspace(0, 2*np.pi, n_rot, endpoint=False).astype(np.float32),
    }


def centerline_qc(path_source_zyx, ct, spacing_zyx, vessel_name):
    p = resample_path(path_source_zyx, spacing_zyx, 0.8)
    idx = np.rint(p).astype(int)
    idx = np.clip(idx, [0,0,0], np.asarray(ct.shape)-1)
    hu = np.asarray(ct)[tuple(idx.T)]
    s = arc_mm(p, spacing_zyx)
    return {
        "vessel": vessel_name,
        "length_mm": float(s[-1]) if len(s) else 0.0,
        "points": int(len(p)),
        "mean_centerline_hu": float(np.mean(hu)) if len(hu) else np.nan,
        "p10_centerline_hu": float(np.percentile(hu,10)) if len(hu) else np.nan,
        "plausible_lumen_fraction": float(np.mean((hu>=120)&(hu<=1000))) if len(hu) else 0.0,
    }
