"""Open-path coronary tracking for Siemens curved coronary reformats.

Presentation/QC only. This module does not alter canonical TPV and does not
establish source-volume co-registration.

V2 intentionally rejects the long chamber-boundary loops seen in the first
image-driven prototype. Candidate paths must be open, coronary-scale, and
reasonably direct. CT image evidence drives path generation; plaque is used
only as an independent selection/QC consistency signal.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from skimage.filters import frangi
from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize


def largest_body_mask(img):
    img = np.asarray(img, float)
    m = img > -700
    m = ndi.binary_closing(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    return ndi.binary_fill_holes(lab == k)


def robust01(a, mask=None, lo_q=2, hi_q=98):
    a = np.asarray(a, float)
    vals = a[mask] if mask is not None and np.any(mask) else a.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(a, float)
    lo, hi = np.percentile(vals, [lo_q, hi_q])
    if hi <= lo:
        return np.zeros_like(a, float)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _neighbors8(p):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                yield (y + dy, x + dx)


def _component_graph_stats(component):
    coords = [tuple(x) for x in np.argwhere(component)]
    s = set(coords)
    if not coords:
        return {"nodes": 0, "edges": 0, "endpoints": 0, "junctions": 0, "cycle_rank": 0}
    degrees = []
    edges2 = 0
    for p in coords:
        d = sum(q in s for q in _neighbors8(p))
        degrees.append(d)
        edges2 += d
    edges = edges2 // 2
    cycle_rank = max(0, int(edges - len(coords) + 1))
    return {
        "nodes": len(coords),
        "edges": int(edges),
        "endpoints": int(sum(d == 1 for d in degrees)),
        "junctions": int(sum(d >= 3 for d in degrees)),
        "cycle_rank": cycle_rank,
    }


def _bfs_farthest(coord_set, start):
    q = deque([start])
    dist = {start: 0}
    parent = {start: None}
    far = start
    while q:
        p = q.popleft()
        if dist[p] > dist[far]:
            far = p
        for z in _neighbors8(p):
            if z in coord_set and z not in dist:
                dist[z] = dist[p] + 1
                parent[z] = p
                q.append(z)
    return far, dist, parent


def diameter_path(component):
    coords = np.argwhere(component)
    if len(coords) < 12:
        return None
    coord_set = {tuple(x) for x in coords}
    a, _, _ = _bfs_farthest(coord_set, tuple(coords[0]))
    b, _, parent = _bfs_farthest(coord_set, a)
    path = []
    p = b
    while p is not None:
        path.append(p)
        p = parent[p]
    return np.asarray(path[::-1], float)


def path_physical_length(path, sp_yx):
    if path is None or len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0) * np.asarray(sp_yx)[None, :]
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def endpoint_distance(path, sp_yx):
    if path is None or len(path) < 2:
        return 0.0
    d = (path[-1] - path[0]) * np.asarray(sp_yx)
    return float(np.sqrt(np.sum(d * d)))


def total_turn_degrees(path, sp_yx):
    if path is None or len(path) < 4:
        return 0.0
    p = path * np.asarray(sp_yx)[None, :]
    y = gaussian_filter1d(p[:, 0], 1.2, mode="nearest")
    x = gaussian_filter1d(p[:, 1], 1.2, mode="nearest")
    d = np.diff(np.column_stack([y, x]), axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-6
    d = d[keep]
    n = n[keep]
    if len(d) < 2:
        return 0.0
    u = d / n[:, None]
    c = np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1)
    return float(np.degrees(np.arccos(c)).sum())


def image_evidence(img):
    """Small bright tubular evidence with broad cardiac chambers suppressed."""
    img = np.asarray(img, float)
    body = largest_body_mask(img)
    h, w = img.shape
    interior = np.zeros_like(body, bool)
    my = max(10, int(0.06 * h))
    mx = max(10, int(0.06 * w))
    interior[my:h-my, mx:w-mx] = True
    valid = body & interior

    band = np.clip((img - 80.0) / 620.0, 0, 1)
    band[(img < 80) | (img > 850)] = 0

    norm = np.clip((img + 100.0) / 900.0, 0, 1)
    vesselness = frangi(norm, sigmas=(1, 1.5, 2, 2.5, 3), black_ridges=False)
    vesselness = robust01(vesselness, valid)

    small = gaussian_filter(img, 1.0)
    broad = gaussian_filter(img, 7.0)
    dog = np.maximum(small - broad, 0)
    dog = robust01(dog, valid, 5, 99)

    local_mean = ndi.uniform_filter(img, size=17, mode="nearest")
    local_contrast = np.maximum(img - local_mean, 0)
    local_contrast = robust01(local_contrast, valid, 5, 99)

    evidence = (0.50 * vesselness + 0.32 * dog + 0.12 * local_contrast + 0.06 * band) * valid
    return evidence, valid, {
        "vesselness": vesselness,
        "dog": dog,
        "local_contrast": local_contrast,
        "band": band,
    }


def path_tube(path, shape, sp_yx, radius_mm=5.0):
    path_mask = np.zeros(shape, bool)
    yy = np.clip(np.rint(path[:, 0]).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.rint(path[:, 1]).astype(int), 0, shape[1] - 1)
    path_mask[yy, xx] = True
    dist = ndi.distance_transform_edt(~path_mask, sampling=sp_yx)
    return dist <= radius_mm, path_mask


def _path_metrics(path, img, evidence_parts, evidence, sp_yx):
    iy = np.clip(np.rint(path[:, 0]).astype(int), 0, img.shape[0] - 1)
    ix = np.clip(np.rint(path[:, 1]).astype(int), 0, img.shape[1] - 1)
    length = path_physical_length(path, sp_yx)
    straight = endpoint_distance(path, sp_yx)
    sinuosity = length / max(straight, 1e-6)
    mean_ev = float(evidence[iy, ix].mean())
    mean_vesselness = float(evidence_parts["vesselness"][iy, ix].mean())
    mean_dog = float(evidence_parts["dog"][iy, ix].mean())
    mean_hu = float(img[iy, ix].mean())
    turn = total_turn_degrees(path, sp_yx)
    return {
        "path_length_mm": length,
        "endpoint_distance_mm": straight,
        "sinuosity": float(sinuosity),
        "mean_evidence": mean_ev,
        "mean_vesselness": mean_vesselness,
        "mean_dog": mean_dog,
        "mean_path_hu": mean_hu,
        "total_turn_deg": turn,
    }


def frame_candidates(
    img,
    sp_yx,
    plaque2d=None,
    min_length_mm=20.0,
    max_length_mm=220.0,
    max_sinuosity=2.6,
    max_cycle_rank=0,
    max_candidates=12,
):
    """Return plausible open coronary-scale paths from one CPR frame."""
    img = np.asarray(img, float)
    evidence, valid, parts = image_evidence(img)
    vals = evidence[valid]
    if vals.size < 100:
        return []

    plaque2d = np.asarray(plaque2d, bool) if plaque2d is not None else np.zeros_like(img, bool)
    records = []
    seen = set()

    for q in (95, 93, 91, 89, 87, 85, 83):
        threshold = float(np.percentile(vals, q))
        candidate = (evidence >= threshold) & valid
        candidate = binary_closing(candidate, disk(1))
        candidate = remove_small_objects(candidate, min_size=12)
        skel = skeletonize(candidate)
        labs = label(skel, connectivity=2)

        for k in range(1, labs.max() + 1):
            comp = labs == k
            if comp.sum() < 12:
                continue
            stats = _component_graph_stats(comp)
            if stats["cycle_rank"] > max_cycle_rank:
                continue
            if stats["endpoints"] < 2:
                continue

            path = diameter_path(comp)
            if path is None or len(path) < 12:
                continue
            m = _path_metrics(path, img, parts, evidence, sp_yx)
            if not (min_length_mm <= m["path_length_mm"] <= max_length_mm):
                continue
            if m["sinuosity"] > max_sinuosity:
                continue
            if not (80 <= m["mean_path_hu"] <= 850):
                continue

            ep = np.r_[path[0], path[-1]]
            key = tuple(np.rint(ep / 4).astype(int))
            key_rev = tuple(np.rint(np.r_[path[-1], path[0]] / 4).astype(int))
            if key in seen or key_rev in seen:
                continue
            seen.add(key)

            tube, path_mask = path_tube(path, img.shape, sp_yx, radius_mm=5.0)
            plaque_frame = int(plaque2d.sum())
            plaque_tube = int((plaque2d & tube).sum())
            plaque_capture = plaque_tube / max(plaque_frame, 1) if plaque_frame else 0.0

            length = m["path_length_mm"]
            length_quality = math.exp(-0.5 * ((length - 95.0) / 65.0) ** 2)
            directness = math.exp(-0.9 * max(0.0, m["sinuosity"] - 1.0))
            curvature = math.exp(-max(0.0, m["total_turn_deg"] - 220.0) / 500.0)
            image_score = (
                0.38 * m["mean_vesselness"]
                + 0.30 * m["mean_dog"]
                + 0.18 * m["mean_evidence"]
                + 0.08 * length_quality
                + 0.04 * directness
                + 0.02 * curvature
            )
            records.append({
                "path": path,
                "tube": tube,
                "path_mask": path_mask,
                "threshold_percentile": q,
                "threshold": threshold,
                "graph_endpoints": stats["endpoints"],
                "graph_junctions": stats["junctions"],
                "cycle_rank": stats["cycle_rank"],
                "plaque_frame_voxels": plaque_frame,
                "plaque_tube_voxels": plaque_tube,
                "plaque_capture_fraction": float(plaque_capture),
                "image_score": float(image_score),
                **m,
            })

    records.sort(key=lambda d: d["image_score"], reverse=True)
    return records[:max_candidates]


def select_global_candidate(candidates):
    if not candidates:
        return None, []
    im = np.array([c["image_score"] for c in candidates], float)
    lo, hi = float(im.min()), float(im.max())
    for c in candidates:
        image_norm = (c["image_score"] - lo) / max(hi - lo, 1e-9)
        c["selection_score"] = (
            0.72 * image_norm
            + 0.23 * float(c["plaque_capture_fraction"])
            + 0.05 * (1.0 if c["plaque_tube_voxels"] > 0 else 0.0)
        )
    positive = [c for c in candidates if c["plaque_tube_voxels"] > 0]
    pool = positive if positive else candidates
    pool = sorted(pool, key=lambda d: d["selection_score"], reverse=True)
    return pool[0], pool


def qc_status(candidate, vessel_total_plaque_voxels):
    if candidate is None:
        return "FAIL", "no plausible open coronary-scale path"
    fail_reasons = []
    if not (20 <= candidate["path_length_mm"] <= 220):
        fail_reasons.append("length")
    if candidate["cycle_rank"] > 0:
        fail_reasons.append("cycle")
    if candidate["sinuosity"] > 2.6:
        fail_reasons.append("sinuosity")
    if not (80 <= candidate["mean_path_hu"] <= 850):
        fail_reasons.append("HU")
    if vessel_total_plaque_voxels > 0 and candidate["plaque_tube_voxels"] == 0:
        fail_reasons.append("zero plaque capture")
    if fail_reasons:
        return "FAIL", ", ".join(fail_reasons)
    if candidate["plaque_frame_voxels"] > 0 and candidate["plaque_capture_fraction"] < 0.25:
        return "REVIEW", "captures <25% of plaque in selected frame"
    if candidate["mean_vesselness"] < 0.20 or candidate["mean_dog"] < 0.15:
        return "REVIEW", "weak tube evidence"
    return "OK", ""


def smooth_resample_path(path, sp_yx, step_mm=0.35):
    y = gaussian_filter1d(path[:, 0].astype(float), 1.2, mode="nearest")
    x = gaussian_filter1d(path[:, 1].astype(float), 1.2, mode="nearest")
    p = np.column_stack([y, x])
    mm = p * np.asarray(sp_yx)[None, :]
    ds = np.sqrt(((np.diff(mm, axis=0)) ** 2).sum(axis=1))
    s = np.r_[0, np.cumsum(ds)]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s = s[keep]
    mm = mm[keep]
    if len(s) < 3 or s[-1] < 1:
        return None
    su = np.arange(0, s[-1] + 1e-9, step_mm)
    ymu = np.interp(su, s, mm[:, 0])
    xmu = np.interp(su, s, mm[:, 1])
    ymu = gaussian_filter1d(ymu, 1.0)
    xmu = gaussian_filter1d(xmu, 1.0)
    return su, ymu, xmu


def straighten(img, plaque, path, sp_yx, half_width_mm=6.0, along_step=0.35, cross_step=0.25):
    rr = smooth_resample_path(path, sp_yx, along_step)
    if rr is None:
        return None
    s, ym, xm = rr
    dym = np.gradient(ym)
    dxm = np.gradient(xm)
    norm = np.hypot(dym, dxm)
    norm[norm < 1e-6] = 1
    ty = dym / norm
    tx = dxm / norm
    ny = -tx
    nx = ty
    off = np.arange(-half_width_mm, half_width_mm + 1e-9, cross_step)
    y_mm = ym[None, :] + off[:, None] * ny[None, :]
    x_mm = xm[None, :] + off[:, None] * nx[None, :]
    y_px = y_mm / float(sp_yx[0])
    x_px = x_mm / float(sp_yx[1])
    strip = map_coordinates(np.asarray(img, float), [y_px, x_px], order=1, mode="nearest")
    plaque_strip = map_coordinates(
        np.asarray(plaque, float), [y_px, x_px], order=0, mode="constant", cval=0
    ) > 0.5
    return s, off, strip, plaque_strip
