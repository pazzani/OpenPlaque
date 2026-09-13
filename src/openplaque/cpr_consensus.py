"""Cross-rotation coronary tracking for Siemens curved coronary reformats.

The Siemens CPR stack is treated as alternate rotations around one coronary path.
The target coronary should remain relatively stable in CPR pixel coordinates while
other anatomy changes with rotation.  We therefore build a persistent evidence map
across all rotations, track on that consensus map, and only then choose the single
rotation that displays the fixed path best.

Presentation/QC only: this module does not alter canonical TPV and does not establish
source-volume co-registration.
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

TRACKER_VERSION = "cross-rotation-consensus-v3.0"


def largest_body_mask(img):
    img = np.asarray(img, float)
    m = img > -700
    m = ndi.binary_closing(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))


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


def image_evidence(img):
    """Return coronary-scale bright tubular evidence for one CPR rotation."""
    img = np.asarray(img, float)
    body = largest_body_mask(img)
    h, w = img.shape
    interior = np.zeros_like(body, bool)
    my = max(12, int(0.085 * h))
    mx = max(12, int(0.085 * w))
    interior[my:h-my, mx:w-mx] = True
    valid = body & interior

    norm = np.clip((img + 100.0) / 950.0, 0, 1)
    vesselness = frangi(norm, sigmas=(1.0, 1.5, 2.0, 2.5, 3.0), black_ridges=False)
    vesselness = robust01(vesselness, valid, 3, 99)

    # Small bright objects survive; broad chambers are largely subtracted.
    small = gaussian_filter(img, 1.0)
    broad = gaussian_filter(img, 8.0)
    dog = robust01(np.maximum(small - broad, 0), valid, 5, 99)

    local = ndi.uniform_filter(img, size=19, mode="nearest")
    contrast = robust01(np.maximum(img - local, 0), valid, 5, 99)

    # Contrast-filled lumen plausibility. This is intentionally weak relative to shape.
    bright = np.exp(-0.5 * ((img - 500.0) / 300.0) ** 2)
    bright[(img < 90) | (img > 950)] = 0

    evidence = (0.50 * vesselness + 0.30 * dog + 0.14 * contrast + 0.06 * bright) * valid
    return np.asarray(evidence, np.float32), valid


def build_consensus(volume):
    """Build persistent coronary evidence across all CPR rotations.

    The lower quartile is deliberately important: a structure that only appears in a
    few rotations cannot score highly merely because it is extremely bright there.
    """
    volume = np.asarray(volume, float)
    evs, valids = [], []
    for z in range(volume.shape[0]):
        e, v = image_evidence(volume[z])
        evs.append(e)
        valids.append(v)
    stack = np.stack(evs, axis=0)
    vstack = np.stack(valids, axis=0)

    q25 = np.quantile(stack, 0.25, axis=0)
    med = np.median(stack, axis=0)
    q70 = np.quantile(stack, 0.70, axis=0)
    support = np.mean(stack >= 0.45, axis=0)
    valid_fraction = np.mean(vstack, axis=0)

    base = 0.42 * q25 + 0.26 * med + 0.17 * q70 + 0.15 * support
    valid = valid_fraction >= 0.72
    base = robust01(base, valid, 2, 99)

    # CPR target vessel is designed to remain in the central field.  This prior is
    # deliberately modest: it cannot create evidence where persistence is absent.
    h, w = base.shape
    yy, xx = np.mgrid[:h, :w]
    dy = (yy - (h - 1) / 2) / (0.5 * h)
    dx = (xx - (w - 1) / 2) / (0.5 * w)
    rr = np.sqrt(dx * dx + dy * dy)
    center_prior = 0.55 + 0.45 * np.exp(-0.5 * (rr / 0.72) ** 2)
    consensus = base * center_prior * valid

    return {
        "consensus": np.asarray(consensus, np.float32),
        "support": np.asarray(support, np.float32),
        "median": np.asarray(med, np.float32),
        "q25": np.asarray(q25, np.float32),
        "valid_fraction": np.asarray(valid_fraction, np.float32),
    }


def _neighbors8(p):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                yield y + dy, x + dx


def _graph_degrees(component):
    coords = [tuple(x) for x in np.argwhere(component)]
    s = set(coords)
    deg = {p: sum(q in s for q in _neighbors8(p)) for p in coords}
    return s, deg


def _bfs(coord_set, start):
    q = deque([start])
    dist = {start: 0}
    parent = {start: None}
    while q:
        p = q.popleft()
        for z in _neighbors8(p):
            if z in coord_set and z not in dist:
                dist[z] = dist[p] + 1
                parent[z] = p
                q.append(z)
    return dist, parent


def endpoint_diameter_path(component):
    """Longest endpoint-to-endpoint path through an open skeleton component."""
    coord_set, deg = _graph_degrees(component)
    endpoints = [p for p, d in deg.items() if d == 1]
    if len(endpoints) < 2:
        return None, {"endpoints": len(endpoints), "junctions": sum(d >= 3 for d in deg.values())}
    best = None
    best_len = -1
    for a in endpoints:
        dist, parent = _bfs(coord_set, a)
        reachable = [b for b in endpoints if b in dist and b != a]
        if not reachable:
            continue
        b = max(reachable, key=lambda x: dist[x])
        if dist[b] > best_len:
            path = []
            p = b
            while p is not None:
                path.append(p)
                if p == a:
                    break
                p = parent[p]
            if path and path[-1] == a:
                best = np.asarray(path[::-1], float)
                best_len = dist[b]
    stats = {"endpoints": len(endpoints), "junctions": sum(d >= 3 for d in deg.values()), "nodes": len(deg)}
    return best, stats


def path_physical_length(path, sp_yx):
    if path is None or len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0) * np.asarray(sp_yx)[None, :]
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def endpoint_distance(path, sp_yx):
    d = (path[-1] - path[0]) * np.asarray(sp_yx)
    return float(np.sqrt((d * d).sum()))


def path_tube(path, shape, sp_yx, radius_mm=5.0):
    pm = np.zeros(shape, bool)
    yy = np.clip(np.rint(path[:, 0]).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.rint(path[:, 1]).astype(int), 0, shape[1] - 1)
    pm[yy, xx] = True
    dist = ndi.distance_transform_edt(~pm, sampling=sp_yx)
    return dist <= radius_mm, pm


def _sample(a, path):
    y = np.clip(np.rint(path[:, 0]).astype(int), 0, a.shape[0] - 1)
    x = np.clip(np.rint(path[:, 1]).astype(int), 0, a.shape[1] - 1)
    return np.asarray(a)[y, x]


def consensus_candidates(consensus_data, sp_yx, min_length_mm=20, max_length_mm=230, max_sinuosity=2.7, max_candidates=12):
    cons = np.asarray(consensus_data["consensus"], float)
    support = np.asarray(consensus_data["support"], float)
    vals = cons[cons > 0]
    if vals.size < 100:
        return []
    h, w = cons.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt(((yy - h / 2) / (0.5 * h)) ** 2 + ((xx - w / 2) / (0.5 * w)) ** 2)
    records, seen = [], set()

    for q in (94, 91, 88, 85, 82, 79, 76, 73):
        th = float(np.percentile(vals, q))
        cand = cons >= th
        cand = binary_closing(cand, disk(1))
        cand = remove_small_objects(cand, min_size=14)
        skel = skeletonize(cand)
        labs = label(skel, connectivity=2)
        for k in range(1, labs.max() + 1):
            comp = labs == k
            if comp.sum() < 14:
                continue
            path, stats = endpoint_diameter_path(comp)
            if path is None or len(path) < 12:
                continue
            length = path_physical_length(path, sp_yx)
            straight = endpoint_distance(path, sp_yx)
            sinuosity = length / max(straight, 1e-6)
            if not (min_length_mm <= length <= max_length_mm) or sinuosity > max_sinuosity:
                continue
            branch_fraction = stats["junctions"] / max(stats["nodes"], 1)
            if branch_fraction > 0.14:
                continue

            ep = np.r_[path[0], path[-1]]
            key = tuple(np.rint(ep / 5).astype(int))
            rev = tuple(np.rint(np.r_[path[-1], path[0]] / 5).astype(int))
            if key in seen or rev in seen:
                continue
            seen.add(key)

            mean_cons = float(_sample(cons, path).mean())
            mean_support = float(_sample(support, path).mean())
            mean_r = float(_sample(rr, path).mean())
            center_score = float(np.exp(-0.5 * (mean_r / 0.65) ** 2))
            directness = float(np.exp(-0.8 * max(0, sinuosity - 1)))
            length_quality = float(np.exp(-0.5 * ((length - 110) / 80) ** 2))
            score = 0.44 * mean_cons + 0.31 * mean_support + 0.12 * center_score + 0.08 * directness + 0.05 * length_quality
            records.append({
                "path": path,
                "threshold_percentile": q,
                "path_length_mm": length,
                "endpoint_distance_mm": straight,
                "sinuosity": float(sinuosity),
                "mean_consensus": mean_cons,
                "mean_support": mean_support,
                "mean_radius_from_center": mean_r,
                "center_score": center_score,
                "branch_fraction": float(branch_fraction),
                "consensus_score": float(score),
                "graph_endpoints": int(stats["endpoints"]),
                "graph_junctions": int(stats["junctions"]),
            })
    records.sort(key=lambda d: d["consensus_score"], reverse=True)
    return records[:max_candidates]


def frame_score(img, path, sp_yx):
    """Score how well one rotation displays a fixed consensus path."""
    img = np.asarray(img, float)
    vals = _sample(img, path)
    hu_ok = (vals >= 80) & (vals <= 950)
    plausible = float(hu_ok.mean())
    mean_hu = float(np.mean(vals[hu_ok])) if np.any(hu_ok) else float(np.mean(vals))

    tube2, pm = path_tube(path, img.shape, sp_yx, radius_mm=1.5)
    tube5, _ = path_tube(path, img.shape, sp_yx, radius_mm=5.0)
    ring = tube5 & ~tube2
    core_mean = float(np.mean(img[tube2])) if np.any(tube2) else mean_hu
    ring_mean = float(np.mean(img[ring])) if np.any(ring) else core_mean
    contrast = core_mean - ring_mean
    contrast_score = float(1 / (1 + np.exp(-(contrast - 40) / 90)))
    hu_score = float(np.exp(-0.5 * ((mean_hu - 500) / 320) ** 2))
    score = 0.50 * plausible + 0.30 * contrast_score + 0.20 * hu_score
    return {"frame_score": float(score), "mean_path_hu": mean_hu, "plausible_hu_fraction": plausible, "core_minus_ring_hu": float(contrast)}


def choose_best_rotation(volume, path, sp_yx):
    rows = []
    for z in range(np.asarray(volume).shape[0]):
        d = frame_score(volume[z], path, sp_yx)
        d["frame"] = int(z)
        rows.append(d)
    rows.sort(key=lambda d: d["frame_score"], reverse=True)
    return rows[0], rows


def qc_status(candidate, rotation):
    if candidate is None:
        return "FAIL", "no persistent open coronary-scale consensus path"
    fail, review = [], []
    if not (20 <= candidate["path_length_mm"] <= 230): fail.append("length")
    if candidate["sinuosity"] > 2.7: fail.append("sinuosity")
    if candidate["mean_support"] < 0.12: fail.append("low cross-rotation persistence")
    elif candidate["mean_support"] < 0.25: review.append("modest cross-rotation persistence")
    if candidate["mean_consensus"] < 0.22: fail.append("weak consensus evidence")
    elif candidate["mean_consensus"] < 0.38: review.append("moderate consensus evidence")
    if candidate["branch_fraction"] > 0.14: fail.append("branching")
    if rotation is not None:
        if rotation["plausible_hu_fraction"] < 0.55: fail.append("poor lumen HU support")
        elif rotation["plausible_hu_fraction"] < 0.75: review.append("partial lumen HU support")
    if fail:
        return "FAIL", ", ".join(fail + review)
    if review:
        return "REVIEW", ", ".join(review)
    return "OK", ""


def smooth_resample_path(path, sp_yx, step_mm=0.35):
    y = gaussian_filter1d(path[:, 0].astype(float), 1.2, mode="nearest")
    x = gaussian_filter1d(path[:, 1].astype(float), 1.2, mode="nearest")
    p = np.column_stack([y, x])
    mm = p * np.asarray(sp_yx)[None, :]
    ds = np.sqrt(((np.diff(mm, axis=0)) ** 2).sum(axis=1))
    s = np.r_[0, np.cumsum(ds)]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, mm = s[keep], mm[keep]
    if len(s) < 3 or s[-1] < 1:
        return None
    su = np.arange(0, s[-1] + 1e-9, step_mm)
    return su, np.interp(su, s, mm[:, 0]), np.interp(su, s, mm[:, 1])


def straighten(img, plaque, path, sp_yx, half_width_mm=6.0, along_step=0.35, cross_step=0.25):
    rr = smooth_resample_path(path, sp_yx, along_step)
    if rr is None:
        return None
    s, ym, xm = rr
    ym, xm = gaussian_filter1d(ym, 1.0), gaussian_filter1d(xm, 1.0)
    dy, dx = np.gradient(ym), np.gradient(xm)
    n = np.hypot(dy, dx); n[n < 1e-6] = 1
    ny, nx = -(dx / n), dy / n
    off = np.arange(-half_width_mm, half_width_mm + 1e-9, cross_step)
    y_mm = ym[None, :] + off[:, None] * ny[None, :]
    x_mm = xm[None, :] + off[:, None] * nx[None, :]
    y_px, x_px = y_mm / float(sp_yx[0]), x_mm / float(sp_yx[1])
    strip = map_coordinates(np.asarray(img, float), [y_px, x_px], order=1, mode="nearest")
    pstrip = map_coordinates(np.asarray(plaque, float), [y_px, x_px], order=0, mode="constant", cval=0) > 0.5
    return s, off, strip, pstrip
