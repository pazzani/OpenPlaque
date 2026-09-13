"""Image-driven 2-D coronary tracking utilities for Siemens CPR visualization.

These functions are intentionally presentation/QC utilities. They do not change
OpenPlaque canonical TPV and do not establish source-volume co-registration.
"""

from collections import deque
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d, map_coordinates
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


def robust01(a, mask=None):
    a = np.asarray(a, float)
    vals = a[mask] if mask is not None and np.any(mask) else a.ravel()
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        return np.zeros_like(a, float)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _bfs_farthest(coord_set, start):
    q = deque([start])
    dist = {start: 0}
    parent = {start: None}
    far = start
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        p = q.popleft()
        if dist[p] > dist[far]:
            far = p
        for dy, dx in nbr:
            z = (p[0] + dy, p[1] + dx)
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


def image_evidence(img):
    """Return image-driven bright-tube evidence and a valid interior/body mask."""
    img = np.asarray(img, float)
    body = largest_body_mask(img)
    h, w = img.shape
    interior = np.zeros_like(body, bool)
    my = max(8, int(0.06 * h))
    mx = max(8, int(0.06 * w))
    interior[my:h-my, mx:w-mx] = True
    valid = body & interior

    x = np.clip((img - 50) / 650, 0, 1)
    vesselness = frangi(x, sigmas=(1, 2, 3, 4), black_ridges=False)
    vesselness = robust01(vesselness, valid)

    intensity = np.exp(-0.5 * ((img - 350.0) / 230.0) ** 2)
    intensity[(img < 80) | (img > 850)] = 0
    evidence = (0.78 * vesselness + 0.22 * intensity) * valid
    return evidence, valid


def track_frame(img, sp_yx):
    """Find the best continuous bright tubular path in one CPR frame.

    The nnU-Net vessel mask is deliberately not used here.
    """
    img = np.asarray(img, float)
    evidence, valid = image_evidence(img)
    vals = evidence[valid]
    if vals.size < 100:
        return None

    best = None
    for q in (92, 89, 86, 83, 80):
        threshold = np.percentile(vals, q)
        candidate = (evidence >= threshold) & valid
        candidate = binary_closing(candidate, disk(2))
        candidate = remove_small_objects(candidate, min_size=18)
        skel = skeletonize(candidate)
        labs = label(skel, connectivity=2)

        for k in range(1, labs.max() + 1):
            comp = labs == k
            if comp.sum() < 15:
                continue
            path = diameter_path(comp)
            if path is None or len(path) < 12:
                continue

            iy = np.clip(np.rint(path[:, 0]).astype(int), 0, img.shape[0] - 1)
            ix = np.clip(np.rint(path[:, 1]).astype(int), 0, img.shape[1] - 1)
            length = path_physical_length(path, sp_yx)
            mean_ev = float(evidence[iy, ix].mean())
            mean_hu = float(img[iy, ix].mean())
            intensity_fit = float(np.exp(-0.5 * ((img[iy, ix] - 350) / 230) ** 2).mean())
            cy, cx = path.mean(axis=0)
            center_dist = np.hypot(
                (cy - img.shape[0] / 2) / img.shape[0],
                (cx - img.shape[1] / 2) / img.shape[1],
            )
            center_factor = float(np.exp(-0.5 * (center_dist / 0.33) ** 2))
            score = length * (0.25 + mean_ev) * (0.45 + 0.55 * intensity_fit) * (0.65 + 0.35 * center_factor)

            rec = {
                "path": path,
                "evidence": evidence,
                "valid": valid,
                "image_score": float(score),
                "path_length_mm": length,
                "mean_path_hu": mean_hu,
                "mean_evidence": mean_ev,
                "intensity_fit": intensity_fit,
                "threshold_percentile": q,
            }
            if best is None or rec["image_score"] > best["image_score"]:
                best = rec
    return best


def path_tube(path, shape, sp_yx, radius_mm=5.0):
    path_mask = np.zeros(shape, bool)
    yy = np.clip(np.rint(path[:, 0]).astype(int), 0, shape[0] - 1)
    xx = np.clip(np.rint(path[:, 1]).astype(int), 0, shape[1] - 1)
    path_mask[yy, xx] = True
    dist = ndi.distance_transform_edt(~path_mask, sampling=sp_yx)
    return dist <= radius_mm, path_mask


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
    """Straighten one 2-D CPR frame around an image-driven path."""
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
