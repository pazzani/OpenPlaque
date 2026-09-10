from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi


@dataclass
class InformativeSlice:
    z: int
    score: float
    y: int
    x: int
    hu: float
    vesselness: float
    local_radius_mm: float


def _central_bounds(shape_yx, fraction=0.78):
    h, w = (int(shape_yx[0]), int(shape_yx[1]))
    fy = float(fraction)
    fx = float(fraction)
    hh = int(round(h * fy / 2))
    hw = int(round(w * fx / 2))
    cy, cx = h // 2, w // 2
    return max(0, cy-hh), min(h, cy+hh), max(0, cx-hw), min(w, cx+hw)


def _slice_feature_map(image_hu, pixel_spacing_yx_mm):
    """Return a 2-D score favoring small, contrast-filled tubular structures."""
    im = np.asarray(image_hu, dtype=np.float32)
    sy, sx = [float(v) for v in pixel_spacing_yx_mm]
    smooth = ndi.gaussian_filter(im, sigma=0.8, mode="nearest")

    norm = np.clip((smooth + 200.0) / 1000.0, 0.0, 1.5)
    vessel = frangi(norm, sigmas=(1.0, 1.6, 2.3, 3.2), black_ridges=False)
    vessel = np.nan_to_num(vessel, nan=0.0, posinf=0.0, neginf=0.0)
    pos = vessel[vessel > 0]
    if pos.size:
        scale = np.percentile(pos, 99.5)
        if scale > 0:
            vessel = np.clip(vessel / scale, 0.0, 1.0)

    contrast = np.clip((smooth - 140.0) / 260.0, 0.0, 1.0)

    blood = smooth >= 150.0
    radius_mm = ndi.distance_transform_edt(blood, sampling=(sy, sx))
    small = np.exp(-0.5 * (radius_mm / 3.0) ** 2)

    score = vessel * (0.35 + 0.65 * contrast) * (0.25 + 0.75 * small)
    score[smooth < 100.0] = 0.0

    return score, vessel, radius_mm


def _diverse_top(entries, n=12, min_separation_slices=5):
    selected = []
    for item in sorted(entries, key=lambda e: e.score, reverse=True):
        if all(abs(item.z - prev.z) >= int(min_separation_slices) for prev in selected):
            selected.append(item)
            if len(selected) >= int(n):
                break
    return selected


def find_informative_slices(
    volume_hu,
    spacing_xyz_mm,
    *,
    z_fraction=(0.42, 0.82),
    n=12,
    step=2,
    min_separation_mm=2.5,
):
    """Find diverse axial slices with strong small-vessel signal.

    Returns InformativeSlice objects. This is a visual triage helper, not an
    anatomical RCA classifier.
    """
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("volume_hu must be 3-D")
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=float)
    if spacing_xyz.size != 3 or np.any(spacing_xyz <= 0):
        raise ValueError("spacing_xyz_mm must contain three positive values")

    z0 = max(0, int(round(vol.shape[0] * float(z_fraction[0]))))
    z1 = min(vol.shape[0], int(round(vol.shape[0] * float(z_fraction[1]))))
    if z1 <= z0:
        raise ValueError("invalid z_fraction")

    y0, y1, x0, x1 = _central_bounds(vol.shape[1:])
    sy, sx = spacing_xyz[1], spacing_xyz[0]
    entries = []

    for z in range(z0, z1, max(1, int(step))):
        crop = vol[z, y0:y1, x0:x1]
        score_map, vessel, radius = _slice_feature_map(crop, (sy, sx))

        yy, xx = np.indices(score_map.shape)
        cy, cx = score_map.shape[0] / 2.0, score_map.shape[1] / 2.0
        center_r_mm = np.sqrt(((yy-cy)*sy)**2 + ((xx-cx)*sx)**2)
        score_map = score_map.copy()
        score_map[center_r_mm < 12.0] *= 0.45

        if not np.any(score_map > 0):
            continue
        iy, ix = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        flat = score_map.ravel()
        k = min(30, flat.size)
        top = np.partition(flat, flat.size-k)[-k:]
        score = float(np.mean(top))
        entries.append(
            InformativeSlice(
                z=int(z),
                score=score,
                y=int(iy+y0),
                x=int(ix+x0),
                hu=float(vol[z, iy+y0, ix+x0]),
                vesselness=float(vessel[iy, ix]),
                local_radius_mm=float(radius[iy, ix]),
            )
        )

    sep_slices = max(1, int(round(float(min_separation_mm) / spacing_xyz[2])))
    return _diverse_top(entries, n=n, min_separation_slices=sep_slices)


def find_root_window(volume_hu, spacing_xyz_mm, *, z_fraction=(0.45, 0.78)):
    """Estimate a broad aortic-root window for visual triage."""
    vol = np.asarray(volume_hu, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=float)
    z0 = max(0, int(round(vol.shape[0] * float(z_fraction[0]))))
    z1 = min(vol.shape[0], int(round(vol.shape[0] * float(z_fraction[1]))))
    y0, y1, x0, x1 = _central_bounds(vol.shape[1:], fraction=0.62)

    metrics = []
    for z in range(z0, z1):
        im = ndi.gaussian_filter(vol[z, y0:y1, x0:x1], sigma=1.0)
        mask = im >= 250.0
        labels, count = ndi.label(mask)
        if count == 0:
            metrics.append((z, 0.0))
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = np.sort(sizes)[-4:]
        compact = largest[(largest >= 300) & (largest <= 12000)]
        metric = float(compact.sum()) if compact.size else 0.0
        metrics.append((z, metric))

    if not metrics:
        return z0, z1
    vals = np.asarray([m for _, m in metrics], dtype=float)
    if np.max(vals) <= 0:
        return z0, z1

    peak_index = int(np.argmax(ndi.gaussian_filter1d(vals, sigma=4.0)))
    peak_z = int(metrics[peak_index][0])
    half_width_mm = 18.0
    dz = max(12, int(round(half_width_mm / spacing_xyz[2])))
    return max(z0, peak_z-dz), min(z1, peak_z+dz+1)


def find_root_informative_slices(
    volume_hu,
    spacing_xyz_mm,
    *,
    n=12,
    step=1,
    min_separation_mm=1.5,
):
    vol = np.asarray(volume_hu, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=float)
    z0, z1 = find_root_window(vol, spacing_xyz)
    frac = (z0 / vol.shape[0], z1 / vol.shape[0])
    return find_informative_slices(
        vol,
        spacing_xyz,
        z_fraction=frac,
        n=n,
        step=step,
        min_separation_mm=min_separation_mm,
    ), (z0, z1)
