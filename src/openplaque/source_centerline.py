from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.graph import route_through_array


@dataclass
class SourceCenterlineResult:
    points_zyx_voxel: np.ndarray
    points_xyz_mm: np.ndarray
    cumulative_length_mm: np.ndarray
    landmarks_xyz_mm: dict[float, np.ndarray]
    start_zyx_voxel: tuple[int, int, int]
    end_zyx_voxel: tuple[int, int, int]
    crop_bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    target_spacing_mm: float
    parameters: dict

    @property
    def length_mm(self) -> float:
        return float(self.cumulative_length_mm[-1]) if len(self.cumulative_length_mm) else 0.0


def _index_zyx_to_physical_xyz(points_zyx: np.ndarray, image) -> np.ndarray:
    pts = np.asarray(points_zyx, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_zyx must have shape (N,3)")
    out = np.empty((len(pts), 3), dtype=float)
    for i, (z, y, x) in enumerate(pts):
        out[i] = image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
    return out


def cumulative_length(points_xyz_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz_mm, dtype=float)
    if len(pts) == 0:
        return np.array([], dtype=float)
    if len(pts) == 1:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def point_at_distance(points_xyz_mm: np.ndarray, cumulative_mm: np.ndarray, distance_mm: float) -> np.ndarray:
    pts = np.asarray(points_xyz_mm, dtype=float)
    cum = np.asarray(cumulative_mm, dtype=float)
    if len(pts) == 0:
        raise ValueError("empty centerline")
    if distance_mm < 0 or distance_mm > cum[-1]:
        raise ValueError("distance outside centerline")
    j = int(np.searchsorted(cum, distance_mm, side="left"))
    if j == 0:
        return pts[0].copy()
    if j >= len(pts):
        return pts[-1].copy()
    lo, hi = cum[j - 1], cum[j]
    if hi <= lo:
        return pts[j].copy()
    t = (distance_mm - lo) / (hi - lo)
    return pts[j - 1] * (1 - t) + pts[j] * t


def _crop_from_seeds(shape, start_zyx, end_zyx, spacing_xyz_mm, margin_mm):
    spacing_zyx = np.asarray(spacing_xyz_mm, dtype=float)[::-1]
    a = np.asarray(start_zyx, dtype=int)
    b = np.asarray(end_zyx, dtype=int)
    margin_vox = np.ceil(float(margin_mm) / spacing_zyx).astype(int)
    lo = np.maximum(0, np.minimum(a, b) - margin_vox)
    hi = np.minimum(np.asarray(shape), np.maximum(a, b) + margin_vox + 1)
    return tuple((int(lo[d]), int(hi[d])) for d in range(3))


def _snap_to_candidate(point_zyx, candidate, max_distance_vox=8):
    p = np.asarray(point_zyx, dtype=int)
    p = np.clip(p, 0, np.asarray(candidate.shape) - 1)
    if candidate[tuple(p)]:
        return tuple(int(v) for v in p)
    coords = np.argwhere(candidate)
    if len(coords) == 0:
        raise ValueError("no candidate vessel voxels in crop")
    d2 = np.sum((coords - p) ** 2, axis=1)
    q = coords[int(np.argmin(d2))]
    if float(np.sqrt(np.min(d2))) > float(max_distance_vox):
        raise ValueError("seed is too far from any vessel candidate")
    return tuple(int(v) for v in q)


def _normalize_positive(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    pos = a[a > 0]
    if pos.size == 0:
        return np.zeros_like(a, dtype=float)
    scale = float(np.percentile(pos, percentile))
    if scale <= 0:
        return np.zeros_like(a, dtype=float)
    return np.clip(a / scale, 0.0, 1.0)


def _resample_path(points_xyz_mm: np.ndarray, step_mm: float = 0.5, smooth_sigma_mm: float = 0.8) -> np.ndarray:
    pts = np.asarray(points_xyz_mm, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    cum = cumulative_length(pts)
    keep = np.concatenate(([True], np.diff(cum) > 1e-6))
    pts = pts[keep]
    cum = cumulative_length(pts)
    if len(pts) < 2 or cum[-1] <= 0:
        return pts
    med = float(np.median(np.diff(cum)))
    if smooth_sigma_mm > 0 and med > 0 and len(pts) >= 5:
        sigma = max(0.5, float(smooth_sigma_mm) / med)
        pts = np.column_stack([
            ndi.gaussian_filter1d(pts[:, j], sigma=sigma, mode="nearest")
            for j in range(3)
        ])
        cum = cumulative_length(pts)
    sample = np.arange(0.0, cum[-1], float(step_mm))
    if len(sample) == 0 or sample[-1] < cum[-1]:
        sample = np.append(sample, cum[-1])
    return np.column_stack([np.interp(sample, cum, pts[:, j]) for j in range(3)])


def trace_seeded_coronary(
    volume_hu: np.ndarray,
    image,
    start_zyx: Sequence[int],
    end_zyx: Sequence[int],
    *,
    crop_margin_mm: float = 25.0,
    target_spacing_mm: float = 0.7,
    min_hu: float = 120.0,
    soft_hu: float = 180.0,
    frangi_sigmas_vox: Iterable[float] = (1.0, 1.5, 2.0, 2.5),
    landmark_distances_mm: Sequence[float] = (0.0, 10.0, 50.0),
    resample_step_mm: float = 0.5,
    smooth_sigma_mm: float = 0.8,
) -> SourceCenterlineResult:
    """Trace a contrast-filled coronary between two source-CCTA seed points.

    The path is computed only inside a seed-defined crop. CT attenuation and
    3-D Frangi vesselness are combined into a weighted shortest-path cost.
    This is a research prototype, not an artery segmentation model, and every
    result must be visually validated before downstream quantitative use.
    """
    vol = np.asarray(volume_hu, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("volume_hu must be 3-D")
    if tuple(image.GetSize())[::-1] != tuple(vol.shape):
        raise ValueError("SimpleITK image geometry does not match NumPy volume")

    spacing_xyz = np.asarray(image.GetSpacing(), dtype=float)
    spacing_zyx = spacing_xyz[::-1]
    bounds = _crop_from_seeds(vol.shape, start_zyx, end_zyx, spacing_xyz, crop_margin_mm)
    slc = tuple(slice(a, b) for a, b in bounds)
    crop = vol[slc]
    offset = np.asarray([a for a, _ in bounds], dtype=float)

    zoom = spacing_zyx / float(target_spacing_mm)
    zoom = np.minimum(1.0, zoom)
    work = ndi.zoom(crop, zoom=zoom, order=1, mode="nearest", prefilter=False)
    if min(work.shape) < 5:
        raise ValueError("seed crop is too small after resampling")

    local_start = np.rint((np.asarray(start_zyx, dtype=float) - offset) * zoom).astype(int)
    local_end = np.rint((np.asarray(end_zyx, dtype=float) - offset) * zoom).astype(int)

    sm = ndi.gaussian_filter(work, sigma=0.6, mode="nearest")
    vesselness = frangi(
        sm,
        sigmas=tuple(float(s) for s in frangi_sigmas_vox),
        black_ridges=False,
    )
    vesselness = _normalize_positive(vesselness)
    intensity = np.clip((sm - float(min_hu)) / max(1.0, 650.0 - float(min_hu)), 0.0, 1.0)

    candidate = sm >= float(min_hu)
    start = _snap_to_candidate(local_start, candidate)
    end = _snap_to_candidate(local_end, candidate)

    support = 0.55 * intensity + 0.45 * vesselness
    cost = 1.0 + 18.0 * (1.0 - support)
    low = sm < float(soft_hu)
    cost[low] += np.clip((float(soft_hu) - sm[low]) / 20.0, 0.0, 20.0)
    cost[~candidate] += 120.0
    cost[~np.isfinite(cost)] = 1000.0

    route, _ = route_through_array(cost, start, end, fully_connected=True, geometric=True)
    route = np.asarray(route, dtype=float)

    original_zyx = route / zoom + offset
    raw_xyz = _index_zyx_to_physical_xyz(original_zyx, image)
    smooth_xyz = _resample_path(raw_xyz, step_mm=resample_step_mm, smooth_sigma_mm=smooth_sigma_mm)
    cum = cumulative_length(smooth_xyz)

    landmarks = {}
    for d in landmark_distances_mm:
        d = float(d)
        if len(cum) and d <= cum[-1]:
            landmarks[d] = point_at_distance(smooth_xyz, cum, d)

    return SourceCenterlineResult(
        points_zyx_voxel=original_zyx,
        points_xyz_mm=smooth_xyz,
        cumulative_length_mm=cum,
        landmarks_xyz_mm=landmarks,
        start_zyx_voxel=tuple(int(v) for v in start_zyx),
        end_zyx_voxel=tuple(int(v) for v in end_zyx),
        crop_bounds_zyx=bounds,
        target_spacing_mm=float(target_spacing_mm),
        parameters={
            "crop_margin_mm": float(crop_margin_mm),
            "target_spacing_mm": float(target_spacing_mm),
            "min_hu": float(min_hu),
            "soft_hu": float(soft_hu),
            "frangi_sigmas_vox": tuple(float(s) for s in frangi_sigmas_vox),
            "resample_step_mm": float(resample_step_mm),
            "smooth_sigma_mm": float(smooth_sigma_mm),
        },
    )
