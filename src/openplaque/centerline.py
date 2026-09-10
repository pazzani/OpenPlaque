"""Experimental coronary centerline utilities for OpenPlaque.

The initial target is an isolated RCA mask. The caller supplies an ostium point
and may optionally supply a distal-RCA hint. The implementation skeletonizes the
3-D vessel mask, builds a 26-connected graph in physical space, and extracts a
single path from the ostium to either the hinted distal point or the farthest
reachable skeleton point.

Research use only. Not clinically validated. Not for diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass
class CenterlineResult:
    points_zyx_voxel: np.ndarray
    points_xyz_mm: np.ndarray
    cumulative_length_mm: np.ndarray
    ostium_zyx_voxel: tuple[int, int, int]
    endpoint_zyx_voxel: tuple[int, int, int]
    skeleton_mask: np.ndarray
    landmarks_xyz_mm: dict[float, np.ndarray]
    parameters: dict

    @property
    def length_mm(self) -> float:
        return float(self.cumulative_length_mm[-1]) if len(self.cumulative_length_mm) else 0.0


def _spacing_zyx(spacing_xyz_mm: Iterable[float]) -> np.ndarray:
    spacing = np.asarray(tuple(float(v) for v in spacing_xyz_mm), dtype=float)
    if spacing.shape != (3,):
        raise ValueError("spacing_xyz_mm must contain exactly three values")
    if np.any(spacing <= 0):
        raise ValueError("spacing_xyz_mm values must be positive")
    return spacing[::-1]


def _nearest_true(mask: np.ndarray, point_zyx: Iterable[float]) -> tuple[int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("mask contains no foreground voxels")
    p = np.asarray(tuple(float(v) for v in point_zyx), dtype=float)
    if p.shape != (3,):
        raise ValueError("point must contain exactly three z,y,x coordinates")
    coords = np.argwhere(mask)
    idx = int(np.argmin(np.sum((coords - p[None, :]) ** 2, axis=1)))
    return tuple(int(v) for v in coords[idx])


def _neighbor_offsets() -> list[tuple[int, int, int]]:
    out = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                out.append((dz, dy, dx))
    return out


def _build_graph(skeleton: np.ndarray, spacing_xyz_mm):
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        raise ValueError("skeleton is empty")
    node_of = {tuple(int(v) for v in c): i for i, c in enumerate(coords)}
    spacing_zyx = _spacing_zyx(spacing_xyz_mm)
    offsets = _neighbor_offsets()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]

    for i, c in enumerate(coords):
        z, y, x = (int(v) for v in c)
        for dz, dy, dx in offsets:
            q = (z + dz, y + dy, x + dx)
            j = node_of.get(q)
            if j is None or j <= i:
                continue
            delta_mm = np.asarray((dz, dy, dx), dtype=float) * spacing_zyx
            w = float(np.linalg.norm(delta_mm))
            adjacency[i].append((j, w))
            adjacency[j].append((i, w))
    return coords, node_of, adjacency


def _dijkstra(adjacency, start: int):
    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=float)
    prev = np.full(n, -1, dtype=int)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def _reconstruct(prev: np.ndarray, start: int, end: int) -> np.ndarray:
    path = [int(end)]
    u = int(end)
    while u != start:
        u = int(prev[u])
        if u < 0:
            raise ValueError("endpoint is not connected to ostium on skeleton")
        path.append(u)
    path.reverse()
    return np.asarray(path, dtype=int)


def _points_xyz_mm(points_zyx: np.ndarray, spacing_xyz_mm) -> np.ndarray:
    spacing_xyz = np.asarray(tuple(float(v) for v in spacing_xyz_mm), dtype=float)
    return points_zyx[:, ::-1].astype(float) * spacing_xyz[None, :]


def cumulative_length(points_xyz_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def point_at_distance(points_xyz_mm: np.ndarray, cumulative_mm: np.ndarray, distance_mm: float) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float)
    cum = np.asarray(cumulative_mm, dtype=float)
    d = float(distance_mm)
    if len(points) == 0:
        raise ValueError("centerline has no points")
    if d < 0:
        raise ValueError("distance_mm must be non-negative")
    if d > cum[-1] + 1e-9:
        raise ValueError(f"requested {d:.1f} mm landmark exceeds centerline length {cum[-1]:.1f} mm")
    if d <= 0:
        return points[0].copy()
    i = int(np.searchsorted(cum, d, side="right"))
    if i >= len(points):
        return points[-1].copy()
    d0, d1 = cum[i - 1], cum[i]
    if d1 <= d0:
        return points[i].copy()
    a = (d - d0) / (d1 - d0)
    return (1.0 - a) * points[i - 1] + a * points[i]


def extract_rca_centerline(
    artery_mask: np.ndarray,
    spacing_xyz_mm,
    ostium_zyx,
    *,
    distal_hint_zyx: Optional[Iterable[float]] = None,
    landmark_distances_mm=(0.0, 10.0, 50.0),
    min_component_voxels: int = 10,
) -> CenterlineResult:
    """Extract a provisional proximal-to-distal RCA centerline.

    Parameters
    ----------
    artery_mask:
        Boolean 3-D mask for an isolated RCA (lumen/vessel foreground).
    spacing_xyz_mm:
        Voxel spacing in SimpleITK order (x, y, z).
    ostium_zyx:
        Approximate RCA ostium in NumPy voxel coordinates (z, y, x). It is
        snapped to the nearest skeleton voxel.
    distal_hint_zyx:
        Optional approximate distal RCA point. If omitted, the algorithm chooses
        the skeleton voxel with the largest geodesic distance from the ostium.
    landmark_distances_mm:
        Arc-length positions to interpolate along the path. Landmarks beyond the
        extracted path length are omitted rather than extrapolated.

    Notes
    -----
    This function intentionally does not perform automatic artery labeling. A
    distal hint is recommended when the RCA mask contains substantial branches.
    """
    mask = np.asarray(artery_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("artery_mask must be a 3-D array")
    if int(np.sum(mask)) < int(min_component_voxels):
        raise ValueError("artery_mask is too small for centerline extraction")

    # Keep the connected component containing the voxel nearest the supplied ostium.
    seed = _nearest_true(mask, ostium_zyx)
    labels, _ = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
    lab = int(labels[seed])
    component = labels == lab

    # scikit-image uses Lee's 3-D thinning when skeletonize is applied to 3-D input.
    skeleton = skeletonize(component).astype(bool)
    coords, node_of, adjacency = _build_graph(skeleton, spacing_xyz_mm)

    ostium_skel = _nearest_true(skeleton, ostium_zyx)
    start = node_of[ostium_skel]
    dist, prev = _dijkstra(adjacency, start)

    if distal_hint_zyx is not None:
        endpoint_skel = _nearest_true(skeleton, distal_hint_zyx)
        end = node_of[endpoint_skel]
        if not np.isfinite(dist[end]):
            raise ValueError("distal hint lies on a skeleton component disconnected from the ostium")
    else:
        reachable = np.where(np.isfinite(dist))[0]
        if len(reachable) == 0:
            raise ValueError("no skeleton point is reachable from the ostium")
        end = int(reachable[np.argmax(dist[reachable])])
        endpoint_skel = tuple(int(v) for v in coords[end])

    path_nodes = _reconstruct(prev, start, end)
    points_zyx = coords[path_nodes].astype(float)
    points_xyz = _points_xyz_mm(points_zyx, spacing_xyz_mm)
    cum = cumulative_length(points_xyz)

    landmarks = {}
    for d in landmark_distances_mm:
        d = float(d)
        if 0.0 <= d <= cum[-1] + 1e-9:
            landmarks[d] = point_at_distance(points_xyz, cum, d)

    return CenterlineResult(
        points_zyx_voxel=points_zyx,
        points_xyz_mm=points_xyz,
        cumulative_length_mm=cum,
        ostium_zyx_voxel=tuple(int(v) for v in ostium_skel),
        endpoint_zyx_voxel=tuple(int(v) for v in endpoint_skel),
        skeleton_mask=skeleton,
        landmarks_xyz_mm=landmarks,
        parameters={
            "distal_hint_supplied": distal_hint_zyx is not None,
            "landmark_distances_mm": tuple(float(d) for d in landmark_distances_mm),
            "min_component_voxels": int(min_component_voxels),
        },
    )


def show_centerline_mip(volume, result: CenterlineResult, *, axis=0, vmin=-200, vmax=800):
    """Show a simple CT maximum-intensity projection with centerline landmarks."""
    import matplotlib.pyplot as plt

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError("volume must be 3-D")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    mip = np.max(vol, axis=axis)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)

    pts = result.points_zyx_voxel
    if axis == 0:      # image axes y,x
        yy, xx = pts[:, 1], pts[:, 2]
    elif axis == 1:    # image axes z,x
        yy, xx = pts[:, 0], pts[:, 2]
    else:              # image axes z,y
        yy, xx = pts[:, 0], pts[:, 1]
    ax.plot(xx, yy, linewidth=1.5, label="RCA centerline")

    spacing_xyz = np.asarray(result.points_xyz_mm[0] * 0 + 1.0)
    # Landmark voxel positions are obtained by interpolation in voxel coordinates
    # using the same cumulative arc-length parameter as the physical path.
    for d, _p_xyz in sorted(result.landmarks_xyz_mm.items()):
        p = np.array([
            np.interp(d, result.cumulative_length_mm, pts[:, 0]),
            np.interp(d, result.cumulative_length_mm, pts[:, 1]),
            np.interp(d, result.cumulative_length_mm, pts[:, 2]),
        ])
        if axis == 0:
            py, px = p[1], p[2]
        elif axis == 1:
            py, px = p[0], p[2]
        else:
            py, px = p[0], p[1]
        ax.scatter([px], [py], s=45)
        ax.text(px + 2, py + 2, f"{d:g} mm")

    ax.legend(loc="upper right")
    ax.set_title(f"Experimental RCA centerline; length {result.length_mm:.1f} mm")
    ax.axis("off")
    return fig, ax
