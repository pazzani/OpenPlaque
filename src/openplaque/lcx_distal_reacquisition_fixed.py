from __future__ import annotations

"""Runtime bugfix layer for LCX distal reacquisition.

The v1.0 scientific implementation stores Dijkstra distances as a flat vector, but
reachable-node discovery treated np.nonzero(dist) as if it returned (z,y,x) indices.
This layer reshapes only the returned distance field to the ROI shape before the
existing extension code consumes it. Scientific thresholds, costs, controls, and
identity rules are unchanged.
"""

import numpy as np

from . import lcx_distal_reacquisition as _base

BASELINE = _base.BASELINE
ALGORITHM = _base.ALGORITHM
PATCH_VERSION = "dijkstra-distance-shape-v1"

_ORIGINAL_DIJKSTRA = _base._dijkstra


def _dijkstra_reshaped(mask, cost, start, spacing, goal_mask=None, max_cost=180.0):
    dist, prev, reached = _ORIGINAL_DIJKSTRA(
        mask, cost, start, spacing, goal_mask=goal_mask, max_cost=max_cost
    )
    dist = np.asarray(dist)
    if dist.ndim == 1:
        expected = int(np.prod(mask.shape))
        if dist.size != expected:
            raise RuntimeError(
                f"Unexpected Dijkstra distance size {dist.size}; expected {expected} for shape {mask.shape}"
            )
        dist = dist.reshape(mask.shape)
    elif dist.shape != mask.shape:
        raise RuntimeError(
            f"Unexpected Dijkstra distance shape {dist.shape}; expected {mask.shape}"
        )
    return dist, prev, reached


def reachable_coordinates(dist, shape):
    """Return local (z,y,x) coordinates from a flat or shaped distance field."""
    d = np.asarray(dist)
    if d.ndim == 1:
        if d.size != int(np.prod(shape)):
            raise ValueError("Flat distance field size does not match ROI shape")
        d = d.reshape(shape)
    if tuple(d.shape) != tuple(shape):
        raise ValueError("Distance field shape does not match ROI shape")
    coords = np.column_stack(np.nonzero(np.isfinite(d)))
    return coords.astype(int, copy=False)


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    old = _base._dijkstra
    _base._dijkstra = _dijkstra_reshaped
    try:
        return _base.run(drive_root, output_dir)
    finally:
        _base._dijkstra = old


def synthetic_reacquisition_self_test():
    return _base.synthetic_reacquisition_self_test()
