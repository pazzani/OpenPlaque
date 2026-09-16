from __future__ import annotations

"""Robust direction-estimation patch for ostium mask-consensus v1.0.

Scientific candidate discovery and acceptance gates are unchanged. This patch only makes the
local vessel-axis estimator well-defined for short/thin mask segments and removes the arbitrary
Y-axis fallback used by v1.0 when fewer than 12 nearby mask voxels were present.
"""

import numpy as np

from . import left_coronary_ostium_mask_consensus as _base

BASELINE = _base.BASELINE
ALGORITHM = "left-coronary-ostium-mask-consensus-v1.1-robust-direction"
OUTPUT_DIRNAME = "Left_Coronary_Ostium_Mask_Consensus_v1_1"

# Re-export the functions used by the experiment tests and notebook.
_score_plane = _base._score_plane
_root_interface_clusters = _base._root_interface_clusters
synthetic_interface_cluster_self_test = _base.synthetic_interface_cluster_self_test


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _sample_outside(outside_dist, point):
    return float(_base._sample_arr(outside_dist, [point], order=1, cval=0.0)[0])


def _outside_gradient(seed, outside_dist, spacing, step_mm=0.8):
    """Finite-difference gradient in physical z/y/x coordinates."""
    seed = np.asarray(seed, float)
    sp = np.asarray(spacing, float)
    g = np.zeros(3, float)
    for axis in range(3):
        dv = np.zeros(3, float)
        dv[axis] = step_mm / max(sp[axis], 1e-9)
        g[axis] = (_sample_outside(outside_dist, seed + dv) -
                   _sample_outside(outside_dist, seed - dv)) / (2.0 * step_mm)
    return g


def _local_direction(seed, union, outside_dist, spacing, radius_mm=6.0):
    """Estimate the local coronary axis and orient it away from the aortic surface.

    v1.0 required >=12 mask voxels before even attempting PCA, which caused a valid thin
    straight synthetic vessel to fall through to an arbitrary [0,1,0] direction. Here PCA is
    allowed with >=3 nondegenerate physical points and >=1 mm spatial span. If that is not
    identifiable, the local extra-aortic-distance gradient is used as the fallback.
    """
    seed = np.asarray(seed, float)
    sp = np.asarray(spacing, float)
    rad = np.ceil(radius_mm / sp).astype(int)
    lo = np.maximum(np.floor(seed).astype(int) - rad, 0)
    hi = np.minimum(np.floor(seed).astype(int) + rad + 1, np.asarray(union.shape))
    sl = tuple(slice(lo[i], hi[i]) for i in range(3))
    pts = np.argwhere(union[sl]) + lo[None, :]

    dmm = (pts - seed[None, :]) * sp[None, :] if len(pts) else np.empty((0, 3), float)
    if len(dmm):
        keep = np.linalg.norm(dmm, axis=1) <= radius_mm + 1e-9
        dmm = dmm[keep]

    t = None
    if len(dmm) >= 3:
        span_mm = float(np.linalg.norm(np.ptp(dmm, axis=0)))
        if span_mm >= 1.0:
            centered = dmm - np.mean(dmm, axis=0, keepdims=True)
            C = centered.T @ centered / max(len(centered) - 1, 1)
            vals, vecs = np.linalg.eigh(C)
            if float(vals[-1]) > 1e-6:
                t = _unit(vecs[:, -1])

    grad = _outside_gradient(seed, outside_dist, sp)
    if t is None or np.linalg.norm(t) < 1e-9:
        t = _unit(grad)
    if np.linalg.norm(t) < 1e-9:
        # Last-resort deterministic axis only for completely uninformative synthetic/degenerate
        # neighborhoods. Production candidates should not reach this case.
        t = np.array([0.0, 1.0, 0.0])

    step = 0.8
    pplus = seed + step * t / sp
    pminus = seed - step * t / sp
    dplus = _sample_outside(outside_dist, pplus)
    dminus = _sample_outside(outside_dist, pminus)
    if dminus > dplus + 1e-9:
        t = -t
    elif abs(dplus - dminus) <= 1e-9 and np.linalg.norm(grad) > 1e-9 and np.dot(t, grad) < 0:
        t = -t
    return _unit(t)


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    """Run v1.0 science with only the robust direction estimator patched in."""
    old_direction = _base._local_direction
    old_algorithm = _base.ALGORITHM
    old_output = _base.OUTPUT_DIRNAME
    _base._local_direction = _local_direction
    _base.ALGORITHM = ALGORITHM
    _base.OUTPUT_DIRNAME = OUTPUT_DIRNAME
    try:
        return _base.run(drive_root, output_dir)
    finally:
        _base._local_direction = old_direction
        _base.ALGORITHM = old_algorithm
        _base.OUTPUT_DIRNAME = old_output
