from __future__ import annotations

"""Indexing bugfix for blind source-CCTA coronary ostium discovery v1.0.

Scientific discovery, RCA calibration, candidate gates, and selection logic are unchanged.
The only change is preserving connected-component voxel coordinates as integer indices for
NumPy array indexing while using float copies for physical-space geometry/interpolation.
"""

import numpy as np
import pandas as pd

from . import left_coronary_source_ostium_discovery as _base

BASELINE = _base.BASELINE
ALGORITHM = "left-coronary-source-ostium-discovery-v1.1-indexfix"
OUTPUT_DIRNAME = "Left_Coronary_Source_Ostium_Discovery_v1_1"

# Re-export helpers used by tests/notebook.
_score_plane = _base._score_plane
_blind_root_components = _base._blind_root_components
synthetic_root_component_self_test = _base.synthetic_root_component_self_test


def _component_index_and_geometry_points(comp, outside):
    """Return integer voxel indices, float geometry coordinates, outside distances, and seed.

    Component points originate from np.argwhere and are voxel indices. They must remain integer
    for direct array indexing. Geometry/search code uses an explicit float copy.
    """
    pts_idx = np.asarray(comp["points"], dtype=np.intp)
    if pts_idx.ndim != 2 or pts_idx.shape[1] != 3 or len(pts_idx) == 0:
        raise ValueError("component points must be a non-empty Nx3 voxel-index array")
    od = np.asarray(outside)[tuple(pts_idx.T)]
    seed_i = int(np.argmin(od))
    pts = pts_idx.astype(float, copy=False)
    seed = pts[seed_i].copy()
    return pts_idx, pts, np.asarray(od, float), seed


def _trace_component(comp, src_roi, vessel, outside, spacing, v_thr, rca_cal, label):
    # Bugfix: direct indexing uses integer voxel coordinates; physical geometry uses floats.
    _, pts, _, seed = _component_index_and_geometry_points(comp, outside)
    sp = np.asarray(spacing, float)
    near = pts[np.linalg.norm((pts - seed[None, :]) * sp[None, :], axis=1) <= 6.0]
    tangent = _base._component_direction(near, seed, spacing, outside)
    finals = _base._beam_source(seed, tangent, src_roi, vessel, outside, spacing, v_thr)

    rows = []
    for st in finals:
        p = np.asarray(st[1])
        m = _base._path_metrics(p, src_roi, vessel, outside, spacing, v_thr)
        qdf, qsum = _base._serial_qc(p, src_roi, spacing, rca_cal, n=9, label=label)
        gate = bool(
            m["length_mm"] >= 5.0
            and m["tortuosity"] <= 1.8
            and m["robust_hu_fraction"] >= .90
            and m["p10_vesselness"] >= .50 * v_thr
            and m["outside_aorta_gain_mm"] >= 3.0
            and qsum["plane_pass_fraction"] >= .60
            and qsum["median_plane_score"] >= .60
            and .55 * rca_cal["median_radius_mm"]
            <= qsum.get("median_radius_mm", np.nan)
            <= min(3.4, 2.0 * rca_cal["median_radius_mm"] + .2)
        )
        score = (
            .42 * qsum["median_plane_score"]
            + .30 * qsum["plane_pass_fraction"]
            + .12 * min(1, m["outside_aorta_gain_mm"] / 5)
            + .10 * min(1, m["median_vesselness"] / max(v_thr, 1e-6))
            + .06 * min(1, m["length_mm"] / 8)
        )
        rows.append((gate, score, m, qsum, p, qdf, float(st[0]), seed, tangent))

    if not rows:
        return None, {
            "accepted": False,
            "reason": "no_path_ge_5mm",
            "component_id": comp["component_id"],
            "seed_zyx_local": seed.tolist(),
        }, pd.DataFrame()

    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    gate, score, m, qsum, p, qdf, bscore, seed, tangent = rows[0]
    sm = {
        "accepted": bool(gate),
        "component_id": comp["component_id"],
        "component_n_voxels": comp["n_voxels"],
        "component_span_mm": comp["physical_span_mm"],
        "seed_zyx_local": seed.tolist(),
        "initial_tangent_mm": tangent.tolist(),
        "selection_score": float(score),
        "beam_score": bscore,
        "path_metrics": {k: v for k, v in m.items() if not isinstance(v, np.ndarray)},
        "serial_qc": qsum,
    }
    return p, sm, qdf


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    """Run v1.0 science with only the component-index dtype bug fixed."""
    old_trace = _base._trace_component
    old_algorithm = _base.ALGORITHM
    old_output = _base.OUTPUT_DIRNAME
    _base._trace_component = _trace_component
    _base.ALGORITHM = ALGORITHM
    _base.OUTPUT_DIRNAME = OUTPUT_DIRNAME
    try:
        return _base.run(drive_root, output_dir)
    finally:
        _base._trace_component = old_trace
        _base.ALGORITHM = old_algorithm
        _base.OUTPUT_DIRNAME = old_output
