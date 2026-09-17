import json

import numpy as np

from openplaque.left_coronary_backbone_branch_discovery_v1_1 import (
    _branch_angle_safe,
    _overlap_metrics_safe,
    _safe_unit_tangents,
    _source_fixed,
    synthetic_local_vesselness_self_test,
    synthetic_short_path_self_test,
)


def test_synthetic_local_vesselness_self_test():
    out = synthetic_local_vesselness_self_test()
    assert out["ok"] is True
    assert out["max_vesselness"] >= 0.0


def test_source_fixed_ignores_incompatible_cached_vesselness(tmp_path):
    src = np.zeros((4, 5, 6), dtype=np.int16)
    np.save(tmp_path / "series7_int16.npy", src)
    # Regression fixture for the exact bug: cached vesselness is a different cropped shape.
    np.save(tmp_path / "vesselness.npy", np.zeros((2, 2, 2), dtype=np.float32))
    meta = {
        "spacing_zyx": [1.0, 1.0, 1.0],
        "positions_lps_mm": [[0.0, 0.0, 0.0]],
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    (tmp_path / "series7_int16.json").write_text(json.dumps(meta), encoding="utf-8")
    ref, loaded, lazy = _source_fixed(tmp_path)
    assert loaded.shape == src.shape
    assert tuple(ref.GetSize()) == (6, 5, 4)
    assert lazy.field is None


def test_short_path_self_test():
    out = synthetic_short_path_self_test()
    assert out["ok"] is True
    assert out["one_point_overlap_distance_mm"] == 0.0


def test_one_point_tangent_and_branch_angle_do_not_call_gradient():
    p = np.array([[1.0, 2.0, 3.0]])
    t = _safe_unit_tangents(p)
    assert t.shape == (1, 3)
    assert np.allclose(t, 0.0)
    assert _branch_angle_safe(p, np.array([1.0, 0.0, 0.0])) == 0.0


def test_one_point_overlap_is_valid_failed_geometry():
    q = np.array([[1.0, 2.0, 3.0]])
    r = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    out = _overlap_metrics_safe(q, r)
    assert out["min_distance_mm"] == 0.0
    assert out["median_tangent_alignment_within_2mm"] == 0.0
