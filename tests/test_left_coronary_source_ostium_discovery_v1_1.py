import numpy as np
from openplaque import left_coronary_source_ostium_discovery_v1_1 as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-source-ostium-discovery-v1.1-indexfix"


def test_synthetic_root_component_self_test():
    r = m.synthetic_root_component_self_test()
    assert r["ok"]
    assert r["components"] >= 2


def test_component_voxel_indices_remain_integer_for_direct_indexing():
    outside = np.arange(7 * 7 * 7, dtype=float).reshape(7, 7, 7)
    pts = np.array([[2, 2, 2], [2, 2, 3], [2, 2, 4]], dtype=np.int64)
    comp = {"points": pts}
    pts_idx, pts_float, od, seed = m._component_index_and_geometry_points(comp, outside)
    assert np.issubdtype(pts_idx.dtype, np.integer)
    assert np.issubdtype(pts_float.dtype, np.floating)
    np.testing.assert_allclose(od, outside[tuple(pts.T)])
    np.testing.assert_allclose(seed, pts_float[np.argmin(od)])


def test_component_helper_accepts_float_serialized_voxel_coordinates_but_indexes_safely():
    outside = np.zeros((9, 9, 9), float)
    outside[4, 4, 5] = 1.0
    outside[4, 4, 6] = 2.0
    comp = {"points": np.array([[4.0, 4.0, 5.0], [4.0, 4.0, 6.0]])}
    pts_idx, pts_float, od, seed = m._component_index_and_geometry_points(comp, outside)
    assert pts_idx.dtype == np.intp
    assert od.tolist() == [1.0, 2.0]
    assert seed.tolist() == [4.0, 4.0, 5.0]


def test_coronary_scale_plane_gate_accepts_rca_like_lumen():
    rca = {"median_radius_mm": 1.60, "median_center_hu": 560.0}
    metrics = {
        "radius_mm": 1.75,
        "centroid_offset_mm": 0.25,
        "circularity": 0.75,
        "core_minus_ring_hu": 180.0,
        "center_hu": 570.0,
    }
    score, passed = m._score_plane(metrics, rca)
    assert passed
    assert score > 0.65


def test_coronary_scale_plane_gate_rejects_aortic_scale():
    rca = {"median_radius_mm": 1.60, "median_center_hu": 560.0}
    metrics = {
        "radius_mm": 4.50,
        "centroid_offset_mm": 0.05,
        "circularity": 0.90,
        "core_minus_ring_hu": 180.0,
        "center_hu": 570.0,
    }
    _, passed = m._score_plane(metrics, rca)
    assert not passed
