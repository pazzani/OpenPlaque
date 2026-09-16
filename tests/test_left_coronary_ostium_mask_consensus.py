import numpy as np
from openplaque import left_coronary_ostium_mask_consensus as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-ostium-mask-consensus-v1.0-lowmem"


def test_synthetic_interface_cluster_self_test():
    r = m.synthetic_interface_cluster_self_test()
    assert r["ok"]
    assert r["clusters"] >= 2


def test_rca_calibrated_plane_gate_accepts_coronary_scale():
    rca = {"median_radius_mm": 1.60, "median_center_hu": 565.0}
    metrics = {
        "radius_mm": 1.75,
        "centroid_offset_mm": 0.20,
        "circularity": 0.75,
        "core_minus_ring_hu": 220.0,
        "center_hu": 560.0,
    }
    score, passed = m._score_plane(metrics, rca)
    assert passed
    assert score > 0.7


def test_rca_calibrated_plane_gate_rejects_aortic_scale():
    rca = {"median_radius_mm": 1.60, "median_center_hu": 565.0}
    metrics = {
        "radius_mm": 4.50,
        "centroid_offset_mm": 0.05,
        "circularity": 0.90,
        "core_minus_ring_hu": 150.0,
        "center_hu": 560.0,
    }
    _, passed = m._score_plane(metrics, rca)
    assert not passed


def test_local_direction_points_toward_greater_extra_aortic_distance():
    shape = (31,31,31)
    union = np.zeros(shape, bool)
    union[15,15,15:25] = True
    outside = np.zeros(shape, float)
    for x in range(shape[2]):
        outside[:,:,x] = max(0, x-15)
    t = m._local_direction(np.array([15.,15.,16.]), union, outside, np.ones(3), radius_mm=6)
    assert t[2] > 0.5
