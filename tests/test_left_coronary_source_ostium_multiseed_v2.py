import numpy as np
from openplaque import left_coronary_source_ostium_multiseed_v2 as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-source-ostium-multiseed-v2.0-control-recovery"


def test_surface_seed_self_test_covers_separated_contacts():
    r = m.synthetic_seed_coverage_self_test()
    assert r["ok"]
    assert r["n_seeds"] >= 4


def test_surface_seeds_are_spatially_separated_and_integer_safe():
    outside = np.ones((16, 16, 16), float) * 8
    pts = np.array([[8, 8, x] for x in range(2, 14)], dtype=np.int64)
    outside[tuple(pts.T)] = 1.0
    seeds = m._surface_contact_seeds({"points": pts}, outside, np.ones(3),
                                     contact_max_mm=2.0, min_sep_mm=2.5, max_seeds=10)
    xyz = np.asarray([s["seed_zyx_local"] for s in seeds], float)
    assert len(xyz) >= 4
    for i in range(len(xyz)):
        for j in range(i):
            assert np.linalg.norm(xyz[i] - xyz[j]) >= 2.5 - 1e-9


def test_no_surface_contact_means_no_seed():
    outside = np.ones((8, 8, 8), float) * 5
    pts = np.array([[3, 3, 3], [3, 3, 4]], dtype=np.int64)
    assert m._surface_contact_seeds({"points": pts}, outside, np.ones(3)) == []


def test_frozen_coronary_scale_gate_still_rejects_aortic_scale():
    rca = {"median_radius_mm": 1.60, "median_center_hu": 560.0}
    metrics = {"radius_mm": 4.50, "centroid_offset_mm": 0.05, "circularity": 0.90,
               "core_minus_ring_hu": 180.0, "center_hu": 570.0}
    _, passed = m._base._score_plane(metrics, rca)
    assert not passed


def test_multiseed_constants_are_prospective_and_bounded():
    assert m.CONTACT_MAX_MM == 2.0
    assert m.SEED_MIN_SEPARATION_MM == 2.5
    assert m.MAX_SEEDS_PER_COMPONENT == 10
    assert m.CHECKPOINTS_MM == (6.0, 9.0, 12.0)
    assert m.TOP_PER_SEED_CHECKPOINT == 3
    assert m.BEAM_WIDTH == 48
