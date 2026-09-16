import numpy as np
from openplaque import left_main_common_parent_reacquisition as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-main-common-parent-reacquisition-v1.0-lowmem"


def test_synthetic_aorta_monotonic_self_test():
    r = m.synthetic_aorta_monotonic_self_test()
    assert r["ok"]
    assert r["distance_reduction_mm"] > 6.0


def test_bifurcation_seed_orients_endpoints_together():
    lad = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])
    c6 = np.array([[0.1,0.,0.],[0.8,0.6,0.],[1.5,1.2,0.],[2.2,1.8,0.]])
    lo, co, seed, tl, tc, meta = m._bifurcation_seed(lad[::-1], c6)
    assert meta["nearest_lad_c6_distance_mm"] < 0.2
    assert np.linalg.norm(seed - np.array([0.05,0.,0.])) < 0.2
    assert np.linalg.norm(tl) > 0.9
    assert np.linalg.norm(tc) > 0.9


def test_cone_directions_are_forward():
    t = np.array([1.,0.,0.])
    dirs = m._cone_dirs(t)
    assert len(dirs) > 20
    assert all(np.dot(d, t) > 0.45 for d in dirs)
