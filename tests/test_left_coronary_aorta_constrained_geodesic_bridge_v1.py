import numpy as np

from openplaque.left_coronary_aorta_constrained_geodesic_bridge_v1 import (
    BASELINE,
    STATUS_CANDIDATE,
    _resample_path,
    _symmetric_path_separation,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_CANDIDATE == "LEFT_CORONARY_AORTA_CONSTRAINED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED"
    assert synthetic_self_test()["ok"] is True


def test_path_resampling():
    p = np.array([[0., 0., 0.], [2., 0., 0.]])
    q, a = _resample_path(p, 0.5)
    assert len(q) == 5
    assert np.isclose(a[-1], 2.0)
    assert np.allclose(q[:, 1:], 0.0)


def test_symmetric_path_separation():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    b = a + np.array([0.25, 0.0, 0.0])
    s = _symmetric_path_separation(a, b)
    assert s["median_mm"] <= 0.25 + 1e-9
    assert s["p90_mm"] <= 0.25 + 1e-9
