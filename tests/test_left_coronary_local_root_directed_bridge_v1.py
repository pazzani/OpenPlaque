import numpy as np

from openplaque.left_coronary_local_root_directed_bridge_v1 import (
    BASELINE,
    STATUS_CANDIDATE,
    _line_distance,
    _resample_path,
    _symmetric_separation,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_CANDIDATE == "LEFT_CORONARY_LOCAL_ROOT_DIRECTED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED"
    assert synthetic_self_test()["ok"] is True


def test_line_distance():
    pts = np.array([[1.,1.,0.],[1.,0.,0.],[3.,0.,0.]])
    d = _line_distance(pts, np.array([0.,0.,0.]), np.array([2.,0.,0.]))
    assert np.allclose(d, [1.,0.,1.])


def test_resample_and_separation():
    p = np.array([[0.,0.,0.],[2.,0.,0.]])
    q, a = _resample_path(p, .5)
    assert len(q) == 5
    assert np.isclose(a[-1], 2.0)
    s = _symmetric_separation(q, q + np.array([.2,0,0]))
    assert s["median_mm"] <= .2 + 1e-9
