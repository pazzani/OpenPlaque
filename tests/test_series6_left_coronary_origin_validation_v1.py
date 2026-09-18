import numpy as np

from openplaque.series6_left_coronary_origin_validation_v1 import (
    MAX_LEFT_ANCHOR_TO_CONTACT_MM,
    MIN_DISTINCT_FROM_RCA_MM,
    RCA_CONTACT_RADIUS_MM,
    _resample_polyline,
    synthetic_self_test,
)


def test_synthetic_self_test():
    out = synthetic_self_test()
    assert out["ok"] is True
    assert out["resampled_points"] == 5


def test_resample_polyline_half_mm():
    p = np.array([[0., 0., 0.], [2., 0., 0.]])
    q = _resample_polyline(p, 0.5)
    assert q.shape == (5, 3)
    assert np.allclose(q[-1], [2., 0., 0.])


def test_distinct_left_contact_geometry():
    rca = np.array([0., 0., 0.])
    left = np.array([18., 0., 0.])
    rca_contact = np.array([1., 0., 0.])
    left_contact = np.array([18.5, .5, 0.])
    assert np.linalg.norm(rca_contact - rca) <= RCA_CONTACT_RADIUS_MM
    assert np.linalg.norm(left_contact - rca) >= MIN_DISTINCT_FROM_RCA_MM
    assert np.linalg.norm(left_contact - left) <= MAX_LEFT_ANCHOR_TO_CONTACT_MM
