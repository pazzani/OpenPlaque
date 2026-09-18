import numpy as np

from openplaque.left_coronary_unresolved_state_lock_v1 import (
    BASELINE,
    STATUS,
    _junction_geometry,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS == "LEFT_CORONARY_UNRESOLVED_STATE_RESEARCH_LOCKED"
    assert synthetic_self_test()["ok"] is True


def test_exact_through_junction_geometry():
    lad = np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    c6 = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    g = _junction_geometry(lad, c6, window_mm=2.0)
    assert np.isclose(g["junction_gap_mm"], 0.0)
    assert np.isclose(g["through_deflection_deg"], 0.0)
    assert np.allclose(g["junction_lps_mm"], [0.,0.,0.])


def test_reversed_input_is_oriented_to_junction():
    lad = np.array([[0.,0.,0.],[-1.,0.,0.],[-2.,0.,0.]])
    c6 = np.array([[2.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
    g = _junction_geometry(lad, c6, window_mm=2.0)
    assert np.isclose(g["junction_gap_mm"], 0.0)
    assert np.isclose(g["through_deflection_deg"], 0.0)
