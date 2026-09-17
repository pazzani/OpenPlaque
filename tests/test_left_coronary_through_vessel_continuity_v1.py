import numpy as np

from openplaque.left_coronary_through_vessel_continuity_v1 import (
    _angle,
    _branch_control,
    _reentry_gate,
    synthetic_continuity_self_test,
)


def test_self_test():
    assert synthetic_continuity_self_test()["ok"] is True


def test_straight_opposite_outgoing_directions_have_zero_deflection():
    outgoing = _angle(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert abs((180.0 - outgoing)) < 1e-6


def test_reentry_gate():
    m = {
        "min_distance_mm": 0.4,
        "fraction_within_2mm": 0.8,
        "max_contiguous_span_within_2mm": 8.0,
        "median_tangent_alignment_within_2mm": 0.95,
    }
    assert _reentry_gate(m)
    m["max_contiguous_span_within_2mm"] = 2.0
    assert not _reentry_gate(m)


def test_branch_control_distinguishes_parent_and_side_branch():
    split = 20.25
    pre = np.column_stack([np.linspace(0, split, 82), np.zeros(82), np.zeros(82)])
    t = np.linspace(0, 8, 33)
    c6post = np.column_stack([split + np.cos(np.deg2rad(10))*t, np.sin(np.deg2rad(10))*t, np.zeros_like(t)])
    c7post = np.column_stack([split + np.cos(np.deg2rad(60))*t, np.sin(np.deg2rad(60))*t, np.zeros_like(t)])
    c6 = np.vstack([pre, c6post[1:]])
    c7 = np.vstack([pre, c7post[1:]])
    out = _branch_control(c6, c7)
    assert out["parent_like_angle_deg"] < 25.0
    assert out["side_branch_like_angle_deg"] > 35.0
    assert out["parent_side_angle_separation_deg"] > 25.0
    assert out["control_gate_pass"]
