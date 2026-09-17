import numpy as np
import pandas as pd

from openplaque.left_coronary_bifurcation_parent_recovery_v1 import (
    _parent_direction,
    _path_overlap,
    _truncate,
    _unit,
    synthetic_parent_recovery_self_test,
)


def test_parent_bisector_points_opposite_daughters():
    d1 = _unit([1.0, 1.0, 0.0])
    d2 = _unit([1.0, -1.0, 0.0])
    p = _parent_direction(d1, d2)
    assert np.dot(p, np.array([-1.0, 0.0, 0.0])) > 0.99


def test_overlap_gate_metrics_for_nearly_same_path():
    x = np.linspace(0, 10, 41)
    a = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    b = np.column_stack([x, np.full_like(x, 0.25), np.zeros_like(x)])
    m = _path_overlap(a, b)
    assert m["min_distance_mm"] < 0.5
    assert m["fraction_within_2mm"] == 1.0
    assert m["median_tangent_alignment_within_2mm"] > 0.99


def test_dense_qc_truncation_requires_three_consecutive_failures():
    df = pd.DataFrame({"arc_mm": np.arange(10) * 0.4,
                       "plane_pass": [1,1,1,1,0,1,1,0,0,0]})
    r = _truncate(df)
    assert r["first_sustained_failure_index"] == 7
    assert r["accepted_last_index"] == 6


def test_synthetic_self_test():
    assert synthetic_parent_recovery_self_test()["ok"] is True
