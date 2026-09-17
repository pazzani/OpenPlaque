import numpy as np

from openplaque.lad_distal_bidirectional_validation_v1 import (
    BASELINE,
    MIN_CONFIRMING_HYPOTHESES,
    MIN_FORWARD_OVERLAP_FRACTION,
    MIN_REVERSE_QC_ARC_MM,
    REVERSE_SEARCH_MM,
    _overlap_metrics,
    _tail_tangent,
    synthetic_self_test,
)


def test_baseline_and_prospective_gates():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert REVERSE_SEARCH_MM >= 15.0
    assert MIN_REVERSE_QC_ARC_MM >= 10.0
    assert MIN_FORWARD_OVERLAP_FRACTION >= 0.80
    assert MIN_CONFIRMING_HYPOTHESES >= 3


def test_reverse_tangent_on_straight_path():
    x = np.linspace(0.0, 15.0, 61)
    p = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    assert np.allclose(_tail_tangent(p, from_end=True), [-1.0, 0.0, 0.0])


def test_overlap_direction_invariant_alignment():
    x = np.linspace(0.0, 15.0, 61)
    p = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    ov = _overlap_metrics(p[::-1].copy(), p[::-1].copy())
    assert ov["fraction_within_2mm"] == 1.0
    assert ov["median_distance_mm"] < 1e-8
    assert ov["median_tangent_alignment_within_2mm"] > 0.99


def test_synthetic_self_test():
    out = synthetic_self_test()
    assert out["ok"] is True
