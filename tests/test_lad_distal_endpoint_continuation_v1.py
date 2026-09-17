import numpy as np
from scipy.spatial import cKDTree

from openplaque.lad_distal_endpoint_continuation_v1 import (
    _angle,
    _arc,
    _interp,
    _orient_lad_distal_first,
    _overlap_metrics,
    _reference_tail,
    _unit,
    synthetic_distal_continuation_self_test,
)


def test_synthetic_distal_continuation_self_test():
    out = synthetic_distal_continuation_self_test()
    assert out["ok"] is True
    assert out["tail_length_mm"] >= 4.9


def test_endpoint_orientation_and_outgoing_tangent():
    x = np.linspace(0.0, 10.0, 41)
    lad = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    discovered = np.column_stack([np.linspace(2.0, -5.0, 29), np.zeros(29), np.zeros(29)])
    oriented, d = _orient_lad_distal_first(lad, discovered)
    assert d == 0.0
    assert np.allclose(oriented[0], [0.0, 0.0, 0.0])
    outgoing = _unit(oriented[0] - _interp(oriented, [4.0])[0])
    assert _angle(outgoing, [-1, 0, 0]) < 1e-6


def test_reference_tail_selects_beyond_endpoint_side():
    x = np.linspace(0.0, 10.0, 41)
    lad = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    discovered = np.column_stack([np.linspace(2.0, -5.0, 29), np.zeros(29), np.zeros(29)])
    tree = cKDTree(lad)
    tail = _reference_tail(discovered, lad[0], tree)
    assert tail[-1, 0] < -4.5
    assert _arc(tail)[-1] >= 4.9


def test_overlap_metrics_identical_paths():
    q = np.column_stack([np.linspace(0, 8, 33), np.zeros(33), np.zeros(33)])
    m = _overlap_metrics(q, q)
    assert m["median_distance_mm"] < 1e-8
    assert m["fraction_within_2mm"] == 1.0
    assert m["median_tangent_alignment_within_2mm"] > 0.99
