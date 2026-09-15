import numpy as np
from openplaque.lcx_local_bifurcation_groove_tangent import (
    _angle_score_from_vectors,
    _consensus_prefix,
    synthetic_local_bifurcation_self_test,
)


def test_angle_score_prefers_true_groove_tangent():
    n_a = np.array([1.0, 0.0, 0.0])
    n_v = np.array([0.0, 1.0, 0.0])
    groove = np.array([0.0, 0.0, 1.0])
    outward = np.array([1.0, 0.0, 1.0])
    outward /= np.linalg.norm(outward)
    assert _angle_score_from_vectors(groove, n_a, n_v) > 0.99
    assert _angle_score_from_vectors(groove, n_a, n_v) > _angle_score_from_vectors(outward, n_a, n_v)


def test_consensus_prefix_detects_split():
    t = np.linspace(0, 20, 81)
    trunk = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    p1 = trunk.copy()
    p2 = trunk.copy(); p2[t > 15, 1] = (t[t > 15] - 15) * 0.8
    p3 = trunk.copy(); p3[t > 15, 2] = (t[t > 15] - 15) * 0.8
    _, _, _, meta = _consensus_prefix([p1, p2, p3], step=0.25, tol_mm=0.4, sustain=3)
    assert 14.5 <= meta["common_prefix_total_mm"] <= 15.5


def test_self_test():
    assert synthetic_local_bifurcation_self_test()["ok"] is True
