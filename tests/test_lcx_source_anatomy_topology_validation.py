import numpy as np
import pandas as pd

from openplaque.lcx_source_anatomy_topology_validation import (
    _first_sustained,
    _score_angle,
    _score_prox_divergence,
    _decision,
    synthetic_anatomy_topology_self_test,
    STATUS_CANDIDATE,
    STATUS_AMBIGUOUS,
    STATUS_NO_CANDIDATE,
)


def test_sustained_divergence():
    x = np.array([False, False, True, True, True, True, True, False])
    assert _first_sustained(x, 5) == 2
    assert _first_sustained(x, 6) is None


def test_anatomy_scores_have_expected_shape():
    assert _score_prox_divergence(2.0) == 1.0
    assert _score_prox_divergence(8.0) == 0.0
    assert _score_angle(80.0) > 0.95
    assert _score_angle(10.0) == 0.0


def test_decision_requires_gate_and_separation():
    q = pd.DataFrame([
        {"anatomy_score": 0.78, "anatomy_gate_pass": True},
        {"anatomy_score": 0.69, "anatomy_gate_pass": True},
    ])
    assert _decision(q) == STATUS_CANDIDATE
    q2 = pd.DataFrame([
        {"anatomy_score": 0.78, "anatomy_gate_pass": True},
        {"anatomy_score": 0.75, "anatomy_gate_pass": True},
    ])
    assert _decision(q2) == STATUS_AMBIGUOUS
    q3 = pd.DataFrame([{"anatomy_score": 0.90, "anatomy_gate_pass": False}])
    assert _decision(q3) == STATUS_NO_CANDIDATE


def test_self_test():
    assert synthetic_anatomy_topology_self_test()["passed"]
