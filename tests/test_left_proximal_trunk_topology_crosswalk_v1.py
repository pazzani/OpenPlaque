import numpy as np

from openplaque import left_proximal_trunk_topology_crosswalk_v1 as e


def test_synthetic_topology_crosswalk():
    r = e.synthetic_topology_crosswalk_self_test()
    assert r["ok"]
    assert r["near_gate"] is True
    assert r["far_gate"] is False


def test_declared_association_thresholds():
    assert e.ASSOC_NEAR_MM == 1.5
    assert e.ASSOC_BAND_MM == 2.0
    assert e.ASSOC_MIN_SPAN_MM == 5.0
    assert e.ASSOC_MIN_FRACTION == 0.20
    assert e.ASSOC_MIN_TANGENT_ALIGNMENT == 0.75


def test_classification_prefers_unique_post_split_branch():
    base = {k: {"association_gate_pass": False} for k in
            ["LAD", "C6_C7_common_trunk", "C6_post_split", "C7_post_split", "RCA"]}
    x = {k: dict(v) for k, v in base.items()}
    x["C6_post_split"]["association_gate_pass"] = True
    assert e._classify(x) == e.STATUS_C6
    x["C7_post_split"]["association_gate_pass"] = True
    assert e._classify(x) == e.STATUS_AMBIG
