import numpy as np
import pandas as pd

from openplaque.left_coronary_backbone_branch_discovery_v1 import (
    _angle,
    _branch_angle,
    _cluster_candidates,
    _control_gate,
    _local_tangent,
    _overlap_metrics,
    synthetic_branch_discovery_self_test,
)


def test_self_test():
    out = synthetic_branch_discovery_self_test()
    assert out["ok"] is True
    assert 55 < out["branch_angle_deg"] < 65


def test_local_tangent_on_straight_path():
    x = np.linspace(0, 20, 81)
    p = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    t = _local_tangent(p, 10.0)
    assert _angle(t, np.array([1.0, 0.0, 0.0])) < 1e-6


def test_overlap_control_gate():
    x = np.linspace(0, 8, 33)
    p = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    m = _overlap_metrics(p, p)
    assert _control_gate({"branch_gate_pass": True}, m)
    assert not _control_gate({"branch_gate_pass": False}, m)


def test_cluster_candidates_keeps_separated_seed_regions():
    d = pd.DataFrame([
        {"hypothesis_id":"a","seed_arc_mm":10.0,"accepted_arc_mm":8.0,"accepted_plane_pass_fraction":1.0,"beam_score":10.0},
        {"hypothesis_id":"b","seed_arc_mm":11.5,"accepted_arc_mm":7.0,"accepted_plane_pass_fraction":1.0,"beam_score":9.0},
        {"hypothesis_id":"c","seed_arc_mm":20.0,"accepted_arc_mm":6.0,"accepted_plane_pass_fraction":.9,"beam_score":8.0},
    ])
    out = _cluster_candidates(d)
    assert list(out.hypothesis_id) == ["a", "c"]
