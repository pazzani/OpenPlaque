import numpy as np
from openplaque.lcx_hierarchical_distal_tree import synthetic_hierarchy_self_test, _sibling_decision

def test_synthetic_hierarchy():
    r = synthetic_hierarchy_self_test()
    assert r["ok"]
    assert r["root_children"] >= 2

def test_sibling_score_margin_rule():
    rows = [
        {"child_node_id":"A","member_source_candidate_ids":"7","local_geometry_score":0.62,
         "incoming_continuation_angle_deg":12.0,"local_groove_alignment_cos":0.62,"branch_gate_pass":True},
        {"child_node_id":"B","member_source_candidate_ids":"6;9","local_geometry_score":0.54,
         "incoming_continuation_angle_deg":8.0,"local_groove_alignment_cos":0.50,"branch_gate_pass":True},
    ]
    d = _sibling_decision(rows)
    assert d["decision_pass"]
    assert d["selected_child_node_id"] == "A"

def test_sibling_joint_continuity_rule():
    rows = [
        {"child_node_id":"A","member_source_candidate_ids":"6;7;9","local_geometry_score":0.49,
         "incoming_continuation_angle_deg":6.0,"local_groove_alignment_cos":0.46,"branch_gate_pass":True},
        {"child_node_id":"B","member_source_candidate_ids":"5;11","local_geometry_score":0.46,
         "incoming_continuation_angle_deg":40.0,"local_groove_alignment_cos":0.33,"branch_gate_pass":True},
    ]
    d = _sibling_decision(rows)
    assert d["decision_pass"]
    assert d["continuity_angle_advantage_deg"] >= 20.0
    assert d["groove_alignment_advantage"] >= 0.10
