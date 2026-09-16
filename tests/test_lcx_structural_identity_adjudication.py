from openplaque import lcx_structural_identity_adjudication as m

def test_synthetic_adjudication():
    r = m.synthetic_adjudication_self_test()
    assert r["ok"]
    assert r["n_gates"] >= 10

def test_algorithm_and_baseline():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "lcx-structural-identity-adjudication-v1.0"

def test_gate_is_conjunctive():
    e = {
        "master_status":"CORONARY_ANATOMY_BASELINE_V2_FROZEN",
        "consensus_trunk_established":True,
        "topology_parent_daughter_established":True,
        "c6_c9_truncation_established":True,
        "chamber_method_replicated_control_pass":True,
        "chamber_winner_id":6,
        "chamber_score_margin":.12,
        "median_distance_advantage_mm":2.0,
        "alignment_advantage":.0,
        "departure_slope_advantage_mm_per_mm":.3,
        "endpoint_distance_advantage_mm":0.,
        "c7_distal_accepted":True,
        "c7_distal_new_length_mm":8.,
        "c7_distal_robust_hu_fraction":1.,
        "c7_distal_union_support_fraction":.95,
        "c7_distal_tortuosity":1.2,
        "c6_robust_hu_fraction":1.,
        "c6_current_support_fraction":.95,
        "c6_legacy_support_fraction":.95,
        "c7_robust_hu_fraction":1.,
        "c7_current_support_fraction":.95,
        "c7_union_support_fraction":.95,
        "c6_dijkstra_control_pass":True,
        "c6_monotonic_control_pass":True,
    }
    gates, ok = m.adjudicate_evidence(e)
    assert ok
    e["c7_distal_accepted"] = False
    _, ok2 = m.adjudicate_evidence(e)
    assert not ok2
