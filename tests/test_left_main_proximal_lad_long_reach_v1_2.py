from openplaque import left_main_proximal_lad_long_reach_v1_2 as exp


def test_long_reach_budget_fixes_impossible_prior_design():
    old = exp.reachability_budget(36.035291878, 20.0)
    new = exp.reachability_budget(36.035291878, exp.LEFT_MAX_MM)
    assert old["geometrically_possible"] is False
    assert new["geometrically_possible"] is True
    assert new["prospectively_adequate"] is True
    assert new["budget_minus_straight_line_mm"] > 9.0


def test_scientific_constants_are_prospective():
    assert exp.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert exp.LEFT_MAX_MM == 46.0
    assert exp.LEFT_GATE_MAX_MM == 44.0
    assert exp.RCA_ENDPOINT_SEPARATION_MM == 8.0
    assert exp.RCA_MAX_MM == 12.0
    assert exp.RCA_GATE_MAX_MM == 10.0


def test_synthetic_reachability_self_test():
    result = exp.synthetic_reachability_self_test()
    assert result["ok"] is True
