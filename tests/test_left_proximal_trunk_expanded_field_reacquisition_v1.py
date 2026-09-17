import pandas as pd
import numpy as np

from openplaque import left_proximal_trunk_expanded_field_reacquisition_v1 as e


def test_synthetic_expanded_field_self_test():
    r = e.synthetic_expanded_field_self_test()
    assert r["ok"]
    assert r["field_margin_mm"] == 18.0
    assert r["search_max_mm"] == 40.0


def test_truncation_requires_three_consecutive_failures():
    d = pd.DataFrame({
        "arc_mm": np.arange(8) * 0.4,
        "plane_pass": [1, 1, 0, 1, 1, 0, 0, 0],
    })
    r = e._truncate_by_sustained_failure(d, 3)
    assert r["first_sustained_failure_index"] == 5
    assert r["accepted_last_index"] == 4
    assert abs(r["accepted_arc_mm"] - 1.6) < 1e-9


def test_search_budget_has_prospective_slack():
    assert e.SEARCH_MAX_MM >= 29.88011340574657 + e.PROSPECTIVE_MARGIN_MM
