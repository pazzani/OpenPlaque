import pandas as pd

from openplaque import left_coronary_source_ostium_multiseed_control_v2_1 as m


def test_accepted_keys_only_returns_blind_accepted_rows():
    df = pd.DataFrame({
        "component_id": [305, 305, 13],
        "seed_rank": [1, 6, 2],
        "accepted": [True, True, False],
    })
    assert m._accepted_keys(df) == [(305, 1), (305, 6)]


def test_control_geometry_limits_are_frozen_v2_values():
    good = {
        "postsearch_median_distance_to_known_RCA_mm": 1.0,
        "postsearch_p90_distance_to_known_RCA_mm": 1.8,
        "postsearch_seed_distance_to_known_RCA_prox_mm": 3.0,
    }
    assert m._control_geometry_pass(good)

    for key, bad_value in [
        ("postsearch_median_distance_to_known_RCA_mm", 1.0001),
        ("postsearch_p90_distance_to_known_RCA_mm", 1.8001),
        ("postsearch_seed_distance_to_known_RCA_prox_mm", 3.0001),
    ]:
        bad = dict(good)
        bad[key] = bad_value
        assert not m._control_geometry_pass(bad)


def test_algorithm_and_baseline_are_explicit():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-source-ostium-multiseed-v2.1-control-adjudication"
