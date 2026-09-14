from openplaque.secondary_anchored_lumen_validation import synthetic_lumen_validator_self_test


def test_compact_lumen_validator_rejects_broad_pool():
    r = synthetic_lumen_validator_self_test()
    assert r["passed"]
    assert 1.5 <= r["tube_radius_mm"] <= 2.1
    assert r["broad_radius_mm"] > 2.65
