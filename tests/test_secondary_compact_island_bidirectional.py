from openplaque.secondary_compact_island_bidirectional import synthetic_island_self_test

def test_synthetic_island_validator():
    r = synthetic_island_self_test()
    assert r["passed"], r
    assert r["n_tracking_directions"] == 33
