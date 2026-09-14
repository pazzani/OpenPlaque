from openplaque.secondary_endpoint_fanout import synthetic_fanout_self_test, initial_directions


def test_synthetic_fanout_validator():
    assert synthetic_fanout_self_test()["passed"]


def test_fanout_direction_count():
    dirs = initial_directions([0.0, 1.0, 0.0])
    assert len(dirs) == 113
