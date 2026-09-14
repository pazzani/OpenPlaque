from openplaque.secondary_3d_vesselness_topology import synthetic_vesselness_self_test


def test_synthetic_vesselness_prefers_tube_center():
    result = synthetic_vesselness_self_test()
    assert result["passed"]
    assert result["center_vesselness"] > result["background_vesselness"]
