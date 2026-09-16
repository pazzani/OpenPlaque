from openplaque import lcx_distal_reacquisition as m


def test_synthetic_vesselness_reacquisition():
    r = m.synthetic_reacquisition_self_test()
    assert r["ok"]
    assert r["center_vesselness"] > r["background_vesselness"]


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "lcx-distal-reacquisition-v1.0-lowmem"
