from openplaque import left_main_lad_mask_gate_diagnostic_v1 as d


def test_synthetic_attrition_self_test():
    out = d.synthetic_attrition_self_test()
    assert out["ok"] is True
    assert out["mask_gate_mm"] == 1.6
    assert out["step_mm"] == 0.40
    assert out["beam_width"] == 110


def test_experiment_is_diagnostic_only():
    assert d.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert "diagnostic" in d.ALGORITHM
    assert d.STATUS_MASK != d.STATUS_SOURCE
