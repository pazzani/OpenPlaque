import numpy as np

from openplaque.lcx_structural_source_qc_freeze_v1 import (
    BASELINE,
    STATUS_PASS,
    BIFURCATION_EXCLUSION_MM,
    _resample,
    _vessel_gate,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_PASS == "LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN"
    assert BIFURCATION_EXCLUSION_MM == 1.0
    assert synthetic_self_test()["ok"] is True


def test_resample_simple_line():
    p = np.array([[0.,0.,0.],[2.,0.,0.]])
    r, q = _resample(p, 0.5)
    assert len(r) == 5
    assert np.isclose(q[-1], 2.0)
    assert np.allclose(r[:,1:], 0.0)


def test_vessel_gate_pass_and_fail():
    good = {
        "station_qc_pass_fraction": 0.90,
        "median_axis_proxy": 1.6,
        "p90_axis_proxy": 2.2,
        "median_center_hu": 450.0,
    }
    assert _vessel_gate(good)
    bad = dict(good)
    bad["station_qc_pass_fraction"] = 0.80
    assert not _vessel_gate(bad)
    bad = dict(good)
    bad["median_axis_proxy"] = 2.2
    assert not _vessel_gate(bad)
    bad = dict(good)
    bad["p90_axis_proxy"] = 2.6
    assert not _vessel_gate(bad)
