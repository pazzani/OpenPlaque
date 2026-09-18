import numpy as np

from openplaque.lad_distal_reference_plaque_self_calibration_v1 import (
    BASELINE,
    LAD_REFERENCE_ARC,
    NOMINAL_SHELL,
    SHELLS,
    STATUS_PASS,
    _auc,
    _join_frozen_and_extension,
    _specificity_pass,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert LAD_REFERENCE_ARC == (-15.0, -3.0)
    assert NOMINAL_SHELL == 1.0
    assert set(SHELLS) == {0.75, 1.0, 1.25, 1.5}
    assert STATUS_PASS == "LAD_DISTAL_REFERENCE_SELF_CALIBRATION_DEVELOPMENTAL_PASS"
    assert synthetic_self_test()["ok"] is True


def test_join_extension_to_frozen_distal_endpoint():
    frozen = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    extension = np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    combined, ext_len, gap = _join_frozen_and_extension(frozen, extension)
    assert np.isclose(ext_len, 2.0)
    assert np.isclose(gap, 0.0)
    assert np.allclose(combined[0], [-2.,0.,0.])
    assert np.allclose(combined[-1], [2.,0.,0.])


def test_auc_and_specificity_gate():
    assert _auc([0,0,1,1], [0.1,0.2,0.8,0.9]) == 1.0
    good = {
        "usable": True,
        "majority_vs_vote_free_auc": 0.82,
        "strict_5of5_bins": 2,
        "strict5_vs_vote_free_auc": 0.92,
        "positive_negative_median_ratio": 2.3,
    }
    assert _specificity_pass(good, 0.85)
    bad = dict(good)
    bad["majority_vs_vote_free_auc"] = 0.75
    assert not _specificity_pass(bad, 0.85)


def test_specificity_requires_shell_stability():
    m = {
        "usable": True,
        "majority_vs_vote_free_auc": 0.85,
        "strict_5of5_bins": 3,
        "strict5_vs_vote_free_auc": 0.95,
        "positive_negative_median_ratio": 2.5,
    }
    assert _specificity_pass(m, 0.81)
    assert not _specificity_pass(m, 0.79)
