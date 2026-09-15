import numpy as np

from openplaque import lcx_curved_template_reacquisition_v2 as v2


def test_align_mask_identity_and_permutation():
    a = np.arange(24 * 5 * 7).reshape(24, 5, 7)
    assert v2._align_mask(a, a.shape).shape == a.shape
    b = np.transpose(a, (0, 2, 1))
    assert v2._align_mask(b, a.shape).shape == a.shape


def test_cache_only_contract():
    assert "Full_DICOM" not in str(v2.RCA_INPUT)
    assert "Full_DICOM" not in str(v2.LCX_INPUT)
    result = v2.synthetic_lcx_template_v2_self_test()
    assert result["passed"]
    assert result["historical_Full_DICOM_zip_used"] is False
