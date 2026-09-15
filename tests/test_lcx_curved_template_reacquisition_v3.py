from openplaque.lcx_curved_template_reacquisition_v3 import (
    MASTER,
    synthetic_lcx_template_v3_self_test,
)


def test_master_path_is_under_cache():
    assert str(MASTER) == "Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json"


def test_v3_self_test():
    r = synthetic_lcx_template_v3_self_test()
    assert r["passed"]
    assert r["historical_Full_DICOM_zip_used"] is False
