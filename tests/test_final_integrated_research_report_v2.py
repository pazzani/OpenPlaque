import pandas as pd
from openplaque.final_integrated_research_report_v2 import _validate_master_anatomy, _lad_zone_summary, synthetic_v2_self_test


def test_synthetic_v2():
    assert synthetic_v2_self_test()["passed"]


def test_master_anatomy_requires_lm_lcx_unresolved():
    obj={"status":"CORONARY_ANATOMY_BASELINE_V2_FROZEN","accepted":{"RCA_length_mm":52.365,"LAD_length_mm":24.997},"unresolved":["LM","LCX"],"downstream_analysis":{"LAD":"allowed only on accepted 24.997-mm-class combined LAD centerline"}}
    got=_validate_master_anatomy(obj)
    assert got["LAD_length_mm"] == 24.997


def test_lad_zone_summary_preserves_zone():
    df=pd.DataFrame([{"support_zone_id":1,"arc_start_mm":17.626,"arc_end_mm":21.484,"longitudinal_span_mm":3.858,"peak_support_pixels":9,"peak_rotation_fraction":0.0833,"mean_plaque_disagreement":0.327}])
    got=_lad_zone_summary(df)
    assert got[0]["arc_start_mm"] == 17.626
    assert got[0]["peak_support_pixels"] == 9
