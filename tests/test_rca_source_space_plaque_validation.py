import pandas as pd
from openplaque.rca_source_space_plaque_validation import (
    _controls, _decision, _positive_windows, _validate_master,
    synthetic_source_plaque_self_test,
)


def test_controls_avoid_primary_window():
    controls = _controls(52.2, [{'arc_start_mm': 0.0, 'arc_end_mm': 5.0}, {'arc_start_mm': 11.0, 'arc_end_mm': 12.0}])
    assert len(controls) == 2
    assert controls[0]['arc_start_mm'] >= 15.0


def test_positive_windows_rank_support():
    df = pd.DataFrame([
        {'confidence_level':'majority_3plus','arc_start_mm':11.0,'arc_end_mm':12.0,'duration_mm':1.0,'mapped_native_voxels_vote_ge3':10},
        {'confidence_level':'majority_3plus','arc_start_mm':0.0,'arc_end_mm':5.0,'duration_mm':5.0,'mapped_native_voxels_vote_ge3':57},
    ])
    w = _positive_windows(df)
    assert w[0]['label'] == 'primary_model_positive'
    assert w[0]['arc_start_mm'] == 0.0


def test_master_gate():
    master = {'status':'CORONARY_ANATOMY_BASELINE_V2_FROZEN','accepted':{'RCA_length_mm':52.365},'unresolved':['LM','LCX']}
    out = _validate_master(master, 52.20)
    assert abs(out['difference_mm']) < 0.5


def test_decision_and_self_test():
    assert _decision(1.0, .8, .05, .4).startswith('SOURCE_CALCIFIC_CANDIDATE_ENRICHED')
    assert _decision(0.0, 0.0, .05, 0.0).startswith('NO_STRICT')
    assert synthetic_source_plaque_self_test()['passed']
