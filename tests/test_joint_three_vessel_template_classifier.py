import numpy as np
import pandas as pd

from openplaque.joint_three_vessel_template_classifier import (
    MASTER, RECIPES, STATUS_CANDIDATE, STATUS_CALIBRATION_FAILED,
    _calibrate, _decision, _score_window, synthetic_joint_classifier_self_test,
)


def _template(phase=0.0, n=100):
    x=np.linspace(0,1,n)
    med=500+80*np.sin(2*np.pi*x+phase)+20*np.sin(6*np.pi*x+phase)
    p90=med+120+30*np.cos(4*np.pi*x+phase)
    plq=np.maximum(np.sin(3*np.pi*x+phase),0)
    return {"start_px":0,"end_px":n,"span_px":n,"hu_median":med,"hu_p90":p90,"plaque_fraction":plq,"support":np.ones(n)}


def _profile(t, length=25.0):
    return {"tube_median":t["hu_median"],"tube_p90":t["hu_p90"],"landmark":t["plaque_fraction"],"robust_fraction":1.0,"median_hu":550.0,"length_mm":length,"points":np.zeros((100,3)),"arc":np.linspace(0,length,100),"center_hu":t["hu_median"]}


def test_master_path_uses_cache():
    assert str(MASTER).startswith("Cache/Master_Coronary_Anatomy_Baseline_v2/")


def test_window_match_prefers_correct_template():
    a=_template(0.0); b=_template(1.2); p=_profile(a)
    sa=_score_window(a,p,"hybrid",0.25)["score"]
    sb=_score_window(b,p,"hybrid",0.25)["score"]
    assert sa > sb


def test_calibration_and_decision():
    templates={"RCA":_template(0.0),"LAD":_template(0.8),"LCX":_template(1.6)}
    profiles={"RCA":_profile(templates["RCA"]),"LAD":_profile(templates["LAD"])}
    c=_calibrate(templates,profiles)
    assert c["recipe"] in RECIPES
    assert c["rca_margin"] > 0
    assert c["lad_margin"] > 0
    ranking=pd.DataFrame([
        {"LCX_score":0.72,"LCX_margin":0.12,"robust_fraction":1.0,"length_mm":20.0},
        {"LCX_score":0.61,"LCX_margin":0.04,"robust_fraction":1.0,"length_mm":18.0},
    ])
    c2={**c,"passed":True,"rca_margin":0.10,"lad_margin":0.10}
    assert _decision(c2,ranking) == STATUS_CANDIDATE
    assert _decision({**c2,"passed":False},ranking) == STATUS_CALIBRATION_FAILED


def test_self_test():
    assert synthetic_joint_classifier_self_test()["passed"]
