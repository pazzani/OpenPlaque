import numpy as np
import pandas as pd

from openplaque.vendor_q3d_proximal_origin_audit_v1 import (
    MIN_RCA_PROX_DISTAL_RATIO,
    MIN_RCA_SIDE_CONSISTENCY,
    _bright_threshold,
    detect_axis,
    summarize_origin_signatures,
    synthetic_self_test,
    view_metrics,
)


def test_synthetic_self_test():
    out=synthetic_self_test()
    assert out["ok"] is True
    assert out["left_score"] > out["right_score"]


def test_detect_horizontal_axis():
    arr=np.full((512,512),-100.0,np.float32)
    arr[250:263,50:470]=350.0
    axis,thr,hu,hs,vs=detect_axis(arr)
    assert axis=="horizontal"
    assert hu is False or isinstance(hu,bool)
    assert hs>=vs


def test_view_origin_expansion():
    arr=np.full((512,512),-100.0,np.float32)
    arr[250:263,70:470]=350.0
    arr[205:305,0:110]=350.0
    m=view_metrics(arr,forced_axis="horizontal")
    assert m["left"]["origin_score"] > m["right"]["origin_score"]
    assert m["left"]["component_area"] > m["right"]["component_area"]


def test_rca_calibrated_signature():
    rows=[]
    for vessel in ("RCA","LAD","CX"):
        for i in range(24):
            lp=4.0 if vessel=="RCA" else (3.0 if vessel=="LAD" else 1.1)
            rp=1.0
            rows.append({
                "vessel":vessel,
                "view_index":i,
                "left_origin_score":lp,
                "right_origin_score":rp,
            })
    df=pd.DataFrame(rows)
    sig,dec=summarize_origin_signatures(df,"horizontal")
    assert dec["rca_control_pass"] is True
    assert dec["LAD_origin_signature_positive"] is True
    assert dec["CX_origin_signature_positive"] is False
