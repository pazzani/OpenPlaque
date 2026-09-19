import numpy as np
import pandas as pd

from openplaque.vendor_q3d_proximal_origin_audit_v1 import (
    PAIR_OFFSET,
    build_rca_angle_template,
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
    assert hs>=vs


def test_view_origin_expansion():
    arr=np.full((512,512),-100.0,np.float32)
    arr[250:263,70:470]=350.0
    arr[205:305,0:110]=350.0
    m=view_metrics(arr)
    assert m["analysis_axis"]=="horizontal"
    assert m["left"]["origin_score"] > m["right"]["origin_score"]


def _row(vessel,idx,left,right,informative=True,axis="vertical"):
    return {
        "vessel":vessel,
        "view_index":idx,
        "angle_index":idx % PAIR_OFFSET,
        "repeat_index":idx // PAIR_OFFSET,
        "analysis_axis":axis,
        "detected_axis":axis,
        "view_informative":informative,
        "dominant_end":"left" if left>=right else "right",
        "left_origin_score":float(left),
        "right_origin_score":float(right),
    }


def test_paired_rca_orientation_excludes_uninformative_views():
    rows=[]
    for i in range(24):
        angle=i % 12
        if angle in (4,5):
            rows.append(_row("RCA",i,0.05,0.02,informative=False))
        else:
            rows.append(_row("RCA",i,3.0,0.2,informative=True))
    df=pd.DataFrame(rows)
    t=build_rca_angle_template(df)
    assert int(t.usable_for_orientation.sum())==10
    assert not bool(t.loc[t.angle_index==4,"usable_for_orientation"].iloc[0])
    assert set(t[t.usable_for_orientation].rca_proximal_end)=={"left"}


def test_rca_angle_template_transfers_to_lad_and_cx():
    rows=[]
    for vessel in ("RCA","LAD","CX"):
        for i in range(24):
            if vessel=="RCA":
                left,right=3.0,0.2
            elif vessel=="LAD":
                left,right=4.5,0.4
            else:
                left,right=4.0,0.3
            rows.append(_row(vessel,i,left,right,True))
    df=pd.DataFrame(rows)
    template=build_rca_angle_template(df)
    sig,dec=summarize_origin_signatures(df,"vertical",template)
    assert dec["rca_control_pass"] is True
    assert dec["rca_valid_oriented_angle_pairs"]==12
    assert dec["LAD_origin_signature_positive"] is True
    assert dec["CX_origin_signature_positive"] is True
    assert set(sig[sig.vessel.isin(["LAD","CX"])].origin_signature_positive)=={True}


def test_inconsistent_rca_pairs_fail_control():
    rows=[]
    for i in range(24):
        # First repeat says left; second repeat says right for every angle.
        left,right=(3.0,0.2) if i<12 else (0.2,3.0)
        rows.append(_row("RCA",i,left,right,True))
    for vessel in ("LAD","CX"):
        for i in range(24):
            rows.append(_row(vessel,i,4.0,0.3,True))
    df=pd.DataFrame(rows)
    template=build_rca_angle_template(df)
    sig,dec=summarize_origin_signatures(df,"vertical",template)
    assert dec["rca_control_pass"] is False
    assert dec["rca_valid_oriented_angle_pairs"]==0
