from __future__ import annotations

"""v1.1 calibration wrapper for proximal LAD-to-aorta ostial bridge.

The RCA control starts 6 mm inside the accepted RCA and must retrace the known proximal RCA to
the aorta. The left experiment starts exactly at the accepted LAD endpoint closest to the aorta.
All source-search thresholds are otherwise inherited unchanged from v1.0.
"""

import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import left_main_proximal_lad_ostial_bridge as b

BASELINE=b.BASELINE
ALGORITHM="left-main-proximal-lad-ostial-bridge-v1.1-calibrated"
OUTPUT_DIRNAME="Left_Main_Proximal_LAD_Ostial_Bridge_v1_1"
STATUS_RCA_FAIL=b.STATUS_RCA_FAIL
STATUS_NO_BRIDGE=b.STATUS_NO_BRIDGE
STATUS_POS=b.STATUS_POS


def _trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,path,max_mm,gate_length_max,start_inside_mm=0.0,require_known_retrace=False):
    p,prox_d,flipped=b._orient_endpoint_nearest_tree(path,aorta_tree)
    pr,pq=b._resample(p,.20)
    si=int(np.argmin(np.abs(pq-float(start_inside_mm))))
    start=pr[si]
    look=min(len(pr)-1,si+max(3,int(round(2.0/.20))))
    outward=b._unit(start-pr[look])
    goal=np.asarray(aorta_tree.data[int(aorta_tree.query(start)[1])],float)
    known=pr[:min(len(pr),look+8)]
    F=b._build_field(ref,src,spacing,cur,leg,start,goal,known)
    reached,frontier=b._beam_to_aorta(ref,F,aorta_tree,start,outward,target_max_mm=max_mm,step=.40,beam_width=110)
    kt=cKDTree(pr[:si+1]) if require_known_retrace and si>=2 else None
    rows=[]
    for st in reached:
        pp=np.asarray(st[1]); m=b._path_metrics(ref,src,cur,leg,F,pp,start,aorta_tree); m["beam_score"]=float(st[0])
        if kt is not None:
            dd=kt.query(pp)[0]
            m["median_distance_to_known_proximal_mm"]=float(np.median(dd))
            m["p90_distance_to_known_proximal_mm"]=float(np.percentile(dd,90))
            m["endpoint_error_to_known_proximal_endpoint_mm"]=float(np.linalg.norm(pp[-1]-pr[0]))
        source_gate=bool(m["endpoint_aorta_distance_mm"]<=.80 and 1.0<=m["length_mm"]<=gate_length_max and m["endpoint_displacement_mm"]>=.8 and m["tortuosity"]<=1.8 and m["max_turn_deg"]<=65 and m["robust_hu_fraction"]>=.88 and m["union_support_fraction"]>=.60 and m["p10_vesselness"]>=.42*F["thr"] and m["aorta_distance_reduction_mm"]>=.8)
        retrace_gate=True
        if kt is not None:
            retrace_gate=bool(m["median_distance_to_known_proximal_mm"]<=.80 and m["p90_distance_to_known_proximal_mm"]<=1.30 and m["endpoint_error_to_known_proximal_endpoint_mm"]<=1.60)
        m["source_gate_pass"]=source_gate; m["known_retrace_gate_pass"]=retrace_gate; m["gate_pass"]=bool(source_gate and retrace_gate)
        rows.append((m,pp))
    meta={"proximal_endpoint_aorta_distance_mm":prox_d,"path_was_reversed":bool(flipped),"start_inside_known_path_mm":float(pq[si]),"start_lps_mm":start.tolist(),"outward_tangent":outward.tolist(),"frontier_count":len(frontier),"vesselness_threshold":F["thr"]}
    if not rows:return None,{**meta,"accepted":False,"reason":"no_path_reached_aorta"},F,p
    ok=[x for x in rows if x[0]["gate_pass"]]
    if not ok:
        best=max(rows,key=lambda x:x[0]["beam_score"])[0]
        return None,{**meta,"accepted":False,"reason":"no_candidate_passed","best":{k:v for k,v in best.items() if not isinstance(v,np.ndarray)}},F,p
    ok.sort(key=lambda x:(x[0]["robust_hu_fraction"],x[0]["union_support_fraction"],-x[0]["tortuosity"],x[0]["beam_score"]),reverse=True)
    m,pp=ok[0]
    return pp,{**meta,"accepted":True,"best":{k:v for k,v in m.items() if not isinstance(v,np.ndarray)}},F,p


def synthetic_calibrated_control_self_test():
    p=np.column_stack([np.linspace(0,8,41),np.zeros(41),np.zeros(41)])
    tree=cKDTree(np.array([[-1.,0,0],[20.,0,0]]))
    q,d,f=b._orient_endpoint_nearest_tree(p[::-1],tree)
    assert np.allclose(q[0],[0,0,0]); assert f==1; assert d==1.0
    rr,rq=b._resample(q,.20); si=int(np.argmin(np.abs(rq-6.0))); assert 5.8<=rq[si]<=6.2
    t=b._unit(rr[si]-rr[min(len(rr)-1,si+10)]); assert t[0] < -.9
    return {"ok":True,"control_start_inside_mm":float(rq[si])}
