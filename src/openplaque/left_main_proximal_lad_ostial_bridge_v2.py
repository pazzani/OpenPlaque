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


def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    b._write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/b.SOURCE_CACHE/"series7_int16.npy",root/b.SOURCE_CACHE/"series7_int16.json",root/b.MASTER,root/b.LAD_PATH,root/b.RCA_PATH,root/b.CUR,root/b.LEG,root/b.AORTA]
    for p in required:b._req(p)
    master=json.loads((root/b.MASTER).read_text()); ref,src,spacing=b._source(root/b.SOURCE_CACHE); lad=b._load_path(root/b.LAD_PATH,ref); rca=b._load_path(root/b.RCA_PATH,ref); cur=b._resample_mask(root/b.CUR,ref); leg=b._resample_mask(root/b.LEG,ref); aorta=b._resample_mask(root/b.AORTA,ref); aorta_tree,aorta_pts=b._surface_tree(aorta,ref)
    print("Running calibrated RCA proximal retrace to aorta...")
    rpath,rca_summary,rF,rca_o=_trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,rca,max_mm=12.,gate_length_max=10.,start_inside_mm=6.0,require_known_retrace=True)
    b._write_json(out/"RCA_proximal_ostial_control.json",rca_summary)
    print("Searching accepted proximal LAD endpoint to aorta...")
    lpath,lad_summary,lF,lad_o=_trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,lad,max_mm=20.,gate_length_max=18.,start_inside_mm=0.0,require_known_retrace=False)
    b._write_json(out/"proximal_LAD_ostial_bridge_summary.json",lad_summary)
    if lpath is not None: pd.DataFrame(lpath,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"proximal_LAD_to_aorta_candidate.csv",index=False)
    if rpath is not None:
        rm=b._path_metrics(ref,src,cur,leg,rF,rpath,rpath[0],aorta_tree); plt.figure(figsize=(6.5,4)); plt.plot(rm["arc_profile_mm"],rm["aorta_distance_profile_mm"],marker="."); plt.xlabel("RCA control arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title(f"RCA known-proximal retrace | accepted={rca_summary.get('accepted',False)}"); plt.tight_layout(); plt.savefig(out/"01_RCA_calibrated_retrace.png",dpi=180); plt.close()
    if lpath is not None:
        lm=b._path_metrics(ref,src,cur,leg,lF,lpath,lpath[0],aorta_tree); plt.figure(figsize=(6.5,4)); plt.plot(lm["arc_profile_mm"],lm["aorta_distance_profile_mm"],marker="."); plt.xlabel("left bridge arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title("Accepted proximal LAD endpoint→aorta"); plt.tight_layout(); plt.savefig(out/"02_left_ostial_aorta_distance.png",dpi=180); plt.close()
        fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d"); lp,_=b._resample(lad_o,.3); ax.plot(lp[:,0],lp[:,1],lp[:,2],lw=2,label="accepted LAD"); ax.plot(lpath[:,0],lpath[:,1],lpath[:,2],lw=3,label="ostial/proximal candidate"); near=aorta_pts[np.linalg.norm(aorta_pts-lpath[-1],axis=1)<=8]; ax.scatter(near[:,0],near[:,1],near[:,2],s=3,alpha=.15,label="local aortic surface"); ax.legend(); ax.set_title("Accepted LAD proximal endpoint to aortic root"); plt.tight_layout(); plt.savefig(out/"03_left_ostial_geometry.png",dpi=180); plt.close()
        fig,axes=plt.subplots(1,4,figsize=(14,4)); pp,qq=b._resample(lpath,.20); tt=np.gradient(pp,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9); picks=np.linspace(0,len(pp)-1,4).astype(int)
        for ax,ix in zip(axes,picks): im,qv=b._plane(ref,src,pp[ix],tt[ix]); ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=18); ax.set_title(f"bridge +{qq[ix]:.1f} mm")
        plt.tight_layout(); plt.savefig(out/"04_left_ostial_orthogonal_source_qc.png",dpi=180); plt.close()
    control_pass=bool(rca_summary.get("accepted",False)); left_pass=bool(lad_summary.get("accepted",False)); status=STATUS_RCA_FAIL if not control_pass else (STATUS_POS if left_pass else STATUS_NO_BRIDGE)
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"RCA_calibrated_proximal_retrace":rca_summary,"proximal_LAD_to_aorta":lad_summary,"C6_or_LCX_geometry_used_in_search":False,"scientific_boundary":"A positive result nominates a source-supported left-coronary ostial/proximal-trunk bridge from the accepted LAD endpoint to the aorta. It does not yet establish LM or prove where the LCX-like C6 joins."}
    b._write_json(out/"summary.json",summary); b._write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    report=out/"OPENPLAQUE_PROXIMAL_LAD_OSTIAL_BRIDGE_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque Proximal LAD Ostial Bridge v1.1</h1><p><b>Status:</b> {status}</p><p>RCA calibrated retrace accepted: {control_pass}.</p><p>Left bridge accepted: {left_pass}.</p><p>Clinical LM remains unresolved; master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_PROXIMAL_LAD_OSTIAL_BRIDGE_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
