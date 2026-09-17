from __future__ import annotations

"""Label-neutral side-branch discovery along the validated left-coronary source-CCTA backbone.

The immediately preceding continuity experiment established that the frozen LAD and the
validated proximal/common-trunk segment behave as one continuous vessel, and that the prior
"parent" candidate mostly re-entered known LAD. This experiment therefore stops trying to
force an LM direction. Instead it builds a label-neutral backbone from the frozen LAD plus the
established C6 parent-continuation trajectory, validates the known C6/C7 split as a same-method
positive control, and then searches blindly for other source-supported side branches along the
backbone.

Aortic geometry and clinical LAD/LM/LCX/OM labels are not used in discovery or ranking. The
frozen master is never modified. Any newly found branch is an unlabeled source-CCTA branch
hypothesis until separately adjudicated.
"""

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-coronary-backbone-branch-discovery-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Backbone_Branch_Discovery_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
PROX = Path("Left_Proximal_Trunk_Continuation_QC_v1/accepted_proximal_trunk_continuation_candidate.csv")
C6 = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7 = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
CONTINUITY = Path("Left_Coronary_Through_Vessel_Continuity_v1/summary.json")

C67_SPLIT_ARC_MM = 20.25
STEP_MM = 0.40
SEARCH_MM = 9.0
BEAM_WIDTH = 36
INITIAL_DIRS = 6
PREVIEW_MM = 1.6
SEED_SPACING_MM = 1.5
SEED_END_MARGIN_MM = 4.0
CONTROL_EXCLUSION_HALF_WIDTH_MM = 3.0
BACKBONE_EXCLUSION_AFTER_MM = 2.0
BACKBONE_EXCLUSION_MM = 0.80
PLANE_STEP_MM = 0.40
SUSTAINED_FAIL_N = 3

MIN_ACCEPTED_BRANCH_MM = 5.0
MIN_PLANE_PASS_FRACTION = 0.80
MIN_BRANCH_ANGLE_DEG = 30.0
MIN_ENDPOINT_BACKBONE_SEPARATION_MM = 3.0
MIN_CONTROL_FRACTION_WITHIN_2MM = 0.70
MAX_CONTROL_MEDIAN_DISTANCE_MM = 1.5
MAX_CONTROL_P90_DISTANCE_MM = 2.5
MAX_CONTROL_ENDPOINT_DISTANCE_MM = 2.5

STATUS_PREREQ = "BACKBONE_BRANCH_DISCOVERY_PREREQUISITE_FAILED"
STATUS_CONTROL_FAIL = "BACKBONE_BRANCH_DISCOVERY_C7_CONTROL_FAILED"
STATUS_NONE = "BACKBONE_BRANCH_DISCOVERY_NO_ADDITIONAL_VALID_BRANCH"
STATUS_FOUND = "BACKBONE_BRANCH_DISCOVERY_ADDITIONAL_SOURCE_BRANCHES_FOUND"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source(cache):
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    ves = np.load(_req(cache / "vesselness.npy"), mmap_mode="r")
    if src.shape != ves.shape:
        raise RuntimeError(f"source/vesselness shape mismatch: {src.shape} vs {ves.shape}")
    meta = _read_json(cache / "series7_int16.json")
    ref = sitk.GetImageFromArray(np.asarray(src))
    sp_zyx = np.asarray(meta["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp_zyx[::-1]))
    ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref, src, ves


def _xyz_to_zyx(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return (((pts - o) @ np.linalg.inv(D).T) / sp)[:, ::-1]


def _zyx_to_xyz(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float)); idx = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx * sp) @ D.T


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"), ("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in (("zyx_z","zyx_y","zyx_x"), ("source_z","source_y","source_x"), ("z","y","x")):
        if all(c in d.columns for c in cols):
            return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    raise ValueError(f"No recognized coordinates in {path}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp(p, q):
    a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:,k]) for k in range(3)])


def _resample(p, step=.25):
    p = np.asarray(p, float); a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0.0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _slice_arc(p, lo=0.0, hi=None, step=.20):
    a = _arc(p)
    hi = float(a[-1] if hi is None else min(hi, a[-1]))
    lo = float(max(0.0, min(lo, hi)))
    q = np.arange(lo, hi + 1e-9, step)
    if len(q) == 0 or q[-1] < hi - 1e-6:
        q = np.r_[q, hi]
    return _interp(np.asarray(p, float), q)


def _unit(v):
    v = np.asarray(v, float); n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def _orth_basis(t):
    t = _unit(t); axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed)); v = _unit(np.cross(t, u))
    return u, v


def _orient_to_point(path, point):
    p = np.asarray(path, float)
    if np.linalg.norm(p[-1] - point) < np.linalg.norm(p[0] - point):
        p = p[::-1].copy()
    return p


def _orient_pair_at_junction(a, b):
    choices=[]
    for ia, pa in enumerate((a[0],a[-1])):
        for ib, pb in enumerate((b[0],b[-1])):
            choices.append((float(np.linalg.norm(pa-pb)), ia, ib))
    gap, ia, ib = min(choices)
    if ia == 1: a = a[::-1].copy()
    if ib == 1: b = b[::-1].copy()
    j = (a[0] + b[0]) / 2.0
    return a, b, j, gap


def _join_paths(parts, tol=.75):
    out = None
    for p in parts:
        p = np.asarray(p, float)
        if out is None:
            out = p.copy(); continue
        if np.linalg.norm(out[-1] - p[0]) > np.linalg.norm(out[-1] - p[-1]):
            p = p[::-1].copy()
        if np.linalg.norm(out[-1] - p[0]) > tol:
            raise RuntimeError(f"cannot join paths; endpoint gap {np.linalg.norm(out[-1]-p[0]):.3f} mm")
        out = np.vstack([out, p[1:]])
    return out


def _sample_array(ref, arr, pts, cval=0.0):
    return map_coordinates(np.asarray(arr), _xyz_to_zyx(ref, pts).T, order=1, mode="constant", cval=cval)


def _local_tangent(path, arc_mm, half=2.0):
    a = _arc(path)
    lo = max(0.0, arc_mm-half); hi = min(float(a[-1]), arc_mm+half)
    if hi-lo < .5:
        lo = max(0.0, arc_mm-1.0); hi = min(float(a[-1]), arc_mm+1.0)
    return _unit(_interp(path,[hi])[0] - _interp(path,[lo])[0])


def _branch_seed_dirs(tangent):
    t = _unit(tangent); u,v = _orth_basis(t)
    dirs=[]
    for deg in (35,45,55,65,75,85,95,105,115,125,135,145):
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,12,endpoint=False):
            dirs.append(_unit(np.cos(a)*t + np.sin(a)*(np.cos(phi)*u + np.sin(phi)*v)))
    return dirs


def _cone_dirs(t, max_deg=45):
    t=_unit(t); u,v=_orth_basis(t); out=[t]
    for deg in (12,24,36,45):
        if deg>max_deg: continue
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,10,endpoint=False):
            out.append(_unit(np.cos(a)*t + np.sin(a)*(np.cos(phi)*u + np.sin(phi)*v)))
    return out


def _preview_dirs(ref, src, ves, start, tangent, backbone_tree, vnorm, n=INITIAL_DIRS):
    ranked=[]
    for d in _branch_seed_dirs(tangent):
        q = np.linspace(.4, PREVIEW_MM, 4)
        pts = start[None,:] + q[:,None]*d[None,:]
        hu = _sample_array(ref,src,pts,cval=-1024.0)
        vv = _sample_array(ref,ves,pts,cval=0.0)
        ep_sep = float(backbone_tree.query(pts[-1])[0])
        if ep_sep < .45 or np.mean((hu>=100)&(hu<=1400)) < .75:
            continue
        sc = 1.8*float(np.mean(np.clip(vv/max(vnorm,1e-6),0,1.5))) + .35*float(np.mean(np.clip((hu-100)/500.,0,1))) + .15*min(ep_sep/2.0,1.0)
        ranked.append((sc,d))
    ranked.sort(key=lambda x:x[0], reverse=True)
    keep=[]
    for sc,d in ranked:
        if all(_angle(d,k[1]) >= 18.0 for k in keep):
            keep.append((sc,d))
        if len(keep)>=n:
            break
    return keep


def _beam(ref, src, ves, start, initial_dir, backbone_tree, vthr, vnorm, max_mm=SEARCH_MM):
    states=[(0.0,[np.asarray(start,float)],_unit(initial_dir))]
    best=states[0]; rows=[]; nsteps=int(math.ceil(max_mm/STEP_MM))
    for si in range(nsteps):
        nxt=[]; cnt={"step":si,"arc_budget_mm":float((si+1)*STEP_MM),"states_in":len(states),"proposals":0,
                    "reject_turn":0,"reject_loop":0,"reject_backbone":0,"reject_hu":0,"reject_vesselness":0,
                    "accepted":0,"kept":0}
        for score,pts,t in states:
            for d in _cone_dirs(t):
                cnt["proposals"]+=1
                if np.dot(d,t) < math.cos(math.radians(55)):
                    cnt["reject_turn"]+=1; continue
                q=np.asarray(pts[-1])+STEP_MM*d
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-q,axis=1)) < .60*STEP_MM:
                    cnt["reject_loop"]+=1; continue
                arc_new=len(pts)*STEP_MM
                sep=float(backbone_tree.query(q)[0])
                if arc_new > BACKBONE_EXCLUSION_AFTER_MM and sep < BACKBONE_EXCLUSION_MM:
                    cnt["reject_backbone"]+=1; continue
                hu=float(_sample_array(ref,src,[q],cval=-1024.0)[0])
                vv=float(_sample_array(ref,ves,[q],cval=0.0)[0])
                if not (100 <= hu <= 1400):
                    cnt["reject_hu"]+=1; continue
                if vv < vthr:
                    cnt["reject_vesselness"]+=1; continue
                vn=min(1.5,max(0.0,vv/max(vnorm,1e-6)))
                align=max(0.0,float(np.dot(d,t)))
                div=min(1.0,sep/3.0) if arc_new>BACKBONE_EXCLUSION_AFTER_MM else 0.0
                ns=score + 1.8*vn + .30*min(1.0,max(0.0,hu-100)/500.0) + .35*align + .20*div
                nt=_unit(.70*t+.30*d); st=(ns,pts+[q],nt); nxt.append(st); cnt["accepted"]+=1
                if len(st[1])>len(best[1]) or (len(st[1])==len(best[1]) and ns>best[0]):
                    best=st
        if not nxt:
            rows.append(cnt); break
        nxt.sort(key=lambda x:x[0], reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(np.asarray(st[1][-1])/.35).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=BEAM_WIDTH: break
        states=keep; cnt["kept"]=len(states); rows.append(cnt)
    finals=[]
    for st in list(states)+[best]:
        ep=np.asarray(st[1][-1])
        if all(np.linalg.norm(ep-np.asarray(x[1][-1]))>1.0 for x in finals):
            finals.append(st)
    finals.sort(key=lambda s:(len(s[1]),s[0]), reverse=True)
    return finals[:4], pd.DataFrame(rows)


def _plane(ref,src,c,t,half=5.5,step=.20):
    t=_unit(t);u,v=_orth_basis(t);q=np.arange(-half,half+1e-9,step);yy,xx=np.meshgrid(q,q,indexing="ij")
    P=c+xx[...,None]*u+yy[...,None]*v
    return _sample_array(ref,src,P.reshape(-1,3),cval=-1024.0).reshape(len(q),len(q)),q


def _component_metrics(im,q):
    iy=ix=int(np.argmin(np.abs(q))); center=float(im[iy,ix]); yy,xx=np.meshgrid(q,q,indexing="ij"); rr=np.sqrt(xx*xx+yy*yy)
    thr=max(220.,min(500.,.55*center)); bw=(im>=thr)&(rr<=3.5); lab,_=ndi.label(bw,np.ones((3,3),int)); labels=[]
    if lab[iy,ix]>0: labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt(q[pts[:,1]]**2+q[pts[:,0]]**2);j=int(np.argmin(dist))
            if dist[j]<=1.0: labels=[int(lab[tuple(pts[j])])]
    if not labels:
        return {"center_hu":center,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
    mask=lab==labels[0];pts=np.argwhere(mask);xs=q[pts[:,1]];ys=q[pts[:,0]];cx=float(np.mean(xs));cy=float(np.mean(ys));off=float(np.hypot(cx,cy));area=float(len(pts)*.04);rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        C=np.cov(np.column_stack([xs,ys]).T);ev=np.linalg.eigvalsh(C);axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else: axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0);contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan
    return {"center_hu":center,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}


def _plane_pass(m):
    return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2 and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)


def _dense_qc(ref,src,path):
    p,q=_resample(path,PLANE_STEP_MM);tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m.update(index=i,arc_mm=float(a),plane_pass=_plane_pass(m));rows.append(m)
    return p,q,pd.DataFrame(rows)


def _truncate_qc(df):
    ps=df.plane_pass.astype(bool).to_numpy();cut=len(df)-1;first=None
    for i in range(len(ps)-SUSTAINED_FAIL_N+1):
        if not np.any(ps[i:i+SUSTAINED_FAIL_N]):
            first=i;cut=max(0,i-1);break
    d=df.iloc[:cut+1];arc=float(d.arc_mm.iloc[-1]) if len(d) else 0.0;frac=float(d.plane_pass.mean()) if len(d) else 0.0
    return {"first_sustained_failure_index":first,"accepted_last_index":int(cut),"accepted_arc_mm":arc,
            "accepted_plane_pass_fraction":frac,"accepted":bool(arc>=MIN_ACCEPTED_BRANCH_MM and frac>=MIN_PLANE_PASS_FRACTION),
            "covers_full_path":bool(cut==len(df)-1)}


def _overlap_metrics(query, ref, step=.25):
    q,qa=_resample(query,step);r,ra=_resample(ref,step);tree=cKDTree(r);d,ix=tree.query(q)
    tq=np.gradient(q,axis=0);tq/=np.maximum(np.linalg.norm(tq,axis=1,keepdims=True),1e-9)
    tr=np.gradient(r,axis=0);tr/=np.maximum(np.linalg.norm(tr,axis=1,keepdims=True),1e-9)
    al=np.abs(np.sum(tq*tr[ix],axis=1));within=d<=2.0;best=cur=0.0
    for v in within:
        cur=cur+step if v else 0.0;best=max(best,cur)
    return {"min_distance_mm":float(np.min(d)),"median_distance_mm":float(np.median(d)),"p90_distance_mm":float(np.percentile(d,90)),
            "endpoint_distance_mm":float(d[-1]),"fraction_within_2mm":float(np.mean(within)),"max_contiguous_span_within_2mm":float(best),
            "median_tangent_alignment_within_2mm":float(np.median(al[within])) if np.any(within) else 0.0,
            "nearest_query_arc_mm":float(qa[int(np.argmin(d))]),"nearest_reference_arc_mm":float(ra[ix[int(np.argmin(d))]])}


def _branch_angle(path, backbone_tangent):
    a=_arc(path); q=min(3.0,float(a[-1])); return _angle(backbone_tangent,_unit(_interp(path,[q])[0]-path[0]))


def _evaluate_seed(ref,src,ves,backbone,backbone_tree,seed_arc,vthr,vnorm,tag):
    start=_interp(backbone,[seed_arc])[0];tan=_local_tangent(backbone,seed_arc);dirs=_preview_dirs(ref,src,ves,start,tan,backbone_tree,vnorm)
    attrs=[];rows=[];paths={};qcdfs={}
    for di,(preview,d) in enumerate(dirs):
        finals,attr=_beam(ref,src,ves,start,d,backbone_tree,vthr,vnorm);attr["seed_arc_mm"]=seed_arc;attr["dir_index"]=di;attr["preview_score"]=preview;attrs.append(attr)
        for fi,st in enumerate(finals[:2]):
            raw=np.asarray(st[1],float);p,q,qc=_dense_qc(ref,src,raw);tr=_truncate_qc(qc);hid=f"{tag}_d{di}_f{fi}";last=int(tr["accepted_last_index"]);acc=p[:last+1] if last>=0 else p[:1]
            epsep=float(backbone_tree.query(acc[-1])[0]);ang=_branch_angle(acc,tan)
            row={"hypothesis_id":hid,"seed_arc_mm":float(seed_arc),"dir_index":di,"final_index":fi,"preview_score":float(preview),
                 "beam_score":float(st[0]),"path_arc_mm":float(q[-1]),"branch_angle_deg":float(ang),"endpoint_backbone_separation_mm":epsep,**tr}
            row["branch_gate_pass"]=bool(tr["accepted"] and ang>=MIN_BRANCH_ANGLE_DEG and epsep>=MIN_ENDPOINT_BACKBONE_SEPARATION_MM)
            rows.append(row);paths[hid]=acc;qcdfs[hid]=qc
    return (pd.concat(attrs,ignore_index=True) if attrs else pd.DataFrame()),pd.DataFrame(rows),paths,qcdfs


def _control_gate(row, overlap):
    return bool(row.get("branch_gate_pass",False) and overlap["fraction_within_2mm"]>=MIN_CONTROL_FRACTION_WITHIN_2MM and
                overlap["median_distance_mm"]<=MAX_CONTROL_MEDIAN_DISTANCE_MM and overlap["p90_distance_mm"]<=MAX_CONTROL_P90_DISTANCE_MM and
                overlap["endpoint_distance_mm"]<=MAX_CONTROL_ENDPOINT_DISTANCE_MM)


def _cluster_candidates(df):
    if df.empty: return df.copy()
    d=df.sort_values(["accepted_arc_mm","accepted_plane_pass_fraction","beam_score"],ascending=[False,False,False]).copy();clusters=[];used=[]
    for _,r in d.iterrows():
        arc=float(r.seed_arc_mm)
        if any(abs(arc-u)<3.0 for u in used): continue
        used.append(arc);clusters.append(r)
    return pd.DataFrame(clusters).reset_index(drop=True)


def synthetic_branch_discovery_self_test():
    x=np.linspace(0,20,81);back=np.column_stack([x,np.zeros_like(x),np.zeros_like(x)])
    t=_local_tangent(back,10.0);assert _angle(t,[1,0,0])<1e-6
    q=np.linspace(0,8,33);br=np.column_stack([10+np.cos(np.deg2rad(60))*q,np.sin(np.deg2rad(60))*q,np.zeros_like(q)])
    assert 55 < _branch_angle(br,t) < 65
    ov=_overlap_metrics(br,br);assert ov["median_distance_mm"]<1e-6 and ov["fraction_within_2mm"]==1.0
    row={"branch_gate_pass":True};assert _control_gate(row,ov)
    return {"ok":True,"branch_angle_deg":_branch_angle(br,t)}


def _finalize(out,summary):
    report=out/"OPENPLAQUE_LEFT_CORONARY_BACKBONE_BRANCH_DISCOVERY_REPORT.html"
    rows="".join(f"<tr><td>{r.get('hypothesis_id')}</td><td>{r.get('seed_arc_mm')}</td><td>{r.get('accepted_arc_mm')}</td><td>{r.get('branch_angle_deg')}</td><td>{r.get('endpoint_backbone_separation_mm')}</td></tr>" for r in summary.get("top_additional_branches",[]))
    report.write_text("<html><body><h1>OpenPlaque Left-Coronary Backbone Branch Discovery v1</h1>"+
        f"<p><b>Status:</b> {summary.get('status')}</p><p>Master: {summary.get('master_status')}; modified: {summary.get('master_modified')}</p>"+
        f"<p>C7 same-method control: {summary.get('control',{}).get('pass')}</p><p>Backbone length: {summary.get('backbone_length_mm')} mm.</p>"+
        "<table border='1'><tr><th>Candidate</th><th>seed arc mm</th><th>accepted arc mm</th><th>branch angle deg</th><th>endpoint separation mm</th></tr>"+rows+"</table>"+
        "<p>Label-neutral research result only. No LAD/LM/LCX/OM relabeling and frozen master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_CORONARY_BACKBONE_BRANCH_DISCOVERY_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root);out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME;out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/SOURCE_CACHE/"vesselness.npy",
              root/MASTER,root/LAD,root/PROX,root/C6,root/C7,root/CONTINUITY]
    for p in required: _req(p)
    master=_read_json(root/MASTER);cont=_read_json(root/CONTINUITY)
    prereq=bool(master.get("status")=="CORONARY_ANATOMY_BASELINE_V2_FROZEN" and cont.get("status")=="LAD_COMMON_TRUNK_THROUGH_VESSEL_CONTINUITY_CONFIRMED_PARENT_CANDIDATE_REENTERS_LAD")
    if not prereq:
        s={"status":STATUS_PREREQ,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"continuity_status":cont.get("status")}
        _write_json(out/"summary.json",s);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_PREREQ});return _finalize(out,s)

    ref,src,ves=_source(root/SOURCE_CACHE);lad=_load_path(root/LAD,ref);prox=_load_path(root/PROX,ref);c6=_load_path(root/C6,ref);c7=_load_path(root/C7,ref)
    prox,lad,junction,gap=_orient_pair_at_junction(prox,lad);c6=_orient_to_point(c6,junction);c7=_orient_to_point(c7,junction)
    backbone=_join_paths([lad[::-1].copy(),c6])
    bpts,barc=_resample(backbone,.20);btree=cKDTree(bpts.astype(np.float32));pd.DataFrame(backbone,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"label_neutral_backbone.csv",index=False)
    lad_len=float(_arc(lad)[-1]);control_arc=lad_len+C67_SPLIT_ARC_MM

    bv=np.asarray(_sample_array(ref,ves,bpts,cval=0.0),float);good=bv[np.isfinite(bv)&(bv>0)]
    if not len(good): raise RuntimeError("No positive vesselness sampled on backbone")
    vthr=max(1e-5,0.20*float(np.percentile(good,20)));vnorm=max(1e-4,float(np.median(good)))
    calibration={"backbone_vesselness_p20":float(np.percentile(good,20)),"backbone_vesselness_median":float(np.median(good)),"vesselness_threshold":vthr,"vesselness_norm":vnorm}
    _write_json(out/"vesselness_calibration.json",calibration)

    attr_c,hc,pc,qcc=_evaluate_seed(ref,src,ves,backbone,btree,control_arc,vthr,vnorm,"control")
    attr_c.to_csv(out/"control_search_attrition.csv",index=False);control_rows=[];c7_post=_slice_arc(c7,C67_SPLIT_ARC_MM,min(float(_arc(c7)[-1]),C67_SPLIT_ARC_MM+SEARCH_MM))
    for _,r in hc.iterrows():
        hid=str(r.hypothesis_id);ov=_overlap_metrics(pc[hid],c7_post);d={**r.to_dict(),**{f"c7_{k}":v for k,v in ov.items()}};d["control_gate_pass"]=_control_gate(r.to_dict(),ov);control_rows.append(d)
    cdf=pd.DataFrame(control_rows);cdf.to_csv(out/"C7_control_candidates.csv",index=False)
    passing=cdf[cdf.control_gate_pass==True].sort_values(["accepted_arc_mm","c7_median_distance_mm"],ascending=[False,True]) if len(cdf) else cdf
    control_ok=bool(len(passing));control_best=passing.iloc[0].to_dict() if control_ok else (cdf.sort_values("c7_median_distance_mm").iloc[0].to_dict() if len(cdf) else {})
    if control_best:
        hid=str(control_best["hypothesis_id"]);pd.DataFrame(pc[hid],columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"C7_control_recovered_path.csv",index=False);qcc[hid].to_csv(out/"C7_control_dense_qc.csv",index=False)
    _write_json(out/"C7_control.json",{"pass":control_ok,"seed_arc_mm":control_arc,"best":control_best})
    if not control_ok:
        s={"status":STATUS_CONTROL_FAIL,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"backbone_length_mm":float(barc[-1]),"calibration":calibration,"control":{"pass":False,"best":control_best}}
        _write_json(out/"summary.json",s);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_CONTROL_FAIL});return _finalize(out,s)

    seed_arcs=np.arange(SEED_END_MARGIN_MM,max(SEED_END_MARGIN_MM,float(barc[-1])-SEED_END_MARGIN_MM)+1e-9,SEED_SPACING_MM)
    seed_arcs=[float(a) for a in seed_arcs if abs(a-control_arc)>CONTROL_EXCLUSION_HALF_WIDTH_MM]
    all_attr=[];all_rows=[];all_paths={};all_qc={}
    for si,a in enumerate(seed_arcs):
        attr,h,p,qc=_evaluate_seed(ref,src,ves,backbone,btree,a,vthr,vnorm,f"s{si:02d}")
        if len(attr): all_attr.append(attr)
        if len(h): all_rows.append(h)
        all_paths.update(p);all_qc.update(qc)
    attr_df=pd.concat(all_attr,ignore_index=True) if all_attr else pd.DataFrame();hyp=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    attr_df.to_csv(out/"blind_branch_search_attrition.csv",index=False);hyp.to_csv(out/"blind_branch_hypotheses.csv",index=False)
    valid=hyp[hyp.branch_gate_pass==True].copy() if len(hyp) else hyp
    clustered=_cluster_candidates(valid);clustered.to_csv(out/"additional_branch_candidates.csv",index=False)
    top=[]
    for rank,(_,r) in enumerate(clustered.head(5).iterrows(),start=1):
        hid=str(r.hypothesis_id);path=all_paths[hid];pd.DataFrame(path,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/f"additional_branch_{rank:02d}_{hid}.csv",index=False);all_qc[hid].to_csv(out/f"additional_branch_{rank:02d}_{hid}_dense_qc.csv",index=False);top.append(r.to_dict())

    fig,axes=plt.subplots(1,3,figsize=(15,4.8));pairs=[(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]
    geom=[(backbone,"label-neutral backbone"),(c7_post,"known C7 control branch")]
    for i,(_,r) in enumerate(clustered.head(5).iterrows()): geom.append((all_paths[str(r.hypothesis_id)],f"additional {i+1}"))
    for ax,(i,j,xl,yl) in zip(axes,pairs):
        for p0,lab in geom: ax.plot(p0[:,i],p0[:,j],lw=1.5,label=lab)
        ax.set_xlabel(xl);ax.set_ylabel(yl);ax.set_aspect("equal",adjustable="box")
    axes[0].legend(fontsize=7);fig.suptitle("Label-neutral left-coronary backbone and source-supported side branches");plt.tight_layout();plt.savefig(out/"01_backbone_branch_geometry.png",dpi=180);plt.close()

    plt.figure(figsize=(10,4.5))
    if len(hyp):
        best_by_seed=hyp.sort_values(["branch_gate_pass","accepted_arc_mm","beam_score"],ascending=[False,False,False]).groupby("seed_arc_mm",as_index=False).first()
        plt.scatter(best_by_seed.seed_arc_mm,best_by_seed.accepted_arc_mm,c=best_by_seed.branch_gate_pass.astype(int),s=28)
    plt.axvline(control_arc,ls="--",lw=.8,label="known C6/C7 split");plt.xlabel("backbone arc (mm)");plt.ylabel("best accepted branch arc (mm)");plt.title("Blind branch scan along validated left-coronary backbone");plt.legend();plt.tight_layout();plt.savefig(out/"02_branch_scan_overview.png",dpi=180);plt.close()

    if len(clustered):
        r=clustered.iloc[0];hid=str(r.hypothesis_id);path=all_paths[hid];p,q=_resample(path,.4);tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);picks=np.linspace(0,len(p)-1,min(12,len(p))).astype(int);cols=4;nr=int(math.ceil(len(picks)/cols));fig,axes=plt.subplots(nr,cols,figsize=(14,3.6*nr));axes=np.atleast_1d(axes).ravel()
        for ax in axes[len(picks):]:ax.axis("off")
        for ax,ix in zip(axes,picks):
            im,g=_plane(ref,src,p[ix],tt[ix]);ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]);ax.scatter([0],[0],s=14);ax.set_title(f"{q[ix]:.1f} mm");ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle("Best additional label-neutral branch: orthogonal source-CCTA QC");plt.tight_layout();plt.savefig(out/"03_best_additional_branch_orthogonal_qc.png",dpi=180);plt.close()

    status=STATUS_FOUND if len(clustered) else STATUS_NONE
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,
             "continuity_status":cont.get("status"),"junction_endpoint_gap_mm":gap,"backbone_length_mm":float(barc[-1]),"backbone_junction_arc_mm":lad_len,
             "known_C67_split_backbone_arc_mm":control_arc,"calibration":calibration,"control":{"pass":control_ok,"best":control_best},
             "blind_seed_count":len(seed_arcs),"blind_hypothesis_count":int(len(hyp)),"valid_branch_hypothesis_count":int(len(valid)),"clustered_additional_branch_count":int(len(clustered)),
             "top_additional_branches":top,
             "scientific_boundary":"Label-neutral source-CCTA branch discovery only. The C7 same-method control must pass. No clinical LAD/LM/LCX/OM label is assigned and frozen anatomy is not modified."}
    _write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    return _finalize(out,summary)
