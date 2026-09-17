from __future__ import annotations

"""Source-CCTA recovery of an unknown parent vessel from a validated two-daughter junction.

The target junction is the frozen LAD endpoint plus the dense-QC 20-mm continuation. The
experiment first verifies that the 20-mm continuation is geometrically the established C6/C7
common trunk. It then uses the LAD and common-trunk directions as two daughters and derives a
parent-search direction from their opposite angular bisector. Discovery is source-led: aortic
geometry is never used in proposal acceptance or ranking.

A same-mechanism positive control is performed at the established C6/C7 split: the two known
post-split daughters must recover their already-established common parent trunk. The target
search is run only if that control passes. Known daughter paths are excluded after the first
2 mm so the beam cannot simply backtrack onto LAD/common-trunk anatomy.

Research use only. A positive target is a source-supported parent-continuation hypothesis, not
a clinical LM label, and the frozen master is never modified.
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
ALGORITHM = "left-coronary-bifurcation-parent-recovery-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Bifurcation_Parent_Recovery_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
PROX = Path("Left_Proximal_Trunk_Continuation_QC_v1/accepted_proximal_trunk_continuation_candidate.csv")
PROX_SUMMARY = Path("Left_Proximal_Trunk_Continuation_QC_v1/summary.json")
C6 = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7 = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
STRUCTURAL = Path("LCX_Structural_Identity_Adjudication_v1/structural_identity_decision.json")
RCA = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
AORTA = Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz")

C67_SPLIT_ARC_MM = 20.25
TANGENT_ARC_MM = 4.0
CONTROL_SEARCH_MM = 12.0
TARGET_SEARCH_MM = 30.0
FIELD_MARGIN_MM = 34.0
STEP_MM = 0.40
BEAM_WIDTH = 72
INITIAL_DIRS = 7
PREVIEW_MM = 1.2
KNOWN_EXCLUSION_AFTER_MM = 2.0
KNOWN_EXCLUSION_MM = 0.90
PLANE_STEP_MM = 0.40
MIN_ACCEPTED_MM = 6.0
MIN_PLANE_PASS_FRACTION = 0.80
SUSTAINED_FAIL_N = 3
AORTA_REACH_MM = 2.0
MEANINGFUL_AORTA_PROGRESS_MM = 4.0
RCA_ENDPOINT_SEPARATION_MM = 8.0

STATUS_PREREQ = "BIFURCATION_PARENT_PREREQUISITE_FAILED"
STATUS_CONTROL_FAIL = "BIFURCATION_PARENT_RECOVERY_CONTROL_FAILED"
STATUS_NEG = "BIFURCATION_PARENT_NO_VALID_SOURCE_CONTINUATION"
STATUS_AWAY = "BIFURCATION_PARENT_SOURCE_CONTINUATION_QC_POSITIVE_NO_AORTIC_PROGRESS"
STATUS_TOWARD = "BIFURCATION_PARENT_SOURCE_CONTINUATION_QC_POSITIVE_TOWARD_AORTA"
STATUS_REACH_RCA = "BIFURCATION_PARENT_REACHES_AORTA_BUT_RCA_ASSOCIATED"
STATUS_REACH = "BIFURCATION_PARENT_REACHES_AORTA_REQUIRES_VISUAL_QC"


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
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    ref = sitk.GetImageFromArray(np.asarray(arr))
    sp_zyx = np.asarray(meta["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp_zyx[::-1]))
    ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref, arr, sp_zyx


def _xyz_to_zyx(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float)); o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float); D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return (((pts - o) @ np.linalg.inv(D).T) / sp)[:, ::-1]


def _zyx_to_xyz(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float)); q = pts[:, ::-1]; o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float); D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (q * sp) @ D.T


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    for cols in (("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")):
        if all(c in d.columns for c in cols): return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    raise ValueError(f"unrecognized coordinates in {path}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _interp(p, q):
    a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=.20):
    p = np.asarray(p, float); a = _arc(p)
    if len(p) < 2 or a[-1] <= 0: return p.copy(), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6: q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _slice_arc(p, lo=0.0, hi=None, step=.20):
    a = _arc(p); hi = float(a[-1] if hi is None else min(hi, a[-1])); lo = float(max(0.0, min(lo, hi)))
    q = np.arange(lo, hi + 1e-9, step)
    if len(q) == 0 or q[-1] < hi - 1e-6: q = np.r_[q, hi]
    return _interp(np.asarray(p, float), q)


def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t); axes = np.eye(3); seed = axes[np.argmin(np.abs(axes @ t))]; u = _unit(np.cross(t, seed)); v = _unit(np.cross(t, u)); return u, v


def _cone_dirs(t, max_deg=55):
    t = _unit(t); u, v = _orth_basis(t); out = [t]
    for deg in (10, 20, 30, 40, 50):
        if deg > max_deg: continue
        ang = np.deg2rad(deg)
        for phi in np.linspace(0, 2*np.pi, 12, endpoint=False): out.append(_unit(np.cos(ang)*t + np.sin(ang)*(np.cos(phi)*u + np.sin(phi)*v)))
    return out


def _sample(ref, arr, pts, cval=-1024.0):
    return map_coordinates(np.asarray(arr), _xyz_to_zyx(ref, pts).T, order=1, mode="constant", cval=cval)


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path))); same = (im.GetSize() == ref.GetSize() and np.allclose(im.GetSpacing(), ref.GetSpacing()) and np.allclose(im.GetOrigin(), ref.GetOrigin()) and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same: im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def _surface_tree(mask, ref, stride=2):
    surf = mask & ~ndi.binary_erosion(mask, iterations=1, border_value=0); z = np.argwhere(surf)[::stride]; pts = _zyx_to_xyz(ref, z.astype(float)); return cKDTree(pts.astype(np.float32)), pts


def _frangi_3d(vol, spacing, scales=(.55, .80, 1.10, 1.45)):
    x = np.clip(np.asarray(vol, np.float32), 80, 1000); x = (x - 80) / 920.0; spacing = np.asarray(spacing, float); best = np.zeros_like(x, np.float32)
    for sm in scales:
        sig = np.maximum(sm / spacing, .55); n = sm * sm
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * n / spacing[0]**2; hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * n / spacing[1]**2; hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * n / spacing[2]**2
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * n / (spacing[0]*spacing[1]); hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * n / (spacing[0]*spacing[2]); hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * n / (spacing[1]*spacing[2])
        H = np.empty(x.shape + (3,3), np.float32); H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx; H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals = np.linalg.eigvalsh(H); vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1); l1,l2,l3 = vals[...,0], vals[...,1], vals[...,2]; eps = 1e-8
        ra = np.abs(l2)/(np.abs(l3)+eps); rb = np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); ss = np.sqrt(l1*l1+l2*l2+l3*l3); nz = ss[ss>0]; c = max(float(np.percentile(nz,90))*.45 if nz.size else .05, 1e-4)
        vv = (1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(ss*ss)/(2*c*c))); vv[(l2>=0)|(l3>=0)] = 0; best = np.maximum(best, np.nan_to_num(vv).astype(np.float32))
    return best


def _field(ref, src, spacing, paths, margin_mm=FIELD_MARGIN_MM):
    pts = np.vstack([_resample(p, .35)[0] for p in paths if len(p)]); z = _xyz_to_zyx(ref, pts); mv = np.asarray([margin_mm]*3) / spacing
    lo = np.maximum(np.floor(z.min(0)-mv).astype(int), 0); hi = np.minimum(np.ceil(z.max(0)+mv).astype(int)+1, np.asarray(src.shape)); sl = tuple(slice(lo[k], hi[k]) for k in range(3)); roi = np.asarray(src[sl]); vessel = _frangi_3d(roi, spacing)
    kz = z - lo[None,:]; good = np.all((kz>=0)&(kz<np.asarray(roi.shape)[None,:]), axis=1); kv = map_coordinates(vessel, kz[good].T, order=1, mode="nearest")
    thr = max(.003, min(.10, .25*float(np.percentile(kv,20)))) if len(kv) else .01; norm = max(float(np.median(kv))*1.5, .02) if len(kv) else .02
    return {"lo":lo, "roi":roi, "v":vessel, "thr":thr, "norm":norm, "shape":list(roi.shape)}


def _sample_field(ref, F, p):
    z = _xyz_to_zyx(ref, [p])[0] - F["lo"]; shape = np.asarray(F["roi"].shape)
    if np.any(z<1) or np.any(z>shape-2): return None
    co = z[:,None]; hu = float(map_coordinates(F["roi"], co, order=1, mode="nearest")[0]); vv = float(map_coordinates(F["v"], co, order=1, mode="nearest")[0]); return hu, vv


def _path_overlap(query, ref):
    q, _ = _resample(query, .25); r, _ = _resample(ref, .25); tree = cKDTree(r); d, ix = tree.query(q)
    tq = np.gradient(q, axis=0); tq /= np.maximum(np.linalg.norm(tq,axis=1,keepdims=True),1e-9); tr = np.gradient(r, axis=0); tr /= np.maximum(np.linalg.norm(tr,axis=1,keepdims=True),1e-9); al = np.abs(np.sum(tq*tr[ix], axis=1))
    return {"min_distance_mm":float(np.min(d)),"median_distance_mm":float(np.median(d)),"fraction_within_1_5mm":float(np.mean(d<=1.5)),"fraction_within_2mm":float(np.mean(d<=2.0)),"median_tangent_alignment_within_2mm":float(np.median(al[d<=2.0])) if np.any(d<=2.0) else 0.0}


def _orient_pair_at_junction(a, b):
    choices=[]
    for ia, pa in enumerate((a[0],a[-1])):
        for ib, pb in enumerate((b[0],b[-1])): choices.append((float(np.linalg.norm(pa-pb)), ia, ib))
    _, ia, ib = min(choices)
    if ia==1: a=a[::-1].copy()
    if ib==1: b=b[::-1].copy()
    junction=(a[0]+b[0])/2.0
    return a,b,junction,float(np.linalg.norm(a[0]-b[0]))


def _tangent_from_start(p, arc_mm=TANGENT_ARC_MM):
    p,_ = _resample(p,.20); a=_arc(p); ix=int(np.argmin(np.abs(a-min(arc_mm,a[-1])))); return _unit(p[ix]-p[0])


def _parent_direction(d1, d2):
    d1,d2=_unit(d1),_unit(d2); s=d1+d2
    if np.linalg.norm(s)<.25: s=d1
    return _unit(-s)


def _preview(ref,F,start,d):
    vals=[]
    for s in np.arange(.4,PREVIEW_MM+1e-9,.4):
        x=_sample_field(ref,F,start+s*d)
        if x is None: return -1e9
        hu,vv=x
        if not (100<=hu<=1400) or vv<.35*F["thr"]: return -1e9
        vals.append(1.8*min(1.,vv/F["norm"])+.3*min(1.,max(0.,hu-100)/500.))
    return float(np.mean(vals)) if vals else -1e9


def _initial_dirs(ref,F,start,parent_dir):
    scored=[]
    for d in _cone_dirs(parent_dir,55):
        if np.dot(d,parent_dir)<math.cos(math.radians(60)): continue
        sc=_preview(ref,F,start,d)
        if sc>-1e8: scored.append((sc,d))
    scored.sort(key=lambda x:x[0], reverse=True); chosen=[]
    for sc,d in scored:
        if all(abs(float(np.dot(d,e)))<.985 for _,e in chosen): chosen.append((sc,d))
        if len(chosen)>=INITIAL_DIRS: break
    return chosen


def _beam(ref,F,start,initial_dir,known_tree,max_mm,step=STEP_MM,beam_width=BEAM_WIDTH):
    states=[(0.,[np.asarray(start,float)],_unit(initial_dir))]; checkpoints=[]; rows=[]; best=states[0]; nsteps=int(math.ceil(max_mm/step))
    for si in range(nsteps):
        nxt=[]; cnt={"step":si,"arc_budget_mm":float((si+1)*step),"states_in":len(states),"proposals":0,"reject_turn":0,"reject_loop":0,"reject_known_daughter":0,"reject_outside":0,"reject_hu":0,"reject_vesselness":0,"accepted":0,"kept":0}
        for score,pts,t in states:
            for d in _cone_dirs(t,55):
                cnt["proposals"]+=1
                if np.dot(d,t)<math.cos(math.radians(65)): cnt["reject_turn"]+=1; continue
                q=pts[-1]+step*d
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-q,axis=1))<.60*step: cnt["reject_loop"]+=1; continue
                arc_new=(len(pts))*step
                if arc_new>KNOWN_EXCLUSION_AFTER_MM and float(known_tree.query(q)[0])<KNOWN_EXCLUSION_MM: cnt["reject_known_daughter"]+=1; continue
                sv=_sample_field(ref,F,q)
                if sv is None: cnt["reject_outside"]+=1; continue
                hu,vv=sv
                if not (100<=hu<=1400): cnt["reject_hu"]+=1; continue
                if vv<.35*F["thr"]: cnt["reject_vesselness"]+=1; continue
                vn=min(1.,vv/F["norm"]); align=max(0.,float(np.dot(d,t))); ns=score+1.8*vn+.30*min(1.,max(0.,hu-100)/500.)+.45*align; nt=_unit(.70*t+.30*d); st=(ns,pts+[q],nt); nxt.append(st); cnt["accepted"]+=1
                if len(st[1])>len(best[1]) or (len(st[1])==len(best[1]) and ns>best[0]): best=st
        if not nxt: rows.append(cnt); break
        nxt.sort(key=lambda x:x[0], reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.30).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=beam_width: break
        states=keep; cnt["kept"]=len(states); rows.append(cnt)
        if (si+1)%12==0 or si==nsteps-1: checkpoints.extend(states[:3])
    finals=[]
    for st in list(states)+checkpoints+[best]:
        ep=np.asarray(st[1][-1])
        if all(np.linalg.norm(ep-np.asarray(x[1][-1]))>1.0 for x in finals): finals.append(st)
    finals.sort(key=lambda s:(len(s[1]),s[0]), reverse=True)
    return finals[:10], pd.DataFrame(rows)


def _plane(ref,src,c,t,half=5.5,step=.20):
    t=_unit(t);u,v=_orth_basis(t);q=np.arange(-half,half+1e-9,step);yy,xx=np.meshgrid(q,q,indexing="ij");P=c+xx[...,None]*u+yy[...,None]*v;return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q


def _component_metrics(im,q):
    iy=ix=int(np.argmin(np.abs(q))); center=float(im[iy,ix]); yy,xx=np.meshgrid(q,q,indexing="ij"); rr=np.sqrt(xx*xx+yy*yy); thr=max(220.,min(500.,.55*center)); bw=(im>=thr)&(rr<=3.5); lab,_=ndi.label(bw,np.ones((3,3),int)); labels=[]
    if lab[iy,ix]>0: labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt(q[pts[:,1]]**2+q[pts[:,0]]**2);j=int(np.argmin(dist))
            if dist[j]<=1.0: labels=[int(lab[tuple(pts[j])])]
    if not labels: return {"center_hu":center,"threshold_hu":thr,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
    mask=lab==labels[0];pts=np.argwhere(mask);xs=q[pts[:,1]];ys=q[pts[:,0]];cx=float(np.mean(xs));cy=float(np.mean(ys));off=float(np.hypot(cx,cy));area=float(len(pts)*.04);rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        C=np.cov(np.column_stack([xs,ys]).T);ev=np.linalg.eigvalsh(C);axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else: axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0); contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan
    return {"center_hu":center,"threshold_hu":thr,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}


def _plane_pass(m): return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2 and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)


def _dense_qc(ref,src,path):
    p,q=_resample(path,PLANE_STEP_MM); tt=np.gradient(p,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9); rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m.update(index=i,arc_mm=float(a),plane_pass=_plane_pass(m));rows.append(m)
    return p,q,pd.DataFrame(rows)


def _truncate(df):
    ps=df.plane_pass.astype(bool).to_numpy(); cut=len(df)-1; first=None
    for i in range(len(ps)-SUSTAINED_FAIL_N+1):
        if not np.any(ps[i:i+SUSTAINED_FAIL_N]): first=i; cut=max(0,i-1); break
    d=df.iloc[:cut+1]; arc=float(d.arc_mm.iloc[-1]) if len(d) else 0.; frac=float(d.plane_pass.mean()) if len(d) else 0.
    return {"first_sustained_failure_index":first,"accepted_last_index":int(cut),"accepted_arc_mm":arc,"accepted_plane_pass_fraction":frac,"accepted":bool(arc>=MIN_ACCEPTED_MM and frac>=MIN_PLANE_PASS_FRACTION),"covers_full_path":bool(cut==len(df)-1)}


def _nearest_metrics(query, ref):
    q,_=_resample(query,.25); r,_=_resample(ref,.25); d,_=cKDTree(r).query(q); return {"median_distance_mm":float(np.median(d)),"p90_distance_mm":float(np.percentile(d,90)),"endpoint_distance_mm":float(d[-1]),"fraction_within_2mm":float(np.mean(d<=2.0))}


def _run_search(ref,src,spacing,start,parent_dir,daughters,calibration,max_mm):
    F=_field(ref,src,spacing,[calibration]+daughters,FIELD_MARGIN_MM); kp=np.vstack([_resample(p,.20)[0] for p in daughters]); known_tree=cKDTree(kp.astype(np.float32)); dirs=_initial_dirs(ref,F,start,parent_dir); all_final=[]; attrs=[]
    for di,(preview,d) in enumerate(dirs):
        finals,attr=_beam(ref,F,start,d,known_tree,max_mm); attr["dir_index"]=di; attr["preview_score"]=preview; attrs.append(attr)
        for fi,st in enumerate(finals): all_final.append((di,fi,preview,st))
    rows=[]; paths={}; qcdfs={}
    for di,fi,preview,st in all_final:
        raw=np.asarray(st[1],float); p,q,qc=_dense_qc(ref,src,raw); tr=_truncate(qc); hid=f"d{di}_f{fi}"; paths[hid]=p; qcdfs[hid]=qc; rows.append({"hypothesis_id":hid,"dir_index":di,"final_index":fi,"preview_score":preview,"beam_score":float(st[0]),"path_arc_mm":float(q[-1]),**tr})
    return F, dirs, (pd.concat(attrs,ignore_index=True) if attrs else pd.DataFrame()), pd.DataFrame(rows), paths, qcdfs


def synthetic_parent_recovery_self_test():
    d1=_unit([1.,1.,0.]); d2=_unit([1.,-1.,0.]); p=_parent_direction(d1,d2); assert np.dot(p,[-1.,0.,0.])>.99
    df=pd.DataFrame({"arc_mm":np.arange(10)*.4,"plane_pass":[1,1,1,1,0,1,1,0,0,0]}); r=_truncate(df); assert r["first_sustained_failure_index"]==7 and r["accepted_last_index"]==6
    return {"ok":True,"parent_direction":p.tolist()}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD,root/PROX,root/PROX_SUMMARY,root/C6,root/C7,root/STRUCTURAL,root/RCA,root/AORTA]
    for p in required: _req(p)
    master=_read_json(root/MASTER); psum=_read_json(root/PROX_SUMMARY); structural=_read_json(root/STRUCTURAL)
    prereq=bool(master.get("status")=="CORONARY_ANATOMY_BASELINE_V2_FROZEN" and str(psum.get("status","")).startswith("PROXIMAL_TRUNK_CONTINUATION_QC_POSITIVE") and structural.get("all_predeclared_structural_gates_pass") is True)
    if not prereq:
        s={"status":STATUS_PREREQ,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False}; _write_json(out/"summary.json",s); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_PREREQ}); return _finalize(out,s)

    ref,src,spacing=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD,ref); prox=_load_path(root/PROX,ref); c6=_load_path(root/C6,ref); c7=_load_path(root/C7,ref); rca=_load_path(root/RCA,ref)
    prox,lad,junction,junc_gap=_orient_pair_at_junction(prox,lad); c6 = c6 if np.linalg.norm(c6[0]-junction)<=np.linalg.norm(c6[-1]-junction) else c6[::-1].copy(); c7 = c7 if np.linalg.norm(c7[0]-junction)<=np.linalg.norm(c7[-1]-junction) else c7[::-1].copy(); common=_slice_arc(c6,0,C67_SPLIT_ARC_MM)
    overlap=_path_overlap(prox,common); overlap_pass=bool(overlap["min_distance_mm"]<=1.0 and overlap["fraction_within_2mm"]>=.80 and overlap["median_tangent_alignment_within_2mm"]>=.80)
    lad_d=_tangent_from_start(lad); prox_d=_tangent_from_start(prox); target_parent=_parent_direction(lad_d,prox_d); daughter_angle=float(np.degrees(np.arccos(np.clip(np.dot(lad_d,prox_d),-1,1))))

    c6_split=_interp(c6,[C67_SPLIT_ARC_MM])[0]; c7_split=_interp(c7,[C67_SPLIT_ARC_MM])[0]; control_j=(c6_split+c7_split)/2
    c6_post=_slice_arc(c6,C67_SPLIT_ARC_MM,min(_arc(c6)[-1],C67_SPLIT_ARC_MM+12)); c7_post=_slice_arc(c7,C67_SPLIT_ARC_MM,min(_arc(c7)[-1],C67_SPLIT_ARC_MM+12))
    d6=_unit(_interp(c6,[min(_arc(c6)[-1],C67_SPLIT_ARC_MM+TANGENT_ARC_MM)])[0]-c6_split); d7=_unit(_interp(c7,[min(_arc(c7)[-1],C67_SPLIT_ARC_MM+TANGENT_ARC_MM)])[0]-c7_split); control_parent=_parent_direction(d6,d7); known_parent=_slice_arc(c6,max(0,C67_SPLIT_ARC_MM-CONTROL_SEARCH_MM),C67_SPLIT_ARC_MM)[::-1].copy()
    _,_,attr_c,hc,pc,qcc=_run_search(ref,src,spacing,control_j,control_parent,[c6_post,c7_post],known_parent,CONTROL_SEARCH_MM); attr_c.to_csv(out/"control_search_attrition.csv",index=False); hc.to_csv(out/"control_hypotheses.csv",index=False)
    control_candidates=[]
    for _,row in hc.iterrows():
        hid=str(row.hypothesis_id); tr=pc[hid][:int(row.accepted_last_index)+1]; gm=_nearest_metrics(tr,known_parent); control_candidates.append({**row.to_dict(),**gm})
    cdf=pd.DataFrame(control_candidates); cdf["control_geometry_pass"]=(cdf.accepted==True)&(cdf.median_distance_mm<=1.5)&(cdf.p90_distance_mm<=2.5)&(cdf.endpoint_distance_mm<=2.5) if len(cdf) else False; cdf.to_csv(out/"control_parent_recovery_metrics.csv",index=False)
    passing=cdf[cdf.control_geometry_pass==True].sort_values(["accepted_arc_mm","median_distance_mm"],ascending=[False,True]) if len(cdf) else cdf; control_ok=bool(len(passing)>0); control_best=passing.iloc[0].to_dict() if control_ok else (cdf.sort_values("median_distance_mm").iloc[0].to_dict() if len(cdf) else {})
    if control_best:
        hid=str(control_best["hypothesis_id"]); pd.DataFrame(pc[hid],columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"control_recovered_parent_path.csv",index=False); qcc[hid].to_csv(out/"control_dense_qc.csv",index=False)

    junction_info={"junction_lps_mm":junction.tolist(),"junction_endpoint_gap_mm":junc_gap,"prox_common_overlap":overlap,"prox_common_overlap_pass":overlap_pass,"lad_daughter_tangent":lad_d.tolist(),"common_trunk_daughter_tangent":prox_d.tolist(),"daughter_angle_deg":daughter_angle,"derived_parent_direction":target_parent.tolist(),"control_parent_direction":control_parent.tolist(),"control_pass":control_ok}; _write_json(out/"junction_geometry.json",junction_info)
    if not (overlap_pass and control_ok):
        st=STATUS_CONTROL_FAIL if overlap_pass else STATUS_PREREQ; s={"status":st,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"junction":junction_info,"control_best":control_best}; _write_json(out/"summary.json",s); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":st}); return _finalize(out,s)

    _,_,attr_t,ht,pt,qct=_run_search(ref,src,spacing,junction,target_parent,[lad,prox],np.vstack([_slice_arc(lad,0,8),_slice_arc(prox,0,8)]),TARGET_SEARCH_MM); attr_t.to_csv(out/"target_search_attrition.csv",index=False); ht.to_csv(out/"target_parent_hypotheses.csv",index=False)
    accepted=ht[ht.accepted==True].copy() if len(ht) else ht
    if len(accepted): accepted=accepted.sort_values(["accepted_arc_mm","accepted_plane_pass_fraction","beam_score"],ascending=[False,False,False]); best=accepted.iloc[0]
    elif len(ht): best=ht.sort_values(["accepted_arc_mm","beam_score"],ascending=[False,False]).iloc[0]
    else: best=None

    aorta=_resample_mask(root/AORTA,ref); aorta_tree,_=_surface_tree(aorta,ref); status=STATUS_NEG; posthoc={}; bestdict={}
    if best is not None:
        hid=str(best.hypothesis_id); raw=pt[hid]; last=int(best.accepted_last_index); path=raw[:last+1] if last>=0 else raw[:1]; pd.DataFrame(path,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"best_target_parent_path.csv",index=False); qct[hid].to_csv(out/"best_target_parent_dense_qc.csv",index=False)
        p,q=_resample(path,.20); ad=np.asarray(aorta_tree.query(p)[0],float); pd.DataFrame({"arc_mm":q,"aorta_distance_mm":ad}).to_csv(out/"best_target_parent_aorta_profile.csv",index=False); start_ad=float(ad[0]); min_ad=float(np.min(ad)); end_ad=float(ad[-1]); progress=start_ad-min_ad; rca_p,_=_resample(rca,.25); rca_sep=float(cKDTree(rca_p).query(p[-1])[0]); reaches=bool(best.accepted and min_ad<=AORTA_REACH_MM); toward=bool(best.accepted and progress>=MEANINGFUL_AORTA_PROGRESS_MM)
        if reaches and rca_sep<RCA_ENDPOINT_SEPARATION_MM: status=STATUS_REACH_RCA
        elif reaches: status=STATUS_REACH
        elif toward: status=STATUS_TOWARD
        elif bool(best.accepted): status=STATUS_AWAY
        posthoc={"start_aorta_distance_mm":start_ad,"minimum_aorta_distance_mm":min_ad,"minimum_aorta_arc_mm":float(q[int(np.argmin(ad))]),"end_aorta_distance_mm":end_ad,"maximum_aorta_progress_mm":progress,"endpoint_distance_to_known_RCA_mm":rca_sep,"reaches_aorta":reaches,"meaningful_progress_toward_aorta":toward}; bestdict=best.to_dict()
        plt.figure(figsize=(8,4.5));plt.plot(q,ad);plt.axhline(AORTA_REACH_MM,ls="--",lw=.8);plt.xlabel("target parent arc (mm)");plt.ylabel("distance to nearest aortic surface (mm)");plt.title("Post-hoc aortic distance: bifurcation-derived parent candidate");plt.tight_layout();plt.savefig(out/"01_target_parent_aorta_profile.png",dpi=180);plt.close()
        bp,bq=_resample(path,.4);tt=np.gradient(bp,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);picks=np.linspace(0,len(bp)-1,min(12,len(bp))).astype(int);cols=4;nr=int(math.ceil(len(picks)/cols));fig,axes=plt.subplots(nr,cols,figsize=(14,3.6*nr));axes=np.atleast_1d(axes).ravel()
        for ax in axes[len(picks):]:ax.axis("off")
        for ax,ix in zip(axes,picks): im,g=_plane(ref,src,bp[ix],tt[ix]);ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]);ax.scatter([0],[0],s=14);ax.set_title(f"{bq[ix]:.1f} mm");ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle("Bifurcation-derived parent candidate: orthogonal source-CCTA QC");plt.tight_layout();plt.savefig(out/"02_target_parent_orthogonal_qc.png",dpi=180);plt.close()

    fig,axes=plt.subplots(1,3,figsize=(15,4.8));pairs=[(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]; geom=[(lad,"frozen LAD"),(prox,"validated common-trunk continuation"),(common,"C6/C7 common trunk")]
    if best is not None and bestdict: geom.append((pt[str(best.hypothesis_id)][:int(best.accepted_last_index)+1],"parent candidate"))
    for ax,(i,j,xl,yl) in zip(axes,pairs):
        for p0,lab in geom: ax.plot(p0[:,i],p0[:,j],lw=1.5,label=lab)
        ax.scatter([junction[i]],[junction[j]],s=25);ax.set_xlabel(xl);ax.set_ylabel(yl);ax.set_aspect("equal",adjustable="box")
    axes[0].legend(fontsize=7);fig.suptitle("LAD/common-trunk junction and bifurcation-derived parent search");plt.tight_layout();plt.savefig(out/"03_bifurcation_parent_geometry.png",dpi=180);plt.close()

    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"junction":junction_info,"control_best":control_best,"target_best":bestdict,"posthoc":posthoc,"candidate_role":"source-supported parent-continuation hypothesis derived from LAD plus validated common-trunk daughter geometry; not clinical LM unless separately established","scientific_boundary":"The C6/C7 parent-recovery control must pass before target interpretation. Discovery does not use aortic geometry. A positive target establishes only source-supported parent continuity; clinical LM identity and frozen anatomy remain unchanged."}; _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); return _finalize(out,summary)


def _finalize(out,summary):
    report=out/"OPENPLAQUE_BIFURCATION_PARENT_RECOVERY_REPORT.html"; report.write_text("<html><body><h1>OpenPlaque Bifurcation-Derived Parent Recovery v1</h1>"+f"<p><b>Status:</b> {summary.get('status')}</p>"+f"<p>Master: {summary.get('master_status')}; modified: {summary.get('master_modified')}</p>"+f"<p>Known C6/C7 parent control: {summary.get('junction',{}).get('control_pass')}</p>"+f"<p>Validated continuation overlaps C6/C7 common trunk: {summary.get('junction',{}).get('prox_common_overlap_pass')}</p>"+f"<p>Target accepted arc: {summary.get('target_best',{}).get('accepted_arc_mm')}</p>"+f"<p>Post-hoc aortic progress: {summary.get('posthoc',{}).get('maximum_aorta_progress_mm')}</p>"+"<p>Research-only parent hypothesis; no clinical LM label and frozen master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_BIFURCATION_PARENT_RECOVERY_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
