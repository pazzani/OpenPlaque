from __future__ import annotations

"""Adjudicate whether frozen LAD and the validated proximal/common-trunk segment form one through-vessel.

This is a topology/identity diagnostic. It does not search for a new vessel and does not modify
frozen anatomy. It directly tests the junction that was previously treated as a possible
bifurcation and checks whether the bifurcation-parent candidate merely re-entered known LAD.

A real branch control is taken from the established C6/C7 split. Dense orthogonal source-CCTA
QC is performed across the LAD-to-proximal-continuation junction. Clinical LM/LCX/LAD labels
remain unchanged regardless of result.
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
ALGORITHM = "left-coronary-through-vessel-continuity-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Through_Vessel_Continuity_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
PROX = Path("Left_Proximal_Trunk_Continuation_QC_v1/accepted_proximal_trunk_continuation_candidate.csv")
C6 = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7 = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
PARENT_SUMMARY = Path("Left_Coronary_Bifurcation_Parent_Recovery_v1/summary.json")
PARENT_PATH = Path("Left_Coronary_Bifurcation_Parent_Recovery_v1/best_target_parent_path.csv")

C67_SPLIT_ARC_MM = 20.25
TANGENT_ARC_MM = 4.0
JUNCTION_QC_HALF_LENGTH_MM = 8.0
PLANE_STEP_MM = 0.40
SUSTAINED_FAIL_N = 3

# Prospective continuity gates.
MAX_ENDPOINT_GAP_MM = 0.50
MAX_THROUGH_DEFLECTION_DEG = 25.0
MIN_PROX_COMMON_FRACTION_WITHIN_2MM = 0.80
MIN_PROX_COMMON_TANGENT_ALIGNMENT = 0.80
MIN_DENSE_QC_PASS_FRACTION = 0.90

# Established C6/C7 split should reproduce a parent-vs-side-branch pattern.
MAX_PARENT_ANGLE_FROM_INCOMING_DEG = 25.0
MIN_SIDE_BRANCH_ANGLE_FROM_INCOMING_DEG = 35.0
MIN_PARENT_SIDE_ANGLE_SEPARATION_DEG = 25.0

# Parent-candidate re-entry diagnostic.
REENTRY_BAND_MM = 2.0
REENTRY_NEAR_MM = 1.5
REENTRY_MIN_FRACTION = 0.20
REENTRY_MIN_CONTIGUOUS_SPAN_MM = 5.0
REENTRY_MIN_TANGENT_ALIGNMENT = 0.75

STATUS_PREREQ = "THROUGH_VESSEL_CONTINUITY_PREREQUISITE_FAILED"
STATUS_CONTROL = "THROUGH_VESSEL_CONTINUITY_C67_SPLIT_CONTROL_FAILED"
STATUS_POS_REENTRY = "LAD_COMMON_TRUNK_THROUGH_VESSEL_CONTINUITY_CONFIRMED_PARENT_CANDIDATE_REENTERS_LAD"
STATUS_POS = "LAD_COMMON_TRUNK_THROUGH_VESSEL_CONTINUITY_CONFIRMED"
STATUS_NEG = "LAD_COMMON_TRUNK_CONTINUITY_NOT_ESTABLISHED"


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
    return ref, arr


def _xyz_to_zyx(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float)
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
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _interp(p, q):
    a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:,k]) for k in range(3)])


def _resample(p, step=.25):
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
    v = np.asarray(v, float); n = float(np.linalg.norm(v))
    return v/n if n > 1e-9 else np.zeros_like(v)


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


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


def _tangent_from_start(p, arc_mm=TANGENT_ARC_MM):
    a = _arc(p); q = min(float(arc_mm), float(a[-1]))
    return _unit(_interp(p, [q])[0] - p[0])


def _sample(ref, arr, pts, cval=-1024.0):
    return map_coordinates(np.asarray(arr), _xyz_to_zyx(ref, pts).T, order=1, mode="constant", cval=cval)


def _orth_basis(t):
    t = _unit(t); axes = np.eye(3); seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed)); v = _unit(np.cross(t, u)); return u, v


def _plane(ref, src, c, t, half=5.5, step=.20):
    t = _unit(t); u,v = _orth_basis(t); q = np.arange(-half, half+1e-9, step)
    yy,xx = np.meshgrid(q,q,indexing="ij"); P = c + xx[...,None]*u + yy[...,None]*v
    return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)), q


def _component_metrics(im, q):
    iy=ix=int(np.argmin(np.abs(q))); center=float(im[iy,ix])
    yy,xx=np.meshgrid(q,q,indexing="ij"); rr=np.sqrt(xx*xx+yy*yy)
    thr=max(220.,min(500.,.55*center)); bw=(im>=thr)&(rr<=3.5)
    lab,_=ndi.label(bw,np.ones((3,3),int)); labels=[]
    if lab[iy,ix]>0: labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt(q[pts[:,1]]**2+q[pts[:,0]]**2); k=int(np.argmin(dist))
            if dist[k] <= 1.0: labels=[int(lab[tuple(pts[k])])]
    if not labels:
        return {"center_hu":center,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
    mask=lab==labels[0]; pts=np.argwhere(mask); xs=q[pts[:,1]]; ys=q[pts[:,0]]
    cx=float(np.mean(xs)); cy=float(np.mean(ys)); off=float(np.hypot(cx,cy)); area=float(len(pts)*.04); rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        C=np.cov(np.column_stack([xs,ys]).T); ev=np.linalg.eigvalsh(C); axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else: axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0); contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan
    return {"center_hu":center,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}


def _plane_pass(m):
    return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2 and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)


def _dense_qc(ref, src, path):
    p,q=_resample(path,PLANE_STEP_MM); tt=np.gradient(p,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9)
    rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,g=_plane(ref,src,c,t); m=_component_metrics(im,g); m.update(index=i,arc_mm=float(a),plane_pass=_plane_pass(m)); rows.append(m)
    return p,q,pd.DataFrame(rows)


def _sustained_failure(df):
    ps=df.plane_pass.astype(bool).to_numpy()
    for i in range(len(ps)-SUSTAINED_FAIL_N+1):
        if not np.any(ps[i:i+SUSTAINED_FAIL_N]): return i
    return None


def _overlap_metrics(query, ref, step=.25):
    q,qa=_resample(query,step); r,ra=_resample(ref,step); tree=cKDTree(r); d,ix=tree.query(q)
    tq=np.gradient(q,axis=0); tq/=np.maximum(np.linalg.norm(tq,axis=1,keepdims=True),1e-9)
    tr=np.gradient(r,axis=0); tr/=np.maximum(np.linalg.norm(tr,axis=1,keepdims=True),1e-9)
    al=np.abs(np.sum(tq*tr[ix],axis=1)); within=d<=REENTRY_BAND_MM
    best=cur=0.0
    for v in within:
        cur = cur + step if v else 0.0; best=max(best,cur)
    return {
        "min_distance_mm":float(np.min(d)), "median_distance_mm":float(np.median(d)), "p90_distance_mm":float(np.percentile(d,90)),
        "endpoint_distance_mm":float(d[-1]), "fraction_within_1_5mm":float(np.mean(d<=REENTRY_NEAR_MM)),
        "fraction_within_2mm":float(np.mean(within)), "max_contiguous_span_within_2mm":float(best),
        "median_tangent_alignment_within_2mm":float(np.median(al[within])) if np.any(within) else 0.0,
        "nearest_query_arc_mm":float(qa[int(np.argmin(d))]), "nearest_reference_arc_mm":float(ra[ix[int(np.argmin(d))]])
    }


def _reentry_gate(m):
    return bool(m["min_distance_mm"]<=REENTRY_NEAR_MM and m["fraction_within_2mm"]>=REENTRY_MIN_FRACTION and
                m["max_contiguous_span_within_2mm"]>=REENTRY_MIN_CONTIGUOUS_SPAN_MM and
                m["median_tangent_alignment_within_2mm"]>=REENTRY_MIN_TANGENT_ALIGNMENT)


def _branch_control(c6, c7):
    a6=_arc(c6); a7=_arc(c7); split=min(C67_SPLIT_ARC_MM,float(a6[-1]),float(a7[-1]))
    p6=_interp(c6,[split])[0]; p7=_interp(c7,[split])[0]; j=(p6+p7)/2.0
    incoming=_unit(j-_interp(c6,[max(0.0,split-TANGENT_ARC_MM)])[0])
    d6=_unit(_interp(c6,[min(float(a6[-1]),split+TANGENT_ARC_MM)])[0]-p6)
    d7=_unit(_interp(c7,[min(float(a7[-1]),split+TANGENT_ARC_MM)])[0]-p7)
    a6in=_angle(incoming,d6); a7in=_angle(incoming,d7); dangle=_angle(d6,d7)
    parent=min(a6in,a7in); side=max(a6in,a7in); sep=side-parent
    gate=bool(parent<=MAX_PARENT_ANGLE_FROM_INCOMING_DEG and side>=MIN_SIDE_BRANCH_ANGLE_FROM_INCOMING_DEG and sep>=MIN_PARENT_SIDE_ANGLE_SEPARATION_DEG)
    return {"split_lps_mm":j.tolist(),"C6_angle_from_incoming_deg":a6in,"C7_angle_from_incoming_deg":a7in,
            "daughter_angle_deg":dangle,"parent_like_angle_deg":parent,"side_branch_like_angle_deg":side,
            "parent_side_angle_separation_deg":sep,"control_gate_pass":gate}


def synthetic_continuity_self_test():
    assert (180.0-_angle([1,0,0],[-1,0,0])) < 1e-6
    m={"min_distance_mm":.5,"fraction_within_2mm":.8,"max_contiguous_span_within_2mm":8.0,"median_tangent_alignment_within_2mm":.9}
    assert _reentry_gate(m)
    return {"ok":True}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD,root/PROX,root/C6,root/C7,root/PARENT_SUMMARY,root/PARENT_PATH]
    for p in required: _req(p)
    master=_read_json(root/MASTER); prior=_read_json(root/PARENT_SUMMARY)
    prereq=bool(master.get("status")=="CORONARY_ANATOMY_BASELINE_V2_FROZEN" and prior.get("junction",{}).get("control_pass") is True and prior.get("target_best",{}).get("accepted") is True)
    if not prereq:
        s={"status":STATUS_PREREQ,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False}
        _write_json(out/"summary.json",s); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_PREREQ}); return _finalize(out,s)

    ref,src=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD,ref); prox=_load_path(root/PROX,ref); c6=_load_path(root/C6,ref); c7=_load_path(root/C7,ref); parent=_load_path(root/PARENT_PATH,ref)
    lad,prox,junction,gap=_orient_pair_at_junction(lad,prox)
    common=_slice_arc(c6,0,C67_SPLIT_ARC_MM)
    prox_common=_overlap_metrics(prox,common)
    prox_common_gate=bool(prox_common["min_distance_mm"]<=1.0 and prox_common["fraction_within_2mm"]>=MIN_PROX_COMMON_FRACTION_WITHIN_2MM and prox_common["median_tangent_alignment_within_2mm"]>=MIN_PROX_COMMON_TANGENT_ALIGNMENT)

    lad_t=_tangent_from_start(lad); prox_t=_tangent_from_start(prox); outgoing_angle=_angle(lad_t,prox_t); deflection=180.0-outgoing_angle
    ln=_slice_arc(lad,0,min(JUNCTION_QC_HALF_LENGTH_MM,_arc(lad)[-1])); pn=_slice_arc(prox,0,min(JUNCTION_QC_HALF_LENGTH_MM,_arc(prox)[-1])); through=np.vstack([ln[::-1],pn[1:]])
    tp,tq,qc=_dense_qc(ref,src,through); qc.to_csv(out/"junction_dense_qc.csv",index=False); pd.DataFrame(tp,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"junction_through_path.csv",index=False)
    qc_frac=float(qc.plane_pass.mean()) if len(qc) else 0.0; fail_ix=_sustained_failure(qc)

    ctrl=_branch_control(c6,c7); _write_json(out/"C67_split_control.json",ctrl)
    reentry=_overlap_metrics(parent,lad); reentry["reentry_gate_pass"]=_reentry_gate(reentry); pd.DataFrame([reentry]).to_csv(out/"parent_candidate_LAD_crosswalk.csv",index=False)

    gates={
        "endpoint_gap_le_0_5mm":bool(gap<=MAX_ENDPOINT_GAP_MM),
        "proximal_segment_matches_C6_C7_common_trunk":prox_common_gate,
        "through_vessel_deflection_le_25deg":bool(deflection<=MAX_THROUGH_DEFLECTION_DEG),
        "dense_source_qc_pass_fraction_ge_0_90":bool(qc_frac>=MIN_DENSE_QC_PASS_FRACTION),
        "no_three_consecutive_dense_qc_failures":bool(fail_ix is None),
        "C6_C7_real_split_control_pass":bool(ctrl["control_gate_pass"]),
    }
    continuity=all(gates.values())
    if not ctrl["control_gate_pass"]: status=STATUS_CONTROL
    elif continuity and reentry["reentry_gate_pass"]: status=STATUS_POS_REENTRY
    elif continuity: status=STATUS_POS
    else: status=STATUS_NEG

    decision={"status":status,"junction_lps_mm":junction.tolist(),"junction_endpoint_gap_mm":gap,"outgoing_daughter_angle_deg":outgoing_angle,
              "through_vessel_deflection_deg":deflection,"prox_common_overlap":prox_common,"dense_qc_pass_fraction":qc_frac,
              "first_sustained_dense_qc_failure_index":fail_ix,"C67_split_control":ctrl,"parent_candidate_LAD_crosswalk":reentry,
              "gates":gates,"continuity_gate_pass":continuity,"master_modified":False,"clinical_identity_established":False}
    _write_json(out/"continuity_decision.json",decision)

    # Geometry plot.
    fig,axes=plt.subplots(1,3,figsize=(15,4.8)); pairs=[(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]
    geom=[(lad,"frozen LAD"),(prox,"validated proximal/common segment"),(common,"C6/C7 common trunk"),(parent,"prior parent candidate")]
    for ax,(i,j,xl,yl) in zip(axes,pairs):
        for p0,lab in geom: ax.plot(p0[:,i],p0[:,j],lw=1.4,label=lab)
        ax.scatter([junction[i]],[junction[j]],s=24); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal",adjustable="box")
    axes[0].legend(fontsize=7); fig.suptitle("Left-coronary junction continuity and prior parent-candidate re-entry"); plt.tight_layout(); plt.savefig(out/"01_junction_geometry.png",dpi=180); plt.close()

    # Dense source-QC montage across junction.
    tt=np.gradient(tp,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9); picks=np.linspace(0,len(tp)-1,min(12,len(tp))).astype(int); cols=4; nr=int(math.ceil(len(picks)/cols)); fig,axes=plt.subplots(nr,cols,figsize=(14,3.6*nr)); axes=np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]: ax.axis("off")
    for ax,ix in zip(axes,picks):
        im,g=_plane(ref,src,tp[ix],tt[ix]); ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]); ax.scatter([0],[0],s=14); ax.set_title(f"{tq[ix]:.1f} mm"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Dense orthogonal CCTA QC across LAD/common-trunk junction"); plt.tight_layout(); plt.savefig(out/"02_junction_orthogonal_qc.png",dpi=180); plt.close()

    # Parent-candidate distance to LAD by arc.
    pq,pa=_resample(parent,.25); lq,la=_resample(lad,.25); d,ix=cKDTree(lq).query(pq)
    pd.DataFrame({"candidate_arc_mm":pa,"distance_to_frozen_LAD_mm":d,"nearest_LAD_arc_mm":la[ix]}).to_csv(out/"parent_candidate_LAD_distance_profile.csv",index=False)
    plt.figure(figsize=(8,4.5)); plt.plot(pa,d); plt.axhline(REENTRY_BAND_MM,ls="--",lw=.8); plt.xlabel("prior parent-candidate arc (mm)"); plt.ylabel("distance to frozen LAD (mm)"); plt.title("Prior parent candidate vs frozen LAD"); plt.tight_layout(); plt.savefig(out/"03_parent_candidate_LAD_crosswalk.png",dpi=180); plt.close()

    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,
             "junction":{"endpoint_gap_mm":gap,"outgoing_angle_deg":outgoing_angle,"through_deflection_deg":deflection,"prox_common_overlap":prox_common},
             "dense_qc":{"plane_count":int(len(qc)),"pass_fraction":qc_frac,"first_sustained_failure_index":fail_ix},"C67_split_control":ctrl,
             "parent_candidate_reentry":reentry,"gates":gates,
             "scientific_boundary":"This adjudicates continuous-vessel geometry and whether the prior parent hypothesis re-entered known LAD. It does not relabel LAD/LM/LCX/OM and does not modify frozen anatomy."}
    _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); return _finalize(out,summary)


def _finalize(out,summary):
    report=out/"OPENPLAQUE_LEFT_CORONARY_THROUGH_VESSEL_CONTINUITY_REPORT.html"
    report.write_text("<html><body><h1>OpenPlaque Left-Coronary Through-Vessel Continuity v1</h1>"+
                      f"<p><b>Status:</b> {summary.get('status')}</p>"+
                      f"<p>Through-vessel deflection: {summary.get('junction',{}).get('through_deflection_deg')} deg.</p>"+
                      f"<p>Dense source-plane pass fraction: {summary.get('dense_qc',{}).get('pass_fraction')}.</p>"+
                      f"<p>C6/C7 real-split control: {summary.get('C67_split_control',{}).get('control_gate_pass')}.</p>"+
                      f"<p>Prior parent candidate re-enters LAD: {summary.get('parent_candidate_reentry',{}).get('reentry_gate_pass')}.</p>"+
                      "<p>Frozen master unchanged; clinical vessel identity remains unresolved.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_CORONARY_THROUGH_VESSEL_CONTINUITY_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
