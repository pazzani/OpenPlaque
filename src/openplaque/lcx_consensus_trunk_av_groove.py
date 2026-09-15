from __future__ import annotations

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
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-consensus-trunk-av-groove-v1.0"
OUTPUT_DIRNAME = "LCX_Consensus_Trunk_AV_Groove_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
HEART = TS / "heartchambers_highres"
COR_CURRENT = TS / "coronary_arteries/coronary_arteries.nii.gz"
COR_LEGACY = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
LA = HEART / "heart_atrium_left.nii.gz"
LV = HEART / "heart_ventricle_left.nii.gz"
RA = HEART / "heart_atrium_right.nii.gz"
RV = HEART / "heart_ventricle_right.nii.gz"
MYO = HEART / "heart_myocardium.nii.gz"

PRIOR = Path("Joint_Three_Vessel_Template_Classifier_v1")
PRIOR_RANKING = PRIOR / "LCX_joint_candidate_ranking.csv"
LEAF_FILES = [PRIOR / f"candidate_{i:02d}_source_path.csv" for i in range(1, 6)]

STATUS_NO_CONSENSUS = "NO_CONSENSUS_LEFT_CORONARY_BRANCH_TRUNK"
STATUS_CONTROL_FAIL = "CONSENSUS_TRUNK_ESTABLISHED_AV_GROOVE_CONTROL_FAILED"
STATUS_AMBIG = "CONSENSUS_TRUNK_ESTABLISHED_DISTAL_AV_GROOVE_AMBIGUOUS"
STATUS_CANDIDATE = "LCX_AV_GROOVE_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC"

def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")

def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    img = sitk.GetImageFromArray(np.asarray(arr))
    spacing_zyx = np.asarray(meta["spacing_zyx"], float)
    img.SetSpacing(tuple(spacing_zyx[::-1]))
    img.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([[row[0], col[0], slc[0]],
                          [row[1], col[1], slc[1]],
                          [row[2], col[2], slc[2]]], float)
    img.SetDirection(tuple(direction.ravel()))
    return img, np.asarray(arr)

def _xyz_to_zyx(img, pts_lps):
    pts = np.atleast_2d(np.asarray(pts_lps, float))
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(direction).T) / sp
    return idx_xyz[:, ::-1]

def _zyx_to_xyz(img, pts_zyx):
    pts = np.atleast_2d(np.asarray(pts_zyx, float))
    idx_xyz = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ direction.T

def _sample(arr, img, pts, order=1, cval=0.0):
    zyx = _xyz_to_zyx(img, pts)
    return map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)

def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize()
            and np.allclose(im.GetSpacing(), ref.GetSpacing())
            and np.allclose(im.GetOrigin(), ref.GetOrigin())
            and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0

def _load_path(path, ref=None):
    d = pd.read_csv(_req(path))
    for cols in [("lps_x_mm","lps_y_mm","lps_z_mm"), ("x_mm","y_mm","z_mm")]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    if ref is not None:
        for cols in [("zyx_z","zyx_y","zyx_x"), ("source_z","source_y","source_x"), ("z","y","x")]:
            if all(c in d.columns for c in cols):
                return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    if all(c in d.columns for c in ("x","y","z")):
        return d[["x","y","z"]].to_numpy(float)
    raise ValueError(f"Unrecognized path columns for {path}: {list(d.columns)}")

def _arc(points):
    p = np.asarray(points, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

def _resample_path(points, step=0.25):
    p = np.asarray(points, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p, a
    q = np.arange(0, a[-1] + 1e-8, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    out = np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])
    return out, q

def _interp_path(points, q):
    p = np.asarray(points, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])

def _orient_common(paths):
    ref = paths[0][0]
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref):
            p = p[::-1].copy()
        out.append(p)
    return out

def _consensus_prefix(paths, step=0.25, tol_mm=0.60, sustain=4):
    paths = _orient_common(paths)
    minlen = min(_arc(p)[-1] for p in paths)
    q = np.arange(0, minlen + 1e-9, step)
    samples = np.stack([_interp_path(p, q) for p in paths], axis=0)
    centroid = np.median(samples, axis=0)
    dev = np.max(np.linalg.norm(samples - centroid[None, :, :], axis=2), axis=0)
    first_bad = None
    for i in range(max(1, int(round(5.0/step))), max(1, len(q)-sustain+1)):
        if np.all(dev[i:i+sustain] > tol_mm):
            first_bad = i
            break
    end_i = (first_bad - 1) if first_bad is not None else (len(q)-1)
    return centroid[:end_i+1], q[:end_i+1], dev[:end_i+1], {
        "common_prefix_total_mm": float(q[end_i]),
        "prefix_tolerance_mm": float(tol_mm),
        "prefix_max_deviation_mm": float(np.max(dev[:end_i+1])),
        "first_sustained_split_arc_mm": None if first_bad is None else float(q[first_bad]),
    }

def _nearest_dist(points, ref):
    return cKDTree(np.asarray(ref,float)).query(np.asarray(points,float))[0]

def _first_sustained(arr, threshold, step, duration_mm=1.0):
    arr = np.asarray(arr,float)
    n = max(1, int(round(duration_mm/step)))
    for i in range(0, len(arr)-n+1):
        if np.all(arr[i:i+n] >= threshold):
            return i
    return None

def _distance_to_surface(mask, spacing_zyx):
    mask = np.asarray(mask, bool)
    dout = distance_transform_edt(~mask, sampling=spacing_zyx)
    din = distance_transform_edt(mask, sampling=spacing_zyx)
    return np.where(mask, din, dout).astype(np.float32)

def _path_surface_profile(path, surface_dist, ref):
    return _sample(surface_dist, ref, path, order=1, cval=100.0)

def _linear_slope(x, y):
    x = np.asarray(x,float); y=np.asarray(y,float)
    if len(x) < 3 or np.ptp(x) < 1e-6:
        return 0.0
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])

def _groove_metrics(path, dist_a, dist_v, ref):
    p, arc = _resample_path(path, 0.25)
    da = _path_surface_profile(p, dist_a, ref)
    dv = _path_surface_profile(p, dist_v, ref)
    both = np.maximum(da, dv)
    balance = np.abs(da-dv)
    point_score = np.exp(-both/12.0) * np.exp(-balance/10.0)
    slope_a = _linear_slope(arc, da)
    slope_both = _linear_slope(arc, both)
    retention = math.exp(-max(0.0, slope_a)/0.35)
    score = float(0.75*np.median(point_score) + 0.25*retention)
    return {
        "length_mm": float(arc[-1]) if len(arc) else 0.0,
        "median_atrium_surface_mm": float(np.median(da)),
        "median_ventricle_surface_mm": float(np.median(dv)),
        "median_max_chamber_surface_mm": float(np.median(both)),
        "median_balance_mm": float(np.median(balance)),
        "endpoint_atrium_surface_mm": float(da[-1]),
        "endpoint_ventricle_surface_mm": float(dv[-1]),
        "atrium_distance_slope_mm_per_mm": float(slope_a),
        "max_chamber_distance_slope_mm_per_mm": float(slope_both),
        "atrium_retention_score": float(retention),
        "groove_score": score,
        "profile_arc": arc,
        "profile_atrium": da,
        "profile_ventricle": dv,
        "profile_point_score": point_score,
        "resampled_points": p,
    }

def _support_fraction(mask, ref, path, radius_mm=1.0):
    p, _ = _resample_path(path, 0.25)
    z = np.argwhere(mask)
    if len(z) == 0:
        return 0.0
    if len(z) > 200000:
        z = z[::int(math.ceil(len(z)/200000))]
    tree = cKDTree(_zyx_to_xyz(ref, z))
    d = tree.query(p)[0]
    return float(np.mean(d <= radius_mm))

def _robust_hu(source, ref, path):
    p, _ = _resample_path(path, 0.25)
    hu = _sample(source, ref, p, order=1, cval=-1024.0)
    return float(np.mean((hu >= 120) & (hu <= 1200))), float(np.median(hu))

def _union_find_groups(points, threshold=1.5):
    points = np.asarray(points,float)
    n=len(points)
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[rb]=ra
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(points[i]-points[j]) <= threshold:
                union(i,j)
    groups={}
    for i in range(n):
        groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

def _plane(ref, source, center, normal, binormal, chamber_masks=None, half=6.0, step=0.20):
    q=np.arange(-half,half+1e-9,step)
    yy,xx=np.meshgrid(q,q,indexing="ij")
    pts=center[None,None,:]+xx[...,None]*normal[None,None,:]+yy[...,None]*binormal[None,None,:]
    ct=_sample(source,ref,pts.reshape(-1,3),order=1,cval=-1024).reshape(len(q),len(q))
    masks={}
    if chamber_masks:
        for name,arr in chamber_masks.items():
            masks[name]=_sample(arr.astype(np.float32),ref,pts.reshape(-1,3),order=0,cval=0).reshape(len(q),len(q))
    return ct,masks,q

def _frame(points, i):
    p=np.asarray(points,float)
    a=max(0,i-2); b=min(len(p)-1,i+2)
    t=p[b]-p[a]; t=t/max(np.linalg.norm(t),1e-9)
    axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]
    n=np.cross(t,seed); n=n/max(np.linalg.norm(n),1e-9)
    bv=np.cross(t,n); bv=bv/max(np.linalg.norm(bv),1e-9)
    return n,bv

def synthetic_av_groove_self_test():
    t=np.linspace(0,20,81)
    trunk=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)])
    p1=trunk.copy()
    p2=trunk.copy()
    p2[t>15,1]=(t[t>15]-15)*0.8
    p3=trunk.copy()
    p3[t>15,2]=(t[t>15]-15)*0.8
    c,q,dev,meta=_consensus_prefix([p1,p2,p3],step=0.25,tol_mm=0.4,sustain=3)
    assert 14.5 <= meta["common_prefix_total_mm"] <= 15.5
    return {"ok":True,"consensus_mm":meta["common_prefix_total_mm"]}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",
        output_dir=None):
    root=Path(drive_root)
    out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})

    required=[root/SOURCE_CACHE/"series7_int16.npy", root/SOURCE_CACHE/"series7_int16.json",
              root/MASTER, root/LAD_PATH, root/RCA_PATH, root/COR_CURRENT, root/COR_LEGACY,
              root/LA, root/LV, root/RA, root/RV, root/MYO, root/PRIOR_RANKING] + [root/p for p in LEAF_FILES]
    for p in required: _req(p)

    master=json.loads((root/MASTER).read_text())
    ref,source=_source(root/SOURCE_CACHE)
    spacing_zyx=np.asarray(ref.GetSpacing()[::-1],float)
    current=_resample_mask(root/COR_CURRENT, ref)
    legacy=_resample_mask(root/COR_LEGACY, ref)
    la=_resample_mask(root/LA, ref); lv=_resample_mask(root/LV, ref)
    ra=_resample_mask(root/RA, ref); rv=_resample_mask(root/RV, ref)
    myo=_resample_mask(root/MYO, ref)
    dLA=_distance_to_surface(la,spacing_zyx); dLV=_distance_to_surface(lv,spacing_zyx)
    dRA=_distance_to_surface(ra,spacing_zyx); dRV=_distance_to_surface(rv,spacing_zyx)
    dMYO=_distance_to_surface(myo,spacing_zyx)

    lad=_load_path(root/LAD_PATH,ref)
    rca=_load_path(root/RCA_PATH,ref)
    leaves=[_load_path(root/p,ref) for p in LEAF_FILES]
    ranking=pd.read_csv(root/PRIOR_RANKING)
    prior_by_file={}
    for i in range(1,6):
        row=ranking.iloc[i-1].to_dict()
        prior_by_file[i]=row

    consensus,q_common,dev,cons_meta=_consensus_prefix(leaves,step=0.25,tol_mm=0.60,sustain=4)
    d_lad=_nearest_dist(consensus,lad)
    div_i=_first_sustained(d_lad,1.50,0.25,1.0)
    if div_i is None:
        div_i=len(consensus)-1
    branch_trunk=consensus[div_i:].copy()
    branch_trunk_arc=_arc(branch_trunk)
    trunk_current=_support_fraction(current,ref,branch_trunk)
    trunk_legacy=_support_fraction(legacy,ref,branch_trunk)
    trunk_hu_frac,trunk_hu_med=_robust_hu(source,ref,branch_trunk)
    trunk_ok=bool(branch_trunk_arc[-1] >= 10.0 and trunk_current >= 0.90 and trunk_legacy >= 0.90 and trunk_hu_frac >= 0.90)
    pd.DataFrame(branch_trunk,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).assign(
        arc_mm=branch_trunk_arc, lad_distance_mm=d_lad[div_i:],
    ).to_csv(out/"consensus_left_coronary_branch_trunk.csv",index=False)

    cons_summary={
        **cons_meta,
        "lad_divergence_arc_from_candidate_origin_mm":float(q_common[div_i]),
        "post_lad_divergence_common_trunk_mm":float(branch_trunk_arc[-1]),
        "current_support_fraction":trunk_current,
        "legacy_support_fraction":trunk_legacy,
        "robust_hu_fraction":trunk_hu_frac,
        "median_hu":trunk_hu_med,
        "consensus_trunk_gate_pass":trunk_ok,
    }
    _write_json(out/"consensus_trunk_summary.json",cons_summary)

    rca_ctrl=_groove_metrics(rca,dRA,dRV,ref)
    lad_ctrl=_groove_metrics(lad,dLA,dLV,ref)
    control_margin=float(rca_ctrl["groove_score"]-lad_ctrl["groove_score"])
    control_pass=bool(rca_ctrl["groove_score"] >= 0.20 and control_margin >= 0.05)
    controls={
        "RCA_right_AV_groove_score":rca_ctrl["groove_score"],
        "RCA_median_right_atrium_surface_mm":rca_ctrl["median_atrium_surface_mm"],
        "RCA_median_right_ventricle_surface_mm":rca_ctrl["median_ventricle_surface_mm"],
        "LAD_left_AV_groove_negative_score":lad_ctrl["groove_score"],
        "LAD_median_left_atrium_surface_mm":lad_ctrl["median_atrium_surface_mm"],
        "LAD_median_left_ventricle_surface_mm":lad_ctrl["median_ventricle_surface_mm"],
        "control_margin":control_margin,
        "control_pass":control_pass,
    }
    _write_json(out/"av_groove_controls.json",controls)

    common_end=float(q_common[-1])
    min_remaining=min(_arc(p)[-1]-common_end for p in leaves)
    probe_delta=max(1.0,min(4.0,0.5*min_remaining))
    probe_arc=common_end+probe_delta
    probe_points=np.vstack([_interp_path(p,[probe_arc])[0] for p in leaves])
    groups=_union_find_groups(probe_points,threshold=1.5)
    family_of={}
    for fi,g in enumerate(groups,1):
        for idx in g: family_of[idx]=fi
    family_rows=[]
    for fi,g in enumerate(groups,1):
        family_rows.append({
            "family_id":fi,
            "member_leaf_indices":";".join(str(i+1) for i in g),
            "member_source_candidate_ids":";".join(str(int(prior_by_file[i+1]["candidate_id"])) for i in g),
            "probe_arc_mm":probe_arc,
            "n_members":len(g),
        })
    pd.DataFrame(family_rows).to_csv(out/"distal_branch_family_membership.csv",index=False)

    leaf_rows=[]; profile_map={}
    for idx,p in enumerate(leaves):
        total=_arc(p)[-1]
        start=min(common_end,total-0.5)
        q=np.arange(start,total+1e-9,0.25)
        if len(q)<3:
            q=np.linspace(start,total,max(3,int(round((total-start)/0.25))+1))
        cont=_interp_path(p,q)
        gm=_groove_metrics(cont,dLA,dLV,ref)
        cur_sup=_support_fraction(current,ref,cont)
        leg_sup=_support_fraction(legacy,ref,cont)
        hu_frac,hu_med=_robust_hu(source,ref,cont)
        myo_dist=_path_surface_profile(gm["resampled_points"],dMYO,ref)
        prior=prior_by_file[idx+1]
        rel_rca=gm["groove_score"]/max(rca_ctrl["groove_score"],1e-6)
        above_lad=gm["groove_score"]-lad_ctrl["groove_score"]
        gate=bool(control_pass and cur_sup>=0.90 and leg_sup>=0.90 and hu_frac>=0.90
                  and rel_rca>=0.70 and above_lad>=0.05 and gm["atrium_retention_score"]>=0.55)
        row={
            "leaf_index":idx+1,
            "source_candidate_id":int(prior["candidate_id"]),
            "family_id":family_of[idx],
            "prior_LCX_score":float(prior["LCX_score"]),
            "prior_LCX_margin":float(prior["LCX_margin"]),
            "continuation_length_mm":gm["length_mm"],
            "groove_score":gm["groove_score"],
            "relative_to_RCA_control":rel_rca,
            "margin_over_LAD_negative":above_lad,
            "median_left_atrium_surface_mm":gm["median_atrium_surface_mm"],
            "median_left_ventricle_surface_mm":gm["median_ventricle_surface_mm"],
            "median_max_chamber_surface_mm":gm["median_max_chamber_surface_mm"],
            "median_balance_mm":gm["median_balance_mm"],
            "endpoint_left_atrium_surface_mm":gm["endpoint_atrium_surface_mm"],
            "endpoint_left_ventricle_surface_mm":gm["endpoint_ventricle_surface_mm"],
            "left_atrium_distance_slope_mm_per_mm":gm["atrium_distance_slope_mm_per_mm"],
            "atrium_retention_score":gm["atrium_retention_score"],
            "median_myocardium_surface_mm":float(np.median(myo_dist)),
            "current_support_fraction":cur_sup,
            "legacy_support_fraction":leg_sup,
            "robust_hu_fraction":hu_frac,
            "median_hu":hu_med,
            "av_groove_gate_pass":gate,
        }
        leaf_rows.append(row)
        profile_map[idx+1]=(gm,cont)
    leaf_df=pd.DataFrame(leaf_rows).sort_values(["groove_score","atrium_retention_score"],ascending=False).reset_index(drop=True)
    leaf_df.insert(0,"av_rank",np.arange(1,len(leaf_df)+1))
    leaf_df.to_csv(out/"distal_av_groove_scores.csv",index=False)

    fam_scores=[]
    for fi in sorted(leaf_df["family_id"].unique()):
        qf=leaf_df[leaf_df["family_id"]==fi].sort_values("groove_score",ascending=False)
        r=qf.iloc[0]
        fam_scores.append({
            "family_id":int(fi),
            "best_leaf_index":int(r["leaf_index"]),
            "best_source_candidate_id":int(r["source_candidate_id"]),
            "family_groove_score":float(r["groove_score"]),
            "relative_to_RCA_control":float(r["relative_to_RCA_control"]),
            "margin_over_LAD_negative":float(r["margin_over_LAD_negative"]),
            "atrium_retention_score":float(r["atrium_retention_score"]),
            "n_leaf_members":int((leaf_df["family_id"]==fi).sum()),
            "family_gate_pass":bool(qf["av_groove_gate_pass"].any()),
        })
    fam_df=pd.DataFrame(fam_scores).sort_values("family_groove_score",ascending=False).reset_index(drop=True)
    fam_df.insert(0,"family_rank",np.arange(1,len(fam_df)+1))
    fam_df.to_csv(out/"distal_branch_family_scores.csv",index=False)

    if not trunk_ok:
        status=STATUS_NO_CONSENSUS
    elif not control_pass:
        status=STATUS_CONTROL_FAIL
    else:
        passing=fam_df[fam_df["family_gate_pass"]]
        if len(passing)==0:
            status=STATUS_AMBIG
        else:
            top=passing.iloc[0]
            second=float(passing.iloc[1]["family_groove_score"]) if len(passing)>1 else -np.inf
            margin=float(top["family_groove_score"]-second) if np.isfinite(second) else 1.0
            status=STATUS_CANDIDATE if margin>=0.04 else STATUS_AMBIG

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for p in leaves:
        axes[0].plot(p[:,0],p[:,1],alpha=.65)
        axes[1].plot(p[:,1],p[:,2],alpha=.65)
    axes[0].plot(lad[:,0],lad[:,1],"k--",linewidth=2,label="accepted LAD")
    axes[1].plot(lad[:,1],lad[:,2],"k--",linewidth=2,label="accepted LAD")
    axes[0].plot(branch_trunk[:,0],branch_trunk[:,1],"k",linewidth=4,label="consensus post-LAD trunk")
    axes[1].plot(branch_trunk[:,1],branch_trunk[:,2],"k",linewidth=4,label="consensus post-LAD trunk")
    axes[0].set_xlabel("LPS x mm"); axes[0].set_ylabel("LPS y mm")
    axes[1].set_xlabel("LPS y mm"); axes[1].set_ylabel("LPS z mm")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("Consensus left-coronary branch trunk and distal leaf paths")
    fig.tight_layout(); fig.savefig(out/"01_consensus_trunk_and_branches.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(leaf_df))
    ax.bar(x,leaf_df["groove_score"])
    ax.axhline(rca_ctrl["groove_score"],linestyle="--",label="RCA right-AV-groove control")
    ax.axhline(lad_ctrl["groove_score"],linestyle=":",label="LAD left-side negative")
    ax.set_xticks(x); ax.set_xticklabels([f"Leaf {int(i)} / C{int(c)}" for i,c in zip(leaf_df["leaf_index"],leaf_df["source_candidate_id"])])
    ax.set_ylabel("AV-groove score"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.2)
    ax.set_title("Distal continuations: chamber-geometry AV-groove score")
    fig.tight_layout(); fig.savefig(out/"02_av_groove_scores.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    fig,axes=plt.subplots(len(leaves),1,figsize=(10,12),sharex=False)
    for ax,row in zip(np.atleast_1d(axes),leaf_df.itertuples(index=False)):
        gm,_=profile_map[int(row.leaf_index)]
        ax.plot(gm["profile_arc"],gm["profile_atrium"],label="left atrium surface")
        ax.plot(gm["profile_arc"],gm["profile_ventricle"],label="left ventricle surface")
        ax.set_ylabel("mm"); ax.set_title(f"Leaf {int(row.leaf_index)} / source C{int(row.source_candidate_id)} / family {int(row.family_id)}")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("arc from common-trunk split (mm)")
    fig.tight_layout(); fig.savefig(out/"03_distal_chamber_distance_profiles.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    top_rows=leaf_df.head(2)
    fig,axes=plt.subplots(len(top_rows),3,figsize=(12,4*len(top_rows)))
    axes=np.atleast_2d(axes)
    for rr,row in enumerate(top_rows.itertuples(index=False)):
        gm,cont=profile_map[int(row.leaf_index)]
        pts=gm["resampled_points"]
        for cc,frac in enumerate((0.15,0.50,0.85)):
            i=int(round(frac*(len(pts)-1)))
            n,bv=_frame(pts,i)
            ct,masks,qq=_plane(ref,source,pts[i],n,bv,{"LA":la,"LV":lv})
            ax=axes[rr,cc]
            ax.imshow(ct,cmap="gray",vmin=-200,vmax=1000,extent=[qq[0],qq[-1],qq[-1],qq[0]])
            try:
                ax.contour(qq,qq,masks["LA"],levels=[0.5],linewidths=1)
                ax.contour(qq,qq,masks["LV"],levels=[0.5],linewidths=1)
            except Exception:
                pass
            ax.scatter([0],[0],marker="+",s=35)
            ax.set_title(f"Leaf {int(row.leaf_index)} — {frac:.0%} distal")
            ax.set_xlabel("mm"); ax.set_ylabel("mm")
    fig.suptitle("Top AV-groove continuations — source CCTA with LA/LV chamber contours")
    fig.tight_layout(); fig.savefig(out/"04_top_av_groove_orthogonal_qc.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    top_leaf=leaf_df.iloc[0].to_dict() if len(leaf_df) else None
    top_family=fam_df.iloc[0].to_dict() if len(fam_df) else None
    summary={
        "status":status,
        "algorithm":ALGORITHM,
        "baseline_commit":BASELINE,
        "master_status":master.get("status"),
        "LCX_master_status":"UNRESOLVED",
        "consensus_trunk":cons_summary,
        "av_groove_controls":controls,
        "n_distal_families":int(len(fam_df)),
        "top_leaf":top_leaf,
        "top_family":top_family,
        "template_similarity_used_in_decision":False,
        "scientific_boundary":"The common source-space branch trunk is independent of curved-template identity. Chamber geometry is used only to nominate an LCX-like distal continuation. No result establishes LM identity, circumferential registration, plaque volume, or clinical vessel labeling without visual/anatomical review."
    }
    _write_json(out/"summary.json",summary)

    html=f"""<!doctype html><html><head><meta charset="utf-8"><title>OpenPlaque LCX consensus trunk / AV groove</title>
<style>body{{font-family:Arial;max-width:1200px;margin:30px auto;line-height:1.4}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px}}.warn{{background:#fff3cd;padding:12px}}</style></head>
<body><h1>OpenPlaque — consensus left-coronary branch trunk and AV-groove adjudication</h1>
<p><b>Status:</b> {status}</p>
<div class="warn">Curved-template similarity contributes zero to the decision. RCA right-AV-groove anatomy is the patient-specific positive control; LAD is the left-side negative control. LCX remains unresolved unless a distal continuation passes chamber-geometry and visual QC.</div>
<h2>Consensus trunk</h2><pre>{json.dumps(cons_summary,indent=2)}</pre>
<h2>AV-groove controls</h2><pre>{json.dumps(controls,indent=2)}</pre>
<h2>Distal leaf scores</h2>{leaf_df.to_html(index=False,float_format=lambda x:f"{x:.4f}")}
<h2>Branch-family scores</h2>{fam_df.to_html(index=False,float_format=lambda x:f"{x:.4f}")}
<h2>Summary</h2><pre>{json.dumps(summary,indent=2,default=str)}</pre>
<img src="01_consensus_trunk_and_branches.png" style="max-width:100%"><img src="02_av_groove_scores.png" style="max-width:100%">
<img src="03_distal_chamber_distance_profiles.png" style="max-width:100%"><img src="04_top_av_groove_orthogonal_qc.png" style="max-width:100%">
</body></html>"""
    report=out/"OPENPLAQUE_LCX_CONSENSUS_TRUNK_AV_GROOVE_REPORT.html"
    report.write_text(html,encoding="utf-8")
    _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"baseline_commit":BASELINE,"algorithm":ALGORITHM})
    zip_path=out/"OPENPLAQUE_LCX_CONSENSUS_TRUNK_AV_GROOVE_REPORT_BACK.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out.iterdir()):
            if fp==zip_path: continue
            zf.write(fp,arcname=fp.name)
    return {"summary":summary,"report":str(report),"zip":str(zip_path)}

if __name__=="__main__":
    print(synthetic_av_groove_self_test())
