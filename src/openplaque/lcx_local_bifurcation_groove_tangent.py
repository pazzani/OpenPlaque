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
ALGORITHM = "lcx-local-bifurcation-groove-tangent-v1.0"
OUTPUT_DIRNAME = "LCX_Local_Bifurcation_Groove_Tangent_v1"

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

STATUS_NO_TRUNK = "NO_CONSENSUS_LEFT_CORONARY_BRANCH_TRUNK"
STATUS_CONTROL_FAIL = "LOCAL_AV_GROOVE_TANGENT_CONTROL_FAILED"
STATUS_AMBIG = "CONSENSUS_TRUNK_ESTABLISHED_LOCAL_BIFURCATION_AMBIGUOUS"
STATUS_CANDIDATE = "LCX_LOCAL_GROOVE_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC"


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
    direction = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    img.SetDirection(tuple(direction.ravel()))
    return img, np.asarray(arr)


def _xyz_to_zyx(img, pts_lps):
    pts = np.atleast_2d(np.asarray(pts_lps, float))
    o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(direction).T) / sp
    return idx_xyz[:, ::-1]


def _zyx_to_xyz(img, pts_zyx):
    pts = np.atleast_2d(np.asarray(pts_zyx, float)); idx_xyz = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float); sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ direction.T


def _sample(arr, img, pts, order=1, cval=0.0):
    zyx = _xyz_to_zyx(img, pts)
    return map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize() and np.allclose(im.GetSpacing(), ref.GetSpacing()) and np.allclose(im.GetOrigin(), ref.GetOrigin()) and np.allclose(im.GetDirection(), ref.GetDirection()))
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
    if len(p) <= 1: return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp_path(points, q):
    p = np.asarray(points, float); a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample_path(points, step=0.25):
    p = np.asarray(points, float); a = _arc(p)
    if len(p) < 2 or a[-1] <= 0: return p, a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6: q = np.r_[q, a[-1]]
    return _interp_path(p, q), q


def _orient_common(paths):
    ref = np.asarray(paths[0], float)[0]; out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref): p = p[::-1].copy()
        out.append(p)
    return out


def _consensus_prefix(paths, step=0.25, tol_mm=0.60, sustain=4):
    paths = _orient_common(paths); minlen = min(_arc(p)[-1] for p in paths)
    q = np.arange(0, minlen + 1e-9, step)
    samples = np.stack([_interp_path(p, q) for p in paths], axis=0)
    centroid = np.median(samples, axis=0)
    dev = np.max(np.linalg.norm(samples - centroid[None, :, :], axis=2), axis=0)
    first_bad = None; start = max(1, int(round(5.0 / step)))
    for i in range(start, max(start, len(q) - sustain + 1)):
        if np.all(dev[i:i+sustain] > tol_mm): first_bad = i; break
    end_i = first_bad - 1 if first_bad is not None else len(q) - 1
    return centroid[:end_i+1], q[:end_i+1], dev[:end_i+1], {"common_prefix_total_mm": float(q[end_i]), "prefix_tolerance_mm": float(tol_mm), "prefix_max_deviation_mm": float(np.max(dev[:end_i+1])), "first_sustained_split_arc_mm": None if first_bad is None else float(q[first_bad])}


def _nearest_dist(points, ref):
    return cKDTree(np.asarray(ref, float)).query(np.asarray(points, float))[0]


def _first_sustained(arr, threshold, step, duration_mm=1.0):
    arr = np.asarray(arr, float); n = max(1, int(round(duration_mm / step)))
    for i in range(0, len(arr) - n + 1):
        if np.all(arr[i:i+n] >= threshold): return i
    return None


def _distance_to_surface(mask, spacing_zyx):
    mask = np.asarray(mask, bool)
    dout = distance_transform_edt(~mask, sampling=spacing_zyx); din = distance_transform_edt(mask, sampling=spacing_zyx)
    return np.where(mask, din, dout).astype(np.float32)


def _distance_gradient_lps(dist, ref):
    spacing_zyx = np.asarray(ref.GetSpacing()[::-1], float)
    gz, gy, gx = np.gradient(np.asarray(dist, np.float32), *spacing_zyx, edge_order=1)
    direction = np.asarray(ref.GetDirection(), float).reshape(3, 3)
    return gx.astype(np.float32), gy.astype(np.float32), gz.astype(np.float32), direction


def _sample_gradient_lps(grad_pack, ref, points):
    gx, gy, gz, direction = grad_pack; pts = np.asarray(points, float)
    vec_idx = np.column_stack([_sample(gx, ref, pts, 1, 0.0), _sample(gy, ref, pts, 1, 0.0), _sample(gz, ref, pts, 1, 0.0)])
    vec_lps = vec_idx @ direction.T; n = np.linalg.norm(vec_lps, axis=1); good = n > 1e-6
    out = np.zeros_like(vec_lps); out[good] = vec_lps[good] / n[good, None]
    return out, good


def _path_tangents(points):
    p = np.asarray(points, float)
    if len(p) < 2: return np.zeros_like(p)
    d = np.gradient(p, axis=0); n = np.linalg.norm(d, axis=1); n[n < 1e-9] = 1.0
    return d / n[:, None]


def _groove_tangent(norm_a, norm_v):
    g = np.cross(norm_a, norm_v); n = np.linalg.norm(g, axis=1); good = n > 1e-6
    out = np.zeros_like(g); out[good] = g[good] / n[good, None]
    return out, good


def _angle_deg_abs(a, b):
    c = np.sum(np.asarray(a,float)*np.asarray(b,float), axis=1); c = np.clip(np.abs(c), 0.0, 1.0)
    return np.degrees(np.arccos(c))


def _continuation_angle_deg(incoming_tangent, daughter_points, lookahead_mm=2.0):
    p, _ = _resample_path(daughter_points, 0.25)
    if len(p) < 2: return 180.0
    idx = min(len(p)-1, max(1, int(round(lookahead_mm/0.25)))); v = p[idx] - p[0]; nv = np.linalg.norm(v)
    if nv < 1e-9: return 180.0
    v /= nv; c = float(np.clip(np.dot(incoming_tangent, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _linear_slope(x, y):
    x = np.asarray(x,float); y = np.asarray(y,float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) < 1e-6: return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def _local_alignment_metrics(path, dist_a, dist_v, grad_a, grad_v, ref, incoming_tangent=None, max_length_mm=4.0):
    p, q = _resample_path(path, 0.25)
    if len(p) == 0: raise ValueError("Empty path")
    keep = q <= min(float(q[-1]), float(max_length_mm)) + 1e-9; p = p[keep]; q = q[keep]
    t = _path_tangents(p); na, ga = _sample_gradient_lps(grad_a, ref, p); nv, gv = _sample_gradient_lps(grad_v, ref, p); groove, gg = _groove_tangent(na, nv); good = ga & gv & gg
    if good.sum() < 3:
        groove_alignment = a_tangency = v_tangency = 0.0; groove_angle_med = groove_angle_p90 = 90.0; angle_profile = np.array([])
    else:
        dotg = np.abs(np.sum(t[good]*groove[good], axis=1)); angle_profile = _angle_deg_abs(t[good], groove[good])
        groove_alignment = float(np.median(dotg)); groove_angle_med = float(np.median(angle_profile)); groove_angle_p90 = float(np.percentile(angle_profile,90))
        a_tangency = float(np.median(1.0 - np.abs(np.sum(t[good]*na[good], axis=1)))); v_tangency = float(np.median(1.0 - np.abs(np.sum(t[good]*nv[good], axis=1))))
    da = _sample(dist_a, ref, p, 1, 100.0); dv = _sample(dist_v, ref, p, 1, 100.0); slope_a = _linear_slope(q, da); retention = float(math.exp(-max(0.0, slope_a)/0.35))
    continuity_angle = None; continuity_score = 1.0
    if incoming_tangent is not None:
        continuity_angle = _continuation_angle_deg(np.asarray(incoming_tangent,float), p, 2.0); continuity_score = float(math.exp(-continuity_angle/55.0))
    score = float(0.55*groove_alignment + 0.15*a_tangency + 0.10*v_tangency + 0.10*retention + 0.10*continuity_score)
    return {"local_length_mm":float(q[-1]), "local_groove_alignment_cos":groove_alignment, "median_groove_tangent_angle_deg":groove_angle_med, "p90_groove_tangent_angle_deg":groove_angle_p90, "left_atrium_surface_tangency":a_tangency, "left_ventricle_surface_tangency":v_tangency, "left_atrium_distance_slope_mm_per_mm":float(slope_a), "atrium_retention_score":retention, "incoming_continuation_angle_deg":continuity_angle, "incoming_continuity_score":continuity_score, "local_geometry_score":score, "profile_arc_mm":q, "profile_groove_angle_deg":angle_profile, "points":p}


def _control_alignment_metrics(path, dist_a, dist_v, grad_a, grad_v, ref):
    p, q = _resample_path(path, 0.50)
    if len(p) > 12: p = p[4:-4]; q = q[4:-4] - q[4]
    t = _path_tangents(p); na, ga = _sample_gradient_lps(grad_a, ref, p); nv, gv = _sample_gradient_lps(grad_v, ref, p); groove, gg = _groove_tangent(na, nv); good = ga & gv & gg
    if good.sum() < 5: return {"score":0.0,"median_angle_deg":90.0,"n_valid":int(good.sum())}
    dotg = np.abs(np.sum(t[good]*groove[good],axis=1)); ang = _angle_deg_abs(t[good],groove[good]); a_t = 1.0-np.abs(np.sum(t[good]*na[good],axis=1)); v_t=1.0-np.abs(np.sum(t[good]*nv[good],axis=1))
    da = _sample(dist_a, ref, p, 1, 100.0); slope = _linear_slope(q,da); retention=float(math.exp(-max(0.0,slope)/0.35)); score=float(0.65*np.median(dotg)+0.15*np.median(a_t)+0.10*np.median(v_t)+0.10*retention)
    return {"score":score,"median_angle_deg":float(np.median(ang)),"p90_angle_deg":float(np.percentile(ang,90)),"median_groove_alignment_cos":float(np.median(dotg)),"median_atrium_tangency":float(np.median(a_t)),"median_ventricle_tangency":float(np.median(v_t)),"atrium_retention_score":retention,"n_valid":int(good.sum())}


def _support_fraction(mask, ref, path, radius_mm=1.0):
    p,_=_resample_path(path,0.25); z=np.argwhere(mask)
    if len(z)==0:return 0.0
    if len(z)>200000:z=z[::int(math.ceil(len(z)/200000))]
    return float(np.mean(cKDTree(_zyx_to_xyz(ref,z)).query(p)[0] <= radius_mm))


def _robust_hu(source, ref, path):
    p,_=_resample_path(path,0.25); hu=_sample(source,ref,p,1,-1024.0)
    return float(np.mean((hu>=120)&(hu<=1200))), float(np.median(hu))


def _union_find_groups(points, threshold=1.5):
    points=np.asarray(points,float); n=len(points); parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(points[i]-points[j]) <= threshold: union(i,j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    return list(groups.values())


def _frame(points,i):
    p=np.asarray(points,float); a=max(0,i-2); b=min(len(p)-1,i+2); t=p[b]-p[a]; t=t/max(np.linalg.norm(t),1e-9); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; n=np.cross(t,seed); n=n/max(np.linalg.norm(n),1e-9); bv=np.cross(t,n); bv=bv/max(np.linalg.norm(bv),1e-9); return n,bv


def _plane(ref,source,center,normal,binormal,chamber_masks=None,half=7.0,step=0.20):
    q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij"); pts=center[None,None,:]+xx[...,None]*normal[None,None,:]+yy[...,None]*binormal[None,None,:]; ct=_sample(source,ref,pts.reshape(-1,3),1,-1024).reshape(len(q),len(q)); masks={}
    if chamber_masks:
        for name,arr in chamber_masks.items(): masks[name]=_sample(arr.astype(np.float32),ref,pts.reshape(-1,3),0,0).reshape(len(q),len(q))
    return ct,masks,q


def _angle_score_from_vectors(tangent,n_a,n_v):
    tangent=np.asarray(tangent,float); tangent/=np.linalg.norm(tangent); n_a=np.asarray(n_a,float); n_a/=np.linalg.norm(n_a); n_v=np.asarray(n_v,float); n_v/=np.linalg.norm(n_v); g=np.cross(n_a,n_v); g/=np.linalg.norm(g); return float(abs(np.dot(tangent,g)))


def synthetic_local_bifurcation_self_test():
    n_a=np.array([1.,0.,0.]); n_v=np.array([0.,1.,0.]); groove=np.array([0.,0.,1.]); om=np.array([1.,0.,1.]); om/=np.linalg.norm(om)
    assert _angle_score_from_vectors(groove,n_a,n_v)>0.99
    assert _angle_score_from_vectors(groove,n_a,n_v)>_angle_score_from_vectors(om,n_a,n_v)
    t=np.linspace(0,20,81); trunk=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)]); p1=trunk.copy(); p2=trunk.copy(); p2[t>15,1]=(t[t>15]-15)*0.8; p3=trunk.copy(); p3[t>15,2]=(t[t>15]-15)*0.8
    _,_,_,meta=_consensus_prefix([p1,p2,p3],0.25,0.4,3); assert 14.5<=meta["common_prefix_total_mm"]<=15.5
    return {"ok":True,"synthetic_consensus_mm":meta["common_prefix_total_mm"]}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD_PATH,root/RCA_PATH,root/COR_CURRENT,root/COR_LEGACY,root/LA,root/LV,root/RA,root/RV,root/MYO,root/PRIOR_RANKING]+[root/p for p in LEAF_FILES]
    for p in required:_req(p)
    master=json.loads((root/MASTER).read_text()); ref,source=_source(root/SOURCE_CACHE); spacing_zyx=np.asarray(ref.GetSpacing()[::-1],float)
    current=_resample_mask(root/COR_CURRENT,ref); legacy=_resample_mask(root/COR_LEGACY,ref); la=_resample_mask(root/LA,ref); lv=_resample_mask(root/LV,ref); ra=_resample_mask(root/RA,ref); rv=_resample_mask(root/RV,ref); myo=_resample_mask(root/MYO,ref)
    dLA=_distance_to_surface(la,spacing_zyx); dLV=_distance_to_surface(lv,spacing_zyx); dRA=_distance_to_surface(ra,spacing_zyx); dRV=_distance_to_surface(rv,spacing_zyx); dMYO=_distance_to_surface(myo,spacing_zyx)
    gLA=_distance_gradient_lps(dLA,ref); gLV=_distance_gradient_lps(dLV,ref); gRA=_distance_gradient_lps(dRA,ref); gRV=_distance_gradient_lps(dRV,ref)
    lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref); leaves=_orient_common([_load_path(root/p,ref) for p in LEAF_FILES]); ranking=pd.read_csv(root/PRIOR_RANKING); prior_by_file={i:ranking.iloc[i-1].to_dict() for i in range(1,6)}
    consensus,q_common,dev,cons_meta=_consensus_prefix(leaves,0.25,0.60,4); d_lad=_nearest_dist(consensus,lad); div_i=_first_sustained(d_lad,1.50,0.25,1.0); div_i=len(consensus)-1 if div_i is None else div_i; trunk=consensus[div_i:].copy(); trunk_arc=_arc(trunk)
    trunk_current=_support_fraction(current,ref,trunk); trunk_legacy=_support_fraction(legacy,ref,trunk); trunk_hu_frac,trunk_hu_med=_robust_hu(source,ref,trunk); trunk_ok=bool(trunk_arc[-1]>=10 and trunk_current>=.90 and trunk_legacy>=.90 and trunk_hu_frac>=.90)
    pd.DataFrame(trunk,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).assign(arc_mm=trunk_arc,lad_distance_mm=d_lad[div_i:]).to_csv(out/"consensus_left_coronary_branch_trunk.csv",index=False)
    common_end=float(q_common[-1]); min_remaining=min(_arc(p)[-1]-common_end for p in leaves); probe_delta=max(1.0,min(4.0,.5*min_remaining)); probe_arc=common_end+probe_delta; probe_points=np.vstack([_interp_path(p,[probe_arc])[0] for p in leaves]); groups=_union_find_groups(probe_points,1.5); family_of={}
    for fi,g in enumerate(groups,1):
        for idx in g:family_of[idx]=fi
    q0=max(0.,common_end-2.0); inc_pts=_interp_path(consensus,[q0,common_end]); incoming_tangent=inc_pts[-1]-inc_pts[0]; incoming_tangent/=max(np.linalg.norm(incoming_tangent),1e-9)
    rca_ctrl=_control_alignment_metrics(rca,dRA,dRV,gRA,gRV,ref); lad_ctrl=_control_alignment_metrics(lad,dLA,dLV,gLA,gLV,ref); ctrl_margin=float(rca_ctrl["score"]-lad_ctrl["score"]); control_pass=bool(rca_ctrl["score"]>=.45 and ctrl_margin>=.08 and rca_ctrl["n_valid"]>=5)
    controls={"RCA_right_AV_local_tangent_score":rca_ctrl["score"],"RCA_median_groove_tangent_angle_deg":rca_ctrl["median_angle_deg"],"LAD_left_AV_local_tangent_negative_score":lad_ctrl["score"],"LAD_median_groove_tangent_angle_deg":lad_ctrl["median_angle_deg"],"control_margin":ctrl_margin,"control_pass":control_pass}; _write_json(out/"local_tangent_controls.json",controls)
    leaf_rows=[]; profiles={}
    for idx,p in enumerate(leaves):
        total=_arc(p)[-1]; start=min(common_end,total-.5); q=np.arange(start,total+1e-9,.25)
        if len(q)<3:q=np.linspace(start,total,max(3,int(round((total-start)/.25))+1))
        cont=_interp_path(p,q); met=_local_alignment_metrics(cont,dLA,dLV,gLA,gLV,ref,incoming_tangent,4.0); cur_sup=_support_fraction(current,ref,cont); leg_sup=_support_fraction(legacy,ref,cont); hu_frac,hu_med=_robust_hu(source,ref,cont); myo_dist=_sample(dMYO,ref,met["points"],1,100.); prior=prior_by_file[idx+1]; rel_rca=met["local_geometry_score"]/max(rca_ctrl["score"],1e-6); above_lad=met["local_geometry_score"]-lad_ctrl["score"]; gate=bool(control_pass and met["local_length_mm"]>=2.0 and cur_sup>=.90 and leg_sup>=.90 and hu_frac>=.90 and rel_rca>=.65 and above_lad>=.05)
        row={"leaf_index":idx+1,"source_candidate_id":int(prior["candidate_id"]),"family_id":family_of[idx],"continuation_total_mm":float(total-common_end),"local_geometry_score":met["local_geometry_score"],"relative_to_RCA_control":rel_rca,"margin_over_LAD_negative":above_lad,"median_groove_tangent_angle_deg":met["median_groove_tangent_angle_deg"],"p90_groove_tangent_angle_deg":met["p90_groove_tangent_angle_deg"],"local_groove_alignment_cos":met["local_groove_alignment_cos"],"left_atrium_surface_tangency":met["left_atrium_surface_tangency"],"left_ventricle_surface_tangency":met["left_ventricle_surface_tangency"],"left_atrium_distance_slope_mm_per_mm":met["left_atrium_distance_slope_mm_per_mm"],"atrium_retention_score":met["atrium_retention_score"],"incoming_continuation_angle_deg":met["incoming_continuation_angle_deg"],"incoming_continuity_score":met["incoming_continuity_score"],"median_myocardium_surface_mm":float(np.median(myo_dist)),"current_support_fraction":cur_sup,"legacy_support_fraction":leg_sup,"robust_hu_fraction":hu_frac,"median_hu":hu_med,"prior_LCX_score":float(prior["LCX_score"]),"prior_LCX_margin":float(prior["LCX_margin"]),"local_gate_pass":gate}; leaf_rows.append(row); profiles[idx+1]=met
    leaf_df=pd.DataFrame(leaf_rows).sort_values(["local_geometry_score","continuation_total_mm"],ascending=False).reset_index(drop=True); leaf_df.insert(0,"local_rank",np.arange(1,len(leaf_df)+1)); leaf_df.to_csv(out/"local_leaf_geometry_scores.csv",index=False)
    family_rows=[]
    for fi,g in enumerate(groups,1):
        sub=leaf_df[leaf_df["family_id"]==fi]; family_score=float(np.median(sub["local_geometry_score"])); best=sub.sort_values("local_geometry_score",ascending=False).iloc[0]; family_rows.append({"family_id":fi,"n_leaf_members":int(len(sub)),"member_leaf_indices":";".join(str(int(x)) for x in sorted(sub["leaf_index"])),"family_local_geometry_score":family_score,"best_leaf_index":int(best["leaf_index"]),"best_source_candidate_id":int(best["source_candidate_id"]),"family_median_groove_tangent_angle_deg":float(np.median(sub["median_groove_tangent_angle_deg"])),"family_median_incoming_continuation_angle_deg":float(np.median(sub["incoming_continuation_angle_deg"])),"relative_to_RCA_control":family_score/max(rca_ctrl["score"],1e-6),"margin_over_LAD_negative":family_score-lad_ctrl["score"],"family_gate_pass":bool(np.any(sub["local_gate_pass"]) and family_score/max(rca_ctrl["score"],1e-6)>=.65 and family_score-lad_ctrl["score"]>=.05)})
    fam_df=pd.DataFrame(family_rows).sort_values("family_local_geometry_score",ascending=False).reset_index(drop=True); fam_df.insert(0,"family_rank",np.arange(1,len(fam_df)+1)); fam_df.to_csv(out/"local_distal_family_scores.csv",index=False)
    fig=plt.figure(figsize=(11,8)); ax=fig.add_subplot(111,projection="3d"); ax.plot(trunk[:,0],trunk[:,1],trunk[:,2],linewidth=5,label="consensus post-LAD trunk")
    for fi,g in enumerate(groups,1):
        for idx in g:
            total=_arc(leaves[idx])[-1]; q=np.arange(common_end,total+1e-9,.25); cont=_interp_path(leaves[idx],q); ax.plot(cont[:,0],cont[:,1],cont[:,2],linewidth=2,label=f"family {fi} leaf {idx+1}" if idx==g[0] else None)
    ax.scatter([trunk[-1,0]],[trunk[-1,1]],[trunk[-1,2]],s=60,marker="o",label="distal split"); ax.set_title("Local distal bifurcation geometry"); ax.set_xlabel("LPS x (mm)"); ax.set_ylabel("LPS y (mm)"); ax.set_zlabel("LPS z (mm)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"01_local_bifurcation_geometry.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); labels=["RCA +"]+[f"F{int(r.family_id)}" for _,r in fam_df.iterrows()]+["LAD -"]; vals=[rca_ctrl["score"]]+fam_df["family_local_geometry_score"].tolist()+[lad_ctrl["score"]]; ax.bar(np.arange(len(vals)),vals); ax.set_xticks(np.arange(len(vals)),labels); ax.set_ylim(0,1); ax.set_ylabel("local AV-groove tangent score"); ax.set_title("Local groove-tangent calibration and distal families"); fig.tight_layout(); fig.savefig(out/"02_local_tangent_scores.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for _,row in leaf_df.iterrows():
        prof=profiles[int(row["leaf_index"])]; a=prof["profile_groove_angle_deg"]
        if len(a):ax.plot(np.linspace(0,prof["local_length_mm"],len(a)),a,label=f"F{int(row['family_id'])}/leaf{int(row['leaf_index'])}")
    ax.set_xlabel("distance after distal split (mm)"); ax.set_ylabel("angle to local LA-LV groove tangent (deg)"); ax.set_title("Local tangent-angle profiles"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"03_local_angle_profiles.png",dpi=160); plt.close(fig)
    nshow=min(2,len(fam_df)); fig,axes=plt.subplots(nshow,3,figsize=(12,4*nshow),squeeze=False)
    for rr in range(nshow):
        famrow=fam_df.iloc[rr]; leaf_idx=int(famrow["best_leaf_index"]); pts=profiles[leaf_idx]["points"]; sample_idx=[0,min(len(pts)-1,max(1,len(pts)//2)),len(pts)-1]
        for cc,ii in enumerate(sample_idx):
            n,bv=_frame(pts,ii); ct,masks,qpl=_plane(ref,source,pts[ii],n,bv,{"LA":la,"LV":lv},7,.20); ax=axes[rr,cc]; ax.imshow(ct,cmap="gray",vmin=0,vmax=900,extent=[qpl[0],qpl[-1],qpl[-1],qpl[0]])
            for _,arr in masks.items():
                if arr.max()>0:
                    try:ax.contour(qpl,qpl,arr,levels=[.5],linewidths=1.2)
                    except Exception:pass
            ax.scatter([0],[0],s=25,marker="+"); ax.set_title(f"F{int(famrow['family_id'])} leaf {leaf_idx}, {['split','mid','local-end'][cc]}"); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    fig.suptitle("Top distal families: local source-CCTA with LA/LV contours"); fig.tight_layout(); fig.savefig(out/"04_top_families_orthogonal_qc.png",dpi=160); plt.close(fig)
    top=fam_df.iloc[0].to_dict() if len(fam_df) else None; second=fam_df.iloc[1].to_dict() if len(fam_df)>1 else None; unique_margin=float(top["family_local_geometry_score"]-second["family_local_geometry_score"]) if top and second else None; unique=bool(trunk_ok and control_pass and top is not None and bool(top["family_gate_pass"]) and (second is None or unique_margin>=.05))
    status=STATUS_NO_TRUNK if not trunk_ok else STATUS_CONTROL_FAIL if not control_pass else STATUS_CANDIDATE if unique else STATUS_AMBIG
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"LCX_master_status":"UNRESOLVED","consensus_trunk":{**cons_meta,"lad_divergence_arc_from_candidate_origin_mm":float(q_common[div_i]),"post_lad_divergence_common_trunk_mm":float(trunk_arc[-1]),"current_support_fraction":trunk_current,"legacy_support_fraction":trunk_legacy,"robust_hu_fraction":trunk_hu_frac,"median_hu":trunk_hu_med,"gate_pass":trunk_ok},"local_tangent_controls":controls,"n_distal_families":int(len(fam_df)),"top_family":top,"second_family":second,"top_family_margin":unique_margin,"template_similarity_used_in_decision":False,"scientific_boundary":"The consensus post-LAD branch trunk is source-space anatomy. Local chamber-surface differential geometry may nominate an LCX-like distal continuation but does not establish LM identity, clinical vessel labeling, circumferential plaque registration, or plaque volume."}; _write_json(out/"summary.json",summary)
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>OpenPlaque LCX local bifurcation</title><style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:30px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;font-size:13px}}img{{max-width:100%}}</style></head><body><h1>OpenPlaque — LCX local bifurcation / AV-groove tangent</h1><p><b>Status:</b> {status}</p><p><b>Algorithm:</b> {ALGORITHM}</p><p>Consensus post-LAD trunk: {trunk_arc[-1]:.3f} mm; current/legacy support {trunk_current:.3f}/{trunk_legacy:.3f}; HU support {trunk_hu_frac:.3f}.</p><p>RCA positive local groove score {rca_ctrl['score']:.3f}; LAD negative {lad_ctrl['score']:.3f}; control pass {control_pass}.</p><h2>Distal families</h2>{fam_df.to_html(index=False)}<h2>Leaf metrics</h2>{leaf_df.to_html(index=False)}<h2>Figures</h2><img src="01_local_bifurcation_geometry.png"><img src="02_local_tangent_scores.png"><img src="03_local_angle_profiles.png"><img src="04_top_families_orthogonal_qc.png"><p><b>Boundary:</b> {summary['scientific_boundary']}</p></body></html>'''; report=out/"OPENPLAQUE_LCX_LOCAL_BIFURCATION_GROOVE_TANGENT_REPORT.html"; report.write_text(html,encoding="utf-8")
    zip_path=out/"OPENPLAQUE_LCX_LOCAL_BIFURCATION_GROOVE_TANGENT_REPORT_BACK.zip"; _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zip_path:z.write(p,arcname=p.name)
    return {"summary":summary,"report":str(report),"zip":str(zip_path)}


if __name__ == "__main__":
    print(synthetic_local_bifurcation_self_test())
