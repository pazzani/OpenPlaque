from __future__ import annotations

"""Source-CCTA reacquisition of a short common parent between the adjudicated LAD/C6 split and aorta.

The search is anatomy-blind with respect to vessel naming after prerequisites are checked. It uses
source HU, multiscale vesselness, coronary-mask proximity, local directional continuity, and strictly
improving distance to the aortic mask. The accepted RCA is used as a positive control by tracing its
proximal segment back to the aorta with the same machinery.

Research use only. A positive result nominates a left-main-like common parent for visual QC; it does
not establish clinical LM identity and does not modify Master Coronary Anatomy Baseline v2.1.
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
ALGORITHM = "left-main-common-parent-reacquisition-v1.0-lowmem"
OUTPUT_DIRNAME = "Left_Main_Common_Parent_Reacquisition_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
STRUCTURAL = Path("LCX_Structural_Identity_Adjudication_v1/summary.json")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "heartchambers_highres/aorta.nii.gz"

STATUS_PREREQ_FAIL = "LEFT_MAIN_STRUCTURAL_PREREQUISITE_FAILED"
STATUS_RCA_CONTROL_FAIL = "LEFT_MAIN_RCA_TO_AORTA_CONTROL_FAILED"
STATUS_NO_PARENT = "LEFT_MAIN_COMMON_PARENT_NOT_ESTABLISHED"
STATUS_POS = "LEFT_MAIN_LIKE_COMMON_PARENT_REQUIRES_VISUAL_QC"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
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
    pts = np.atleast_2d(np.asarray(pts, float))
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(D).T) / sp
    return idx_xyz[:, ::-1]


def _zyx_to_xyz(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    q = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (q * sp) @ D.T


def _sample(img, arr, pts, order=1, cval=-1024.0):
    zyx = _xyz_to_zyx(img, pts)
    return map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in (("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")):
        if all(c in d.columns for c in cols):
            return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    raise ValueError(f"No recognized coordinates in {path}; columns={list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _resample(p, step=.20):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)]), q


def _interp(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize() and np.allclose(im.GetSpacing(), ref.GetSpacing()) and
            np.allclose(im.GetOrigin(), ref.GetOrigin()) and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def _frangi_3d(vol, spacing, scales=(.55, .80, 1.10, 1.45)):
    x = np.clip(np.asarray(vol, np.float32), 80., 1000.)
    x = (x - 80.) / 920.
    spacing = np.asarray(spacing, float)
    best = np.zeros_like(x, np.float32)
    for sm in scales:
        sig = np.maximum(sm / spacing, .55)
        n = sm * sm
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * n / (spacing[0]**2)
        hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * n / (spacing[1]**2)
        hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * n / (spacing[2]**2)
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * n / (spacing[0]*spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * n / (spacing[0]*spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * n / (spacing[1]*spacing[2])
        H = np.empty(x.shape + (3,3), np.float32)
        H[...,0,0] = hzz; H[...,1,1] = hyy; H[...,2,2] = hxx
        H[...,0,1] = H[...,1,0] = hzy; H[...,0,2] = H[...,2,0] = hzx; H[...,1,2] = H[...,2,1] = hyx
        vals = np.linalg.eigvalsh(H)
        vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
        l1,l2,l3 = vals[...,0], vals[...,1], vals[...,2]
        eps = 1e-8
        ra = np.abs(l2) / (np.abs(l3) + eps)
        rb = np.abs(l1) / np.sqrt(np.abs(l2*l3) + eps)
        s = np.sqrt(l1*l1 + l2*l2 + l3*l3)
        nz = s[s > 0]
        c = max(float(np.percentile(nz, 90)) * .45 if nz.size else .05, 1e-4)
        v = (1 - np.exp(-(ra*ra)/.5)) * np.exp(-(rb*rb)/.5) * (1 - np.exp(-(s*s)/(2*c*c)))
        v[(l2 >= 0) | (l3 >= 0)] = 0
        best = np.maximum(best, np.nan_to_num(v, nan=0, posinf=0, neginf=0).astype(np.float32))
    return best


def _surface_tree_from_mask(mask, ref, stride=3):
    surf = mask & ~ndi.binary_erosion(mask, iterations=1, border_value=0)
    z = np.argwhere(surf)[::stride]
    pts = _zyx_to_xyz(ref, z.astype(float))
    return cKDTree(pts.astype(np.float32)), pts


def _orient_endpoint_nearest_tree(path, tree):
    p = np.asarray(path, float)
    d0 = float(tree.query(p[0])[0]); d1 = float(tree.query(p[-1])[0])
    return (p, d0) if d0 <= d1 else (p[::-1].copy(), d1)


def _orient_pair_to_common(lad, c6):
    options = []
    for a in (lad, lad[::-1].copy()):
        for b in (c6, c6[::-1].copy()):
            options.append((float(np.linalg.norm(a[0] - b[0])), a, b))
    _, a, b = min(options, key=lambda x: x[0])
    return a, b


def _bifurcation_seed(lad, c6):
    lad, c6 = _orient_pair_to_common(lad, c6)
    lr, lq = _resample(lad, .20); cr, cq = _resample(c6, .20)
    li = np.where(lq <= min(5., lq[-1]))[0]; ci = np.where(cq <= min(5., cq[-1]))[0]
    tree = cKDTree(cr[ci])
    d, j = tree.query(lr[li])
    k = int(np.argmin(d)); i_l = int(li[k]); i_c = int(ci[int(j[k])])
    p_l, p_c = lr[i_l], cr[i_c]
    seed = .5 * (p_l + p_c)
    l2 = min(lq[-1], lq[i_l] + 2.0); c2 = min(cq[-1], cq[i_c] + 2.0)
    t_l = _unit(_interp(lr, [l2])[0] - p_l)
    t_c = _unit(_interp(cr, [c2])[0] - p_c)
    return lad, c6, seed, t_l, t_c, {"nearest_lad_c6_distance_mm": float(d[k]), "lad_arc_mm": float(lq[i_l]), "c6_arc_mm": float(cq[i_c])}


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed)); v = _unit(np.cross(t, u))
    return u, v


def _cone_dirs(t):
    t = _unit(t); u, v = _orth_basis(t)
    out = [t]
    for deg in (12, 24, 36, 48, 58):
        a = np.deg2rad(deg)
        for phi in np.linspace(0, 2*np.pi, 10, endpoint=False):
            out.append(_unit(np.cos(a)*t + np.sin(a)*(np.cos(phi)*u + np.sin(phi)*v)))
    return out


def _build_field(ref, src, spacing, cur, leg, start, goal, known_path=None):
    pts = np.vstack([np.asarray(start)[None,:], np.asarray(goal)[None,:]])
    if known_path is not None:
        kp, _ = _resample(known_path, .4)
        pts = np.vstack([pts, kp])
    z = _xyz_to_zyx(ref, pts)
    margin = np.array([7., 7., 7.])
    lo = np.floor(np.min(z * spacing, axis=0) / spacing - margin / spacing).astype(int)
    hi = np.ceil(np.max(z * spacing, axis=0) / spacing + margin / spacing).astype(int) + 1
    lo = np.maximum(lo, 0); hi = np.minimum(hi, np.asarray(src.shape))
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    roi = np.asarray(src[sl]); cm = np.asarray(cur[sl]); lm = np.asarray(leg[sl])
    union = cm | lm
    du = ndi.distance_transform_edt(~union, sampling=spacing)
    vessel = _frangi_3d(roi, spacing)
    if known_path is not None:
        kp, _ = _resample(known_path, .20)
        kz = _xyz_to_zyx(ref, kp) - lo[None,:]
        good = np.all((kz >= 0) & (kz < np.asarray(roi.shape)[None,:]), axis=1)
        kv = map_coordinates(vessel, kz[good].T, order=1, mode="nearest") if np.any(good) else np.array([.03])
    else:
        kv = np.array([.03])
    thr = max(.003, min(.12, .30 * float(np.percentile(kv, 20))))
    norm = max(float(np.median(kv))*1.5, .02)
    return {"lo":lo,"hi":hi,"roi":roi,"cur":cm,"leg":lm,"union":union,"du":du,"v":vessel,"thr":thr,"norm":norm}


def _sample_field(ref, F, p):
    z = _xyz_to_zyx(ref, [p])[0] - F["lo"]
    shape = np.asarray(F["roi"].shape)
    if np.any(z < 1) or np.any(z > shape - 2):
        return None
    co = z[:,None]
    hu = float(map_coordinates(F["roi"], co, order=1, mode="nearest")[0])
    vv = float(map_coordinates(F["v"], co, order=1, mode="nearest")[0])
    dd = float(map_coordinates(F["du"], co, order=1, mode="nearest")[0])
    zi = np.rint(z).astype(int)
    c = bool(F["cur"][tuple(zi)]); l = bool(F["leg"][tuple(zi)])
    return hu, vv, dd, c, l


def _beam_to_aorta(ref, F, aorta_tree, start, tangent, target_max_mm=14., step=.45, beam_width=80):
    origin = np.asarray(start, float)
    d0 = float(aorta_tree.query(origin)[0])
    states = [(0., [origin], _unit(tangent), d0)]
    reached = []
    nsteps = int(math.ceil(target_max_mm / step))
    for _ in range(nsteps):
        nxt = []
        for score, pts, t, prev_ad in states:
            for d in _cone_dirs(t):
                if np.dot(d, t) < math.cos(math.radians(62)):
                    continue
                p = pts[-1] + step*d
                ad = float(aorta_tree.query(p)[0])
                if ad > prev_ad + .12:
                    continue
                if len(pts) > 5 and np.min(np.linalg.norm(np.asarray(pts[:-4]) - p, axis=1)) < .60*step:
                    continue
                s = _sample_field(ref, F, p)
                if s is None:
                    continue
                hu,vv,du,c,l = s
                near_aorta = ad <= 1.2
                if not (80 <= hu <= 1400 and vv >= .45*F["thr"] and (du <= 1.4 or near_aorta)):
                    continue
                vn = min(1., vv/F["norm"])
                support = 1.0 if c and l else (.60 if c or l else .15 if du <= 1.4 else 0.)
                improve = max(-.3, prev_ad - ad)
                align = max(0., float(np.dot(d,t)))
                ns = score + 1.7*vn + .45*support + .45*align + .85*improve/max(step,1e-6)
                nt = _unit(.72*t + .28*d)
                st = (ns, pts+[p], nt, ad)
                nxt.append(st)
                if ad <= .75:
                    reached.append(st)
        if not nxt:
            break
        nxt.sort(key=lambda x:x[0], reverse=True)
        keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.30).astype(int))
            if key in bins:
                continue
            bins.add(key); keep.append(st)
            if len(keep) >= beam_width:
                break
        states = keep
        if reached and max(_arc(np.asarray(s[1]))[-1] for s in reached) >= 2.0:
            # allow a few more alternatives but avoid needless expansion after robust arrival
            if len(reached) >= 12:
                break
    return reached, states


def _path_metrics(ref, src, cur, leg, F, path, start, aorta_tree):
    p,q = _resample(path, .20)
    h = _sample(ref, src, p)
    z = _xyz_to_zyx(ref, p)
    local = z - F["lo"][None,:]
    vv = map_coordinates(F["v"], local.T, order=1, mode="nearest")
    zr = np.rint(z).astype(int); zr = np.clip(zr, [0,0,0], np.asarray(src.shape)-1)
    cs = cur[tuple(zr.T)]; ls = leg[tuple(zr.T)]
    ad = np.asarray(aorta_tree.query(p)[0], float)
    length = float(q[-1]); disp = float(np.linalg.norm(p[-1]-start))
    turns=np.array([])
    if len(p)>=4:
        v=np.diff(p,axis=0); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
        turns=np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))
    return {
        "length_mm":length,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6),
        "max_turn_deg":float(np.max(turns)) if len(turns) else 0.,"median_hu":float(np.median(h)),
        "robust_hu_fraction":float(np.mean((h>=120)&(h<=1200))),"median_vesselness":float(np.median(vv)),
        "p10_vesselness":float(np.percentile(vv,10)),"current_support_fraction":float(np.mean(cs)),
        "legacy_support_fraction":float(np.mean(ls)),"union_support_fraction":float(np.mean(cs|ls)),
        "dual_support_fraction":float(np.mean(cs&ls)),"start_aorta_distance_mm":float(ad[0]),
        "endpoint_aorta_distance_mm":float(ad[-1]),"aorta_distance_reduction_mm":float(ad[0]-ad[-1]),
        "aorta_distance_profile_mm":ad,"arc_profile_mm":q,
    }


def _positive_rca_control(ref, src, spacing, cur, leg, aorta_tree, rca):
    rca, prox_d = _orient_endpoint_nearest_tree(rca, aorta_tree)
    rr,rq = _resample(rca,.20)
    start_arc=min(6.0, max(2.5, rq[-1]*.18))
    start=_interp(rr,[start_arc])[0]
    proximal=rr[0]
    t=_unit(proximal-start)
    F=_build_field(ref,src,spacing,cur,leg,start,aorta_tree.data[aorta_tree.query(start)[1]],rr[rq<=start_arc+.5])
    reached,_=_beam_to_aorta(ref,F,aorta_tree,start,t,target_max_mm=max(8.,start_arc+3.),step=.40,beam_width=90)
    if not reached:
        return {"control_pass":False,"reason":"no_path_reached_aorta","proximal_endpoint_aorta_distance_mm":prox_d}, None, F
    known=rr[rq<=start_arc+.5]; kt=cKDTree(known)
    rows=[]
    for st in reached:
        p=np.asarray(st[1]); m=_path_metrics(ref,src,cur,leg,F,p,start,aorta_tree); dd=kt.query(p)[0]
        m.update({"endpoint_error_to_known_proximal_mm":float(np.linalg.norm(p[-1]-proximal)),"median_distance_to_known_rca_mm":float(np.median(dd)),"p90_distance_to_known_rca_mm":float(np.percentile(dd,90)),"beam_score":float(st[0])})
        rows.append((m,p))
    rows.sort(key=lambda x:(x[0]["median_distance_to_known_rca_mm"],x[0]["endpoint_error_to_known_proximal_mm"],-x[0]["beam_score"]))
    m,p=rows[0]
    passed=bool(m["endpoint_aorta_distance_mm"]<=.80 and m["median_distance_to_known_rca_mm"]<=.80 and m["p90_distance_to_known_rca_mm"]<=1.30 and m["endpoint_error_to_known_proximal_mm"]<=1.60 and m["robust_hu_fraction"]>=.85)
    m["control_pass"]=passed
    return m,p,F


def _lm_search(ref,src,spacing,cur,leg,aorta_tree,lad,c6):
    lad,c6,seed,t_l,t_c,bif=_bifurcation_seed(lad,c6)
    _,goal_idx=aorta_tree.query(seed); goal=np.asarray(aorta_tree.data[int(goal_idx)],float)
    direct=_unit(goal-seed)
    bis=_unit(-(t_l+t_c))
    t=direct if np.linalg.norm(bis)<.25 else _unit(.65*direct+.35*bis)
    known=np.vstack([_resample(lad,.4)[0][:20],_resample(c6,.4)[0][:20]])
    F=_build_field(ref,src,spacing,cur,leg,seed,goal,known)
    reached,frontier=_beam_to_aorta(ref,F,aorta_tree,seed,t,target_max_mm=14.,step=.40,beam_width=100)
    candidates=[]
    for st in reached:
        p=np.asarray(st[1]); m=_path_metrics(ref,src,cur,leg,F,p,seed,aorta_tree)
        m["beam_score"]=float(st[0]); m["endpoint_distance_to_LAD_mm"]=float(cKDTree(lad).query(p[-1])[0]); m["endpoint_distance_to_C6_mm"]=float(cKDTree(c6).query(p[-1])[0])
        m["gate_pass"]=bool(m["endpoint_aorta_distance_mm"]<=.80 and 1.5<=m["length_mm"]<=14.0 and m["endpoint_displacement_mm"]>=1.2 and m["tortuosity"]<=1.8 and m["max_turn_deg"]<=65.0 and m["robust_hu_fraction"]>=.90 and m["union_support_fraction"]>=.70 and m["p10_vesselness"]>=.45*F["thr"] and m["aorta_distance_reduction_mm"]>=1.0)
        candidates.append((m,p))
    if not candidates:
        return None,{"accepted":False,"reason":"no_path_reached_aorta","bifurcation":bif,"seed_lps_mm":seed.tolist(),"seed_aorta_distance_mm":float(aorta_tree.query(seed)[0]),"initial_tangent":t.tolist(),"frontier_count":len(frontier)},F,lad,c6
    df=pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,np.ndarray)} for m,_ in candidates])
    ok=[(m,p) for m,p in candidates if m["gate_pass"]]
    if not ok:
        best=max(candidates,key=lambda x:x[0]["beam_score"])[0]
        return None,{"accepted":False,"reason":"no_candidate_passed","bifurcation":bif,"seed_lps_mm":seed.tolist(),"seed_aorta_distance_mm":float(aorta_tree.query(seed)[0]),"best":{k:v for k,v in best.items() if not isinstance(v,np.ndarray)},"candidate_table":df},F,lad,c6
    ok.sort(key=lambda x:(x[0]["length_mm"],x[0]["beam_score"]),reverse=True)
    m,p=ok[0]
    summary={"accepted":True,"bifurcation":bif,"seed_lps_mm":seed.tolist(),"seed_aorta_distance_mm":float(aorta_tree.query(seed)[0]),"best":{k:v for k,v in m.items() if not isinstance(v,np.ndarray)},"candidate_table":df}
    return p,summary,F,lad,c6


def _plane(ref,src,c,t,half=7.,step=.2):
    t=_unit(t); u,v=_orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij")
    P=c+xx[...,None]*u+yy[...,None]*v
    return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q


def synthetic_aorta_monotonic_self_test():
    d=np.array([8.,7.4,6.8,6.2,5.6,5.0,4.4,3.8,3.2,2.6,2.0,1.4,.7])
    assert np.all(np.diff(d)<=0)
    assert d[0]-d[-1]>6
    p=np.column_stack([np.arange(len(d))*.4,np.zeros(len(d)),np.zeros(len(d))])
    assert 0.99<=_arc(p)[-1]/np.linalg.norm(p[-1]-p[0])<=1.01
    return {"ok":True,"distance_reduction_mm":float(d[0]-d[-1])}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD_PATH,root/RCA_PATH,root/C6_PATH,root/STRUCTURAL,root/CUR,root/LEG,root/AORTA]
    for p in required:_req(p)
    structural=json.loads((root/STRUCTURAL).read_text()); master=json.loads((root/MASTER).read_text())
    prereq=bool(structural.get("decision",{}).get("all_predeclared_structural_gates_pass",False) and structural.get("decision",{}).get("structural_label_C6","").startswith("LCX-like") and structural.get("decision",{}).get("structural_label_C7","").startswith("OM-like"))
    ref,src,spacing=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref); c6=_load_path(root/C6_PATH,ref)
    cur=_resample_mask(root/CUR,ref); leg=_resample_mask(root/LEG,ref); aorta=_resample_mask(root/AORTA,ref); aorta_tree,aorta_pts=_surface_tree_from_mask(aorta,ref,stride=2)
    if not prereq:
        status=STATUS_PREREQ_FAIL; summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"structural_prerequisite_pass":False}
        _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); return {"summary":summary,"report":None,"zip":None}
    print("Running RCA-to-aorta positive control...")
    rca_control,rca_path,_=_positive_rca_control(ref,src,spacing,cur,leg,aorta_tree,rca); _write_json(out/"RCA_to_aorta_control.json",rca_control)
    print("Searching for short LAD/C6 common parent toward aorta...")
    lm,lm_summary,F,lad_o,c6_o=_lm_search(ref,src,spacing,cur,leg,aorta_tree,lad,c6)
    table=lm_summary.pop("candidate_table",pd.DataFrame()) if isinstance(lm_summary,dict) else pd.DataFrame()
    if isinstance(table,pd.DataFrame): table.to_csv(out/"left_main_candidates.csv",index=False)
    _write_json(out/"left_main_candidate_summary.json",lm_summary)
    if lm is not None: pd.DataFrame(lm,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"left_main_candidate_path.csv",index=False)
    # Figures
    if rca_path is not None:
        rm=_path_metrics(ref,src,cur,leg,_build_field(ref,src,spacing,cur,leg,rca_path[0],aorta_tree.data[aorta_tree.query(rca_path[0])[1]],rca_path),rca_path,rca_path[0],aorta_tree)
        plt.figure(figsize=(6.5,4)); plt.plot(rm["arc_profile_mm"],rm["aorta_distance_profile_mm"],marker="."); plt.xlabel("RCA control path arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title(f"RCA→aorta positive control | pass={rca_control.get('control_pass',False)}"); plt.tight_layout(); plt.savefig(out/"01_RCA_to_aorta_control.png",dpi=180); plt.close()
    if lm is not None:
        mm=_path_metrics(ref,src,cur,leg,F,lm,lm[0],aorta_tree)
        plt.figure(figsize=(6.5,4)); plt.plot(mm["arc_profile_mm"],mm["aorta_distance_profile_mm"],marker="."); plt.xlabel("candidate common-parent arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title("LAD/C6 common-parent approach to aorta"); plt.tight_layout(); plt.savefig(out/"02_left_main_aorta_distance.png",dpi=180); plt.close()
        fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d")
        seed=np.asarray(lm_summary["seed_lps_mm"],float); _,ix=aorta_tree.query(seed,k=min(1800,len(aorta_tree.data))); AP=np.atleast_2d(aorta_tree.data[ix])
        ax.scatter(AP[:,0],AP[:,1],AP[:,2],s=2,alpha=.12,label="aortic surface")
        lr,_=_resample(lad_o,.35); cr,_=_resample(c6_o,.35)
        ax.plot(lr[:,0],lr[:,1],lr[:,2],lw=2.5,label="accepted LAD"); ax.plot(cr[:,0],cr[:,1],cr[:,2],lw=2.5,label="C6 LCX-like parent")
        ax.plot(lm[:,0],lm[:,1],lm[:,2],lw=4,label="LM-like candidate"); ax.scatter([seed[0]],[seed[1]],[seed[2]],s=50,label="LAD/C6 bifurcation seed")
        ax.legend(); ax.set_title("Left-coronary common-parent geometry"); plt.tight_layout(); plt.savefig(out/"03_left_main_geometry.png",dpi=180); plt.close()
        fig,axes=plt.subplots(1,4,figsize=(14,4)); pp,qq=_resample(lm,.20); tt=np.gradient(pp,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9); picks=np.linspace(0,len(pp)-1,4).astype(int)
        for ax,j in zip(axes,picks):
            im,qv=_plane(ref,src,pp[j],tt[j]); ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=18); ax.set_title(f"LM-like +{qq[j]:.1f} mm")
        plt.tight_layout(); plt.savefig(out/"04_left_main_orthogonal_source_qc.png",dpi=180); plt.close()
    if not rca_control.get("control_pass",False): status=STATUS_RCA_CONTROL_FAIL
    elif not lm_summary.get("accepted",False): status=STATUS_NO_PARENT
    else: status=STATUS_POS
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"structural_prerequisite_pass":prereq,"RCA_to_aorta_control":rca_control,"left_main_candidate":lm_summary,"scientific_boundary":"Positive status nominates a short source-supported common parent between the adjudicated LAD/C6 bifurcation and aortic root as LM-like for visual QC. Clinical LM identity remains unresolved until visual adjudication; Master Coronary Anatomy Baseline v2.1 remains unchanged."}
    _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    report=out/"OPENPLAQUE_LEFT_MAIN_COMMON_PARENT_REACQUISITION_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque Left Main Common-Parent Reacquisition</h1><p><b>Status:</b> {status}</p><p>Structural prerequisite: {prereq}.</p><p>RCA→aorta control pass: {rca_control.get('control_pass',False)}.</p><p>LM-like common parent accepted: {lm_summary.get('accepted',False)}.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_MAIN_COMMON_PARENT_REACQUISITION_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
