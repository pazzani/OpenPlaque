from __future__ import annotations

"""RCA-calibrated discovery of a second coronary ostial exit at the aortic root.

This experiment deliberately does not use LAD or C6 coordinates to discover a candidate. It:
1. aligns the TotalSegmentator aorta/current/legacy coronary masks to source CCTA;
2. finds coronary-mask clusters within 2 mm of the aortic surface near the RCA ostial level;
3. identifies the RCA interface cluster automatically and uses it as a positive control;
4. traces all other root-interface clusters away from the aorta using source HU, local vesselness,
   current/legacy mask support, directional continuity, and increasing extra-aortic distance;
5. applies serial orthogonal lumen QC calibrated to the frozen RCA.

LAD/C6 distances are computed only after a candidate path is frozen and never affect selection.
Research use only. A positive result nominates a left-coronary ostial exit for visual QC and does
not establish clinical left-main identity or modify Master Coronary Anatomy Baseline v2.1.
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
ALGORITHM = "left-coronary-ostium-mask-consensus-v1.0-lowmem"
OUTPUT_DIRNAME = "Left_Coronary_Ostium_Mask_Consensus_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "heartchambers_highres/aorta.nii.gz"

STATUS_RCA_INTERFACE_FAIL = "OSTIUM_CONSENSUS_RCA_INTERFACE_CONTROL_FAILED"
STATUS_NO_SECOND = "SECOND_CORONARY_OSTIAL_EXIT_NOT_ESTABLISHED"
STATUS_POS = "LEFT_CORONARY_OSTIAL_EXIT_REQUIRES_VISUAL_QC"


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


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return _xyz_to_zyx(ref, d[list(cols)].to_numpy(float))
    for cols in (("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No recognized path coordinates in {path}; columns={list(d.columns)}")


def _arc(path_zyx, spacing_zyx):
    p = np.asarray(path_zyx, float)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def _resample(path_zyx, spacing_zyx, step_mm=.25):
    p = np.asarray(path_zyx, float)
    s = _arc(p, spacing_zyx)
    if len(p) < 2 or s[-1] <= 0:
        return p.copy(), s
    keep = np.r_[True, np.diff(s) > 1e-7]
    p, s = p[keep], s[keep]
    q = np.arange(0, s[-1] + 1e-9, step_mm)
    if len(q) == 0 or q[-1] < s[-1] - 1e-6:
        q = np.r_[q, s[-1]]
    return np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)]), q


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


def _sample_arr(arr, pts_zyx, order=1, cval=0.0):
    p = np.atleast_2d(np.asarray(pts_zyx, float))
    return map_coordinates(np.asarray(arr), p.T, order=order, mode="constant", cval=cval, prefilter=False)


def _orthogonal_basis(tangent_mm):
    t = _unit(tangent_mm)
    ref = np.array([1., 0., 0.]) if abs(t[0]) < .82 else np.array([0., 1., 0.])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def _orthogonal_plane(ct, point_zyx, tangent_mm, spacing_zyx, half_mm=5.5, pix_mm=.18):
    u, v = _orthogonal_basis(tangent_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, [vox[..., 0], vox[..., 1], vox[..., 2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def _plane_lumen_metrics(im, coords_mm):
    im = np.asarray(im, float)
    c = np.asarray(coords_mm, float)
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    Y, X = c[yy], c[xx]
    R = np.sqrt(X * X + Y * Y)
    center_hu = float(np.median(im[R <= .7]))
    threshold = float(np.clip(.55 * center_hu, 180, 520))
    mask = (im >= threshold) & (im <= 1200) & (R <= 4.0)
    lab, _ = ndi.label(mask)
    central = lab[R <= .50]
    central = central[central > 0]
    if central.size == 0:
        return {"center_hu": center_hu, "threshold_hu": threshold, "radius_mm": np.nan,
                "centroid_offset_mm": np.inf, "circularity": 0., "core_minus_ring_hu": np.nan,
                "component_area_mm2": 0.}
    vals, counts = np.unique(central, return_counts=True)
    k = int(vals[np.argmax(counts)])
    comp = lab == k
    pix = float(abs(c[1] - c[0]))
    area = float(comp.sum() * pix * pix)
    radius = math.sqrt(area / math.pi)
    weights = comp.astype(float)
    cy = float((Y * weights).sum() / max(weights.sum(), 1))
    cx = float((X * weights).sum() / max(weights.sum(), 1))
    offset = float(math.hypot(cx, cy))
    edge = comp & ~ndi.binary_erosion(comp)
    perimeter = float(edge.sum() * pix)
    circularity = float(np.clip(4 * math.pi * area / max(perimeter * perimeter, 1e-6), 0, 1.2))
    core = float(np.mean(im[R <= .9]))
    ring_mask = (R >= 2.7) & (R <= 4.0)
    ring = float(np.mean(im[ring_mask])) if np.any(ring_mask) else core
    return {"center_hu": center_hu, "threshold_hu": threshold, "radius_mm": radius,
            "centroid_offset_mm": offset, "circularity": circularity,
            "core_minus_ring_hu": core - ring, "component_area_mm2": area}


def _score_plane(m, rca):
    r = float(m["radius_mm"]) if np.isfinite(m["radius_mm"]) else np.nan
    rr = float(rca["median_radius_mm"])
    rh = float(rca["median_center_hu"])
    radius_target = 1.15 * rr
    radius_score = float(np.exp(-.5 * ((r / max(radius_target, 1e-6) - 1) / .42) ** 2)) if np.isfinite(r) else 0.
    offset_score = float(np.exp(-.5 * (m["centroid_offset_mm"] / .65) ** 2)) if np.isfinite(m["centroid_offset_mm"]) else 0.
    circ_score = float(np.clip(m["circularity"] / .60, 0, 1))
    contrast_score = float(1 / (1 + np.exp(-(m["core_minus_ring_hu"] - 25) / 80))) if np.isfinite(m["core_minus_ring_hu"]) else 0.
    hu_score = float(np.exp(-.5 * ((m["center_hu"] - rh) / 300) ** 2))
    score = .34 * radius_score + .26 * offset_score + .17 * circ_score + .14 * contrast_score + .09 * hu_score
    passed = bool(np.isfinite(r) and .55 * rr <= r <= min(3.4, 2.0 * rr + .2)
                  and m["centroid_offset_mm"] <= 1.10 and m["circularity"] >= .28
                  and 150 <= m["center_hu"] <= 1100 and np.isfinite(m["core_minus_ring_hu"])
                  and m["core_minus_ring_hu"] >= -20)
    return score, passed


def _serial_qc(path, ct, spacing, rca, n=10, label="candidate"):
    p, s = _resample(path, spacing, .35)
    if len(p) < 5 or s[-1] < 2:
        return pd.DataFrame(), {"label": label, "length_mm": float(s[-1]) if len(s) else 0.,
                                "median_plane_score": 0., "plane_pass_fraction": 0.}
    ss = np.linspace(min(.7, .08 * s[-1]), max(min(.7, .08 * s[-1]), s[-1] - .7), n)
    rows = []
    for x in ss:
        i = int(np.argmin(np.abs(s - x)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing, float)
        if np.linalg.norm(t) < 1e-6:
            continue
        im, c = _orthogonal_plane(ct, p[i], t, spacing)
        m = _plane_lumen_metrics(im, c)
        score, passed = _score_plane(m, rca)
        rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_score": score, "plane_pass": passed})
    df = pd.DataFrame(rows)
    return df, {"label": label, "length_mm": float(s[-1]),
                "median_plane_score": float(df["plane_score"].median()) if len(df) else 0.,
                "plane_pass_fraction": float(df["plane_pass"].mean()) if len(df) else 0.,
                "median_radius_mm": float(df["radius_mm"].median()) if len(df) else np.nan,
                "median_offset_mm": float(df["centroid_offset_mm"].median()) if len(df) else np.nan,
                "median_circularity": float(df["circularity"].median()) if len(df) else np.nan,
                "median_center_hu": float(df["center_hu"].median()) if len(df) else np.nan}


def _calibrate_rca(rca_path, ct, spacing):
    p, s = _resample(rca_path, spacing, .45)
    sample_s = np.linspace(min(.8, s[-1] * .04), max(min(.8, s[-1] * .04), s[-1] - .8), 12)
    rows = []
    for x in sample_s:
        i = int(np.argmin(np.abs(s - x)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing, float)
        im, c = _orthogonal_plane(ct, p[i], t, spacing)
        rows.append({"arc_mm": float(s[i]), **_plane_lumen_metrics(im, c)})
    raw = pd.DataFrame(rows)
    good = raw[np.isfinite(raw["radius_mm"])]
    if len(good) < 6:
        raise RuntimeError("RCA calibration produced too few finite lumen sections")
    rca = {"median_radius_mm": float(good["radius_mm"].median()),
           "median_center_hu": float(good["center_hu"].median()),
           "median_offset_mm": float(good["centroid_offset_mm"].median()),
           "median_circularity": float(good["circularity"].median())}
    scores, passes = zip(*[_score_plane(row, rca) for _, row in raw.iterrows()])
    raw["plane_score"] = scores; raw["plane_pass"] = passes
    rca["median_plane_score"] = float(raw["plane_score"].median())
    rca["plane_pass_fraction"] = float(raw["plane_pass"].mean())
    return raw, rca


def _frangi_3d(vol, spacing, scales=(.55, .80, 1.10, 1.45)):
    x = np.clip(np.asarray(vol, np.float32), 80., 1000.); x = (x - 80.) / 920.
    spacing = np.asarray(spacing, float); best = np.zeros_like(x, np.float32)
    for sm in scales:
        sig = np.maximum(sm / spacing, .55); n = sm * sm
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * n / spacing[0]**2
        hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * n / spacing[1]**2
        hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * n / spacing[2]**2
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * n / (spacing[0]*spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * n / (spacing[0]*spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * n / (spacing[1]*spacing[2])
        H = np.empty(x.shape + (3,3), np.float32)
        H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals = np.linalg.eigvalsh(H); vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
        l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); ss=np.sqrt(l1*l1+l2*l2+l3*l3)
        nz=ss[ss>0]; cc=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        v=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(ss*ss)/(2*cc*cc)))
        v[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(v).astype(np.float32))
    return best


def _root_interface_clusters(cur, leg, aorta, spacing, rca_prox_zyx):
    surface = aorta & ~ndi.binary_erosion(aorta, iterations=1, border_value=0)
    surf_dist = ndi.distance_transform_edt(~surface, sampling=spacing)
    zz = np.arange(aorta.shape[0], dtype=float)
    zwin = np.abs((zz - float(rca_prox_zyx[0])) * float(spacing[0])) <= 12.0
    interface = (cur | leg) & (surf_dist <= 2.0) & zwin[:,None,None]
    # bridge tiny mask gaps before component labeling, but retain only original interface points for metrics
    dil = ndi.binary_dilation(interface, structure=np.ones((3,3,3), bool), iterations=3)
    lab, n = ndi.label(dil, structure=np.ones((3,3,3), int))
    rows=[]
    for k in range(1,n+1):
        pts=np.argwhere(interface & (lab==k))
        if len(pts)<3: continue
        ctr=np.mean(pts,axis=0); dual=float(np.mean((cur&leg)[tuple(pts.T)]))
        rows.append({"cluster_id":k,"n_interface_voxels":int(len(pts)),"centroid_zyx":ctr,
                     "dual_fraction":dual,"points":pts})
    rows.sort(key=lambda r:(r["n_interface_voxels"],r["dual_fraction"]),reverse=True)
    return rows, surf_dist, surface


def _local_direction(seed, union, outside_dist, spacing, radius_mm=6.0):
    seed=np.asarray(seed,float); sp=np.asarray(spacing,float); rad=np.ceil(radius_mm/sp).astype(int)
    lo=np.maximum(np.floor(seed).astype(int)-rad,0); hi=np.minimum(np.floor(seed).astype(int)+rad+1,np.asarray(union.shape))
    sl=tuple(slice(lo[i],hi[i]) for i in range(3)); pts=np.argwhere(union[sl])+lo[None,:]
    if len(pts)>=12:
        dmm=(pts-seed[None,:])*sp[None,:]; keep=np.linalg.norm(dmm,axis=1)<=radius_mm; dmm=dmm[keep]
    else: dmm=np.empty((0,3))
    if len(dmm)>=8:
        C=np.cov(dmm.T); vals,vecs=np.linalg.eigh(C); t=_unit(vecs[:,np.argmax(vals)])
    else:
        t=np.array([0.,1.,0.])
    step=.8; pplus=seed+step*t/sp; pminus=seed-step*t/sp
    dplus=float(_sample_arr(outside_dist,[pplus])[0]); dminus=float(_sample_arr(outside_dist,[pminus])[0])
    if dminus>dplus: t=-t
    return t


def _cone_dirs(t):
    t=_unit(t); u,v=_orthogonal_basis(t); out=[t]
    for deg in (10,20,30,40,50):
        a=math.radians(deg)
        for phi in np.linspace(0,2*math.pi,10,endpoint=False):
            out.append(_unit(math.cos(a)*t+math.sin(a)*(math.cos(phi)*u+math.sin(phi)*v)))
    return out


def _build_local_field(src, cur, leg, seed, spacing, half_mm=15.0):
    rad=np.ceil(half_mm/np.asarray(spacing,float)).astype(int); c=np.floor(seed).astype(int)
    lo=np.maximum(c-rad,0); hi=np.minimum(c+rad+1,np.asarray(src.shape)); sl=tuple(slice(lo[i],hi[i]) for i in range(3))
    roi=np.asarray(src[sl]); cm=np.asarray(cur[sl]); lm=np.asarray(leg[sl]); union=cm|lm
    du=ndi.distance_transform_edt(~union,sampling=spacing); vessel=_frangi_3d(roi,spacing)
    vals=vessel[union]
    thr=max(.003,min(.12,.25*float(np.percentile(vals,20)))) if vals.size else .01
    norm=max(float(np.median(vals))*1.5,.02) if vals.size else .02
    return {"lo":lo,"roi":roi,"cur":cm,"leg":lm,"du":du,"v":vessel,"thr":thr,"norm":norm}


def _beam_outward(seed, tangent, field, outside_dist, spacing, max_mm=12.0, step=.45, beam_width=90):
    sp=np.asarray(spacing,float); origin=np.asarray(seed,float); d0=float(_sample_arr(outside_dist,[origin])[0])
    states=[(0.,[origin],_unit(tangent),d0)]; finals=[]; nsteps=int(math.ceil(max_mm/step))
    for _ in range(nsteps):
        nxt=[]
        for score,pts,t,prev_od in states:
            for d in _cone_dirs(t):
                if np.dot(d,t)<math.cos(math.radians(58)): continue
                p=pts[-1]+step*d/sp
                local=p-field["lo"]; shape=np.asarray(field["roi"].shape)
                if np.any(local<1) or np.any(local>shape-2): continue
                od=float(_sample_arr(outside_dist,[p])[0])
                if od<prev_od-.10: continue
                hu=float(_sample_arr(field["roi"],[local],order=1,cval=-1024)[0])
                vv=float(_sample_arr(field["v"],[local],order=1,cval=0)[0])
                dd=float(_sample_arr(field["du"],[local],order=1,cval=9)[0])
                zi=np.rint(local).astype(int); c=bool(field["cur"][tuple(zi)]); l=bool(field["leg"][tuple(zi)])
                if not (120<=hu<=1200 and vv>=.40*field["thr"] and dd<=1.6): continue
                if len(pts)>5 and np.min(np.linalg.norm((np.asarray(pts[:-4])-p)*sp[None,:],axis=1))<.55: continue
                vn=min(1.,vv/field["norm"]); support=1. if c and l else .62 if c or l else .12
                progress=max(-.1,od-prev_od); align=max(0.,float(np.dot(d,t)))
                ns=score+1.7*vn+.55*support+.45*align+.55*progress/max(step,1e-6)
                nt=_unit(.72*t+.28*d); st=(ns,pts+[p],nt,od); nxt.append(st)
                length=_arc(np.asarray(st[1]),spacing)[-1]
                if length>=5.0: finals.append(st)
        if not nxt: break
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]*np.asarray(spacing)/.35).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=beam_width: break
        states=keep
    return finals,states


def _path_metrics(path, src, cur, leg, vessel_field, outside_dist, spacing):
    p,s=_resample(path,spacing,.20); h=_sample_arr(src,p,order=1,cval=-1024); z=np.rint(p).astype(int); z=np.clip(z,[0,0,0],np.asarray(src.shape)-1)
    cs=cur[tuple(z.T)]; ls=leg[tuple(z.T)]; od=_sample_arr(outside_dist,p,order=1,cval=0)
    local=p-vessel_field["lo"][None,:]; vv=_sample_arr(vessel_field["v"],local,order=1,cval=0)
    length=float(s[-1]); disp=float(np.linalg.norm((p[-1]-p[0])*np.asarray(spacing)))
    return {"length_mm":length,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6),
            "median_hu":float(np.median(h)),"robust_hu_fraction":float(np.mean((h>=120)&(h<=1200))),
            "median_vesselness":float(np.median(vv)),"p10_vesselness":float(np.percentile(vv,10)),
            "current_support_fraction":float(np.mean(cs)),"legacy_support_fraction":float(np.mean(ls)),
            "union_support_fraction":float(np.mean(cs|ls)),"dual_support_fraction":float(np.mean(cs&ls)),
            "start_outside_aorta_mm":float(od[0]),"endpoint_outside_aorta_mm":float(od[-1]),
            "outside_aorta_gain_mm":float(od[-1]-od[0]),"arc_profile_mm":s,"outside_profile_mm":od}


def _candidate_from_cluster(cluster, src, cur, leg, outside_dist, spacing, rca_cal, label):
    pts=np.asarray(cluster["points"],float); # choose the interface point already farthest outside the aorta
    od=_sample_arr(outside_dist,pts,order=1,cval=0); seed=pts[int(np.argmax(od))]
    tangent=_local_direction(seed,cur|leg,outside_dist,spacing)
    field=_build_local_field(src,cur,leg,seed,spacing)
    finals,_=_beam_outward(seed,tangent,field,outside_dist,spacing)
    rows=[]
    for st in finals:
        p=np.asarray(st[1]); m=_path_metrics(p,src,cur,leg,field,outside_dist,spacing)
        qdf,qsum=_serial_qc(p,src,spacing,rca_cal,n=9,label=label)
        gate=bool(m["length_mm"]>=5.0 and m["tortuosity"]<=1.8 and m["robust_hu_fraction"]>=.90
                  and m["union_support_fraction"]>=.70 and m["dual_support_fraction"]>=.20
                  and m["outside_aorta_gain_mm"]>=3.0 and qsum["plane_pass_fraction"]>=.60
                  and .55*rca_cal["median_radius_mm"]<=qsum.get("median_radius_mm",np.nan)<=min(3.4,2.0*rca_cal["median_radius_mm"]+.2))
        score=.42*qsum["median_plane_score"]+.30*qsum["plane_pass_fraction"]+.12*min(1,m["outside_aorta_gain_mm"]/5)+.10*m["dual_support_fraction"]+.06*min(1,m["length_mm"]/8)
        rows.append((gate,score,m,qsum,p,qdf,float(st[0]),seed,tangent))
    if not rows:
        return None,{"accepted":False,"reason":"no_path_ge_5mm","cluster_id":cluster["cluster_id"],"seed_zyx":seed.tolist(),"initial_tangent_mm":tangent.tolist()},pd.DataFrame()
    rows.sort(key=lambda r:(r[0],r[1]),reverse=True); gate,score,m,qsum,p,qdf,bscore,seed,tangent=rows[0]
    summary={"accepted":bool(gate),"cluster_id":cluster["cluster_id"],"n_interface_voxels":cluster["n_interface_voxels"],"cluster_dual_fraction":cluster["dual_fraction"],
             "seed_zyx":seed.tolist(),"initial_tangent_mm":tangent.tolist(),"selection_score":float(score),"beam_score":bscore,
             "path_metrics":{k:v for k,v in m.items() if not isinstance(v,np.ndarray)},"serial_qc":qsum}
    return p,summary,qdf


def synthetic_interface_cluster_self_test():
    shape=(40,40,40); a=np.zeros(shape,bool); zz,yy,xx=np.indices(shape); a=((yy-20)**2+(xx-20)**2<=8**2)&(zz>=10)&(zz<=30)
    c=np.zeros(shape,bool); c[20,20,28:34]=True; c[20,20,6:13]=True
    clusters,_,_=_root_interface_clusters(c,c,a,np.ones(3),np.array([20.,20.,28.]))
    assert len(clusters)>=2
    cents=[r["centroid_zyx"] for r in clusters]
    assert max(np.linalg.norm(cents[i]-cents[j]) for i in range(len(cents)) for j in range(i))>10
    return {"ok":True,"clusters":len(clusters)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/RCA_PATH,root/LAD_PATH,root/C6_PATH,root/CUR,root/LEG,root/AORTA]
    for p in required:_req(p)
    master=json.loads((root/MASTER).read_text()); ref,src,spacing=_source(root/SOURCE_CACHE)
    rca=_load_path(root/RCA_PATH,ref); lad=_load_path(root/LAD_PATH,ref); c6=_load_path(root/C6_PATH,ref)
    cur=_resample_mask(root/CUR,ref); leg=_resample_mask(root/LEG,ref); aorta=_resample_mask(root/AORTA,ref)
    surface=aorta&~ndi.binary_erosion(aorta,iterations=1,border_value=0); surf_tree=cKDTree(np.argwhere(surface)*spacing[None,:])
    # orient accepted RCA from its aortic end outward
    d0=float(surf_tree.query(rca[0]*spacing)[0]); d1=float(surf_tree.query(rca[-1]*spacing)[0]); rca_o=rca if d0<=d1 else rca[::-1].copy(); rca_prox=rca_o[0]
    rca_raw,rca_cal=_calibrate_rca(rca_o,src,spacing); rca_raw.to_csv(out/"RCA_serial_calibration.csv",index=False); _write_json(out/"RCA_lumen_calibration.json",rca_cal)
    clusters,surf_dist,surface=_root_interface_clusters(cur,leg,aorta,spacing,rca_prox)
    if not clusters:
        status=STATUS_RCA_INTERFACE_FAIL; summary={"status":status,"reason":"no_root_interface_clusters","algorithm":ALGORITHM,"baseline_commit":BASELINE}
        _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); return {"summary":summary,"report":None,"zip":None}
    for r in clusters:
        r["distance_to_rca_prox_mm"]=float(np.linalg.norm((r["centroid_zyx"]-rca_prox)*spacing))
    rca_cluster=min(clusters,key=lambda r:r["distance_to_rca_prox_mm"])
    outside_dist=ndi.distance_transform_edt(~aorta,sampling=spacing)
    rca_trace,rca_sum,rca_qdf=_candidate_from_cluster(rca_cluster,src,cur,leg,outside_dist,spacing,rca_cal,"RCA_interface_control")
    if rca_trace is not None:
        known,_=_resample(rca_o,spacing,.20); known=known[_arc(known,spacing)<=8.0+1e-6]; kt=cKDTree(known*spacing[None,:]); dd=kt.query(rca_trace*spacing[None,:])[0]
        rca_sum["median_distance_to_known_rca_mm"]=float(np.median(dd)); rca_sum["p90_distance_to_known_rca_mm"]=float(np.percentile(dd,90))
    else:
        rca_sum["median_distance_to_known_rca_mm"]=float("inf"); rca_sum["p90_distance_to_known_rca_mm"]=float("inf")
    rca_control_pass=bool(rca_sum.get("accepted",False) and rca_sum["distance_to_rca_prox_mm"] if False else True)
    rca_control_pass=bool(rca_trace is not None and rca_sum.get("accepted",False) and rca_cluster["distance_to_rca_prox_mm"]<=3.0 and rca_sum["median_distance_to_known_rca_mm"]<=1.0 and rca_sum["p90_distance_to_known_rca_mm"]<=1.8)
    rca_sum["control_pass"]=rca_control_pass; rca_sum["interface_centroid_distance_to_known_rca_mm"]=rca_cluster["distance_to_rca_prox_mm"]
    _write_json(out/"RCA_interface_control.json",rca_sum); rca_qdf.to_csv(out/"RCA_interface_serial_qc.csv",index=False)
    # discover every other cluster without using LAD/C6 geometry
    candidate_rows=[]; candidate_objects=[]
    if rca_control_pass:
        for cl in clusters:
            sep=float(np.linalg.norm((cl["centroid_zyx"]-rca_cluster["centroid_zyx"])*spacing))
            if cl["cluster_id"]==rca_cluster["cluster_id"] or sep<8.0: continue
            path,csum,qdf=_candidate_from_cluster(cl,src,cur,leg,outside_dist,spacing,rca_cal,f"cluster_{cl['cluster_id']}")
            csum["separation_from_rca_interface_mm"]=sep
            # post-hoc only: never used in acceptance/selection
            if path is not None:
                ltree=cKDTree(lad*spacing[None,:]); ctree=cKDTree(c6*spacing[None,:]); pm=path*spacing[None,:]
                csum["posthoc_min_distance_to_LAD_mm"]=float(np.min(ltree.query(pm)[0])); csum["posthoc_min_distance_to_C6_mm"]=float(np.min(ctree.query(pm)[0]))
            candidate_rows.append({"cluster_id":cl["cluster_id"],"n_interface_voxels":cl["n_interface_voxels"],"cluster_dual_fraction":cl["dual_fraction"],"separation_from_rca_interface_mm":sep,"accepted":csum.get("accepted",False),"selection_score":csum.get("selection_score",np.nan),"length_mm":csum.get("path_metrics",{}).get("length_mm",np.nan),"outside_gain_mm":csum.get("path_metrics",{}).get("outside_aorta_gain_mm",np.nan),"union_support":csum.get("path_metrics",{}).get("union_support_fraction",np.nan),"dual_support":csum.get("path_metrics",{}).get("dual_support_fraction",np.nan),"plane_pass_fraction":csum.get("serial_qc",{}).get("plane_pass_fraction",np.nan),"median_radius_mm":csum.get("serial_qc",{}).get("median_radius_mm",np.nan),"posthoc_min_distance_to_LAD_mm":csum.get("posthoc_min_distance_to_LAD_mm",np.nan),"posthoc_min_distance_to_C6_mm":csum.get("posthoc_min_distance_to_C6_mm",np.nan)})
            candidate_objects.append((path,csum,qdf,cl))
    cdf=pd.DataFrame(candidate_rows); cdf.to_csv(out/"ostial_interface_candidates.csv",index=False)
    accepted=[x for x in candidate_objects if x[0] is not None and x[1].get("accepted",False)]
    best=None
    if accepted:
        accepted.sort(key=lambda x:x[1].get("selection_score",0),reverse=True); best=accepted[0]
        path,bsum,bqdf,bcl=best; pd.DataFrame(path,columns=["source_z","source_y","source_x"]).to_csv(out/"left_coronary_ostial_candidate_path.csv",index=False)
        xyz=_zyx_to_xyz(ref,path); pd.DataFrame(xyz,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"left_coronary_ostial_candidate_lps.csv",index=False)
        bqdf.to_csv(out/"left_coronary_ostial_serial_qc.csv",index=False); _write_json(out/"left_coronary_ostial_candidate_summary.json",bsum)
    status=STATUS_RCA_INTERFACE_FAIL if not rca_control_pass else STATUS_POS if best is not None else STATUS_NO_SECOND
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status","CORONARY_ANATOMY_BASELINE_V2_FROZEN"),"master_modified":False,"RCA_interface_control":rca_sum,"RCA_lumen_calibration":rca_cal,"n_root_interface_clusters":len(clusters),"left_candidate":best[1] if best is not None else {"accepted":False}}
    _write_json(out/"summary.json",summary)
    # Figures
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection="3d")
    for cl in clusters[:10]:
        p=cl["points"][::max(1,len(cl["points"])//150)]*spacing[None,:]; ax.scatter(p[:,2],p[:,1],p[:,0],s=8,label=f"C{cl['cluster_id']}")
    rp=rca_o*spacing[None,:]; ax.plot(rp[:,2],rp[:,1],rp[:,0],linewidth=2,label="known RCA")
    if best is not None:
        bp=best[0]*spacing[None,:]; ax.plot(bp[:,2],bp[:,1],bp[:,0],linewidth=3,label="accepted second ostial exit")
    ax.set_title("Aortic-root coronary-mask interface clusters"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/"01_root_interface_clusters.png",dpi=180); plt.close()
    if rca_trace is not None:
        rm=_path_metrics(rca_trace,src,cur,leg,_build_local_field(src,cur,leg,rca_trace[0],spacing),outside_dist,spacing)
        plt.figure(figsize=(6.5,4)); plt.plot(rm["arc_profile_mm"],rm["outside_profile_mm"],marker="."); plt.xlabel("arc (mm)"); plt.ylabel("distance outside aorta (mm)"); plt.title(f"RCA interface control | pass={rca_control_pass}"); plt.tight_layout(); plt.savefig(out/"02_RCA_outside_aorta_profile.png",dpi=180); plt.close()
    if best is not None:
        bm=_path_metrics(best[0],src,cur,leg,_build_local_field(src,cur,leg,best[0][0],spacing),outside_dist,spacing)
        plt.figure(figsize=(6.5,4)); plt.plot(bm["arc_profile_mm"],bm["outside_profile_mm"],marker="."); plt.xlabel("arc (mm)"); plt.ylabel("distance outside aorta (mm)"); plt.title("Accepted second coronary ostial exit"); plt.tight_layout(); plt.savefig(out/"03_left_candidate_outside_aorta_profile.png",dpi=180); plt.close()
        # orthogonal QC montage
        p,s=_resample(best[0],spacing,.25); ss=np.linspace(.7,max(.7,s[-1]-.7),6); fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,x in zip(axes.ravel(),ss):
            i=int(np.argmin(np.abs(s-x))); i0,i1=max(0,i-3),min(len(p)-1,i+3); t=(p[i1]-p[i0])*spacing; im,c=_orthogonal_plane(src,p[i],t,spacing,half_mm=5.0,pix_mm=.20); ax.imshow(im,cmap="gray",vmin=0,vmax=900,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,linewidth=.5); ax.axvline(0,linewidth=.5); ax.set_title(f"arc {s[i]:.1f} mm"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Second ostial exit — source-CCTA orthogonal QC"); plt.tight_layout(); plt.savefig(out/"04_left_candidate_orthogonal_qc.png",dpi=180); plt.close()
    report=out/"OPENPLAQUE_LEFT_CORONARY_OSTIUM_MASK_CONSENSUS_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque Left Coronary Ostium Mask Consensus</h1><p><b>Status:</b> {status}</p><p>RCA interface control pass: {rca_control_pass}.</p><p>Accepted second coronary ostial exit: {best is not None}.</p><p>LAD/C6 are post-hoc QC only. Clinical LM remains unresolved; master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_CORONARY_OSTIUM_MASK_CONSENSUS_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
