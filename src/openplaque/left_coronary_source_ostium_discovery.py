from __future__ import annotations

"""Blind source-CCTA discovery of coronary ostial exits at the aortic root.

The discovery stage does not use LAD/C6 coordinates or coronary masks. It uses the aortic
surface plus source-CCTA HU/vesselness and RCA-calibrated coronary lumen scale. The known RCA
is used after blind candidate generation as a positive control. Current/legacy coronary masks,
LAD, and C6 are post-hoc QC only and never affect candidate acceptance or selection.

Research use only. A positive result nominates a second coronary-sized ostial exit for visual QC;
it does not establish clinical left-main identity or modify Master Coronary Anatomy Baseline v2.1.
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
ALGORITHM = "left-coronary-source-ostium-discovery-v1.0-lowmem"
OUTPUT_DIRNAME = "Left_Coronary_Source_Ostium_Discovery_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "heartchambers_highres/aorta.nii.gz"

STATUS_CONTROL_FAIL = "SOURCE_OSTIUM_RCA_DISCOVERY_CONTROL_FAILED"
STATUS_NO_SECOND = "SOURCE_SECOND_CORONARY_OSTIAL_EXIT_NOT_ESTABLISHED"
STATUS_POS = "SOURCE_LEFT_CORONARY_OSTIAL_EXIT_REQUIRES_VISUAL_QC"


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
    D = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
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


def _arc(path, spacing):
    p = np.asarray(path, float)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.diff(p, axis=0) * np.asarray(spacing, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def _resample(path, spacing, step_mm=.25):
    p = np.asarray(path, float)
    s = _arc(p, spacing)
    if len(p) < 2 or not len(s) or s[-1] <= 0:
        return p.copy(), s
    keep = np.r_[True, np.diff(s) > 1e-7]
    p, s = p[keep], s[keep]
    q = np.arange(0.0, s[-1] + 1e-9, step_mm)
    if len(q) == 0 or q[-1] < s[-1] - 1e-6:
        q = np.r_[q, s[-1]]
    return np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)]), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orthogonal_basis(t):
    t = _unit(t)
    ref = np.array([1., 0., 0.]) if abs(t[0]) < .82 else np.array([0., 1., 0.])
    u = _unit(np.cross(t, ref)); v = _unit(np.cross(t, u))
    return u, v


def _orthogonal_plane(ct, point, tangent_mm, spacing, half_mm=5.2, pix_mm=.18):
    u, v = _orthogonal_basis(tangent_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing, np.float32)
    pm = np.asarray(point, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, np.float32)
    map_coordinates(ct, [vox[...,0], vox[...,1], vox[...,2]], output=out,
                    order=1, mode="nearest", prefilter=False)
    return out, c


def _plane_metrics(im, c):
    im = np.asarray(im, float); c = np.asarray(c, float)
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    Y, X = c[yy], c[xx]; R = np.sqrt(X*X + Y*Y)
    center_hu = float(np.median(im[R <= .7]))
    threshold = float(np.clip(.55 * center_hu, 180, 520))
    mask = (im >= threshold) & (im <= 1200) & (R <= 4.0)
    lab, _ = ndi.label(mask)
    central = lab[R <= .50]; central = central[central > 0]
    if central.size == 0:
        return {"center_hu":center_hu,"radius_mm":np.nan,"centroid_offset_mm":np.inf,
                "circularity":0.,"core_minus_ring_hu":np.nan}
    vals, counts = np.unique(central, return_counts=True); k = int(vals[np.argmax(counts)])
    comp = lab == k; pix = float(abs(c[1]-c[0])); area = float(comp.sum()*pix*pix)
    radius = math.sqrt(area/math.pi); w = comp.astype(float)
    cy = float((Y*w).sum()/max(w.sum(),1)); cx = float((X*w).sum()/max(w.sum(),1))
    offset = float(math.hypot(cx,cy)); edge = comp & ~ndi.binary_erosion(comp)
    perimeter = float(edge.sum()*pix); circ = float(np.clip(4*math.pi*area/max(perimeter*perimeter,1e-6),0,1.2))
    core = float(np.mean(im[R <= .9])); ringm = (R >= 2.7) & (R <= 4.0)
    ring = float(np.mean(im[ringm])) if np.any(ringm) else core
    return {"center_hu":center_hu,"radius_mm":radius,"centroid_offset_mm":offset,
            "circularity":circ,"core_minus_ring_hu":core-ring}


def _score_plane(m, rca):
    r = float(m["radius_mm"]) if np.isfinite(m["radius_mm"]) else np.nan
    rr = float(rca["median_radius_mm"]); rh = float(rca["median_center_hu"])
    radius_target = 1.15 * rr
    rs = float(np.exp(-.5*((r/max(radius_target,1e-6)-1)/.42)**2)) if np.isfinite(r) else 0.
    os = float(np.exp(-.5*(m["centroid_offset_mm"]/.65)**2)) if np.isfinite(m["centroid_offset_mm"]) else 0.
    cs = float(np.clip(m["circularity"]/.60,0,1))
    ks = float(1/(1+np.exp(-(m["core_minus_ring_hu"]-25)/80))) if np.isfinite(m["core_minus_ring_hu"]) else 0.
    hs = float(np.exp(-.5*((m["center_hu"]-rh)/300)**2))
    score = .34*rs + .26*os + .17*cs + .14*ks + .09*hs
    passed = bool(np.isfinite(r) and .55*rr <= r <= min(3.4,2.0*rr+.2)
                  and m["centroid_offset_mm"] <= 1.10 and m["circularity"] >= .28
                  and 150 <= m["center_hu"] <= 1100 and np.isfinite(m["core_minus_ring_hu"])
                  and m["core_minus_ring_hu"] >= -20)
    return score, passed


def _serial_qc(path, ct, spacing, rca, n=9, label="candidate"):
    p, s = _resample(path, spacing, .35)
    if len(p) < 5 or s[-1] < 2:
        return pd.DataFrame(), {"label":label,"length_mm":float(s[-1]) if len(s) else 0.,
                                "median_plane_score":0.,"plane_pass_fraction":0.}
    ss = np.linspace(min(.7,.08*s[-1]), max(min(.7,.08*s[-1]),s[-1]-.7), n)
    rows=[]
    for x in ss:
        i=int(np.argmin(np.abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3)
        t=(p[i1]-p[i0])*np.asarray(spacing,float)
        if np.linalg.norm(t)<1e-6: continue
        im,c=_orthogonal_plane(ct,p[i],t,spacing); m=_plane_metrics(im,c); score,passed=_score_plane(m,rca)
        rows.append({"label":label,"arc_mm":float(s[i]),**m,"plane_score":score,"plane_pass":passed})
    df=pd.DataFrame(rows)
    return df,{"label":label,"length_mm":float(s[-1]),
               "median_plane_score":float(df["plane_score"].median()) if len(df) else 0.,
               "plane_pass_fraction":float(df["plane_pass"].mean()) if len(df) else 0.,
               "median_radius_mm":float(df["radius_mm"].median()) if len(df) else np.nan,
               "median_offset_mm":float(df["centroid_offset_mm"].median()) if len(df) else np.nan,
               "median_center_hu":float(df["center_hu"].median()) if len(df) else np.nan}


def _calibrate_rca(rca, ct, spacing):
    p,s=_resample(rca,spacing,.45)
    ss=np.linspace(min(.8,s[-1]*.04),max(min(.8,s[-1]*.04),s[-1]-.8),12)
    rows=[]
    for x in ss:
        i=int(np.argmin(np.abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3)
        t=(p[i1]-p[i0])*np.asarray(spacing,float); im,c=_orthogonal_plane(ct,p[i],t,spacing)
        rows.append({"arc_mm":float(s[i]),**_plane_metrics(im,c)})
    raw=pd.DataFrame(rows); good=raw[np.isfinite(raw["radius_mm"])]
    if len(good)<6: raise RuntimeError("RCA calibration produced too few finite lumen sections")
    rca_cal={"median_radius_mm":float(good["radius_mm"].median()),
             "median_center_hu":float(good["center_hu"].median()),
             "median_offset_mm":float(good["centroid_offset_mm"].median()),
             "median_circularity":float(good["circularity"].median())}
    scores,passes=zip(*[_score_plane(row,rca_cal) for _,row in raw.iterrows()])
    raw["plane_score"]=scores; raw["plane_pass"]=passes
    rca_cal["median_plane_score"]=float(raw["plane_score"].median())
    rca_cal["plane_pass_fraction"]=float(raw["plane_pass"].mean())
    return raw,rca_cal


def _frangi_3d(vol, spacing, scales=(.60,.90,1.25)):
    x=np.clip(np.asarray(vol,np.float32),80.,1000.); x=(x-80.)/920.; spacing=np.asarray(spacing,float)
    best=np.zeros_like(x,np.float32)
    for sm in scales:
        sig=np.maximum(sm/spacing,.55); n=sm*sm
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0),mode="nearest")*n/spacing[0]**2
        hyy=ndi.gaussian_filter(x,sig,order=(0,2,0),mode="nearest")*n/spacing[1]**2
        hxx=ndi.gaussian_filter(x,sig,order=(0,0,2),mode="nearest")*n/spacing[2]**2
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0),mode="nearest")*n/(spacing[0]*spacing[1])
        hzx=ndi.gaussian_filter(x,sig,order=(1,0,1),mode="nearest")*n/(spacing[0]*spacing[2])
        hyx=ndi.gaussian_filter(x,sig,order=(0,1,1),mode="nearest")*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),np.float32)
        H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1)
        l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); ss=np.sqrt(l1*l1+l2*l2+l3*l3)
        nz=ss[ss>0]; cc=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        v=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(ss*ss)/(2*cc*cc)))
        v[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(v).astype(np.float32))
        del H,vals,hzz,hyy,hxx,hzy,hzx,hyx
    return best


def _root_crop(src_shape, center, spacing, half_z_mm=18., half_xy_mm=45.):
    sp=np.asarray(spacing,float); c=np.asarray(center,float)
    rad=np.ceil(np.array([half_z_mm,half_xy_mm,half_xy_mm])/sp).astype(int)
    lo=np.maximum(np.floor(c).astype(int)-rad,0); hi=np.minimum(np.floor(c).astype(int)+rad+1,np.asarray(src_shape))
    return lo,hi,tuple(slice(lo[i],hi[i]) for i in range(3))


def _component_direction(points, seed, spacing, outside_dist):
    pts=np.asarray(points,float); seed=np.asarray(seed,float); sp=np.asarray(spacing,float)
    dmm=(pts-seed[None,:])*sp[None,:]
    if len(dmm)>=3 and float(np.linalg.norm(np.ptp(dmm,axis=0)))>=1.0:
        q=dmm-np.mean(dmm,axis=0,keepdims=True); vals,vecs=np.linalg.eigh(q.T@q/max(len(q)-1,1)); t=_unit(vecs[:,-1])
    else:
        t=np.array([0.,1.,0.])
    step=.8; dplus=float(_sample_arr(outside_dist,[seed+step*t/sp])[0]); dminus=float(_sample_arr(outside_dist,[seed-step*t/sp])[0])
    if dminus>dplus: t=-t
    return _unit(t)


def _blind_root_components(src_roi,aorta_roi,vessel,spacing,v_thr,rca_local_z):
    outside=ndi.distance_transform_edt(~aorta_roi,sampling=spacing).astype(np.float32)
    zmm=np.abs((np.arange(src_roi.shape[0])-float(rca_local_z))*float(spacing[0]))
    cand=(outside>=.45)&(outside<=14.0)&(zmm[:,None,None]<=12.0)&(src_roi>=180)&(src_roi<=1000)&(vessel>=v_thr)
    # One closing heals sub-voxel ridge gaps without using coronary masks.
    cand=ndi.binary_closing(cand,structure=np.ones((3,3,3),bool),iterations=1)
    lab,n=ndi.label(cand,structure=np.ones((3,3,3),int)); rows=[]
    for k in range(1,n+1):
        pts=np.argwhere(lab==k)
        if len(pts)<8: continue
        od=outside[tuple(pts.T)]; mn=float(np.min(od)); mx=float(np.max(od))
        if mn>2.0 or mx<4.0: continue
        span=float(np.linalg.norm(np.ptp(pts*np.asarray(spacing)[None,:],axis=0)))
        if span<4.0: continue
        hu=src_roi[tuple(pts.T)].astype(float); vv=vessel[tuple(pts.T)].astype(float)
        rows.append({"component_id":int(k),"n_voxels":int(len(pts)),"points":pts,
                     "min_outside_mm":mn,"max_outside_mm":mx,"physical_span_mm":span,
                     "median_hu":float(np.median(hu)),"median_vesselness":float(np.median(vv))})
    rows.sort(key=lambda r:(r["max_outside_mm"],r["median_vesselness"],r["n_voxels"]),reverse=True)
    return rows,outside


def _cone_dirs(t):
    t=_unit(t); u,v=_orthogonal_basis(t); out=[t]
    for deg in (10,20,30,40,50):
        a=math.radians(deg)
        for phi in np.linspace(0,2*math.pi,10,endpoint=False):
            out.append(_unit(math.cos(a)*t+math.sin(a)*(math.cos(phi)*u+math.sin(phi)*v)))
    return out


def _beam_source(seed,tangent,src_roi,vessel,outside,spacing,v_thr,max_mm=12.,step=.45,beam_width=90):
    sp=np.asarray(spacing,float); origin=np.asarray(seed,float); d0=float(_sample_arr(outside,[origin])[0])
    norm=max(float(np.median(vessel[vessel>=v_thr]))*1.5,.02) if np.any(vessel>=v_thr) else .02
    states=[(0.,[origin],_unit(tangent),d0)]; finals=[]; shape=np.asarray(src_roi.shape)
    for _ in range(int(math.ceil(max_mm/step))):
        nxt=[]
        for score,pts,t,prev_od in states:
            for d in _cone_dirs(t):
                if np.dot(d,t)<math.cos(math.radians(58)): continue
                p=pts[-1]+step*d/sp
                if np.any(p<1) or np.any(p>shape-2): continue
                od=float(_sample_arr(outside,[p])[0])
                if od<prev_od-.10: continue
                hu=float(_sample_arr(src_roi,[p],order=1,cval=-1024)[0]); vv=float(_sample_arr(vessel,[p],order=1,cval=0)[0])
                if not (120<=hu<=1200 and vv>=.40*v_thr): continue
                if len(pts)>5 and np.min(np.linalg.norm((np.asarray(pts[:-4])-p)*sp[None,:],axis=1))<.55: continue
                vn=min(1.,vv/norm); progress=max(-.1,od-prev_od); align=max(0.,float(np.dot(d,t)))
                hs=float(np.exp(-.5*((hu-560.)/360.)**2)); ns=score+1.75*vn+.42*align+.55*progress/max(step,1e-6)+.18*hs
                nt=_unit(.72*t+.28*d); st=(ns,pts+[p],nt,od); nxt.append(st)
                if _arc(np.asarray(st[1]),spacing)[-1]>=5.0: finals.append(st)
        if not nxt: break
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]*sp/.35).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=beam_width: break
        states=keep
    return finals


def _path_metrics(path,src_roi,vessel,outside,spacing,v_thr):
    p,s=_resample(path,spacing,.20); hu=_sample_arr(src_roi,p,order=1,cval=-1024); vv=_sample_arr(vessel,p,order=1,cval=0); od=_sample_arr(outside,p,order=1,cval=0)
    length=float(s[-1]); disp=float(np.linalg.norm((p[-1]-p[0])*np.asarray(spacing)))
    return {"length_mm":length,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6),
            "median_hu":float(np.median(hu)),"robust_hu_fraction":float(np.mean((hu>=120)&(hu<=1200))),
            "median_vesselness":float(np.median(vv)),"p10_vesselness":float(np.percentile(vv,10)),
            "vesselness_threshold":float(v_thr),"start_outside_aorta_mm":float(od[0]),
            "endpoint_outside_aorta_mm":float(od[-1]),"outside_aorta_gain_mm":float(od[-1]-od[0]),
            "arc_profile_mm":s,"outside_profile_mm":od}


def _trace_component(comp,src_roi,vessel,outside,spacing,v_thr,rca_cal,label):
    pts=np.asarray(comp["points"],float); od=outside[tuple(pts.T)]; seed=pts[int(np.argmin(od))]
    near=pts[np.linalg.norm((pts-seed[None,:])*np.asarray(spacing)[None,:],axis=1)<=6.0]
    tangent=_component_direction(near,seed,spacing,outside); finals=_beam_source(seed,tangent,src_roi,vessel,outside,spacing,v_thr)
    rows=[]
    for st in finals:
        p=np.asarray(st[1]); m=_path_metrics(p,src_roi,vessel,outside,spacing,v_thr); qdf,qsum=_serial_qc(p,src_roi,spacing,rca_cal,n=9,label=label)
        gate=bool(m["length_mm"]>=5.0 and m["tortuosity"]<=1.8 and m["robust_hu_fraction"]>=.90
                  and m["p10_vesselness"]>=.50*v_thr and m["outside_aorta_gain_mm"]>=3.0
                  and qsum["plane_pass_fraction"]>=.60 and qsum["median_plane_score"]>=.60
                  and .55*rca_cal["median_radius_mm"]<=qsum.get("median_radius_mm",np.nan)<=min(3.4,2.0*rca_cal["median_radius_mm"]+.2))
        score=.42*qsum["median_plane_score"]+.30*qsum["plane_pass_fraction"]+.12*min(1,m["outside_aorta_gain_mm"]/5)+.10*min(1,m["median_vesselness"]/max(v_thr,1e-6))+.06*min(1,m["length_mm"]/8)
        rows.append((gate,score,m,qsum,p,qdf,float(st[0]),seed,tangent))
    if not rows:
        return None,{"accepted":False,"reason":"no_path_ge_5mm","component_id":comp["component_id"],"seed_zyx_local":seed.tolist()},pd.DataFrame()
    rows.sort(key=lambda r:(r[0],r[1]),reverse=True); gate,score,m,qsum,p,qdf,bscore,seed,tangent=rows[0]
    sm={"accepted":bool(gate),"component_id":comp["component_id"],"component_n_voxels":comp["n_voxels"],
        "component_span_mm":comp["physical_span_mm"],"seed_zyx_local":seed.tolist(),"initial_tangent_mm":tangent.tolist(),
        "selection_score":float(score),"beam_score":bscore,"path_metrics":{k:v for k,v in m.items() if not isinstance(v,np.ndarray)},"serial_qc":qsum}
    return p,sm,qdf


def synthetic_root_component_self_test():
    shape=(41,41,41); zz,yy,xx=np.indices(shape); a=((yy-20)**2+(xx-20)**2<=8**2)&(zz>=8)&(zz<=32)
    src=np.zeros(shape,np.float32); vessel=np.zeros(shape,np.float32)
    # two narrow bright exits on opposite sides of the synthetic aorta
    src[20,20,28:37]=560; vessel[20,20,28:37]=.8
    src[20,20,4:13]=560; vessel[20,20,4:13]=.8
    comps,out=_blind_root_components(src,a,vessel,np.ones(3),.2,20.)
    assert len(comps)>=2
    return {"ok":True,"components":len(comps)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    for p in [root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/RCA_PATH,root/LAD_PATH,root/C6_PATH,root/AORTA,root/CUR,root/LEG]: _req(p)
    master=json.loads((root/MASTER).read_text()); ref,src,spacing=_source(root/SOURCE_CACHE)
    rca=_load_path(root/RCA_PATH,ref); lad=_load_path(root/LAD_PATH,ref); c6=_load_path(root/C6_PATH,ref)
    aorta=_resample_mask(root/AORTA,ref); cur=_resample_mask(root/CUR,ref); leg=_resample_mask(root/LEG,ref)
    surface=aorta&~ndi.binary_erosion(aorta,iterations=1,border_value=0); surfpts=np.argwhere(surface); stree=cKDTree(surfpts*spacing[None,:])
    d0=float(stree.query(rca[0]*spacing)[0]); d1=float(stree.query(rca[-1]*spacing)[0]); rca_o=rca if d0<=d1 else rca[::-1].copy(); rca_prox=rca_o[0]
    rca_raw,rca_cal=_calibrate_rca(rca_o,src,spacing); rca_raw.to_csv(out/"RCA_lumen_calibration_sections.csv",index=False); _write_json(out/"RCA_lumen_calibration.json",rca_cal)
    lo,hi,sl=_root_crop(src.shape,rca_prox,spacing); src_roi=np.asarray(src[sl]); aorta_roi=np.asarray(aorta[sl]); cur_roi=np.asarray(cur[sl]); leg_roi=np.asarray(leg[sl]); rca_local=rca_o-lo[None,:]
    vessel=_frangi_3d(src_roi,spacing); rp,rs=_resample(rca_local,spacing,.25); rp=rp[rs<=8.0+1e-9]; rvals=_sample_arr(vessel,rp,order=1,cval=0)
    v_thr=max(.003,min(.12,.35*float(np.percentile(rvals,25)))) if len(rvals) else .01
    comps,outside=_blind_root_components(src_roi,aorta_roi,vessel,spacing,v_thr,rca_prox[0]-lo[0])
    comp_rows=[]; objects=[]
    for comp in comps[:40]:
        p,sm,qdf=_trace_component(comp,src_roi,vessel,outside,spacing,v_thr,rca_cal,f"component_{comp['component_id']}")
        if p is not None:
            pg=p+lo[None,:]; known,_=_resample(rca_o,spacing,.20); known=known[_arc(known,spacing)<=10.0+1e-9]; kt=cKDTree(known*spacing[None,:]); dd=kt.query(pg*spacing[None,:])[0]
            sm["postsearch_median_distance_to_known_RCA_mm"]=float(np.median(dd)); sm["postsearch_p90_distance_to_known_RCA_mm"]=float(np.percentile(dd,90))
            sm["postsearch_seed_distance_to_known_RCA_prox_mm"]=float(np.linalg.norm((pg[0]-rca_prox)*spacing))
        else:
            sm["postsearch_median_distance_to_known_RCA_mm"]=float("inf"); sm["postsearch_p90_distance_to_known_RCA_mm"]=float("inf"); sm["postsearch_seed_distance_to_known_RCA_prox_mm"]=float("inf")
        comp_rows.append({"component_id":comp["component_id"],"n_voxels":comp["n_voxels"],"span_mm":comp["physical_span_mm"],"min_outside_mm":comp["min_outside_mm"],"max_outside_mm":comp["max_outside_mm"],"accepted":sm.get("accepted",False),"selection_score":sm.get("selection_score",np.nan),"length_mm":sm.get("path_metrics",{}).get("length_mm",np.nan),"plane_pass_fraction":sm.get("serial_qc",{}).get("plane_pass_fraction",np.nan),"median_radius_mm":sm.get("serial_qc",{}).get("median_radius_mm",np.nan),"median_distance_to_RCA_mm":sm.get("postsearch_median_distance_to_known_RCA_mm",np.nan)})
        objects.append((p,sm,qdf,comp))
    pd.DataFrame(comp_rows).to_csv(out/"blind_source_root_candidates.csv",index=False)
    traced=[x for x in objects if x[0] is not None]
    rca_obj=min(traced,key=lambda x:x[1].get("postsearch_median_distance_to_known_RCA_mm",np.inf)) if traced else None
    rca_control=False
    if rca_obj is not None:
        rsu=rca_obj[1]; rca_control=bool(rsu.get("accepted",False) and rsu["postsearch_median_distance_to_known_RCA_mm"]<=1.0 and rsu["postsearch_p90_distance_to_known_RCA_mm"]<=1.8 and rsu["postsearch_seed_distance_to_known_RCA_prox_mm"]<=3.0)
        rsu["control_pass"]=rca_control; rca_obj[2].to_csv(out/"RCA_blind_discovery_serial_qc.csv",index=False); _write_json(out/"RCA_blind_discovery_control.json",rsu)
    else:
        _write_json(out/"RCA_blind_discovery_control.json",{"control_pass":False,"reason":"no_traced_candidates"})
    best=None
    if rca_control:
        eligible=[]
        rseed=(rca_obj[0][0]+lo)*spacing
        for obj in objects:
            p,sm,qdf,comp=obj
            if p is None or not sm.get("accepted",False) or obj is rca_obj: continue
            seed_sep=float(np.linalg.norm((p[0]+lo)*spacing-rseed))
            if sm.get("postsearch_median_distance_to_known_RCA_mm",np.inf)<=4.0 or seed_sep<8.0: continue
            eligible.append(obj)
        if eligible:
            eligible.sort(key=lambda x:x[1].get("selection_score",0),reverse=True); best=eligible[0]
            p,bsum,bqdf,bcomp=best; pg=p+lo[None,:]
            # Post-hoc only: masks and established left-system geometry do not affect selection.
            zi=np.rint(pg).astype(int); zi=np.clip(zi,[0,0,0],np.asarray(src.shape)-1); cs=cur[tuple(zi.T)]; ls=leg[tuple(zi.T)]
            ltree=cKDTree(lad*spacing[None,:]); ctree=cKDTree(c6*spacing[None,:]); pm=pg*spacing[None,:]
            bsum["posthoc_current_mask_support_fraction"]=float(np.mean(cs)); bsum["posthoc_legacy_mask_support_fraction"]=float(np.mean(ls)); bsum["posthoc_union_mask_support_fraction"]=float(np.mean(cs|ls))
            bsum["posthoc_min_distance_to_LAD_mm"]=float(np.min(ltree.query(pm)[0])); bsum["posthoc_min_distance_to_C6_mm"]=float(np.min(ctree.query(pm)[0]))
            pd.DataFrame(pg,columns=["source_z","source_y","source_x"]).to_csv(out/"second_coronary_ostial_exit_path.csv",index=False)
            xyz=_zyx_to_xyz(ref,pg); pd.DataFrame(xyz,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"second_coronary_ostial_exit_lps.csv",index=False)
            bqdf.to_csv(out/"second_coronary_ostial_exit_serial_qc.csv",index=False); _write_json(out/"second_coronary_ostial_exit_summary.json",bsum)
    status=STATUS_CONTROL_FAIL if not rca_control else STATUS_POS if best is not None else STATUS_NO_SECOND
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status","CORONARY_ANATOMY_BASELINE_V2_FROZEN"),"master_modified":False,"RCA_lumen_calibration":rca_cal,"source_vesselness_threshold":v_thr,"n_blind_root_components":len(comps),"n_traced_components":len(traced),"RCA_blind_control":rca_obj[1] if rca_obj is not None else {"control_pass":False},"second_coronary_candidate":best[1] if best is not None else {"accepted":False}}
    _write_json(out/"summary.json",summary)
    # QC figures.
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d")
    for comp in comps[:12]:
        pts=(comp["points"][::max(1,len(comp["points"])//120)]+lo[None,:])*spacing[None,:]
        ax.scatter(pts[:,2],pts[:,1],pts[:,0],s=5,alpha=.45)
    if rca_obj is not None:
        q=(rca_obj[0]+lo[None,:])*spacing[None,:]; ax.plot(q[:,2],q[:,1],q[:,0],linewidth=3,label="blind RCA control")
    if best is not None:
        q=(best[0]+lo[None,:])*spacing[None,:]; ax.plot(q[:,2],q[:,1],q[:,0],linewidth=3,label="second ostial exit")
    ax.set_title("Blind source-CCTA aortic-root tubular components"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/"01_blind_source_root_components.png",dpi=180); plt.close()
    if rca_obj is not None:
        m=_path_metrics(rca_obj[0],src_roi,vessel,outside,spacing,v_thr); plt.figure(figsize=(6.5,4)); plt.plot(m["arc_profile_mm"],m["outside_profile_mm"],marker="."); plt.xlabel("arc (mm)"); plt.ylabel("distance outside aorta (mm)"); plt.title(f"Blind RCA discovery | control pass={rca_control}"); plt.tight_layout(); plt.savefig(out/"02_RCA_blind_outside_profile.png",dpi=180); plt.close()
    if best is not None:
        m=_path_metrics(best[0],src_roi,vessel,outside,spacing,v_thr); plt.figure(figsize=(6.5,4)); plt.plot(m["arc_profile_mm"],m["outside_profile_mm"],marker="."); plt.xlabel("arc (mm)"); plt.ylabel("distance outside aorta (mm)"); plt.title("Second coronary ostial exit"); plt.tight_layout(); plt.savefig(out/"03_second_exit_outside_profile.png",dpi=180); plt.close()
        p,s=_resample(best[0],spacing,.25); ss=np.linspace(.7,max(.7,s[-1]-.7),6); fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,x in zip(axes.ravel(),ss):
            i=int(np.argmin(np.abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3); t=(p[i1]-p[i0])*spacing; im,c=_orthogonal_plane(src_roi,p[i],t,spacing,half_mm=5.,pix_mm=.20); ax.imshow(im,cmap="gray",vmin=0,vmax=900,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,linewidth=.5); ax.axvline(0,linewidth=.5); ax.set_title(f"arc {s[i]:.1f} mm"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Second coronary ostial exit — source-CCTA orthogonal QC"); plt.tight_layout(); plt.savefig(out/"04_second_exit_orthogonal_qc.png",dpi=180); plt.close()
    report=out/"OPENPLAQUE_LEFT_CORONARY_SOURCE_OSTIUM_DISCOVERY_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque Blind Source-CCTA Coronary Ostium Discovery</h1><p><b>Status:</b> {status}</p><p>Blind RCA discovery control pass: {rca_control}.</p><p>Accepted second coronary ostial exit: {best is not None}.</p><p>Coronary masks, LAD, and C6 are post-hoc QC only. Clinical LM remains unresolved; master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_CORONARY_SOURCE_OSTIUM_DISCOVERY_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
