from __future__ import annotations

import gc, heapq, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-distal-source-extension-v1.0"
OUTPUT_DIRNAME = "LCX_Distal_Source_Extension_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
HIER_SUMMARY = Path("LCX_Hierarchical_Distal_Tree_v1/summary.json")
PRIOR = Path("Joint_Three_Vessel_Template_Classifier_v1")
C7_FILE = PRIOR / "candidate_05_source_path.csv"   # source candidate 7
C9_FILE = PRIOR / "candidate_02_source_path.csv"   # source candidate 9
C6_FILE = PRIOR / "candidate_04_source_path.csv"   # source candidate 6

TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CURRENT = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEGACY = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
LA = TS / "heartchambers_highres/heart_atrium_left.nii.gz"
LV = TS / "heartchambers_highres/heart_ventricle_left.nii.gz"
MYO = TS / "heartchambers_highres/heart_myocardium.nii.gz"

STATUS_CONTROL_FAIL = "DISTAL_SOURCE_EXTENSION_POSITIVE_CONTROL_FAILED"
STATUS_NO_LONG = "NO_LONG_DISTAL_SOURCE_EXTENSION"
STATUS_AMBIG = "LONG_DISTAL_SOURCE_EXTENSIONS_AV_GROOVE_AMBIGUOUS"
STATUS_C7 = "C7_LONGITUDINAL_LCX_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC"
STATUS_C96 = "C9_C6_LONGITUDINAL_LCX_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC"

SCALES_MM = (0.55, 0.80, 1.10, 1.45)

def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")

def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)

def _arc_lps(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

def _resample_lps(p, step=0.20):
    p = np.asarray(p, float)
    s = _arc_lps(p)
    if len(p) < 2 or s[-1] <= 0:
        return p.copy(), s
    q = np.arange(0.0, s[-1] + 1e-9, step)
    if q[-1] < s[-1] - 1e-6:
        q = np.r_[q, s[-1]]
    out = np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)])
    return out, q

def _interp_lps(p, q):
    p = np.asarray(p, float)
    s = _arc_lps(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)])

def _load_lps(path):
    d = pd.read_csv(_req(path))
    cols = ["lps_x_mm", "lps_y_mm", "lps_z_mm"]
    if not all(c in d.columns for c in cols):
        raise ValueError(f"Expected LPS columns in {path}: {list(d.columns)}")
    return d[cols].to_numpy(float)

def _orient_same(paths):
    ref = np.asarray(paths[0], float)[0]
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref):
            p = p[::-1].copy()
        out.append(p)
    return out

def _common_prefix(paths, step=0.25, tol_mm=0.60, sustain=4, start_check_mm=10.0):
    ps = _orient_same(paths)
    minlen = min(_arc_lps(p)[-1] for p in ps)
    q = np.arange(0.0, minlen + 1e-9, step)
    sam = np.stack([_interp_lps(p, q) for p in ps], axis=0)
    med = np.median(sam, axis=0)
    dev = np.max(np.linalg.norm(sam - med[None, :, :], axis=2), axis=0)
    start = int(round(start_check_mm / step))
    bad = None
    for i in range(start, max(start, len(q)-sustain+1)):
        if np.all(dev[i:i+sustain] > tol_mm):
            bad = i
            break
    end = bad-1 if bad is not None else len(q)-1
    return med[:end+1], q[:end+1], dev[:end+1]

@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray
    spacing_xyz: np.ndarray
    direction: np.ndarray
    shape_zyx: np.ndarray

    def lps_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx_xyz = ((pts - self.origin) @ np.linalg.inv(self.direction).T) / self.spacing_xyz
        return idx_xyz[:, ::-1]

    def zyx_to_lps(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx_xyz = pts[:, ::-1]
        return self.origin + (idx_xyz * self.spacing_xyz) @ self.direction.T

    @property
    def spacing_zyx(self):
        return self.spacing_xyz[::-1]

def _load_source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    sp_zyx = np.asarray(meta["spacing_zyx"], float)
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([[row[0], col[0], slc[0]],
                          [row[1], col[1], slc[1]],
                          [row[2], col[2], slc[2]]], float)
    g = Geometry(np.asarray(meta["positions_lps_mm"][0], float),
                 sp_zyx[::-1], direction, np.asarray(arr.shape, int))
    return g, arr

def _sample_source(arr, g, pts_lps, order=1, cval=-1024.0):
    zyx = g.lps_to_zyx(pts_lps)
    return ndi.map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)

def _img_surface_tree(path, surface=True, max_points=250000):
    im = sitk.ReadImage(str(_req(path)))
    work = sitk.LabelContour(sitk.Cast(im > 0, sitk.sitkUInt8), False) if surface else sitk.Cast(im > 0, sitk.sitkUInt8)
    a = sitk.GetArrayViewFromImage(work)
    z = np.argwhere(a > 0)
    if len(z) == 0:
        raise ValueError(f"No foreground in {path}")
    if len(z) > max_points:
        z = z[::int(math.ceil(len(z)/max_points))]
    idx_xyz = z[:, ::-1]
    origin = np.asarray(work.GetOrigin(), float)
    spacing = np.asarray(work.GetSpacing(), float)
    direction = np.asarray(work.GetDirection(), float).reshape(3, 3)
    xyz = origin + (idx_xyz * spacing) @ direction.T
    del z, a, work, im
    gc.collect()
    return cKDTree(xyz.astype(np.float32, copy=False))

def _frangi_3d(volume_hu, spacing_zyx, scales_mm=SCALES_MM, alpha=0.5, beta=0.5):
    x = np.clip(np.asarray(volume_hu, np.float32), 80.0, 1000.0)
    x = (x - 80.0) / 920.0
    spacing = np.asarray(spacing_zyx, float)
    best = np.zeros_like(x, dtype=np.float32)
    best_scale = np.zeros_like(x, dtype=np.float32)
    for sigma_mm in scales_mm:
        sig = np.maximum(float(sigma_mm) / spacing, 0.55)
        norm = float(sigma_mm) ** 2
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * norm/(spacing[0]**2)
        hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * norm/(spacing[1]**2)
        hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * norm/(spacing[2]**2)
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * norm/(spacing[0]*spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * norm/(spacing[0]*spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * norm/(spacing[1]*spacing[2])
        H = np.empty(x.shape + (3,3), dtype=np.float32)
        H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy
        H[...,0,2]=H[...,2,0]=hzx
        H[...,1,2]=H[...,2,1]=hyx
        vals = np.linalg.eigvalsh(H)
        order = np.argsort(np.abs(vals), axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        l1,l2,l3 = vals[...,0], vals[...,1], vals[...,2]
        eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps)
        rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps)
        ss=np.sqrt(l1*l1+l2*l2+l3*l3)
        nz=ss[ss>0]
        c=max(float(np.percentile(nz,90))*0.45 if nz.size else 0.05,1e-4)
        v=(1-np.exp(-(ra*ra)/(2*alpha*alpha)))
        v*=np.exp(-(rb*rb)/(2*beta*beta))
        v*=(1-np.exp(-(ss*ss)/(2*c*c)))
        v[(l2>=0)|(l3>=0)]=0
        v=np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float32)
        take=v>best
        best[take]=v[take]
        best_scale[take]=float(sigma_mm)
        del H, vals, hzz,hyy,hxx,hzy,hzx,hyx
        gc.collect()
    return best,best_scale

def _terminal_tangent(path, span_mm=2.5):
    p,s=_resample_lps(path,0.20)
    j=np.searchsorted(s,max(0.0,s[-1]-span_mm))
    return _unit(p[-1]-p[j])

def _make_roi(g, seed, forward_mm=22.0, margin_mm=10.0):
    endpoint=np.asarray(seed[-1],float)
    tang=_terminal_tangent(seed)
    base=np.vstack([_interp_lps(seed, np.linspace(max(0,_arc_lps(seed)[-1]-5), _arc_lps(seed)[-1], 16)),
                    endpoint[None,:]+np.linspace(0,forward_mm,8)[:,None]*tang[None,:]])
    pts=[base]
    for p in base[::max(1,len(base)//8)]:
        for sx in (-margin_mm,margin_mm):
            for sy in (-margin_mm,margin_mm):
                for sz in (-margin_mm,margin_mm):
                    pts.append((p+np.array([sx,sy,sz],float))[None,:])
    allpts=np.vstack(pts)
    zyx=g.lps_to_zyx(allpts)
    lo=np.floor(np.min(zyx,axis=0)).astype(int)
    hi=np.ceil(np.max(zyx,axis=0)).astype(int)+1
    lo=np.maximum(lo,0); hi=np.minimum(hi,g.shape_zyx)
    if np.any(hi-lo<8):
        raise RuntimeError(f"Degenerate ROI {lo} {hi}")
    return lo,hi,tang

def _crop_source(src, lo, hi):
    return np.asarray(src[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]], dtype=np.float32)

def _sample_crop(vol, g, lo, pts_lps, order=1, cval=0.0):
    zyx=g.lps_to_zyx(pts_lps)-lo[None,:]
    return ndi.map_coordinates(np.asarray(vol), zyx.T, order=order, mode="constant", cval=cval)

def _neighbors(spacing_zyx):
    out=[]
    sp=np.asarray(spacing_zyx,float)
    for dz in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dz==dy==dx==0: continue
                step=float(np.linalg.norm(np.array([dz,dy,dx])*sp))
                out.append((dz,dy,dx,step))
    return out

def _dijkstra(mask,cost,start,spacing_zyx,max_cost=210.0,goal_mask=None):
    shape=mask.shape; n=int(np.prod(shape))
    dist=np.full(n,np.inf,np.float32); prev=np.full(n,-1,np.int32)
    sf=np.ravel_multi_index(tuple(start),shape); dist[sf]=0.0
    heap=[(0.0,sf)]; neigh=_neighbors(spacing_zyx); reached=-1
    while heap:
        dcur,flat=heapq.heappop(heap)
        if dcur>float(dist[flat])+1e-6: continue
        if dcur>max_cost: break
        z,y,x=np.unravel_index(flat,shape)
        if goal_mask is not None and goal_mask[z,y,x]:
            reached=flat; break
        for dz,dy,dx,step in neigh:
            zz,yy,xx=z+dz,y+dy,x+dx
            if zz<0 or yy<0 or xx<0 or zz>=shape[0] or yy>=shape[1] or xx>=shape[2]: continue
            if not mask[zz,yy,xx]: continue
            nf=np.ravel_multi_index((zz,yy,xx),shape)
            nd=dcur+step*0.5*(float(cost[z,y,x])+float(cost[zz,yy,xx]))
            if nd<float(dist[nf]):
                dist[nf]=nd; prev[nf]=flat; heapq.heappush(heap,(nd,nf))
    return dist,prev,reached

def _reconstruct(prev, flat, shape):
    if flat<0:return None
    seq=[]; seen=set()
    while flat>=0 and flat not in seen:
        seen.add(flat); seq.append(np.asarray(np.unravel_index(flat,shape),float)); flat=int(prev[flat])
    seq.reverse()
    return np.asarray(seq,float)

def _sphere_mask(g,lo,shape,center_lps,radius_mm):
    c=g.lps_to_zyx(center_lps)[0]-lo
    zz,yy,xx=np.indices(shape)
    sp=g.spacing_zyx
    d2=((zz-c[0])*sp[0])**2+((yy-c[1])*sp[1])**2+((xx-c[2])*sp[2])**2
    return d2<=radius_mm**2

def _local_to_lps(g,lo,p):
    return g.zyx_to_lps(np.asarray(p,float)+lo[None,:])

def _path_turn_p90(path):
    p,_=_resample_lps(path,0.30)
    if len(p)<4:return 180.0
    v=np.diff(p,axis=0); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
    ang=np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))
    return float(np.percentile(ang,90)) if len(ang) else 180.0

def _initial_angle(seed_tangent,path,lookahead_mm=2.5):
    p,s=_resample_lps(path,0.20)
    j=min(len(p)-1,max(1,int(round(lookahead_mm/0.20))))
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(seed_tangent),_unit(p[j]-p[0])),-1,1))))

def _surface_metrics(path, la_tree, lv_tree):
    p,s=_resample_lps(path,0.25)
    da,ia=la_tree.query(p); dv,iv=lv_tree.query(p)
    na=p-np.asarray(la_tree.data[ia],float); nv=p-np.asarray(lv_tree.data[iv],float)
    nla=np.linalg.norm(na,axis=1); nlv=np.linalg.norm(nv,axis=1)
    good=(nla>0.15)&(nlv>0.15)
    na[good]/=nla[good,None]; nv[good]/=nlv[good,None]
    t=np.gradient(p,axis=0); t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
    groove=np.cross(na,nv); ng=np.linalg.norm(groove,axis=1); gg=good&(ng>1e-6)
    groove[gg]/=ng[gg,None]
    if gg.sum()>=3:
        align=np.abs(np.sum(t[gg]*groove[gg],axis=1))
        med_align=float(np.median(align))
        med_angle=float(np.median(np.degrees(np.arccos(np.clip(align,0,1)))))
    else:
        med_align=0.0; med_angle=90.0
    half=s>=0.5*s[-1]
    if half.sum()<3: half=np.ones(len(p),bool)
    def slope(x,y):
        if len(x)<3 or np.ptp(x)<1e-6:return 0.0
        return float(np.polyfit(x,y,1)[0])
    la_slope=slope(s,da)
    retention=float(np.exp(-max(0.0,la_slope)/0.35))
    bal=float(np.median(np.abs(da[half]-dv[half])))
    maxd=float(np.median(np.maximum(da[half],dv[half])))
    balance_score=float(np.exp(-bal/10.0))
    proximity_score=float(np.exp(-maxd/20.0))
    score=float(0.50*med_align+0.20*retention+0.15*balance_score+0.15*proximity_score)
    return {
        "median_groove_alignment_cos":med_align,
        "median_groove_tangent_angle_deg":med_angle,
        "left_atrium_distance_slope_mm_per_mm":la_slope,
        "atrium_retention_score":retention,
        "distal_median_left_atrium_surface_mm":float(np.median(da[half])),
        "distal_median_left_ventricle_surface_mm":float(np.median(dv[half])),
        "distal_balance_mm":bal,
        "distal_max_chamber_distance_mm":maxd,
        "longitudinal_av_score":score,
        "profile_arc_mm":s,
        "profile_la_mm":da,
        "profile_lv_mm":dv,
    }

def _source_support(path,current_tree,legacy_tree):
    p,_=_resample_lps(path,0.25)
    dc=current_tree.query(p)[0]; dl=legacy_tree.query(p)[0]
    return float(np.mean(dc<=1.0)),float(np.mean(dl<=1.0))

def _seed_control(seed, g, lo, vessel, threshold, mask, cost, max_cost=90.0):
    p,s=_resample_lps(seed,0.20)
    end=p[-1]; start_arc=max(0.0,s[-1]-2.5)
    start=p[int(np.argmin(np.abs(s-start_arc)))]
    start_local=np.rint(g.lps_to_zyx(start)[0]-lo).astype(int)
    start_local=np.clip(start_local,0,np.asarray(mask.shape)-1)
    gm=_sphere_mask(g,lo,mask.shape,end,0.80)
    mm=mask.copy()
    mm[_sphere_mask(g,lo,mask.shape,start,0.60)]=True
    mm[gm]=True
    _,prev,reached=_dijkstra(mm,cost,start_local,g.spacing_zyx,max_cost,gm)
    if reached<0:
        return None,{"control_pass":False,"reason":"endpoint_unreachable"}
    loc=_reconstruct(prev,reached,mask.shape)
    path=_local_to_lps(g,lo,loc)
    path,_=_resample_lps(path,0.20)
    known=_interp_lps(seed,np.linspace(start_arc,s[-1],max(3,int(round((s[-1]-start_arc)/0.15))+1)))
    dd=cKDTree(known).query(path)[0]
    vv=_sample_crop(vessel,g,lo,path,1,0)
    target=float(np.linalg.norm(path[-1]-end))
    length=float(_arc_lps(path)[-1])
    passed=bool(target<=0.9 and np.median(dd)<=0.75 and np.percentile(dd,90)<=1.2 and length<=5.0 and np.median(vv)>=0.8*threshold)
    return path,{"control_pass":passed,"target_distance_mm":target,"control_path_length_mm":length,
                 "median_distance_to_seed_mm":float(np.median(dd)),"p90_distance_to_seed_mm":float(np.percentile(dd,90)),
                 "median_vesselness":float(np.median(vv))}

def _build_mask_cost(crop,vessel,best_scale,threshold,control_median):
    vn=np.clip(vessel/max(1.5*control_median,1e-4),0,1)
    mask=(crop>=170)&(crop<=1200)&(vessel>=threshold)
    cost=0.30+4.5*(1-vn)**2
    cost+=0.30*np.clip((best_scale-1.15)/0.50,0,1)
    cost+=0.20*np.clip((crop-900)/300,0,1)
    return mask,cost.astype(np.float32)

def _candidate_metrics(path, seed, seed_tangent, vessel, g, lo, current_tree, legacy_tree, la_tree, lv_tree):
    p,s=_resample_lps(path,0.20)
    vv=_sample_crop(vessel,g,lo,p,1,0)
    cur,leg=_source_support(p,current_tree,legacy_tree)
    ext=float(s[-1]); disp=float(np.linalg.norm(p[-1]-p[0]))
    angle=_initial_angle(seed_tangent,p)
    p10=float(np.percentile(vv,10)); med=float(np.median(vv))
    source_score=float(
        0.26*min(ext/15.0,1.0)+
        0.20*min(disp/12.0,1.0)+
        0.20*min(med/max(CURRENT_THRESHOLD*2.0,1e-4),1.0)+
        0.12*min(p10/max(CURRENT_THRESHOLD,1e-4),1.0)+
        0.12*(cur+leg)/2.0+
        0.10*math.exp(-angle/70.0)
    )
    av=_surface_metrics(p,la_tree,lv_tree)
    return {"extension_length_mm":ext,"endpoint_displacement_mm":disp,"initial_continuity_angle_deg":angle,
            "turn_angle_p90_deg":_path_turn_p90(p),"median_vesselness":med,"p10_vesselness":p10,
            "current_support_fraction":cur,"legacy_support_fraction":leg,"source_evidence_score":source_score,**av}

CURRENT_THRESHOLD=0.12

def _run_seed(label,seed,g,src,out,current_tree,legacy_tree,la_tree,lv_tree,forward_mm=22.0,margin_mm=10.0):
    global CURRENT_THRESHOLD
    lo,hi,tangent=_make_roi(g,seed,forward_mm,margin_mm)
    crop=_crop_source(src,lo,hi)
    vessel,bscale=_frangi_3d(crop,g.spacing_zyx)
    p,s=_resample_lps(seed,0.15)
    known=p[s>=max(0,s[-1]-2.5)]
    vals=_sample_crop(vessel,g,lo,known,1,0)
    vals=vals[np.isfinite(vals)]
    p25=float(np.percentile(vals,25)) if len(vals) else 0.02
    medc=float(np.median(vals)) if len(vals) else 0.05
    threshold=float(max(0.003,min(0.12,0.35*p25)))
    CURRENT_THRESHOLD=threshold
    mask,cost=_build_mask_cost(crop,vessel,bscale,threshold,medc)
    control_path,control=_seed_control(seed,g,lo,vessel,threshold,mask,cost)
    meta={"label":label,"roi_lo_zyx":lo.tolist(),"roi_hi_zyx":hi.tolist(),"roi_shape":list(map(int,crop.shape)),
          "seed_endpoint_lps_mm":np.asarray(seed[-1]).tolist(),"seed_terminal_tangent_lps":tangent.tolist(),
          "vessel_threshold":threshold,"control_vesselness_p25":p25,"control_vesselness_median":medc,**control}
    _write_json(out/f"{label}_search_meta.json",meta)
    np.save(out/f"{label}_vesselness.npy",vessel.astype(np.float16))
    np.save(out/f"{label}_best_scale.npy",bscale.astype(np.float16))
    if not control["control_pass"]:
        return meta,pd.DataFrame(),None,None
    end=seed[-1]
    start=np.rint(g.lps_to_zyx(end)[0]-lo).astype(int); start=np.clip(start,0,np.asarray(mask.shape)-1)
    extmask=mask.copy()
    # Do not let the target-free search simply retrace the known seed and escape from
    # an older proximal branch. Block voxels within 0.9 mm of the historical seed,
    # except the final 1.2 mm used as the launch neighborhood.
    seed_r,seed_s=_resample_lps(seed,0.25)
    old_seed=seed_r[seed_s<=max(0.0,seed_s[-1]-1.2)]
    if len(old_seed):
        vox=np.column_stack(np.nonzero(extmask))
        if len(vox):
            vlps=_local_to_lps(g,lo,vox.astype(float))
            dd=cKDTree(old_seed).query(vlps)[0]
            bad=vox[dd<0.9]
            if len(bad): extmask[tuple(bad.T)]=False
    extmask[_sphere_mask(g,lo,mask.shape,end,0.65)]=True
    dist,prev,_=_dijkstra(extmask,cost,start,g.spacing_zyx,220.0,None)
    finite=np.isfinite(dist)
    coords=np.column_stack(np.nonzero(finite))
    if len(coords)==0:
        return meta,pd.DataFrame(),None,None
    lps=_local_to_lps(g,lo,coords.astype(float))
    disp=np.linalg.norm(lps-end[None,:],axis=1)
    proj=(lps-end[None,:])@tangent
    vv=vessel[tuple(coords.T)]
    prelim=0.60*disp+2.0*vv+0.20*np.maximum(proj,0)
    ok=(disp>=6.0)&(vv>=threshold)&(proj>=-1.0)
    coords=coords[ok]; prelim=prelim[ok]
    order=np.argsort(prelim)[::-1]
    rows=[]; paths=[]; chosen=[]
    for ii in order:
        glps=_local_to_lps(g,lo,coords[ii:ii+1].astype(float))[0]
        if any(np.linalg.norm(glps-q)<1.5 for q in chosen): continue
        flat=np.ravel_multi_index(tuple(coords[ii]),mask.shape)
        loc=_reconstruct(prev,flat,mask.shape)
        if loc is None or len(loc)<2: continue
        path=_local_to_lps(g,lo,loc); path,_=_resample_lps(path,0.20)
        extlen=float(_arc_lps(path)[-1])
        if not (8.0<=extlen<=28.0): continue
        m=_candidate_metrics(path,seed,tangent,vessel,g,lo,current_tree,legacy_tree,la_tree,lv_tree)
        source_hu=_sample_source(src,g,path)
        robust_hu=float(np.mean((source_hu>=120)&(source_hu<=1200)))
        m["robust_hu_fraction"]=robust_hu
        supported=bool(m["extension_length_mm"]>=10.0 and m["current_support_fraction"]>=0.80 and
                       m["legacy_support_fraction"]>=0.80 and robust_hu>=0.90 and
                       m["turn_angle_p90_deg"]<=70.0 and m["initial_continuity_angle_deg"]<=85.0 and
                       m["p10_vesselness"]>=0.60*threshold)
        row={"candidate_rank_input":len(rows)+1,**m,"supported_extension":supported}
        rows.append(row); paths.append(path); chosen.append(glps)
        if len(rows)>=30: break
    if not rows:
        return meta,pd.DataFrame(),None,None
    df=pd.DataFrame(rows).sort_values(["supported_extension","source_evidence_score","extension_length_mm"],
                                     ascending=[False,False,False]).reset_index(drop=True)
    df.insert(0,"source_rank",np.arange(1,len(df)+1))
    selected=paths[int(df.iloc[0]["candidate_rank_input"])-1]
    selected_metrics=df.iloc[0].to_dict()
    df.to_csv(out/f"{label}_extension_candidates.csv",index=False)
    pd.DataFrame(selected,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).assign(
        arc_mm=_arc_lps(selected)).to_csv(out/f"{label}_selected_extension.csv",index=False)
    del crop,vessel,bscale,mask,cost,dist,prev
    gc.collect()
    return meta,df,selected,selected_metrics

def _orth_basis(t):
    t=_unit(t); ref=np.array([1.,0,0]) if abs(t[0])<0.82 else np.array([0,1.,0])
    u=_unit(np.cross(t,ref)); v=_unit(np.cross(t,u)); return u,v

def _plane(src,g,path,i,half=6.0,pix=0.20):
    p=np.asarray(path,float); a=max(0,i-2); b=min(len(p)-1,i+2); t=_unit(p[b]-p[a]); u,v=_orth_basis(t)
    grid=np.arange(-half,half+1e-9,pix); yy,xx=np.meshgrid(grid,grid,indexing="ij")
    pts=p[i][None,None,:]+xx[...,None]*u+yy[...,None]*v
    im=_sample_source(src,g,pts.reshape(-1,3),1,-1024).reshape(len(grid),len(grid))
    return im,grid

def synthetic_extension_self_test():
    t=np.linspace(0,10,51); p=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)])
    q=p.copy(); q[t>7,1]=(t[t>7]-7)*0.3
    c,s,d=_common_prefix([p,q],0.2,0.25,3,2.0)
    assert 6.5<=s[-1]<=7.5
    assert _initial_angle(np.array([1.,0,0]),p)<1.0
    return {"ok":True,"common_prefix_mm":float(s[-1])}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",
              root/HIER_SUMMARY,root/C7_FILE,root/C9_FILE,root/C6_FILE,
              root/CURRENT,root/LEGACY,root/LA,root/LV,root/MYO]
    for p in required:_req(p)
    hier=json.loads((root/HIER_SUMMARY).read_text())
    nd=hier.get("node_decisions",[])
    if not nd or not bool(nd[0].get("decision_pass",False)) or set(map(int,str(nd[0].get("selected_members","")).split(";")))!={9,6,7}:
        raise RuntimeError("Hierarchical prerequisite does not show decisive 9/6/7 first-branch selection.")
    g,src=_load_source(root/SOURCE_CACHE)
    c7=_load_lps(root/C7_FILE); c9=_load_lps(root/C9_FILE); c6=_load_lps(root/C6_FILE)
    c9,c6=_orient_same([c9,c6])
    c96,q96,dev96=_common_prefix([c9,c6],0.25,0.60,4,15.0)
    seeds={"c7":c7,"c96":c96}
    current_tree=_img_surface_tree(root/CURRENT,False); legacy_tree=_img_surface_tree(root/LEGACY,False)
    la_tree=_img_surface_tree(root/LA,True); lv_tree=_img_surface_tree(root/LV,True); myo_tree=_img_surface_tree(root/MYO,True)
    results={}
    for label,seed in seeds.items():
        meta,df,path,metrics=_run_seed(label,seed,g,src,out,current_tree,legacy_tree,la_tree,lv_tree)
        results[label]={"meta":meta,"df":df,"path":path,"metrics":metrics}
    controls_pass=all(results[k]["meta"].get("control_pass",False) for k in results)
    supported={k:(results[k]["metrics"] is not None and bool(results[k]["metrics"].get("supported_extension",False))) for k in results}
    if not controls_pass:
        status=STATUS_CONTROL_FAIL
    elif not all(supported.values()):
        status=STATUS_NO_LONG
    else:
        s7=float(results["c7"]["metrics"]["longitudinal_av_score"])
        s96=float(results["c96"]["metrics"]["longitudinal_av_score"])
        margin=s7-s96
        if margin>=0.08: status=STATUS_C7
        elif margin<=-0.08: status=STATUS_C96
        else: status=STATUS_AMBIG
    comp=[]
    for k in ("c7","c96"):
        m=results[k]["metrics"] or {}
        comp.append({"alternative":k,"control_pass":results[k]["meta"].get("control_pass",False),
                     "supported_extension":supported[k],**{x:m.get(x) for x in [
                         "extension_length_mm","source_evidence_score","longitudinal_av_score",
                         "median_groove_alignment_cos","median_groove_tangent_angle_deg",
                         "left_atrium_distance_slope_mm_per_mm","atrium_retention_score",
                         "distal_median_left_atrium_surface_mm","distal_median_left_ventricle_surface_mm",
                         "distal_balance_mm","distal_max_chamber_distance_mm",
                         "current_support_fraction","legacy_support_fraction","robust_hu_fraction",
                         "initial_continuity_angle_deg","turn_angle_p90_deg","median_vesselness","p10_vesselness"]}})
    compdf=pd.DataFrame(comp)
    compdf.to_csv(out/"longitudinal_extension_comparison.csv",index=False)
    fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection="3d")
    for label,seed in seeds.items():
        ax.plot(seed[:,0],seed[:,1],seed[:,2],lw=2,label=f"{label} seed")
        path=results[label]["path"]
        if path is not None: ax.plot(path[:,0],path[:,1],path[:,2],lw=4,label=f"{label} extension")
    ax.legend(fontsize=8); ax.set_title(status); fig.tight_layout(); fig.savefig(out/"01_longitudinal_extension_geometry.png",dpi=170); plt.close(fig)
    plt.figure(figsize=(10,6))
    for label in ("c7","c96"):
        path=results[label]["path"]
        if path is None: continue
        av=_surface_metrics(path,la_tree,lv_tree)
        plt.plot(av["profile_arc_mm"],av["profile_la_mm"],label=f"{label} → LA")
        plt.plot(av["profile_arc_mm"],av["profile_lv_mm"],ls="--",label=f"{label} → LV")
    plt.xlabel("Extension arc (mm)"); plt.ylabel("Chamber-surface distance (mm)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/"02_longitudinal_chamber_distance_profiles.png",dpi=170); plt.close()
    plt.figure(figsize=(8,5))
    x=np.arange(len(compdf)); plt.bar(x,compdf["longitudinal_av_score"].fillna(0))
    plt.xticks(x,compdf["alternative"]); plt.ylabel("Longitudinal AV-groove score"); plt.title(status)
    plt.tight_layout(); plt.savefig(out/"03_longitudinal_av_scores.png",dpi=170); plt.close()
    valid=[k for k in ("c7","c96") if results[k]["path"] is not None]
    fig,axes=plt.subplots(max(1,len(valid)),3,figsize=(12,4*max(1,len(valid))),squeeze=False)
    for r,label in enumerate(valid):
        path=results[label]["path"]
        for c,f in enumerate((0.15,0.50,0.85)):
            i=min(len(path)-1,max(0,int(round(f*(len(path)-1)))))
            im,gr=_plane(src,g,path,i)
            axes[r,c].imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[gr[0],gr[-1],gr[-1],gr[0]])
            axes[r,c].scatter([0],[0],marker="+")
            axes[r,c].set_title(f"{label} {f:.0%}")
    fig.tight_layout(); fig.savefig(out/"04_selected_extensions_orthogonal_qc.png",dpi=170); plt.close(fig)
    score_margin=None
    if all(results[k]["metrics"] is not None for k in ("c7","c96")):
        score_margin=float(results["c7"]["metrics"]["longitudinal_av_score"]-results["c96"]["metrics"]["longitudinal_av_score"])
    summary={
        "status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,
        "hierarchical_prerequisite_status":hier.get("status"),
        "c96_common_prefix_mm":float(q96[-1]),"c96_prefix_max_deviation_mm":float(np.max(dev96)),
        "alternatives":{k:{"search_meta":results[k]["meta"],"selected_metrics":results[k]["metrics"]} for k in ("c7","c96")},
        "c7_minus_c96_longitudinal_av_score_margin":score_margin,
        "decision_rule":"Both seed-positive-controls and both >=10 mm source-supported extensions required; longitudinal AV score margin >=0.08 nominates an LCX-like continuation.",
        "source_path_selection_policy":"Within each seed, target-free graph path is selected by source evidence only; LA/LV anatomy is evaluated only after source-path selection.",
        "template_similarity_used_in_decision":False,
        "scientific_boundary":"Research anatomy only; no Master Anatomy update. LM remains unresolved."
    }
    _write_json(out/"summary.json",summary)
    report=out/"OPENPLAQUE_LCX_DISTAL_SOURCE_EXTENSION_REPORT.html"
    report.write_text(
        f"<html><body><h1>OpenPlaque LCX distal source extension</h1>"
        f"<p><b>Status:</b> {status}</p><p><b>C7-C96 AV score margin:</b> {score_margin}</p>"
        f"<p>Two independent endpoint-centered source-space vesselness searches; path selection is source-only.</p>"
        "<img src='01_longitudinal_extension_geometry.png' style='max-width:95%'><br>"
        "<img src='02_longitudinal_chamber_distance_profiles.png' style='max-width:95%'><br>"
        "<img src='03_longitudinal_av_scores.png' style='max-width:95%'><br>"
        "<img src='04_selected_extensions_orthogonal_qc.png' style='max-width:95%'></body></html>",encoding="utf-8")
    _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    zp=out/"OPENPLAQUE_LCX_DISTAL_SOURCE_EXTENSION_REPORT_BACK.zip"
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.name!=zp.name and not p.name.endswith("_vesselness.npy") and not p.name.endswith("_best_scale.npy"):
                z.write(p,arcname=p.name)
    return {"summary":summary,"report":str(report),"zip":str(zp)}
