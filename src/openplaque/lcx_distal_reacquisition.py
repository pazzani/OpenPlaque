from __future__ import annotations

"""Targeted source-CCTA distal reacquisition for the established C6/C7 daughters.

Path extension is deliberately blind to chamber identity. Each endpoint must first
rediscover its own known terminal segment using source-CCTA vesselness plus current/
legacy coronary-mask support. Only after accepted source-space extensions are frozen
is the independently validated LA-LV chamber-interface metric applied longitudinally.

Research use only. A positive result nominates LCX-like/OM-like identities for visual
QC; it does not establish clinical vessel identity and does not alter the frozen master.
"""

import heapq
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
ALGORITHM = "lcx-distal-reacquisition-v1.0-lowmem"
OUTPUT_DIRNAME = "LCX_Distal_Reacquisition_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
COR_CURRENT = TS / "coronary_arteries/coronary_arteries.nii.gz"
COR_LEGACY = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
LA = TS / "heartchambers_highres/heart_atrium_left.nii.gz"
LV = TS / "heartchambers_highres/heart_ventricle_left.nii.gz"
RA = TS / "heartchambers_highres/heart_atrium_right.nii.gz"
RV = TS / "heartchambers_highres/heart_ventricle_right.nii.gz"
PRIOR = Path("Joint_Three_Vessel_Template_Classifier_v1")
PRIOR_RANKING = PRIOR / "LCX_joint_candidate_ranking.csv"
LEAF_FILES = [PRIOR / f"candidate_{i:02d}_source_path.csv" for i in range(1, 6)]

STATUS_SOURCE_CONTROL_FAIL = "DISTAL_REACQUISITION_CONTROL_FAILED"
STATUS_SOURCE_INSUFFICIENT = "DISTAL_REACQUISITION_INSUFFICIENT"
STATUS_CHAMBER_CONTROL_FAIL = "DISTAL_REACQUISITION_CHAMBER_CONTROL_FAILED"
STATUS_AMBIG = "DISTAL_REACQUISITION_COMPLETE_CHAMBER_IDENTITY_AMBIGUOUS"
STATUS_C6 = "C6_LCX_C7_OM_LONGITUDINAL_CANDIDATES_REQUIRE_VISUAL_QC"
STATUS_C7 = "C7_LCX_C6_OM_LONGITUDINAL_CANDIDATES_REQUIRE_VISUAL_QC"


def _req(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    ref = sitk.GetImageFromArray(np.asarray(arr))
    spacing_zyx = np.asarray(meta["spacing_zyx"], float)
    ref.SetSpacing(tuple(spacing_zyx[::-1]))
    ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(direction.ravel()))
    return ref, arr, spacing_zyx


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


def _sample_source(img, arr, pts_lps):
    zyx = _xyz_to_zyx(img, pts_lps)
    return map_coordinates(np.asarray(arr), zyx.T, order=1, mode="constant", cval=-1024.0)


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


def _resample(p, step=.25):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    out = np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])
    return out, q


def _interp(p, q):
    p = np.asarray(p, float); a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orient_three(paths):
    best = None
    for bits in range(8):
        pp = [p[::-1].copy() if (bits >> i) & 1 else p.copy() for i, p in enumerate(paths)]
        s = np.array([x[0] for x in pp])
        score = sum(np.linalg.norm(s[i] - s[j]) for i in range(3) for j in range(i+1, 3))
        if best is None or score < best[0]: best = (score, pp)
    return best[1]


def _consensus_split(paths, step=.25, tol=.60, sustain=4):
    rs = []
    for p in paths:
        r, q = _resample(p, step); rs.append((r, q))
    maxq = min(q[-1] for _, q in rs)
    q = np.arange(0, maxq + 1e-9, step)
    X = np.stack([_interp(r, q) for r, _ in rs])
    dev = np.max(np.linalg.norm(X - X.mean(axis=0)[None,:,:], axis=2), axis=0)
    bad = dev > tol
    idx = None
    if len(bad) >= sustain:
        run = np.convolve(bad.astype(int), np.ones(sustain, int), mode="valid")
        w = np.where(run == sustain)[0]
        if len(w): idx = int(w[0])
    split = float(q[idx-1]) if idx is not None and idx > 0 else float(q[-1])
    return split, q, dev


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize() and np.allclose(im.GetSpacing(), ref.GetSpacing()) and np.allclose(im.GetOrigin(), ref.GetOrigin()) and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def _frangi_3d(volume_hu, spacing, scales_mm=(0.55, .80, 1.10, 1.45), alpha=.5, beta=.5):
    x = np.clip(np.asarray(volume_hu, np.float32), 80., 1000.)
    x = (x - 80.) / 920.
    spacing = np.asarray(spacing, float)
    best = np.zeros_like(x, np.float32)
    for sigma_mm in scales_mm:
        sig = np.maximum(float(sigma_mm)/spacing, .55); norm = float(sigma_mm)**2
        hzz = ndi.gaussian_filter(x, sig, order=(2,0,0), mode="nearest") * norm/(spacing[0]**2)
        hyy = ndi.gaussian_filter(x, sig, order=(0,2,0), mode="nearest") * norm/(spacing[1]**2)
        hxx = ndi.gaussian_filter(x, sig, order=(0,0,2), mode="nearest") * norm/(spacing[2]**2)
        hzy = ndi.gaussian_filter(x, sig, order=(1,1,0), mode="nearest") * norm/(spacing[0]*spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1,0,1), mode="nearest") * norm/(spacing[0]*spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0,1,1), mode="nearest") * norm/(spacing[1]*spacing[2])
        H = np.empty(x.shape+(3,3), np.float32)
        H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); order=np.argsort(np.abs(vals),axis=-1); vals=np.take_along_axis(vals,order,axis=-1)
        l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); s=np.sqrt(l1*l1+l2*l2+l3*l3)
        nz=s[s>0]; c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        v=(1-np.exp(-(ra*ra)/(2*alpha*alpha)))*np.exp(-(rb*rb)/(2*beta*beta))*(1-np.exp(-(s*s)/(2*c*c)))
        v[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float32))
    return best


def synthetic_reacquisition_self_test():
    shape=(45,45,70); z,y,x=np.indices(shape)
    center_y=22 + .035*(x-15); center_z=22 + .02*(x-15)
    tube=np.exp(-((y-center_y)**2+(z-center_z)**2)/(2*2.0**2))
    vol=60+650*tube
    v=_frangi_3d(vol,np.array([.4,.4,.4]))
    center=float(np.median(v[22,22,15:55])); bg=float(np.median(v[:5,:5,:]))
    assert center > max(.01, 4*bg)
    return {"ok":True,"center_vesselness":center,"background_vesselness":bg}


def _neighbors(spacing):
    out=[]
    for dz in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dz==dy==dx==0: continue
                step=math.sqrt((dz*spacing[0])**2+(dy*spacing[1])**2+(dx*spacing[2])**2)
                out.append((dz,dy,dx,step))
    return out


def _dijkstra(mask, cost, start, spacing, goal_mask=None, max_cost=180.):
    shape=mask.shape; n=int(np.prod(shape)); dist=np.full(n,np.inf,np.float32); prev=np.full(n,-1,np.int32)
    sf=np.ravel_multi_index(start,shape); dist[sf]=0.; heap=[(0.,sf)]; neigh=_neighbors(spacing); reached=-1
    while heap:
        dcur,flat=heapq.heappop(heap)
        if dcur>float(dist[flat])+1e-6: continue
        if dcur>max_cost: break
        z,y,x=np.unravel_index(flat,shape)
        if goal_mask is not None and goal_mask[z,y,x]: reached=flat; break
        for dz,dy,dx,step in neigh:
            zz,yy,xx=z+dz,y+dy,x+dx
            if zz<0 or yy<0 or xx<0 or zz>=shape[0] or yy>=shape[1] or xx>=shape[2] or not mask[zz,yy,xx]: continue
            nf=np.ravel_multi_index((zz,yy,xx),shape)
            nd=dcur+step*.5*(float(cost[z,y,x])+float(cost[zz,yy,xx]))
            if nd<float(dist[nf]): dist[nf]=nd; prev[nf]=flat; heapq.heappush(heap,(nd,nf))
    return dist,prev,reached


def _reconstruct(prev, flat, shape):
    if flat<0: return None
    seq=[]; seen=set()
    while flat>=0 and flat not in seen:
        seen.add(flat); seq.append(np.array(np.unravel_index(flat,shape),float)); flat=int(prev[flat])
    seq.reverse(); return np.asarray(seq,float)


def _sphere(shape, point_local, spacing, radius_mm):
    p=np.asarray(point_local,float); zz,yy,xx=np.indices(shape)
    d2=((zz-p[0])*spacing[0])**2+((yy-p[1])*spacing[1])**2+((xx-p[2])*spacing[2])**2
    return d2<=radius_mm**2


def _turns_mm(path_zyx, spacing):
    p=np.asarray(path_zyx,float)*np.asarray(spacing,float)
    if len(p)<4: return np.array([])
    v=np.diff(p,axis=0); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))


def _search_extension(ref, source, spacing, cur_mask, leg_mask, path_lps, target_mm=14.0):
    p_lps,q=_resample(path_lps,.20); p_zyx=_xyz_to_zyx(ref,p_lps)
    total=q[-1]; control_start=max(0,total-3.0); ci=int(np.argmin(np.abs(q-control_start)))
    endpoint=p_zyx[-1]; start_control=p_zyx[ci]
    t=_unit((endpoint-p_zyx[max(0,len(p_zyx)-10)])*spacing)
    phys_tail=p_zyx[max(0,ci-6):]*spacing; forward=endpoint*spacing+np.outer(np.linspace(0,target_mm,8),t)
    pts=np.vstack([phys_tail,forward]); margin=5.0
    lo=np.floor((pts.min(axis=0)-margin)/spacing).astype(int); hi=np.ceil((pts.max(axis=0)+margin)/spacing).astype(int)+1
    lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(source.shape)); sl=tuple(slice(lo[k],hi[k]) for k in range(3))
    roi=np.asarray(source[sl]); cur=np.asarray(cur_mask[sl]); leg=np.asarray(leg_mask[sl]); union=cur|leg
    dist_union=ndi.distance_transform_edt(~union,sampling=spacing); permitted=dist_union<=1.0
    v=_frangi_3d(roi,spacing)
    local_known=p_zyx-lo[None,:]
    known_v=ndi.map_coordinates(v,local_known.T,order=1,mode="nearest")
    tail_v=known_v[q>=control_start]
    thr=max(.003,min(.12,.35*float(np.percentile(tail_v,25)) if len(tail_v) else .01))
    vn=np.clip(v/max(float(np.median(tail_v))*1.5 if len(tail_v) else .05,1e-4),0,1)
    hu=np.asarray(roi,float)
    mask=(hu>=120)&(hu<=1200)&(v>=thr)&permitted
    support_pen=np.where(cur&leg,0.,np.where(union,.25,.85))
    cost=.30+4.5*(1-vn)**2+support_pen+.15*np.clip((hu-950)/250,0,1)
    start=np.rint(start_control-lo).astype(int); goal_local=endpoint-lo
    goal=_sphere(mask.shape,goal_local,spacing,.75); mask[_sphere(mask.shape,start,spacing,.60)]=True; mask[goal]=True
    dist,prev,reached=_dijkstra(mask,cost,tuple(start),spacing,goal_mask=goal,max_cost=100.)
    control={"control_pass":False}
    if reached>=0:
        loc=_reconstruct(prev,reached,mask.shape); glob=loc+lo[None,:]
        rec_lps=_zyx_to_xyz(ref,glob); rec_lps,_=_resample(rec_lps,.20)
        known=p_lps[ci:]; tree=cKDTree(known); dd=tree.query(rec_lps)[0]
        turns=_turns_mm(glob,spacing); control={"control_pass":bool(np.median(dd)<=.75 and np.percentile(dd,90)<=1.15 and (np.max(turns) if len(turns) else 0)<=70),"median_distance_to_known_mm":float(np.median(dd)),"p90_distance_to_known_mm":float(np.percentile(dd,90)),"max_turn_deg":float(np.max(turns)) if len(turns) else 0.,"vesselness_threshold":float(thr)}
    if not control["control_pass"]:
        return None,{"control":control,"accepted":False,"reason":"positive_control_failed"},None
    start2=np.rint(goal_local).astype(int); extmask=mask.copy(); extmask[_sphere(mask.shape,start2,spacing,.60)]=True
    dist,prev,_=_dijkstra(extmask,cost,tuple(start2),spacing,goal_mask=None,max_cost=185.)
    finite=np.isfinite(dist); coords=np.column_stack(np.nonzero(finite)); rows=[]; candidates=[]
    if len(coords):
        phys=(coords+lo[None,:])*spacing; ep=endpoint*spacing
        proj=(phys-ep)@t; disp=np.linalg.norm(phys-ep,axis=1); vals=v[tuple(coords.T)]
        ok=(proj>=2.0)&(disp>=2.2)&(proj<=target_mm+3)&(vals>=thr)
        coords=coords[ok]; proj=proj[ok]; vals=vals[ok]
        dual=(cur&leg)[tuple(coords.T)] if len(coords) else np.array([])
        score=proj+1.2*vals+.35*dual
        order=np.argsort(score)[::-1]; chosen=[]
        for ii in order:
            g=coords[ii]+lo
            if any(np.linalg.norm((g-c)*spacing)<1.0 for c in chosen): continue
            chosen.append(g.copy()); flat=np.ravel_multi_index(tuple(coords[ii]),mask.shape); loc=_reconstruct(prev,flat,mask.shape)
            if loc is None or len(loc)<2: continue
            glob=loc+lo[None,:]; lps=_zyx_to_xyz(ref,glob); lps,arc=_resample(lps,.20); z=_xyz_to_zyx(ref,lps)
            vl=ndi.map_coordinates(v,(z-lo[None,:]).T,order=1,mode="nearest")
            h=_sample_source(ref,source,lps); zround=np.rint(z).astype(int); zround=np.clip(zround,[0,0,0],np.asarray(source.shape)-1)
            cs=cur_mask[tuple(zround.T)]; ls=leg_mask[tuple(zround.T)]; turns=_turns_mm(z,spacing)
            length=float(arc[-1]); disp2=float(np.linalg.norm((z[-1]-endpoint)*spacing)); proj2=float(np.dot((z[-1]-endpoint)*spacing,t))
            m={"new_length_mm":length,"endpoint_displacement_mm":disp2,"forward_projection_mm":proj2,"tortuosity":length/max(disp2,1e-6),"max_turn_deg":float(np.max(turns)) if len(turns) else 0.,"median_vesselness":float(np.median(vl)),"p10_vesselness":float(np.percentile(vl,10)),"median_hu":float(np.median(h)),"robust_hu_fraction":float(np.mean((h>=120)&(h<=1200))),"current_support_fraction":float(np.mean(cs)),"legacy_support_fraction":float(np.mean(ls)),"union_support_fraction":float(np.mean(cs|ls)),"dual_support_fraction":float(np.mean(cs&ls))}
            m["gate_pass"]=bool(length>=3.0 and proj2>=2.0 and m["tortuosity"]<=1.9 and m["max_turn_deg"]<=65 and m["p10_vesselness"]>=.60*thr and m["robust_hu_fraction"]>=.90 and m["union_support_fraction"]>=.90)
            rows.append(m); candidates.append(lps)
            if len(rows)>=12: break
    if not rows:
        return None,{"control":control,"accepted":False,"reason":"no_distal_candidate"},pd.DataFrame()
    df=pd.DataFrame(rows); passed=df[df.gate_pass].copy()
    if passed.empty:
        return None,{"control":control,"accepted":False,"reason":"no_candidate_passed","best":df.iloc[0].to_dict()},df
    sel=passed.sort_values(["forward_projection_mm","dual_support_fraction","median_vesselness"],ascending=False).iloc[0]
    idx=int(sel.name); extension=candidates[idx]
    return extension,{"control":control,"accepted":True,"best":sel.to_dict(),"roi_lo_zyx":lo.tolist(),"roi_hi_zyx":hi.tolist()},df


def _surface_points(path, stride=4):
    im=sitk.ReadImage(str(_req(path))); a=sitk.GetArrayFromImage(im)>0
    er=ndi.binary_erosion(a,iterations=1,border_value=0); z=np.argwhere(a&~er)[::stride]
    pts=np.array([im.TransformContinuousIndexToPhysicalPoint((float(x),float(y),float(zz))) for zz,y,x in z],float)
    return pts


def _pair_interface(anchor, atrium_pts, ventricle_pts, radius=18.0):
    at=cKDTree(atrium_pts); vt=cKDTree(ventricle_pts); anchor=np.asarray(anchor,float); last=None
    for r in (radius,22.,28.,35.):
        ia=at.query_ball_point(anchor,r); iv=vt.query_ball_point(anchor,r)
        A=atrium_pts[np.asarray(ia,int)] if len(ia) else np.empty((0,3)); V=ventricle_pts[np.asarray(iv,int)] if len(iv) else np.empty((0,3))
        if len(A)<20 or len(V)<20: last=(r,len(A),len(V),0); continue
        tv=cKDTree(V); ta=cKDTree(A); d_av,jv=tv.query(A); d_va,ja=ta.query(V); mids=[]; sep=[]
        for cutoff in (5.,7.5,10.,12.5,15.):
            mids.clear(); sep.clear()
            for i,(d,j) in enumerate(zip(d_av,jv)):
                if d>cutoff: continue
                back=int(ja[int(j)])
                if np.linalg.norm(A[i]-A[back])<=1.5: mids.append((A[i]+V[int(j)])/2); sep.append(float(d))
            for j,(d,i) in enumerate(zip(d_va,ja)):
                if d>cutoff: continue
                back=int(jv[int(i)])
                if np.linalg.norm(V[j]-V[back])<=1.5: mids.append((V[j]+A[int(i)])/2); sep.append(float(d))
            if len(mids)>=30:
                P=np.asarray(mids); S=np.asarray(sep); keep=np.linalg.norm(P-anchor[None,:],axis=1)<=r; P,S=P[keep],S[keep]
                if len(P)>=30:
                    key=np.round(P/.6).astype(int); _,ix=np.unique(key,axis=0,return_index=True); P,S=P[np.sort(ix)],S[np.sort(ix)]
                    if len(P)>=20:
                        return cKDTree(P.astype(np.float32)),{"used_radius_mm":float(r),"pair_separation_cut_mm":float(cutoff),"n_interface_midpoints":int(len(P)),"median_pair_separation_mm":float(np.median(S)),"p90_pair_separation_mm":float(np.percentile(S,90))}
        last=(r,len(A),len(V),len(mids))
    raise RuntimeError(f"Unable to build chamber interface; last={last}")


def _tangents(p):
    p=np.asarray(p,float); d=np.gradient(p,axis=0); n=np.linalg.norm(d,axis=1); n[n<1e-9]=1; return d/n[:,None]


def _local_interface_tangent(tree,x,k=30):
    kk=min(k,len(tree.data)); _,idx=tree.query(np.asarray(x,float),k=kk); h=np.atleast_2d(np.asarray(tree.data[idx],float)); h=h-h.mean(axis=0)
    if len(h)<3: return np.zeros(3),0.
    _,s,vh=np.linalg.svd(h,full_matrices=False); v=_unit(vh[0]); an=float(s[0]/max(s[1],1e-9)) if len(s)>1 else float("inf"); return v,an


def _slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    return 0. if m.sum()<3 or np.ptp(x[m])<1e-6 else float(np.polyfit(x[m],y[m],1)[0])


def _interface_score(path,tree):
    p,q=_resample(path,.25); d,_=tree.query(p); t=_tangents(p); al=[]; an=[]
    for x,v in zip(p,t):
        r,a=_local_interface_tangent(tree,x); al.append(abs(float(np.dot(v,r))) if np.linalg.norm(r)>0 else 0.); an.append(a)
    al=np.asarray(al); an=np.asarray(an); med=float(np.median(d)); end=float(d[-1]); sl=_slope(q,d); align=float(np.median(al)); anis=float(np.median(an))
    closeness=float(np.exp(-med/6.)); retention=float(np.exp(-max(0.,sl)/.45)); linearity=float(np.clip((anis-1.)/2.,0,1)); score=float(.45*closeness+.30*align+.15*retention+.10*linearity)
    return {"interface_identity_score":score,"median_interface_distance_mm":med,"p90_interface_distance_mm":float(np.percentile(d,90)),"endpoint_interface_distance_mm":end,"interface_distance_slope_mm_per_mm":sl,"interface_retention_score":retention,"median_interface_tangent_alignment":align,"median_local_interface_anisotropy":anis,"profile_arc_mm":q,"profile_interface_distance_mm":d,"profile_alignment":al}


def _slice_arc(path,start,max_len=None):
    a=_arc(path); end=a[-1] if max_len is None else min(a[-1],start+max_len); q=np.arange(start,end+1e-9,.25)
    if len(q)<2: q=np.array([start,end])
    return _interp(path,q)


def _combine(original, extension):
    if extension is None or len(extension)<2: return original.copy()
    e=np.asarray(extension,float)
    if np.linalg.norm(e[0]-original[-1])>np.linalg.norm(e[-1]-original[-1]): e=e[::-1].copy()
    return np.vstack([original,e[1:]])


def _plane(ref,src,center,tangent,half=7.,step=.2):
    t=_unit(tangent); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; n=_unit(np.cross(t,seed)); b=_unit(np.cross(t,n)); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij"); pts=center[None,None,:]+xx[...,None]*n+yy[...,None]*b
    return _sample_source(ref,src,pts.reshape(-1,3)).reshape(len(q),len(q)),q


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD_PATH,root/RCA_PATH,root/COR_CURRENT,root/COR_LEGACY,root/LA,root/LV,root/RA,root/RV,root/PRIOR_RANKING]+[root/p for p in LEAF_FILES]
    for p in required: _req(p)
    master=json.loads((root/MASTER).read_text()); ref,src,spacing=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref)
    leaves=[_load_path(root/p,ref) for p in LEAF_FILES]; ranking=pd.read_csv(root/PRIOR_RANKING); mapping={int(ranking.iloc[i]["candidate_id"]):leaves[i] for i in range(5)}
    c6,c7,c9=_orient_three([mapping[6],mapping[7],mapping[9]]); split_arc,_,_=_consensus_split([c6,c7,c9])
    cur=_resample_mask(root/COR_CURRENT,ref); leg=_resample_mask(root/COR_LEGACY,ref)
    print("Reacquiring C6 distal continuation..."); ext6,s6,df6=_search_extension(ref,src,spacing,cur,leg,c6)
    print("Reacquiring C7 distal continuation..."); ext7,s7,df7=_search_extension(ref,src,spacing,cur,leg,c7)
    if df6 is not None: df6.to_csv(out/"C6_distal_candidates.csv",index=False)
    if df7 is not None: df7.to_csv(out/"C7_distal_candidates.csv",index=False)
    _write_json(out/"C6_reacquisition_summary.json",s6); _write_json(out/"C7_reacquisition_summary.json",s7)
    c6x=_combine(c6,ext6); c7x=_combine(c7,ext7)
    pd.DataFrame(c6x,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"C6_extended_path.csv",index=False)
    pd.DataFrame(c7x,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"C7_extended_path.csv",index=False)
    ext_pass=bool(s6.get("accepted") and s7.get("accepted")); source_control_pass=bool(s6.get("control",{}).get("control_pass") and s7.get("control",{}).get("control_pass"))
    la=_surface_points(root/LA); lv=_surface_points(root/LV); ra=_surface_points(root/RA); rv=_surface_points(root/RV)
    left_anchor=_interp(c6,[split_arc])[0]; left_tree,left_meta=_pair_interface(left_anchor,la,lv,18.)
    rca_arc=_arc(rca); rca_anchor_arc=min(30.,max(8.,.58*rca_arc[-1])); right_anchor=_interp(rca,[rca_anchor_arc])[0]; right_tree,right_meta=_pair_interface(right_anchor,ra,rv,18.)
    rca_seg=_slice_arc(rca,max(0.,rca_anchor_arc-3.),10.); lad_rs,lad_q=_resample(lad,.25); j=int(np.argmin(np.linalg.norm(lad_rs-left_anchor[None,:],axis=1))); lad_seg=_slice_arc(lad,max(0.,float(lad_q[j])-3.),6.)
    rcam=_interface_score(rca_seg,right_tree); ladm=_interface_score(lad_seg,left_tree); control_margin=float(rcam["interface_identity_score"]-ladm["interface_identity_score"])
    chamber_control_pass=bool(rcam["median_interface_distance_mm"]<=8. and rcam["median_interface_tangent_alignment"]>=.45 and control_margin>=.08)
    controls={"RCA_right_interface_score":rcam["interface_identity_score"],"RCA_median_interface_distance_mm":rcam["median_interface_distance_mm"],"RCA_median_tangent_alignment":rcam["median_interface_tangent_alignment"],"LAD_left_negative_score":ladm["interface_identity_score"],"LAD_median_interface_distance_mm":ladm["median_interface_distance_mm"],"LAD_median_tangent_alignment":ladm["median_interface_tangent_alignment"],"control_margin":control_margin,"control_pass":chamber_control_pass}
    _write_json(out/"chamber_interface_controls.json",controls); _write_json(out/"chamber_interface_build_summary.json",{"left":left_meta,"right":right_meta,"split_arc_mm":split_arc,"rca_anchor_arc_mm":rca_anchor_arc})
    c6post=_slice_arc(c6x,split_arc); c7post=_slice_arc(c7x,split_arc); m6=_interface_score(c6post,left_tree); m7=_interface_score(c7post,left_tree)
    win_id=6 if m6["interface_identity_score"]>=m7["interface_identity_score"] else 7; lose_id=7 if win_id==6 else 6; win=m6 if win_id==6 else m7; lose=m7 if win_id==6 else m6
    score_margin=float(win["interface_identity_score"]-lose["interface_identity_score"]); dist_adv=float(lose["median_interface_distance_mm"]-win["median_interface_distance_mm"]); align_adv=float(win["median_interface_tangent_alignment"]-lose["median_interface_tangent_alignment"]); end_adv=float(lose["endpoint_interface_distance_mm"]-win["endpoint_interface_distance_mm"]); slope_adv=float(lose["interface_distance_slope_mm_per_mm"]-win["interface_distance_slope_mm_per_mm"])
    long_enough=bool(_arc(c6post)[-1]>=10. and _arc(c7post)[-1]>=8.)
    identity_gate=bool(ext_pass and chamber_control_pass and long_enough and score_margin>=.08 and win["interface_retention_score"]>=.55 and (dist_adv>=1.5 or align_adv>=.15) and (slope_adv>=.20 or end_adv>=2.0))
    decision={"winner_source_candidate_id":win_id,"loser_source_candidate_id":lose_id,"score_margin":score_margin,"median_interface_distance_advantage_mm":dist_adv,"tangent_alignment_advantage":align_adv,"departure_slope_advantage_mm_per_mm":slope_adv,"endpoint_interface_distance_advantage_mm":end_adv,"longitudinal_length_gate_pass":long_enough,"identity_gate_pass":identity_gate}
    _write_json(out/"LCX_vs_OM_longitudinal_decision.json",decision)
    plt.figure(figsize=(7,4)); plt.plot(m6["profile_arc_mm"],m6["profile_interface_distance_mm"],label="C6 extended"); plt.plot(m7["profile_arc_mm"],m7["profile_interface_distance_mm"],label="C7 extended"); plt.xlabel("post-split arc (mm)"); plt.ylabel("distance to LA-LV interface (mm)"); plt.legend(); plt.tight_layout(); plt.savefig(out/"01_extended_interface_distance_profiles.png",dpi=180); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(m6["profile_arc_mm"],m6["profile_alignment"],label="C6 extended"); plt.plot(m7["profile_arc_mm"],m7["profile_alignment"],label="C7 extended"); plt.xlabel("post-split arc (mm)"); plt.ylabel("|tangent · interface tangent|"); plt.ylim(0,1.05); plt.legend(); plt.tight_layout(); plt.savefig(out/"02_extended_interface_alignment_profiles.png",dpi=180); plt.close()
    fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111,projection="3d"); lp=np.asarray(left_tree.data); ax.scatter(lp[:,0],lp[:,1],lp[:,2],s=5,alpha=.22,label="LA-LV interface"); ax.plot(c6post[:,0],c6post[:,1],c6post[:,2],lw=3,label="C6 extended"); ax.plot(c7post[:,0],c7post[:,1],c7post[:,2],lw=3,label="C7 extended"); ax.scatter([left_anchor[0]],[left_anchor[1]],[left_anchor[2]],s=55,label="F2 split"); ax.legend(); ax.set_title("Distal reacquisition + independent chamber interface"); plt.tight_layout(); plt.savefig(out/"03_extended_geometry.png",dpi=180); plt.close()
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    for r,(cid,p) in enumerate(((6,c6post),(7,c7post))):
        pp,qq=_resample(p,.25); picks=np.linspace(0,len(pp)-1,4).astype(int); tt=_tangents(pp)
        for j,ix in enumerate(picks):
            im,qv=_plane(ref,src,pp[ix],tt[ix]); ax=axes[r,j]; ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=25); ax.set_title(f"C{cid} +{qq[ix]:.1f} mm"); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    plt.tight_layout(); plt.savefig(out/"04_extended_orthogonal_source_qc.png",dpi=180); plt.close()
    if not source_control_pass: status=STATUS_SOURCE_CONTROL_FAIL
    elif not ext_pass: status=STATUS_SOURCE_INSUFFICIENT
    elif not chamber_control_pass: status=STATUS_CHAMBER_CONTROL_FAIL
    elif identity_gate: status=STATUS_C6 if win_id==6 else STATUS_C7
    else: status=STATUS_AMBIG
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"LCX_master_status":"UNRESOLVED","split_arc_mm":split_arc,"source_reacquisition":{"C6":s6,"C7":s7,"both_accepted":ext_pass},"chamber_interface_controls":controls,"C6_longitudinal_metrics":{k:v for k,v in m6.items() if not isinstance(v,np.ndarray)},"C7_longitudinal_metrics":{k:v for k,v in m7.items() if not isinstance(v,np.ndarray)},"decision":decision,"chamber_identity_used_during_source_extension":False,"template_similarity_used_in_decision":False,"prior_AV_groove_score_used_in_decision":False,"scientific_boundary":"Source-space distal reacquisition is frozen before chamber scoring. A positive result nominates an LCX-like parent and OM-like daughter for visual QC; it does not establish clinical vessel identity. LM remains unresolved."}
    _write_json(out/"summary.json",summary); report=out/"OPENPLAQUE_LCX_DISTAL_REACQUISITION_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque LCX Distal Reacquisition</h1><p><b>Status:</b> {status}</p><p>Source controls: C6={s6.get('control',{}).get('control_pass')}, C7={s7.get('control',{}).get('control_pass')}; extensions accepted C6={s6.get('accepted')}, C7={s7.get('accepted')}.</p><p>Chamber control pass: {chamber_control_pass}; margin {control_margin:.3f}.</p><p>C6 longitudinal score {m6['interface_identity_score']:.3f}; C7 {m7['interface_identity_score']:.3f}; winner C{win_id}; identity gate {identity_gate}.</p></body></html>",encoding="utf-8")
    _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); zpath=out/"OPENPLAQUE_LCX_DISTAL_REACQUISITION_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
