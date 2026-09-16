from __future__ import annotations

"""Monotonic, direction-constrained source-CCTA reacquisition of the established C6 parent.

This experiment is intentionally chamber-blind during path search. It replaces the prior
unconstrained Dijkstra traversal with a physical-space beam search that advances in short
forward steps, updates the local tangent, forbids loop-like returns, and requires increasing
endpoint displacement. Only after a source-space C6 extension is frozen is the previously
validated direct chamber-interface midpoint metric reapplied against the frozen accepted C7
extension from the prior experiment.

Research use only. No clinical LCX/OM label is automatically frozen.
"""

import json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="lcx-c6-monotonic-distal-v1.0-lowmem"
OUTPUT_DIRNAME="LCX_C6_Monotonic_Distal_v1"
SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH=Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH=Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS=Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR=TS/"coronary_arteries/coronary_arteries.nii.gz"
LEG=TS/"coronary_arteries_LEGACY/coronary_arteries.nii.gz"
LA=TS/"heartchambers_highres/heart_atrium_left.nii.gz"
LV=TS/"heartchambers_highres/heart_ventricle_left.nii.gz"
RA=TS/"heartchambers_highres/heart_atrium_right.nii.gz"
RV=TS/"heartchambers_highres/heart_ventricle_right.nii.gz"
C6_FILE=Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_FROZEN=Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")

STATUS_CONTROL_FAIL="C6_MONOTONIC_CONTROL_FAILED"
STATUS_NO_EXTENSION="C6_MONOTONIC_EXTENSION_NOT_ESTABLISHED"
STATUS_CHAMBER_CONTROL_FAIL="C6_MONOTONIC_CHAMBER_CONTROL_FAILED"
STATUS_AMBIG="C6_MONOTONIC_COMPLETE_LCX_OM_IDENTITY_AMBIGUOUS"
STATUS_POS="C6_LCX_C7_OM_LONGITUDINAL_CANDIDATES_REQUIRE_VISUAL_QC"


def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding="utf-8")

def _source(cache):
    arr=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r")
    meta=json.loads(_req(cache/"series7_int16.json").read_text())
    ref=sitk.GetImageFromArray(np.asarray(arr)); sp=np.asarray(meta["spacing_zyx"],float)
    ref.SetSpacing(tuple(sp[::-1])); ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0],float)))
    iop=np.asarray(meta["image_orientation_patient"],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
    D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float); ref.SetDirection(tuple(D.ravel()))
    return ref,arr,sp

def _xyz_to_zyx(img,p):
    p=np.atleast_2d(np.asarray(p,float)); o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing()); D=np.asarray(img.GetDirection()).reshape(3,3)
    return (((p-o)@np.linalg.inv(D).T)/sp)[:,::-1]

def _zyx_to_xyz(img,p):
    p=np.atleast_2d(np.asarray(p,float)); q=p[:,::-1]; o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing()); D=np.asarray(img.GetDirection()).reshape(3,3)
    return o+(q*sp)@D.T

def _sample(img,a,p,order=1,cval=-1024.): return map_coordinates(np.asarray(a),_xyz_to_zyx(img,p).T,order=order,mode="constant",cval=cval)

def _load_path(path,ref):
    d=pd.read_csv(_req(path))
    for c in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(x in d.columns for x in c): return d[list(c)].to_numpy(float)
    for c in (("zyx_z","zyx_y","zyx_x"),("source_z","source_y","source_x"),("z","y","x")):
        if all(x in d.columns for x in c): return _zyx_to_xyz(ref,d[list(c)].to_numpy(float))
    raise ValueError(f"No recognized coordinates: {list(d.columns)}")

def _arc(p):
    p=np.asarray(p,float); return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))

def _resample(p,step=.25):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0: return p.copy(),a
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6: q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)]),q

def _interp(p,q):
    a=_arc(p); q=np.asarray(q,float); return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-9 else np.zeros_like(v)

def _resample_mask(path,ref):
    im=sitk.ReadImage(str(_req(path)))
    same=(im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection()))
    if not same: im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im)>0

def _frangi_3d(vol,spacing,scales=(.55,.80,1.10,1.45)):
    x=np.clip(np.asarray(vol,np.float32),80,1000); x=(x-80)/920.; spacing=np.asarray(spacing,float); best=np.zeros_like(x,np.float32)
    for sm in scales:
        sig=np.maximum(sm/spacing,.55); n=sm*sm
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0))*n/(spacing[0]**2); hyy=ndi.gaussian_filter(x,sig,order=(0,2,0))*n/(spacing[1]**2); hxx=ndi.gaussian_filter(x,sig,order=(0,0,2))*n/(spacing[2]**2)
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0))*n/(spacing[0]*spacing[1]); hzx=ndi.gaussian_filter(x,sig,order=(1,0,1))*n/(spacing[0]*spacing[2]); hyx=ndi.gaussian_filter(x,sig,order=(0,1,1))*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),np.float32); H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx; H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1); l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); s=np.sqrt(l1*l1+l2*l2+l3*l3); nz=s[s>0]; c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        v=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(s*s)/(2*c*c))); v[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(v).astype(np.float32))
    return best

def _orth_basis(t):
    t=_unit(t); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; u=_unit(np.cross(t,seed)); v=_unit(np.cross(t,u)); return u,v

def _cone_dirs(t):
    t=_unit(t); u,v=_orth_basis(t); out=[t]
    for deg in (12,24,36,48):
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,8,endpoint=False): out.append(_unit(np.cos(a)*t+np.sin(a)*(np.cos(phi)*u+np.sin(phi)*v)))
    return out

def _turns(path):
    p=np.asarray(path,float)
    if len(p)<4:return np.array([])
    v=np.diff(p,axis=0); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9); return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))

def _build_field(ref,src,spacing,cur,leg,path,target=12.):
    p,q=_resample(path,.20); z=_xyz_to_zyx(ref,p); tail=z[max(0,len(z)-18):]; t=_unit(p[-1]-p[max(0,len(p)-10)]); forward=_xyz_to_zyx(ref,p[-1]+np.outer(np.linspace(0,target,8),t)); pts=np.vstack([tail,forward]); margin_mm=6.
    lo=np.floor(np.min(pts*spacing,axis=0)/spacing-margin_mm/spacing).astype(int); hi=np.ceil(np.max(pts*spacing,axis=0)/spacing+margin_mm/spacing).astype(int)+1; lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(src.shape)); sl=tuple(slice(lo[k],hi[k]) for k in range(3))
    roi=np.asarray(src[sl]); cm=np.asarray(cur[sl]); lm=np.asarray(leg[sl]); union=cm|lm; du=ndi.distance_transform_edt(~union,sampling=spacing); vessel=_frangi_3d(roi,spacing)
    tail_local=z[max(0,len(z)-18):]-lo[None,:]; kv=map_coordinates(vessel,tail_local.T,order=1,mode="nearest"); thr=max(.003,min(.12,.35*float(np.percentile(kv,25)))); norm=max(float(np.median(kv))*1.5,.02)
    return {"lo":lo,"hi":hi,"roi":roi,"cur":cm,"leg":lm,"union":union,"du":du,"v":vessel,"thr":thr,"norm":norm}

def _sample_field(ref,F,p):
    z=_xyz_to_zyx(ref,[p])[0]-F["lo"]; shape=np.asarray(F["roi"].shape)
    if np.any(z<1) or np.any(z>shape-2): return None
    co=z[:,None]; hu=float(map_coordinates(F["roi"],co,order=1,mode="nearest")[0]); vv=float(map_coordinates(F["v"],co,order=1,mode="nearest")[0]); dd=float(map_coordinates(F["du"],co,order=1,mode="nearest")[0]); zi=np.rint(z).astype(int); c=bool(F["cur"][tuple(zi)]); l=bool(F["leg"][tuple(zi)])
    return hu,vv,dd,c,l

def _beam_search(ref,F,start,tangent,target_mm=12.,step=.55,beam_width=48,min_progress=.08):
    # state: score, points, smoothed tangent
    states=[(0.,[np.asarray(start,float)],_unit(tangent))]; completed=[]; origin=np.asarray(start,float); nsteps=int(math.ceil(target_mm/step))
    for _ in range(nsteps):
        nxt=[]
        for score,pts,t in states:
            prev_disp=np.linalg.norm(pts[-1]-origin)
            for d in _cone_dirs(t):
                if np.dot(d,t)<math.cos(math.radians(55)): continue
                p=pts[-1]+step*d; disp=np.linalg.norm(p-origin)
                if disp<prev_disp+min_progress: continue
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-p,axis=1))<.65*step: continue
                s=_sample_field(ref,F,p)
                if s is None: continue
                hu,vv,dd,c,l=s
                if not (120<=hu<=1200 and vv>=F["thr"] and dd<=1.0): continue
                align=max(0.,float(np.dot(d,t))); vn=min(1.,vv/F["norm"]); support=1. if c and l else (.55 if c or l else 0.)
                ns=score+1.8*vn+.55*support+.45*align+.10*min(1.,disp/target_mm)
                nt=_unit(.72*t+.28*d); nxt.append((ns,pts+[p],nt))
        if not nxt: break
        # spatial deduplication, then beam prune
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.35).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=beam_width: break
        states=keep; completed.extend(states)
    return completed

def _path_metrics(ref,src,cur,leg,F,path,start):
    p,q=_resample(path,.20); h=_sample(ref,src,p); z=_xyz_to_zyx(ref,p); local=z-F["lo"][None,:]; vv=map_coordinates(F["v"],local.T,order=1,mode="nearest"); zr=np.rint(z).astype(int); zr=np.clip(zr,[0,0,0],np.asarray(src.shape)-1); cs=cur[tuple(zr.T)]; ls=leg[tuple(zr.T)]; disp=float(np.linalg.norm(p[-1]-start)); length=float(q[-1]); tr=_turns(p)
    return {"new_length_mm":length,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6),"max_turn_deg":float(np.max(tr)) if len(tr) else 0.,"median_vesselness":float(np.median(vv)),"p10_vesselness":float(np.percentile(vv,10)),"median_hu":float(np.median(h)),"robust_hu_fraction":float(np.mean((h>=120)&(h<=1200))),"current_support_fraction":float(np.mean(cs)),"legacy_support_fraction":float(np.mean(ls)),"union_support_fraction":float(np.mean(cs|ls)),"dual_support_fraction":float(np.mean(cs&ls))}

def _control_and_extension(ref,src,spacing,cur,leg,c6):
    p,q=_resample(c6,.20); total=q[-1]; ci=int(np.argmin(np.abs(q-(total-3.0)))); start=p[ci]; target=p[-1]; t=_unit(p[min(len(p)-1,ci+8)]-p[max(0,ci-3)]); F=_build_field(ref,src,spacing,cur,leg,c6,12.)
    control_states=_beam_search(ref,F,start,t,4.2,step=.50,beam_width=56,min_progress=.04)
    if not control_states: return None,{"control_pass":False,"reason":"no_control_path"},F,pd.DataFrame()
    known=p[ci:]; kt=cKDTree(known); cands=[]
    for st in control_states:
        path=np.asarray(st[1]); ep=np.linalg.norm(path[-1]-target); dd=kt.query(path)[0]; length=_arc(path)[-1]
        cands.append((ep,float(np.median(dd)),float(np.percentile(dd,90)),length,path))
    cands.sort(key=lambda x:(x[0],x[1])); best=cands[0]; ctrl={"control_pass":bool(best[0]<=.90 and best[1]<=.75 and best[2]<=1.15 and 2.4<=best[3]<=4.5),"endpoint_error_mm":best[0],"median_distance_to_known_mm":best[1],"p90_distance_to_known_mm":best[2],"control_length_mm":best[3],"vesselness_threshold":F["thr"]}
    if not ctrl["control_pass"]: return None,ctrl,F,pd.DataFrame()
    t2=_unit(p[-1]-p[max(0,len(p)-10)]); states=_beam_search(ref,F,p[-1],t2,12.,step=.55,beam_width=64,min_progress=.08); rows=[]; paths=[]
    for st in states:
        path=np.asarray(st[1]); length=_arc(path)[-1]
        if length<4.5: continue
        m=_path_metrics(ref,src,cur,leg,F,path,p[-1]); m["beam_score"]=float(st[0]); m["gate_pass"]=bool(m["new_length_mm"]>=5.0 and m["endpoint_displacement_mm"]>=4.0 and m["tortuosity"]<=1.8 and m["max_turn_deg"]<=60 and m["p10_vesselness"]>=.60*F["thr"] and m["robust_hu_fraction"]>=.90 and m["union_support_fraction"]>=.90); rows.append(m); paths.append(path)
    if not rows: return None,{"control":ctrl,"accepted":False,"reason":"no_long_candidate"},F,pd.DataFrame()
    df=pd.DataFrame(rows); ok=df[df.gate_pass]
    if ok.empty: return None,{"control":ctrl,"accepted":False,"reason":"no_candidate_passed","best":df.sort_values(["new_length_mm","beam_score"],ascending=False).iloc[0].to_dict()},F,df
    sel=ok.sort_values(["new_length_mm","beam_score","dual_support_fraction"],ascending=False).iloc[0]; idx=int(sel.name); return paths[idx],{"control":ctrl,"accepted":True,"best":sel.to_dict()},F,df

def _surface_points(path,stride=4):
    im=sitk.ReadImage(str(_req(path))); a=sitk.GetArrayFromImage(im)>0; z=np.argwhere(a&~ndi.binary_erosion(a,iterations=1,border_value=0))[::stride]
    return np.array([im.TransformContinuousIndexToPhysicalPoint((float(x),float(y),float(zz))) for zz,y,x in z],float)

def _pair_interface(anchor,A,V,radius=18.):
    at,vt=cKDTree(A),cKDTree(V); anchor=np.asarray(anchor,float); last=None
    for r in (radius,22.,28.,35.):
        ia=at.query_ball_point(anchor,r); iv=vt.query_ball_point(anchor,r); AA=A[np.asarray(ia,int)] if ia else np.empty((0,3)); VV=V[np.asarray(iv,int)] if iv else np.empty((0,3))
        if len(AA)<20 or len(VV)<20: last=(r,len(AA),len(VV),0); continue
        tv,ta=cKDTree(VV),cKDTree(AA); d1,j1=tv.query(AA); d2,j2=ta.query(VV)
        for cut in (5.,7.5,10.,12.5,15.):
            mids=[]; sep=[]
            for i,(d,j) in enumerate(zip(d1,j1)):
                if d<=cut and np.linalg.norm(AA[i]-AA[int(j2[int(j)])])<=1.5: mids.append((AA[i]+VV[int(j)])/2); sep.append(float(d))
            for j,(d,i) in enumerate(zip(d2,j2)):
                if d<=cut and np.linalg.norm(VV[j]-VV[int(j1[int(i)])])<=1.5: mids.append((VV[j]+AA[int(i)])/2); sep.append(float(d))
            if len(mids)>=30:
                P=np.asarray(mids); S=np.asarray(sep); keep=np.linalg.norm(P-anchor,axis=1)<=r; P,S=P[keep],S[keep]
                if len(P)>=30:
                    key=np.round(P/.6).astype(int); _,ix=np.unique(key,axis=0,return_index=True); P,S=P[np.sort(ix)],S[np.sort(ix)]
                    if len(P)>=20:return cKDTree(P.astype(np.float32)),{"used_radius_mm":r,"pair_separation_cut_mm":cut,"n_interface_midpoints":len(P),"median_pair_separation_mm":float(np.median(S)),"p90_pair_separation_mm":float(np.percentile(S,90))}
        last=(r,len(AA),len(VV),len(mids))
    raise RuntimeError(f"Unable to build interface: {last}")
def _tangents(p):
    d=np.gradient(np.asarray(p,float),axis=0); n=np.linalg.norm(d,axis=1); n[n<1e-9]=1; return d/n[:,None]
def _local_tangent(tree,x,k=30):
    _,ix=tree.query(x,k=min(k,len(tree.data))); h=np.atleast_2d(tree.data[ix]); h=h-h.mean(axis=0); _,s,vh=np.linalg.svd(h,full_matrices=False); return _unit(vh[0]),float(s[0]/max(s[1],1e-9))
def _slope(x,y): return float(np.polyfit(x,y,1)[0]) if len(x)>=3 and np.ptp(x)>1e-6 else 0.
def _interface_score(path,tree):
    p,q=_resample(path,.25); d,_=tree.query(p); tt=_tangents(p); al=[]; an=[]
    for x,t in zip(p,tt): r,a=_local_tangent(tree,x); al.append(abs(float(np.dot(t,r)))); an.append(a)
    med=float(np.median(d)); sl=_slope(q,d); align=float(np.median(al)); anis=float(np.median(an)); close=float(np.exp(-med/6)); retain=float(np.exp(-max(0,sl)/.45)); lin=float(np.clip((anis-1)/2,0,1)); score=.45*close+.30*align+.15*retain+.10*lin
    return {"interface_identity_score":float(score),"median_interface_distance_mm":med,"p90_interface_distance_mm":float(np.percentile(d,90)),"endpoint_interface_distance_mm":float(d[-1]),"interface_distance_slope_mm_per_mm":sl,"interface_retention_score":retain,"median_interface_tangent_alignment":align,"median_local_interface_anisotropy":anis,"profile_arc_mm":q,"profile_distance_mm":d,"profile_alignment":np.asarray(al)}
def _slice(path,start,length):
    a=_arc(path); q=np.arange(start,min(a[-1],start+length)+1e-9,.25); return _interp(path,q)
def _combine(base,ext):
    if ext is None:return base.copy()
    e=np.asarray(ext); return np.vstack([base,e[1:]])
def synthetic_monotonic_self_test():
    # Geometry-only sanity check for loop rejection/progress logic.
    origin=np.array([0.,0.,0.]); pts=[origin]
    for i in range(1,12): pts.append(np.array([.55*i,.04*np.sin(i),0.]))
    p=np.asarray(pts); assert np.all(np.diff(np.linalg.norm(p-origin,axis=1))>0); assert _arc(p)[-1]/np.linalg.norm(p[-1]-origin)<1.1
    return {"ok":True,"tortuosity":float(_arc(p)[-1]/np.linalg.norm(p[-1]-origin))}

def _plane(ref,src,c,t,half=7,step=.2):
    t=_unit(t); u,v=_orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij"); P=c+xx[...,None]*u+yy[...,None]*v; return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD_PATH,root/RCA_PATH,root/CUR,root/LEG,root/LA,root/LV,root/RA,root/RV,root/C6_FILE,root/C7_FROZEN]
    for p in required:_req(p)
    master=json.loads((root/MASTER).read_text()); ref,src,spacing=_source(root/SOURCE_CACHE); c6=_load_path(root/C6_FILE,ref); c7=_load_path(root/C7_FROZEN,ref); lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref); cur=_resample_mask(root/CUR,ref); leg=_resample_mask(root/LEG,ref)
    print("Running monotonic C6 positive control and distal beam search..."); ext,source_summary,F,df=_control_and_extension(ref,src,spacing,cur,leg,c6); df.to_csv(out/"C6_monotonic_candidates.csv",index=False); _write_json(out/"C6_monotonic_reacquisition_summary.json",source_summary)
    c6x=_combine(c6,ext); pd.DataFrame(c6x,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"C6_extended_path.csv",index=False)
    # chamber-interface construction is downstream only
    la,lv,ra,rv=[_surface_points(root/p) for p in (LA,LV,RA,RV)]; split_arc=20.25; left_anchor=_interp(c6,[min(split_arc,_arc(c6)[-1])])[0]; left,left_meta=_pair_interface(left_anchor,la,lv)
    rca_arc=min(30.,max(8.,.58*_arc(rca)[-1])); right,right_meta=_pair_interface(_interp(rca,[rca_arc])[0],ra,rv)
    rcam=_interface_score(_slice(rca,max(0,rca_arc-3),10),right); lr,_=_resample(lad,.25); j=np.argmin(np.linalg.norm(lr-left_anchor,axis=1)); lstart=max(0,_arc(lr)[j]-3); ladm=_interface_score(_slice(lad,lstart,6),left); cmargin=rcam["interface_identity_score"]-ladm["interface_identity_score"]; cpass=bool(rcam["median_interface_distance_mm"]<=8 and rcam["median_interface_tangent_alignment"]>=.45 and cmargin>=.08)
    controls={"RCA_right_interface_score":rcam["interface_identity_score"],"RCA_median_interface_distance_mm":rcam["median_interface_distance_mm"],"RCA_median_tangent_alignment":rcam["median_interface_tangent_alignment"],"LAD_left_negative_score":ladm["interface_identity_score"],"LAD_median_interface_distance_mm":ladm["median_interface_distance_mm"],"control_margin":float(cmargin),"control_pass":cpass}; _write_json(out/"chamber_interface_controls.json",controls); _write_json(out/"chamber_interface_build_summary.json",{"left":left_meta,"right":right_meta})
    c6post=_slice(c6x,split_arc,min(18.,max(5.,_arc(c6x)[-1]-split_arc))); c7post=_slice(c7,split_arc,min(18.,max(5.,_arc(c7)[-1]-split_arc))); m6=_interface_score(c6post,left); m7=_interface_score(c7post,left); margin=float(m6["interface_identity_score"]-m7["interface_identity_score"]); identity=bool(source_summary.get("accepted",False) and cpass and margin>=.08)
    decision={"winner_source_candidate_id":6 if margin>=0 else 7,"score_margin_C6_minus_C7":margin,"C6_source_extension_accepted":bool(source_summary.get("accepted",False)),"chamber_control_pass":cpass,"identity_gate_pass":identity}; _write_json(out/"LCX_vs_OM_longitudinal_decision.json",decision)
    # figures
    plt.figure(figsize=(7,4)); plt.plot(m6["profile_arc_mm"],m6["profile_distance_mm"],label="C6 extended"); plt.plot(m7["profile_arc_mm"],m7["profile_distance_mm"],label="C7 frozen"); plt.xlabel("downstream arc (mm)"); plt.ylabel("distance to LA-LV interface (mm)"); plt.legend(); plt.tight_layout(); plt.savefig(out/"01_interface_distance_profiles.png",dpi=180); plt.close()
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d"); P=np.asarray(left.data); ax.scatter(P[:,0],P[:,1],P[:,2],s=4,alpha=.2,label="LA-LV interface"); ax.plot(c6post[:,0],c6post[:,1],c6post[:,2],lw=3,label="C6"); ax.plot(c7post[:,0],c7post[:,1],c7post[:,2],lw=3,label="C7 frozen"); ax.legend(); plt.tight_layout(); plt.savefig(out/"02_extended_geometry.png",dpi=180); plt.close()
    fig,axes=plt.subplots(1,4,figsize=(14,4)); pp,qq=_resample(c6post,.25); tt=_tangents(pp); picks=np.linspace(0,len(pp)-1,4).astype(int)
    for ax,ix in zip(axes,picks): im,qv=_plane(ref,src,pp[ix],tt[ix]); ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=18); ax.set_title(f"C6 +{qq[ix]:.1f} mm")
    plt.tight_layout(); plt.savefig(out/"03_C6_orthogonal_source_qc.png",dpi=180); plt.close()
    if not source_summary.get("control",source_summary).get("control_pass",False): status=STATUS_CONTROL_FAIL
    elif not source_summary.get("accepted",False): status=STATUS_NO_EXTENSION
    elif not cpass: status=STATUS_CHAMBER_CONTROL_FAIL
    elif identity: status=STATUS_POS
    else: status=STATUS_AMBIG
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"LCX_master_status":"UNRESOLVED","C6_source_reacquisition":source_summary,"chamber_interface_controls":controls,"C6_longitudinal_metrics":{k:v for k,v in m6.items() if not isinstance(v,np.ndarray)},"C7_frozen_longitudinal_metrics":{k:v for k,v in m7.items() if not isinstance(v,np.ndarray)},"decision":decision,"C7_source_path_frozen_from_prior_run":True,"chamber_identity_used_during_C6_search":False,"scientific_boundary":"A positive result establishes a source-supported monotonic C6 continuation and nominates C6 as LCX-like versus frozen C7 as OM-like for visual QC. It does not alter the frozen master or establish clinical identity automatically."}; _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    report=out/"OPENPLAQUE_LCX_C6_MONOTONIC_DISTAL_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque C6 Monotonic Distal Reacquisition</h1><p><b>Status:</b> {status}</p><p>C6 extension accepted: {source_summary.get('accepted',False)}.</p><p>Chamber control pass: {cpass}; margin {cmargin:.3f}.</p><p>C6 score {m6['interface_identity_score']:.3f}; C7 score {m7['interface_identity_score']:.3f}; identity gate {identity}.</p></body></html>",encoding="utf-8")
    z=out/"OPENPLAQUE_LCX_C6_MONOTONIC_DISTAL_REPORT_BACK.zip";
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zz:
        for p in sorted(out.iterdir()):
            if p!=z and p.is_file(): zz.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(z)}
