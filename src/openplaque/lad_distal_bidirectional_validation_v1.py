from __future__ import annotations

"""Bidirectional source-CCTA validation of the independently traced distal LAD extension.

Prerequisite: LAD_Distal_Endpoint_Continuation_v1 produced a source-supported 15.2-mm
extension but did not reproduce the earlier blind trajectory over its full length. This
experiment asks a different question: does the newly traced distal segment itself behave as
a stable vessel when traced in reverse, from its far endpoint back toward the frozen LAD?

The forward candidate is used only to define the reverse starting endpoint and initial reverse
tangent, and then for post-hoc agreement. During reverse discovery, candidate coordinates are
not used in scoring or gating. A known-terminal-LAD reverse recovery is required as a positive
control. Frozen anatomy is never modified.
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
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lad-distal-bidirectional-validation-v1.0"
OUTPUT_DIRNAME = "LAD_Distal_Bidirectional_Validation_v1"
SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
PRIOR_DIR = Path("LAD_Distal_Endpoint_Continuation_v1")
PRIOR_SUMMARY = PRIOR_DIR / "summary.json"
FORWARD_PATH = PRIOR_DIR / "best_independent_distal_extension.csv"

STEP_MM = 0.40
PLANE_STEP_MM = 0.40
BEAM_WIDTH = 72
CONTROL_SEARCH_MM = 6.0
REVERSE_SEARCH_MM = 17.0
VESSELNESS_MARGIN_MM = 22.0
SUSTAINED_FAIL_N = 3
HU_MIN = 100.0
HU_MAX = 1400.0
MIN_VESSELNESS_FACTOR = 0.20
MAX_STEP_TURN_DEG = 55.0
MIN_CONTROL_ARC_MM = 5.0
MIN_CONTROL_OVERLAP_FRACTION = 0.80
MAX_CONTROL_MEDIAN_DISTANCE_MM = 1.0
MIN_REVERSE_QC_ARC_MM = 12.0
MIN_PLANE_PASS_FRACTION = 0.80
MAX_REACH_ENDPOINT_MM = 1.50
MIN_FORWARD_OVERLAP_FRACTION = 0.80
MAX_FORWARD_MEDIAN_DISTANCE_MM = 1.0
MAX_FORWARD_P90_DISTANCE_MM = 2.0
MIN_FORWARD_TANGENT_ALIGNMENT = 0.80
MIN_CONFIRMING_HYPOTHESES = 3

STATUS_PREREQ = "LAD_DISTAL_BIDIRECTIONAL_PREREQUISITE_FAILED"
STATUS_CONTROL = "LAD_DISTAL_BIDIRECTIONAL_KNOWN_LAD_CONTROL_FAILED"
STATUS_NONE = "LAD_DISTAL_BIDIRECTIONAL_NO_CONFIRMING_REVERSE_TRACE"
STATUS_CONFIRMED = "LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED"


def _req(p):
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _read_json(p): return json.loads(_req(p).read_text(encoding="utf-8"))
def _write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding="utf-8")
def _load_path(p):
    d=pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinate columns in {p}: {list(d.columns)}")
def _save_path(p,path): pd.DataFrame(np.asarray(path,float),columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(p,index=False)
def _arc(p):
    p=np.asarray(p,float)
    return np.zeros(len(p)) if len(p)<=1 else np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
def _interp(p,q):
    p=np.asarray(p,float);a=_arc(p);q=np.asarray(q,float)
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])
def _resample(p,step=.25):
    p=np.asarray(p,float);a=_arc(p)
    if len(p)<2 or a[-1]<=0:return p.copy(),a
    q=np.arange(0.,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6:q=np.r_[q,a[-1]]
    return _interp(p,q),q
def _unit(v):
    v=np.asarray(v,float);n=float(np.linalg.norm(v));return v/n if n>1e-9 else np.zeros_like(v)
def _angle(a,b): return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a),_unit(b)),-1.,1.))))

class SourceGeometry:
    def __init__(self,meta):
        self.spacing_zyx=np.asarray(meta["spacing_zyx"],float);self.spacing_xyz=self.spacing_zyx[::-1];self.origin=np.asarray(meta["positions_lps_mm"][0],float)
        iop=np.asarray(meta["image_orientation_patient"],float);row,col=iop[:3],iop[3:];slc=np.cross(row,col)
        self.D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float);self.invD=np.linalg.inv(self.D)
    def xyz_to_zyx(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float));xyz=((pts-self.origin)@self.invD.T)/self.spacing_xyz;return xyz[:,::-1]

def _source(cache):
    cache=Path(cache);src=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r");meta=_read_json(cache/"series7_int16.json");return SourceGeometry(meta),src
def _sample(geom,arr,pts,cval=0.0): return map_coordinates(np.asarray(arr),geom.xyz_to_zyx(pts).T,order=1,mode="constant",cval=float(cval))

def _frangi_3d(vol,spacing_zyx,scales=(0.55,0.80,1.10,1.45)):
    x=np.clip(np.asarray(vol,np.float32),80.,1000.);x=(x-80.)/920.;spacing=np.asarray(spacing_zyx,float);best=np.zeros_like(x,dtype=np.float32);eps=1e-8
    for sm in scales:
        sig=np.maximum(sm/spacing,.55);n=float(sm*sm)
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0),mode="nearest")*n/spacing[0]**2;hyy=ndi.gaussian_filter(x,sig,order=(0,2,0),mode="nearest")*n/spacing[1]**2;hxx=ndi.gaussian_filter(x,sig,order=(0,0,2),mode="nearest")*n/spacing[2]**2
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0),mode="nearest")*n/(spacing[0]*spacing[1]);hzx=ndi.gaussian_filter(x,sig,order=(1,0,1),mode="nearest")*n/(spacing[0]*spacing[2]);hyx=ndi.gaussian_filter(x,sig,order=(0,1,1),mode="nearest")*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),dtype=np.float32);H[...,0,0],H[...,1,1],H[...,2,2]=hzz,hyy,hxx;H[...,0,1]=H[...,1,0]=hzy;H[...,0,2]=H[...,2,0]=hzx;H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H);vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1);l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]
        ra=np.abs(l2)/(np.abs(l3)+eps);rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps);ss=np.sqrt(l1*l1+l2*l2+l3*l3);nz=ss[ss>0];c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        vv=(1.-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1.-np.exp(-(ss*ss)/(2.*c*c)));vv[(l2>=0)|(l3>=0)]=0.;best=np.maximum(best,np.nan_to_num(vv).astype(np.float32));del H,vals,hzz,hyy,hxx,hzy,hzx,hyx,vv
    return best

class LocalVesselness:
    def __init__(self,geom,src,anchor_pts,margin_mm=VESSELNESS_MARGIN_MM):
        z=geom.xyz_to_zyx(np.asarray(anchor_pts,float));mv=float(margin_mm)/geom.spacing_zyx;lo=np.maximum(np.floor(z.min(axis=0)-mv).astype(int),0);hi=np.minimum(np.ceil(z.max(axis=0)+mv).astype(int)+1,np.asarray(src.shape));sl=tuple(slice(int(lo[k]),int(hi[k])) for k in range(3));roi=np.asarray(src[sl],dtype=np.float32)
        if np.any(hi-lo<7):raise RuntimeError(f"local vesselness ROI too small: {lo.tolist()} {hi.tolist()}")
        self.field=_frangi_3d(roi,geom.spacing_zyx);self.lo=lo.astype(float);self.shape=tuple(int(v) for v in roi.shape)
    def sample(self,geom,pts):return map_coordinates(self.field,(geom.xyz_to_zyx(pts)-self.lo[None,:]).T,order=1,mode="constant",cval=0.)

def _orth_basis(t):
    t=_unit(t);axes=np.eye(3);seed=axes[np.argmin(np.abs(axes@t))];u=_unit(np.cross(t,seed));return u,_unit(np.cross(t,u))
def _cone_dirs(t,degrees=(0,10,20,30,40),nphi=10):
    t=_unit(t);u,v=_orth_basis(t);out=[t]
    for deg in degrees:
        if deg==0:continue
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,nphi,endpoint=False):out.append(_unit(np.cos(a)*t+np.sin(a)*(np.cos(phi)*u+np.sin(phi)*v)))
    return out
def _rank_initial_dirs(geom,src,ves,start,tangent,vnorm,n=8):
    ranked=[]
    for d in _cone_dirs(tangent,degrees=(0,8,16,24,32,40),nphi=12):
        q=np.array([.4,.8,1.2,1.6]);pts=np.asarray(start)[None,:]+q[:,None]*d[None,:];hu=_sample(geom,src,pts,cval=-1024.);vv=ves.sample(geom,pts)
        if float(np.mean((hu>=HU_MIN)&(hu<=HU_MAX)))<.75:continue
        sc=1.8*float(np.mean(np.clip(vv/max(vnorm,1e-6),0,1.5)))+.30*float(np.mean(np.clip((hu-HU_MIN)/500.,0,1)))+.20*max(0.,float(np.dot(_unit(d),_unit(tangent))));ranked.append((sc,d))
    ranked.sort(key=lambda x:x[0],reverse=True);keep=[]
    for sc,d in ranked:
        if all(_angle(d,kd)>=10. for _,kd in keep):keep.append((sc,d))
        if len(keep)>=n:break
    return keep

def _beam(geom,src,ves,start,initial_dir,vthr,vnorm,max_mm):
    states=[(0.,[np.asarray(start,float)],_unit(initial_dir))];best=states[0];rows=[]
    for si in range(int(math.ceil(max_mm/STEP_MM))):
        nxt=[];cnt={"step":si,"arc_budget_mm":float((si+1)*STEP_MM),"states_in":len(states),"proposals":0,"reject_loop":0,"reject_hu":0,"reject_vesselness":0,"accepted":0,"kept":0}
        for score,pts,t in states:
            for d in _cone_dirs(t):
                cnt["proposals"]+=1
                if np.dot(d,t)<math.cos(math.radians(MAX_STEP_TURN_DEG)):continue
                q=np.asarray(pts[-1])+STEP_MM*d
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-q,axis=1))<.60*STEP_MM:cnt["reject_loop"]+=1;continue
                hu=float(_sample(geom,src,[q],cval=-1024.)[0]);vv=float(ves.sample(geom,[q])[0])
                if not(HU_MIN<=hu<=HU_MAX):cnt["reject_hu"]+=1;continue
                if vv<vthr:cnt["reject_vesselness"]+=1;continue
                vn=min(1.5,max(0.,vv/max(vnorm,1e-6)));align=max(0.,float(np.dot(d,t)));ns=score+1.8*vn+.30*min(1.,max(0.,hu-HU_MIN)/500.)+.35*align;nt=_unit(.72*t+.28*d);st=(ns,pts+[q],nt);nxt.append(st);cnt["accepted"]+=1
                if len(st[1])>len(best[1]) or (len(st[1])==len(best[1]) and ns>best[0]):best=st
        if not nxt:rows.append(cnt);break
        nxt.sort(key=lambda x:x[0],reverse=True);keep=[];bins=set()
        for st in nxt:
            key=tuple(np.round(np.asarray(st[1][-1])/.35).astype(int))
            if key in bins:continue
            bins.add(key);keep.append(st)
            if len(keep)>=BEAM_WIDTH:break
        states=keep;cnt["kept"]=len(states);rows.append(cnt)
    finals=[]
    for st in list(states)+[best]:
        ep=np.asarray(st[1][-1])
        if all(np.linalg.norm(ep-np.asarray(x[1][-1]))>1. for x in finals):finals.append(st)
    finals.sort(key=lambda s:(len(s[1]),s[0]),reverse=True);return finals[:6],pd.DataFrame(rows)

def _plane(geom,src,c,t,half=5.5,step=.20):
    t=_unit(t);u,v=_orth_basis(t);q=np.arange(-half,half+1e-9,step);yy,xx=np.meshgrid(q,q,indexing="ij");P=c+xx[...,None]*u+yy[...,None]*v;return _sample(geom,src,P.reshape(-1,3),cval=-1024.).reshape(len(q),len(q)),q
def _component_metrics(im,q):
    iy=ix=int(np.argmin(np.abs(q)));center=float(im[iy,ix]);yy,xx=np.meshgrid(q,q,indexing="ij");rr=np.sqrt(xx*xx+yy*yy);thr=max(220.,min(500.,.55*center));bw=(im>=thr)&(rr<=3.5);lab,_=ndi.label(bw,np.ones((3,3),int));labels=[]
    if lab[iy,ix]>0:labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt(q[pts[:,1]]**2+q[pts[:,0]]**2);k=int(np.argmin(dist));labels=[int(lab[tuple(pts[k])])] if dist[k]<=1. else []
    if not labels:return {"center_hu":center,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
    mask=lab==labels[0];pts=np.argwhere(mask);xs=q[pts[:,1]];ys=q[pts[:,0]];off=float(np.hypot(np.mean(xs),np.mean(ys)));rad=float(np.sqrt((len(pts)*.04)/np.pi));axis=np.inf
    if len(pts)>=4:
        ev=np.linalg.eigvalsh(np.cov(np.column_stack([xs,ys]).T));axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    ring=(rr>=3.5)&(rr<=5.);contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan;return {"center_hu":center,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}
def _plane_pass(m):return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2 and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)
def _dense_qc(geom,src,path):
    p,q=_resample(path,PLANE_STEP_MM)
    if len(p)<2:return p,q,pd.DataFrame([{"index":0,"arc_mm":0.,"plane_pass":False}])
    tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,g=_plane(geom,src,c,t);m=_component_metrics(im,g);m.update(index=i,arc_mm=float(a),plane_pass=_plane_pass(m));rows.append(m)
    return p,q,pd.DataFrame(rows)
def _truncate_qc(df):
    ps=df.plane_pass.astype(bool).to_numpy();cut=len(df)-1;first=None
    for i in range(max(0,len(ps)-SUSTAINED_FAIL_N+1)):
        if not np.any(ps[i:i+SUSTAINED_FAIL_N]):first=i;cut=max(0,i-1);break
    d=df.iloc[:cut+1];return {"first_sustained_failure_index":first,"accepted_last_index":int(cut),"accepted_arc_mm":float(d.arc_mm.iloc[-1]) if len(d) else 0.,"accepted_plane_pass_fraction":float(d.plane_pass.mean()) if len(d) else 0.,"covers_full_path":bool(cut==len(df)-1)}
def _safe_tangents(p):
    p=np.asarray(p,float)
    if len(p)<2:return np.zeros_like(p)
    t=np.gradient(p,axis=0);return t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
def _overlap_metrics(query,ref,step=.25):
    q,_=_resample(query,step);r,_=_resample(ref,step)
    if len(q)==0 or len(r)==0:return {"min_distance_mm":np.inf,"median_distance_mm":np.inf,"p90_distance_mm":np.inf,"endpoint_distance_mm":np.inf,"fraction_within_2mm":0.,"max_contiguous_span_within_2mm":0.,"median_tangent_alignment_within_2mm":0.}
    tree=cKDTree(r);d,ix=tree.query(q);al=np.abs(np.sum(_safe_tangents(q)*_safe_tangents(r)[ix],axis=1));within=d<=2.;best=cur=0.
    for v in within:cur=cur+step if v else 0.;best=max(best,cur)
    return {"min_distance_mm":float(np.min(d)),"median_distance_mm":float(np.median(d)),"p90_distance_mm":float(np.percentile(d,90)),"endpoint_distance_mm":float(d[-1]),"fraction_within_2mm":float(np.mean(within)),"max_contiguous_span_within_2mm":float(best),"median_tangent_alignment_within_2mm":float(np.median(al[within])) if np.any(within) else 0.}
def _orient_forward(forward,lad):
    f=np.asarray(forward,float);tree=cKDTree(lad);return f if float(tree.query(f[0])[0])<=float(tree.query(f[-1])[0]) else f[::-1].copy()
def _tail_tangent(path,from_end=True,span=4.):
    p=np.asarray(path,float);a=_arc(p)
    return _unit(_interp(p,[max(0.,float(a[-1])-span)])[0]-p[-1]) if from_end else _unit(_interp(p,[min(float(a[-1]),span)])[0]-p[0])
def _evaluate_trace(geom,src,path,forward,lad_endpoint):
    p,q,qc=_dense_qc(geom,src,path);tr=_truncate_qc(qc);last=int(tr["accepted_last_index"]);acc=p[:last+1];d=np.linalg.norm(acc-lad_endpoint[None,:],axis=1);k=int(np.argmin(d));reach=float(d[k]);reach_arc=float(q[k]);prefix=acc[:k+1];qcp=qc.iloc[:k+1];frac=float(qcp.plane_pass.mean()) if len(qcp) else 0.;ov=_overlap_metrics(prefix,forward[::-1].copy());confirmed=bool(reach<=MAX_REACH_ENDPOINT_MM and reach_arc>=MIN_REVERSE_QC_ARC_MM and frac>=MIN_PLANE_PASS_FRACTION and ov["fraction_within_2mm"]>=MIN_FORWARD_OVERLAP_FRACTION and ov["median_distance_mm"]<=MAX_FORWARD_MEDIAN_DISTANCE_MM and ov["p90_distance_mm"]<=MAX_FORWARD_P90_DISTANCE_MM and ov["median_tangent_alignment_within_2mm"]>=MIN_FORWARD_TANGENT_ALIGNMENT)
    return p,q,qc,tr,prefix,{"reach_frozen_endpoint_distance_mm":reach,"reach_arc_mm":reach_arc,"prefix_plane_pass_fraction":frac,**{f"forward_{kk}":v for kk,v in ov.items()},"confirmation_gate_pass":confirmed}
def _plot_geometry(out,lad,forward,reverse):
    fig=plt.figure(figsize=(16,5));pairs=[(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]
    for i,(a,b,xl,yl) in enumerate(pairs,1):
        ax=fig.add_subplot(1,3,i);ax.plot(lad[:,a],lad[:,b],label="frozen LAD");ax.plot(forward[:,a],forward[:,b],label="forward candidate");ax.plot(reverse[:,a],reverse[:,b],label="best reverse trace");ax.set_xlabel(xl);ax.set_ylabel(yl);ax.axis("equal");ax.legend() if i==1 else None
    fig.suptitle("Distal LAD bidirectional source-CCTA validation");fig.tight_layout();fig.savefig(out/"01_bidirectional_geometry.png",dpi=170);plt.close(fig)
def _plot_qc(out,geom,src,path):
    p,q=_resample(path,PLANE_STEP_MM);tt=_safe_tangents(p);idx=np.linspace(0,max(0,len(p)-1),12).round().astype(int);fig,axs=plt.subplots(3,4,figsize=(16,11))
    for ax,j in zip(axs.ravel(),idx):im,g=_plane(geom,src,p[j],tt[j]);ax.imshow(im,cmap="gray",vmin=-200,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]);ax.scatter([0],[0],s=9);ax.set_title(f"{q[j]:.1f} mm");ax.axis("off")
    fig.suptitle("Best reverse trace: orthogonal source-CCTA QC");fig.tight_layout();fig.savefig(out/"02_best_reverse_orthogonal_qc.png",dpi=170);plt.close(fig)
def synthetic_self_test():
    x=np.linspace(0,15,61);p=np.column_stack([x,np.zeros_like(x),np.zeros_like(x)]);ov=_overlap_metrics(p[::-1],p[::-1]);assert ov["fraction_within_2mm"]==1. and ov["median_tangent_alignment_within_2mm"]>.99;assert np.allclose(_tail_tangent(p,True),[-1,0,0]);return {"ok":True,"overlap":ov}
def _finalize(out,summary):
    report=out/"OPENPLAQUE_LAD_DISTAL_BIDIRECTIONAL_VALIDATION_REPORT.html";b=summary.get("best_reverse",{});report.write_text(f"<html><body><h1>OpenPlaque LAD Distal Bidirectional Validation v1</h1><p><b>Status:</b> {summary.get('status')}</p><p>Master: {summary.get('master_status')}; modified: {summary.get('master_modified')}</p><p>Known LAD reverse control: {summary.get('control',{}).get('pass')}</p><p>Confirming reverse hypotheses: {summary.get('confirming_reverse_hypothesis_count')}</p><p>Best reverse reach distance: {b.get('reach_frozen_endpoint_distance_mm')} mm; reach arc: {b.get('reach_arc_mm')} mm; overlap within 2 mm: {b.get('forward_fraction_within_2mm')}.</p><p>Research validation only; frozen anatomy unchanged.</p></body></html>",encoding="utf-8");z=out/"OPENPLAQUE_LAD_DISTAL_BIDIRECTIONAL_VALIDATION_RESULTS.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zz:
        for p in out.iterdir():
            if p.is_file() and p!=z:zz.write(p,p.name)
    return report,z

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root);out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME;out.mkdir(parents=True,exist_ok=True);_write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE});master=_read_json(root/MASTER);prior=_read_json(root/PRIOR_SUMMARY);allowed={"LAD_DISTAL_ENDPOINT_EXTENSION_INDEPENDENTLY_REPRODUCED","LAD_DISTAL_ENDPOINT_EXTENSION_SOURCE_SUPPORTED_DIFFERENT_TRAJECTORY"}
    if prior.get("status") not in allowed or not prior.get("control",{}).get("pass") or not prior.get("best_target",{}).get("extension_gate_pass"):
        summary={"status":STATUS_PREREQ,"master_status":master.get("status"),"master_modified":False,"prior_status":prior.get("status")};_write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","final_status":summary["status"]});_finalize(out,summary);return {"summary":summary}
    lad=_load_path(root/LAD);forward=_orient_forward(_load_path(root/FORWARD_PATH),lad);distal=forward[0].copy();geom,src=_source(root/SOURCE_CACHE);ves=LocalVesselness(geom,src,np.vstack([lad,forward]));terminal,_=_resample(lad,.25);tv=ves.sample(geom,terminal);vnorm=float(np.median(tv));vthr=float(np.percentile(tv,20))*MIN_VESSELNESS_FACTOR;calibration={"terminal_lad_vesselness_p20":float(np.percentile(tv,20)),"terminal_lad_vesselness_median":vnorm,"vesselness_threshold":vthr,"vesselness_norm":vnorm,"local_vesselness_roi_shape_zyx":list(ves.shape)};_write_json(out/"vesselness_calibration.json",calibration)
    d0=np.linalg.norm(lad[0]-distal);d1=np.linalg.norm(lad[-1]-distal);lad_or=lad if d0<=d1 else lad[::-1].copy();control_tan=_unit(_interp(lad_or,[min(4.,_arc(lad_or)[-1])])[0]-lad_or[0]);cstart=lad_or[0];cdirs=_rank_initial_dirs(geom,src,ves,cstart,control_tan,vnorm,n=6);crows=[];bestc=None;cref=_interp(lad_or,np.arange(0,min(CONTROL_SEARCH_MM,_arc(lad_or)[-1])+1e-9,.25))
    for di,(psc,d) in enumerate(cdirs):
        finals,attr=_beam(geom,src,ves,cstart,d,vthr,vnorm,CONTROL_SEARCH_MM);attr["dir_index"]=di;attr.to_csv(out/f"control_attrition_d{di}.csv",index=False)
        for fi,st in enumerate(finals[:2]):
            raw=np.asarray(st[1],float);p,q,qc=_dense_qc(geom,src,raw);tr=_truncate_qc(qc);acc=p[:int(tr["accepted_last_index"])+1];ov=_overlap_metrics(acc,cref);row={"id":f"control_d{di}_f{fi}","preview_score":psc,"beam_score":float(st[0]),**tr,**{f"lad_{k}":v for k,v in ov.items()}};row["control_gate_pass"]=bool(tr["accepted_arc_mm"]>=MIN_CONTROL_ARC_MM and tr["accepted_plane_pass_fraction"]>=MIN_PLANE_PASS_FRACTION and ov["fraction_within_2mm"]>=MIN_CONTROL_OVERLAP_FRACTION and ov["median_distance_mm"]<=MAX_CONTROL_MEDIAN_DISTANCE_MM);crows.append(row)
            if bestc is None or (row["control_gate_pass"],row["accepted_arc_mm"],row["beam_score"])>(bestc[0]["control_gate_pass"],bestc[0]["accepted_arc_mm"],bestc[0]["beam_score"]):bestc=(row,acc,qc)
    pd.DataFrame(crows).to_csv(out/"known_lad_reverse_control_candidates.csv",index=False);control_pass=bool(bestc and bestc[0]["control_gate_pass"]);_write_json(out/"known_lad_reverse_control.json",{"pass":control_pass,"best":bestc[0] if bestc else None})
    if not control_pass:
        summary={"status":STATUS_CONTROL,"master_status":master.get("status"),"master_modified":False,"control":{"pass":False,"best":bestc[0] if bestc else None},"calibration":calibration};_write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","final_status":summary["status"]});report,z=_finalize(out,summary);return {"summary":summary,"report":str(report),"zip":str(z)}
    reverse_tan=_tail_tangent(forward,True,4.);rstart=forward[-1].copy();rdirs=_rank_initial_dirs(geom,src,ves,rstart,reverse_tan,vnorm,n=8);rows=[];paths={};qcs={}
    for di,(psc,d) in enumerate(rdirs):
        finals,attr=_beam(geom,src,ves,rstart,d,vthr,vnorm,REVERSE_SEARCH_MM);attr["dir_index"]=di;attr.to_csv(out/f"reverse_attrition_d{di}.csv",index=False)
        for fi,st in enumerate(finals[:2]):
            raw=np.asarray(st[1],float);p,q,qc,tr,prefix,metrics=_evaluate_trace(geom,src,raw,forward,distal);hid=f"reverse_d{di}_f{fi}";row={"hypothesis_id":hid,"preview_score":psc,"beam_score":float(st[0]),"path_arc_mm":float(q[-1]) if len(q) else 0.,**tr,**metrics};rows.append(row);paths[hid]=prefix;qcs[hid]=qc
    rdf=pd.DataFrame(rows);rdf.to_csv(out/"reverse_hypotheses.csv",index=False);confirm=rdf[rdf.confirmation_gate_pass==True].copy() if len(rdf) else pd.DataFrame();nconfirm=int(len(confirm))
    if nconfirm:bestrow=confirm.sort_values(["reach_arc_mm","prefix_plane_pass_fraction","beam_score"],ascending=[False,False,False]).iloc[0].to_dict();bestid=bestrow["hypothesis_id"]
    elif len(rdf):bestrow=rdf.sort_values(["reach_frozen_endpoint_distance_mm","accepted_arc_mm","beam_score"],ascending=[True,False,False]).iloc[0].to_dict();bestid=bestrow["hypothesis_id"]
    else:bestrow={};bestid=None
    if bestid:_save_path(out/"best_reverse_trace.csv",paths[bestid]);qcs[bestid].to_csv(out/"best_reverse_trace_dense_qc.csv",index=False);_plot_geometry(out,lad,forward,paths[bestid]);_plot_qc(out,geom,src,paths[bestid])
    status=STATUS_CONFIRMED if nconfirm>=MIN_CONFIRMING_HYPOTHESES else STATUS_NONE;summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"prior_status":prior.get("status"),"forward_candidate_length_mm":float(_arc(forward)[-1]),"calibration":calibration,"control":{"pass":True,"best":bestc[0]},"reverse_hypothesis_count":int(len(rdf)),"confirming_reverse_hypothesis_count":nconfirm,"required_confirming_hypotheses":MIN_CONFIRMING_HYPOTHESES,"best_reverse":bestrow,"scientific_boundary":"Bidirectional source-CCTA validation only. Prior forward candidate defines reverse start and tangent but is not used in reverse discovery scoring. Frozen anatomy and clinical labels remain unchanged."};_write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","final_status":status,"algorithm":ALGORITHM});report,z=_finalize(out,summary);return {"summary":summary,"report":str(report),"zip":str(z)}
