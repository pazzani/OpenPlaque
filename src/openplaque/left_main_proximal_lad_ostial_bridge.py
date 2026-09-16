from __future__ import annotations

"""Source-CCTA bridge from the independently accepted proximal LAD endpoint to the aortic root.

The RCA proximal endpoint is traced to the aorta with the same machinery as a positive control.
The left search uses only the accepted LAD endpoint closest to the aortic surface, its reverse
local tangent, source HU, multiscale vesselness, coronary-mask proximity, and monotonically
improving distance to the aorta. C6/LCX geometry is deliberately excluded from the search.

Research use only. A positive result nominates a left-coronary ostial/proximal-trunk candidate
for visual QC; it does not establish clinical LM identity and does not modify the frozen master.
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
ALGORITHM="left-main-proximal-lad-ostial-bridge-v1.0-lowmem"
OUTPUT_DIRNAME="Left_Main_Proximal_LAD_Ostial_Bridge_v1"
SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH=Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH=Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS=Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR=TS/"coronary_arteries/coronary_arteries.nii.gz"
LEG=TS/"coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA=TS/"heartchambers_highres/aorta.nii.gz"

STATUS_RCA_FAIL="PROXIMAL_LAD_OSTIAL_RCA_CONTROL_FAILED"
STATUS_NO_BRIDGE="PROXIMAL_LAD_TO_AORTA_BRIDGE_NOT_ESTABLISHED"
STATUS_POS="LEFT_CORONARY_OSTIAL_PROXIMAL_TRUNK_REQUIRES_VISUAL_QC"


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

def _resample(p,step=.20):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0:return p.copy(),a
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6:q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)]),q

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-9 else np.zeros_like(v)

def _resample_mask(path,ref):
    im=sitk.ReadImage(str(_req(path)))
    same=(im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection()))
    if not same: im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im)>0

def _surface_tree(mask,ref,stride=2):
    surf=mask&~ndi.binary_erosion(mask,iterations=1,border_value=0); z=np.argwhere(surf)[::stride]; pts=_zyx_to_xyz(ref,z.astype(float)); return cKDTree(pts.astype(np.float32)),pts

def _orient_endpoint_nearest_tree(path,tree):
    p=np.asarray(path,float); d0=float(tree.query(p[0])[0]); d1=float(tree.query(p[-1])[0])
    return (p.copy(),d0,0) if d0<=d1 else (p[::-1].copy(),d1,1)

def _frangi_3d(vol,spacing,scales=(.55,.80,1.10,1.45)):
    x=np.clip(np.asarray(vol,np.float32),80,1000); x=(x-80)/920.; spacing=np.asarray(spacing,float); best=np.zeros_like(x,np.float32)
    for sm in scales:
        sig=np.maximum(sm/spacing,.55); n=sm*sm
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0),mode="nearest")*n/(spacing[0]**2); hyy=ndi.gaussian_filter(x,sig,order=(0,2,0),mode="nearest")*n/(spacing[1]**2); hxx=ndi.gaussian_filter(x,sig,order=(0,0,2),mode="nearest")*n/(spacing[2]**2)
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0),mode="nearest")*n/(spacing[0]*spacing[1]); hzx=ndi.gaussian_filter(x,sig,order=(1,0,1),mode="nearest")*n/(spacing[0]*spacing[2]); hyx=ndi.gaussian_filter(x,sig,order=(0,1,1),mode="nearest")*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),np.float32); H[...,0,0]=hzz; H[...,1,1]=hyy; H[...,2,2]=hxx; H[...,0,1]=H[...,1,0]=hzy; H[...,0,2]=H[...,2,0]=hzx; H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1); l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); s=np.sqrt(l1*l1+l2*l2+l3*l3); nz=s[s>0]; c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        v=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(s*s)/(2*c*c))); v[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(v).astype(np.float32))
    return best

def _orth_basis(t):
    t=_unit(t); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; u=_unit(np.cross(t,seed)); v=_unit(np.cross(t,u)); return u,v

def _cone_dirs(t):
    t=_unit(t); u,v=_orth_basis(t); out=[t]
    for deg in (10,20,30,40,50,60):
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,12,endpoint=False): out.append(_unit(np.cos(a)*t+np.sin(a)*(np.cos(phi)*u+np.sin(phi)*v)))
    return out

def _build_field(ref,src,spacing,cur,leg,start,goal,known):
    kp,_=_resample(known,.35); pts=np.vstack([start[None,:],goal[None,:],kp]); z=_xyz_to_zyx(ref,pts); margin=np.array([7.,7.,7.])
    lo=np.floor(np.min(z*spacing,axis=0)/spacing-margin/spacing).astype(int); hi=np.ceil(np.max(z*spacing,axis=0)/spacing+margin/spacing).astype(int)+1; lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(src.shape)); sl=tuple(slice(lo[k],hi[k]) for k in range(3))
    roi=np.asarray(src[sl]); cm=np.asarray(cur[sl]); lm=np.asarray(leg[sl]); union=cm|lm; du=ndi.distance_transform_edt(~union,sampling=spacing); vessel=_frangi_3d(roi,spacing)
    kz=_xyz_to_zyx(ref,kp)-lo[None,:]; good=np.all((kz>=0)&(kz<np.asarray(roi.shape)[None,:]),axis=1); kv=map_coordinates(vessel,kz[good].T,order=1,mode="nearest") if np.any(good) else np.array([.03])
    thr=max(.003,min(.12,.30*float(np.percentile(kv,20)))); norm=max(float(np.median(kv))*1.5,.02)
    return {"lo":lo,"roi":roi,"cur":cm,"leg":lm,"du":du,"v":vessel,"thr":thr,"norm":norm}

def _sample_field(ref,F,p):
    z=_xyz_to_zyx(ref,[p])[0]-F["lo"]; shape=np.asarray(F["roi"].shape)
    if np.any(z<1) or np.any(z>shape-2): return None
    co=z[:,None]; hu=float(map_coordinates(F["roi"],co,order=1,mode="nearest")[0]); vv=float(map_coordinates(F["v"],co,order=1,mode="nearest")[0]); dd=float(map_coordinates(F["du"],co,order=1,mode="nearest")[0]); zi=np.rint(z).astype(int); c=bool(F["cur"][tuple(zi)]); l=bool(F["leg"][tuple(zi)])
    return hu,vv,dd,c,l

def _beam_to_aorta(ref,F,aorta_tree,start,tangent,target_max_mm=18.,step=.40,beam_width=100):
    origin=np.asarray(start,float); d0=float(aorta_tree.query(origin)[0]); states=[(0.,[origin],_unit(tangent),d0)]; reached=[]; nsteps=int(math.ceil(target_max_mm/step))
    for _ in range(nsteps):
        nxt=[]
        for score,pts,t,prev_ad in states:
            for d in _cone_dirs(t):
                if np.dot(d,t)<math.cos(math.radians(65)): continue
                p=pts[-1]+step*d; ad=float(aorta_tree.query(p)[0])
                if ad>prev_ad+.10: continue
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-p,axis=1))<.60*step: continue
                s=_sample_field(ref,F,p)
                if s is None: continue
                hu,vv,du,c,l=s; near=ad<=1.3
                if not (80<=hu<=1400 and vv>=.42*F["thr"] and (du<=1.6 or near)): continue
                vn=min(1.,vv/F["norm"]); support=1. if c and l else (.60 if c or l else .15 if du<=1.6 else 0.); improve=max(-.25,prev_ad-ad); align=max(0.,float(np.dot(d,t)))
                ns=score+1.7*vn+.45*support+.45*align+.9*improve/max(step,1e-6); nt=_unit(.70*t+.30*d); st=(ns,pts+[p],nt,ad); nxt.append(st)
                if ad<=.75: reached.append(st)
        if not nxt: break
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.30).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep)>=beam_width: break
        states=keep
        if len(reached)>=12: break
    return reached,states

def _path_metrics(ref,src,cur,leg,F,path,start,aorta_tree):
    p,q=_resample(path,.20); h=_sample(ref,src,p); z=_xyz_to_zyx(ref,p); local=z-F["lo"][None,:]; vv=map_coordinates(F["v"],local.T,order=1,mode="nearest"); zr=np.rint(z).astype(int); zr=np.clip(zr,[0,0,0],np.asarray(src.shape)-1); cs=cur[tuple(zr.T)]; ls=leg[tuple(zr.T)]; ad=np.asarray(aorta_tree.query(p)[0],float); length=float(q[-1]); disp=float(np.linalg.norm(p[-1]-start)); tr=np.array([])
    if len(p)>=4:
        v=np.diff(p,axis=0); v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9); tr=np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))
    return {"length_mm":length,"endpoint_displacement_mm":disp,"tortuosity":length/max(disp,1e-6),"max_turn_deg":float(np.max(tr)) if len(tr) else 0.,"median_hu":float(np.median(h)),"robust_hu_fraction":float(np.mean((h>=120)&(h<=1200))),"median_vesselness":float(np.median(vv)),"p10_vesselness":float(np.percentile(vv,10)),"current_support_fraction":float(np.mean(cs)),"legacy_support_fraction":float(np.mean(ls)),"union_support_fraction":float(np.mean(cs|ls)),"dual_support_fraction":float(np.mean(cs&ls)),"start_aorta_distance_mm":float(ad[0]),"endpoint_aorta_distance_mm":float(ad[-1]),"aorta_distance_reduction_mm":float(ad[0]-ad[-1]),"arc_profile_mm":q,"aorta_distance_profile_mm":ad}

def _trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,path,max_mm,gate_length_max,control_known=False):
    p,prox_d,flipped=_orient_endpoint_nearest_tree(path,aorta_tree); pr,pq=_resample(p,.20); start=pr[0]; distal=_resample(p,.20)[0]; ix=min(len(pr)-1,max(3,int(round(2.0/.20)))); outward=_unit(start-pr[ix]); goal=np.asarray(aorta_tree.data[int(aorta_tree.query(start)[1])],float); known=pr[pq<=min(4.,pq[-1])]; F=_build_field(ref,src,spacing,cur,leg,start,goal,known); reached,frontier=_beam_to_aorta(ref,F,aorta_tree,start,outward,target_max_mm=max_mm,step=.40,beam_width=110)
    rows=[]
    for st in reached:
        pp=np.asarray(st[1]); m=_path_metrics(ref,src,cur,leg,F,pp,start,aorta_tree); m["beam_score"]=float(st[0]); m["gate_pass"]=bool(m["endpoint_aorta_distance_mm"]<=.80 and 1.0<=m["length_mm"]<=gate_length_max and m["endpoint_displacement_mm"]>=.8 and m["tortuosity"]<=1.8 and m["max_turn_deg"]<=65 and m["robust_hu_fraction"]>=.88 and m["union_support_fraction"]>=.60 and m["p10_vesselness"]>=.42*F["thr"] and m["aorta_distance_reduction_mm"]>=.8); rows.append((m,pp))
    meta={"proximal_endpoint_aorta_distance_mm":prox_d,"path_was_reversed":bool(flipped),"start_lps_mm":start.tolist(),"outward_tangent":outward.tolist(),"frontier_count":len(frontier),"vesselness_threshold":F["thr"]}
    if not rows: return None,{**meta,"accepted":False,"reason":"no_path_reached_aorta"},F,p
    ok=[x for x in rows if x[0]["gate_pass"]]
    if not ok:
        best=max(rows,key=lambda x:x[0]["beam_score"])[0]; return None,{**meta,"accepted":False,"reason":"no_candidate_passed","best":{k:v for k,v in best.items() if not isinstance(v,np.ndarray)}},F,p
    ok.sort(key=lambda x:(x[0]["robust_hu_fraction"],x[0]["union_support_fraction"],-x[0]["tortuosity"],x[0]["beam_score"]),reverse=True); m,pp=ok[0]
    return pp,{**meta,"accepted":True,"best":{k:v for k,v in m.items() if not isinstance(v,np.ndarray)}},F,p

def _plane(ref,src,c,t,half=7.,step=.2):
    t=_unit(t); u,v=_orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij"); P=c+xx[...,None]*u+yy[...,None]*v; return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q

def synthetic_endpoint_orientation_self_test():
    pts=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]); tree=cKDTree(np.array([[-.5,0,0],[10,0,0]],float)); p,d,f=_orient_endpoint_nearest_tree(pts[::-1],tree); assert np.allclose(p[0],[0,0,0]); assert f==1; assert d<1.; return {"ok":True,"nearest_distance_mm":float(d)}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    required=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/LAD_PATH,root/RCA_PATH,root/CUR,root/LEG,root/AORTA]
    for p in required:_req(p)
    master=json.loads((root/MASTER).read_text()); ref,src,spacing=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref); cur=_resample_mask(root/CUR,ref); leg=_resample_mask(root/LEG,ref); aorta=_resample_mask(root/AORTA,ref); aorta_tree,aorta_pts=_surface_tree(aorta,ref)
    print("Running RCA proximal-endpoint to aorta control..."); rpath,rca_summary,rF,rca_o=_trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,rca,max_mm=12.,gate_length_max=10.); _write_json(out/"RCA_proximal_ostial_control.json",rca_summary)
    print("Searching accepted proximal LAD endpoint to aorta..."); lpath,lad_summary,lF,lad_o=_trace_endpoint(ref,src,spacing,cur,leg,aorta_tree,lad,max_mm=20.,gate_length_max=18.); _write_json(out/"proximal_LAD_ostial_bridge_summary.json",lad_summary)
    if lpath is not None: pd.DataFrame(lpath,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"proximal_LAD_to_aorta_candidate.csv",index=False)
    if rpath is not None:
        rm=_path_metrics(ref,src,cur,leg,rF,rpath,rpath[0],aorta_tree); plt.figure(figsize=(6.5,4)); plt.plot(rm["arc_profile_mm"],rm["aorta_distance_profile_mm"],marker="."); plt.xlabel("RCA bridge arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title(f"RCA proximal→aorta control | accepted={rca_summary.get('accepted',False)}"); plt.tight_layout(); plt.savefig(out/"01_RCA_proximal_ostial_control.png",dpi=180); plt.close()
    if lpath is not None:
        lm=_path_metrics(ref,src,cur,leg,lF,lpath,lpath[0],aorta_tree); plt.figure(figsize=(6.5,4)); plt.plot(lm["arc_profile_mm"],lm["aorta_distance_profile_mm"],marker="."); plt.xlabel("left bridge arc (mm)"); plt.ylabel("distance to aorta (mm)"); plt.title("Accepted proximal LAD endpoint→aorta"); plt.tight_layout(); plt.savefig(out/"02_left_ostial_aorta_distance.png",dpi=180); plt.close()
        fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d"); lp,_=_resample(lad_o,.3); ax.plot(lp[:,0],lp[:,1],lp[:,2],lw=2,label="accepted LAD"); ax.plot(lpath[:,0],lpath[:,1],lpath[:,2],lw=3,label="ostial/proximal candidate"); near=aorta_pts[np.linalg.norm(aorta_pts-lpath[-1],axis=1)<=8]; ax.scatter(near[:,0],near[:,1],near[:,2],s=3,alpha=.15,label="local aortic surface"); ax.legend(); ax.set_title("Accepted LAD proximal endpoint to aortic root"); plt.tight_layout(); plt.savefig(out/"03_left_ostial_geometry.png",dpi=180); plt.close()
        fig,axes=plt.subplots(1,4,figsize=(14,4)); pp,qq=_resample(lpath,.20); tt=np.gradient(pp,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9); picks=np.linspace(0,len(pp)-1,4).astype(int)
        for ax,ix in zip(axes,picks): im,qv=_plane(ref,src,pp[ix],tt[ix]); ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=18); ax.set_title(f"bridge +{qq[ix]:.1f} mm")
        plt.tight_layout(); plt.savefig(out/"04_left_ostial_orthogonal_source_qc.png",dpi=180); plt.close()
    control_pass=bool(rca_summary.get("accepted",False)); left_pass=bool(lad_summary.get("accepted",False)); status=STATUS_RCA_FAIL if not control_pass else (STATUS_POS if left_pass else STATUS_NO_BRIDGE)
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"RCA_proximal_ostial_control":rca_summary,"proximal_LAD_to_aorta":lad_summary,"C6_or_LCX_geometry_used_in_search":False,"scientific_boundary":"A positive result nominates a source-supported left-coronary ostial/proximal-trunk bridge from the accepted LAD endpoint to the aorta. It does not yet establish LM or prove where the LCX-like C6 joins."}; _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    report=out/"OPENPLAQUE_PROXIMAL_LAD_OSTIAL_BRIDGE_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque Proximal LAD Ostial Bridge</h1><p><b>Status:</b> {status}</p><p>RCA control accepted: {control_pass}.</p><p>Left bridge accepted: {left_pass}.</p><p>Clinical LM remains unresolved; master unchanged.</p></body></html>",encoding="utf-8"); zpath=out/"OPENPLAQUE_PROXIMAL_LAD_OSTIAL_BRIDGE_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
