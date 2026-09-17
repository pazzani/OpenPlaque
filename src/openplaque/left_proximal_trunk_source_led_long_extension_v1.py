from __future__ import annotations

"""Long source-led continuation from the dense-QC positive local multidirection endpoint.

Prospective experiment. The prior local multidirection experiment established a 10-mm continuation
with dense plane QC while every accepted local hypothesis moved farther from the nearest aortic
surface. This experiment therefore removes aortic geometry from discovery entirely and asks a
narrower question: does that source-supported vessel continue for a substantially longer arc, and
what does its aortic-distance trajectory do only after the source-led path has been recovered?

The frozen master is never modified. A positive result is a source-supported continuation only;
it is not labeled LM and does not establish LCX topology.
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
ALGORITHM="left-proximal-trunk-source-led-long-extension-v1.0"
OUTPUT_DIRNAME="Left_Proximal_Trunk_Source_Led_Long_Extension_v1"
PRIOR_DIR="Left_Proximal_Trunk_Local_Multidirection_v1"
CONT_DIR="Left_Proximal_Trunk_Continuation_QC_v1"

SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PATH=Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS=Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR=TS/"coronary_arteries/coronary_arteries.nii.gz"
LEG=TS/"coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA=TS/"heartchambers_highres/aorta.nii.gz"
PRIOR_SUMMARY=Path(PRIOR_DIR)/"summary.json"
PRIOR_PATH=Path(PRIOR_DIR)/"best_local_multidirection_path.csv"
CONT_PATH=Path(CONT_DIR)/"accepted_proximal_trunk_continuation_candidate.csv"

FIELD_MARGIN_MM=28.0
STEP_MM=.40
SEARCH_MAX_MM=25.0
BEAM_WIDTH=100
TOP_INITIAL_DIRS=6
PREVIEW_MM=1.6
PLANE_STEP_MM=.40
SUSTAINED_FAIL_N=3
MIN_ACCEPTED_EXTENSION_MM=5.0
MIN_PLANE_PASS_FRACTION=.80
AORTA_REACH_MM=.90
MEANINGFUL_AORTA_RETURN_MM=2.0
KNOWN_PATH_EXCLUSION_MM=.70

STATUS_RCA_FAIL="SOURCE_LED_LONG_EXTENSION_RCA_CONTROL_FAILED"
STATUS_REACH="SOURCE_LED_LONG_EXTENSION_REACHES_AORTA_REQUIRES_VISUAL_QC"
STATUS_RETURN="SOURCE_LED_LONG_EXTENSION_QC_POSITIVE_EVENTUALLY_TURNS_TOWARD_AORTA"
STATUS_AWAY="SOURCE_LED_LONG_EXTENSION_QC_POSITIVE_CONTINUES_AWAY_FROM_AORTA"
STATUS_NEG="SOURCE_LED_LONG_EXTENSION_NO_VALID_CONTINUATION"

def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _write_json(p,o): Path(p).write_text(json.dumps(o,indent=2,default=str,allow_nan=True),encoding="utf-8")

def _source(cache):
    a=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r")
    m=_load_json(_req(cache/"series7_int16.json"))
    ref=sitk.GetImageFromArray(np.asarray(a)); sp=np.asarray(m["spacing_zyx"],float)
    ref.SetSpacing(tuple(sp[::-1])); ref.SetOrigin(tuple(np.asarray(m["positions_lps_mm"][0],float)))
    iop=np.asarray(m["image_orientation_patient"],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
    D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
    ref.SetDirection(tuple(D.ravel())); return ref,a,sp

def _xyz_to_zyx(img,p):
    p=np.atleast_2d(np.asarray(p,float)); o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing())
    D=np.asarray(img.GetDirection()).reshape(3,3)
    return (((p-o)@np.linalg.inv(D).T)/sp)[:,::-1]

def _zyx_to_xyz(img,p):
    p=np.atleast_2d(np.asarray(p,float)); q=p[:,::-1]; o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing())
    D=np.asarray(img.GetDirection()).reshape(3,3); return o+(q*sp)@D.T

def _sample(img,a,p,cval=-1024.):
    return map_coordinates(np.asarray(a),_xyz_to_zyx(img,p).T,order=1,mode="constant",cval=cval)

def _load_path(path,ref):
    d=pd.read_csv(_req(path))
    for c in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(x in d.columns for x in c): return d[list(c)].to_numpy(float)
    for c in (("zyx_z","zyx_y","zyx_x"),("z","y","x")):
        if all(x in d.columns for x in c): return _zyx_to_xyz(ref,d[list(c)].to_numpy(float))
    raise ValueError(f"unrecognized coordinates: {path}")

def _arc(p):
    p=np.asarray(p,float); return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))

def _resample(p,step=.2):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0:return p.copy(),a
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6:q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)]),q

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-9 else np.zeros_like(v)

def _orth_basis(t):
    t=_unit(t); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]
    u=_unit(np.cross(t,seed)); v=_unit(np.cross(t,u)); return u,v

def _cone_dirs(t):
    t=_unit(t); u,v=_orth_basis(t); out=[t]
    for deg in (10,20,30,40,50,60):
        a=np.deg2rad(deg)
        for phi in np.linspace(0,2*np.pi,12,endpoint=False):
            out.append(_unit(np.cos(a)*t+np.sin(a)*(np.cos(phi)*u+np.sin(phi)*v)))
    return out

def _resample_mask(path,ref):
    im=sitk.ReadImage(str(_req(path)))
    same=(im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection()))
    if not same: im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im)>0

def _surface_tree(mask,ref,stride=2):
    surf=mask&~ndi.binary_erosion(mask,iterations=1,border_value=0); z=np.argwhere(surf)[::stride]
    pts=_zyx_to_xyz(ref,z.astype(float)); return cKDTree(pts.astype(np.float32)),pts

def _frangi_3d(vol,spacing,scales=(.55,.80,1.10,1.45)):
    x=np.clip(np.asarray(vol,np.float32),80,1000); x=(x-80)/920.; spacing=np.asarray(spacing,float); best=np.zeros_like(x,np.float32)
    for si,sm in enumerate(scales,1):
        print(f"[source-led-long] vesselness scale {si}/{len(scales)} = {sm:.2f} mm")
        sig=np.maximum(sm/spacing,.55); n=sm*sm
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0),mode="nearest")*n/(spacing[0]**2)
        hyy=ndi.gaussian_filter(x,sig,order=(0,2,0),mode="nearest")*n/(spacing[1]**2)
        hxx=ndi.gaussian_filter(x,sig,order=(0,0,2),mode="nearest")*n/(spacing[2]**2)
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0),mode="nearest")*n/(spacing[0]*spacing[1])
        hzx=ndi.gaussian_filter(x,sig,order=(1,0,1),mode="nearest")*n/(spacing[0]*spacing[2])
        hyx=ndi.gaussian_filter(x,sig,order=(0,1,1),mode="nearest")*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),np.float32); H[...,0,0]=hzz;H[...,1,1]=hyy;H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy;H[...,0,2]=H[...,2,0]=hzx;H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1)
        l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps); ss=np.sqrt(l1*l1+l2*l2+l3*l3)
        nz=ss[ss>0]; c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        vv=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(ss*ss)/(2*c*c))); vv[(l2>=0)|(l3>=0)]=0
        best=np.maximum(best,np.nan_to_num(vv).astype(np.float32))
    return best

def _field(ref,src,spacing,cur,leg,known_path,start,margin_mm=FIELD_MARGIN_MM):
    known,_=_resample(known_path,.35); recent=known[-min(30,len(known)):]
    pts=np.vstack([recent,start[None,:]]); z=_xyz_to_zyx(ref,pts); mv=np.asarray([margin_mm]*3)/spacing
    lo=np.maximum(np.floor(z.min(0)-mv).astype(int),0); hi=np.minimum(np.ceil(z.max(0)+mv).astype(int)+1,np.asarray(src.shape))
    sl=tuple(slice(lo[k],hi[k]) for k in range(3)); roi=np.asarray(src[sl]); cm=np.asarray(cur[sl]); lm=np.asarray(leg[sl]); union=cm|lm
    du=ndi.distance_transform_edt(~union,sampling=spacing); vessel=_frangi_3d(roi,spacing)
    kz=_xyz_to_zyx(ref,recent)-lo[None,:]; good=np.all((kz>=0)&(kz<np.asarray(roi.shape)[None,:]),axis=1)
    kv=map_coordinates(vessel,kz[good].T,order=1,mode="nearest")
    thr=max(.003,min(.12,.30*float(np.percentile(kv,20)))); norm=max(float(np.median(kv))*1.5,.02)
    return {"lo":lo,"roi":roi,"cur":cm,"leg":lm,"du":du,"v":vessel,"thr":thr,"norm":norm,"shape":list(roi.shape)}

def _sample_field(ref,F,p):
    z=_xyz_to_zyx(ref,[p])[0]-F["lo"]; shape=np.asarray(F["roi"].shape)
    if np.any(z<1) or np.any(z>shape-2):return None
    co=z[:,None]; hu=float(map_coordinates(F["roi"],co,order=1,mode="nearest")[0]); vv=float(map_coordinates(F["v"],co,order=1,mode="nearest")[0]); dd=float(map_coordinates(F["du"],co,order=1,mode="nearest")[0])
    zi=np.rint(z).astype(int); c=bool(F["cur"][tuple(zi)]); l=bool(F["leg"][tuple(zi)]); return hu,vv,dd,c,l

def _preview(ref,F,start,d):
    vals=[]
    for s in np.arange(.4,PREVIEW_MM+1e-9,.4):
        x=_sample_field(ref,F,start+s*d)
        if x is None:return -1e9
        hu,vv,du,c,l=x
        if not (80<=hu<=1400) or vv<.42*F["thr"] or du>1.6:return -1e9
        vals.append(1.8*min(1.,vv/F["norm"])+.25*min(1.,max(0.,hu-80)/500.)+.25*(1 if c or l else 0))
    return float(np.mean(vals)) if vals else -1e9

def _initial_dirs(ref,F,start,tangent):
    scored=[]
    for d in _cone_dirs(tangent):
        if np.dot(d,tangent)<math.cos(math.radians(65)):continue
        sc=_preview(ref,F,start,d)
        if np.isfinite(sc) and sc>-1e8:scored.append((sc,d))
    scored.sort(key=lambda x:x[0],reverse=True); chosen=[]
    for sc,d in scored:
        if all(abs(float(np.dot(d,e)))<.985 for _,e in chosen):chosen.append((sc,d))
        if len(chosen)>=TOP_INITIAL_DIRS:break
    return chosen

def _beam(ref,F,start,tangent,known_tree,max_mm=SEARCH_MAX_MM,step=STEP_MM,beam_width=BEAM_WIDTH):
    states=[(0.,[np.asarray(start,float)],_unit(tangent))]; best=states[0]; rows=[]; checkpoints=[]
    nsteps=int(math.ceil(max_mm/step))
    for si in range(nsteps):
        nxt=[]; cnt={"step":si,"arc_budget_mm":float((si+1)*step),"states_in":len(states),"proposals":0,"reject_turn":0,"reject_loop":0,"reject_known_path":0,"reject_outside":0,"reject_hu":0,"reject_vesselness":0,"reject_mask":0,"accepted":0,"kept":0}
        for score,pts,t in states:
            for d in _cone_dirs(t):
                cnt["proposals"]+=1
                if np.dot(d,t)<math.cos(math.radians(65)):cnt["reject_turn"]+=1;continue
                q=pts[-1]+step*d
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-q,axis=1))<.60*step:cnt["reject_loop"]+=1;continue
                if len(pts)>4 and float(known_tree.query(q)[0])<KNOWN_PATH_EXCLUSION_MM:cnt["reject_known_path"]+=1;continue
                s=_sample_field(ref,F,q)
                if s is None:cnt["reject_outside"]+=1;continue
                hu,vv,du,c,l=s
                if not (80<=hu<=1400):cnt["reject_hu"]+=1;continue
                if vv<.42*F["thr"]:cnt["reject_vesselness"]+=1;continue
                if du>1.6:cnt["reject_mask"]+=1;continue
                vn=min(1.,vv/F["norm"]); support=1. if c and l else (.60 if c or l else .15); align=max(0.,float(np.dot(d,t)))
                ns=score+1.7*vn+.45*support+.45*align; nt=_unit(.70*t+.30*d); st=(ns,pts+[q],nt); nxt.append(st);cnt["accepted"]+=1
                if len(st[1])>len(best[1]) or (len(st[1])==len(best[1]) and ns>best[0]):best=st
        if not nxt:rows.append(cnt);break
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.30).astype(int))
            if key in bins:continue
            bins.add(key);keep.append(st)
            if len(keep)>=beam_width:break
        states=keep;cnt["kept"]=len(states);rows.append(cnt)
        if (si+1)%12==0 or si==nsteps-1:
            for st in states[:3]:checkpoints.append(st)
        if si%10==0:print(f"[source-led-long] step {si+1}/{nsteps} states={len(states)}")
    finals=[]
    for st in list(states)+checkpoints+[best]:
        ep=np.asarray(st[1][-1]);
        if all(np.linalg.norm(ep-np.asarray(x[1][-1]))>1.0 for x in finals):finals.append(st)
    finals.sort(key=lambda s:(len(s[1]),s[0]),reverse=True)
    return finals[:8],pd.DataFrame(rows)

def _plane(ref,src,c,t,half=5.5,step=.20):
    t=_unit(t);u,v=_orth_basis(t);q=np.arange(-half,half+1e-9,step);yy,xx=np.meshgrid(q,q,indexing="ij");P=c+xx[...,None]*u+yy[...,None]*v
    return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q

def _component_metrics(im,q):
    iy=ix=int(np.argmin(np.abs(q)));center=float(im[iy,ix]);yy,xx=np.meshgrid(q,q,indexing="ij");rr=np.sqrt(xx*xx+yy*yy);thr=max(220.,min(500.,.55*center));bw=(im>=thr)&(rr<=3.5);lab,_=ndi.label(bw,np.ones((3,3),int));labels=[]
    if lab[iy,ix]>0:labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt(q[pts[:,1]]**2+q[pts[:,0]]**2);j=int(np.argmin(dist))
            if dist[j]<=1.0:labels=[int(lab[tuple(pts[j])])]
    if not labels:return {"center_hu":center,"threshold_hu":thr,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
    mask=lab==labels[0];pts=np.argwhere(mask);xs=q[pts[:,1]];ys=q[pts[:,0]];cx=float(np.mean(xs));cy=float(np.mean(ys));off=float(np.hypot(cx,cy));area=float(len(pts)*.04);rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        C=np.cov(np.column_stack([xs,ys]).T);ev=np.linalg.eigvalsh(C);axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else:axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0);contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan
    return {"center_hu":center,"threshold_hu":thr,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}

def _pass(m):return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2 and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)

def _dense_qc(ref,src,path):
    p,q=_resample(path,PLANE_STEP_MM);tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m.update(index=i,arc_mm=float(a),plane_pass=_pass(m));rows.append(m)
    return p,q,pd.DataFrame(rows)

def _truncate(df):
    ps=df.plane_pass.astype(bool).to_numpy();cut=len(df)-1;first=None
    for i in range(len(ps)-SUSTAINED_FAIL_N+1):
        if not np.any(ps[i:i+SUSTAINED_FAIL_N]):first=i;cut=max(0,i-1);break
    d=df.iloc[:cut+1];arc=float(d.arc_mm.iloc[-1]) if len(d) else 0.;frac=float(d.plane_pass.mean()) if len(d) else 0.
    return {"first_sustained_failure_index":first,"accepted_last_index":int(cut),"accepted_arc_mm":arc,"accepted_plane_pass_fraction":frac,"accepted":bool(arc>=MIN_ACCEPTED_EXTENSION_MM and frac>=MIN_PLANE_PASS_FRACTION),"covers_full_path":bool(cut==len(df)-1)}

def _rca_control(ref,src,rca,aorta_tree):
    d0=float(aorta_tree.query(rca[0])[0]);d1=float(aorta_tree.query(rca[-1])[0]);p=rca if d0<=d1 else rca[::-1];rp,rq=_resample(p,.8);keep=(rq>=2)&(rq<=min(18,rq[-1]));rp=rp[keep];tt=np.gradient(rp,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t) in enumerate(zip(rp,tt)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m["plane_pass"]=_pass(m);m["index"]=i;rows.append(m)
    df=pd.DataFrame(rows);frac=float(df.plane_pass.mean()) if len(df) else 0.;return df,bool(frac>=.75),frac,p

def synthetic_long_extension_self_test():
    assert SEARCH_MAX_MM>=20.;assert FIELD_MARGIN_MM>SEARCH_MAX_MM
    d=pd.DataFrame({"arc_mm":np.arange(10)*.4,"plane_pass":[1,1,1,1,0,1,1,0,0,0]});r=_truncate(d);assert r["first_sustained_failure_index"]==7 and r["accepted_last_index"]==6
    return {"ok":True,"search_max_mm":SEARCH_MAX_MM,"field_margin_mm":FIELD_MARGIN_MM}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root);out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME;out.mkdir(parents=True,exist_ok=True);_write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    req=[root/PRIOR_SUMMARY,root/PRIOR_PATH,root/CONT_PATH,root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/RCA_PATH,root/CUR,root/LEG,root/AORTA]
    for p in req:_req(p)
    prior=_load_json(root/PRIOR_SUMMARY);master=_load_json(root/MASTER)
    if prior.get("status")!="PROXIMAL_TRUNK_LOCAL_MULTIDIRECTION_CONTINUATION_QC_POSITIVE":raise RuntimeError(f"unexpected prior status {prior.get('status')}")
    if int(prior.get("n_accepted_hypotheses",0))<1:raise RuntimeError("no accepted local hypothesis")
    ref,src,spacing=_source(root/SOURCE_CACHE);cur=_resample_mask(root/CUR,ref);leg=_resample_mask(root/LEG,ref);aorta=_resample_mask(root/AORTA,ref);aorta_tree,_=_surface_tree(aorta,ref)
    rca=_load_path(root/RCA_PATH,ref);rdf,rca_ok,rca_frac,rca_o=_rca_control(ref,src,rca,aorta_tree);rdf.to_csv(out/"RCA_plane_qc_control.csv",index=False)
    if not rca_ok:
        s={"status":STATUS_RCA_FAIL,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"RCA_plane_qc_control":{"plane_count":len(rdf),"pass_fraction":rca_frac,"accepted":False}};_write_json(out/"summary.json",s);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_RCA_FAIL});return _finalize(out,s)
    prior_path=_load_path(root/PRIOR_PATH,ref);cont=_load_path(root/CONT_PATH,ref)
    cont_end=cont[-1]
    if np.linalg.norm(prior_path[-1]-cont_end)<np.linalg.norm(prior_path[0]-cont_end):prior_path=prior_path[::-1].copy()
    start=prior_path[-1];pp,aa=_resample(prior_path,.2);tangent=_unit(pp[-1]-pp[max(0,len(pp)-9)])
    combined_known=np.vstack([cont,prior_path[1:]]);known_p,known_a=_resample(combined_known,.2);known_for_tree=known_p[:-6] if len(known_p)>8 else known_p[:1];known_tree=cKDTree(known_for_tree.astype(np.float32))
    F=_field(ref,src,spacing,cur,leg,combined_known,start);_write_json(out/"field_metadata.json",{"roi_shape":F["shape"],"margin_mm":FIELD_MARGIN_MM,"vesselness_threshold":float(F["thr"]),"normalizer":float(F["norm"])})
    dirs=_initial_dirs(ref,F,start,tangent);print("[source-led-long] initial directions:",len(dirs));all_final=[];attrs=[]
    for di,(preview,d) in enumerate(dirs):
        finals,attr=_beam(ref,F,start,d,known_tree);attr["dir_index"]=di;attr["preview_score"]=preview;attrs.append(attr)
        for fi,st in enumerate(finals):all_final.append((di,fi,preview,st))
    if attrs:pd.concat(attrs,ignore_index=True).to_csv(out/"search_attrition.csv",index=False)
    rows=[];qcdfs={};paths={}
    for di,fi,preview,st in all_final:
        path=np.asarray(st[1],float);p,q,qc=_dense_qc(ref,src,path);tr=_truncate(qc);last=tr["accepted_last_index"];accepted=p[:last+1] if last>=0 else p[:1];ad=np.asarray(aorta_tree.query(accepted)[0],float);start_ad=float(ad[0]);end_ad=float(ad[-1]);min_ad=float(np.min(ad));min_ix=int(np.argmin(ad));reduction=start_ad-min_ad
        hid=f"d{di}_f{fi}";qc["hypothesis_id"]=hid;qcdfs[hid]=qc;paths[hid]=path
        rows.append({"hypothesis_id":hid,"dir_index":di,"final_index":fi,"preview_score":preview,"beam_score":float(st[0]),"path_arc_mm":float(q[-1]),**tr,"start_aorta_distance_mm":start_ad,"accepted_endpoint_aorta_distance_mm":end_ad,"minimum_aorta_distance_mm":min_ad,"minimum_aorta_arc_mm":float(q[min_ix]) if min_ix<len(q) else np.nan,"maximum_aorta_reduction_mm":reduction})
    h=pd.DataFrame(rows);h.to_csv(out/"long_extension_hypotheses.csv",index=False)
    accepted=h[h.accepted==True].copy() if len(h) else h
    if len(accepted):accepted=accepted.sort_values(["accepted_arc_mm","accepted_plane_pass_fraction","beam_score"],ascending=[False,False,False]);bestrow=accepted.iloc[0]
    elif len(h):bestrow=h.sort_values(["accepted_arc_mm","beam_score"],ascending=[False,False]).iloc[0]
    else:raise RuntimeError("No source-led finalists produced")
    hid=str(bestrow.hypothesis_id);path=paths[hid];qc=qcdfs[hid];qc.to_csv(out/"best_long_dense_qc.csv",index=False);pd.DataFrame(path,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"best_long_extension_path.csv",index=False)
    p,q=_resample(path,.2);ad=np.asarray(aorta_tree.query(p)[0],float);pd.DataFrame({"arc_mm":q,"aorta_distance_mm":ad}).to_csv(out/"best_long_aorta_distance_profile.csv",index=False)
    accepted_positive=bool(bestrow.accepted);reaches=bool(accepted_positive and float(np.min(ad))<=AORTA_REACH_MM);returns=bool(accepted_positive and float(bestrow.maximum_aorta_reduction_mm)>=MEANINGFUL_AORTA_RETURN_MM)
    status=STATUS_REACH if reaches else STATUS_RETURN if returns else STATUS_AWAY if accepted_positive else STATUS_NEG
    plt.figure(figsize=(8,4.5));plt.plot(q,ad);plt.axhline(AORTA_REACH_MM,ls="--",lw=.8);plt.xlabel("source-led extension arc (mm)");plt.ylabel("distance to nearest aortic surface (mm)");plt.title("Post-hoc aortic distance along source-led continuation");plt.tight_layout();plt.savefig(out/"01_posthoc_aorta_distance.png",dpi=180);plt.close()
    bp,bq=_resample(path,.4);tt=np.gradient(bp,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);picks=np.linspace(0,len(bp)-1,min(12,len(bp))).astype(int);cols=4;nr=int(math.ceil(len(picks)/cols));fig,axes=plt.subplots(nr,cols,figsize=(14,3.6*nr));axes=np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:ax.axis("off")
    for ax,ix in zip(axes,picks):im,g=_plane(ref,src,bp[ix],tt[ix]);ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]);ax.scatter([0],[0],s=14);ax.set_title(f"{bq[ix]:.1f} mm");ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Best long source-led continuation: orthogonal CCTA QC");plt.tight_layout();plt.savefig(out/"02_best_long_orthogonal_qc.png",dpi=180);plt.close()
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"prior_status":prior.get("status"),"RCA_plane_qc_control":{"plane_count":int(len(rdf)),"pass_fraction":rca_frac,"accepted":True},"design":{"search_max_mm":SEARCH_MAX_MM,"field_margin_mm":FIELD_MARGIN_MM,"aortic_geometry_used_during_discovery":False,"aortic_geometry_posthoc_only":True,"initial_direction_count":len(dirs)},"n_hypotheses":int(len(h)),"n_accepted_hypotheses":int(h.accepted.sum()) if len(h) else 0,"best_hypothesis":bestrow.to_dict(),"posthoc":{"reaches_aorta":reaches,"eventually_turns_toward_aorta":returns,"start_aorta_distance_mm":float(ad[0]),"minimum_aorta_distance_mm":float(np.min(ad)),"end_aorta_distance_mm":float(ad[-1])},"candidate_role":"source-supported long continuation beyond the dense-QC local multidirection endpoint; not labeled LM and not added to frozen master","scientific_boundary":"A positive result establishes only long source-CCTA vessel continuity. Aortic geometry is evaluated post-hoc; LM identity and LCX topology remain unresolved."}
    _write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE});return _finalize(out,summary)

def _finalize(out,s):
    report=out/"OPENPLAQUE_SOURCE_LED_LONG_EXTENSION_REPORT.html";b=s.get("best_hypothesis",{});p=s.get("posthoc",{});report.write_text(f"<html><body><h1>OpenPlaque Source-Led Long Proximal-Trunk Extension v1</h1><p><b>Status:</b> {s.get('status')}</p><p>RCA plane-QC control: {s.get('RCA_plane_qc_control',{}).get('accepted')}</p><p>Accepted new extension: {b.get('accepted_arc_mm')} mm; plane pass fraction {b.get('accepted_plane_pass_fraction')}</p><p>Post-hoc aorta distance: {p.get('start_aorta_distance_mm')} -> min {p.get('minimum_aorta_distance_mm')} -> end {p.get('end_aorta_distance_mm')} mm.</p><p>Frozen master unchanged; LM identity unresolved.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_SOURCE_LED_LONG_EXTENSION_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file():z.write(p,p.name)
    return {"summary":s,"report":str(report),"zip":str(zpath)}
