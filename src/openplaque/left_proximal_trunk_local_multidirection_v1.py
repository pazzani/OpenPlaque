from __future__ import annotations

"""Local multi-direction source-CCTA reacquisition at the validated proximal-trunk endpoint.

This is a prospective experiment. It starts from the dense-QC accepted 20-mm proximal-trunk
continuation and asks whether the previous endpoint search failed because the terminal tangent
plus strict monotonic-aorta rule forced the beam off a locally valid vessel branch. Discovery is
local and source-led: several seeds near the validated endpoint and multiple forward directions
are explored without using aortic distance in proposal acceptance or ranking. HU, vesselness,
coronary-mask support, turn, and loop gates are retained. Aortic distance is evaluated only
post-hoc. The frozen master is never modified and no candidate is labeled LM.
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
ALGORITHM="left-proximal-trunk-local-multidirection-v1.0"
OUTPUT_DIRNAME="Left_Proximal_Trunk_Local_Multidirection_v1"
PRIOR_DIR="Left_Proximal_Trunk_Continuation_QC_v1"
EXPANDED_DIR="Left_Proximal_Trunk_Expanded_Field_Reacquisition_v1"

SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_PATH=Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS=Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR=TS/"coronary_arteries/coronary_arteries.nii.gz"
LEG=TS/"coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA=TS/"heartchambers_highres/aorta.nii.gz"
PRIOR_PATH=Path(PRIOR_DIR)/"accepted_proximal_trunk_continuation_candidate.csv"

FIELD_MARGIN_MM=14.0
STEP_MM=.40
LOCAL_MAX_MM=10.0
BEAM_WIDTH=60
SEED_OFFSETS_MM=(0.0,.8,1.6,2.4)
TOP_INITIAL_DIRS=5
PREVIEW_MM=1.2
PLANE_STEP_MM=.40
MIN_ACCEPTED_EXTENSION_MM=5.0
MIN_PLANE_PASS_FRACTION=.80
SUSTAINED_FAIL_N=3

STATUS_RCA_FAIL="LOCAL_MULTIDIRECTION_RCA_CONTROL_FAILED"
STATUS_POS="PROXIMAL_TRUNK_LOCAL_MULTIDIRECTION_CONTINUATION_QC_POSITIVE"
STATUS_NEG="PROXIMAL_TRUNK_LOCAL_MULTIDIRECTION_NO_VALID_CONTINUATION"

def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _write_json(p,o):
    Path(p).write_text(json.dumps(o,indent=2,default=str,allow_nan=True),encoding="utf-8")

def _load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))

def _source(cache):
    a=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r")
    m=_load_json(_req(cache/"series7_int16.json"))
    ref=sitk.GetImageFromArray(np.asarray(a))
    sp=np.asarray(m["spacing_zyx"],float)
    ref.SetSpacing(tuple(sp[::-1])); ref.SetOrigin(tuple(np.asarray(m["positions_lps_mm"][0],float)))
    iop=np.asarray(m["image_orientation_patient"],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
    D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
    ref.SetDirection(tuple(D.ravel()))
    return ref,a,sp

def _xyz_to_zyx(img,p):
    p=np.atleast_2d(np.asarray(p,float)); o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing())
    D=np.asarray(img.GetDirection()).reshape(3,3)
    return (((p-o)@np.linalg.inv(D).T)/sp)[:,::-1]

def _zyx_to_xyz(img,p):
    p=np.atleast_2d(np.asarray(p,float)); q=p[:,::-1]
    o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing()); D=np.asarray(img.GetDirection()).reshape(3,3)
    return o+(q*sp)@D.T

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
    p=np.asarray(p,float)
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))

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
    same=(im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing())
          and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection()))
    if not same: im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im)>0

def _surface_tree(mask,ref,stride=2):
    surf=mask&~ndi.binary_erosion(mask,iterations=1,border_value=0)
    z=np.argwhere(surf)[::stride]; pts=_zyx_to_xyz(ref,z.astype(float))
    return cKDTree(pts.astype(np.float32)),pts

def _frangi_3d(vol,spacing,scales=(.55,.80,1.10,1.45)):
    x=np.clip(np.asarray(vol,np.float32),80,1000); x=(x-80)/920.
    spacing=np.asarray(spacing,float); best=np.zeros_like(x,np.float32)
    for sm in scales:
        sig=np.maximum(sm/spacing,.55); n=sm*sm
        hzz=ndi.gaussian_filter(x,sig,order=(2,0,0),mode="nearest")*n/(spacing[0]**2)
        hyy=ndi.gaussian_filter(x,sig,order=(0,2,0),mode="nearest")*n/(spacing[1]**2)
        hxx=ndi.gaussian_filter(x,sig,order=(0,0,2),mode="nearest")*n/(spacing[2]**2)
        hzy=ndi.gaussian_filter(x,sig,order=(1,1,0),mode="nearest")*n/(spacing[0]*spacing[1])
        hzx=ndi.gaussian_filter(x,sig,order=(1,0,1),mode="nearest")*n/(spacing[0]*spacing[2])
        hyx=ndi.gaussian_filter(x,sig,order=(0,1,1),mode="nearest")*n/(spacing[1]*spacing[2])
        H=np.empty(x.shape+(3,3),np.float32)
        H[...,0,0]=hzz;H[...,1,1]=hyy;H[...,2,2]=hxx
        H[...,0,1]=H[...,1,0]=hzy;H[...,0,2]=H[...,2,0]=hzx;H[...,1,2]=H[...,2,1]=hyx
        vals=np.linalg.eigvalsh(H); vals=np.take_along_axis(vals,np.argsort(np.abs(vals),axis=-1),axis=-1)
        l1,l2,l3=vals[...,0],vals[...,1],vals[...,2]; eps=1e-8
        ra=np.abs(l2)/(np.abs(l3)+eps); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+eps)
        ss=np.sqrt(l1*l1+l2*l2+l3*l3); nz=ss[ss>0]
        c=max(float(np.percentile(nz,90))*.45 if nz.size else .05,1e-4)
        vv=(1-np.exp(-(ra*ra)/.5))*np.exp(-(rb*rb)/.5)*(1-np.exp(-(ss*ss)/(2*c*c)))
        vv[(l2>=0)|(l3>=0)]=0; best=np.maximum(best,np.nan_to_num(vv).astype(np.float32))
    return best

def _local_field(ref,src,spacing,cur,leg,path,margin_mm=FIELD_MARGIN_MM):
    p,_=_resample(path,.35); pts=p[-min(len(p),24):]
    z=_xyz_to_zyx(ref,pts); mv=np.asarray([margin_mm]*3)/spacing
    lo=np.maximum(np.floor(z.min(0)-mv).astype(int),0); hi=np.minimum(np.ceil(z.max(0)+mv).astype(int)+1,np.asarray(src.shape))
    sl=tuple(slice(lo[k],hi[k]) for k in range(3))
    roi=np.asarray(src[sl]); cm=np.asarray(cur[sl]); lm=np.asarray(leg[sl]); union=cm|lm
    du=ndi.distance_transform_edt(~union,sampling=spacing)
    vessel=_frangi_3d(roi,spacing)
    kz=_xyz_to_zyx(ref,pts)-lo[None,:]
    good=np.all((kz>=0)&(kz<np.asarray(roi.shape)[None,:]),axis=1)
    kv=map_coordinates(vessel,kz[good].T,order=1,mode="nearest")
    thr=max(.003,min(.12,.30*float(np.percentile(kv,20))))
    norm=max(float(np.median(kv))*1.5,.02)
    return {"lo":lo,"roi":roi,"cur":cm,"leg":lm,"du":du,"v":vessel,"thr":thr,"norm":norm,"shape":list(roi.shape)}

def _sample_field(ref,F,p):
    z=_xyz_to_zyx(ref,[p])[0]-F["lo"]; shape=np.asarray(F["roi"].shape)
    if np.any(z<1) or np.any(z>shape-2): return None
    co=z[:,None]
    hu=float(map_coordinates(F["roi"],co,order=1,mode="nearest")[0])
    vv=float(map_coordinates(F["v"],co,order=1,mode="nearest")[0])
    dd=float(map_coordinates(F["du"],co,order=1,mode="nearest")[0])
    zi=np.rint(z).astype(int); c=bool(F["cur"][tuple(zi)]); l=bool(F["leg"][tuple(zi)])
    return hu,vv,dd,c,l

def _seed_points(path):
    p,a=_resample(path,.20); total=float(a[-1]); out=[]
    for off in SEED_OFFSETS_MM:
        target=max(0,total-off); ix=int(np.argmin(np.abs(a-target)))
        i0=max(0,ix-6); tangent=_unit(p[ix]-p[i0])
        out.append((float(off),p[ix].copy(),tangent))
    return out

def _preview_score(ref,F,start,d):
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
        if np.dot(d,tangent)<math.cos(math.radians(65)): continue
        sc=_preview_score(ref,F,start,d)
        if np.isfinite(sc) and sc>-1e8: scored.append((sc,d))
    scored.sort(key=lambda x:x[0],reverse=True)
    chosen=[]
    for sc,d in scored:
        if all(abs(float(np.dot(d,e)))<.985 for _,e in chosen):
            chosen.append((sc,d))
        if len(chosen)>=TOP_INITIAL_DIRS:break
    return chosen

def _beam_local(ref,F,start,tangent,max_mm=LOCAL_MAX_MM,step=STEP_MM,beam_width=BEAM_WIDTH):
    states=[(0.,[np.asarray(start,float)],_unit(tangent))]; best=states[0]; rows=[]
    nsteps=int(math.ceil(max_mm/step))
    for si in range(nsteps):
        nxt=[]; count={"step":si,"arc_budget_mm":(si+1)*step,"states_in":len(states),"proposals":0,
                        "reject_turn":0,"reject_loop":0,"reject_outside":0,"reject_hu":0,
                        "reject_vesselness":0,"reject_mask":0,"accepted":0,"kept":0}
        for score,pts,t in states:
            for d in _cone_dirs(t):
                count["proposals"]+=1
                if np.dot(d,t)<math.cos(math.radians(65)):count["reject_turn"]+=1;continue
                q=pts[-1]+step*d
                if len(pts)>5 and np.min(np.linalg.norm(np.asarray(pts[:-4])-q,axis=1))<.60*step:
                    count["reject_loop"]+=1;continue
                s=_sample_field(ref,F,q)
                if s is None:count["reject_outside"]+=1;continue
                hu,vv,du,c,l=s
                if not (80<=hu<=1400):count["reject_hu"]+=1;continue
                if vv<.42*F["thr"]:count["reject_vesselness"]+=1;continue
                if du>1.6:count["reject_mask"]+=1;continue
                vn=min(1.,vv/F["norm"]); support=1. if c and l else (.60 if c or l else .15)
                align=max(0.,float(np.dot(d,t)))
                ns=score+1.7*vn+.45*support+.45*align
                nt=_unit(.70*t+.30*d); st=(ns,pts+[q],nt); nxt.append(st);count["accepted"]+=1
                if ns>best[0]:best=st
        if not nxt:rows.append(count);break
        nxt.sort(key=lambda x:x[0],reverse=True); keep=[];bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]/.30).astype(int))
            if key in bins:continue
            bins.add(key);keep.append(st)
            if len(keep)>=beam_width:break
        states=keep;count["kept"]=len(states);rows.append(count)
    finals=states if states else [best]
    finals.sort(key=lambda x:x[0],reverse=True)
    sel=[]
    for st in finals:
        e=np.asarray(st[1][-1])
        if all(np.linalg.norm(e-np.asarray(q[1][-1]))>=1.0 for q in sel):sel.append(st)
        if len(sel)>=3:break
    return sel,pd.DataFrame(rows)

def _plane(ref,src,c,t,half=5.5,step=.20):
    t=_unit(t);u,v=_orth_basis(t);q=np.arange(-half,half+1e-9,step);yy,xx=np.meshgrid(q,q,indexing="ij")
    P=c+xx[...,None]*u+yy[...,None]*v
    return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q

def _component_metrics(im,q):
    iy=ix=int(np.argmin(np.abs(q))); center=float(im[iy,ix]);yy,xx=np.meshgrid(q,q,indexing="ij");rr=np.sqrt(xx*xx+yy*yy)
    thr=max(220.,min(500.,.55*center));bw=(im>=thr)&(rr<=3.5);lab,_=ndi.label(bw,np.ones((3,3),int))
    labid=int(lab[iy,ix])
    if labid<=0:
        pts=np.argwhere(bw)
        if not len(pts):return {"center_hu":center,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
        dist=np.hypot(q[pts[:,1]],q[pts[:,0]]);j=int(np.argmin(dist))
        if dist[j]>1.0:return {"center_hu":center,"component_found":False,"radius_mm":np.nan,"centroid_offset_mm":np.inf,"axis_ratio":np.inf,"contrast_hu":-np.inf}
        labid=int(lab[tuple(pts[j])])
    mask=lab==labid;pts=np.argwhere(mask);xs=q[pts[:,1]];ys=q[pts[:,0]]
    off=float(np.hypot(xs.mean(),ys.mean()));area=len(pts)*.04;rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        ev=np.linalg.eigvalsh(np.cov(np.column_stack([xs,ys]).T));axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else:axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0);contrast=float(np.median(im[mask])-np.median(im[ring]))
    return {"center_hu":center,"component_found":True,"radius_mm":rad,"centroid_offset_mm":off,"axis_ratio":axis,"contrast_hu":contrast}

def _plane_pass(m):
    return bool(m["component_found"] and m["center_hu"]>=200 and .55<=m["radius_mm"]<=3.2
                and m["centroid_offset_mm"]<=1.10 and m["axis_ratio"]<=2.2 and m["contrast_hu"]>=40)

def _dense_qc(ref,src,path):
    p,a=_resample(path,PLANE_STEP_MM)
    if len(p)<3:return p,a,pd.DataFrame()
    tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t,x) in enumerate(zip(p,tt,a)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m.update(index=i,arc_mm=float(x),plane_pass=_plane_pass(m));rows.append(m)
    return p,a,pd.DataFrame(rows)

def _truncate(df):
    if df.empty:return {"accepted":False,"accepted_arc_mm":0.,"accepted_plane_pass_fraction":0.,"accepted_last_index":-1}
    z=df["plane_pass"].astype(bool).to_numpy();cut=len(z)-1;first=None
    for i in range(len(z)-SUSTAINED_FAIL_N+1):
        if not np.any(z[i:i+SUSTAINED_FAIL_N]):first=i;cut=max(0,i-1);break
    acc=df.iloc[:cut+1];arc=float(acc["arc_mm"].iloc[-1]);frac=float(acc["plane_pass"].mean())
    return {"first_sustained_failure_index":first,"accepted_last_index":int(cut),"accepted_arc_mm":arc,
            "accepted_plane_pass_fraction":frac,"accepted":bool(arc>=MIN_ACCEPTED_EXTENSION_MM and frac>=MIN_PLANE_PASS_FRACTION)}

def _rca_control(ref,src,rca,aorta_tree):
    d0=float(aorta_tree.query(rca[0])[0]);d1=float(aorta_tree.query(rca[-1])[0]);r=rca if d0<=d1 else rca[::-1]
    p,a=_resample(r,.8);keep=(a>=2)&(a<=min(18.,a[-1]));p=p[keep] if np.sum(keep)>=8 else p
    tt=np.gradient(p,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9);rows=[]
    for i,(c,t) in enumerate(zip(p,tt)):
        im,g=_plane(ref,src,c,t);m=_component_metrics(im,g);m["plane_pass"]=_plane_pass(m);m["index"]=i;rows.append(m)
    d=pd.DataFrame(rows);frac=float(d["plane_pass"].mean());return d,bool(frac>=.75),frac

def synthetic_local_multidirection_self_test():
    assert LOCAL_MAX_MM>=MIN_ACCEPTED_EXTENSION_MM
    assert len(SEED_OFFSETS_MM)>=3
    assert TOP_INITIAL_DIRS>=3
    return {"ok":True,"seed_offsets_mm":SEED_OFFSETS_MM,"local_max_mm":LOCAL_MAX_MM,"top_initial_dirs":TOP_INITIAL_DIRS}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root);out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME;out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    req=[root/SOURCE_CACHE/"series7_int16.npy",root/SOURCE_CACHE/"series7_int16.json",root/MASTER,root/RCA_PATH,
         root/CUR,root/LEG,root/AORTA,root/PRIOR_DIR/"summary.json",root/PRIOR_PATH,root/EXPANDED_DIR/"summary.json"]
    for p in req:_req(p)
    prior=_load_json(root/PRIOR_DIR/"summary.json"); expanded=_load_json(root/EXPANDED_DIR/"summary.json"); master=_load_json(root/MASTER)
    if prior.get("status")!="PROXIMAL_TRUNK_CONTINUATION_QC_POSITIVE":raise RuntimeError("prior continuation prerequisite not positive")
    if expanded.get("status")!="PROXIMAL_TRUNK_EXPANDED_FIELD_NO_VALID_EXTENSION":raise RuntimeError(f"unexpected expanded prerequisite {expanded.get('status')}")
    ref,src,spacing=_source(root/SOURCE_CACHE);cur=_resample_mask(root/CUR,ref);leg=_resample_mask(root/LEG,ref);aorta=_resample_mask(root/AORTA,ref)
    aorta_tree,_=_surface_tree(aorta,ref);path=_load_path(root/PRIOR_PATH,ref);rca=_load_path(root/RCA_PATH,ref)
    rdf,rca_ok,rca_frac=_rca_control(ref,src,rca,aorta_tree);rdf.to_csv(out/"RCA_plane_qc_control.csv",index=False)
    if not rca_ok:
        summary={"status":STATUS_RCA_FAIL,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_modified":False,
                 "RCA_plane_qc_control":{"pass_fraction":rca_frac,"accepted":False}}
        _write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":STATUS_RCA_FAIL})
        return _finalize(out,summary)
    F=_local_field(ref,src,spacing,cur,leg,path);_write_json(out/"local_field_metadata.json",{"roi_shape":F["shape"],"vesselness_threshold":F["thr"],"field_margin_mm":FIELD_MARGIN_MM})
    seeds=_seed_points(path); hyps=[]; all_qc=[]; all_attr=[]
    for si,(off,start,tan) in enumerate(seeds):
        dirs=_initial_dirs(ref,F,start,tan)
        for di,(preview,d0) in enumerate(dirs):
            finals,attr=_beam_local(ref,F,start,d0);attr["seed_index"]=si;attr["dir_index"]=di;all_attr.append(attr)
            for fi,st in enumerate(finals):
                p=np.asarray(st[1],float);pp,aa,qc=_dense_qc(ref,src,p);tr=_truncate(qc)
                last=tr["accepted_last_index"];acc=pp[:last+1] if last>=0 else np.empty((0,3))
                start_ad=float(aorta_tree.query(start)[0]);end_ad=float(aorta_tree.query(acc[-1])[0]) if len(acc) else np.nan
                red=start_ad-end_ad if np.isfinite(end_ad) else np.nan
                hid=f"s{si}_d{di}_f{fi}"
                qc["hypothesis_id"]=hid;all_qc.append(qc)
                pd.DataFrame(p,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/f"path_{hid}.csv",index=False)
                hyps.append({"hypothesis_id":hid,"seed_index":si,"seed_offset_mm":off,"dir_index":di,"final_index":fi,
                             "preview_score":preview,"beam_score":float(st[0]),**tr,
                             "start_aorta_distance_mm":start_ad,"accepted_endpoint_aorta_distance_mm":end_ad,
                             "aorta_distance_reduction_mm":red})
    hdf=pd.DataFrame(hyps)
    if len(hdf):
        hdf=hdf.sort_values(["accepted","accepted_arc_mm","accepted_plane_pass_fraction","aorta_distance_reduction_mm"],
                            ascending=[False,False,False,False]).reset_index(drop=True)
    hdf.to_csv(out/"local_multidirection_hypotheses.csv",index=False)
    if all_attr:pd.concat(all_attr,ignore_index=True).to_csv(out/"local_multidirection_attrition.csv",index=False)
    if all_qc:pd.concat(all_qc,ignore_index=True).to_csv(out/"local_multidirection_dense_qc.csv",index=False)
    accepted=hdf[hdf["accepted"]==True] if len(hdf) else hdf
    best=accepted.iloc[0].to_dict() if len(accepted) else (hdf.iloc[0].to_dict() if len(hdf) else None)
    status=STATUS_POS if len(accepted) else STATUS_NEG
    if best:
        srcp=out/f"path_{best['hypothesis_id']}.csv"; pd.read_csv(srcp).to_csv(out/"best_local_multidirection_path.csv",index=False)
    if best:
        bp=pd.read_csv(out/f"path_{best['hypothesis_id']}.csv")[["lps_x_mm","lps_y_mm","lps_z_mm"]].to_numpy(float)
        pp,aa,qc=_dense_qc(ref,src,bp);picks=np.linspace(0,len(pp)-1,min(12,len(pp))).astype(int);cols=4;rows=int(math.ceil(len(picks)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(14,3.6*rows));axes=np.atleast_1d(axes).ravel();tt=np.gradient(pp,axis=0);tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9)
        for ax in axes[len(picks):]:ax.axis("off")
        for ax,ix in zip(axes,picks):
            im,g=_plane(ref,src,pp[ix],tt[ix]);ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]);ax.scatter([0],[0],s=14)
            ax.set_title(f"{aa[ix]:.1f} mm pass={bool(qc.iloc[ix].plane_pass)}");ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle("Best local multi-direction proximal-trunk hypothesis");plt.tight_layout();plt.savefig(out/"01_best_local_multidirection_orthogonal_qc.png",dpi=180);plt.close()
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,
             "prior_continuation_status":prior.get("status"),"expanded_field_status":expanded.get("status"),
             "RCA_plane_qc_control":{"plane_count":int(len(rdf)),"pass_fraction":rca_frac,"accepted":rca_ok},
             "design":{"seed_offsets_mm":list(SEED_OFFSETS_MM),"top_initial_dirs":TOP_INITIAL_DIRS,"local_max_mm":LOCAL_MAX_MM,
                       "aortic_monotonic_gate_during_discovery":False,"aortic_distance_used_posthoc_only":True},
             "n_hypotheses":int(len(hdf)),"n_accepted_hypotheses":int(len(accepted)),"best_hypothesis":best,
             "candidate_role":"local source-supported branch/continuation hypothesis beyond validated proximal-trunk endpoint; not LM and not frozen master",
             "scientific_change":"multi-seed, multi-direction local source search removes only the aortic-monotonic discovery constraint while retaining HU, vesselness, mask, turning and loop gates; aortic distance is post-hoc",
             "scientific_boundary":"A positive local continuation establishes only source-supported local vessel continuity. It does not establish an aortic bridge, LM identity, LCX topology, or modify frozen anatomy."}
    _write_json(out/"summary.json",summary);_write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    return _finalize(out,summary)

def _finalize(out,summary):
    report=out/"OPENPLAQUE_LOCAL_MULTIDIRECTION_PROXIMAL_TRUNK_REPORT.html"
    b=summary.get("best_hypothesis") or {}
    report.write_text(f"<html><body><h1>OpenPlaque Local Multi-Direction Proximal-Trunk v1</h1><p><b>Status:</b> {summary.get('status')}</p><p>RCA control: {summary.get('RCA_plane_qc_control',{}).get('accepted')}</p><p>Hypotheses: {summary.get('n_hypotheses')}; accepted: {summary.get('n_accepted_hypotheses')}</p><p>Best accepted arc: {b.get('accepted_arc_mm')} mm; post-hoc aorta reduction: {b.get('aorta_distance_reduction_mm')} mm.</p><p>Frozen master unchanged; no LM label assigned.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LOCAL_MULTIDIRECTION_PROXIMAL_TRUNK_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file():z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
