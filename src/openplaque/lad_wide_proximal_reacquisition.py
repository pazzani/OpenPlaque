from __future__ import annotations

"""Wide proximal LAD reacquisition from the validated ~25 mm LAD endpoint.

Searches a 3-D shell around the accepted proximal LAD endpoint for coronary-sized
bright tubular candidates. Candidate discovery is independent of the aorta. Any
survivor must have local bidirectional source-CCTA support and must reconnect to
an independently traced LAD-side path after both traces are revalidated using
planes perpendicular to their final local tangents.
"""

import json, math, zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "lad-wide-proximal-reacquisition-v1.0"
SHELL_MIN_MM = 3.0
SHELL_MAX_MM = 12.0
CANDIDATE_N = 36
POSTHOC_STEP_MM = 0.35

def _json_read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _json_write(o,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2),encoding="utf-8")
def _unit(v):
    v=np.asarray(v,float); n=float(np.linalg.norm(v)); return v/max(n,1e-12)
def arc_mm(path,spacing):
    p=np.asarray(path,float)
    if len(p)<2:return np.zeros(len(p))
    d=np.diff(p,axis=0)*np.asarray(spacing)[None,:]
    return np.r_[0.,np.cumsum(np.linalg.norm(d,axis=1))]
def resample_path(path,spacing,step=.35):
    p=np.asarray(path,float)
    if len(p)<2:return p.copy()
    s=arc_mm(p,spacing); keep=np.r_[True,np.diff(s)>1e-6]; p,s=p[keep],s[keep]
    q=np.arange(0.,s[-1]+.5*step,step)
    if not len(q) or q[-1]<s[-1]-.1:q=np.r_[q,s[-1]]
    q[-1]=min(q[-1],s[-1]); return np.column_stack([np.interp(q,s,p[:,j]) for j in range(3)])
def source_zyx_to_lps(meta,zyx):
    p=np.atleast_2d(np.asarray(zyx,float)); pos=np.asarray(meta["positions_lps_mm"],float)
    o=np.asarray(meta["image_orientation_patient"],float); row,col=o[:3],o[3:]
    sp=np.asarray(meta["spacing_zyx"],float)
    sv=(pos[-1]-pos[0])/max(len(pos)-1,1) if len(pos)>1 else np.cross(row,col)*sp[0]
    out=pos[0][None,:]+p[:,0,None]*sv[None,:]
    out+=p[:,2,None]*sp[2]*row[None,:]; out+=p[:,1,None]*sp[1]*col[None,:]
    return out
def _orth_basis(t):
    t=_unit(t); ref=np.array([1.,0.,0.]) if abs(t[0])<.82 else np.array([0.,1.,0.])
    u=_unit(np.cross(t,ref)); v=_unit(np.cross(t,u)); return u,v
def _cone_dirs(base,angles=(10.,20.,32.),n_az=8):
    base=_unit(base); u,v=_orth_basis(base); out=[base]
    for ad in angles:
        a=math.radians(float(ad))
        for k in range(n_az):
            ph=2*math.pi*k/n_az
            out.append(_unit(math.cos(a)*base+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v)))
    return out
def orthogonal_plane(ct,p,t,spacing,half=5.,pix=.16):
    u,v=_orth_basis(t); g=np.arange(-half,half+1e-9,pix,dtype=np.float32)
    V,U=np.meshgrid(g,g,indexing="ij"); sp=np.asarray(spacing,np.float32); c=np.asarray(p,np.float32)*sp
    pos=c[None,None,:]+U[...,None]*u.astype(np.float32)+V[...,None]*v.astype(np.float32); vox=pos/sp
    out=np.empty(U.shape,np.float32)
    map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=out,order=1,mode="nearest",prefilter=False)
    return out,g,u,v
def lumen_metrics(im,g,min_hu=180.,max_hu=1200.):
    pix=float(abs(g[1]-g[0])); bright=(im>=min_hu)&(im<=max_hu)
    lab,_=ndi.label(bright,structure=np.ones((3,3),np.uint8)); cy=cx=len(g)//2; chosen=int(lab[cy,cx])
    if chosen==0:
        yy,xx=np.nonzero(bright)
        if len(yy):
            d=np.hypot(g[yy],g[xx]); j=int(np.argmin(d)); chosen=int(lab[yy[j],xx[j]]) if float(d[j])<=.9 else 0
    if chosen<=0:
        return {"radius_mm":np.nan,"centroid_shift_mm":np.nan,"centroid_u_mm":np.nan,"centroid_v_mm":np.nan,"circularity":np.nan,"component_median_hu":np.nan}
    comp=lab==chosen; yy,xx=np.nonzero(comp); area=float(comp.sum())*pix*pix; r=math.sqrt(area/math.pi)
    cu,cv=float(np.mean(g[xx])),float(np.mean(g[yy])); shift=float(math.hypot(cu,cv))
    er=ndi.binary_erosion(comp,structure=np.ones((3,3),bool)); per=max(float(np.logical_and(comp,~er).sum())*pix,pix)
    circ=float(np.clip(4*math.pi*area/(per*per),0,1.2))
    return {"radius_mm":r,"centroid_shift_mm":shift,"centroid_u_mm":cu,"centroid_v_mm":cv,"circularity":circ,"component_median_hu":float(np.median(im[comp]))}
def score_metrics(m,ref):
    rr,rh=float(ref["median_radius_mm"]),float(ref["median_center_hu"])
    r,sh,circ,med=m["radius_mm"],m["centroid_shift_mm"],m["circularity"],m["component_median_hu"]
    rs=math.exp(-.5*((r-rr)/max(.48*rr,.60))**2) if np.isfinite(r) else 0.
    ss=math.exp(-.5*(sh/.62)**2) if np.isfinite(sh) else 0.; cs=min(1.,max(0.,circ/.50)) if np.isfinite(circ) else 0.
    hs=math.exp(-.5*((med-rh)/320.)**2) if np.isfinite(med) else 0.; score=.38*rs+.30*ss+.18*cs+.14*hs
    hard=bool(np.isfinite(r) and .55*rr<=r<=min(3.05,1.70*rr+.35) and np.isfinite(sh) and sh<=.90 and np.isfinite(circ) and circ>=.20 and np.isfinite(med) and max(150.,.34*rh)<=med<=1200. and score>=.60)
    return float(score),hard
def _sampled_hu_ok(mm):
    a=np.asarray(mm); idx=tuple(slice(None,None,max(1,n//16)) for n in a.shape); s=np.asarray(a[idx],float)
    if not np.isfinite(s).any():return False
    p1,p99=np.nanpercentile(s,[1,99]); return bool(p99-p1>300 and p99>200 and p1<-200)

class LADWideProximalReacquisitionWorkflow:
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        self.root=Path(root); self.cache=self.root/"Cache"/"LAD_Wide_Proximal_Reacquisition_v1"; self.out=self.root/"LAD_Wide_Proximal_Reacquisition_Report"
        self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={"source_ct":True,"inputs":True,"candidates":False,"traces":False,"figures":False,"report":False}; self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()})
        self.ct=self.meta=self.spacing=self.lad=self.ref=self.anchor=self.aorta_tree=None; self.candidates=self.candidate_qc=self.lad_trace=self.summary=None; self.prov=[]
    def _record(self,c,a,p="",note=""):
        self.prov.append({"component":c,"reuse_requested":self.reuse.get(c,False),"action":a,"path":str(p),"note":note}); pd.DataFrame(self.prov).to_csv(self.out/"cache_provenance.csv",index=False)
    def cache_status(self):
        return pd.DataFrame([{"component":k,"reuse":v} for k,v in self.reuse.items()])
    def load_source_ct(self):
        metas=[self.root/"Cache"/"LAD_Frozen_Proximal_Reacquisition_v1"/"series7_geometry.json",self.root/"Cache"/"Coronary_Anatomy_Reconciliation_v1"/"series7_dicom_geometry.json"]
        meta_fp=next((p for p in metas if p.exists()),None)
        if meta_fp is None:raise FileNotFoundError("Series-7 geometry metadata not found")
        self.meta=_json_read(meta_fp); expected=tuple(self.meta["shape"]); paths=[]; src=self.meta.get("ct_cache_source")
        if src:paths.append(Path(src))
        paths += [self.root/"Cache"/"Secondary_3D_Vesselness_Topology_v1"/"series7_int16.npy",self.root/"Cache"/"Coronary_Anatomy_Reconciliation_v1"/"series7_int16.npy",self.root/"Cache"/"LAD_Confirmed_Backtrack_v1"/"series7_int16.npy"]
        hit=None
        for p in paths:
            if not p.exists():continue
            try:mm=np.load(p,mmap_mode="r")
            except Exception:continue
            if tuple(mm.shape)==expected and _sampled_hu_ok(mm):hit=p; self.ct=mm; break
        if hit is None:raise RuntimeError("No content-validated Series-7 CT cache found")
        self.spacing=np.asarray(self.meta["spacing_zyx"],float); self._record("source_ct","reused_content_validated_cache",hit,"shape+sampled-HU validation passed"); return self.ct
    def load_inputs(self):
        if self.ct is None:self.load_source_ct()
        base=self.root/"Cache"/"LAD_Frozen_Proximal_Reacquisition_v1"; summary_fp=base/"tracking_summary.json"; lad_fp=base/"combined_lad_centerline.csv"; cal_fp=base/"lad_lumen_calibration.json"
        if not all(p.exists() for p in [summary_fp,lad_fp,cal_fp]):raise FileNotFoundError("Validated proximal-LAD artifacts not found")
        s=_json_read(summary_fp)
        if s.get("status")!="POSTHOC_TANGENT_VALIDATED_PROXIMAL_EXTENSION_AORTA_NOT_REACHED":raise RuntimeError(f"Prior LAD state not validated for wide search: {s.get('status')}")
        ext=float(s.get("posthoc_validated_extension_mm",np.nan))
        if not (2.0<=ext<=3.0):raise RuntimeError(f"Unexpected validated proximal extension: {ext}")
        df=pd.read_csv(lad_fp); self.lad=df[["z","y","x"]].to_numpy(float); self.ref=_json_read(cal_fp); self.anchor=self.lad[0].copy()
        self._record("inputs","loaded_validated_24.997mm_lad",lad_fp,f"combined={float(s['combined_lad_length_mm']):.3f} mm; extension={ext:.3f} mm"); return s
    def load_aorta_ranker(self):
        fp=self.root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz"
        if not fp.exists():self._record("inputs","aorta_ranker_unavailable",fp,"candidate discovery unaffected"); return None
        arr=sitk.GetArrayFromImage(sitk.ReadImage(str(fp)))>0
        if arr.shape!=self.ct.shape:self._record("inputs","aorta_ranker_shape_mismatch",fp,str(arr.shape)); return None
        b=np.logical_and(arr,~ndi.binary_erosion(arr,iterations=1)); pts=np.argwhere(b)
        if len(pts)>50000:pts=pts[::int(math.ceil(len(pts)/50000))]
        self.aorta_tree=cKDTree(pts*self.spacing[None,:]); self._record("inputs","loaded_aorta_for_ranking_only",fp,f"boundary_points={len(pts)}"); return self.aorta_tree
    def _crop(self):
        rvox=np.ceil((SHELL_MAX_MM+3.)/self.spacing).astype(int); c=np.round(self.anchor).astype(int); lo=np.maximum(0,c-rvox); hi=np.minimum(np.array(self.ct.shape),c+rvox+1)
        sl=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3)); return sl,lo,np.asarray(self.ct[sl],np.float32)
    def _vesselness(self,crop):
        best=np.zeros(crop.shape,np.float32)
        for scale in (.7,1.0,1.4):
            sig=np.maximum(.6,scale/self.spacing); Hzz=ndi.gaussian_filter(crop,sig,order=(2,0,0))*scale*scale; Hyy=ndi.gaussian_filter(crop,sig,order=(0,2,0))*scale*scale; Hxx=ndi.gaussian_filter(crop,sig,order=(0,0,2))*scale*scale
            Hzy=ndi.gaussian_filter(crop,sig,order=(1,1,0))*scale*scale; Hzx=ndi.gaussian_filter(crop,sig,order=(1,0,1))*scale*scale; Hyx=ndi.gaussian_filter(crop,sig,order=(0,1,1))*scale*scale
            M=np.empty(crop.shape+(3,3),np.float32); M[...,0,0]=Hzz;M[...,1,1]=Hyy;M[...,2,2]=Hxx;M[...,0,1]=M[...,1,0]=Hzy;M[...,0,2]=M[...,2,0]=Hzx;M[...,1,2]=M[...,2,1]=Hyx
            ev=np.linalg.eigvalsh(M).astype(np.float32); order=np.argsort(np.abs(ev),axis=-1); ev=np.take_along_axis(ev,order,axis=-1); l1,l2,l3=ev[...,0],ev[...,1],ev[...,2]; good=(l2<0)&(l3<0)
            ra=np.abs(l2)/(np.abs(l3)+1e-6); rb=np.abs(l1)/np.sqrt(np.abs(l2*l3)+1e-6); ss=np.sqrt(l1*l1+l2*l2+l3*l3); c0=max(float(np.percentile(ss[good],90)) if np.any(good) else 1.,1e-3)
            v=(1-np.exp(-(ra*ra)/(2*.5*.5)))*np.exp(-(rb*rb)/(2*.5*.5))*(1-np.exp(-(ss*ss)/(2*c0*c0))); v[~good]=0; best=np.maximum(best,v.astype(np.float32))
        return best
    def _estimate_tangent(self,p):
        rad=2.4; rv=np.ceil(rad/self.spacing).astype(int); c=np.round(p).astype(int); lo=np.maximum(0,c-rv); hi=np.minimum(np.array(self.ct.shape),c+rv+1); sl=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3)); a=np.asarray(self.ct[sl])
        zz,yy,xx=np.nonzero((a>=180)&(a<=1200))
        if len(zz)<8:return None
        q=np.column_stack([zz+lo[0],yy+lo[1],xx+lo[2]]).astype(float); phys=(q-p[None,:])*self.spacing[None,:]; phys=phys[np.linalg.norm(phys,axis=1)<=rad]
        if len(phys)<8:return None
        w,V=np.linalg.eigh(np.cov(phys,rowvar=False)); return _unit(V[:,int(np.argmax(w))])
    def _eval(self,p,d,max_recenter=.8):
        im,g,u,v=orthogonal_plane(self.ct,p,d,self.spacing); m=lumen_metrics(im,g)
        if not np.isfinite(m["centroid_shift_mm"]):return p,m,0.,False,0.
        rec=float(m["centroid_shift_mm"])
        if rec>max_recenter:return p,m,0.,False,rec
        phys=np.asarray(p)*self.spacing+m["centroid_u_mm"]*u+m["centroid_v_mm"]*v; p2=phys/self.spacing; im2,g2,_,_=orthogonal_plane(self.ct,p2,d,self.spacing); m2=lumen_metrics(im2,g2); sc,ok=score_metrics(m2,self.ref); return p2,m2,sc,ok,rec
    def discover_candidates(self):
        if self.lad is None:self.load_inputs()
        sl,lo,crop=self._crop(); ves=self._vesselness(crop); zz,yy,xx=np.indices(crop.shape); glob=np.stack([zz+lo[0],yy+lo[1],xx+lo[2]],axis=-1).astype(float)
        dist=np.linalg.norm((glob-self.anchor[None,None,None,:])*self.spacing[None,None,None,:],axis=-1); lad_dense=resample_path(self.lad,self.spacing,.3); tree=cKDTree(lad_dense*self.spacing[None,:]); pts=glob.reshape(-1,3); near,_=tree.query(pts*self.spacing[None,:],k=1); near=near.reshape(crop.shape)
        loc=ves==ndi.maximum_filter(ves,size=5,mode="nearest"); mask=loc&(dist>=SHELL_MIN_MM)&(dist<=SHELL_MAX_MM)&(near>=1.8)&(crop>=180)&(crop<=1200); ids=np.argwhere(mask)
        if not len(ids):self.candidates=pd.DataFrame(); return self.candidates
        vals=ves[mask]; kept=[]
        for j in np.argsort(vals)[::-1]:
            p=ids[j].astype(float)+lo
            if all(np.linalg.norm((p-q)*self.spacing)>1.5 for q in kept):kept.append(p)
            if len(kept)>=CANDIDATE_N:break
        rows=[]
        for i,p in enumerate(kept):
            tangent=self._estimate_tangent(p)
            if tangent is None:continue
            tests=[]
            for d in _cone_dirs(tangent,angles=(8.,16.),n_az=6):
                p2,m,sc,ok,rec=self._eval(p,d); tests.append((sc,ok,p2,m,d,rec))
            tests.sort(key=lambda x:(x[1],x[0]),reverse=True); sc,ok,p2,m,d,rec=tests[0]; stable=sum(bool(x[1]) for x in tests); ad=np.nan
            if self.aorta_tree is not None:ad=float(self.aorta_tree.query(p2*self.spacing,k=1)[0])
            ijk=tuple(np.clip(np.round(p-lo).astype(int),0,np.array(ves.shape)-1))
            rows.append({"candidate_id":i,"z":p2[0],"y":p2[1],"x":p2[2],"shell_distance_mm":float(np.linalg.norm((p2-self.anchor)*self.spacing)),"vesselness":float(ves[ijk]),"radius_mm":float(m["radius_mm"]),"centroid_shift_mm":float(m["centroid_shift_mm"]),"circularity":float(m["circularity"]),"median_hu":float(m["component_median_hu"]),"plane_score":float(sc),"plane_pass":bool(ok),"orientation_passes":stable,"tangent_z":d[0],"tangent_y":d[1],"tangent_x":d[2],"aorta_distance_mm":ad})
        q=pd.DataFrame(rows)
        if len(q):q=q.sort_values(["plane_pass","orientation_passes","plane_score","vesselness"],ascending=False).reset_index(drop=True)
        self.candidates=q; q.to_csv(self.cache/"wide_candidates.csv",index=False); self._record("candidates","computed_wide_shell_candidates",self.cache/"wide_candidates.csv",f"n={len(q)}"); return q
    def _trace(self,start,d0,max_mm=5.0,beam_width=6,cone=(10.,20.,32.)):
        step=.60; beams=[{"pts":[np.asarray(start,float)],"dir":_unit(d0),"rows":[],"obj":0.}]
        for _ in range(int(math.ceil(max_mm/step))):
            props=[]
            for st in beams:
                for d in _cone_dirs(st["dir"],angles=cone,n_az=8):
                    guess=st["pts"][-1]+(step*d/self.spacing); p2,m,sc,ok,rec=self._eval(guess,d)
                    if not ok:continue
                    actual=float(np.linalg.norm((p2-st["pts"][-1])*self.spacing))
                    if not .30<=actual<=1.05:continue
                    row={"z":p2[0],"y":p2[1],"x":p2[2],"radius_mm":m["radius_mm"],"plane_score":sc,"recenter_mm":rec}; props.append({"pts":st["pts"]+[p2],"dir":d,"rows":st["rows"]+[row],"obj":st["obj"]+sc-.06*rec})
            if not props:break
            props.sort(key=lambda s:(len(s["pts"]),s["obj"]),reverse=True); beams=props[:beam_width]
        if not beams:return np.asarray([start],float),pd.DataFrame()
        best=max(beams,key=lambda s:(len(s["pts"]),s["obj"])); return np.asarray(best["pts"],float),pd.DataFrame(best["rows"])
    def _posthoc(self,path):
        p=resample_path(path,self.spacing,POSTHOC_STEP_MM)
        if len(p)<2:return p,pd.DataFrame(),0.
        s=arc_mm(p,self.spacing); rows=[]; flags=[]
        for i,x in enumerate(p):
            a=max(0,i-3);b=min(len(p)-1,i+3);t=_unit((p[b]-p[a])*self.spacing); p2,m,sc,ok,rec=self._eval(x,t); flags.append(bool(ok)); rows.append({"track_length_mm":s[i],"z":x[0],"y":x[1],"x":x[2],"radius_mm":m["radius_mm"],"centroid_shift_mm":m["centroid_shift_mm"],"plane_score":sc,"plane_pass":bool(ok),"recenter_mm":rec})
        fail=None; f=np.asarray(flags,bool)
        for i in range(len(f)):
            if not f[i] and int(np.sum(~f[i:min(len(f),i+3)]))>=2:fail=i;break
        end=max(0,(fail-1) if fail is not None else len(p)-1); q=pd.DataFrame(rows); q["within_validated_extent"]=np.arange(len(q))<=end; return p[:end+1],q,float(s[end])
    def run_reconnection(self):
        if self.candidates is None:self.discover_candidates()
        if self.aorta_tree is None:self.load_aorta_ranker()
        d0=_unit((self.lad[0]-self.lad[min(8,len(self.lad)-1)])*self.spacing); lad_raw,_=self._trace(self.anchor,d0,max_mm=6.0,beam_width=12,cone=(15.,30.,50.)); lad_valid,lad_qc,lad_mm=self._posthoc(lad_raw); self.lad_trace=lad_valid
        lad_qc.to_csv(self.cache/"lad_side_trace_qc.csv",index=False); pd.DataFrame(lad_valid,columns=["z","y","x"]).to_csv(self.cache/"lad_side_trace.csv",index=False)
        viable=self.candidates[(self.candidates.plane_pass==True)&(self.candidates.orientation_passes>=3)&(self.candidates.plane_score>=.72)].head(10).copy() if len(self.candidates) else pd.DataFrame(); results=[]
        for _,r in viable.iterrows():
            p=np.array([r.z,r.y,r.x],float); t=np.array([r.tangent_z,r.tangent_y,r.tangent_x],float)
            if np.dot(t,(self.anchor-p)*self.spacing)<0:t=-t
            inward_raw,_=self._trace(p,t,max_mm=6.0,beam_width=6); outward_raw,_=self._trace(p,-t,max_mm=3.0,beam_width=4); inward,iq,imm=self._posthoc(inward_raw); outward,oq,omm=self._posthoc(outward_raw)
            cid=int(r.candidate_id); pd.DataFrame(inward,columns=["z","y","x"]).to_csv(self.cache/f"candidate_{cid:02d}_inward.csv",index=False); pd.DataFrame(outward,columns=["z","y","x"]).to_csv(self.cache/f"candidate_{cid:02d}_outward.csv",index=False); iq.to_csv(self.cache/f"candidate_{cid:02d}_inward_qc.csv",index=False); oq.to_csv(self.cache/f"candidate_{cid:02d}_outward_qc.csv",index=False)
            sep=np.inf; align=np.nan
            if len(inward) and len(lad_valid):
                T=cKDTree(lad_valid*self.spacing[None,:]); ds,js=T.query(inward*self.spacing[None,:],k=1); ii=int(np.argmin(ds)); sep=float(ds[ii]); jj=int(js[ii])
                if len(inward)>=2 and len(lad_valid)>=2:
                    ia=max(0,ii-2);ib=min(len(inward)-1,ii+2);ja=max(0,jj-2);jb=min(len(lad_valid)-1,jj+2); ti=_unit((inward[ib]-inward[ia])*self.spacing); tl=_unit((lad_valid[jb]-lad_valid[ja])*self.spacing); align=float(np.degrees(np.arccos(np.clip(abs(np.dot(ti,tl)),-1,1))))
            reconnect=bool(imm>=1.2 and omm>=1.2 and lad_mm>=1.2 and sep<=1.25 and (np.isnan(align) or align<=40.)); results.append({"candidate_id":cid,"inward_valid_mm":imm,"outward_valid_mm":omm,"lad_side_valid_mm":lad_mm,"meeting_distance_mm":sep,"tangent_mismatch_deg":align,"bidirectional_reconnect":reconnect,"aorta_distance_mm":r.aorta_distance_mm,"candidate_plane_score":r.plane_score})
        rr=pd.DataFrame(results)
        if len(rr):rr=rr.sort_values(["bidirectional_reconnect","meeting_distance_mm","candidate_plane_score"],ascending=[False,True,False])
        rr.to_csv(self.cache/"reconnection_results.csv",index=False); n=int(len(viable)); nr=int(rr.bidirectional_reconnect.sum()) if len(rr) else 0
        status="WIDE_REACQUISITION_BIDIRECTIONAL_RECONNECTION_SUPPORTED" if nr else ("WIDE_REACQUISITION_CANDIDATES_NO_RECONNECTION" if n else "NO_COMPACT_WIDE_REACQUISITION_CANDIDATES")
        best=rr.iloc[0].replace({np.inf:None}).to_dict() if len(rr) else None
        self.candidate_qc=rr; self.summary={"algorithm":ALGORITHM_VERSION,"status":status,"validated_lad_input_mm":float(arc_mm(self.lad,self.spacing)[-1]),"shell_mm":[SHELL_MIN_MM,SHELL_MAX_MM],"n_candidates":int(len(self.candidates)),"n_viable_candidates":n,"n_bidirectional_reconnections":nr,"lad_side_posthoc_valid_mm":lad_mm,"best_candidate":best,"aorta_used_for_candidate_discovery":False,"aorta_used_for_trace_objective":False,"aorta_used_only_for_posthoc_ranking":True,"final_tangent_qc_required_for_all_traces":True}; _json_write(self.summary,self.cache/"summary.json"); self._record("traces","computed_bidirectional_reconnection",self.cache/"reconnection_results.csv",status); return self.summary
    def make_figures(self):
        if self.summary is None:self.run_reconnection()
        names=[]; lps=source_zyx_to_lps(self.meta,self.lad); anc=source_zyx_to_lps(self.meta,self.anchor)[0]; fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d"); ax.plot(lps[:,0],lps[:,1],lps[:,2],lw=3,label="Validated LAD")
        if self.lad_trace is not None and len(self.lad_trace)>1:q=source_zyx_to_lps(self.meta,self.lad_trace);ax.plot(q[:,0],q[:,1],q[:,2],lw=2,label="LAD-side trace")
        if len(self.candidates):cp=source_zyx_to_lps(self.meta,self.candidates[["z","y","x"]].to_numpy(float));ax.scatter(cp[:,0],cp[:,1],cp[:,2],s=20,label="Wide candidates")
        ax.scatter([anc[0]],[anc[1]],[anc[2]],s=80,label="Validated proximal endpoint");ax.legend();ax.set_title(self.summary["status"]);fp=self.out/"01_wide_reacquisition_lps.png";fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name)
        fig,ax=plt.subplots(figsize=(11,6));ax.axis("off"); show=self.candidates.head(12).copy() if len(self.candidates) else pd.DataFrame({"message":["No candidates"]}); cols=[c for c in ["candidate_id","shell_distance_mm","vesselness","radius_mm","plane_score","orientation_passes","aorta_distance_mm"] if c in show.columns]; txt=show[cols].round(3).to_string(index=False) if cols else show.to_string(index=False); ax.text(.01,.99,txt,va="top",family="monospace",fontsize=9);ax.set_title("Top wide-shell candidates");fp=self.out/"02_candidate_summary.png";fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name)
        fig,ax=plt.subplots(figsize=(10,6))
        if self.candidate_qc is not None and len(self.candidate_qc):x=np.arange(len(self.candidate_qc));ax.bar(x,self.candidate_qc.meeting_distance_mm.clip(upper=15));ax.axhline(1.25,ls="--");ax.set_xticks(x);ax.set_xticklabels(self.candidate_qc.candidate_id.astype(int));ax.set_ylabel("Meeting distance mm");ax.set_xlabel("Candidate ID")
        ax.set_title("Post-hoc validated bidirectional reconnection");fp=self.out/"03_reconnection_distances.png";fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name);_json_write(names,self.out/"figure_manifest.json");return names
    def build_report(self):
        if self.summary is None:self.run_reconnection()
        if not (self.out/"figure_manifest.json").exists():self.make_figures()
        figs=_json_read(self.out/"figure_manifest.json"); s=pd.DataFrame([self.summary]).to_html(index=False); c=self.candidates.head(20).to_html(index=False,float_format=lambda x:f"{x:.3f}") if len(self.candidates) else "<p>No candidates.</p>"; r=self.candidate_qc.to_html(index=False,float_format=lambda x:f"{x:.3f}") if self.candidate_qc is not None and len(self.candidate_qc) else "<p>No reconnection candidates.</p>"
        html=f"""<html><head><meta charset='utf-8'><title>OpenPlaque LAD Wide Proximal Reacquisition</title><style>body{{font-family:Arial;max-width:1500px;margin:24px auto;padding:0 18px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:5px;border-bottom:1px solid #ddd}}</style></head><body><h1>OpenPlaque — LAD Wide Proximal Reacquisition</h1><p><b>Status: {self.summary['status']}</b></p><p>Candidate discovery is independent of the aorta. Aorta distance is added only after candidate generation for endpoint plausibility/ranking. Every LAD-side and candidate-side trace is revalidated on planes perpendicular to its final local tangent before reconnection distances are computed.</p><h2>Summary</h2>{s}<h2>Wide-shell candidates</h2>{c}<h2>Bidirectional reconnection</h2>{r}{''.join(f"<h3>{n}</h3><img src='{n}'>" for n in figs)}</body></html>"""; fp=self.out/"OPENPLAQUE_LAD_WIDE_PROXIMAL_REACQUISITION_REPORT.html";fp.write_text(html,encoding="utf-8");return fp
    def package(self):
        report=self.build_report(); zp=self.out/"OPENPLAQUE_LAD_WIDE_PROXIMAL_REACQUISITION_REPORT_BACK.zip"; files=[self.cache/"summary.json",self.cache/"wide_candidates.csv",self.cache/"reconnection_results.csv",self.cache/"lad_side_trace.csv",self.cache/"lad_side_trace_qc.csv",self.out/"cache_provenance.csv",report,self.out/"figure_manifest.json"]+[self.out/n for n in _json_read(self.out/"figure_manifest.json")]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for fp in files:
                fp=Path(fp)
                if fp.exists():z.write(fp,arcname=fp.name)
            for fp in sorted(self.cache.glob("candidate_*_*.csv")):z.write(fp,arcname=fp.name)
        return zp

def synthetic_wide_reacquisition_self_test():
    ref={"median_radius_mm":1.5,"median_center_hu":450.}; good={"radius_mm":1.55,"centroid_shift_mm":.1,"circularity":.7,"component_median_hu":470.}; bad={"radius_mm":3.5,"centroid_shift_mm":2.0,"circularity":.2,"component_median_hu":470.}; sg,hg=score_metrics(good,ref);sb,hb=score_metrics(bad,ref);return {"passed":bool(hg and not hb and sg>sb and len(_cone_dirs([1,0,0]))==25),"good_score":sg,"bad_score":sb}
