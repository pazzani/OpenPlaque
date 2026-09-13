from __future__ import annotations

"""Focused persistence test for a non-LAD branch at the validated LAD takeoff candidate.

This workflow freezes the prior validated LAD backbone, the strong takeoff candidate,
and the 8.8 mm proximal trunk that already passed source-plane QC. It does not
rediscover the LAD or trunk. It searches only for an independent coronary-sized
secondary branch from the candidate, explicitly rewarding geometric separation
from the established LAD/trunk reference while preserving source-resolution lumen
quality. No automatic LCX label is assigned. Research use only.
"""

import base64, json, math, shutil, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

ALGORITHM_VERSION = "secondary-branch-persistence-v1.0"
PINNED_RCA = {
    "median_radius_mm": 1.616906,
    "median_center_hu": 551.061478,
    "plane_pass_fraction": 0.916667,
}


def _unit(v):
    v=np.asarray(v,float); n=float(np.linalg.norm(v)); return v/max(n,1e-9)

def arc_mm(path, spacing):
    p=np.asarray(path,float)
    if len(p)<2: return np.zeros(len(p),float)
    d=np.diff(p,axis=0)*np.asarray(spacing,float)
    return np.r_[0.,np.cumsum(np.linalg.norm(d,axis=1))]

def resample_path(path, spacing, step_mm=.3):
    p=np.asarray(path,float)
    if len(p)<2: return p.copy()
    s=arc_mm(p,spacing)
    if s[-1]<=step_mm: return p.copy()
    q=np.arange(0,s[-1],step_mm)
    if not len(q) or q[-1]<s[-1]-1e-6: q=np.r_[q,s[-1]]
    return np.column_stack([np.interp(q,s,p[:,k]) for k in range(3)])

def _json(path): return json.loads(Path(path).read_text())
def _write_json(obj,path): Path(path).write_text(json.dumps(obj,indent=2,default=lambda x: float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else x))

def _orth_basis(t):
    t=_unit(t); ref=np.array([1.,0,0]) if abs(t[0])<.82 else np.array([0.,1.,0.])
    u=_unit(np.cross(t,ref)); v=_unit(np.cross(t,u)); return u,v

def orthogonal_plane(ct,p,tangent,spacing,half_mm=5.5,pix_mm=.18):
    u,v=_orth_basis(tangent); c=np.arange(-half_mm,half_mm+1e-9,pix_mm,dtype=np.float32)
    U,V=np.meshgrid(c,c,indexing="xy"); sp=np.asarray(spacing,np.float32); pm=np.asarray(p,np.float32)*sp
    pos=pm[None,None,:]+U[...,None]*u.astype(np.float32)+V[...,None]*v.astype(np.float32)
    vox=pos/sp[None,None,:]; out=np.empty(U.shape,np.float32)
    map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=out,order=1,mode="nearest",prefilter=False)
    return out,c

def _component_metrics(im,c,comp,cu,cv):
    pix=float(abs(c[1]-c[0])); area=int(comp.sum()); r=math.sqrt(area*pix*pix/math.pi)
    er=ndi.binary_erosion(comp,structure=np.ones((3,3),bool)); per=max(1,int((comp & ~er).sum()))*pix
    circ=float(np.clip(4*math.pi*(area*pix*pix)/max(per*per,1e-8),0,1.2))
    U,V=np.meshgrid(c,c,indexing="xy"); R=np.hypot(U-cu,V-cv)
    center=float(np.median(im[R<=.7])); core=float(np.median(im[R<=1.0])); rv=im[(R>=2.5)&(R<=4.0)]; ring=float(np.median(rv)) if rv.size else center
    return {"radius_mm":r,"circularity":circ,"center_hu":center,"core_minus_ring_hu":core-ring,"centroid_u_mm":float(cu),"centroid_v_mm":float(cv),"recenter_shift_mm":float(math.hypot(cu,cv))}

def find_component(im,c,rca,search_mm=1.7,rescue=False):
    rr=float(rca["median_radius_mm"]); rh=float(rca["median_center_hu"])
    lo=max(170.,min(260.,.38*rh)); lab,n=ndi.label((im>=lo)&(im<=1200),structure=np.ones((3,3),np.uint8)); pix=float(abs(c[1]-c[0])); lim=3.0 if rescue else search_mm
    best=None
    for k in range(1,int(n)+1):
        comp=lab==k; area=int(comp.sum())
        if area<7: continue
        yy,xx=np.nonzero(comp); cu,cv=float(np.mean(c[xx])),float(np.mean(c[yy])); shift=float(math.hypot(cu,cv)); radius=math.sqrt(area*pix*pix/math.pi)
        if shift>lim or not (.60<=radius<=3.25): continue
        m=_component_metrics(im,c,comp,cu,cv)
        rs=math.exp(-.5*((m["radius_mm"]-1.05*rr)/max(.65*rr,.75))**2); ps=math.exp(-.5*(shift/(1.3 if rescue else 1.0))**2)
        cs=float(np.clip(m["circularity"]/.50,0,1)); hs=math.exp(-.5*((m["center_hu"]-rh)/350.)**2); xs=1/(1+math.exp(-(m["core_minus_ring_hu"]-10)/85))
        score=.34*rs+.27*ps+.17*cs+.12*hs+.10*xs
        if best is None or score>best[0]: m["choose_score"]=float(score); best=(score,m)
    return None if best is None else best[1]

def score_component(m,rca,rescue=False):
    rr=float(rca["median_radius_mm"]); rh=float(rca["median_center_hu"]); r=float(m["radius_mm"]); sh=float(m["recenter_shift_mm"]); circ=float(m["circularity"]); hu=float(m["center_hu"]); con=float(m["core_minus_ring_hu"])
    rs=math.exp(-.5*((r-1.05*rr)/max(.58*rr,.70))**2); ss=math.exp(-.5*(sh/(1.25 if rescue else .95))**2); cs=float(np.clip(circ/.55,0,1)); hs=math.exp(-.5*((hu-rh)/330)**2); xs=1/(1+math.exp(-(con-15)/80))
    score=.34*rs+.23*ss+.18*cs+.14*hs+.11*xs
    hard=bool(.55*rr<=r<=min(3.15,2.0*rr+.1) and sh<=(1.9 if rescue else 1.5) and circ>=.16 and 120<=hu<=1150 and score>=.58)
    soft=bool(.45*rr<=r<=3.25 and sh<=(2.8 if rescue else 2.2) and circ>=.08 and 90<=hu<=1200 and score>=.48)
    return float(score),hard,soft

def _angle(a,b): return math.degrees(math.acos(float(np.clip(np.dot(_unit(a),_unit(b)),-1,1))))
def _cone(base,angles=(0,16,30,44),n_az=12):
    base=_unit(base); u,v=_orth_basis(base); out=[]
    for deg in angles:
        if deg==0: out.append(base); continue
        a=math.radians(deg)
        for k in range(n_az):
            ph=2*math.pi*k/n_az; out.append(_unit(math.cos(a)*base+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v)))
    return out

def nearest_dist_mm(p,path,spacing,exclude_origin_mm=0):
    q=np.asarray(path,float)
    if len(q)==0: return float("inf")
    if exclude_origin_mm>0:
        s=arc_mm(q,spacing); q=q[s>=exclude_origin_mm]
        if len(q)==0: return float("inf")
    return float(np.linalg.norm((q-np.asarray(p,float))*np.asarray(spacing,float),axis=1).min())

def serial_qc(path,ct,spacing,rca,step=.75):
    p=resample_path(path,spacing,.28); s=arc_mm(p,spacing); rows=[]
    if len(p)>=4:
        for x in np.arange(.25,s[-1]+1e-6,step):
            i=int(np.argmin(abs(s-x))); i0=max(0,i-5); i1=min(len(p)-1,i+5); t=(p[i1]-p[i0])*spacing
            im,c=orthogonal_plane(ct,p[i],t,spacing); m=find_component(im,c,rca,1.45,False)
            if m is None: rows.append({"arc_mm":float(s[i]),"plane_score":0.,"plane_pass":False,"soft_pass":False,"radius_mm":np.nan,"recenter_shift_mm":np.inf,"circularity":np.nan,"center_hu":np.nan}); continue
            sc,hp,sp=score_component(m,rca); rows.append({"arc_mm":float(s[i]),**m,"plane_score":sc,"plane_pass":hp,"soft_pass":sp})
    df=pd.DataFrame(rows)
    return df,{"length_mm":float(s[-1]) if len(s) else 0.,"plane_pass_fraction":float(df.plane_pass.mean()) if len(df) else 0.,"soft_pass_fraction":float(df.soft_pass.mean()) if len(df) else 0.,"median_plane_score":float(df.plane_score.median()) if len(df) else 0.,"median_radius_mm":float(df.radius_mm.median()) if len(df) else np.nan,"median_recenter_shift_mm":float(df.recenter_shift_mm.replace([np.inf],np.nan).median()) if len(df) else np.nan,"median_circularity":float(df.circularity.median()) if len(df) else np.nan}


class SecondaryBranchPersistenceWorkflow:
    COMPONENTS=("source_ct","frozen_geometry","branch_search","figures","report")
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        self.root=Path(root); self.cache=self.root/"Cache"/"Secondary_Branch_Persistence_v1"; self.out=self.root/"Secondary_Branch_Persistence_Report"; self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS}; self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()}); self.prov=[]; self.ct=self.meta=self.spacing=self.rca=None; self.reference=self.trunk=self.takeoff=self.lad_distal=None; self.branch=self.branch_qc=self.candidates=self.summary=None
    def _record(self,c,a,p="",n=""):
        self.prov.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":n}); pd.DataFrame(self.prov).to_csv(self.out/"cache_provenance.csv",index=False)
    def cache_status(self):
        names={"source_ct":"series7_int16.npy","frozen_geometry":"frozen_geometry.json","branch_search":"branch_summary.json","figures":"figures.done","report":"report.done"}; return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])
    def _first(self,rels):
        for r in rels:
            p=self.root/r
            if p.exists(): return p
        return None
    def load_source_ct(self):
        own=self.cache/"series7_int16.npy"; meta=self.cache/"series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and meta.exists(): self.ct=np.load(own,mmap_mode="r"); self.meta=_json(meta); self.spacing=np.asarray(self.meta["spacing_zyx"],float); self._record("source_ct","reused",own); return self.ct
        src=self._first(["Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy","Cache/LAD_Proximal_Recenter_v1/series7_int16.npy","Cache/LAD_Confirmed_Backtrack_v1/series7_int16.npy"])
        if src is None: raise FileNotFoundError("Disk-backed series-7 CCTA cache not found")
        sm=src.with_suffix(".json"); shutil.copyfile(src,own); shutil.copyfile(sm,meta); self.ct=np.load(own,mmap_mode="r"); self.meta=_json(meta); self.spacing=np.asarray(self.meta["spacing_zyx"],float); self._record("source_ct","imported_prior_cache",src); return self.ct
    def load_frozen_geometry(self):
        f=self.cache/"frozen_geometry.json"
        if self.ct is None: self.load_source_ct()
        alt=self.root/"LAD_Takeoff_Root_Alternatives_Report"/"alternative_01_centerline.csv"; cand=self.root/"LAD_Takeoff_Root_Alternatives_Report"/"takeoff_candidate.json"; trunk=self.root/"LAD_Takeoff_Confirmation_Report"/"trunk_centerline.csv"; tsum=self.root/"LAD_Takeoff_Confirmation_Report"/"trunk_summary.json"; rca=self.root/"LAD_Takeoff_Confirmation_Report"/"validated_rca_calibration.json"
        if not all(p.exists() for p in (alt,cand,trunk,tsum)): raise FileNotFoundError("Required prior takeoff-confirmation outputs not found")
        ts=_json(tsum)
        if not bool(ts.get("accepted",False)): raise RuntimeError("Prior proximal trunk did not pass; refusing secondary-only search")
        self.reference=pd.read_csv(alt)[["z","y","x"]].to_numpy(float); self.trunk=pd.read_csv(trunk)[["z","y","x"]].to_numpy(float); info=_json(cand); self.rca=_json(rca) if rca.exists() else dict(PINNED_RCA)
        if not (1.0<=float(self.rca.get("median_radius_mm",np.nan))<=2.3): self.rca=dict(PINNED_RCA)
        p=resample_path(self.reference,self.spacing,.2); s=arc_mm(p,self.spacing); i=int(np.argmin(abs(s-float(info["arc_from_proximal_mm"])))); self.takeoff=p[i].copy(); self.lad_distal=p[i:].copy(); _write_json({"candidate":info,"trunk_summary":ts},f); self._record("frozen_geometry","loaded_prior_confirmed_geometry",trunk)
        return {"candidate":info,"trunk_summary":ts}
    def _propose(self,cur,d,rescue=False):
        step=.68; guess=cur+(step*_unit(d))/self.spacing
        if np.any(guess<2) or np.any(guess>=np.asarray(self.ct.shape)-3): return None
        im,c=orthogonal_plane(self.ct,guess,d,self.spacing); m=find_component(im,c,self.rca,2.4 if rescue else 1.55,rescue)
        if m is None: return None
        u,v=_orth_basis(d); p=(guess*self.spacing+m["centroid_u_mm"]*u+m["centroid_v_mm"]*v)/self.spacing; vec=(p-cur)*self.spacing; L=float(np.linalg.norm(vec))
        if not (.38<=L<=(2.5 if rescue else 1.65)): return None
        sc,hp,sp=score_component(m,self.rca,rescue); 
        if not sp: return None
        sep=nearest_dist_mm(p,self.reference,self.spacing,exclude_origin_mm=.8)
        return {"point":p,"direction":_unit(vec),"score":sc,"hard":hp,"soft":sp,"sep":sep,"step":L,**m}
    def search_branch(self,max_length_mm=12.0,beam_width=24):
        sf=self.cache/"branch_summary.json"; pf=self.cache/"branch_centerline.csv"; qf=self.cache/"branch_qc.csv"; cf=self.cache/"branch_candidates.csv"
        if self.takeoff is None: self.load_frozen_geometry()
        if self.reuse["branch_search"] and all(p.exists() for p in (sf,pf,qf,cf)):
            self.summary=_json(sf); self.branch=pd.read_csv(pf)[["z","y","x"]].to_numpy(float); self.branch_qc=pd.read_csv(qf); self.candidates=pd.read_csv(cf); self._record("branch_search","reused",sf); return self.summary
        td=resample_path(self.lad_distal,self.spacing,.25); tr=resample_path(self.trunk,self.spacing,.25); lad_dir=_unit((td[min(len(td)-1,10)]-td[0])*self.spacing); trunk_dir=_unit((tr[0]-tr[-1])*self.spacing)
        seeds=[]
        u,v=_orth_basis(lad_dir)
        for ph in np.linspace(0,2*math.pi,48,endpoint=False):
            for deg in (45,60,75,90,105,120,135):
                a=math.radians(deg); d=_unit(math.cos(a)*lad_dir+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v))
                if _angle(d,trunk_dir)<42: continue
                r=self._propose(self.takeoff,d,True)
                if r is None: continue
                seeds.append({"point":r["point"],"direction":r["direction"],"path":[self.takeoff.copy(),r["point"].copy()],"scores":[r["score"]],"hard":[r["hard"]],"sep":[r["sep"]],"obj":r["score"]+.08*min(r["sep"]/2,1)})
        seeds.sort(key=lambda st:st["obj"],reverse=True); beams=[]
        for st in seeds:
            if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>.45 for q in beams): beams.append(st)
            if len(beams)>=beam_width: break
        pool=list(beams); nsteps=int(math.ceil(max_length_mm/.68))
        for _ in range(1,nsteps):
            props=[]
            for st in beams:
                L0=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1]); minsep=.25 if L0<1.5 else (.65 if L0<3 else (1.1 if L0<5 else 1.6))
                local=[]
                for d in _cone(st["direction"],(0,14,26,38,50),14):
                    r=self._propose(st["point"],d,False)
                    if r is not None and r["sep"]>=minsep: local.append(r)
                if not local:
                    for d in _cone(st["direction"],(55,70),12):
                        r=self._propose(st["point"],d,True)
                        if r is not None and r["sep"]>=minsep: local.append(r)
                for r in local:
                    smooth=(float(np.clip(np.dot(_unit(st["direction"]),_unit(r["direction"])),-1,1))+1)/2; obj=.50*r["score"]+.14*float(r["hard"])+.12*smooth+.24*min(r["sep"]/3.5,1)
                    props.append({"point":r["point"],"direction":r["direction"],"path":st["path"]+[r["point"].copy()],"scores":st["scores"]+[r["score"]],"hard":st["hard"]+[r["hard"]],"sep":st["sep"]+[r["sep"]],"obj":st["obj"]+obj})
            if not props: break
            def rank(st):
                L=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1]); return .28*np.mean(st["scores"])+.18*np.mean(st["hard"])+.28*min(L/9,1)+.26*min(st["sep"][-1]/4,1)
            props.sort(key=rank,reverse=True); keep=[]
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=.55 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            beams=keep; pool.extend(keep)
        rows=[]; ev=[]
        for st in pool:
            path=np.asarray(st["path"],float); L=float(arc_mm(path,self.spacing)[-1])
            if L<3: continue
            qdf,qsum=serial_qc(path,self.ct,self.spacing,self.rca,.75); endsep=float(st["sep"][-1]); medsep=float(np.median(st["sep"])); angle=_angle((path[1]-path[0])*self.spacing,lad_dir); rank=.30*min(L/9,1)+.27*qsum["plane_pass_fraction"]+.20*qsum["median_plane_score"]+.23*min(endsep/4,1)
            rows.append({"candidate":len(rows)+1,"length_mm":L,"plane_pass_fraction":qsum["plane_pass_fraction"],"median_plane_score":qsum["median_plane_score"],"median_radius_mm":qsum["median_radius_mm"],"median_reference_separation_mm":medsep,"endpoint_reference_separation_mm":endsep,"initial_angle_from_lad_deg":angle,"rank_score":rank}); ev.append((rank,L,qsum,qdf,path,st,angle))
        if not ev: self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.candidates=pd.DataFrame(rows); self.summary={"status":"FAIL","accepted":False,"reason":"no_candidate_persisted_beyond_3_mm"}
        else:
            ev.sort(key=lambda x:x[0],reverse=True); rank,L,qsum,qdf,path,st,angle=ev[0]; self.branch=resample_path(path,self.spacing,.28); self.branch_qc=qdf; endsep=float(st["sep"][-1]); medsep=float(np.median(st["sep"])); accepted=bool(L>=6 and qsum["plane_pass_fraction"]>=.75 and qsum["soft_pass_fraction"]>=.85 and qsum["median_plane_score"]>=.80 and endsep>=2.5 and medsep>=1.2 and 40<=angle<=140)
            self.summary={**qsum,"status":"PASS" if accepted else "REVIEW","accepted":accepted,"rank_score":float(rank),"endpoint_reference_separation_mm":endsep,"median_reference_separation_mm":medsep,"initial_angle_from_lad_deg":float(angle),"interpretation":"persistent independent coronary-like secondary branch from validated takeoff candidate; not automatically labeled LCX" if accepted else "best secondary trajectory remains insufficiently persistent or independent"}
            self.candidates=pd.DataFrame(rows).sort_values("rank_score",ascending=False)
        pd.DataFrame({"arc_mm":arc_mm(self.branch,self.spacing),"z":self.branch[:,0],"y":self.branch[:,1],"x":self.branch[:,2]}).to_csv(pf,index=False); self.branch_qc.to_csv(qf,index=False); self.candidates.to_csv(cf,index=False); _write_json(self.summary,sf); self._record("branch_search","recomputed_and_cached",sf,f"status={self.summary.get('status')}, length={self.summary.get('length_mm',0):.1f}"); return self.summary
    def plot_qc(self):
        done=self.cache/"figures.done"; figs=[self.out/f for f in ("01_secondary_branch_cross_sections.png","02_branch_vs_lad_mips.png","03_separation_profile.png","04_candidate_leaderboard.png")]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs): self._record("figures","reused",done); return figs
        if self.summary is None: self.search_branch()
        p=resample_path(self.branch,self.spacing,.28); s=arc_mm(p,self.spacing); fig,axs=plt.subplots(2,4,figsize=(11,5.5))
        if len(p)>=4:
            for ax,x in zip(axs.ravel(),np.linspace(.3,max(.3,s[-1]),8)):
                i=int(np.argmin(abs(s-x))); i0=max(0,i-5); i1=min(len(p)-1,i+5); im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+"); ax.set_aspect("equal"); ax.set_title(f"{x:.1f} mm")
        fig.suptitle(f"Secondary branch persistence — {self.summary.get('status')}"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[0],dpi=180,bbox_inches="tight"); plt.close(fig)
        pts=np.vstack([self.reference,self.branch]); pad=np.ceil(8/self.spacing).astype(int); lo=np.maximum(0,np.floor(pts.min(0)).astype(int)-pad); hi=np.minimum(np.asarray(self.ct.shape),np.ceil(pts.max(0)).astype(int)+pad+1); crop=np.asarray(self.ct[tuple(slice(int(lo[d]),int(hi[d])) for d in range(3))],dtype=np.int16); r=self.reference-lo; b=self.branch-lo
        fig,axs=plt.subplots(1,3,figsize=(15,5)); ims=[crop.max(0),crop.max(1),crop.max(2)]
        for ax,im,title in zip(axs,ims,("Axial MIP","Coronal MIP","Sagittal MIP")): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(title); ax.axis("off")
        axs[0].plot(r[:,2],r[:,1],lw=1.5,label="LAD/trunk reference"); axs[0].plot(b[:,2],b[:,1],lw=2,label="secondary"); axs[1].plot(r[:,2],r[:,0],lw=1.5); axs[1].plot(b[:,2],b[:,0],lw=2); axs[2].plot(r[:,1],r[:,0],lw=1.5); axs[2].plot(b[:,1],b[:,0],lw=2); axs[0].legend(fontsize=8); fig.suptitle("Secondary branch vs established LAD/trunk"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[1],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True); q=self.branch_qc
        if len(q): axs[0].plot(q.arc_mm,q.plane_score,marker="o"); axs[0].axhline(.8,ls="--"); axs[0].set_ylabel("plane score"); axs[1].plot(q.arc_mm,q.radius_mm,marker="o"); axs[1].axhline(float(self.rca["median_radius_mm"]),ls="--"); axs[1].set_ylabel("radius mm")
        if len(self.branch)>1:
            ss=arc_mm(self.branch,self.spacing); sep=[nearest_dist_mm(x,self.reference,self.spacing,.8) for x in self.branch]; axs[2].plot(ss,sep,marker="."); axs[2].axhline(2.5,ls="--"); axs[2].set_ylabel("reference sep mm"); axs[2].set_xlabel("branch arc mm")
        fig.suptitle("Secondary branch QC and independence"); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(figs[2],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5)); d=self.candidates.head(15) if self.candidates is not None else pd.DataFrame()
        if len(d): ax.scatter(d.endpoint_reference_separation_mm,d.length_mm,s=50+200*d.plane_pass_fraction); ax.set_xlabel("endpoint separation from LAD/trunk (mm)"); ax.set_ylabel("branch length (mm)"); [ax.annotate(str(int(r.candidate)),(r.endpoint_reference_separation_mm,r.length_mm)) for _,r in d.iterrows()]
        ax.axvline(2.5,ls="--"); ax.axhline(6,ls="--"); ax.set_title("Candidate branch persistence tradeoff"); fig.tight_layout(); fig.savefig(figs[3],dpi=180,bbox_inches="tight"); plt.close(fig); done.write_text(ALGORITHM_VERSION); self._record("figures","recomputed_and_cached",done); return figs
    def package(self):
        done=self.cache/"report.done"; zpath=self.out/"OPENPLAQUE_SECONDARY_BRANCH_PERSISTENCE_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists(): self._record("report","reused",zpath); return zpath
        figs=self.plot_qc(); html=self.out/"OPENPLAQUE_SECONDARY_BRANCH_PERSISTENCE_REPORT.html"
        def img(f): return f"<h2>{Path(f).name}</h2><img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(Path(f).read_bytes()).decode()}'>"
        html.write_text("<html><body><h1>OpenPlaque — Secondary branch persistence</h1><p><b>Research use only.</b> The LAD, takeoff candidate, and passed proximal trunk are frozen. This experiment tests only whether one non-LAD coronary-like branch persists independently. No LCX label is assigned automatically.</p>"+"".join(img(f) for f in figs)+"<h2>Result</h2><pre>"+json.dumps(self.summary,indent=2)+"</pre></body></html>",encoding="utf-8")
        names=[f.name for f in figs]+[html.name,"cache_provenance.csv"]
        for n in ("branch_summary.json","branch_centerline.csv","branch_qc.csv","branch_candidates.csv","frozen_geometry.json"):
            p=self.cache/n
            if p.exists(): shutil.copyfile(p,self.out/n); names.append(n)
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for n in names:
                p=self.out/n
                if p.exists(): z.write(p,arcname=n)
        done.write_text(ALGORITHM_VERSION); self._record("report","recomputed_and_cached",zpath); return zpath
