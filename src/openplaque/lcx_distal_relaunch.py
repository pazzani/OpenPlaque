from __future__ import annotations

"""Relaunch the accepted secondary coronary branch before its broadened endpoint.

The accepted laterally-diverging secondary branch is frozen. This workflow
re-QCs that branch in source CCTA, identifies several rollback anchors 1.5-4 mm
before the distal endpoint where the lumen is still compact/coronary-sized, and
launches independent distal continuation searches from those anchors. The old
terminal tail is not forced into any continuation. LAD/trunk geometry remains
an immutable exclusion/reference. No LCX label is assigned automatically.
Research use only.
"""

import json, math, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

ALGORITHM_VERSION = "lcx-distal-relaunch-v1.0"
PINNED_RCA = {"median_radius_mm":1.616906,"median_center_hu":551.061478,"plane_pass_fraction":0.916667}


def _unit(v):
    v=np.asarray(v,float); n=float(np.linalg.norm(v)); return v/max(n,1e-9)

def _json(p): return json.loads(Path(p).read_text())
def _write_json(x,p): Path(p).write_text(json.dumps(x,indent=2,default=lambda v: float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v))

def arc_mm(path,spacing):
    p=np.asarray(path,float)
    if len(p)<=1: return np.zeros(len(p),float)
    d=np.diff(p,axis=0)*np.asarray(spacing,float); return np.r_[0.,np.cumsum(np.linalg.norm(d,axis=1))]

def resample_path(path,spacing,step=.3):
    p=np.asarray(path,float)
    if len(p)<2: return p.copy()
    s=arc_mm(p,spacing)
    if s[-1]<=step: return p.copy()
    q=np.arange(0.,s[-1],step)
    if len(q)==0 or q[-1]<s[-1]-1e-6: q=np.r_[q,s[-1]]
    return np.column_stack([np.interp(q,s,p[:,k]) for k in range(3)])

def _orth_basis(t):
    t=_unit(t); ref=np.array([1.,0.,0.]) if abs(t[0])<.82 else np.array([0.,1.,0.]); u=_unit(np.cross(t,ref)); v=_unit(np.cross(t,u)); return u,v

def orthogonal_plane(ct,p,t,spacing,half_mm=5.5,pix_mm=.18):
    u,v=_orth_basis(t); c=np.arange(-half_mm,half_mm+1e-9,pix_mm,dtype=np.float32); U,V=np.meshgrid(c,c,indexing='xy')
    sp=np.asarray(spacing,np.float32); pm=np.asarray(p,np.float32)*sp
    pos=pm[None,None,:]+U[...,None]*u.astype(np.float32)+V[...,None]*v.astype(np.float32); vox=pos/sp[None,None,:]
    out=np.empty(U.shape,dtype=np.float32); map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=out,order=1,mode='nearest',prefilter=False)
    return out,c

def _component_metrics(im,c,comp,cu,cv):
    pix=float(abs(c[1]-c[0])); area=int(comp.sum()); radius=math.sqrt(area*pix*pix/math.pi)
    er=ndi.binary_erosion(comp,structure=np.ones((3,3),bool)); per=max(1,int((comp & ~er).sum()))*pix
    circ=float(np.clip(4*math.pi*area*pix*pix/max(per*per,1e-8),0,1.2)); U,V=np.meshgrid(c,c,indexing='xy'); R=np.hypot(U-cu,V-cv)
    center=float(np.median(im[R<=.70])); core=float(np.median(im[R<=1.])); rv=im[(R>=2.5)&(R<=4.)]; ring=float(np.median(rv)) if rv.size else center
    return {"radius_mm":radius,"circularity":circ,"center_hu":center,"core_minus_ring_hu":core-ring,"centroid_u_mm":float(cu),"centroid_v_mm":float(cv),"recenter_shift_mm":float(math.hypot(cu,cv))}

def find_component(im,c,rca,search_mm=1.8,rescue=False):
    rr,rh=float(rca['median_radius_mm']),float(rca['median_center_hu']); lo=max(170.,min(260.,.38*rh)); bright=(im>=lo)&(im<=1200.)
    lab,nlab=ndi.label(bright,structure=np.ones((3,3),np.uint8)); pix=float(abs(c[1]-c[0])); lim=3.0 if rescue else float(search_mm); best=None
    for k in range(1,int(nlab)+1):
        comp=lab==k; area=int(comp.sum())
        if area<7: continue
        yy,xx=np.nonzero(comp); cu,cv=float(np.mean(c[xx])),float(np.mean(c[yy])); sh=float(math.hypot(cu,cv)); r=math.sqrt(area*pix*pix/math.pi)
        if sh>lim or not (.55<=r<=3.15): continue
        m=_component_metrics(im,c,comp,cu,cv)
        rs=math.exp(-.5*((m['radius_mm']-1.08*rr)/max(.72*rr,.82))**2); ps=math.exp(-.5*(m['recenter_shift_mm']/(1.45 if rescue else 1.0))**2)
        cs=float(np.clip(m['circularity']/.48,0,1)); hs=math.exp(-.5*((m['center_hu']-rh)/360.)**2); xs=1./(1.+math.exp(-(m['core_minus_ring_hu']-5.)/90.))
        choose=.33*rs+.28*ps+.16*cs+.12*hs+.11*xs
        if best is None or choose>best[0]: m['choose_score']=float(choose); best=(choose,m)
    return None if best is None else best[1]

def score_component(m,rca,rescue=False):
    rr,rh=float(rca['median_radius_mm']),float(rca['median_center_hu']); r=float(m['radius_mm']); sh=float(m['recenter_shift_mm']); circ=float(m['circularity']); hu=float(m['center_hu']); con=float(m['core_minus_ring_hu'])
    rs=math.exp(-.5*((r-1.05*rr)/max(.70*rr,.80))**2); ss=math.exp(-.5*(sh/(1.40 if rescue else 1.0))**2); cs=float(np.clip(circ/.52,0,1)); hs=math.exp(-.5*((hu-rh)/350.)**2); xs=1./(1.+math.exp(-(con-10.)/85.))
    sc=.32*rs+.25*ss+.17*cs+.14*hs+.12*xs
    hard=bool(.52*rr<=r<=min(3.15,2.*rr+.12) and sh<=(2.0 if rescue else 1.55) and circ>=.15 and 120<=hu<=1150 and sc>=.57)
    soft=bool(.42*rr<=r<=3.15 and sh<=(3.0 if rescue else 2.3) and circ>=.07 and 90<=hu<=1200 and sc>=.46)
    return float(sc),hard,soft

def _nearest_dist(p,ref,spacing):
    r=np.asarray(ref,float); return float(np.min(np.linalg.norm((r-np.asarray(p,float))*np.asarray(spacing,float),axis=1)))
def _angle(a,b): return math.degrees(math.acos(float(np.clip(np.dot(_unit(a),_unit(b)),-1,1))))

def _serial_qc(path,ct,spacing,rca,step=.75):
    p=resample_path(path,spacing,.28); s=arc_mm(p,spacing); rows=[]
    if len(p)>=4:
        xs=np.arange(min(.28,.08*s[-1]),s[-1]+1e-6,step)
        if len(xs)==0 or xs[-1]<s[-1]-.3: xs=np.r_[xs,s[-1]]
        for x in xs:
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); t=(p[i1]-p[i0])*np.asarray(spacing,float)
            im,c=orthogonal_plane(ct,p[i],t,spacing); m=find_component(im,c,rca,1.6,False)
            if m is None: rows.append({"arc_mm":float(s[i]),"radius_mm":np.nan,"circularity":np.nan,"center_hu":np.nan,"core_minus_ring_hu":np.nan,"recenter_shift_mm":np.inf,"plane_score":0.,"plane_pass":False,"soft_pass":False})
            else:
                sc,hp,sp=score_component(m,rca,False); rows.append({"arc_mm":float(s[i]),**m,"plane_score":sc,"plane_pass":hp,"soft_pass":sp})
    df=pd.DataFrame(rows)
    return df,{"length_mm":float(s[-1]) if len(s) else 0.,"plane_pass_fraction":float(df.plane_pass.mean()) if len(df) else 0.,"soft_pass_fraction":float(df.soft_pass.mean()) if len(df) else 0.,"median_plane_score":float(df.plane_score.median()) if len(df) else 0.,"median_radius_mm":float(df.radius_mm.median()) if len(df) else np.nan,"median_recenter_shift_mm":float(df.recenter_shift_mm.replace([np.inf],np.nan).median()) if len(df) else np.nan,"median_circularity":float(df.circularity.median()) if len(df) else np.nan}

class LCXDistalRelaunchWorkflow:
    COMPONENTS=("source_ct","frozen_geometry","anchors","search","figures","report")
    def __init__(self,root='/content/drive/MyDrive/OpenPlaque',reuse=None):
        self.root=Path(root); self.cache=self.root/'Cache'/'LCX_Distal_Relaunch_v1'; self.out=self.root/'LCX_Distal_Relaunch_Report'; self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS}; self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()}); self.prov=[]; self.ct=self.meta=self.spacing=None; self.seed=self.reference=self.trunk=self.rca=None; self.seed_qc=None; self.anchors=None; self.best=None; self.qc=None; self.candidates=None; self.summary=None; self.diag=None
    def _record(self,c,a,p='',n=''):
        self.prov.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":n}); pd.DataFrame(self.prov).to_csv(self.out/'cache_provenance.csv',index=False)
    def cache_status(self):
        n={"source_ct":"series7_int16.npy","frozen_geometry":"frozen_geometry.json","anchors":"anchors.csv","search":"search_summary.json","figures":"figures.done","report":"report.done"}; return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in n.items()])
    def _first(self,rels):
        for r in rels:
            p=self.root/r
            if p.exists(): return p
        return None
    def load_source_ct(self):
        own=self.cache/'series7_int16.npy'; om=self.cache/'series7_int16.json'
        if self.reuse['source_ct'] and own.exists() and om.exists(): self.ct=np.load(own,mmap_mode='r'); self.meta=_json(om); self.spacing=np.asarray(self.meta['spacing_zyx'],float); self._record('source_ct','reused',own); return self.ct
        src=self._first(['Cache/LCX_Course_Confirmation_v2/series7_int16.npy','Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy','Cache/LAD_Proximal_Recenter_v1/series7_int16.npy'])
        if src is None: raise FileNotFoundError('Disk-backed series-7 CCTA cache not found')
        sm=src.with_suffix('.json'); shutil.copyfile(src,own); shutil.copyfile(sm,om); self.ct=np.load(own,mmap_mode='r'); self.meta=_json(om); self.spacing=np.asarray(self.meta['spacing_zyx'],float); self._record('source_ct','imported_prior_cache',src); return self.ct
    def load_frozen_geometry(self):
        if self.ct is None: self.load_source_ct()
        seed=self.root/'Secondary_Branch_Lateral_Divergence_Report'/'branch_centerline.csv'; ss=self.root/'Secondary_Branch_Lateral_Divergence_Report'/'branch_summary.json'; ref=self.root/'LAD_Takeoff_Root_Alternatives_Report'/'alternative_01_centerline.csv'; trunk=self.root/'LAD_Takeoff_Confirmation_Report'/'trunk_centerline.csv'; rca=self.root/'LAD_Takeoff_Confirmation_Report'/'validated_rca_calibration.json'
        if not all(p.exists() for p in (seed,ss,ref,trunk)): raise FileNotFoundError('Required accepted branch/LAD/trunk outputs not found')
        info=_json(ss)
        if not bool(info.get('accepted',False)): raise RuntimeError('Accepted lateral-divergence branch missing')
        self.seed=pd.read_csv(seed)[['z','y','x']].to_numpy(float); self.reference=pd.read_csv(ref)[['z','y','x']].to_numpy(float); self.trunk=pd.read_csv(trunk)[['z','y','x']].to_numpy(float); self.rca=_json(rca) if rca.exists() else dict(PINNED_RCA)
        if not (1.0<=float(self.rca.get('median_radius_mm',np.nan))<=2.3): self.rca=dict(PINNED_RCA)
        self.seed_qc,_=_serial_qc(self.seed,self.ct,self.spacing,self.rca,.55); self.seed_qc.to_csv(self.cache/'seed_qc.csv',index=False)
        snap={"algorithm":ALGORITHM_VERSION,"accepted_seed_summary":info,"seed_length_mm":float(arc_mm(self.seed,self.spacing)[-1])}; _write_json(snap,self.cache/'frozen_geometry.json'); self._record('frozen_geometry','loaded_accepted_secondary_branch',seed); return snap
    def select_anchors(self):
        af=self.cache/'anchors.csv'
        if self.seed is None: self.load_frozen_geometry()
        if self.reuse['anchors'] and af.exists(): self.anchors=pd.read_csv(af); self._record('anchors','reused',af); return self.anchors
        p=resample_path(self.seed,self.spacing,.20); s=arc_mm(p,self.spacing); total=float(s[-1]); q=self.seed_qc.copy(); good=q[(q.plane_pass==True)&(q.plane_score>=.82)&(q.radius_mm<=2.35)&(q.recenter_shift_mm<=.95)&(q.arc_mm<=total-.8)]; preferred=[]
        for rollback in (1.5,2.0,2.5,3.0,3.5,4.0):
            target=total-rollback
            if len(good): r=good.iloc[(good.arc_mm-target).abs().argmin()]; a=float(r.arc_mm); source='qc_good'
            else: a=max(0.,target); source='fallback_distance'
            if all(abs(a-x[0])>=.55 for x in preferred): preferred.append((a,rollback,source))
        rows=[]
        for idx,(a,rollback,source) in enumerate(preferred,1):
            i=int(np.argmin(abs(s-a))); actual=float(s[i]); rows.append({"anchor_id":idx,"arc_mm":actual,"rollback_mm":total-actual,"z":p[i,0],"y":p[i,1],"x":p[i,2],"source":source})
        self.anchors=pd.DataFrame(rows).sort_values('arc_mm',ascending=False); self.anchors.to_csv(af,index=False); self._record('anchors','recomputed_and_cached',af,f'n={len(self.anchors)}'); return self.anchors
    def _direction_hypotheses(self,p,s,idx):
        out=[]; a=float(s[idx])
        for back in (.8,1.4,2.2,3.2,4.5):
            j=int(np.argmin(abs(s-max(0.,a-back))))
            if j>=idx: continue
            d=_unit((p[idx]-p[j])*self.spacing)
            if all(_angle(d,q)>=7 for q in out): out.append(d)
        return out or [_unit((p[idx]-p[max(0,idx-2)])*self.spacing)]
    def _propose(self,cur,d,step_mm,rescue=False):
        guess=np.asarray(cur,float)+(float(step_mm)*_unit(d))/self.spacing
        if np.any(guess<2) or np.any(guess>=np.asarray(self.ct.shape)-3): return None
        im,c=orthogonal_plane(self.ct,guess,d,self.spacing); m=find_component(im,c,self.rca,3.2 if rescue else 2.0,rescue)
        if m is None: return None
        u,v=_orth_basis(d); p=(guess*self.spacing+m['centroid_u_mm']*u+m['centroid_v_mm']*v)/self.spacing; vec=(p-cur)*self.spacing; L=float(np.linalg.norm(vec))
        if not (.22<=L<=(3.0 if rescue else 2.0)): return None
        sc,hp,sp=score_component(m,self.rca,rescue)
        if not sp: return None
        sep=_nearest_dist(p,np.vstack([self.reference,self.trunk]),self.spacing)
        return {"point":p,"direction":_unit(vec),"score":sc,"hard":hp,"soft":sp,"sep":sep,"step":L,**m}
    def search(self,max_new_mm=24.,beam_width=32):
        sf=self.cache/'search_summary.json'; pf=self.cache/'best_combined_centerline.csv'; qf=self.cache/'best_combined_qc.csv'; cf=self.cache/'candidates.csv'; df=self.cache/'step_diagnostics.csv'
        if self.anchors is None: self.select_anchors()
        if self.reuse['search'] and all(x.exists() for x in (sf,pf,qf,cf)):
            self.summary=_json(sf); self.best=pd.read_csv(pf)[['z','y','x']].to_numpy(float); self.qc=pd.read_csv(qf); self.candidates=pd.read_csv(cf); self.diag=pd.read_csv(df) if df.exists() else pd.DataFrame(); self._record('search','reused',sf); return self.summary
        seed=resample_path(self.seed,self.spacing,.20); ss=arc_mm(seed,self.spacing); ref=np.vstack([self.reference,self.trunk]); pool=[]; diag=[]
        for _,ar in self.anchors.iterrows():
            ai=int(np.argmin(abs(ss-float(ar.arc_mm)))); anchor=seed[ai].copy(); prefix=seed[:ai+1].copy(); sep0=_nearest_dist(anchor,ref,self.spacing); beams=[]
            for d in self._direction_hypotheses(seed,ss,ai): beams.append({"anchor_id":int(ar.anchor_id),"anchor_arc":float(ss[ai]),"prefix":prefix,"point":anchor,"direction":d,"ext":[anchor.copy()],"scores":[],"hard":[],"seps":[sep0],"turns":[]})
            nsteps=int(math.ceil(max_new_mm/.42))
            for step_i in range(nsteps):
                step_mm=.38 if step_i<8 else (.48 if step_i<18 else .60); props=[]
                for bi,st in enumerate(beams):
                    angles=(0,18,32,48,64,80) if step_i<10 else (0,14,28,42,56,70); local=[]
                    for ang in angles:
                        if ang==0: dirs=[st['direction']]
                        else:
                            u,v=_orth_basis(st['direction']); aa=math.radians(ang); naz=18 if step_i<10 else 14; dirs=[_unit(math.cos(aa)*st['direction']+math.sin(aa)*(math.cos(ph)*u+math.sin(ph)*v)) for ph in np.linspace(0,2*math.pi,naz,endpoint=False)]
                        for d in dirs:
                            r=self._propose(st['point'],d,step_mm,False)
                            if r is None or r['sep']<1.7: continue
                            local.append((r,ang,False))
                    if not local:
                        u,v=_orth_basis(st['direction'])
                        for ang in (88,105):
                            aa=math.radians(ang)
                            for ph in np.linspace(0,2*math.pi,20,endpoint=False):
                                d=_unit(math.cos(aa)*st['direction']+math.sin(aa)*(math.cos(ph)*u+math.sin(ph)*v)); r=self._propose(st['point'],d,max(.30,step_mm-.08),True)
                                if r is not None and r['sep']>=1.55: local.append((r,ang,True))
                    for r,ang,rescue in local:
                        smooth=float(np.clip(np.dot(st['direction'],r['direction']),-1,1)); obj=.56*r['score']+.12*float(r['hard'])+.10*((smooth+1)/2)+.14*min(r['sep']/5.,1)+.08*min(r['step']/.65,1)
                        ns={**st,"point":r['point'],"direction":r['direction'],"ext":st['ext']+[r['point'].copy()],"scores":st['scores']+[r['score']],"hard":st['hard']+[r['hard']],"seps":st['seps']+[r['sep']],"turns":st['turns']+[float(ang)]}; props.append((obj,ns)); diag.append({"anchor_id":st['anchor_id'],"step_index":step_i,"beam_index":bi,"rescue":rescue,"turn_deg":ang,"step_mm":r['step'],"plane_score":r['score'],"hard":r['hard'],"reference_separation_mm":r['sep'],"recenter_shift_mm":r['recenter_shift_mm']})
                if not props: break
                def rank(item):
                    _,st=item; ext=np.asarray(st['ext']); L=float(arc_mm(ext,self.spacing)[-1]); return .40*np.mean(st['scores'])+.15*np.mean(st['hard'])+.28*min(L/14.,1)+.17*min(st['seps'][-1]/5.,1)
                props.sort(key=rank,reverse=True); keep=[]
                for _,st in props:
                    if all(np.linalg.norm((st['point']-q['point'])*self.spacing)>=.45 for q in keep): keep.append(st)
                    if len(keep)>=beam_width: break
                pool.extend(keep); beams=keep
                if max(float(arc_mm(np.asarray(st['ext']),self.spacing)[-1]) for st in beams)>=max_new_mm: break
        rows=[]; evaluated=[]
        for st in pool:
            ext=np.asarray(st['ext'],float); new=float(arc_mm(ext,self.spacing)[-1])
            if new<1.5: continue
            combined=np.vstack([st['prefix'][:-1],ext]); eqdf,eqsum=_serial_qc(ext,self.ct,self.spacing,self.rca,.60); qdf,qsum=_serial_qc(combined,self.ct,self.spacing,self.rca,.80); sep_end=float(st['seps'][-1]); sep_med=float(np.median(st['seps'])); total=float(qsum['length_mm']); score=.27*min(new/12.,1)+.25*eqsum['plane_pass_fraction']+.20*eqsum['median_plane_score']+.12*min(sep_end/5.,1)+.08*min(sep_med/4.,1)+.08*qsum['median_plane_score']
            row={"candidate":len(rows)+1,"anchor_id":st['anchor_id'],"anchor_arc_mm":st['anchor_arc'],"new_length_mm":new,"total_length_mm":total,"extension_plane_pass_fraction":eqsum['plane_pass_fraction'],"extension_soft_pass_fraction":eqsum['soft_pass_fraction'],"extension_median_plane_score":eqsum['median_plane_score'],"extension_median_radius_mm":eqsum['median_radius_mm'],"combined_plane_pass_fraction":qsum['plane_pass_fraction'],"combined_median_plane_score":qsum['median_plane_score'],"endpoint_reference_separation_mm":sep_end,"median_extension_reference_separation_mm":sep_med,"max_turn_deg":max(st['turns']) if st['turns'] else 0.,"rank_score":score}; rows.append(row); evaluated.append((score,row,qsum,qdf,eqsum,combined,st))
        self.diag=pd.DataFrame(diag); self.diag.to_csv(df,index=False)
        if not evaluated:
            self.best=self.seed.copy(); self.qc=self.seed_qc.copy(); self.candidates=pd.DataFrame(); self.summary={"status":"NO_RELAUNCH_PERSISTED","accepted_topology":False,"interpretation":"Rollback relaunches found no new coronary-like distal course beyond 1.5 mm; accepted 13.8-mm secondary branch remains unchanged."}
        else:
            evaluated.sort(key=lambda x:x[0],reverse=True); score,row,qsum,qdf,eqsum,combined,st=evaluated[0]; self.best=resample_path(combined,self.spacing,.30); self.qc=qdf; self.candidates=pd.DataFrame(rows).sort_values('rank_score',ascending=False); continuity=bool(row['total_length_mm']>=20 and row['new_length_mm']>=7 and row['extension_plane_pass_fraction']>=.72 and row['extension_soft_pass_fraction']>=.82 and row['extension_median_plane_score']>=.76); independence=bool(row['endpoint_reference_separation_mm']>=3.0 and row['median_extension_reference_separation_mm']>=2.15); status='LCX_TOPOLOGY_SUPPORTED_ORIENTATION_UNVERIFIED' if continuity and independence else 'RELAUNCH_REQUIRES_REVIEW'; self.summary={**qsum,**row,"continuity_gate":continuity,"independence_gate":independence,"status":status,"accepted_topology":bool(continuity and independence),"interpretation":"Rollback relaunch from the last compact coronary-sized segment produced an independent persistent distal coronary-like course. LCX label still requires patient-coordinate compatibility or expert review." if continuity and independence else "A rollback relaunch produced some continuation but did not satisfy the persistent independent-course gate."}
        pd.DataFrame({"arc_mm":arc_mm(self.best,self.spacing),"z":self.best[:,0],"y":self.best[:,1],"x":self.best[:,2]}).to_csv(pf,index=False); self.qc.to_csv(qf,index=False); self.candidates.to_csv(cf,index=False); _write_json(self.summary,sf); self._record('search','recomputed_and_cached',sf,f"status={self.summary.get('status')}"); return self.summary
    def make_figures(self):
        done=self.cache/'figures.done'
        if self.reuse['figures'] and done.exists() and all((self.out/f).exists() for f in ('01_anchor_qc.png','02_best_course_cross_sections.png','03_course_vs_reference.png','04_candidate_tradeoff.png')): self._record('figures','reused',done); return
        if self.summary is None: self.search()
        q=self.seed_qc; fig,ax=plt.subplots(figsize=(9,4)); ax.plot(q.arc_mm,q.radius_mm,'o-',label='radius'); ax.axhline(2.35,ls='--',lw=1); [ax.axvline(float(a),lw=.8,alpha=.6) for a in self.anchors.arc_mm]; ax.set_xlabel('accepted branch arc (mm)'); ax.set_ylabel('radius (mm)'); ax.set_title('Rollback anchors before broadened endpoint'); fig.tight_layout(); fig.savefig(self.out/'01_anchor_qc.png',dpi=150); plt.close(fig)
        p=resample_path(self.best,self.spacing,.30); s=arc_mm(p,self.spacing); xs=np.linspace(.4,s[-1],12); fig,axs=plt.subplots(3,4,figsize=(14,10)); axs=axs.ravel()
        for ax,x in zip(axs,xs):
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing); ax.imshow(im,cmap='gray',vmin=-100,vmax=850,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,lw=.5); ax.axvline(0,lw=.5); ax.set_title(f'{s[i]:.1f} mm'); ax.axis('off')
        fig.suptitle('Best rollback-relaunched secondary course'); fig.tight_layout(); fig.savefig(self.out/'02_best_course_cross_sections.png',dpi=150); plt.close(fig)
        ref=np.vstack([self.reference,self.trunk]); fig,axs=plt.subplots(1,3,figsize=(16,5))
        for ax,(a,b,name) in zip(axs,[(1,2,'axial projection'),(0,2,'coronal projection'),(0,1,'sagittal projection')]): ax.plot(ref[:,b],ref[:,a],lw=1,label='LAD/trunk'); ax.plot(self.seed[:,b],self.seed[:,a],lw=1,label='accepted old branch'); ax.plot(p[:,b],p[:,a],lw=2,label='rollback relaunch'); ax.set_title(name); ax.set_aspect('equal'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(self.out/'03_course_vs_reference.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4));
        if self.candidates is not None and len(self.candidates):
            sc=ax.scatter(self.candidates.total_length_mm,self.candidates.extension_median_plane_score,c=self.candidates.endpoint_reference_separation_mm); ax.set_xlabel('total course length (mm)'); ax.set_ylabel('extension median plane score'); fig.colorbar(sc,ax=ax,label='endpoint reference separation (mm)')
        ax.set_title('Rollback relaunch candidates'); fig.tight_layout(); fig.savefig(self.out/'04_candidate_tradeoff.png',dpi=150); plt.close(fig); done.write_text('ok'); self._record('figures','recomputed_and_cached',done)
    def make_report(self):
        done=self.cache/'report.done'; html=self.out/'OPENPLAQUE_LCX_DISTAL_RELAUNCH_REPORT.html'
        if self.reuse['report'] and done.exists() and html.exists(): self._record('report','reused',html); return html
        if self.summary is None: self.search()
        if not (self.out/'01_anchor_qc.png').exists(): self.make_figures()
        _write_json(self.summary,self.out/'search_summary.json'); self.anchors.to_csv(self.out/'anchors.csv',index=False); self.seed_qc.to_csv(self.out/'accepted_seed_qc.csv',index=False); self.qc.to_csv(self.out/'best_combined_qc.csv',index=False); self.candidates.to_csv(self.out/'candidates.csv',index=False); self.diag.to_csv(self.out/'step_diagnostics.csv',index=False); pd.DataFrame({"arc_mm":arc_mm(self.best,self.spacing),"z":self.best[:,0],"y":self.best[:,1],"x":self.best[:,2]}).to_csv(self.out/'best_combined_centerline.csv',index=False)
        rows=''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in self.summary.items() if not isinstance(v,(dict,list))); body=f'''<html><body><h1>OpenPlaque LCX Distal Rollback Relaunch</h1><p>Algorithm: {ALGORITHM_VERSION}</p><p><b>Status: {self.summary.get('status')}</b></p><table border="1" cellspacing="0" cellpadding="4">{rows}</table><h2>Rollback anchors</h2><img src="01_anchor_qc.png" width="90%"><h2>Best course cross-sections</h2><img src="02_best_course_cross_sections.png" width="95%"><h2>Course versus frozen reference</h2><img src="03_course_vs_reference.png" width="95%"><h2>Candidate tradeoff</h2><img src="04_candidate_tradeoff.png" width="90%"><p>Research use only. No LCX label is assigned automatically.</p></body></html>'''; html.write_text(body); done.write_text('ok'); self._record('report','recomputed_and_cached',html); return html
    def package(self):
        self.make_report(); z=self.out/'OPENPLAQUE_LCX_DISTAL_RELAUNCH_REPORT_BACK.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
            for p in self.out.iterdir():
                if p.is_file() and p!=z: q.write(p,p.name)
        return z
