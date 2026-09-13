from __future__ import annotations

"""Focused source-CCTA confirmation of an LCX-compatible course.

This workflow freezes the previously accepted LAD/takeoff/proximal-trunk geometry
and the accepted laterally diverging secondary branch. It does not rediscover
those structures. It extends only the accepted secondary branch in source CCTA,
re-centering every proposal on a coronary-sized contrast-filled component.

The primary conclusion is intentionally topological/anatomic-course based:
continuous coronary-like lumen + sustained independence from the established
LAD/trunk + continued distal course. If an explicit voxel->LPS affine is present
in the source cache metadata, left/posterior patient-coordinate displacement is
reported as an additional check. No LCX label is assigned solely from geometry.
Research use only.
"""

import base64
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

ALGORITHM_VERSION = "lcx-course-confirmation-v1.0"
PINNED_RCA = {"median_radius_mm": 1.616906, "median_center_hu": 551.061478, "plane_pass_fraction": 0.916667}


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
    if q[-1]<s[-1]-1e-6: q=np.r_[q,s[-1]]
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

def find_component(im,c,rca,search_mm=1.65,rescue=False):
    rr,rh=float(rca['median_radius_mm']),float(rca['median_center_hu']); lo=max(170.,min(260.,.38*rh)); bright=(im>=lo)&(im<=1200.)
    lab,nlab=ndi.label(bright,structure=np.ones((3,3),np.uint8)); pix=float(abs(c[1]-c[0])); lim=2.8 if rescue else float(search_mm); best=None
    for k in range(1,int(nlab)+1):
        comp=lab==k; area=int(comp.sum())
        if area<7: continue
        yy,xx=np.nonzero(comp); cu,cv=float(np.mean(c[xx])),float(np.mean(c[yy])); sh=float(math.hypot(cu,cv)); r=math.sqrt(area*pix*pix/math.pi)
        if sh>lim or not (.60<=r<=3.25): continue
        m=_component_metrics(im,c,comp,cu,cv)
        rs=math.exp(-.5*((m['radius_mm']-1.08*rr)/max(.70*rr,.82))**2); ps=math.exp(-.5*(m['recenter_shift_mm']/(1.30 if rescue else .95))**2)
        cs=float(np.clip(m['circularity']/.48,0,1)); hs=math.exp(-.5*((m['center_hu']-rh)/360.)**2); xs=1./(1.+math.exp(-(m['core_minus_ring_hu']-5.)/90.))
        choose=.33*rs+.28*ps+.16*cs+.12*hs+.11*xs
        if best is None or choose>best[0]: m['choose_score']=float(choose); best=(choose,m)
    return None if best is None else best[1]

def score_component(m,rca,rescue=False):
    rr,rh=float(rca['median_radius_mm']),float(rca['median_center_hu']); r=float(m['radius_mm']); sh=float(m['recenter_shift_mm']); circ=float(m['circularity']); hu=float(m['center_hu']); con=float(m['core_minus_ring_hu'])
    rs=math.exp(-.5*((r-1.05*rr)/max(.70*rr,.80))**2); ss=math.exp(-.5*(sh/(1.30 if rescue else .95))**2); cs=float(np.clip(circ/.52,0,1)); hs=math.exp(-.5*((hu-rh)/350.)**2); xs=1./(1.+math.exp(-(con-10.)/85.))
    sc=.32*rs+.25*ss+.17*cs+.14*hs+.12*xs
    hard=bool(.56*rr<=r<=min(3.25,2.*rr+.15) and sh<=(1.9 if rescue else 1.5) and circ>=.16 and 120<=hu<=1150 and sc>=.58)
    soft=bool(.45*rr<=r<=3.25 and sh<=(2.8 if rescue else 2.2) and circ>=.08 and 90<=hu<=1200 and sc>=.48)
    return float(sc),hard,soft

def _nearest_dist(p,ref,spacing):
    r=np.asarray(ref,float); return float(np.min(np.linalg.norm((r-np.asarray(p,float))*np.asarray(spacing,float),axis=1)))

def _serial_qc(path,ct,spacing,rca,step=.8):
    p=resample_path(path,spacing,.28); s=arc_mm(p,spacing); rows=[]
    if len(p)>=4:
        xs=np.arange(min(.28,.08*s[-1]),s[-1]+1e-6,step)
        if len(xs)==0 or xs[-1]<s[-1]-.3: xs=np.r_[xs,s[-1]]
        for x in xs:
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); t=(p[i1]-p[i0])*np.asarray(spacing,float); im,c=orthogonal_plane(ct,p[i],t,spacing); m=find_component(im,c,rca,1.55,False)
            if m is None: rows.append({"arc_mm":float(s[i]),"radius_mm":np.nan,"circularity":np.nan,"center_hu":np.nan,"core_minus_ring_hu":np.nan,"recenter_shift_mm":np.inf,"plane_score":0.,"plane_pass":False,"soft_pass":False})
            else:
                sc,hp,sp=score_component(m,rca,False); rows.append({"arc_mm":float(s[i]),**m,"plane_score":sc,"plane_pass":hp,"soft_pass":sp})
    df=pd.DataFrame(rows)
    return df,{"length_mm":float(s[-1]) if len(s) else 0.,"plane_pass_fraction":float(df.plane_pass.mean()) if len(df) else 0.,"soft_pass_fraction":float(df.soft_pass.mean()) if len(df) else 0.,"median_plane_score":float(df.plane_score.median()) if len(df) else 0.,"median_radius_mm":float(df.radius_mm.median()) if len(df) else np.nan,"median_recenter_shift_mm":float(df.recenter_shift_mm.replace([np.inf],np.nan).median()) if len(df) else np.nan,"median_circularity":float(df.circularity.median()) if len(df) else np.nan}

def _explicit_affine(meta):
    for k in ('affine_zyx_to_lps','zyx_to_lps_affine','voxel_zyx_to_lps'):
        if k in meta:
            a=np.asarray(meta[k],float)
            if a.shape==(4,4): return a,k
    return None,None

def _to_lps(points,aff):
    p=np.asarray(points,float); h=np.c_[p,np.ones(len(p))]; return (h@aff.T)[:,:3]


class LCXCourseConfirmationWorkflow:
    COMPONENTS=("source_ct","frozen_seed","course_extension","figures","report")
    def __init__(self,root='/content/drive/MyDrive/OpenPlaque',reuse=None):
        self.root=Path(root); self.cache=self.root/'Cache'/'LCX_Course_Confirmation_v1'; self.out=self.root/'LCX_Course_Confirmation_Report'; self.cache.mkdir(parents=True,exist_ok=True); self.out.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS}; self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()}); self.prov=[]; self.ct=self.meta=self.spacing=None; self.seed=self.reference=self.trunk=self.rca=None; self.combined=self.qc=self.candidates=self.summary=None
    def _record(self,c,a,p='',n=''):
        self.prov.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":n}); pd.DataFrame(self.prov).to_csv(self.out/'cache_provenance.csv',index=False)
    def cache_status(self):
        n={"source_ct":"series7_int16.npy","frozen_seed":"frozen_seed.json","course_extension":"course_summary.json","figures":"figures.done","report":"report.done"}; return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in n.items()])
    def _first(self,rels):
        for r in rels:
            p=self.root/r
            if p.exists(): return p
        return None
    def load_source_ct(self):
        own=self.cache/'series7_int16.npy'; om=self.cache/'series7_int16.json'
        if self.reuse['source_ct'] and own.exists() and om.exists(): self.ct=np.load(own,mmap_mode='r'); self.meta=_json(om); self.spacing=np.asarray(self.meta['spacing_zyx'],float); self._record('source_ct','reused',own); return self.ct
        src=self._first(['Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy','Cache/LAD_Proximal_Recenter_v1/series7_int16.npy','Cache/LAD_Confirmed_Backtrack_v1/series7_int16.npy'])
        if src is None: raise FileNotFoundError('Disk-backed series-7 CCTA cache not found')
        sm=src.with_suffix('.json'); shutil.copyfile(src,own); shutil.copyfile(sm,om); self.ct=np.load(own,mmap_mode='r'); self.meta=_json(om); self.spacing=np.asarray(self.meta['spacing_zyx'],float); self._record('source_ct','imported_prior_cache',src); return self.ct
    def load_frozen_seed(self):
        if self.ct is None: self.load_source_ct()
        seed=self.root/'Secondary_Branch_Lateral_Divergence_Report'/'branch_centerline.csv'; ss=self.root/'Secondary_Branch_Lateral_Divergence_Report'/'branch_summary.json'; ref=self.root/'LAD_Takeoff_Root_Alternatives_Report'/'alternative_01_centerline.csv'; trunk=self.root/'LAD_Takeoff_Confirmation_Report'/'trunk_centerline.csv'; rca=self.root/'LAD_Takeoff_Confirmation_Report'/'validated_rca_calibration.json'
        if not all(p.exists() for p in (seed,ss,ref,trunk)): raise FileNotFoundError('Required accepted secondary-branch/LAD/trunk outputs not found')
        info=_json(ss)
        if not bool(info.get('accepted',False)): raise RuntimeError('Lateral-divergence branch is not accepted; refusing LCX-course confirmation')
        self.seed=pd.read_csv(seed)[['z','y','x']].to_numpy(float); self.reference=pd.read_csv(ref)[['z','y','x']].to_numpy(float); self.trunk=pd.read_csv(trunk)[['z','y','x']].to_numpy(float); self.rca=_json(rca) if rca.exists() else dict(PINNED_RCA)
        if not (1.0<=float(self.rca.get('median_radius_mm',np.nan))<=2.3): self.rca=dict(PINNED_RCA)
        snap={"algorithm":ALGORITHM_VERSION,"accepted_seed_summary":info,"seed_length_mm":float(arc_mm(self.seed,self.spacing)[-1])}; _write_json(snap,self.cache/'frozen_seed.json'); self._record('frozen_seed','loaded_accepted_lateral_branch',seed); return snap
    def _propose(self,cur,d,rescue=False):
        step=.66; guess=cur+(step*_unit(d))/self.spacing
        if np.any(guess<2) or np.any(guess>=np.asarray(self.ct.shape)-3): return None
        im,c=orthogonal_plane(self.ct,guess,d,self.spacing); m=find_component(im,c,self.rca,2.5 if rescue else 1.65,rescue)
        if m is None: return None
        u,v=_orth_basis(d); p=(guess*self.spacing+m['centroid_u_mm']*u+m['centroid_v_mm']*v)/self.spacing; vec=(p-cur)*self.spacing; L=float(np.linalg.norm(vec))
        if not (.35<=L<=(2.4 if rescue else 1.65)): return None
        sc,hp,sp=score_component(m,self.rca,rescue)
        if not sp: return None
        sep=_nearest_dist(p,np.vstack([self.reference,self.trunk]),self.spacing)
        return {"point":p,"direction":_unit(vec),"score":sc,"hard":hp,"soft":sp,"sep":sep,"step":L,**m}
    def extend_course(self,max_extra_mm=24.,beam_width=18):
        sf=self.cache/'course_summary.json'; pf=self.cache/'combined_branch_centerline.csv'; qf=self.cache/'combined_branch_qc.csv'; cf=self.cache/'course_candidates.csv'
        if self.seed is None: self.load_frozen_seed()
        if self.reuse['course_extension'] and all(p.exists() for p in (sf,pf,qf,cf)):
            self.summary=_json(sf); self.combined=pd.read_csv(pf)[['z','y','x']].to_numpy(float); self.qc=pd.read_csv(qf); self.candidates=pd.read_csv(cf); self._record('course_extension','reused',sf); return self.summary
        seed=resample_path(self.seed,self.spacing,.25); s0=arc_mm(seed,self.spacing); tail=max(0,len(seed)-10); d0=_unit((seed[-1]-seed[tail])*self.spacing); sep0=_nearest_dist(seed[-1],np.vstack([self.reference,self.trunk]),self.spacing)
        beams=[{"point":seed[-1].copy(),"direction":d0,"ext":[seed[-1].copy()],"scores":[],"hard":[],"seps":[sep0],"obj":0.}]; pool=[]; nsteps=int(math.ceil(max_extra_mm/.66))
        for step_i in range(nsteps):
            props=[]
            for st in beams:
                dirs=[]
                for ang in (0,14,26,40):
                    if ang==0: dirs.append(st['direction']); continue
                    u,v=_orth_basis(st['direction']); a=math.radians(ang)
                    for ph in np.linspace(0,2*math.pi,12,endpoint=False): dirs.append(_unit(math.cos(a)*st['direction']+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v)))
                local=[]
                for d in dirs:
                    r=self._propose(st['point'],d,False)
                    if r is None: continue
                    if r['sep']<max(1.8,st['seps'][-1]-.35): continue
                    local.append(r)
                if not local and step_i<8:
                    u,v=_orth_basis(st['direction'])
                    for ang in (52,68):
                        a=math.radians(ang)
                        for ph in np.linspace(0,2*math.pi,14,endpoint=False):
                            d=_unit(math.cos(a)*st['direction']+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v)); r=self._propose(st['point'],d,True)
                            if r is not None and r['sep']>=max(1.5,st['seps'][-1]-.5): local.append(r)
                for r in local:
                    smooth=float(np.clip(np.dot(st['direction'],r['direction']),-1,1)); sep_gain=float(r['sep']-st['seps'][-1]); obj=.58*r['score']+.14*float(r['hard'])+.12*((smooth+1)/2)+.10*min(r['sep']/5.,1)+.06*min(max(sep_gain+.15,0)/.5,1)
                    props.append({"point":r['point'],"direction":r['direction'],"ext":st['ext']+[r['point'].copy()],"scores":st['scores']+[r['score']],"hard":st['hard']+[r['hard']],"seps":st['seps']+[r['sep']],"obj":st['obj']+obj})
            if not props: break
            def rank(st):
                ext=np.asarray(st['ext']); L=float(arc_mm(ext,self.spacing)[-1]); return .42*np.mean(st['scores'])+.18*np.mean(st['hard'])+.22*min(L/18.,1)+.18*min(st['seps'][-1]/5.,1)
            props.sort(key=rank,reverse=True); keep=[]
            for st in props:
                if all(np.linalg.norm((st['point']-q['point'])*self.spacing)>=.55 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            pool.extend(keep); beams=keep
        rows=[]; evaluated=[]
        for st in pool:
            ext=np.asarray(st['ext'],float); extra=float(arc_mm(ext,self.spacing)[-1])
            if extra<4.: continue
            combined=np.vstack([seed[:-1],ext]); qdf,qsum=_serial_qc(combined,self.ct,self.spacing,self.rca,.85); total=qsum['length_mm']; sep_end=float(st['seps'][-1]); sep_med=float(np.median(st['seps'])); score=.28*min(extra/15.,1)+.28*qsum['plane_pass_fraction']+.22*qsum['median_plane_score']+.12*min(sep_end/5.,1)+.10*min(sep_med/4.,1)
            row={"candidate":len(rows)+1,"extra_length_mm":extra,"total_length_mm":total,"plane_pass_fraction":qsum['plane_pass_fraction'],"soft_pass_fraction":qsum['soft_pass_fraction'],"median_plane_score":qsum['median_plane_score'],"median_radius_mm":qsum['median_radius_mm'],"endpoint_reference_separation_mm":sep_end,"median_extension_reference_separation_mm":sep_med,"rank_score":score}; rows.append(row); evaluated.append((score,row,qsum,qdf,combined,st))
        if not evaluated: raise RuntimeError('No coronary-like extension persisted beyond 4 mm')
        evaluated.sort(key=lambda x:x[0],reverse=True); score,row,qsum,qdf,combined,st=evaluated[0]; self.combined=resample_path(combined,self.spacing,.30); self.qc=qdf; self.candidates=pd.DataFrame(rows).sort_values('rank_score',ascending=False)
        explicit_aff,key=_explicit_affine(self.meta or {}); orient={"patient_coordinate_status":"UNAVAILABLE","affine_key":None}
        if explicit_aff is not None:
            lps=_to_lps(self.combined,explicit_aff); j=int(np.argmin(abs(arc_mm(self.combined,self.spacing)-min(3.,arc_mm(self.combined,self.spacing)[-1])))); delta=lps[-1]-lps[j]
            orient={"patient_coordinate_status":"AVAILABLE","affine_key":key,"delta_left_mm":float(delta[0]),"delta_posterior_mm":float(delta[1]),"delta_superior_mm":float(delta[2]),"leftward_component":bool(delta[0]>0),"posterior_component":bool(delta[1]>-2.)}
        continuity=bool(row['total_length_mm']>=20 and row['plane_pass_fraction']>=.80 and row['soft_pass_fraction']>=.88 and row['median_plane_score']>=.80)
        independence=bool(row['endpoint_reference_separation_mm']>=3.5 and row['median_extension_reference_separation_mm']>=2.4)
        if orient['patient_coordinate_status']=='AVAILABLE': orientation_support=bool(orient['leftward_component'] and orient['posterior_component'])
        else: orientation_support=None
        if continuity and independence and orientation_support is True: status='LCX_COMPATIBLE_COURSE_SUPPORTED'
        elif continuity and independence: status='LCX_TOPOLOGY_SUPPORTED_ORIENTATION_UNVERIFIED'
        else: status='COURSE_REQUIRES_REVIEW'
        self.summary={**qsum,**row,**orient,"continuity_gate":continuity,"independence_gate":independence,"orientation_gate":orientation_support,"status":status,"accepted_topology":bool(continuity and independence),"interpretation":"Accepted secondary branch continues as a distinct coronary-like course from the supported left-main bifurcation. LCX label requires patient-coordinate compatibility or expert review."}
        pd.DataFrame({"arc_mm":arc_mm(self.combined,self.spacing),"z":self.combined[:,0],"y":self.combined[:,1],"x":self.combined[:,2]}).to_csv(pf,index=False); self.qc.to_csv(qf,index=False); self.candidates.to_csv(cf,index=False); _write_json(self.summary,sf); self._record('course_extension','recomputed_and_cached',sf,f"status={status}"); return self.summary
    def make_figures(self):
        done=self.cache/'figures.done'
        if self.reuse['figures'] and done.exists() and all((self.out/f).exists() for f in ('01_course_cross_sections.png','02_course_vs_lad_mips.png','03_course_profiles.png')): self._record('figures','reused',done); return
        if self.summary is None: self.extend_course()
        p=self.combined; s=arc_mm(p,self.spacing); xs=np.linspace(.4,s[-1],12); fig,axs=plt.subplots(3,4,figsize=(14,10)); axs=axs.ravel()
        for ax,x in zip(axs,xs):
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing); ax.imshow(im,cmap='gray',vmin=-100,vmax=850,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,lw=.5); ax.axvline(0,lw=.5); ax.set_title(f'{s[i]:.1f} mm'); ax.axis('off')
        fig.suptitle('Secondary coronary course — orthogonal source-CCTA planes'); fig.tight_layout(); fig.savefig(self.out/'01_course_cross_sections.png',dpi=150); plt.close(fig)
        ref=np.vstack([self.reference,self.trunk]); fig,axs=plt.subplots(1,3,figsize=(16,5))
        for ax,(a,b,name) in zip(axs,[(1,2,'axial projection'),(0,2,'coronal projection'),(0,1,'sagittal projection')]):
            ax.plot(ref[:,b],ref[:,a],lw=1,label='LAD/trunk reference'); ax.plot(p[:,b],p[:,a],lw=2,label='secondary course'); ax.scatter([p[0,b]],[p[0,a]],s=30,label='bifurcation seed'); ax.set_title(name); ax.set_aspect('equal'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(self.out/'02_course_vs_lad_mips.png',dpi=150); plt.close(fig)
        sep=np.array([_nearest_dist(q,ref,self.spacing) for q in p]); fig,ax=plt.subplots(figsize=(9,4)); ax.plot(s,sep); ax.axhline(3.5,ls='--',lw=1); ax.set_xlabel('arc length (mm)'); ax.set_ylabel('distance to LAD/trunk reference (mm)'); ax.set_title('Sustained independence of secondary course'); fig.tight_layout(); fig.savefig(self.out/'03_course_profiles.png',dpi=150); plt.close(fig)
        done.write_text('ok'); self._record('figures','recomputed_and_cached',done)
    def make_report(self):
        done=self.cache/'report.done'; html=self.out/'OPENPLAQUE_LCX_COURSE_CONFIRMATION_REPORT.html'
        if self.reuse['report'] and done.exists() and html.exists(): self._record('report','reused',html); return html
        if self.summary is None: self.extend_course()
        if not (self.out/'01_course_cross_sections.png').exists(): self.make_figures()
        _write_json(self.summary,self.out/'course_summary.json');
        pd.DataFrame({"arc_mm":arc_mm(self.combined,self.spacing),"z":self.combined[:,0],"y":self.combined[:,1],"x":self.combined[:,2]}).to_csv(self.out/'combined_branch_centerline.csv',index=False); self.qc.to_csv(self.out/'combined_branch_qc.csv',index=False); self.candidates.to_csv(self.out/'course_candidates.csv',index=False)
        rows=''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in self.summary.items() if not isinstance(v,(dict,list)))
        body=f'''<html><body><h1>OpenPlaque LCX Course Confirmation</h1><p>Algorithm: {ALGORITHM_VERSION}</p><p><b>Status: {self.summary.get('status')}</b></p><table border="1" cellspacing="0" cellpadding="4">{rows}</table><h2>Orthogonal source-CCTA planes</h2><img src="01_course_cross_sections.png" width="95%"><h2>Course versus frozen LAD/trunk</h2><img src="02_course_vs_lad_mips.png" width="95%"><h2>Independence profile</h2><img src="03_course_profiles.png" width="80%"><p>Research use only. The workflow does not automatically assign an LCX label.</p></body></html>'''; html.write_text(body); done.write_text('ok'); self._record('report','recomputed_and_cached',html); return html
    def package(self):
        self.make_report(); z=self.out/'OPENPLAQUE_LCX_COURSE_CONFIRMATION_REPORT_BACK.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
            for p in self.out.iterdir():
                if p.is_file() and p!=z: q.write(p,p.name)
        return z
