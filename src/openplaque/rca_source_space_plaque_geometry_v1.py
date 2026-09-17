from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

ARC_STEP_MM=.50; N_ANGLES=72; RADIAL_STEP_MM=.10
LUMEN_MAX_RADIUS_MM=4.0; LUMEN_MIN_RADIUS_MM=.55
SHELL_THICKNESSES_MM=(.75,1.0,1.25,1.5); NOMINAL_SHELL_MM=1.0


def arc(p):
    p=np.asarray(p,float)
    return np.zeros(len(p)) if len(p)<=1 else np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def interp(p,q):
    a=arc(p); q=np.asarray(q,float); p=np.asarray(p,float)
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])

def resample(p,step=ARC_STEP_MM):
    a=arc(p); q=np.arange(0.,float(a[-1])+1e-9,step)
    if len(q)==0 or q[-1]<a[-1]-1e-6: q=np.r_[q,a[-1]]
    return interp(p,q),q

def unit(v):
    v=np.asarray(v,float); n=float(np.linalg.norm(v)); return v/n if n>1e-9 else np.zeros_like(v)

def orth_basis(t):
    t=unit(t); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]
    u=unit(np.cross(t,seed)); v=unit(np.cross(t,u)); return u,v

def arc_weights(q):
    q=np.asarray(q,float)
    if len(q)==1:return np.ones(1)
    w=np.empty(len(q)); w[0]=.5*(q[1]-q[0]); w[-1]=.5*(q[-1]-q[-2])
    if len(q)>2:w[1:-1]=.5*(q[2:]-q[:-2])
    return w

def load_path(path):
    d=pd.read_csv(path)
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"),("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS columns in {path}: {list(d.columns)}")

class SourceGeometry:
    def __init__(self,meta):
        self.spacing_zyx=np.asarray(meta["spacing_zyx"],float); self.spacing_xyz=self.spacing_zyx[::-1]
        self.origin=np.asarray(meta["positions_lps_mm"][0],float)
        iop=np.asarray(meta["image_orientation_patient"],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
        self.D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
        self.invD=np.linalg.inv(self.D)
    def xyz_to_zyx(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float)); xyz=((pts-self.origin)@self.invD.T)/self.spacing_xyz
        return xyz[:,::-1]

def load_source(cache):
    cache=Path(cache); src=np.load(cache/"series7_int16.npy",mmap_mode="r")
    meta=json.loads((cache/"series7_int16.json").read_text()); geom=SourceGeometry(meta)
    vv=float(np.prod(geom.spacing_zyx))
    if not(.001<vv<.5): raise RuntimeError(f"Unexpected Series-7 voxel volume {vv}")
    return geom,src,vv

def sample(geom,src,pts,cval=-1024.):
    return map_coordinates(np.asarray(src),geom.xyz_to_zyx(pts).T,order=1,mode="constant",cval=float(cval))

def _cmedian(x):
    return np.median(np.stack([np.roll(x,k) for k in range(-2,3)]),axis=0)

def lumen(geom,src,c,t):
    u,v=orth_basis(t); th=np.linspace(0,2*np.pi,N_ANGLES,endpoint=False)
    dirs=np.cos(th)[:,None]*u+np.sin(th)[:,None]*v
    center_pts=np.vstack([c,c+.15*u,c-.15*u,c+.15*v,c-.15*v]); center_hu=float(np.median(sample(geom,src,center_pts)))
    threshold=float(np.clip(.55*center_hu,220.,500.)); rr=np.arange(.2,LUMEN_MAX_RADIUS_MM+1e-9,RADIAL_STEP_MM)
    P=c[None,None,:]+dirs[:,None,:]*rr[None,:,None]; hu=sample(geom,src,P.reshape(-1,3)).reshape(N_ANGLES,len(rr))
    r=np.full(N_ANGLES,np.nan)
    for i in range(N_ANGLES):
        b=(hu[i]<threshold)&(rr>=LUMEN_MIN_RADIUS_MM); ix=np.flatnonzero(b[:-1]&b[1:])
        if len(ix): r[i]=rr[int(ix[0])]
    valid=np.isfinite(r); vf=float(valid.mean())
    if valid.any():
        med=float(np.median(r[valid])); r[~valid]=med; r=_cmedian(r); r=np.clip(r,med-.75,med+.75); r=np.clip(r,.55,4.)
        p10,p50,p90=np.percentile(r,[10,50,90]); axis=float(p90/max(p10,1e-6)); area=float(.5*np.sum(r*r)*(2*np.pi/N_ANGLES))
    else:p10=p50=p90=axis=area=np.nan
    qc=bool(center_hu>=200 and vf>=.60 and np.isfinite(axis) and axis<=2.5)
    return dict(center_hu=center_hu,lumen_threshold_hu=threshold,valid_radial_fraction=vf,lumen_radius_p10_mm=float(p10),
                lumen_radius_median_mm=float(p50),lumen_radius_p90_mm=float(p90),lumen_axis_proxy=float(axis),
                lumen_area_mm2=float(area),station_qc_pass=qc,radii=r,theta=th,u=u,v=v)

def integrate_shell(geom,src,c,L,shell,ds):
    th=L["theta"]; dirs=np.cos(th)[:,None]*L["u"]+np.sin(th)[:,None]*L["v"]
    off=np.arange(RADIAL_STEP_MM/2.,shell,RADIAL_STEP_MM); off=off if len(off) else np.array([shell/2])
    r=L["radii"][:,None]+off[None,:]; P=c[None,None,:]+dirs[:,None,:]*r[...,None]
    hu=sample(geom,src,P.reshape(-1,3)).reshape(r.shape); area=r*RADIAL_STEP_MM*(2*np.pi/N_ANGLES); vol=area*float(ds)
    finite=np.isfinite(hu); masks={"fatlike_excluded_mm3":finite&(hu<-30),"low_attenuation_mm3":finite&(hu>=-30)&(hu<30),
        "noncalcified_mm3":finite&(hu>=30)&(hu<130),"mixed_intermediate_mm3":finite&(hu>=130)&(hu<350),"calcified_mm3":finite&(hu>=350)}
    vals={k:float(vol[m].sum()) for k,m in masks.items()}; vals["low_attenuation_raw_lt30_mm3"]=float(vol[finite&(hu<30)].sum())
    vals["shell_volume_mm3"]=float(vol[finite].sum()); vals["total_plaque_proxy_mm3"]=sum(vals[k] for k in ("low_attenuation_mm3","noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3"))
    sv=vals["shell_volume_mm3"]; pv=vals["total_plaque_proxy_mm3"]; vessel=pv+float(L["lumen_area_mm2"]*ds)
    vals["fatlike_fraction"]=vals["fatlike_excluded_mm3"]/sv if sv>0 else np.nan
    vals["plaque_proxy_fraction_of_shell"]=pv/sv if sv>0 else np.nan; vals["plaque_burden_proxy"]=pv/vessel if vessel>0 else np.nan
    vals["shell_mean_hu"]=float(np.average(hu[finite],weights=area[finite])) if finite.any() else np.nan
    return vals

def profile_1mm(stations,nominal=NOMINAL_SHELL_MM):
    d=stations[np.isclose(stations.shell_thickness_mm,nominal)].copy(); rows=[]
    if d.empty:return pd.DataFrame()
    cols=["low_attenuation_mm3","noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3","total_plaque_proxy_mm3","fatlike_excluded_mm3"]
    for a0 in np.arange(0.,math.floor(float(d.arc_mm.max()))+1.,1.):
        x=d[(d.arc_mm>=a0)&(d.arc_mm<a0+1.)]
        if x.empty:continue
        r={"arc_start_mm":float(a0),"arc_end_mm":float(a0+1),"station_count":len(x),"station_qc_fraction":float(x.station_qc_pass.mean()),
           "mean_lumen_radius_mm":float(x.lumen_radius_median_mm.mean()),"mean_plaque_burden_proxy":float(x.plaque_burden_proxy.mean())}
        for c in cols:r[c]=float(x[c].sum())
        rows.append(r)
    return pd.DataFrame(rows)

def plane_image(geom,src,c,t,half=5.,step=.15):
    u,v=orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing="ij")
    P=c+xx[...,None]*u+yy[...,None]*v; im=sample(geom,src,P.reshape(-1,3)).reshape(len(q),len(q)); return im,q

def synthetic_self_test():
    th=np.linspace(0,2*np.pi,N_ANGLES,endpoint=False); r=np.full(N_ANGLES,1.5); area=.5*np.sum(r*r)*(2*np.pi/N_ANGLES)
    assert abs(area-np.pi*1.5**2)<1e-10; assert abs(arc_weights(np.array([0.,.5,1.])).sum()-1.)<1e-12
    return {"ok":True,"circle_area_mm2":float(area)}
