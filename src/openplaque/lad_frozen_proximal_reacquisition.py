from __future__ import annotations

"""Target-free proximal reacquisition from the independently frozen LAD segment.

Starts only from the proximal endpoint and tangent of the accepted frozen LAD.
No RCA, prior trunk, or secondary-branch coordinates are loaded or used.
The aorta is used only as a stop/anatomical constraint.
"""

import gc,json,math,shutil,zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from .study import OpenPlaqueStudy

ALGORITHM_VERSION='lad-frozen-proximal-reacquisition-v1.0'
SOURCE_SERIES=7

def _json_write(obj,path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def _json_read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _unit(v):
 v=np.asarray(v,float);return v/max(float(np.linalg.norm(v)),1e-12)
def arc_mm(path,spacing):
 p=np.asarray(path,float)
 if len(p)<2:return np.zeros(len(p),float)
 d=np.diff(p,axis=0)*np.asarray(spacing,float)[None,:];return np.r_[0.,np.cumsum(np.linalg.norm(d,axis=1))]
def resample_path(path,spacing,step=.35):
 p=np.asarray(path,float)
 if len(p)<2:return p.copy()
 s=arc_mm(p,spacing);keep=np.r_[True,np.diff(s)>1e-6];p,s=p[keep],s[keep]
 q=np.arange(0.,s[-1]+.5*step,step)
 if len(q)==0 or q[-1]<s[-1]-.1:q=np.r_[q,s[-1]]
 q[-1]=min(q[-1],s[-1]);return np.column_stack([np.interp(q,s,p[:,j]) for j in range(3)])
def source_zyx_to_lps(meta,zyx):
 p=np.atleast_2d(np.asarray(zyx,float));pos=np.asarray(meta['positions_lps_mm'],float);o=np.asarray(meta['image_orientation_patient'],float);row,col=o[:3],o[3:];sp=np.asarray(meta['spacing_zyx'],float)
 sv=(pos[-1]-pos[0])/max(len(pos)-1,1) if len(pos)>1 else np.cross(row,col)*sp[0]
 out=pos[0][None,:]+p[:,0,None]*sv[None,:];out+=p[:,2,None]*sp[2]*row[None,:];out+=p[:,1,None]*sp[1]*col[None,:];return out
def _orth_basis(t):
 t=_unit(t);ref=np.array([1.,0.,0.]) if abs(t[0])<.82 else np.array([0.,1.,0.]);u=_unit(np.cross(t,ref));v=_unit(np.cross(t,u));return u,v

def _series7_files(root,extract_root='/content/full_dicom_frozen_lad_proximal'):
 root=Path(root);src=root/'Full_DICOM.zip'
 if not src.exists():raise FileNotFoundError(src)
 local=Path('/content/Full_DICOM_lad_proximal.zip')
 if not local.exists() or local.stat().st_size!=src.stat().st_size:shutil.copyfile(src,local)
 shutil.rmtree(extract_root,ignore_errors=True);study=OpenPlaqueStudy(str(local),extract_root=extract_root);m=[s for s in study.series if s.get('series_number')==SOURCE_SERIES]
 if not m:raise RuntimeError('Source CCTA series 7 not found')
 r=sitk.ImageSeriesReader();files=list(r.GetGDCMSeriesFileNames(m[0]['folder'],m[0]['uid']))
 if not files:raise RuntimeError('No DICOM files for source CCTA series 7')
 return files
def _geometry_from_dicom(files):
 ds0=pydicom.dcmread(files[0],stop_before_pixels=True,force=True);positions=[]
 for fp in files:positions.append([float(x) for x in pydicom.dcmread(fp,stop_before_pixels=True,force=True).ImagePositionPatient])
 ps=[float(x) for x in ds0.PixelSpacing];pos=np.asarray(positions,float);dz=float(np.median(np.linalg.norm(np.diff(pos,axis=0),axis=1))) if len(pos)>1 else float(getattr(ds0,'SliceThickness',1.))
 return {'shape':[len(files),int(ds0.Rows),int(ds0.Columns)],'spacing_zyx':[dz,ps[0],ps[1]],'positions_lps_mm':positions,'image_orientation_patient':[float(x) for x in ds0.ImageOrientationPatient],'source':'Full_DICOM.zip series 7'}
def _load_or_build_source_ct(root,cache):
 root,cache=Path(root),Path(cache);cache.mkdir(parents=True,exist_ok=True);own=cache/'series7_int16.npy';meta_fp=cache/'series7_geometry.json'
 if own.exists() and meta_fp.exists():
  meta=_json_read(meta_fp);mm=np.load(own,mmap_mode='r')
  if tuple(meta.get('shape',[]))==tuple(mm.shape) and 'positions_lps_mm' in meta:return mm,meta,'reused_own_cache'
 files=_series7_files(root);meta=_geometry_from_dicom(files);expected=tuple(meta['shape'])
 for fp in [root/'Cache'/'Coronary_Anatomy_Reconciliation_v1'/'series7_int16.npy',root/'Cache'/'LAD_Confirmed_Backtrack_v1'/'series7_int16.npy',root/'Cache'/'LAD_Origin_Backtrack_v1'/'series7_int16.npy']:
  if not fp.exists():continue
  try:mm=np.load(fp,mmap_mode='r')
  except Exception:continue
  if tuple(mm.shape)==expected:
   meta['ct_cache_source']=str(fp);_json_write(meta,meta_fp);return mm,meta,f'reused_shape_verified_cache:{fp}'
 mm=np.lib.format.open_memmap(own,mode='w+',dtype=np.int16,shape=expected)
 for i,fp in enumerate(files):
  ds=pydicom.dcmread(fp,force=True);arr=ds.pixel_array.astype(np.float32,copy=False);hu=arr*float(getattr(ds,'RescaleSlope',1.))+float(getattr(ds,'RescaleIntercept',0.));mm[i]=np.rint(np.clip(hu,-32768,32767)).astype(np.int16)
  if i%100==0:mm.flush()
 mm.flush();del mm;gc.collect();_json_write(meta,meta_fp);return np.load(own,mmap_mode='r'),meta,'rebuilt_from_full_dicom'

def orthogonal_plane(ct,p,t,spacing,half=5.,pix=.16):
 u,v=_orth_basis(t);g=np.arange(-half,half+1e-9,pix,dtype=np.float32);V,U=np.meshgrid(g,g,indexing='ij');sp=np.asarray(spacing,np.float32);c=np.asarray(p,np.float32)*sp;pos=c[None,None,:]+U[...,None]*u.astype(np.float32)+V[...,None]*v.astype(np.float32);vox=pos/sp;out=np.empty(U.shape,np.float32);map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=out,order=1,mode='nearest',prefilter=False);return out,g,u,v
def lumen_metrics(im,g,min_hu=180.,max_hu=1200.):
 pix=float(abs(g[1]-g[0]));bright=(im>=min_hu)&(im<=max_hu);lab,_=ndi.label(bright,structure=np.ones((3,3),np.uint8));cy=cx=len(g)//2;chosen=int(lab[cy,cx])
 if chosen==0:
  yy,xx=np.nonzero(bright)
  if len(yy):
   d=np.hypot(g[yy],g[xx]);j=int(np.argmin(d));chosen=int(lab[yy[j],xx[j]]) if float(d[j])<=.9 else 0
 if chosen<=0:return {'radius_mm':np.nan,'centroid_shift_mm':np.nan,'centroid_u_mm':np.nan,'centroid_v_mm':np.nan,'circularity':np.nan,'center_hu':float(im[cy,cx]),'component_median_hu':np.nan}
 comp=lab==chosen;yy,xx=np.nonzero(comp);area=float(comp.sum())*pix*pix;r=math.sqrt(area/math.pi);cu,cv=float(np.mean(g[xx])),float(np.mean(g[yy]));shift=float(math.hypot(cu,cv));er=ndi.binary_erosion(comp,structure=np.ones((3,3),bool));per=max(float(np.logical_and(comp,~er).sum())*pix,pix);circ=float(np.clip(4*math.pi*area/(per*per),0,1.2))
 return {'radius_mm':r,'centroid_shift_mm':shift,'centroid_u_mm':cu,'centroid_v_mm':cv,'circularity':circ,'center_hu':float(im[cy,cx]),'component_median_hu':float(np.median(im[comp]))}
def score_metrics(m,ref):
 rr,rh=float(ref['median_radius_mm']),float(ref['median_center_hu']);r,sh,circ,med=m['radius_mm'],m['centroid_shift_mm'],m['circularity'],m['component_median_hu'];rs=math.exp(-.5*((r-rr)/max(.48*rr,.60))**2) if np.isfinite(r) else 0.;ss=math.exp(-.5*(sh/.62)**2) if np.isfinite(sh) else 0.;cs=min(1.,max(0.,circ/.50)) if np.isfinite(circ) else 0.;hs=math.exp(-.5*((med-rh)/320.)**2) if np.isfinite(med) else 0.;score=.38*rs+.30*ss+.18*cs+.14*hs
 hard=bool(np.isfinite(r) and .55*rr<=r<=min(3.05,1.70*rr+.35) and np.isfinite(sh) and sh<=.90 and np.isfinite(circ) and circ>=.20 and np.isfinite(med) and max(150.,.34*rh)<=med<=1200. and score>=.60);return float(score),hard
def _cone_dirs(base,max_angle=24.,n_az=10):
 base=_unit(base);u,v=_orth_basis(base);dirs=[base]
 for ad in (8.,16.,float(max_angle)):
  a=math.radians(ad)
  for k in range(n_az):
   ph=2*math.pi*k/n_az;dirs.append(_unit(math.cos(a)*base+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v)))
 return dirs
def _turn(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a),_unit(b)),-1.,1.))))

class LADFrozenProximalReacquisitionWorkflow:
 COMPONENTS=('source_ct','frozen_lad','aorta','calibration','tracking','figures','report')
 def __init__(self,root='/content/drive/MyDrive/OpenPlaque',reuse=None):
  self.root=Path(root);self.cache=self.root/'Cache'/'LAD_Frozen_Proximal_Reacquisition_v1';self.out=self.root/'LAD_Frozen_Proximal_Reacquisition_Report';self.cache.mkdir(parents=True,exist_ok=True);self.out.mkdir(parents=True,exist_ok=True);self.reuse={k:True for k in self.COMPONENTS};self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()});self.ct=self.meta=self.spacing=self.frozen=self.aorta=self.aorta_dist=self.ref=self.track=self.track_qc=self.terminal=self.summary=None;self.prov=[]
 def _record(self,c,a,p='',note=''):
  self.prov.append({'component':c,'reuse_requested':self.reuse[c],'action':a,'path':str(p),'note':note});pd.DataFrame(self.prov).to_csv(self.out/'cache_provenance.csv',index=False)
 def cache_status(self):
  names={'source_ct':'series7_geometry.json','frozen_lad':'frozen_lad_snapshot.csv','aorta':'aorta_distance.npz','calibration':'lad_lumen_calibration.json','tracking':'tracking_summary.json','figures':'figures.done','report':'report.done'};return pd.DataFrame([{'component':k,'reuse':self.reuse[k],'cache_exists':(self.cache/v).exists()} for k,v in names.items()])
 def load_source_ct(self):
  self.ct,self.meta,act=_load_or_build_source_ct(self.root,self.cache);self.spacing=np.asarray(self.meta['spacing_zyx'],float);self._record('source_ct',act,self.cache/'series7_geometry.json');return self.ct
 def load_frozen_lad(self):
  if self.spacing is None:self.load_source_ct()
  fp=self.root/'Cache'/'LAD_Confirmed_Backtrack_v1'/'frozen_lad_centerline.csv';sf=self.root/'Cache'/'LAD_Confirmed_Backtrack_v1'/'frozen_lad_summary.json'
  if not fp.exists():fp=self.root/'LAD_Confirmed_Backtrack_Report'/'frozen_lad_centerline.csv'
  if not fp.exists():raise FileNotFoundError('Validated frozen LAD centerline not found')
  if sf.exists() and _json_read(sf).get('accepted') is False:raise RuntimeError('Prior frozen LAD summary explicitly marks segment rejected')
  df=pd.read_csv(fp)
  if not {'z','y','x'}.issubset(df.columns):raise ValueError('Frozen LAD CSV lacks z,y,x')
  self.frozen=resample_path(df[['z','y','x']].to_numpy(float),self.spacing,.35);s=arc_mm(self.frozen,self.spacing);pd.DataFrame({'arc_mm':s,'z':self.frozen[:,0],'y':self.frozen[:,1],'x':self.frozen[:,2]}).to_csv(self.cache/'frozen_lad_snapshot.csv',index=False);self._record('frozen_lad','imported_explicit_accepted_artifact',fp,f'length={s[-1]:.3f} mm');return self.frozen
 def load_aorta(self):
  if self.ct is None:self.load_source_ct()
  fp=self.root/'RCA_Ostium_TotalSegmentator'/'aorta_series7_totalseg.nii.gz'
  if not fp.exists():raise FileNotFoundError(fp)
  a=sitk.GetArrayFromImage(sitk.ReadImage(str(fp))).astype(bool)
  if tuple(a.shape)!=tuple(self.ct.shape):raise RuntimeError(f'Aorta mask shape {a.shape} != source CT {self.ct.shape}')
  self.aorta=a;self.aorta_dist=ndi.distance_transform_edt(~a,sampling=self.spacing).astype(np.float32);np.savez_compressed(self.cache/'aorta_distance.npz',aorta=a.astype(np.uint8),distance=self.aorta_dist);self._record('aorta','loaded_stop_constraint',fp,'Not used in proposal direction or objective');return self.aorta_dist
 def _aorta_state(self,p):
  q=np.asarray(p,float)
  if np.any(q<0) or np.any(q>=np.asarray(self.aorta.shape)-1):return float('inf'),False
  d=float(map_coordinates(self.aorta_dist,q[:,None],order=1,mode='nearest',prefilter=False)[0]);qi=np.clip(np.rint(q).astype(int),0,np.asarray(self.aorta.shape)-1);return d,bool(self.aorta[tuple(qi)])
 def calibrate_from_frozen_lad(self,end_mm=8.):
  if self.ct is None:self.load_source_ct()
  if self.frozen is None:self.load_frozen_lad()
  p=self.frozen;s=arc_mm(p,self.spacing);rows=[]
  for ss in np.arange(.6,min(float(end_mm),float(s[-1])-.4)+1e-6,.8):
   i=int(np.argmin(abs(s-ss)));t=_unit((p[min(len(p)-1,i+4)]-p[max(0,i-4)])*self.spacing);im,g,_,_=orthogonal_plane(self.ct,p[i],t,self.spacing);rows.append({'arc_mm':float(s[i]),**lumen_metrics(im,g)})
  q=pd.DataFrame(rows);good=q[np.isfinite(q.radius_mm)&np.isfinite(q.component_median_hu)&(q.centroid_shift_mm<=1.)]
  if len(good)<4:raise RuntimeError('Could not calibrate proximal tracking from frozen LAD')
  self.ref={'median_radius_mm':float(good.radius_mm.median()),'median_center_hu':float(good.component_median_hu.median()),'median_shift_mm':float(good.centroid_shift_mm.median()),'median_circularity':float(good.circularity.median()),'n_planes':int(len(good)),'source':'first ~8 mm of independently frozen LAD'};q.to_csv(self.cache/'frozen_lad_calibration_planes.csv',index=False);_json_write(self.ref,self.cache/'lad_lumen_calibration.json');self._record('calibration','calibrated_from_frozen_lad',self.cache/'lad_lumen_calibration.json');return self.ref
 def _eval(self,p,d,max_recenter=.80):
  im,g,u,v=orthogonal_plane(self.ct,p,d,self.spacing);m0=lumen_metrics(im,g)
  if not np.isfinite(m0['centroid_shift_mm']):return None,'no_component'
  if m0['centroid_shift_mm']>max_recenter:return None,'recenter_too_large'
  shift=m0['centroid_u_mm']*u+m0['centroid_v_mm']*v;p2=np.asarray(p,float)+shift/self.spacing;im2,g2,_,_=orthogonal_plane(self.ct,p2,d,self.spacing);m=lumen_metrics(im2,g2);score,hard=score_metrics(m,self.ref)
  if not hard:
   hi=min(3.05,1.70*self.ref['median_radius_mm']+.35)
   why='no_component' if not np.isfinite(m['radius_mm']) else 'radius_too_large' if m['radius_mm']>hi else 'radius_too_small' if m['radius_mm']<.55*self.ref['median_radius_mm'] else 'off_center' if m['centroid_shift_mm']>.90 else 'low_circularity' if m['circularity']<.20 else 'low_score';return None,why
  return (p2,m,score,float(np.linalg.norm(shift))),None
 def run_tracking(self,step_mm=.65,max_length_mm=32.,beam_width=6,max_turn_deg=34.):
  if self.ct is None:self.load_source_ct()
  if self.frozen is None:self.load_frozen_lad()
  if self.aorta_dist is None:self.load_aorta()
  if self.ref is None:self.calibrate_from_frozen_lad()
  p=self.frozen;s=arc_mm(p,self.spacing);j=int(np.argmin(abs(s-min(3.,float(s[-1])))));init_dir=-_unit((p[j]-p[0])*self.spacing);seed=p[0].copy();sd,si=self._aorta_state(seed);beams=[{'point':seed,'direction':init_dir,'points':[seed.copy()],'rows':[],'objective':0.,'length_mm':0.,'terminal':None,'aorta_reached':bool(si or sd<=.75)}];completed=[];reject=[];shape=np.asarray(self.ct.shape,float)
  for step in range(int(math.ceil(max_length_mm/step_mm))):
   props=[]
   for bi,st in enumerate(beams):
    if st['aorta_reached']:completed.append(st);continue
    gen=0
    for di,d in enumerate(_cone_dirs(st['direction'],24.,10)):
     turn=_turn(st['direction'],d)
     if turn>max_turn_deg:reject.append({'step':step+1,'beam':bi,'dir_index':di,'reason':'turn_too_large','turn_deg':turn});continue
     raw=st['point']+(step_mm*d)/self.spacing
     if np.any(raw<2) or np.any(raw>=shape-3):reject.append({'step':step+1,'beam':bi,'dir_index':di,'reason':'out_of_bounds'});continue
     ev,why=self._eval(raw,d)
     if ev is None:reject.append({'step':step+1,'beam':bi,'dir_index':di,'reason':why,'turn_deg':turn});continue
     p2,m,score,recenter=ev;actual=float(np.linalg.norm((p2-st['point'])*self.spacing))
     if not .35<=actual<=1.05:reject.append({'step':step+1,'beam':bi,'dir_index':di,'reason':'step_length','actual_step_mm':actual});continue
     if len(st['points'])>5:
      hist=np.asarray(st['points'][:-4])*self.spacing
      if float(np.min(np.linalg.norm(hist-p2*self.spacing,axis=1)))<.9:reject.append({'step':step+1,'beam':bi,'dir_index':di,'reason':'loop'});continue
     ad,inside=self._aorta_state(p2);reached=bool(inside or ad<=.75);length=st['length_mm']+actual;row={'step_index':step+1,'track_length_mm':length,'z':float(p2[0]),'y':float(p2[1]),'x':float(p2[2]),'radius_mm':float(m['radius_mm']),'centroid_shift_mm':float(m['centroid_shift_mm']),'circularity':float(m['circularity']),'component_median_hu':float(m['component_median_hu']),'plane_score':float(score),'turn_deg':turn,'actual_step_mm':actual,'recenter_mm':recenter,'aorta_distance_mm':ad,'inside_aorta':inside};obj=st['objective']+score-.010*turn-.10*recenter;props.append({'point':p2,'direction':d,'points':st['points']+[p2.copy()],'rows':st['rows']+[row],'objective':obj,'length_mm':length,'terminal':'AORTA_REACHED' if reached else None,'aorta_reached':reached});gen+=1
    if gen==0:
     dead=dict(st);dead['terminal']=dead.get('terminal') or 'NO_COMPACT_LUMEN_PROPOSAL';completed.append(dead)
   if not props:break
   props.sort(key=lambda x:(x['aorta_reached'],x['length_mm'],x['objective']),reverse=True);beams=props[:beam_width]
   if any(x['aorta_reached'] for x in beams):completed.extend([x for x in beams if x['aorta_reached']]);break
  completed.extend(beams);completed=completed or [{'point':seed,'direction':init_dir,'points':[seed],'rows':[],'objective':0.,'length_mm':0.,'terminal':'NO_COMPACT_LUMEN_PROPOSAL','aorta_reached':False}];completed.sort(key=lambda x:(x['aorta_reached'],x['length_mm'],x['objective']),reverse=True);best=completed[0];pts=np.asarray(best['points']);self.track=pd.DataFrame({'track_length_mm':np.r_[0.,[r['track_length_mm'] for r in best['rows']]],'z':pts[:,0],'y':pts[:,1],'x':pts[:,2]});self.track_qc=pd.DataFrame(best['rows']);self.terminal=pd.DataFrame(reject)
  if len(self.track_qc):last_ad=float(self.track_qc.aorta_distance_mm.iloc[-1]);min_ad=float(self.track_qc.aorta_distance_mm.min());med_score=float(self.track_qc.plane_score.median());med_r=float(self.track_qc.radius_mm.median());L=float(best['length_mm'])
  else:last_ad=min_ad=sd;med_score=med_r=float('nan');L=0.
  if best['aorta_reached'] and L>=3.:status='PROXIMAL_LAD_COMPACT_CONTINUATION_REACHES_AORTA';accepted=True
  elif L>=3. and min_ad<=2.:status='PROXIMAL_LAD_CONTINUATION_REACHES_OSTIUM_NEIGHBORHOOD';accepted=True
  elif L>=3.:status='PROXIMAL_LAD_EXTENSION_SUPPORTED_AORTA_NOT_REACHED';accepted=True
  else:status='NO_PROXIMAL_COMPACT_LUMEN_CONTINUATION';accepted=False
  combined=np.vstack([pts[::-1][:-1],self.frozen]);cs=arc_mm(combined,self.spacing);lps=source_zyx_to_lps(self.meta,combined);self.summary={'algorithm':ALGORITHM_VERSION,'status':status,'accepted_proximal_extension':accepted,'target_free':True,'used_rca_or_prior_trunk_as_target':False,'aorta_used_only_as_stop_constraint':True,'frozen_lad_length_mm':float(arc_mm(self.frozen,self.spacing)[-1]),'new_proximal_track_length_mm':L,'combined_lad_length_mm':float(cs[-1]),'seed_aorta_distance_mm':float(sd),'terminal_aorta_distance_mm':last_ad,'minimum_aorta_distance_mm':min_ad,'aorta_reached':bool(best['aorta_reached']),'median_track_plane_score':med_score,'median_track_radius_mm':med_r,'terminal_reason':best.get('terminal'),'calibration':self.ref};self.track.to_csv(self.cache/'proximal_track_centerline.csv',index=False);self.track_qc.to_csv(self.cache/'proximal_track_qc.csv',index=False);self.terminal.to_csv(self.cache/'proposal_rejections.csv',index=False);pd.DataFrame({'arc_mm':cs,'z':combined[:,0],'y':combined[:,1],'x':combined[:,2],'lps_x_mm':lps[:,0],'lps_y_mm':lps[:,1],'lps_z_mm':lps[:,2]}).to_csv(self.cache/'combined_lad_centerline.csv',index=False);_json_write(self.summary,self.cache/'tracking_summary.json');self._record('tracking','computed_target_free',self.cache/'tracking_summary.json',status);return self.summary
 def make_figures(self):
  if self.summary is None:self.run_tracking()
  names=[];track=self.track[['z','y','x']].to_numpy(float);f_lps=source_zyx_to_lps(self.meta,self.frozen);t_lps=source_zyx_to_lps(self.meta,track)
  fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d');ax.plot(f_lps[:,0],f_lps[:,1],f_lps[:,2],lw=3,label='Frozen LAD');ax.plot(t_lps[:,0],t_lps[:,1],t_lps[:,2],lw=3,label='New proximal track');pts=np.argwhere(self.aorta);pts=pts[::max(1,len(pts)//5000)] if len(pts)>5000 else pts;alps=source_zyx_to_lps(self.meta,pts);ax.scatter(alps[:,0],alps[:,1],alps[:,2],s=1,alpha=.05,label='Aorta');ax.set_title('Frozen LAD proximal reacquisition — LPS');ax.legend();fp=self.out/'01_lps_geometry.png';fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name)
  arr=np.asarray(self.ct);fig,axs=plt.subplots(1,3,figsize=(16,5))
  for ax,axis,title in zip(axs,[0,1,2],['max z','max y','max x']):
   ax.imshow(np.max(arr,axis=axis),cmap='gray',vmin=-100,vmax=800,origin='lower')
   if axis==0:ax.plot(self.frozen[:,2],self.frozen[:,1],lw=2,label='Frozen LAD');ax.plot(track[:,2],track[:,1],lw=2,label='Proximal')
   elif axis==1:ax.plot(self.frozen[:,2],self.frozen[:,0],lw=2,label='Frozen LAD');ax.plot(track[:,2],track[:,0],lw=2,label='Proximal')
   else:ax.plot(self.frozen[:,1],self.frozen[:,0],lw=2,label='Frozen LAD');ax.plot(track[:,1],track[:,0],lw=2,label='Proximal')
   ax.set_title(title);ax.legend(fontsize=8)
  fp=self.out/'02_source_mips.png';fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name)
  fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
  if len(self.track_qc):
   x=self.track_qc.track_length_mm;axs[0].plot(x,self.track_qc.radius_mm,marker='o');axs[0].axhline(self.ref['median_radius_mm'],ls='--');axs[1].plot(x,self.track_qc.plane_score,marker='o');axs[1].axhline(.60,ls='--');axs[2].plot(x,self.track_qc.aorta_distance_mm,marker='o');axs[2].axhline(2.,ls='--');axs[2].axhline(.75,ls=':')
  axs[0].set_ylabel('Radius mm');axs[1].set_ylabel('Plane score');axs[2].set_ylabel('Aorta distance mm');axs[2].set_xlabel('New proximal track mm');fig.suptitle(self.summary['status']);fp=self.out/'03_longitudinal_qc.png';fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name)
  picks=sorted(set([0,len(track)//4,len(track)//2,(3*len(track))//4,len(track)-1]));cols=min(3,len(picks));rows=int(math.ceil(len(picks)/cols));fig,axs=plt.subplots(rows,cols,figsize=(4.3*cols,4.1*rows));axs=np.atleast_1d(axs).ravel()
  for ax,idx in zip(axs,picks):
   if idx==0:t=-_unit((self.frozen[min(len(self.frozen)-1,8)]-self.frozen[0])*self.spacing)
   else:a=max(0,idx-2);b=min(len(track)-1,idx+2);t=_unit((track[b]-track[a])*self.spacing) if b>a else -_unit((self.frozen[8]-self.frozen[0])*self.spacing)
   im,g,_,_=orthogonal_plane(self.ct,track[idx],t,self.spacing);m=lumen_metrics(im,g);sc,_=score_metrics(m,self.ref);ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[g[0],g[-1],g[0],g[-1]],origin='lower');tl=float(self.track.track_length_mm.iloc[idx]);ad=self._aorta_state(track[idx])[0];ax.set_title(f'{tl:.1f} mm | r {m["radius_mm"]:.2f} | score {sc:.2f}\naorta {ad:.2f} mm')
  for ax in axs[len(picks):]:ax.axis('off')
  fp=self.out/'04_informative_cross_sections.png';fig.tight_layout();fig.savefig(fp,dpi=170);plt.close(fig);names.append(fp.name);(self.cache/'figures.done').write_text('done');_json_write(names,self.out/'figure_manifest.json');self._record('figures','generated',self.out);return names
 def build_report(self):
  if self.summary is None:self.run_tracking()
  if not (self.cache/'figures.done').exists():self.make_figures()
  manifest=_json_read(self.out/'figure_manifest.json');tab=pd.DataFrame([{'status':self.summary['status'],'frozen_lad_mm':self.summary['frozen_lad_length_mm'],'new_proximal_mm':self.summary['new_proximal_track_length_mm'],'combined_mm':self.summary['combined_lad_length_mm'],'min_aorta_mm':self.summary['minimum_aorta_distance_mm'],'aorta_reached':self.summary['aorta_reached'],'median_score':self.summary['median_track_plane_score']}]).to_html(index=False,float_format=lambda x:f'{x:.3f}');q=self.track_qc.to_html(index=False,float_format=lambda x:f'{x:.3f}') if len(self.track_qc) else '<p>No accepted proximal steps.</p>';html=f"""<html><head><meta charset='utf-8'><title>OpenPlaque Frozen LAD Proximal Reacquisition</title><style>body{{font-family:Arial;max-width:1400px;margin:24px auto;padding:0 18px}}img{{max-width:100%;margin:8px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px;border-bottom:1px solid #ddd}}.note{{background:#eef5ff;padding:12px;border-left:4px solid #356aa0}}</style></head><body><h1>OpenPlaque — Frozen LAD Proximal Reacquisition</h1><div class='note'><b>Status: {self.summary['status']}</b><br>Starts only from the independently frozen LAD. No RCA, old trunk, or secondary centerline is loaded or used as a target. The aorta is only a stop constraint.</div><h2>Summary</h2>{tab}<h2>Accepted proximal tracking planes</h2>{q}<h2>Figures</h2>{''.join(f"<h3>{n}</h3><img src='{n}'>" for n in manifest)}</body></html>""";fp=self.out/'OPENPLAQUE_LAD_FROZEN_PROXIMAL_REACQUISITION_REPORT.html';fp.write_text(html,encoding='utf-8');(self.cache/'report.done').write_text('done');self._record('report','generated',fp);return fp
 def package(self):
  if self.summary is None:self.run_tracking()
  self.build_report();zp=self.out/'OPENPLAQUE_LAD_FROZEN_PROXIMAL_REACQUISITION_REPORT_BACK.zip';files=[self.cache/'tracking_summary.json',self.cache/'proximal_track_centerline.csv',self.cache/'proximal_track_qc.csv',self.cache/'proposal_rejections.csv',self.cache/'combined_lad_centerline.csv',self.cache/'lad_lumen_calibration.json',self.cache/'frozen_lad_calibration_planes.csv',self.cache/'frozen_lad_snapshot.csv',self.cache/'series7_geometry.json',self.out/'cache_provenance.csv',self.out/'OPENPLAQUE_LAD_FROZEN_PROXIMAL_REACQUISITION_REPORT.html']+[self.out/n for n in _json_read(self.out/'figure_manifest.json')]
  with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for fp in files:
    if Path(fp).exists():z.write(fp,arcname=Path(fp).name)
  return zp

def synthetic_proximal_reacquisition_self_test():
 ref={'median_radius_mm':1.7,'median_center_hu':500.};good={'radius_mm':1.72,'centroid_shift_mm':.15,'circularity':.85,'component_median_hu':520.};bad={'radius_mm':4.2,'centroid_shift_mm':.15,'circularity':.85,'component_median_hu':520.};sg,hg=score_metrics(good,ref);sb,hb=score_metrics(bad,ref);dirs=_cone_dirs(np.array([1.,0.,0.]),24.,10);return {'passed':bool(hg and not hb and len(dirs)==31),'good_score':sg,'bad_score':sb,'n_directions':len(dirs)}
