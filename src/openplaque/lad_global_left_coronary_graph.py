from __future__ import annotations

"""Global left-coronary graph reconstruction from source Series-7 CCTA.

The validated ~25 mm LAD is the only positive coronary anchor. The validated RCA
is a hard exclusion corridor and is never a target. Aorta geometry is used only
for broad ROI construction and post-hoc ranking, never for node discovery, edge
cost, or path tracing. Candidate graph paths are accepted only after dense
source-CCTA validation on planes perpendicular to their final local tangent.
"""

import json, math, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates, zoom
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import frangi
from skimage.feature import peak_local_max

ALGORITHM_VERSION='lad-global-left-coronary-graph-v1.0'
ISO_MM=.90
RCA_EXCLUSION_MM=3.0
MAX_EDGE_MM=6.5
MAX_NODES=700
DENSE_STEP_MM=.35

def _json_read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def _json_write(o,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def _unit(v):
 v=np.asarray(v,float);return v/max(float(np.linalg.norm(v)),1e-12)
def arc_mm(path,spacing):
 p=np.asarray(path,float)
 if len(p)<2:return np.zeros(len(p))
 d=np.diff(p,axis=0)*np.asarray(spacing,float)[None,:];return np.r_[0.,np.cumsum(np.linalg.norm(d,axis=1))]
def resample_path(path,spacing,step=DENSE_STEP_MM):
 p=np.asarray(path,float)
 if len(p)<2:return p.copy()
 s=arc_mm(p,spacing);keep=np.r_[True,np.diff(s)>1e-6];p,s=p[keep],s[keep]
 q=np.arange(0.,s[-1]+.5*step,step)
 if len(q)==0 or q[-1]<s[-1]-.1:q=np.r_[q,s[-1]]
 q[-1]=min(q[-1],s[-1]);return np.column_stack([np.interp(q,s,p[:,j]) for j in range(3)])
def _orth_basis(t):
 t=_unit(t);ref=np.array([1.,0.,0.]) if abs(t[0])<.82 else np.array([0.,1.,0.]);u=_unit(np.cross(t,ref));v=_unit(np.cross(t,u));return u,v
def orthogonal_plane(ct,p,t,spacing,half=5.,pix=.16):
 u,v=_orth_basis(t);g=np.arange(-half,half+1e-9,pix,dtype=np.float32);V,U=np.meshgrid(g,g,indexing='ij');sp=np.asarray(spacing,np.float32);c=np.asarray(p,np.float32)*sp
 pos=c[None,None,:]+U[...,None]*u.astype(np.float32)+V[...,None]*v.astype(np.float32);vox=pos/sp;out=np.empty(U.shape,np.float32)
 map_coordinates(ct,[vox[...,0],vox[...,1],vox[...,2]],output=out,order=1,mode='nearest',prefilter=False);return out,g,u,v
def lumen_metrics(im,g,min_hu=180.,max_hu=1200.):
 pix=float(abs(g[1]-g[0]));bright=(im>=min_hu)&(im<=max_hu);lab,_=ndi.label(bright,structure=np.ones((3,3),np.uint8));cy=cx=len(g)//2;chosen=int(lab[cy,cx])
 if chosen==0:
  yy,xx=np.nonzero(bright)
  if len(yy):
   d=np.hypot(g[yy],g[xx]);j=int(np.argmin(d));chosen=int(lab[yy[j],xx[j]]) if float(d[j])<=.9 else 0
 if chosen<=0:return {'radius_mm':np.nan,'centroid_shift_mm':np.nan,'centroid_u_mm':np.nan,'centroid_v_mm':np.nan,'circularity':np.nan,'component_median_hu':np.nan}
 comp=lab==chosen;yy,xx=np.nonzero(comp);area=float(comp.sum())*pix*pix;r=math.sqrt(area/math.pi);cu,cv=float(np.mean(g[xx])),float(np.mean(g[yy]));shift=float(math.hypot(cu,cv));er=ndi.binary_erosion(comp,structure=np.ones((3,3),bool));per=max(float(np.logical_and(comp,~er).sum())*pix,pix);circ=float(np.clip(4*math.pi*area/(per*per),0,1.2))
 return {'radius_mm':r,'centroid_shift_mm':shift,'centroid_u_mm':cu,'centroid_v_mm':cv,'circularity':circ,'component_median_hu':float(np.median(im[comp]))}
def score_metrics(m,ref):
 rr,rh=float(ref['median_radius_mm']),float(ref['median_center_hu']);r,sh,circ,med=m['radius_mm'],m['centroid_shift_mm'],m['circularity'],m['component_median_hu']
 rs=math.exp(-.5*((r-rr)/max(.48*rr,.60))**2) if np.isfinite(r) else 0.;ss=math.exp(-.5*(sh/.62)**2) if np.isfinite(sh) else 0.;cs=min(1.,max(0.,circ/.50)) if np.isfinite(circ) else 0.;hs=math.exp(-.5*((med-rh)/320.)**2) if np.isfinite(med) else 0.;score=.38*rs+.30*ss+.18*cs+.14*hs
 hard=bool(np.isfinite(r) and .55*rr<=r<=min(3.05,1.70*rr+.35) and np.isfinite(sh) and sh<=.90 and np.isfinite(circ) and circ>=.20 and np.isfinite(med) and max(150.,.34*rh)<=med<=1200. and score>=.60);return float(score),hard
def _sampled_hu_valid(mm):
 try:
  a=np.asarray(mm);zs=np.linspace(0,a.shape[0]-1,min(9,a.shape[0]),dtype=int);ys=np.linspace(0,a.shape[1]-1,min(9,a.shape[1]),dtype=int);xs=np.linspace(0,a.shape[2]-1,min(9,a.shape[2]),dtype=int);s=np.asarray(a[np.ix_(zs,ys,xs)],float);p1,p99=np.nanpercentile(s,[1,99]);return bool(p99-p1>300 and p99>250 and p1<100)
 except Exception:return False
def _load_source_ct(root):
 root=Path(root);metas=[root/'Cache'/'LAD_Frozen_Proximal_Reacquisition_v1'/'series7_geometry.json',root/'Cache'/'Coronary_Anatomy_Reconciliation_v1'/'series7_dicom_geometry.json']
 meta=meta_fp=None
 for fp in metas:
  if fp.exists():
   try:
    m=_json_read(fp)
    if 'spacing_zyx' in m and 'positions_lps_mm' in m and 'image_orientation_patient' in m:meta,meta_fp=m,fp;break
   except Exception:pass
 if meta is None:raise FileNotFoundError('No verified Series-7 geometry metadata found')
 expected=tuple(meta.get('shape',[]));cands=[]
 if meta.get('ct_cache_source'):cands.append(Path(meta['ct_cache_source']))
 cands += [root/'Cache'/'Secondary_3D_Vesselness_Topology_v1'/'series7_int16.npy',root/'Cache'/'Coronary_Anatomy_Reconciliation_v1'/'series7_int16.npy',root/'Cache'/'LAD_Confirmed_Backtrack_v1'/'series7_int16.npy']
 seen=set()
 for fp in cands:
  if str(fp) in seen or not fp.exists():continue
  seen.add(str(fp))
  try:mm=np.load(fp,mmap_mode='r')
  except Exception:continue
  if expected and tuple(mm.shape)!=expected:continue
  if _sampled_hu_valid(mm):return mm,meta,str(fp),str(meta_fp)
 raise RuntimeError('No shape- and HU-validated Series-7 CT cache found')
def _load_xyz(fp):
 d=pd.read_csv(fp)
 if not {'z','y','x'}.issubset(d.columns):raise ValueError(f'{fp} lacks z,y,x')
 return d[['z','y','x']].to_numpy(float)
def _tangent_mismatch(a,b):
 c=abs(float(np.dot(_unit(a),_unit(b))));return float(np.degrees(np.arccos(np.clip(c,-1,1))))

def _line_support(vessel,a,b,step=.75):
 a=np.asarray(a,float);b=np.asarray(b,float);n=max(2,int(math.ceil(float(np.linalg.norm(b-a))/step))+1);w=np.linspace(0,1,n)[:,None];q=(1-w)*a[None,:]+w*b[None,:];vals=map_coordinates(vessel,[q[:,0],q[:,1],q[:,2]],order=1,mode='nearest');return float(np.mean(vals)),float(np.quantile(vals,.25))

class GlobalLeftCoronaryGraphWorkflow:
 def __init__(self,root='/content/drive/MyDrive/OpenPlaque',reuse=True):
  self.root=Path(root);self.cache=self.root/'Cache'/'LAD_Global_Left_Coronary_Graph_v1';self.out=self.root/'LAD_Global_Left_Coronary_Graph_Report';self.cache.mkdir(parents=True,exist_ok=True);self.out.mkdir(parents=True,exist_ok=True);self.reuse=bool(reuse);self.ct=self.meta=self.spacing=self.lad=self.rca=self.aorta=self.ref=None;self.nodes=self.edges=self.paths=self.validation=self.summary=None
 def load_inputs(self):
  self.ct,self.meta,ctsrc,metasrc=_load_source_ct(self.root);self.spacing=np.asarray(self.meta['spacing_zyx'],float)
  ladc=[self.root/'LAD_Frozen_Proximal_Reacquisition_Report'/'combined_lad_centerline.csv',self.root/'Cache'/'LAD_Frozen_Proximal_Reacquisition_v1'/'combined_lad_centerline.csv'];ladfp=next((p for p in ladc if p.exists()),None)
  if ladfp is None:raise FileNotFoundError('Validated ~25 mm LAD artifact not found')
  self.lad=_load_xyz(ladfp);ll=float(arc_mm(self.lad,self.spacing)[-1])
  if not 24.0<=ll<=26.0:raise RuntimeError(f'Expected validated LAD ~25 mm, got {ll:.3f} mm')
  rcafp=self.root/'PCAT_RCA_10_50'/'rca_centerline_smoothed_zyx.csv';self.rca=_load_xyz(rcafp)
  calfp=self.root/'Cache'/'LAD_Frozen_Proximal_Reacquisition_v1'/'lad_lumen_calibration.json';self.ref=_json_read(calfp)
  afp=self.root/'RCA_Ostium_TotalSegmentator'/'aorta_series7_totalseg.nii.gz'
  if afp.exists():
   a=sitk.GetArrayFromImage(sitk.ReadImage(str(afp)))>0
   if tuple(a.shape)==tuple(self.ct.shape):self.aorta=a
  self.prov={'ct_cache':ctsrc,'geometry':metasrc,'lad':str(ladfp),'lad_length_mm':ll,'rca':str(rcafp),'calibration':str(calfp),'aorta':str(afp) if self.aorta is not None else None};_json_write(self.prov,self.cache/'input_provenance.json');return self.prov
 def _roi(self):
  pts=[self.lad,self.rca[-20:]]
  if self.aorta is not None:
   zz,yy,xx=np.nonzero(self.aorta);pts.append(np.column_stack([zz,yy,xx])[::max(1,len(zz)//5000)])
  p=np.vstack(pts);m=np.ceil(18/self.spacing).astype(int);lo=np.maximum(0,np.floor(p.min(0)).astype(int)-m);hi=np.minimum(np.asarray(self.ct.shape),np.ceil(p.max(0)).astype(int)+m+1);ep=self.lad[0];lo=np.maximum(lo,np.floor((ep*self.spacing-65)/self.spacing).astype(int));hi=np.minimum(hi,np.ceil((ep*self.spacing+65)/self.spacing).astype(int));return lo,hi
 def build_vesselness(self):
  if self.ct is None:self.load_inputs()
  fp=self.cache/'roi_vesselness.npz'
  if self.reuse and fp.exists():
   z=np.load(fp);self.roi_lo=z['roi_lo'];self.roi_hi=z['roi_hi'];self.iso=z['iso'].astype(np.float32);self.vessel=z['vessel'].astype(np.float32);return self.vessel
  lo,hi=self._roi();sub=np.asarray(self.ct[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]],np.float32);iso=zoom(sub,self.spacing/ISO_MM,order=1,prefilter=False);norm=np.clip((iso-120)/650,0,1).astype(np.float32);v=frangi(norm,sigmas=[1.,1.5,2.,2.5],black_ridges=False,gamma=None).astype(np.float32);self.roi_lo,self.roi_hi,self.iso,self.vessel=lo,hi,iso,v;np.savez_compressed(fp,roi_lo=lo,roi_hi=hi,iso=iso.astype(np.float16),vessel=v.astype(np.float16));return v
 def _iso_to_source(self,q):return self.roi_lo[None,:]+np.asarray(q,float)*(ISO_MM/self.spacing)[None,:]
 def _source_to_iso(self,p):return (np.asarray(p,float)-self.roi_lo[None,:])*(self.spacing/ISO_MM)[None,:]
 def _aorta_distance(self,p):
  if self.aorta is None:return np.nan
  if not hasattr(self,'_atree'):
   b=self.aorta^ndi.binary_erosion(self.aorta);pts=np.column_stack(np.nonzero(b));pts=pts[::max(1,len(pts)//30000)];self._atree=cKDTree(pts*self.spacing)
  return float(self._atree.query(np.asarray(p)*self.spacing)[0])
 def discover_nodes(self):
  if not hasattr(self,'vessel'):self.build_vesselness()
  vv=self.vessel.copy();vv[(self.iso<170)|(self.iso>1100)]=0;pos=vv[vv>0];thr=float(np.quantile(pos,.992)) if len(pos) else 1.;pk=peak_local_max(vv,min_distance=2,threshold_abs=max(thr,1e-6),num_peaks=MAX_NODES,exclude_border=3);src=self._iso_to_source(pk);rtree=cKDTree(self.rca*self.spacing)
  sig=1.2;Hs=[ndi.gaussian_filter(self.iso,sigma=sig,order=o,mode='nearest') for o in [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)]];rows=[]
  for qi,p in zip(pk,src):
   rd=float(rtree.query(p*self.spacing)[0])
   if rd<RCA_EXCLUSION_MM:continue
   iz,iy,ix=[int(v) for v in qi];a,b,c,d,e,f=[h[iz,iy,ix] for h in Hs];H=np.array([[a,d,e],[d,b,f],[e,f,c]],float);w,V=np.linalg.eigh(H);t=_unit(V[:,int(np.argmin(np.abs(w)))])
   dirs=[t,-t];u,v=_orth_basis(t)
   for ang in (12.,24.):
    aa=math.radians(ang)
    for ph in np.linspace(0,2*math.pi,6,endpoint=False):dirs.append(_unit(math.cos(aa)*t+math.sin(aa)*(math.cos(ph)*u+math.sin(ph)*v)))
   best=None;passes=0
   for dd in dirs:
    im,g,_,_=orthogonal_plane(self.ct,p,dd,self.spacing);m=lumen_metrics(im,g);sc,ok=score_metrics(m,self.ref);passes+=int(ok)
    if best is None or sc>best[0]:best=(sc,ok,m,dd)
   sc,ok,m,dd=best
   if not ok:continue
   rows.append({'node_id':len(rows),'z':p[0],'y':p[1],'x':p[2],'iso_z':qi[0],'iso_y':qi[1],'iso_x':qi[2],'vesselness':float(vv[tuple(qi)]),'plane_score':sc,'radius_mm':m['radius_mm'],'shift_mm':m['centroid_shift_mm'],'orientation_passes':passes,'tz':dd[0],'ty':dd[1],'tx':dd[2],'rca_distance_mm':rd,'aorta_distance_mm':self._aorta_distance(p)})
  self.nodes=pd.DataFrame(rows);self.nodes.to_csv(self.cache/'graph_nodes.csv',index=False);return self.nodes
 def build_graph(self):
  if self.nodes is None:self.discover_nodes()
  rows=[]
  if len(self.nodes):
   P=self.nodes[['z','y','x']].to_numpy(float);T=self.nodes[['tz','ty','tx']].to_numpy(float);tree=cKDTree(P*self.spacing)
   for i,j in tree.query_pairs(MAX_EDGE_MM):
    dist=float(np.linalg.norm((P[i]-P[j])*self.spacing));tm=_tangent_mismatch(T[i],T[j])
    if tm>55:continue
    lm,lq=_line_support(self.vessel,self._source_to_iso(P[i:i+1])[0],self._source_to_iso(P[j:j+1])[0])
    if lm<.015 and lq<.004:continue
    rows.append({'i':i,'j':j,'distance_mm':dist,'tangent_mismatch_deg':tm,'line_mean':lm,'line_q25':lq,'cost':dist*(1+.012*tm)+1.5/(lm+.03)})
  self.edges=pd.DataFrame(rows,columns=['i','j','distance_mm','tangent_mismatch_deg','line_mean','line_q25','cost']);self.edges.to_csv(self.cache/'graph_edges.csv',index=False);return self.edges
 def _anchor_links(self):
  ep=self.lad[0];t=_unit((self.lad[min(5,len(self.lad)-1)]-ep)*self.spacing);P=self.nodes[['z','y','x']].to_numpy(float);T=self.nodes[['tz','ty','tx']].to_numpy(float);out=[]
  for i,p in enumerate(P):
   d=float(np.linalg.norm((p-ep)*self.spacing))
   if d>9:continue
   tm=_tangent_mismatch(t,T[i])
   if tm>70:continue
   lm,lq=_line_support(self.vessel,self._source_to_iso(ep[None,:])[0],self._source_to_iso(p[None,:])[0])
   if d>3 and lm<.008:continue
   out.append((i,d*(1+.01*tm)+1/(lm+.025)))
  return out
 def enumerate_paths(self):
  if self.edges is None:self.build_graph()
  n=len(self.nodes);rows=[];arr=[]
  if not n:self.paths=pd.DataFrame(rows);return self.paths
  ri=[];ci=[];da=[]
  for r in self.edges.itertuples():ri += [int(r.i),int(r.j)];ci += [int(r.j),int(r.i)];da += [float(r.cost),float(r.cost)]
  anchor=n
  for i,c in self._anchor_links():ri += [anchor,int(i)];ci += [int(i),anchor];da += [c,c]
  G=csr_matrix((da,(ri,ci)),shape=(n+1,n+1));dist,pred=dijkstra(G,directed=False,indices=anchor,return_predecessors=True);P=self.nodes[['z','y','x']].to_numpy(float)
  for target in np.argsort(dist[:n]):
   if not np.isfinite(dist[target]):continue
   epd=float(np.linalg.norm((P[target]-self.lad[0])*self.spacing))
   if epd<5:continue
   chain=[];cur=int(target);guard=0
   while cur!=anchor and cur>=0 and guard<n+5:chain.append(cur);cur=int(pred[cur]);guard+=1
   if cur!=anchor:continue
   chain=chain[::-1];pts=np.vstack([self.lad[0],P[chain]]);plen=float(arc_mm(pts,self.spacing)[-1]);rows.append({'path_id':len(rows),'target_node':target,'n_nodes':len(chain),'graph_cost':float(dist[target]),'graph_length_mm':plen,'endpoint_from_lad_mm':epd,'endpoint_aorta_mm':self._aorta_distance(pts[-1]),'node_ids':';'.join(map(str,chain))});arr.append(pts)
   if len(rows)>=12:break
  self._path_arrays=arr;self.paths=pd.DataFrame(rows);self.paths.to_csv(self.cache/'candidate_graph_paths.csv',index=False);return self.paths
 def _dense_qc(self,path):
  d=resample_path(path,self.spacing);s=arc_mm(d,self.spacing);rows=[]
  for i,p in enumerate(d):
   a=max(0,i-3);b=min(len(d)-1,i+3);t=_unit((d[b]-d[a])*self.spacing) if b>a else np.array([1.,0.,0.]);im,g,u,v=orthogonal_plane(self.ct,p,t,self.spacing);m=lumen_metrics(im,g);p2=p.copy();rec=0.
   if np.isfinite(m['centroid_shift_mm']) and m['centroid_shift_mm']<=.8:
    cu,cv=m['centroid_u_mm'],m['centroid_v_mm'];rec=float(math.hypot(cu,cv));p2=(p*self.spacing+cu*u+cv*v)/self.spacing;im,g,_,_=orthogonal_plane(self.ct,p2,t,self.spacing);m=lumen_metrics(im,g)
   sc,ok=score_metrics(m,self.ref);rows.append({'arc_mm':float(s[i]),'z':p2[0],'y':p2[1],'x':p2[2],'radius_mm':m['radius_mm'],'shift_mm':m['centroid_shift_mm'],'plane_score':sc,'plane_pass':bool(ok),'recenter_mm':rec,'aorta_distance_mm':self._aorta_distance(p2)})
  q=pd.DataFrame(rows);f=q.plane_pass.to_numpy(bool);fail=None
  for i in range(len(f)):
   if not f[i] and np.sum(~f[i:min(len(f),i+3)])>=2:fail=i;break
  valid=float(q.arc_mm.iloc[-1]) if fail is None else float(q.arc_mm.iloc[max(0,fail-1)]);return q,valid,fail
 def validate_paths(self):
  if self.paths is None:self.enumerate_paths()
  rows=[];best=None
  for r,pts in zip(self.paths.itertuples(),getattr(self,'_path_arrays',[])):
   q,valid,fail=self._dense_qc(pts);q.to_csv(self.cache/f'path_{int(r.path_id):02d}_dense_qc.csv',index=False);pf=float(q.plane_pass.mean()) if len(q) else 0.;accepted=bool(valid>=max(3.,float(r.graph_length_mm)-.7) and pf>=.72);rec={'path_id':int(r.path_id),'graph_length_mm':float(r.graph_length_mm),'dense_valid_mm':valid,'dense_pass_fraction':pf,'first_sustained_failure_index':None if fail is None else int(fail),'endpoint_aorta_mm':float(r.endpoint_aorta_mm) if np.isfinite(r.endpoint_aorta_mm) else np.nan,'accepted_extension':accepted};rows.append(rec);rank=(0 if accepted else 1,-valid,rec['endpoint_aorta_mm'] if np.isfinite(rec['endpoint_aorta_mm']) else 1e6)
   if best is None or rank<best[0]:best=(rank,rec,q,pts)
  self.validation=pd.DataFrame(rows);self.validation.to_csv(self.cache/'path_validation_summary.csv',index=False)
  if best:
   _,rec,q,pts=best;self.best_qc=q;self.best_path=pts;q.to_csv(self.cache/'best_path_dense_qc.csv',index=False);pd.DataFrame(pts,columns=['z','y','x']).to_csv(self.cache/'best_graph_path_centerline.csv',index=False)
  accepted=int(self.validation.accepted_extension.sum()) if len(self.validation) else 0;status='GLOBAL_GRAPH_VALIDATED_LAD_EXTENSION_SUPPORTED' if accepted else ('GLOBAL_GRAPH_COMPONENTS_NEAR_LAD_NO_VALIDATED_PATH' if len(self.nodes) else 'NO_GLOBAL_LEFT_CORONARY_GRAPH_NODES');self.summary={'algorithm':ALGORITHM_VERSION,'status':status,'validated_lad_input_mm':float(arc_mm(self.lad,self.spacing)[-1]),'n_nodes':int(len(self.nodes)),'n_edges':int(len(self.edges)),'n_candidate_paths':int(len(self.paths)),'n_validated_extensions':accepted,'rca_exclusion_mm':RCA_EXCLUSION_MM,'aorta_used_for_discovery':False,'aorta_used_for_edge_cost':False,'aorta_used_only_for_roi_and_posthoc_ranking':True,'final_tangent_dense_qc_required':True,'best_path':None if not best else best[1]};_json_write(self.summary,self.cache/'summary.json');return self.summary
 def make_figures(self):
  if self.summary is None:self.validate_paths()
  names=[];sub=np.asarray(self.ct[self.roi_lo[0]:self.roi_hi[0],self.roi_lo[1]:self.roi_hi[1],self.roi_lo[2]:self.roi_hi[2]],np.float32);fig,axs=plt.subplots(1,3,figsize=(16,5))
  for ax,im,title in zip(axs,[sub.max(0),sub.max(1),sub.max(2)],['z-MIP','y-MIP','x-MIP']):ax.imshow(im,cmap='gray',vmin=0,vmax=850);ax.set_title(title);ax.axis('off')
  fp=self.out/'01_source_roi_mips.png';fig.tight_layout();fig.savefig(fp,dpi=160);plt.close(fig);names.append(fp.name);fig,axs=plt.subplots(1,3,figsize=(16,5))
  for ax,im,title in zip(axs,[self.vessel.max(0),self.vessel.max(1),self.vessel.max(2)],['z vesselness','y vesselness','x vesselness']):ax.imshow(im,cmap='gray');ax.set_title(title);ax.axis('off')
  fp=self.out/'02_vesselness_mips.png';fig.tight_layout();fig.savefig(fp,dpi=160);plt.close(fig);names.append(fp.name);fig=plt.figure(figsize=(9,8));ax=fig.add_subplot(111,projection='3d')
  if len(self.nodes):
   p=self.nodes[['x','y','z']].to_numpy();ax.scatter(p[:,0],p[:,1],p[:,2],s=8,alpha=.65)
   for r in self.edges.itertuples():a=p[int(r.i)];b=p[int(r.j)];ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],linewidth=.5,alpha=.35)
  lad=self.lad[:,[2,1,0]];rca=self.rca[:,[2,1,0]];ax.plot(lad[:,0],lad[:,1],lad[:,2],linewidth=3,label='validated LAD');ax.plot(rca[:,0],rca[:,1],rca[:,2],linewidth=2,label='RCA exclusion');ax.legend();ax.set_title('Global candidate graph');fp=self.out/'03_global_graph_3d.png';fig.tight_layout();fig.savefig(fp,dpi=160);plt.close(fig);names.append(fp.name)
  if hasattr(self,'best_qc') and self.best_qc is not None and len(self.best_qc):
   q=self.best_qc;fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True);axs[0].plot(q.arc_mm,q.plane_score);axs[0].axhline(.6,linestyle='--');axs[0].set_ylabel('score');axs[1].plot(q.arc_mm,q.radius_mm);axs[1].set_ylabel('radius mm');axs[2].plot(q.arc_mm,q.aorta_distance_mm);axs[2].set_ylabel('aorta mm');axs[2].set_xlabel('arc mm');fp=self.out/'04_best_path_dense_qc.png';fig.tight_layout();fig.savefig(fp,dpi=160);plt.close(fig);names.append(fp.name)
  _json_write(names,self.out/'figure_manifest.json');return names
 def build_report(self):
  if self.summary is None:self.validate_paths()
  names=self.make_figures();nt=self.nodes.sort_values('plane_score',ascending=False).head(40).to_html(index=False,float_format=lambda x:f'{x:.3f}') if len(self.nodes) else '<p>No nodes.</p>';pt=self.validation.to_html(index=False,float_format=lambda x:f'{x:.3f}') if len(self.validation) else '<p>No paths.</p>';s=self.summary;html=f"<html><head><meta charset='utf-8'><title>OpenPlaque Global Left-Coronary Graph</title></head><body><h1>OpenPlaque — Global Left-Coronary Graph Reconstruction</h1><p><b>Status: {s['status']}</b></p><p>Validated LAD is the only positive coronary anchor. RCA is excluded within {RCA_EXCLUSION_MM:.1f} mm. Aorta is not used for discovery, edge cost, or path tracing.</p><pre>{json.dumps(s,indent=2)}</pre><h2>Top nodes</h2>{nt}<h2>Path validation</h2>{pt}{''.join(f'<h3>{n}</h3><img style=\"max-width:100%\" src=\"{n}\">' for n in names)}</body></html>";fp=self.out/'OPENPLAQUE_LAD_GLOBAL_LEFT_CORONARY_GRAPH_REPORT.html';fp.write_text(html,encoding='utf-8');return fp
 def package(self):
  report=self.build_report();zp=self.out/'OPENPLAQUE_LAD_GLOBAL_LEFT_CORONARY_GRAPH_REPORT_BACK.zip'
  with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as z:
   for fp in self.out.iterdir():
    if fp.is_file() and fp!=zp:z.write(fp,arcname=fp.name)
   for fp in self.cache.glob('*'):
    if fp.is_file() and fp.suffix in ('.csv','.json'):z.write(fp,arcname=fp.name)
  return report,zp

def synthetic_global_graph_self_test():
 return {'passed':bool(abs(_tangent_mismatch([1,0,0],[-1,0,0]))<1e-6 and np.allclose(arc_mm([[0,0,0],[1,0,0],[2,0,0]],[1,1,1]),[0,1,2])),'algorithm':ALGORITHM_VERSION}
