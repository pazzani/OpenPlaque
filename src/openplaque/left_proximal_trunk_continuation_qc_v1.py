from __future__ import annotations

"""Dense source-CCTA QC of the vessel continuation found beyond the accepted LAD endpoint.

This experiment does not change the frozen master. It adjudicates the best source-supported
frontier path from the LAD mask-gate diagnostic using dense orthogonal source-CCTA planes,
with the canonical RCA as an independent positive control for the plane-QC machinery.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE='0593b453959f5a353d644267fbeef24b514ef4d7'
ALGORITHM='left-proximal-trunk-continuation-qc-v1.0'
OUTPUT_DIRNAME='Left_Proximal_Trunk_Continuation_QC_v1'
DIAG_DIR='Left_Main_LAD_Mask_Gate_Diagnostic_v1'
SOURCE_CACHE=Path('Cache/Secondary_3D_Vesselness_Topology_v1')
LAD_PATH=Path('Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv')
RCA_PATH=Path('PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv')
AORTA=Path('TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz')
MASTER=Path('Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
CANDIDATE_REL=Path(DIAG_DIR)/'hard_mask_best_frontier_path.csv'
PRIOR_SUMMARY_REL=Path(DIAG_DIR)/'summary.json'

PLANE_STEP_MM=.4
SUSTAINED_FAIL_N=3
MIN_ACCEPTED_EXTENSION_MM=5.0
MIN_ACCEPTED_PASS_FRACTION=.80


def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding='utf-8')

def _source(cache):
    arr=np.load(_req(cache/'series7_int16.npy'),mmap_mode='r')
    meta=json.loads(_req(cache/'series7_int16.json').read_text())
    ref=sitk.GetImageFromArray(np.asarray(arr)); sp=np.asarray(meta['spacing_zyx'],float)
    ref.SetSpacing(tuple(sp[::-1])); ref.SetOrigin(tuple(np.asarray(meta['positions_lps_mm'][0],float)))
    iop=np.asarray(meta['image_orientation_patient'],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
    D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
    ref.SetDirection(tuple(D.ravel())); return ref,arr

def _xyz_to_zyx(img,p):
    p=np.atleast_2d(np.asarray(p,float)); o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing()); D=np.asarray(img.GetDirection()).reshape(3,3)
    return (((p-o)@np.linalg.inv(D).T)/sp)[:,::-1]

def _zyx_to_xyz(img,p):
    p=np.atleast_2d(np.asarray(p,float)); q=p[:,::-1]; o=np.asarray(img.GetOrigin()); sp=np.asarray(img.GetSpacing()); D=np.asarray(img.GetDirection()).reshape(3,3)
    return o+(q*sp)@D.T

def _sample(img,a,p,cval=-1024.):
    return map_coordinates(np.asarray(a),_xyz_to_zyx(img,p).T,order=1,mode='constant',cval=cval)

def _load_path(path,ref):
    d=pd.read_csv(_req(path))
    for c in (('lps_x_mm','lps_y_mm','lps_z_mm'),('x_mm','y_mm','z_mm')):
        if all(x in d.columns for x in c): return d[list(c)].to_numpy(float)
    for c in (('zyx_z','zyx_y','zyx_x'),('z','y','x')):
        if all(x in d.columns for x in c): return _zyx_to_xyz(ref,d[list(c)].to_numpy(float))
    raise ValueError(f'No recognized coordinate columns in {path}: {list(d.columns)}')

def _arc(p):
    p=np.asarray(p,float); return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p)>1 else np.zeros(len(p))

def _resample(p,step=PLANE_STEP_MM):
    p=np.asarray(p,float); a=_arc(p)
    if len(p)<2 or a[-1]<=0:return p.copy(),a
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6:q=np.r_[q,a[-1]]
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)]),q

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-9 else np.zeros_like(v)

def _orth_basis(t):
    t=_unit(t); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; u=_unit(np.cross(t,seed)); v=_unit(np.cross(t,u)); return u,v

def _plane(ref,src,c,t,half=5.5,step=.20):
    t=_unit(t); u,v=_orth_basis(t); q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing='ij')
    P=c+xx[...,None]*u+yy[...,None]*v
    return _sample(ref,src,P.reshape(-1,3)).reshape(len(q),len(q)),q

def _component_metrics(im,q):
    n=len(q); iy=ix=int(np.argmin(np.abs(q))); center=float(im[iy,ix])
    yy,xx=np.meshgrid(q,q,indexing='ij'); rr=np.sqrt(xx*xx+yy*yy)
    thr=max(220.,min(500.,.55*center))
    bw=(im>=thr)&(rr<=3.5)
    lab,nlab=ndi.label(bw,np.ones((3,3),int))
    labels=[]
    if lab[iy,ix]>0: labels=[int(lab[iy,ix])]
    else:
        pts=np.argwhere(bw)
        if len(pts):
            dist=np.sqrt((q[pts[:,1]])**2+(q[pts[:,0]])**2); j=int(np.argmin(dist))
            if dist[j]<=1.0: labels=[int(lab[tuple(pts[j])])]
    if not labels:
        return {'center_hu':center,'threshold_hu':thr,'component_found':False,'radius_mm':np.nan,'centroid_offset_mm':np.inf,'axis_ratio':np.inf,'contrast_hu':-np.inf}
    mask=lab==labels[0]; pts=np.argwhere(mask); xs=q[pts[:,1]]; ys=q[pts[:,0]]
    cx=float(np.mean(xs)); cy=float(np.mean(ys)); off=float(np.hypot(cx,cy)); area=float(len(pts)*(.20**2)); rad=float(np.sqrt(area/np.pi))
    if len(pts)>=4:
        C=np.cov(np.column_stack([xs,ys]).T); ev=np.linalg.eigvalsh(C); axis=float(np.sqrt(max(ev[-1],1e-6)/max(ev[0],1e-6)))
    else: axis=np.inf
    ring=(rr>=3.5)&(rr<=5.0); contrast=float(np.median(im[mask])-np.median(im[ring])) if np.any(ring) else np.nan
    return {'center_hu':center,'threshold_hu':thr,'component_found':True,'radius_mm':rad,'centroid_offset_mm':off,'axis_ratio':axis,'contrast_hu':contrast}

def _fixed_plane_pass(m):
    return bool(m['component_found'] and m['center_hu']>=200 and .55<=m['radius_mm']<=3.2 and m['centroid_offset_mm']<=1.10 and m['axis_ratio']<=2.2 and m['contrast_hu']>=40)

def _dense_qc(ref,src,path,label):
    p,q=_resample(path,PLANE_STEP_MM); tt=np.gradient(p,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9)
    rows=[]
    for i,(c,t,a) in enumerate(zip(p,tt,q)):
        im,grid=_plane(ref,src,c,t); m=_component_metrics(im,grid); m.update(index=i,arc_mm=float(a),label=label,plane_pass=_fixed_plane_pass(m)); rows.append(m)
    return p,q,pd.DataFrame(rows)

def _truncate_by_sustained_failure(df,nfail=SUSTAINED_FAIL_N):
    passes=df['plane_pass'].astype(bool).to_numpy(); cutoff=len(df)-1; first_fail=None
    for i in range(0,len(passes)-nfail+1):
        if not np.any(passes[i:i+nfail]):
            first_fail=i; cutoff=max(0,i-1); break
    accepted=df.iloc[:cutoff+1].copy(); frac=float(accepted['plane_pass'].mean()) if len(accepted) else 0.
    arc=float(accepted['arc_mm'].iloc[-1]) if len(accepted) else 0.
    return {'first_sustained_failure_index':first_fail,'accepted_last_index':int(cutoff),'accepted_arc_mm':arc,'accepted_plane_pass_fraction':frac,'accepted':bool(arc>=MIN_ACCEPTED_EXTENSION_MM and frac>=MIN_ACCEPTED_PASS_FRACTION)}

def synthetic_truncation_self_test():
    d=pd.DataFrame({'arc_mm':np.arange(10)*.4,'plane_pass':[1,1,1,1,0,1,1,0,0,0]})
    r=_truncate_by_sustained_failure(d,3); assert r['first_sustained_failure_index']==7 and r['accepted_last_index']==6
    return {'ok':True,**r}

def _surface_tree(mask_path,ref):
    im=sitk.ReadImage(str(_req(mask_path)))
    if not (im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection())):
        im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    m=sitk.GetArrayFromImage(im)>0; surf=m&~ndi.binary_erosion(m,iterations=1,border_value=0); z=np.argwhere(surf)[::2]
    return cKDTree(_zyx_to_xyz(ref,z.astype(float)).astype(np.float32))

def _orient_nearest(path,tree):
    d0=float(tree.query(path[0])[0]); d1=float(tree.query(path[-1])[0]); return (path.copy(),d0) if d0<=d1 else (path[::-1].copy(),d1)

def run(drive_root='/content/drive/MyDrive/OpenPlaque',output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/'run_state.json',{'status':'STARTED','algorithm':ALGORITHM,'baseline_commit':BASELINE})
    required=[root/SOURCE_CACHE/'series7_int16.npy',root/SOURCE_CACHE/'series7_int16.json',root/LAD_PATH,root/RCA_PATH,root/AORTA,root/MASTER,root/CANDIDATE_REL,root/PRIOR_SUMMARY_REL]
    for p in required:_req(p)
    prior=json.loads((root/PRIOR_SUMMARY_REL).read_text()); master=json.loads((root/MASTER).read_text())
    if prior.get('status')!='LAD_SOURCE_SUPPORT_FAILS_EVEN_WITH_MASK_GATE_REMOVED': raise RuntimeError(f'Unexpected diagnostic prerequisite: {prior.get("status")}')
    ref,src=_source(root/SOURCE_CACHE); aorta_tree=_surface_tree(root/AORTA,ref)
    lad=_load_path(root/LAD_PATH,ref); rca=_load_path(root/RCA_PATH,ref); cand=_load_path(root/CANDIDATE_REL,ref)
    lad_o,lad_aorta=_orient_nearest(lad,aorta_tree); rca_o,_=_orient_nearest(rca,aorta_tree)
    if np.linalg.norm(cand[0]-lad_o[0])>np.linalg.norm(cand[-1]-lad_o[0]): cand=cand[::-1].copy()
    start_gap=float(np.linalg.norm(cand[0]-lad_o[0]));
    if start_gap>1.0: raise RuntimeError(f'Candidate does not start at accepted LAD proximal endpoint: gap={start_gap:.3f} mm')

    # Independent RCA plane-QC positive control.
    rp,rq=_resample(rca_o,.8); keep=(rq>=2.)&(rq<=min(18.,rq[-1])); rp=rp[keep]
    if len(rp)<8: rp,_=_resample(rca_o,.8)
    rtt=np.gradient(rp,axis=0); rtt/=np.maximum(np.linalg.norm(rtt,axis=1,keepdims=True),1e-9)
    rrows=[]
    for i,(c,t) in enumerate(zip(rp,rtt)):
        im,g=_plane(ref,src,c,t); m=_component_metrics(im,g); m['plane_pass']=_fixed_plane_pass(m); m['index']=i; rrows.append(m)
    rdf=pd.DataFrame(rrows); rdf.to_csv(out/'RCA_plane_qc_control.csv',index=False)
    rca_pass_fraction=float(rdf['plane_pass'].mean()); rca_control_pass=bool(rca_pass_fraction>=.75)

    pp,qq,qcdf=_dense_qc(ref,src,cand,'proximal_trunk_candidate'); qcdf.to_csv(out/'proximal_trunk_dense_plane_qc.csv',index=False)
    trunc=_truncate_by_sustained_failure(qcdf)
    last=int(trunc['accepted_last_index']); accepted_path=pp[:last+1]
    endpoint_aorta=float(aorta_tree.query(accepted_path[-1])[0]); start_aorta=float(aorta_tree.query(accepted_path[0])[0]); reduction=start_aorta-endpoint_aorta
    scientific_positive=bool(rca_control_pass and trunc['accepted'])
    status='PROXIMAL_TRUNK_CONTINUATION_QC_POSITIVE' if scientific_positive else ('PROXIMAL_TRUNK_QC_RCA_CONTROL_FAILED' if not rca_control_pass else 'PROXIMAL_TRUNK_CONTINUATION_NOT_ESTABLISHED')
    if scientific_positive:
        pd.DataFrame(accepted_path,columns=['lps_x_mm','lps_y_mm','lps_z_mm']).to_csv(out/'accepted_proximal_trunk_continuation_candidate.csv',index=False)
        combined=np.vstack([accepted_path[:0:-1],lad_o])
        pd.DataFrame(combined,columns=['lps_x_mm','lps_y_mm','lps_z_mm']).to_csv(out/'combined_LAD_plus_proximal_trunk_candidate.csv',index=False)

    # Compact QC figures.
    plt.figure(figsize=(8,4)); plt.plot(qcdf['arc_mm'],qcdf['center_hu'],label='center HU'); plt.axhline(200,ls='--',lw=.8); plt.xlabel('candidate arc (mm)'); plt.ylabel('HU'); plt.twinx().plot(qcdf['arc_mm'],qcdf['plane_pass'].astype(int),alpha=.45,label='plane pass'); plt.title('Dense source-CCTA QC along proximal continuation'); plt.tight_layout(); plt.savefig(out/'01_dense_qc_profile.png',dpi=180); plt.close()
    picks=np.linspace(0,len(pp)-1,min(12,len(pp))).astype(int); cols=4; rows=int(math.ceil(len(picks)/cols)); fig,axes=plt.subplots(rows,cols,figsize=(14,3.6*rows)); axes=np.atleast_1d(axes).ravel(); tt=np.gradient(pp,axis=0); tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-9)
    for ax in axes[len(picks):]:ax.axis('off')
    for ax,ix in zip(axes,picks):
        im,g=_plane(ref,src,pp[ix],tt[ix]); ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[g[0],g[-1],g[-1],g[0]]); ax.scatter([0],[0],s=14); row=qcdf.iloc[ix]; ax.set_title(f"{row.arc_mm:.1f} mm | pass={bool(row.plane_pass)}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Proximal continuation: dense orthogonal source-CCTA QC'); plt.tight_layout(); plt.savefig(out/'02_proximal_trunk_dense_orthogonal_qc.png',dpi=180); plt.close()

    summary={'status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'master_status':master.get('status'),'master_modified':False,'prior_diagnostic_status':prior.get('status'),'candidate_start_gap_to_accepted_LAD_mm':start_gap,'RCA_plane_qc_control':{'plane_count':int(len(rdf)),'pass_fraction':rca_pass_fraction,'accepted':rca_control_pass},'candidate_dense_qc':{**trunc,'candidate_total_arc_mm':float(qq[-1]),'start_aorta_distance_mm':start_aorta,'accepted_endpoint_aorta_distance_mm':endpoint_aorta,'aorta_distance_reduction_mm':reduction},'candidate_role':'source-supported proximal-trunk continuation from accepted LAD endpoint; not yet labeled LM and not added to frozen master','scientific_boundary':'A positive result validates continuity of a proximal vessel segment beyond the accepted LAD endpoint using dense source-CCTA plane QC. It does not establish LM identity, LCX topology, or modify the frozen master.'}
    _write_json(out/'summary.json',summary); _write_json(out/'run_state.json',{'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE})
    report=out/'OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_QC_REPORT.html'; report.write_text(f"<html><body><h1>OpenPlaque Proximal Trunk Continuation QC v1</h1><p><b>Status:</b> {status}</p><p>RCA plane-QC control pass: {rca_control_pass} ({rca_pass_fraction:.2f}).</p><p>Accepted continuation: {trunc['accepted_arc_mm']:.2f} mm; pass fraction {trunc['accepted_plane_pass_fraction']:.2f}.</p><p>Aortic distance: {start_aorta:.2f} → {endpoint_aorta:.2f} mm.</p><p>Frozen master unchanged; LM identity remains unresolved.</p></body></html>",encoding='utf-8')
    zpath=out/'OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_QC_RESULTS.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {'summary':summary,'report':str(report),'zip':str(zpath)}
