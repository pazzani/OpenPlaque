from __future__ import annotations
import gc, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = '0593b453959f5a353d644267fbeef24b514ef4d7'
ALGORITHM = 'lcx-family2-recursive-av-groove-v1.0-lowmem'
OUTPUT_DIRNAME = 'LCX_Family2_Recursive_AV_Groove_v1'
SOURCE_CACHE = Path('Cache/Secondary_3D_Vesselness_Topology_v1')
MASTER = Path('Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
LAD_PATH = Path('Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv')
RCA_PATH = Path('PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv')
TS = Path('TotalSegmentator_Cardiovascular_Cache_v1')
HEART = TS / 'heartchambers_highres'
COR_CURRENT = TS / 'coronary_arteries/coronary_arteries.nii.gz'
COR_LEGACY = TS / 'coronary_arteries_LEGACY/coronary_arteries.nii.gz'
LA = HEART / 'heart_atrium_left.nii.gz'
LV = HEART / 'heart_ventricle_left.nii.gz'
RA = HEART / 'heart_atrium_right.nii.gz'
RV = HEART / 'heart_ventricle_right.nii.gz'
MYO = HEART / 'heart_myocardium.nii.gz'
PRIOR = Path('Joint_Three_Vessel_Template_Classifier_v1')
PRIOR_RANKING = PRIOR / 'LCX_joint_candidate_ranking.csv'
LEAF_FILES = [PRIOR / f'candidate_{i:02d}_source_path.csv' for i in range(1, 6)]
F2_SOURCE_CANDIDATE_IDS = (6, 7, 9)
STATUS_CONTROL_FAIL = 'F2_RECURSIVE_LOCAL_TANGENT_CONTROL_FAILED'
STATUS_NO_SPLIT = 'F2_SUBTREE_NO_INTERNAL_SPLIT'
STATUS_AMBIG = 'F2_INTERNAL_AV_GROOVE_BRANCH_AMBIGUOUS'
STATUS_CANDIDATE = 'LCX_F2_INTERNAL_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC'

def _req(p):
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=True, default=str), encoding='utf-8')

@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray
    def xyz_to_zyx(self, pts):
        pts=np.atleast_2d(np.asarray(pts,float)); idx=((pts-self.origin)@np.linalg.inv(self.direction).T)/self.spacing
        return idx[:,::-1]
    def zyx_to_xyz(self, pts):
        idx=np.atleast_2d(np.asarray(pts,float))[:,::-1]
        return self.origin+(idx*self.spacing)@self.direction.T

def _source(cache):
    arr=np.load(_req(cache/'series7_int16.npy'),mmap_mode='r')
    m=json.loads(_req(cache/'series7_int16.json').read_text())
    sp=np.asarray(m['spacing_zyx'],float)[::-1]; iop=np.asarray(m['image_orientation_patient'],float); row,col=iop[:3],iop[3:]; slc=np.cross(row,col)
    d=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
    return Geometry(np.asarray(m['positions_lps_mm'][0],float),sp,d),arr

def _sample(arr,g,pts,order=1,cval=0.):
    return map_coordinates(arr,g.xyz_to_zyx(pts).T,order=order,mode='constant',cval=cval)

def _load_path(path,g):
    d=pd.read_csv(_req(path))
    for cols in [('lps_x_mm','lps_y_mm','lps_z_mm'),('x_mm','y_mm','z_mm')]:
        if all(c in d.columns for c in cols): return d[list(cols)].to_numpy(float)
    for cols in [('zyx_z','zyx_y','zyx_x'),('source_z','source_y','source_x'),('z','y','x')]:
        if all(c in d.columns for c in cols): return g.zyx_to_xyz(d[list(cols)].to_numpy(float))
    raise ValueError(f'Unrecognized path columns in {path}: {list(d.columns)}')

def _arc(p):
    p=np.asarray(p,float)
    if len(p)<=1:return np.zeros(len(p))
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def _interp_path(p,q):
    p=np.asarray(p,float); a=_arc(p); q=np.asarray(q,float)
    return np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])

def _resample_path(p,step=.25):
    a=_arc(p)
    if len(a)<2 or a[-1]<=0:return np.asarray(p,float),a
    q=np.arange(0,a[-1]+1e-9,step)
    if q[-1]<a[-1]-1e-6:q=np.r_[q,a[-1]]
    return _interp_path(p,q),q

def _orient_common(paths):
    ref=np.asarray(paths[0],float)[0]; out=[]
    for p in paths:
        p=np.asarray(p,float)
        if np.linalg.norm(p[-1]-ref)<np.linalg.norm(p[0]-ref):p=p[::-1].copy()
        out.append(p)
    return out

def _consensus_prefix(paths,step=.25,tol_mm=.60,sustain=4):
    paths=_orient_common(paths); minlen=min(_arc(p)[-1] for p in paths); q=np.arange(0,minlen+1e-9,step)
    s=np.stack([_interp_path(p,q) for p in paths],axis=0); centroid=np.median(s,axis=0); dev=np.max(np.linalg.norm(s-centroid[None,:,:],axis=2),axis=0)
    first_bad=None; start=max(1,int(round(5.0/step)))
    for i in range(start,max(start,len(q)-sustain+1)):
        if np.all(dev[i:i+sustain]>tol_mm):first_bad=i;break
    end_i=first_bad-1 if first_bad is not None else len(q)-1
    return centroid[:end_i+1],q[:end_i+1],dev[:end_i+1],{'common_prefix_total_mm':float(q[end_i]),'prefix_tolerance_mm':float(tol_mm),'prefix_max_deviation_mm':float(np.max(dev[:end_i+1])),'first_sustained_split_arc_mm':None if first_bad is None else float(q[first_bad])}

def _img_zyx_to_xyz(im,pts):
    idx=np.atleast_2d(np.asarray(pts,float))[:,::-1]; o=np.asarray(im.GetOrigin()); sp=np.asarray(im.GetSpacing()); d=np.asarray(im.GetDirection()).reshape(3,3)
    return o+(idx*sp)@d.T

def _tree(path,surface=False,max_points=250000):
    im=sitk.ReadImage(str(_req(path))); work=sitk.LabelContour(sitk.Cast(im>0,sitk.sitkUInt8),False) if surface else sitk.Cast(im>0,sitk.sitkUInt8)
    a=sitk.GetArrayViewFromImage(work); z=np.argwhere(a>0)
    if not len(z):raise ValueError(f'No foreground in {path}')
    if len(z)>max_points:z=z[::int(math.ceil(len(z)/max_points))]
    xyz=_img_zyx_to_xyz(work,z).astype(np.float32,copy=False); del z,a,work,im; gc.collect(); return cKDTree(xyz)

def _dist_normal(tree,pts):
    p=np.asarray(pts,float); d,idx=tree.query(p); near=np.asarray(tree.data[idx],float); v=p-near; n=np.linalg.norm(v,axis=1); good=n>.15; out=np.zeros_like(v); out[good]=v[good]/n[good,None]
    return d,out,good

def _tangents(p):
    p=np.asarray(p,float); d=np.gradient(p,axis=0); n=np.linalg.norm(d,axis=1); n[n<1e-9]=1; return d/n[:,None]

def _groove(na,nv):
    x=np.cross(na,nv); n=np.linalg.norm(x,axis=1); good=n>1e-6; out=np.zeros_like(x); out[good]=x[good]/n[good,None]; return out,good

def _slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    return 0. if m.sum()<3 or np.ptp(x[m])<1e-6 else float(np.polyfit(x[m],y[m],1)[0])

def _continuity(inc,p):
    pp,_=_resample_path(p,.25); j=min(len(pp)-1,8); v=pp[j]-pp[0]; v/=max(np.linalg.norm(v),1e-9); c=float(np.clip(np.dot(inc,v),-1,1)); ang=float(np.degrees(np.arccos(c)))
    return ang,float(np.exp(-ang/55.))

def _metrics(path,ta,tv,inc=None,max_mm=5.):
    p,q=_resample_path(path,.25); keep=q<=min(q[-1],max_mm)+1e-9; p,q=p[keep],q[keep]; t=_tangents(p); da,na,ga=_dist_normal(ta,p); dv,nv,gv=_dist_normal(tv,p); groove,gg=_groove(na,nv); good=ga&gv&gg
    if good.sum()<3:align=at=vt=0.; med=p90=90.; prof=np.full(len(p),np.nan)
    else:
        dot=np.abs(np.sum(t[good]*groove[good],axis=1)); ang=np.degrees(np.arccos(np.clip(dot,0,1))); align=float(np.median(dot)); med=float(np.median(ang)); p90=float(np.percentile(ang,90)); at=float(np.median(1-np.abs(np.sum(t[good]*na[good],axis=1)))); vt=float(np.median(1-np.abs(np.sum(t[good]*nv[good],axis=1)))); prof=np.full(len(p),np.nan); prof[np.flatnonzero(good)]=ang
    sa=_slope(q,da); ret=float(np.exp(-max(0.,sa)/.35)); ca,cs=(0.,1.) if inc is None else _continuity(inc,p); score=float(.55*align+.15*at+.10*vt+.10*ret+.10*cs)
    return dict(local_geometry_score=score,local_groove_alignment_cos=align,median_groove_tangent_angle_deg=med,p90_groove_tangent_angle_deg=p90,left_atrium_surface_tangency=at,left_ventricle_surface_tangency=vt,left_atrium_distance_slope_mm_per_mm=sa,atrium_retention_score=ret,incoming_continuation_angle_deg=ca,incoming_continuity_score=cs,profile_arc_mm=q,profile_groove_angle_deg=prof,points=p,da=da,dv=dv)

def _control(path,ta,tv):
    p,q=_resample_path(path,.5)
    if len(p)>12:p,q=p[4:-4],q[4:-4]-q[4]
    t=_tangents(p); da,na,ga=_dist_normal(ta,p); dv,nv,gv=_dist_normal(tv,p); groove,gg=_groove(na,nv); good=ga&gv&gg
    if good.sum()<5:return dict(score=0.,median_angle_deg=90.,n_valid=int(good.sum()))
    dot=np.abs(np.sum(t[good]*groove[good],axis=1)); ang=np.degrees(np.arccos(np.clip(dot,0,1))); at=1-np.abs(np.sum(t[good]*na[good],axis=1)); vt=1-np.abs(np.sum(t[good]*nv[good],axis=1)); ret=float(np.exp(-max(0.,_slope(q,da))/.35)); score=float(.65*np.median(dot)+.15*np.median(at)+.10*np.median(vt)+.10*ret)
    return dict(score=score,median_angle_deg=float(np.median(ang)),n_valid=int(good.sum()))

def _support(tree,path):return float(np.mean(tree.query(_resample_path(path,.25)[0])[0]<=1.0))
def _hu(src,g,path):
    h=_sample(src,g,_resample_path(path,.25)[0],1,-1024.); return float(np.mean((h>=120)&(h<=1200))),float(np.median(h))

def _union_find_groups(points,threshold=1.25):
    points=np.asarray(points,float); n=len(points); parent=list(range(n))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:parent[rb]=ra
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(points[i]-points[j])<=threshold:union(i,j)
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

def _plane(g,src,c,n,b,half=7.,step=.2):
    q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing='ij'); pts=c[None,None,:]+xx[...,None]*n+yy[...,None]*b
    return _sample(src,g,pts.reshape(-1,3),1,-1024).reshape(len(q),len(q)),q

def synthetic_family2_self_test():
    t=np.linspace(0,25,101); base=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)]); p6=base.copy(); p9=base.copy(); p7=base.copy(); p7[t>20,1]=(t[t>20]-20)*.8; p6[t>22,2]=(t[t>22]-22)*.8; p9[t>22,2]=(t[t>22]-22)*.8
    _,_,_,m=_consensus_prefix([p6,p7,p9],.25,.4,3); assert 19.5<=m['common_prefix_total_mm']<=20.5; return {'ok':True,'f2_common_mm':m['common_prefix_total_mm']}

def run(drive_root='/content/drive/MyDrive/OpenPlaque',output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/'run_state.json',{'status':'STARTED','algorithm':ALGORITHM,'baseline_commit':BASELINE})
    req=[root/SOURCE_CACHE/'series7_int16.npy',root/SOURCE_CACHE/'series7_int16.json',root/MASTER,root/LAD_PATH,root/RCA_PATH,root/COR_CURRENT,root/COR_LEGACY,root/LA,root/LV,root/RA,root/RV,root/MYO,root/PRIOR_RANKING]+[root/p for p in LEAF_FILES]
    for p in req:_req(p)
    master=json.loads((root/MASTER).read_text()); g,src=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,g); rca=_load_path(root/RCA_PATH,g); all_leaves=_orient_common([_load_path(root/p,g) for p in LEAF_FILES]); ranking=pd.read_csv(root/PRIOR_RANKING)
    mapping=[]
    for i in range(1,6):
        row=ranking.iloc[i-1].to_dict(); mapping.append((i,int(row['candidate_id']),all_leaves[i-1],row))
    selected=[x for x in mapping if x[1] in F2_SOURCE_CANDIDATE_IDS]
    if {x[1] for x in selected}!=set(F2_SOURCE_CANDIDATE_IDS):raise RuntimeError(f'Could not resolve F2 candidates {F2_SOURCE_CANDIDATE_IDS}; got {[x[1] for x in selected]}')
    f2_paths=[x[2] for x in selected]
    print('Building sparse support/surface trees...')
    cur=_tree(root/COR_CURRENT,False); leg=_tree(root/COR_LEGACY,False); la=_tree(root/LA,True); lv=_tree(root/LV,True); ra=_tree(root/RA,True); rv=_tree(root/RV,True); myo=_tree(root/MYO,True)
    _,global_q,_,_=_consensus_prefix(all_leaves,.25,.60,4); f2_con,f2_q,f2_dev,f2_meta=_consensus_prefix(f2_paths,.25,.60,4); global_end=float(global_q[-1]); f2_end=float(f2_q[-1]); extra=max(0.,f2_end-global_end); f2_has_split=f2_meta['first_sustained_split_arc_mm'] is not None and min(_arc(p)[-1]-f2_end for p in f2_paths)>=.75
    pd.DataFrame(f2_con,columns=['lps_x_mm','lps_y_mm','lps_z_mm']).assign(arc_from_candidate_origin_mm=f2_q,max_between_path_deviation_mm=f2_dev).to_csv(out/'family2_consensus_trunk.csv',index=False)
    subtree={**f2_meta,'global_five_path_common_prefix_mm':global_end,'family2_common_prefix_mm':f2_end,'additional_family2_shared_mm_beyond_global_trunk':extra,'family2_internal_split_detected':bool(f2_has_split),'source_candidate_ids':list(F2_SOURCE_CANDIDATE_IDS)}; _write_json(out/'family2_subtree_summary.json',subtree)
    rc=_control(rca,ra,rv); lc=_control(lad,la,lv); cm=float(rc['score']-lc['score']); cpass=bool(rc['score']>=.45 and cm>=.08 and rc['n_valid']>=5); controls={'RCA_right_AV_local_tangent_score':rc['score'],'RCA_median_groove_tangent_angle_deg':rc['median_angle_deg'],'LAD_left_AV_local_tangent_negative_score':lc['score'],'LAD_median_groove_tangent_angle_deg':lc['median_angle_deg'],'control_margin':cm,'control_pass':cpass,'normal_method':'nearest sparse chamber-surface point'}; _write_json(out/'local_tangent_controls.json',controls)
    q0=max(0.,f2_end-2.); incpts=_interp_path(f2_con,[q0,f2_end]); incoming=incpts[-1]-incpts[0]; incoming/=max(np.linalg.norm(incoming),1e-9)
    if f2_has_split:
        rem=min(_arc(p)[-1]-f2_end for p in f2_paths); probe=f2_end+max(.75,min(2.,.5*rem)); probe_points=np.vstack([_interp_path(p,[probe])[0] for p in f2_paths]); groups=_union_find_groups(probe_points,1.25)
    else:probe=f2_end; groups=[list(range(len(f2_paths)))]
    fam={}
    for fi,grp in enumerate(groups,1):
        for idx in grp:fam[idx]=fi
    rows=[]; profiles={}
    for idx,(leaf_index,candidate_id,p,prior) in enumerate(selected):
        total=_arc(p)[-1]; start=min(f2_end,total-.5); q=np.arange(start,total+1e-9,.25)
        if len(q)<3:q=np.linspace(start,total,max(3,int(round((total-start)/.25))+1))
        cont=_interp_path(p,q); m=_metrics(cont,la,lv,incoming,5.); csup,lsup=_support(cur,cont),_support(leg,cont); hfrac,hmed=_hu(src,g,cont); md=myo.query(m['points'])[0]; control_position=(m['local_geometry_score']-lc['score'])/max(cm,1e-6)
        rows.append({'leaf_index':leaf_index,'source_candidate_id':candidate_id,'subfamily_id':fam[idx],'prior_LCX_score':float(prior['LCX_score']),'prior_LCX_margin':float(prior['LCX_margin']),'continuation_length_mm':float(_arc(cont)[-1]),'local_geometry_score':m['local_geometry_score'],'control_axis_position':control_position,'margin_over_LAD_negative':m['local_geometry_score']-lc['score'],'local_groove_alignment_cos':m['local_groove_alignment_cos'],'median_groove_tangent_angle_deg':m['median_groove_tangent_angle_deg'],'p90_groove_tangent_angle_deg':m['p90_groove_tangent_angle_deg'],'left_atrium_surface_tangency':m['left_atrium_surface_tangency'],'left_ventricle_surface_tangency':m['left_ventricle_surface_tangency'],'left_atrium_distance_slope_mm_per_mm':m['left_atrium_distance_slope_mm_per_mm'],'atrium_retention_score':m['atrium_retention_score'],'incoming_continuation_angle_deg':m['incoming_continuation_angle_deg'],'incoming_continuity_score':m['incoming_continuity_score'],'median_left_atrium_surface_mm':float(np.median(m['da'])),'median_left_ventricle_surface_mm':float(np.median(m['dv'])),'median_myocardium_surface_mm':float(np.median(md)),'current_support_fraction':csup,'legacy_support_fraction':lsup,'robust_hu_fraction':hfrac,'median_hu':hmed}); profiles[candidate_id]=m
    df=pd.DataFrame(rows).sort_values('local_geometry_score',ascending=False).reset_index(drop=True); df.insert(0,'rank',np.arange(1,len(df)+1)); df.to_csv(out/'family2_leaf_scores.csv',index=False)
    fam_rows=[]
    for fi in sorted(df.subfamily_id.unique()):
        x=df[df.subfamily_id==fi]; fam_rows.append({'subfamily_id':int(fi),'n_members':len(x),'member_source_candidate_ids':';'.join(str(int(v)) for v in sorted(x.source_candidate_id)),'median_local_score':float(x.local_geometry_score.median()),'best_local_score':float(x.local_geometry_score.max()),'median_control_axis_position':float(x.control_axis_position.median()),'median_groove_alignment':float(x.local_groove_alignment_cos.median()),'median_tangent_angle_deg':float(x.median_groove_tangent_angle_deg.median()),'median_continuity_angle_deg':float(x.incoming_continuation_angle_deg.median()),'min_current_support':float(x.current_support_fraction.min()),'min_legacy_support':float(x.legacy_support_fraction.min()),'min_robust_hu_fraction':float(x.robust_hu_fraction.min())})
    fdf=pd.DataFrame(fam_rows).sort_values('median_local_score',ascending=False).reset_index(drop=True); fdf.insert(0,'subfamily_rank',np.arange(1,len(fdf)+1)); fdf['subfamily_gate_pass']=(fdf['median_control_axis_position']>=.30)&(fdf['median_continuity_angle_deg']<=25)&(fdf['min_current_support']>=.90)&(fdf['min_legacy_support']>=.90)&(fdf['min_robust_hu_fraction']>=.90); fdf.to_csv(out/'family2_subfamily_scores.csv',index=False)
    margin=float(fdf.iloc[0].median_local_score-fdf.iloc[1].median_local_score) if len(fdf)>1 else 0.; winner=bool(f2_has_split and cpass and len(fdf)>1 and bool(fdf.iloc[0].subfamily_gate_pass) and margin>=.05)
    status=STATUS_NO_SPLIT if not f2_has_split else STATUS_CONTROL_FAIL if not cpass else STATUS_CANDIDATE if winner else STATUS_AMBIG
    plt.figure(figsize=(7,4)); plt.bar([str(int(x)) for x in fdf.subfamily_id],fdf.median_local_score); plt.axhline(lc['score'],ls='--',label='LAD negative'); plt.axhline(rc['score'],ls=':',label='RCA positive'); plt.xlabel('F2 internal subfamily'); plt.ylabel('median local geometry score'); plt.title(f'F2 recursive AV-groove adjudication | margin={margin:.3f}'); plt.legend(); plt.tight_layout(); plt.savefig(out/'01_family2_subfamily_scores.png',dpi=180); plt.close()
    plt.figure(figsize=(8,5))
    for cid,m in profiles.items():plt.plot(m['profile_arc_mm'],m['profile_groove_angle_deg'],marker='o',ms=2,label=f'C{cid}')
    plt.axhline(rc['median_angle_deg'],ls=':',label='RCA control median'); plt.axhline(lc['median_angle_deg'],ls='--',label='LAD control median'); plt.xlabel('arc after F2 internal split (mm)'); plt.ylabel('|angle to local AV-groove tangent| (deg)'); plt.ylim(0,90); plt.legend(); plt.tight_layout(); plt.savefig(out/'02_family2_tangent_angle_profiles.png',dpi=180); plt.close()
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d'); ax.plot(f2_con[:,0],f2_con[:,1],f2_con[:,2],c='k',lw=3,label='F2 shared trunk')
    for _,cid,p,_ in selected:
        q=np.linspace(f2_end,min(_arc(p)[-1],f2_end+7),40); s=_interp_path(p,q); ax.plot(s[:,0],s[:,1],s[:,2],lw=2,label=f'C{cid}')
    ax.scatter(f2_con[-1,0],f2_con[-1,1],f2_con[-1,2],s=50,c='k'); ax.set_title('Family-2 internal split'); ax.legend(); plt.tight_layout(); plt.savefig(out/'03_family2_internal_split_geometry.png',dpi=180); plt.close()
    top_ids=[int(x) for x in df.source_candidate_id.head(2)]; fig,axes=plt.subplots(2,3,figsize=(12,8))
    for r,cid in enumerate(top_ids):
        p=next(x for x in selected if x[1]==cid)[2]; total=_arc(p)[-1]; qs=np.linspace(f2_end,min(total,f2_end+5),3)
        for j,qa in enumerate(qs):
            pts=_interp_path(p,[max(0,qa-.5),qa,min(total,qa+.5)]); t=pts[-1]-pts[0]; t/=max(np.linalg.norm(t),1e-9); axes3=np.eye(3); seed=axes3[np.argmin(np.abs(axes3@t))]; n=np.cross(t,seed); n/=max(np.linalg.norm(n),1e-9); b=np.cross(t,n); b/=max(np.linalg.norm(b),1e-9); im,qv=_plane(g,src,pts[1],n,b); ax=axes[r,j]; ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=25); ax.set_title(f'C{cid} +{qa-f2_end:.1f} mm'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    plt.tight_layout(); plt.savefig(out/'04_family2_top_candidates_orthogonal_qc.png',dpi=180); plt.close()
    summary={'status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'master_status':master.get('status'),'LCX_master_status':'UNRESOLVED','family2_subtree':subtree,'local_tangent_controls':controls,'n_internal_subfamilies':int(len(fdf)),'top_subfamily':None if fdf.empty else fdf.iloc[0].to_dict(),'top_subfamily_margin':margin,'top_leaf':None if df.empty else df.iloc[0].to_dict(),'template_similarity_used_in_decision':False,'decision_rule':'same 0.05 family-separation margin reapplied one level deeper; top subgroup must pass patient-control-axis, continuity, source-support, and HU gates','scientific_boundary':'This can nominate an LCX-like continuation after the Family-2 internal split but does not establish clinical vessel identity; LM remains unresolved.'}; _write_json(out/'summary.json',summary)
    report=out/'OPENPLAQUE_LCX_FAMILY2_RECURSIVE_AV_GROOVE_REPORT.html'; report.write_text(f"<html><body><h1>OpenPlaque LCX Family-2 Recursive AV-Groove Adjudication</h1><p><b>Status:</b> {status}</p><p>F2 common prefix: {f2_end:.3f} mm; additional beyond five-path trunk: {extra:.3f} mm.</p><p>Control: RCA {rc['score']:.3f}, LAD {lc['score']:.3f}, margin {cm:.3f}, pass {cpass}.</p><p>Internal subfamily margin: {margin:.3f}.</p><p>Template similarity has zero decision weight.</p></body></html>",encoding='utf-8')
    _write_json(out/'run_state.json',{'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE}); zp=out/'OPENPLAQUE_LCX_FAMILY2_RECURSIVE_AV_GROOVE_REPORT_BACK.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p!=zp and p.is_file():z.write(p,p.name)
    return {'summary':summary,'report':str(report),'zip':str(zp)}
