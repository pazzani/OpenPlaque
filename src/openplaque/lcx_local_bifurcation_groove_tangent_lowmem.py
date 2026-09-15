from __future__ import annotations
import gc, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from openplaque.lcx_local_bifurcation_groove_tangent import (
    BASELINE, SOURCE_CACHE, MASTER, LAD_PATH, RCA_PATH, COR_CURRENT, COR_LEGACY,
    LA, LV, RA, RV, MYO, PRIOR_RANKING, LEAF_FILES,
    _req, _write_json, _arc, _interp_path, _resample_path, _orient_common,
    _consensus_prefix, _nearest_dist, _first_sustained, _union_find_groups, _frame,
)
ALGORITHM='lcx-local-bifurcation-groove-tangent-v1.1-lowmem'
OUTPUT_DIRNAME='LCX_Local_Bifurcation_Groove_Tangent_v1_lowmem'
STATUS_NO_TRUNK='NO_CONSENSUS_LEFT_CORONARY_BRANCH_TRUNK'
STATUS_CONTROL_FAIL='LOCAL_AV_GROOVE_TANGENT_CONTROL_FAILED'
STATUS_AMBIG='CONSENSUS_TRUNK_ESTABLISHED_LOCAL_BIFURCATION_AMBIGUOUS'
STATUS_CANDIDATE='LCX_LOCAL_GROOVE_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC'

@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray; spacing: np.ndarray; direction: np.ndarray
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
    raise ValueError(f'Unrecognized path columns: {list(d.columns)}')

def _img_zyx_to_xyz(im,pts):
    idx=np.atleast_2d(np.asarray(pts,float))[:,::-1]; o=np.asarray(im.GetOrigin()); sp=np.asarray(im.GetSpacing()); d=np.asarray(im.GetDirection()).reshape(3,3)
    return o+(idx*sp)@d.T

def _tree(path,surface=False,max_points=250000):
    im=sitk.ReadImage(str(_req(path))); work=sitk.LabelContour(sitk.Cast(im>0,sitk.sitkUInt8),False) if surface else sitk.Cast(im>0,sitk.sitkUInt8)
    a=sitk.GetArrayViewFromImage(work); z=np.argwhere(a>0)
    if not len(z): raise ValueError(f'No foreground in {path}')
    if len(z)>max_points: z=z[::int(math.ceil(len(z)/max_points))]
    xyz=_img_zyx_to_xyz(work,z).astype(np.float32,copy=False); del z,a,work,im; gc.collect(); return cKDTree(xyz)

def _dist_normal(tree,pts):
    p=np.asarray(pts,float); d,idx=tree.query(p); near=np.asarray(tree.data[idx],float); v=p-near; n=np.linalg.norm(v,axis=1); good=n>.15; out=np.zeros_like(v); out[good]=v[good]/n[good,None]; return d,out,good

def _tangents(p):
    p=np.asarray(p,float); d=np.gradient(p,axis=0); n=np.linalg.norm(d,axis=1); n[n<1e-9]=1; return d/n[:,None]

def _groove(na,nv):
    x=np.cross(na,nv); n=np.linalg.norm(x,axis=1); good=n>1e-6; out=np.zeros_like(x); out[good]=x[good]/n[good,None]; return out,good

def _slope(x,y):
    x=np.asarray(x); y=np.asarray(y); m=np.isfinite(x)&np.isfinite(y)
    return 0. if m.sum()<3 or np.ptp(x[m])<1e-6 else float(np.polyfit(x[m],y[m],1)[0])

def _continuity(inc,p):
    q,_=_resample_path(p,.25); j=min(len(q)-1,8); v=q[j]-q[0]; v/=max(np.linalg.norm(v),1e-9); c=np.clip(np.dot(inc,v),-1,1); ang=float(np.degrees(np.arccos(c))); return ang,float(np.exp(-ang/55.))

def _metrics(path,ta,tv,inc=None,max_mm=4.):
    p,q=_resample_path(path,.25); keep=q<=min(q[-1],max_mm)+1e-9; p,q=p[keep],q[keep]; t=_tangents(p); da,na,ga=_dist_normal(ta,p); dv,nv,gv=_dist_normal(tv,p); g,gg=_groove(na,nv); good=ga&gv&gg
    if good.sum()<3: align=at=vt=0.; med=p90=90.; prof=np.full(len(p),np.nan)
    else:
        dot=np.abs(np.sum(t[good]*g[good],axis=1)); ang=np.degrees(np.arccos(np.clip(dot,0,1))); align=float(np.median(dot)); med=float(np.median(ang)); p90=float(np.percentile(ang,90)); at=float(np.median(1-np.abs(np.sum(t[good]*na[good],axis=1)))); vt=float(np.median(1-np.abs(np.sum(t[good]*nv[good],axis=1)))); prof=np.full(len(p),np.nan); prof[np.flatnonzero(good)]=ang
    sa=_slope(q,da); ret=float(np.exp(-max(0.,sa)/.35)); ca,cs=(0.,1.) if inc is None else _continuity(inc,p); score=float(.55*align+.15*at+.10*vt+.10*ret+.10*cs)
    return dict(local_geometry_score=score,local_groove_alignment_cos=align,median_groove_tangent_angle_deg=med,p90_groove_tangent_angle_deg=p90,left_atrium_surface_tangency=at,left_ventricle_surface_tangency=vt,left_atrium_distance_slope_mm_per_mm=sa,atrium_retention_score=ret,incoming_continuation_angle_deg=ca,incoming_continuity_score=cs,profile_arc_mm=q,profile_groove_angle_deg=prof,points=p,da=da,dv=dv)

def _control(path,ta,tv):
    p,q=_resample_path(path,.5)
    if len(p)>12: p,q=p[4:-4],q[4:-4]-q[4]
    t=_tangents(p); da,na,ga=_dist_normal(ta,p); dv,nv,gv=_dist_normal(tv,p); g,gg=_groove(na,nv); good=ga&gv&gg
    if good.sum()<5:return dict(score=0.,median_angle_deg=90.,n_valid=int(good.sum()))
    dot=np.abs(np.sum(t[good]*g[good],axis=1)); ang=np.degrees(np.arccos(np.clip(dot,0,1))); at=1-np.abs(np.sum(t[good]*na[good],axis=1)); vt=1-np.abs(np.sum(t[good]*nv[good],axis=1)); ret=float(np.exp(-max(0.,_slope(q,da))/.35)); score=float(.65*np.median(dot)+.15*np.median(at)+.10*np.median(vt)+.10*ret)
    return dict(score=score,median_angle_deg=float(np.median(ang)),n_valid=int(good.sum()))

def _support(tree,path): return float(np.mean(tree.query(_resample_path(path,.25)[0])[0]<=1.0))
def _hu(src,g,path):
    h=_sample(src,g,_resample_path(path,.25)[0],1,-1024.); return float(np.mean((h>=120)&(h<=1200))),float(np.median(h))

def _plane(g,src,c,n,b,half=7.,step=.2):
    q=np.arange(-half,half+1e-9,step); yy,xx=np.meshgrid(q,q,indexing='ij'); pts=c[None,None,:]+xx[...,None]*n+yy[...,None]*b; return _sample(src,g,pts.reshape(-1,3),1,-1024).reshape(len(q),len(q)),q

def synthetic_local_bifurcation_self_test():
    t=np.linspace(0,20,81); p=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)]); p2=p.copy(); p3=p.copy(); p2[t>15,1]=(t[t>15]-15)*.8; p3[t>15,2]=(t[t>15]-15)*.8; *_,m=_consensus_prefix([p,p2,p3],.25,.4,3); assert 14.5<=m['common_prefix_total_mm']<=15.5; return {'ok':True,'consensus_mm':m['common_prefix_total_mm']}

def run(drive_root='/content/drive/MyDrive/OpenPlaque',output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True); _write_json(out/'run_state.json',{'status':'STARTED','algorithm':ALGORITHM,'baseline_commit':BASELINE})
    req=[root/SOURCE_CACHE/'series7_int16.npy',root/SOURCE_CACHE/'series7_int16.json',root/MASTER,root/LAD_PATH,root/RCA_PATH,root/COR_CURRENT,root/COR_LEGACY,root/LA,root/LV,root/RA,root/RV,root/MYO,root/PRIOR_RANKING]+[root/p for p in LEAF_FILES]
    for p in req:_req(p)
    master=json.loads((root/MASTER).read_text()); g,src=_source(root/SOURCE_CACHE); lad=_load_path(root/LAD_PATH,g); rca=_load_path(root/RCA_PATH,g); leaves=_orient_common([_load_path(root/p,g) for p in LEAF_FILES]); rank=pd.read_csv(root/PRIOR_RANKING); prior={i:rank.iloc[i-1].to_dict() for i in range(1,6)}
    print('Building sparse support/surface trees...')
    cur=_tree(root/COR_CURRENT,False); leg=_tree(root/COR_LEGACY,False); la=_tree(root/LA,True); lv=_tree(root/LV,True); ra=_tree(root/RA,True); rv=_tree(root/RV,True); myo=_tree(root/MYO,True)
    con,qc,dev,cm=_consensus_prefix(leaves,.25,.60,4); dl=_nearest_dist(con,lad); di=_first_sustained(dl,1.5,.25,1.0); di=len(con)-1 if di is None else di; trunk=con[di:]; ta=_arc(trunk); tc,tl=_support(cur,trunk),_support(leg,trunk); hf,hm=_hu(src,g,trunk); tok=bool(ta[-1]>=10 and tc>=.9 and tl>=.9 and hf>=.9); pd.DataFrame(trunk,columns=['lps_x_mm','lps_y_mm','lps_z_mm']).assign(arc_mm=ta,lad_distance_mm=dl[di:]).to_csv(out/'consensus_left_coronary_branch_trunk.csv',index=False)
    cs={**cm,'lad_divergence_arc_from_candidate_origin_mm':float(qc[di]),'post_lad_divergence_common_trunk_mm':float(ta[-1]),'current_support_fraction':tc,'legacy_support_fraction':tl,'robust_hu_fraction':hf,'median_hu':hm,'consensus_trunk_gate_pass':tok}; _write_json(out/'consensus_trunk_summary.json',cs)
    ce=float(qc[-1]); rem=min(_arc(p)[-1]-ce for p in leaves); pa=ce+max(1.,min(4.,.5*rem)); pp=np.vstack([_interp_path(p,[pa])[0] for p in leaves]); groups=_union_find_groups(pp,1.5); fam={}; [fam.__setitem__(i,fi) for fi,gr in enumerate(groups,1) for i in gr]
    inc=_interp_path(con,[max(0.,ce-2),ce]); it=inc[-1]-inc[0]; it/=max(np.linalg.norm(it),1e-9)
    rc=_control(rca,ra,rv); lc=_control(lad,la,lv); margin=float(rc['score']-lc['score']); cpass=bool(rc['score']>=.45 and margin>=.08 and rc['n_valid']>=5); controls={'RCA_right_AV_local_tangent_score':rc['score'],'RCA_median_groove_tangent_angle_deg':rc['median_angle_deg'],'LAD_left_AV_local_tangent_negative_score':lc['score'],'LAD_median_groove_tangent_angle_deg':lc['median_angle_deg'],'control_margin':margin,'control_pass':cpass,'normal_method':'nearest sparse chamber-surface point'}; _write_json(out/'local_tangent_controls.json',controls)
    rows=[]; profiles={}
    for i,p in enumerate(leaves):
        total=_arc(p)[-1]; start=min(ce,total-.5); q=np.arange(start,total+1e-9,.25); q=np.linspace(start,total,max(3,int(round((total-start)/.25))+1)) if len(q)<3 else q; cont=_interp_path(p,q); m=_metrics(cont,la,lv,it,4.); csup,lsup=_support(cur,cont),_support(leg,cont); hfrac,hmed=_hu(src,g,cont); md=myo.query(m['points'])[0]; pr=prior[i+1]; rel=m['local_geometry_score']/max(rc['score'],1e-6); above=m['local_geometry_score']-lc['score']; gate=bool(cpass and csup>=.9 and lsup>=.9 and hfrac>=.9 and rel>=.7 and above>=.05 and m['local_groove_alignment_cos']>=.45 and m['atrium_retention_score']>=.55)
        rows.append({'leaf_index':i+1,'source_candidate_id':int(pr['candidate_id']),'family_id':fam[i],'prior_LCX_score':float(pr['LCX_score']),'prior_LCX_margin':float(pr['LCX_margin']),'continuation_length_mm':float(_arc(cont)[-1]),'local_geometry_score':m['local_geometry_score'],'relative_to_RCA_control':rel,'margin_over_LAD_negative':above,'local_groove_alignment_cos':m['local_groove_alignment_cos'],'median_groove_tangent_angle_deg':m['median_groove_tangent_angle_deg'],'p90_groove_tangent_angle_deg':m['p90_groove_tangent_angle_deg'],'left_atrium_surface_tangency':m['left_atrium_surface_tangency'],'left_ventricle_surface_tangency':m['left_ventricle_surface_tangency'],'left_atrium_distance_slope_mm_per_mm':m['left_atrium_distance_slope_mm_per_mm'],'atrium_retention_score':m['atrium_retention_score'],'incoming_continuation_angle_deg':m['incoming_continuation_angle_deg'],'incoming_continuity_score':m['incoming_continuity_score'],'median_left_atrium_surface_mm':float(np.median(m['da'])),'median_left_ventricle_surface_mm':float(np.median(m['dv'])),'median_myocardium_surface_mm':float(np.median(md)),'current_support_fraction':csup,'legacy_support_fraction':lsup,'robust_hu_fraction':hfrac,'median_hu':hmed,'local_gate_pass':gate}); profiles[i+1]=m
    df=pd.DataFrame(rows).sort_values('local_geometry_score',ascending=False).reset_index(drop=True); df.insert(0,'local_rank',np.arange(1,len(df)+1)); df.to_csv(out/'local_bifurcation_leaf_scores.csv',index=False)
    fr=[]
    for fi,gr in enumerate(groups,1):
        s=df[df.family_id==fi]; fr.append({'family_id':fi,'n_leaf_members':len(s),'family_local_score_median':float(s.local_geometry_score.median()),'family_local_score_best':float(s.local_geometry_score.max()),'family_groove_alignment_median':float(s.local_groove_alignment_cos.median()),'family_tangent_angle_median_deg':float(s.median_groove_tangent_angle_deg.median()),'family_continuity_angle_median_deg':float(s.incoming_continuation_angle_deg.median()),'family_gate_pass':bool(s.local_gate_pass.all()),'member_leaf_indices':';'.join(map(str,sorted(s.leaf_index.astype(int)))),'member_source_candidate_ids':';'.join(map(str,sorted(s.source_candidate_id.astype(int))))})
    fd=pd.DataFrame(fr).sort_values('family_local_score_median',ascending=False).reset_index(drop=True); fd.insert(0,'family_rank',np.arange(1,len(fd)+1)); fd.to_csv(out/'local_bifurcation_family_scores.csv',index=False); top=fd.iloc[0].to_dict() if len(fd) else None; fmargin=float(fd.iloc[0].family_local_score_median-fd.iloc[1].family_local_score_median) if len(fd)>1 else float('inf'); status=STATUS_NO_TRUNK if not tok else STATUS_CONTROL_FAIL if not cpass else STATUS_CANDIDATE if top and bool(top['family_gate_pass']) and fmargin>=.05 else STATUS_AMBIG
    plt.figure(figsize=(8,5)); x=np.arange(len(fd)); plt.bar(x,fd.family_local_score_median); plt.xticks(x,[f'Family {int(v)}' for v in fd.family_id]); plt.axhline(rc['score'],ls='--',label='RCA control'); plt.axhline(lc['score'],ls=':',label='LAD negative'); plt.legend(); plt.ylabel('Local geometry score'); plt.title(status); plt.tight_layout(); plt.savefig(out/'01_local_bifurcation_family_scores.png',dpi=170); plt.close()
    plt.figure(figsize=(9,6));
    for _,r in df.iterrows():
        m=profiles[int(r.leaf_index)]; plt.plot(m['profile_arc_mm'],m['profile_groove_angle_deg'],label=f"C{int(r.source_candidate_id)} F{int(r.family_id)}")
    plt.xlabel('Arc after split (mm)'); plt.ylabel('|angle to LA-LV groove tangent| (deg)'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/'02_local_groove_tangent_angle_profiles.png',dpi=170); plt.close()
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d'); ax.plot(trunk[:,0],trunk[:,1],trunk[:,2],lw=4,label='Consensus trunk');
    for _,r in df.iterrows():
        p=profiles[int(r.leaf_index)]['points']; ax.plot(p[:,0],p[:,1],p[:,2],lw=2,label=f"C{int(r.source_candidate_id)} / F{int(r.family_id)}")
    ax.legend(fontsize=8); ax.set_title('Local distal bifurcation geometry'); plt.tight_layout(); plt.savefig(out/'03_local_bifurcation_geometry.png',dpi=170); plt.close(fig)
    top2=df.head(2); fig,axes=plt.subplots(max(1,len(top2)),3,figsize=(12,4*max(1,len(top2))),squeeze=False)
    for rr,(_,r) in enumerate(top2.iterrows()):
        p=profiles[int(r.leaf_index)]['points']
        for cc,f in enumerate([.1,.5,.9]):
            ii=min(len(p)-1,max(0,int(round(f*(len(p)-1))))); n,b=_frame(p,ii); ct,qg=_plane(g,src,p[ii],n,b); axes[rr,cc].imshow(ct,cmap='gray',vmin=-100,vmax=900,extent=[qg[0],qg[-1],qg[-1],qg[0]]); axes[rr,cc].scatter([0],[0],marker='+'); axes[rr,cc].set_title(f"C{int(r.source_candidate_id)} F{int(r.family_id)} {f:.0%}\nscore {r.local_geometry_score:.3f}")
    fig.tight_layout(); fig.savefig(out/'04_top_local_candidates_orthogonal_qc.png',dpi=170); plt.close(fig)
    summary={'status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'master_status':master.get('status'),'LCX_master_status':'UNRESOLVED','consensus_trunk':cs,'local_tangent_controls':controls,'n_families':len(fd),'top_family':top,'top_family_margin':fmargin,'top_leaf':df.iloc[0].to_dict() if len(df) else None,'template_similarity_used_in_decision':False,'memory_strategy':'source memmap + sparse point-cloud KD-trees; no full-volume distance transforms/gradients','scientific_boundary':'Local geometry can nominate an LCX-like continuation but does not establish clinical vessel identity without visual/anatomic review. LM remains unresolved.'}; _write_json(out/'summary.json',summary)
    report=out/'OPENPLAQUE_LCX_LOCAL_BIFURCATION_GROOVE_TANGENT_REPORT.html'; report.write_text(f"<html><body><h1>OpenPlaque LCX local bifurcation / groove tangent</h1><p><b>Status:</b> {status}</p><p><b>Post-LAD consensus trunk:</b> {cs['post_lad_divergence_common_trunk_mm']:.2f} mm</p><p><b>RCA control:</b> {rc['score']:.3f}; <b>LAD negative:</b> {lc['score']:.3f}; margin {margin:.3f}</p><p><b>Top family margin:</b> {fmargin:.3f}</p><p>Low-memory sparse-surface implementation. Curved-template scores are metadata only.</p><img src='01_local_bifurcation_family_scores.png' style='max-width:95%'><br><img src='02_local_groove_tangent_angle_profiles.png' style='max-width:95%'><br><img src='03_local_bifurcation_geometry.png' style='max-width:95%'><br><img src='04_top_local_candidates_orthogonal_qc.png' style='max-width:95%'></body></html>",encoding='utf-8')
    _write_json(out/'run_state.json',{'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE}); zp=out/'OPENPLAQUE_LCX_LOCAL_BIFURCATION_GROOVE_TANGENT_REPORT_BACK.zip';
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.name!=zp.name:z.write(p,arcname=p.name)
    return {'summary':summary,'report':str(report),'zip':str(zp)}
