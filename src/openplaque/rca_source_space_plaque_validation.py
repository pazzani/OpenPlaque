from __future__ import annotations
import json, math, zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.spatial import cKDTree

BASELINE='0593b453959f5a353d644267fbeef24b514ef4d7'
ALGORITHM='rca-source-space-plaque-validation-v1.0'
OUTPUT_DIRNAME='RCA_Source_Space_Plaque_Validation_v1'
SRC=Path('Cache/Secondary_3D_Vesselness_Topology_v1')
CL=Path('Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv')
MASTER=Path('Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
FUSION=Path('Longitudinal_Plaque_PCAT_Fusion_v1/RCA_longitudinal_plaque_intervals.csv')
TS=Path('TotalSegmentator_Cardiovascular_Cache_v1')
CUR=TS/'coronary_arteries/coronary_arteries.nii.gz'
LEG=TS/'coronary_arteries_LEGACY/coronary_arteries.nii.gz'
AORTA=TS/'total/aorta.nii.gz'

def _json(p,x): Path(p).write_text(json.dumps(x,indent=2,default=str,allow_nan=True),encoding='utf-8')
def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _source(cache):
    a=np.load(_req(cache/'series7_int16.npy'),mmap_mode='r'); m=json.loads(_req(cache/'series7_int16.json').read_text())
    im=sitk.GetImageFromArray(np.asarray(a)); sp=np.asarray(m['spacing_zyx'],float); im.SetSpacing(tuple(sp[::-1])); im.SetOrigin(tuple(np.asarray(m['positions_lps_mm'][0],float)))
    iop=np.asarray(m['image_orientation_patient'],float); r,c=iop[:3],iop[3:]; s=np.cross(r,c); d=np.array([[r[0],c[0],s[0]],[r[1],c[1],s[1]],[r[2],c[2],s[2]]]); im.SetDirection(tuple(d.ravel()))
    return im,a

def _mask(path,ref):
    im=sitk.ReadImage(str(_req(path)))
    same=im.GetSize()==ref.GetSize() and np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection())
    if not same: im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im)>0

def _load_cl(p):
    d=pd.read_csv(_req(p)).copy(); opts=[('lps_x_mm','lps_y_mm','lps_z_mm'),('x_mm','y_mm','z_mm'),('x','y','z')]; cols=next((x for x in opts if all(c in d for c in x)),None)
    if cols is None: raise ValueError('No LPS XYZ columns')
    pts=d[list(cols)].to_numpy(float)
    if 'arc_mm' not in d: d['arc_mm']=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    d=d.sort_values('arc_mm').drop_duplicates('arc_mm').reset_index(drop=True)
    return d.rename(columns={cols[0]:'lps_x_mm',cols[1]:'lps_y_mm',cols[2]:'lps_z_mm'})

def _resample(d,step=.2):
    a=d.arc_mm.to_numpy(float); p=d[['lps_x_mm','lps_y_mm','lps_z_mm']].to_numpy(float); q=np.arange(a.min(),a.max()+1e-6,step)
    if q[-1]<a.max()-1e-6: q=np.r_[q,a.max()]
    return q,np.column_stack([np.interp(q,a,p[:,k]) for k in range(3)])

def _p2z(im,p):
    o=np.asarray(im.GetOrigin()); sp=np.asarray(im.GetSpacing()); D=np.asarray(im.GetDirection()).reshape(3,3); ix=((np.asarray(p)-o)@np.linalg.inv(D).T)/sp; return ix[:,::-1]
def _z2p(im,z):
    ix=np.asarray(z,float)[:,::-1]; o=np.asarray(im.GetOrigin()); sp=np.asarray(im.GetSpacing()); D=np.asarray(im.GetDirection()).reshape(3,3); return o+(ix*sp)@D.T
def _sample(im,a,p,order=1,cval=-1024): return map_coordinates(np.asarray(a),_p2z(im,p).T,order=order,mode='constant',cval=cval)

def _validate_master(m,L):
    if m.get('status')!='CORONARY_ANATOMY_BASELINE_V2_FROZEN': raise RuntimeError('Master Anatomy v2 not frozen')
    x=float(m['accepted']['RCA_length_mm'])
    if abs(x-L)>.5: raise RuntimeError(f'RCA length mismatch {L:.3f} vs {x:.3f}')
    if not {'LM','LCX'}.issubset(set(m.get('unresolved',[]))): raise RuntimeError('LM/LCX status changed')
    return {'master_rca_length_mm':x,'input_rca_length_mm':L,'difference_mm':L-x}

def _positive_windows(d):
    q=d[d.confidence_level.eq('majority_3plus')].sort_values(['mapped_native_voxels_vote_ge3','duration_mm'],ascending=False)
    if q.empty: raise RuntimeError('No RCA majority intervals')
    return [{'label':'primary_model_positive' if i==0 else f'secondary_model_positive_{i}','kind':'model_positive','arc_start_mm':float(r.arc_start_mm),'arc_end_mm':float(r.arc_end_mm)} for i,r in enumerate(q.itertuples())]
def _overlap(a,b,c,d,pad=0): return max(a-pad,c-pad)<min(b+pad,d+pad)
def _controls(L,pos):
    out=[]
    for a,b in [(20,25),(35,40),(25,30),(40,45),(15,20),(45,50)]:
        if b>L-.25 or any(_overlap(a,b,x['arc_start_mm'],x['arc_end_mm'],2) for x in pos): continue
        out.append({'label':f'negative_control_{len(out)+1}','kind':'negative_control','arc_start_mm':float(a),'arc_end_mm':float(b)})
        if len(out)==2:return out
    raise RuntimeError('No control windows')

def _bbox(im,p,margin=6):
    z=_p2z(im,p); sp=np.asarray(im.GetSpacing()[::-1]); m=np.ceil(margin/sp).astype(int); lo=np.floor(z.min(0)).astype(int)-m; hi=np.ceil(z.max(0)).astype(int)+m+1; sh=np.asarray(sitk.GetArrayViewFromImage(im).shape); lo=np.maximum(lo,0); hi=np.minimum(hi,sh); return tuple(slice(int(lo[k]),int(hi[k])) for k in range(3)),lo

def _components(mask,hu,im,lo,tree,arc,vv,minvol=.15,minspan=.4):
    lab,n=label(mask,structure=np.ones((3,3,3),int)); rows=[]; keep=np.zeros_like(mask,bool)
    for i in range(1,n+1):
        z=np.argwhere(lab==i); g=z+lo; phys=_z2p(im,g); dist,near=tree.query(phys); aa=arc[near]; vals=hu[lab==i]; vol=len(z)*vv; span=aa.max()-aa.min()
        ok=vol>=minvol and span>=minspan
        rows.append({'component_id':i,'voxel_count':len(z),'volume_mm3':vol,'arc_start_mm':aa.min(),'arc_end_mm':aa.max(),'arc_span_mm':span,'median_hu':np.median(vals),'p90_hu':np.quantile(vals,.9),'max_hu':vals.max(),'median_centerline_distance_mm':np.median(dist),'passes_component_gate':ok})
        if ok: keep|=lab==i
    return pd.DataFrame(rows),keep

def _decision(vol,span,cd,pd):
    if vol<=0 or span<.4:return 'NO_STRICT_SOURCE_CALCIFIC_CANDIDATE_IN_PRIMARY'
    if span>=.5 and vol>=.2 and pd/max(cd,.02)>=3:return 'SOURCE_CALCIFIC_CANDIDATE_ENRICHED_PRIMARY_REQUIRES_VISUAL_QC'
    return 'SOURCE_CALCIFIC_CANDIDATE_PRESENT_PRIMARY_REQUIRES_VISUAL_QC'
def synthetic_source_plaque_self_test():
    c=_controls(52,[{'arc_start_mm':0,'arc_end_mm':5}]); a=_decision(1.2,1.1,.05,.4); b=_decision(0,0,.05,0); return {'passed':len(c)==2 and a.startswith('SOURCE_CALCIFIC_CANDIDATE_ENRICHED') and b.startswith('NO_STRICT'),'controls':c,'algorithm':ALGORITHM}

def _frame(p,i):
    t=p[min(len(p)-1,i+2)]-p[max(0,i-2)]; t/=max(np.linalg.norm(t),1e-9); axes=np.eye(3); seed=axes[np.argmin(np.abs(axes@t))]; n=np.cross(t,seed); n/=np.linalg.norm(n); b=np.cross(t,n); b/=np.linalg.norm(b); return n,b
def _plane(im,a,p,n,b,h=4.5,pix=.15,order=1,cval=-1024):
    q=np.arange(-h,h+pix/2,pix); y,x=np.meshgrid(q,q,indexing='ij'); pts=p[None,None,:]+x[...,None]*n+y[...,None]*b; return _sample(im,a,pts.reshape(-1,3),order,cval).reshape(len(q),len(q)),q

def run(drive_root='/content/drive/MyDrive/OpenPlaque',output_root=None):
    root=Path(drive_root); out=Path(output_root or root/OUTPUT_DIRNAME); out.mkdir(parents=True,exist_ok=True); _json(out/'run_state.json',{'status':'STARTED','baseline_commit':BASELINE,'algorithm':ALGORITHM})
    master=json.loads(_req(root/MASTER).read_text()); cl=_load_cl(root/CL); check=_validate_master(master,float(cl.arc_mm.iloc[-1])); ints=pd.read_csv(_req(root/FUSION)); pos=_positive_windows(ints); ctr=_controls(float(cl.arc_mm.iloc[-1]),pos); wins=pos+ctr
    im,src=_source(root/SRC); cur=_mask(root/CUR,im); leg=_mask(root/LEG,im); ao=_mask(root/AORTA,im); union=cur|leg; arc,pts=_resample(cl); med=float(np.median(_sample(im,src,pts))); strict_thr=max(600.,med+75.); vv=float(np.prod(im.GetSpacing()))
    crop,lo=_bbox(im,pts); ct=np.asarray(src[crop],np.int16); un=union[crop]; aoc=ao[crop]; sp=np.asarray(im.GetSpacing()[::-1]); dout=distance_transform_edt(~un,sampling=sp); dao=distance_transform_edt(~aoc,sampling=sp); shell=(~un)&(dout>=.20)&(dout<=2.25)&(dao>.75)
    zl=np.argwhere(shell); zg=zl+lo; phys=_z2p(im,zg); tree=cKDTree(pts); dc,nn=tree.query(phys); aa=arc[nn]; hu=ct[shell].astype(float); near=dc<=4.75; broad=near&(hu>=350)&(hu<=1600); strict=near&(hu>=strict_thr)&(hu<=1800)
    high=pd.DataFrame({'z':zg[:,0],'y':zg[:,1],'x':zg[:,2],'lps_x_mm':phys[:,0],'lps_y_mm':phys[:,1],'lps_z_mm':phys[:,2],'arc_mm':aa,'centerline_distance_mm':dc,'distance_outside_coronary_union_mm':dout[shell],'hu':hu,'broad_ge350':broad,'strict_calcific':strict}); high=high[high.broad_ge350].reset_index(drop=True); high.to_csv(out/'RCA_source_high_density_shell_voxels.csv',index=False)
    sm=np.zeros_like(shell,bool); sidx=zl[strict]
    if len(sidx): sm[tuple(sidx.T)]=True
    comp,keep=_components(sm,ct,im,lo,tree,arc,vv); comp.to_csv(out/'RCA_source_calcific_candidate_components.csv',index=False)
    full=np.zeros(src.shape,np.uint8); full[crop][keep]=1; ni=sitk.GetImageFromArray(full); ni.CopyInformation(im); sitk.WriteImage(ni,str(out/'RCA_source_strict_calcific_candidate.nii.gz'))
    fk=np.argwhere(keep); fg=fk+lo
    if len(fk): fp=_z2p(im,fg); fd,fn=tree.query(fp); fa=arc[fn]; fh=ct[keep].astype(float)
    else: fp=np.empty((0,3)); fd=np.array([]); fa=np.array([]); fh=np.array([])
    pd.DataFrame({'z':fg[:,0] if len(fg) else [],'y':fg[:,1] if len(fg) else [],'x':fg[:,2] if len(fg) else [],'lps_x_mm':fp[:,0] if len(fp) else [],'lps_y_mm':fp[:,1] if len(fp) else [],'lps_z_mm':fp[:,2] if len(fp) else [],'arc_mm':fa,'centerline_distance_mm':fd,'hu':fh}).to_csv(out/'RCA_source_strict_calcific_candidate_voxels.csv',index=False)
    bins=np.arange(0,math.ceil(arc[-1]*2)/2+.5001,.5); ba=aa[broad]; lr=[]
    for a,b in zip(bins[:-1],bins[1:]):
        nb=int(((ba>=a)&(ba<b)).sum()); ns=int(((fa>=a)&(fa<b)).sum()); lr.append({'arc_start_mm':a,'arc_end_mm':b,'broad_ge350_volume_mm3':nb*vv,'strict_filtered_volume_mm3':ns*vv})
    long=pd.DataFrame(lr); long.to_csv(out/'RCA_source_candidate_longitudinal_0p5mm.csv',index=False)
    wr=[]
    for w in wins:
        a,b=w['arc_start_mm'],w['arc_end_mm']; bv=((ba>=a)&(ba<b)).sum()*vv; sv=((fa>=a)&(fa<b)).sum()*vv; cc=comp[(comp.passes_component_gate==True)&(comp.arc_end_mm>=a)&(comp.arc_start_mm<b)] if len(comp) else comp; dur=b-a
        wr.append({**w,'duration_mm':dur,'broad_ge350_volume_mm3':bv,'strict_calcific_candidate_volume_mm3':sv,'strict_volume_density_mm3_per_mm':sv/dur,'passing_component_count':len(cc),'max_component_arc_span_mm':float(cc.arc_span_mm.max()) if len(cc) else 0.,'max_component_volume_mm3':float(cc.volume_mm3.max()) if len(cc) else 0.})
    win=pd.DataFrame(wr); win.to_csv(out/'RCA_source_candidate_window_summary.csv',index=False); primary=win[win.label.eq('primary_model_positive')].iloc[0]; cd=float(np.median(win[win.kind.eq('negative_control')].strict_volume_density_mm3_per_mm)); status=_decision(primary.strict_calcific_candidate_volume_mm3,primary.max_component_arc_span_mm,cd,primary.strict_volume_density_mm3_per_mm); enrich=float(primary.strict_volume_density_mm3_per_mm/max(cd,.02))
    fig,ax=plt.subplots(figsize=(12,4.8)); x=long.arc_start_mm+.25; ax.plot(x,long.broad_ge350_volume_mm3,label='>=350 HU shell sensitivity'); ax.plot(x,long.strict_filtered_volume_mm3,label='strict filtered calcific candidate'); [ax.axvspan(w['arc_start_mm'],w['arc_end_mm'],alpha=.12) for w in pos]; ax.set(xlabel='Accepted RCA arc (mm)',ylabel='Unique source-CCTA volume per 0.5 mm bin (mm3)',title='RCA source-space high-density periluminal candidates'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'01_RCA_source_candidate_longitudinal.png',dpi=180); plt.close(fig)
    top=list((long.sort_values('strict_filtered_volume_mm3',ascending=False).head(5).arc_start_mm+.25).to_numpy(float))+[.5*(w['arc_start_mm']+w['arc_end_mm']) for w in ctr]; u=[]
    for a in top:
        if all(abs(a-x)>.35 for x in u):u.append(a)
    fig,axes=plt.subplots(2,4,figsize=(14,7)); axes=np.asarray(axes).ravel()
    for ax,a in zip(axes,u[:8]):
        i=int(np.argmin(abs(arc-a))); n,b=_frame(pts,i); pl,q=_plane(im,src,pts[i],n,b); pm,_=_plane(im,full,pts[i],n,b,order=0,cval=0); um,_=_plane(im,union.astype(np.uint8),pts[i],n,b,order=0,cval=0); ax.imshow(pl,cmap='gray',vmin=-100,vmax=1000,extent=[q[0],q[-1],q[-1],q[0]])
        if np.any(um>.5):ax.contour(q,q,um>.5,levels=[.5],linewidths=.8)
        if np.any(pm>.5):ax.contourf(q,q,pm>.5,levels=[.5,1.5],alpha=.35)
        ax.set_title(f'arc {arc[i]:.1f} mm');ax.set_xticks([]);ax.set_yticks([])
    [a.axis('off') for a in axes[len(u[:8]):]]; fig.suptitle('Automatically selected RCA source-CCTA orthogonal planes'); fig.tight_layout(); fig.savefig(out/'02_RCA_source_candidate_cross_sections.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(win.label,win.strict_volume_density_mm3_per_mm);ax.set_ylabel('Strict calcific candidate volume density (mm3/mm)');ax.set_title('Model-positive RCA windows vs source-space controls');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(out/'03_RCA_source_candidate_windows.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');ax.plot(pts[:,0],pts[:,1],pts[:,2],label='RCA centerline')
    if len(fp): s=fp[::max(1,len(fp)//1500)];ax.scatter(s[:,0],s[:,1],s[:,2],s=5,label='strict candidate voxels')
    ax.set_title('RCA source-space candidate geometry');ax.legend();fig.tight_layout();fig.savefig(out/'04_RCA_source_candidate_geometry.png',dpi=180);plt.close(fig)
    summary={'algorithm':ALGORITHM,'status':status,'baseline_commit':BASELINE,'scientific_scope':'Independent source-CCTA calcific plaque candidate validation on accepted RCA; curved plaque model supplies longitudinal windows only','master_anatomy_check':check,'source_voxel_volume_mm3':vv,'source_spacing_xyz_mm':list(im.GetSpacing()),'rca_centerline_length_mm':float(arc[-1]),'source_centerline_median_hu':med,'strict_calcific_threshold_hu':strict_thr,'periluminal_shell_mm':[.20,2.25],'aorta_exclusion_margin_mm':.75,'primary_window':primary.to_dict(),'negative_control_median_density_mm3_per_mm':cd,'primary_to_control_enrichment':enrich,'passing_component_count_total':int(comp.passes_component_gate.sum()) if len(comp) else 0,'strict_calcific_candidate_volume_mm3_total':len(fa)*vv,'physical_candidate_volume_reported':True,'source_space_TPV_reported':False,'noncalcified_plaque_volume_reported':False,'interpretation':'Candidate mm3 is a unique source-CCTA physical volume, but not TPV because no validated outer-wall segmentation is used. Visual/source-plane QC is required before promotion.','prior_registration_constraint':'Earlier curved-to-source registration passed longitudinal mapping but failed angular validation; no circumferential coordinates are used here.'};_json(out/'source_candidate_summary.json',summary)
    _json(out/'input_provenance.json',{'master_anatomy':str(root/MASTER),'source_cache':str(root/SRC),'rca_centerline':str(root/CL),'rca_longitudinal_intervals':str(root/FUSION),'totalseg_current_coronary':str(root/CUR),'totalseg_legacy_coronary':str(root/LEG),'totalseg_aorta':str(root/AORTA),'rule':'No circumferential/angular curved-plaque mapping is used.'})
    html=out/'OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_VALIDATION_REPORT.html'; html.write_text(f"""<html><head><meta charset='utf-8'><title>OpenPlaque RCA Source-Space Plaque Validation</title><style>body{{font-family:Arial;max-width:1200px;margin:24px auto;line-height:1.45}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:5px;border-bottom:1px solid #ddd}}.warn{{background:#fff4e5;border-left:5px solid #d68910;padding:12px}}</style></head><body><h1>OpenPlaque — RCA Source-Space Calcific Plaque Candidate Validation</h1><div class='warn'><b>Research use only.</b> Curved-model plaque supplies longitudinal target windows only. Reported mm3 is unique source-CCTA calcific candidate volume, not TPV.</div><h2>Summary</h2><pre>{json.dumps(summary,indent=2)}</pre><h2>Target/control windows</h2>{win.to_html(index=False,float_format=lambda x:f'{x:.3f}')}<h2>Strict components</h2>{comp.to_html(index=False,float_format=lambda x:f'{x:.3f}') if len(comp) else '<p>No strict components.</p>'}<h2>Figures</h2><img src='01_RCA_source_candidate_longitudinal.png'><img src='02_RCA_source_candidate_cross_sections.png'><img src='03_RCA_source_candidate_windows.png'><img src='04_RCA_source_candidate_geometry.png'><h2>Boundary</h2><p>A positive result supports a spatially localized high-density periluminal candidate in source CCTA. It does not establish TPV, noncalcified plaque volume, or a validated outer-wall boundary.</p></body></html>""",encoding='utf-8')
    zf=out/'OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_VALIDATION_REPORT_BACK.zip';state={'status':'COMPLETE','scientific_status':status,'baseline_commit':BASELINE,'report':str(html),'zip':str(zf),'output_dir':str(out)};_json(out/'run_state.json',state)
    with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p!=zf:z.write(p,p.name)
    return {'summary':summary,'report':str(html),'zip':str(zf),'output_dir':str(out),'status':status}
