from __future__ import annotations

import json, zipfile, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, map_coordinates

from openplaque.study import OpenPlaqueStudy


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def _find_one(root: Path, name: str) -> Path:
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def _find_optional(root: Path, name: str):
    hits = list(Path(root).rglob(name))
    if not hits:
        return None
    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def _find_dir_recursive(root: Path, dirname: str) -> Path:
    direct = Path(root) / dirname
    if direct.is_dir():
        return direct
    hits = [p for p in Path(root).rglob(dirname) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f'Could not find directory {dirname} under {root}')
    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def reconstruct_source(cache_dir, out='/content/source_ccta_registration.nii.gz'):
    cache = Path(cache_dir)
    arr = np.load(cache/'series7_int16.npy', mmap_mode='r')
    meta = json.loads((cache/'series7_int16.json').read_text())
    img = sitk.GetImageFromArray(arr)
    sp = np.asarray(meta['spacing_zyx'], float)
    img.SetSpacing(tuple(sp[::-1]))
    img.SetOrigin(tuple(np.asarray(meta['positions_lps_mm'][0], float)))
    iop = np.asarray(meta['image_orientation_patient'], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    d = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    img.SetDirection(tuple(d.ravel()))
    sitk.WriteImage(img, out)
    return Path(out)


def audit_dicom_files(files, vessel, series_number):
    rows=[]
    for f in files:
        ds=pydicom.dcmread(f, stop_before_pixels=True, force=True)
        px=getattr(ds,'PixelSpacing',None)
        ipp=getattr(ds,'ImagePositionPatient',None)
        iop=getattr(ds,'ImageOrientationPatient',None)
        rows.append({
            'vessel':vessel,'series_number':series_number,'file':Path(f).name,
            'instance_number':int(getattr(ds,'InstanceNumber',-1)),
            'rows':int(getattr(ds,'Rows',0)),'cols':int(getattr(ds,'Columns',0)),
            'pixel_spacing_row_mm':float(px[0]) if px is not None else np.nan,
            'pixel_spacing_col_mm':float(px[1]) if px is not None else np.nan,
            'slice_thickness_mm':float(getattr(ds,'SliceThickness',np.nan)),
            'spacing_between_slices_mm':float(getattr(ds,'SpacingBetweenSlices',np.nan)),
            'ipp':tuple(float(x) for x in ipp) if ipp is not None else None,
            'iop':tuple(float(x) for x in iop) if iop is not None else None,
            'image_type':'\\'.join(str(x) for x in getattr(ds,'ImageType',[])),
            'series_description':str(getattr(ds,'SeriesDescription','')),
            'sop_class_uid':str(getattr(ds,'SOPClassUID','')),
        })
    df=pd.DataFrame(rows).sort_values('instance_number').reset_index(drop=True)
    ipp_unique=len(set(str(x) for x in df['ipp'])) if len(df) else 0
    iop_unique=len(set(str(x) for x in df['iop'])) if len(df) else 0
    spatial_geometry = bool(len(df) and df['ipp'].notna().all() and df['iop'].notna().all() and ipp_unique>1)
    classification = 'spatial_stack' if spatial_geometry else 'nonspatial_or_rotation_stack'
    meta={
        'vessel':vessel,'series_number':series_number,'instances':int(len(df)),
        'rows':int(df['rows'].mode().iloc[0]) if len(df) else 0,
        'cols':int(df['cols'].mode().iloc[0]) if len(df) else 0,
        'unique_ipp':ipp_unique,'unique_iop':iop_unique,
        'classification':classification,
        'series_description':str(df['series_description'].iloc[0]) if len(df) else '',
        'pixel_spacing_row_mm':float(df['pixel_spacing_row_mm'].median()) if len(df) else np.nan,
        'pixel_spacing_col_mm':float(df['pixel_spacing_col_mm'].median()) if len(df) else np.nan,
    }
    return df,meta


def resample_centerline(df, step_mm=0.40):
    arc=df['arc_mm'].to_numpy(float)
    pts=df[['lps_x_mm','lps_y_mm','lps_z_mm']].to_numpy(float)
    q=np.arange(float(arc.min()), float(arc.max())+1e-6, step_mm)
    xyz=np.column_stack([np.interp(q,arc,pts[:,k]) for k in range(3)])
    return q,xyz


def parallel_transport_frames(points):
    p=np.asarray(points,float)
    if len(p)<3: raise ValueError('Need >=3 points')
    t=np.gradient(p,axis=0)
    t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
    axes=np.eye(3)
    seed=axes[np.argmin(np.abs(axes@t[0]))]
    n=np.cross(t[0],seed); n/=np.linalg.norm(n)
    b=np.cross(t[0],n); b/=np.linalg.norm(b)
    ns=[n]; bs=[b]
    for i in range(1,len(p)):
        v=ns[-1]-np.dot(ns[-1],t[i])*t[i]
        if np.linalg.norm(v)<1e-6:
            v=np.cross(t[i],bs[-1])
        v/=np.linalg.norm(v)
        w=np.cross(t[i],v); w/=np.linalg.norm(w)
        if np.dot(v,ns[-1])<0:
            v=-v; w=-w
        ns.append(v); bs.append(w)
    return t,np.asarray(ns),np.asarray(bs)


def _phys_to_zyx(img, pts):
    pts=np.asarray(pts,float)
    origin=np.asarray(img.GetOrigin(),float)
    spacing=np.asarray(img.GetSpacing(),float)
    direction=np.asarray(img.GetDirection(),float).reshape(3,3)
    idx_xyz=((pts-origin)@np.linalg.inv(direction).T)/spacing
    return idx_xyz[:,::-1]


def sample_source_points(img, arr, pts, cval=-1024.0):
    zyx=_phys_to_zyx(img,pts)
    return map_coordinates(arr, zyx.T, order=1, mode='constant', cval=cval)


def _feature(x):
    x=np.asarray(x,float)
    x=np.clip(x,-150,850)
    if len(x)>5: x=gaussian_filter1d(x,1.2)
    z=(x-np.nanmedian(x))/(np.nanstd(x)+1e-6)
    g=np.gradient(z)
    g=(g-np.nanmedian(g))/(np.nanstd(g)+1e-6)
    return z,g


def _corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<12: return -1.0
    a=a[m]-np.mean(a[m]); b=b[m]-np.mean(b[m])
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>1e-8 else -1.0


def fit_longitudinal_mapping(source_arc, source_hu, stack, pixel_spacing=(1.0,1.0), center_search_px=40):
    stack=np.asarray(stack,float)
    best=None
    Lmm=float(source_arc[-1]-source_arc[0])
    for long_axis in (1,2):
        oriented = stack if long_axis==2 else np.transpose(stack,(0,2,1))
        nrot,R,L=oriented.shape
        mid=R//2
        centers=range(max(1,mid-center_search_px), min(R-1,mid+center_search_px+1), 4)
        nominal=float(pixel_spacing[1] if long_axis==2 else pixel_spacing[0])
        if not np.isfinite(nominal) or nominal<=0: nominal=0.5
        scales=np.linspace(0.72,1.28,15)
        for c in centers:
            obs=np.median(oriented[:,max(0,c-1):min(R,c+2),:],axis=(0,1))
            for scale in scales:
                mmpp=nominal*float(scale)
                npix=max(20,int(round(Lmm/mmpp))+1)
                if npix>L: continue
                src_i=np.linspace(0,len(source_hu)-1,npix)
                src=np.interp(src_i,np.arange(len(source_hu)),source_hu)
                sz,sg=_feature(src)
                for flip in (False,True):
                    zz=sz[::-1] if flip else sz
                    gg=sg[::-1] if flip else sg
                    for start in range(0,L-npix+1,max(1,npix//40)):
                        seg=obs[start:start+npix]
                        oz,og=_feature(seg)
                        ci=_corr(zz,oz); cg=_corr(gg,og)
                        score=.55*ci+.45*cg
                        rec={'score':score,'intensity_corr':ci,'gradient_corr':cg,
                             'long_axis':long_axis,'radial_center_px':int(c),'start_px':int(start),
                             'n_pixels':int(npix),'mm_per_long_pixel':float(mmpp),'long_flip':bool(flip),
                             'oriented_shape':[int(x) for x in oriented.shape]}
                        if best is None or score>best['score']: best=rec
    if best is None: raise RuntimeError('No longitudinal mapping candidate')
    return best


def synth_ribbon(img, arr, center_points, nvec, bvec, angle_rad, radial_offsets):
    d=np.cos(angle_rad)*nvec+np.sin(angle_rad)*bvec
    pts=center_points[:,None,:]+radial_offsets[None,:,None]*d[:,None,:]
    vals=sample_source_points(img,arr,pts.reshape(-1,3)).reshape(len(center_points),len(radial_offsets))
    return vals.T


def _ncc2(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape: return -1.0
    a=np.clip(a,-150,850); b=np.clip(b,-150,850)
    a=(a-np.median(a))/(np.std(a)+1e-6); b=(b-np.median(b))/(np.std(b)+1e-6)
    return _corr(a.ravel(),b.ravel())


def fit_angular_mapping(source_img,source_arr,arc,points,nvec,bvec,stack,longfit,pixel_spacing):
    oriented=stack if longfit['long_axis']==2 else np.transpose(stack,(0,2,1))
    N,R,L=oriented.shape
    c=longfit['radial_center_px']; start=longfit['start_px']; npix=longfit['n_pixels']
    long_idx=np.linspace(0,len(points)-1,npix)
    cp=np.column_stack([np.interp(long_idx,np.arange(len(points)),points[:,k]) for k in range(3)])
    nv=np.column_stack([np.interp(long_idx,np.arange(len(points)),nvec[:,k]) for k in range(3)])
    nv/=np.maximum(np.linalg.norm(nv,axis=1,keepdims=True),1e-9)
    bv=np.column_stack([np.interp(long_idx,np.arange(len(points)),bvec[:,k]) for k in range(3)])
    bv/=np.maximum(np.linalg.norm(bv,axis=1,keepdims=True),1e-9)
    radial_spacing=float(pixel_spacing[0] if longfit['long_axis']==2 else pixel_spacing[1])
    if not np.isfinite(radial_spacing) or radial_spacing<=0: radial_spacing=longfit['mm_per_long_pixel']
    half_px=max(6,min(24,int(round(7.0/radial_spacing))))
    r0=max(0,c-half_px); r1=min(R,c+half_px+1)
    roff=(np.arange(r0,r1)-c)*radial_spacing
    obs=oriented[:,r0:r1,start:start+npix]
    if longfit['long_flip']: obs=obs[:,:,::-1]
    synth_cache=[]
    for j in range(N):
        theta=2*np.pi*j/max(N,1)
        synth_cache.append(synth_ribbon(source_img,source_arr,cp,nv,bv,theta,roff))
    pair=np.zeros((N,N),float); pair_flip=np.zeros((N,N),float)
    for i in range(N):
        for j in range(N):
            pair[i,j]=_ncc2(obs[i],synth_cache[j])
            pair_flip[i,j]=_ncc2(obs[i],synth_cache[j][::-1])
    best=None
    for period_deg in (180.0,360.0):
        period_steps = max(1, int(round(N * period_deg / 360.0)))
        for radial_flip,mat in [(False,pair),(True,pair_flip)]:
            for direction in (1,-1):
                for phase in range(N):
                    jj=np.array([(phase + direction * (i % period_steps)) % N for i in range(N)],int)
                    vals=np.array([mat[i,jj[i]] for i in range(N)])
                    score=float(np.median(vals))
                    rec={'median_ncc':score,'p25_ncc':float(np.quantile(vals,.25)),
                         'phase_index':int(phase),'angle_direction':int(direction),'radial_flip':bool(radial_flip),
                         'per_slice_ncc':vals.tolist(),'radial_spacing_mm':float(radial_spacing),
                         'radial_half_width_px':int(half_px),'angle_period_deg':float(period_deg)}
                    if best is None or score>best['median_ncc']: best=rec
    flat=np.concatenate([pair.ravel(),pair_flip.ravel()])
    best['background_pairwise_median']=float(np.median(flat))
    best['sequence_margin']=float(best['median_ncc']-np.median(flat))
    best['pairwise_matrix']=pair.tolist()
    return best,obs,synth_cache


def _interp_frame_at_arc(arc_query, arc, points, nvec, bvec):
    aq=np.asarray(arc_query,float)
    cp=np.column_stack([np.interp(aq,arc,points[:,k]) for k in range(3)])
    nv=np.column_stack([np.interp(aq,arc,nvec[:,k]) for k in range(3)])
    bv=np.column_stack([np.interp(aq,arc,bvec[:,k]) for k in range(3)])
    nv/=np.maximum(np.linalg.norm(nv,axis=1,keepdims=True),1e-9)
    bv/=np.maximum(np.linalg.norm(bv,axis=1,keepdims=True),1e-9)
    return cp,nv,bv


def map_vote_voxels_to_source(vote, longfit, anglefit, arc, points, nvec, bvec, pixel_spacing, threshold=3, max_points=750000):
    vote=np.asarray(vote)
    z,r,c=np.where(vote>=threshold)
    vals=vote[z,r,c].astype(np.uint8)
    if len(z)>max_points:
        sel=np.linspace(0,len(z)-1,max_points).astype(int); z,r,c,vals=z[sel],r[sel],c[sel],vals[sel]
    if longfit['long_axis']==2:
        lp=c.astype(float); rp=r.astype(float); radial_spacing=float(pixel_spacing[0])
    else:
        lp=r.astype(float); rp=c.astype(float); radial_spacing=float(pixel_spacing[1])
    if not np.isfinite(radial_spacing) or radial_spacing<=0: radial_spacing=anglefit['radial_spacing_mm']
    frac=(lp-longfit['start_px'])/max(longfit['n_pixels']-1,1)
    if longfit['long_flip']: frac=1-frac
    valid=(frac>=0)&(frac<=1)&(z>=0)&(z<vote.shape[0])
    z,rp,frac,vals=z[valid],rp[valid],frac[valid],vals[valid]
    aq=arc[0]+frac*(arc[-1]-arc[0])
    cp,nv,bv=_interp_frame_at_arc(aq,arc,points,nvec,bvec)
    radial=(rp-longfit['radial_center_px'])*radial_spacing
    if anglefit['radial_flip']: radial=-radial
    N=vote.shape[0]
    period_steps=max(1,int(round(N*anglefit['angle_period_deg']/360.0)))
    j=np.array([(anglefit['phase_index']+anglefit['angle_direction']*(int(zz)%period_steps))%N for zz in z],int)
    theta=np.deg2rad(360.0*j/float(N))
    d=np.cos(theta)[:,None]*nv+np.sin(theta)[:,None]*bv
    xyz=cp+radial[:,None]*d
    return pd.DataFrame({'rotation_index':z,'source_angle_index':j,'arc_mm':aq,'radial_mm':radial,
                         'lps_x_mm':xyz[:,0],'lps_y_mm':xyz[:,1],'lps_z_mm':xyz[:,2],'vote':vals.astype(int)})


def splat_points_to_source(source_img, mapped_df):
    out=np.zeros(tuple(reversed(source_img.GetSize())),np.uint8)
    if not len(mapped_df): return out
    zyx=np.rint(_phys_to_zyx(source_img,mapped_df[['lps_x_mm','lps_y_mm','lps_z_mm']].to_numpy(float))).astype(int)
    valid=np.ones(len(zyx),bool)
    for k,s in enumerate(out.shape): valid&=(zyx[:,k]>=0)&(zyx[:,k]<s)
    zyx=zyx[valid]; v=mapped_df['vote'].to_numpy(np.uint8)[valid]
    flat=np.ravel_multi_index(zyx.T,out.shape); np.maximum.at(out.ravel(),flat,v)
    return out


def _save_registration_figures(out,vessel,stack,source_hu,longfit,anglefit,obs,synth):
    figs=[]; oriented=stack if longfit['long_axis']==2 else np.transpose(stack,(0,2,1))
    c=longfit['radial_center_px']; start=longfit['start_px']; n=longfit['n_pixels']
    prof=np.median(oriented[:,max(0,c-1):min(oriented.shape[1],c+2),:],axis=(0,1))[start:start+n]
    if longfit['long_flip']: prof=prof[::-1]
    src=np.interp(np.linspace(0,len(source_hu)-1,n),np.arange(len(source_hu)),source_hu)
    fig,ax=plt.subplots(figsize=(11,4.5)); ax.plot(src,label='source centerline HU'); ax.plot(prof,label='curved-series central profile',alpha=.8)
    ax.set_title(f'{vessel}: longitudinal registration'); ax.legend(); fig.tight_layout(); p=out/f'{vessel}_01_longitudinal_registration.png'; fig.savefig(p,dpi=160); plt.close(fig); figs.append(p)
    pair=np.asarray(anglefit['pairwise_matrix']); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(pair,aspect='auto'); fig.colorbar(im,ax=ax,label='NCC'); ax.set_xlabel('synthetic angle index'); ax.set_ylabel('curved image index'); ax.set_title(f"{vessel}: rotation matching ({anglefit['angle_period_deg']:.0f}° convention)"); fig.tight_layout(); p=out/f'{vessel}_02_rotation_score_matrix.png'; fig.savefig(p,dpi=160); plt.close(fig); figs.append(p)
    picks=np.linspace(0,stack.shape[0]-1,min(4,stack.shape[0])).astype(int); fig,axes=plt.subplots(len(picks),2,figsize=(10,3*len(picks))); axes=np.atleast_2d(axes)
    for rr,i in enumerate(picks):
        j=(anglefit['phase_index']+anglefit['angle_direction']*i)%stack.shape[0]; syn=synth[j][::-1] if anglefit['radial_flip'] else synth[j]
        axes[rr,0].imshow(obs[i],cmap='gray',vmin=-150,vmax=850,aspect='auto'); axes[rr,0].set_title(f'observed rotation {i}')
        axes[rr,1].imshow(syn,cmap='gray',vmin=-150,vmax=850,aspect='auto'); axes[rr,1].set_title(f'source synthetic angle {j}')
        for a in axes[rr]: a.axis('off')
    fig.tight_layout(); p=out/f'{vessel}_03_observed_vs_source_ribbons.png'; fig.savefig(p,dpi=160); plt.close(fig); figs.append(p)
    return figs


def _status(longfit,anglefit,meta):
    rotation_like = meta['classification']=='nonspatial_or_rotation_stack' or meta['instances']<=64
    long_ok=longfit['score']>=0.30 and longfit['gradient_corr']>=0.15
    angle_ok=anglefit['median_ncc']>=0.22 and anglefit['sequence_margin']>=0.04 and anglefit['p25_ncc']>=0.05
    if rotation_like and long_ok and angle_ok: return 'VALIDATED_CURVED_TO_SOURCE_RESEARCH_MAPPING'
    if long_ok: return 'PARTIAL_LONGITUDINAL_MAPPING_ONLY'
    return 'NO_VALIDATED_MAPPING'


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None, study_zip=None, source_cache=None, series_map=None):
    drive_root=Path(drive_root); out=Path(output_root or drive_root/'Curved_Plaque_to_Source_Registration_v1'); out.mkdir(parents=True,exist_ok=True)
    _json(out/'run_state.json', {'state':'STARTED'})
    try:
        study_zip=Path(study_zip or drive_root/'Full_DICOM.zip'); source_cache=Path(source_cache or drive_root/'Cache/Secondary_3D_Vesselness_Topology_v1')
        series_map=series_map or {'RCA':1035,'LAD':1043}
        source_path=reconstruct_source(source_cache); source_img=sitk.ReadImage(str(source_path)); source_arr=sitk.GetArrayFromImage(source_img).astype(np.float32,copy=False)
        study=OpenPlaqueStudy(str(study_zip),extract_root='/content/full_dicom_curved_registration')
        centerline_root=_find_dir_recursive(drive_root,'Source_Volume_Coronary_Centerlines')
        atlas_root=_find_dir_recursive(drive_root,'Plaque_Ensemble_Confidence_Atlas_v1')
        _json(out/'input_resolution.json', {'centerline_root':str(centerline_root),'atlas_root':str(atlas_root),'study_zip':str(study_zip),'source_cache':str(source_cache)})
        global_summary=[]; all_figs=[]
        for vessel,series_number in series_map.items():
            vdir=out/vessel; vdir.mkdir(exist_ok=True)
            _,stack,files=study.load_series(series_number)
            aud,meta=audit_dicom_files(files,vessel,series_number); aud.to_csv(vdir/'dicom_instance_audit.csv',index=False); _json(vdir/'dicom_geometry_summary.json',meta)
            cl=pd.read_csv(_find_one(centerline_root,f'{vessel}_source_centerline.csv'))
            nominal=min([x for x in [meta['pixel_spacing_row_mm'],meta['pixel_spacing_col_mm']] if np.isfinite(x) and x>0] or [0.4]); step=min(max(nominal,0.25),0.60)
            arc,pts=resample_centerline(cl,step); _,nvec,bvec=parallel_transport_frames(pts); source_hu=sample_source_points(source_img,source_arr,pts)
            px=(meta['pixel_spacing_row_mm'],meta['pixel_spacing_col_mm'])
            longfit=fit_longitudinal_mapping(arc,source_hu,stack,px); anglefit,obs,synth=fit_angular_mapping(source_img,source_arr,arc,pts,nvec,bvec,stack,longfit,px)
            status=_status(longfit,anglefit,meta)
            reg={'vessel':vessel,'series_number':series_number,'status':status,'dicom_geometry':meta,'longitudinal_fit':longfit,'angular_fit':{k:v for k,v in anglefit.items() if k!='pairwise_matrix'}}
            _json(vdir/'registration_model.json',reg); all_figs+=_save_registration_figures(vdir,vessel,stack,source_hu,longfit,anglefit,obs,synth)
            mapped_counts={}
            if status=='VALIDATED_CURVED_TO_SOURCE_RESEARCH_MAPPING':
                vote=sitk.GetArrayFromImage(sitk.ReadImage(str(_find_one(atlas_root,f'{vessel}_plaque_vote_0to5.nii.gz')))).astype(np.uint8)
                for th in (3,4,5):
                    mdf=map_vote_voxels_to_source(vote,longfit,anglefit,arc,pts,nvec,bvec,px,threshold=th); mdf.to_csv(vdir/f'{vessel}_mapped_plaque_vote_ge{th}.csv',index=False)
                    support=splat_points_to_source(source_img,mdf); im=sitk.GetImageFromArray(support); im.CopyInformation(source_img); sitk.WriteImage(im,str(vdir/f'{vessel}_source_space_plaque_vote_ge{th}.nii.gz'))
                    vv=float(np.prod(source_img.GetSpacing())); mapped_counts[f'vote_ge{th}_mapped_points']=int(len(mdf)); mapped_counts[f'vote_ge{th}_source_support_voxels']=int((support>=th).sum()); mapped_counts[f'vote_ge{th}_source_support_volume_mm3']=float((support>=th).sum()*vv)
                m4=pd.read_csv(vdir/f'{vessel}_mapped_plaque_vote_ge4.csv'); bins=np.arange(arc[0],arc[-1]+1.0,1.0); h,_=np.histogram(m4['arc_mm'],bins=bins); pd.DataFrame({'arc_start_mm':bins[:-1],'arc_end_mm':bins[1:],'mapped_high_conf_points':h}).to_csv(vdir/f'{vessel}_source_arc_plaque_support.csv',index=False)
            reg['mapped_support']=mapped_counts; _json(vdir/'registration_model.json',reg)
            global_summary.append({'vessel':vessel,'status':status,'series_number':series_number,'instances':meta['instances'],'series_classification':meta['classification'],'longitudinal_score':longfit['score'],'longitudinal_gradient_corr':longfit['gradient_corr'],'angular_period_deg':anglefit['angle_period_deg'],'angular_median_ncc':anglefit['median_ncc'],'angular_p25_ncc':anglefit['p25_ncc'],'angular_sequence_margin':anglefit['sequence_margin'],**mapped_counts})
            _json(out/'run_state.json', {'state':'RUNNING','completed_vessels':[x['vessel'] for x in global_summary]})
        summary_df=pd.DataFrame(global_summary); summary_df.to_csv(out/'registration_validation_summary.csv',index=False)
        overall='VALIDATED' if all(x['status']=='VALIDATED_CURVED_TO_SOURCE_RESEARCH_MAPPING' for x in global_summary) else ('PARTIAL' if any(x['status']!='NO_VALIDATED_MAPPING' for x in global_summary) else 'REJECTED')
        summary={'status':overall,'vessels':global_summary,'interpretation':'Source-space plaque overlays are created only for vessels that pass conservative longitudinal and angular matching gates.','volume_warning':'If the curved series is nonspatial/rotation-stack data, plaque mm3 measured directly in stack coordinates is not an anatomical source-space plaque volume. Source-space support volumes in this report are reconstruction-support quantities, not yet validated TPV.'}; _json(out/'summary.json',summary)
        html=out/'OPENPLAQUE_CURVED_PLAQUE_TO_SOURCE_REGISTRATION_REPORT.html'; blocks=''.join(f'<h3>{p.parent.name} — {p.stem}</h3><img src="{p.relative_to(out)}" style="max-width:100%">' for p in all_figs)
        html.write_text('<html><head><meta charset="utf-8"><title>OpenPlaque Curved Plaque Registration</title></head><body style="font-family:Arial;max-width:1200px;margin:auto"><h1>OpenPlaque Curved-Series Plaque → Source-CCTA Registration</h1><p><b>Research use only.</b> No source-space plaque overlay is accepted unless validation gates pass.</p><p><b>Important:</b> when a curved DICOM series is a rotation stack rather than a true 3-D spatial volume, its native stack voxel count must not be interpreted as anatomical plaque volume.</p>'+summary_df.to_html(index=False)+blocks+'</body></html>')
        zf=out/'OPENPLAQUE_CURVED_PLAQUE_TO_SOURCE_REGISTRATION_REPORT_BACK.zip'
        with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file() and p!=zf: z.write(p,p.relative_to(out))
        _json(out/'run_state.json', {'state':'COMPLETE','overall_status':overall})
        return {'output_dir':str(out),'report':str(html),'zip':str(zf),'summary':summary}
    except Exception as e:
        err={'state':'ERROR','error':str(e),'traceback':traceback.format_exc()}
        _json(out/'run_state.json',err)
        (out/'ERROR.txt').write_text(err['traceback'])
        raise
