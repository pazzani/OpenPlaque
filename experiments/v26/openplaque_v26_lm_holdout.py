"""OpenPlaque V26: LM-specific candidate selection with an untouched within-DEV80 evaluation subset.

Research-only diagnostic: ImageCAS-X is sparsely annotated positive; unlabeled voxels are unknown.
No model training, no test-set reads, no reference-based candidate construction, no anatomy promotion.
"""
import argparse
import gc
import hashlib
import html
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

VERSION='openplaque-v26-lm-ridge-soft15-holdout-20261016-1'
ROOT=Path('/content/drive/MyDrive/OpenPlaque')
BASE=ROOT/'OpenPlaque_Evaluation_Dataset_v0_2'
V22=BASE/'evaluation_v22_full_volume'
V23=BASE/'evaluation_v23_heart_aorta_roi'
V24=BASE/'evaluation_v24_vessel_candidates'
V25=BASE/'evaluation_v25_soft_vessel_candidates'
OUT=BASE/'evaluation_v26_lm_holdout'
CT_ROOT=ROOT/'ExternalData/ImageCAS_OriginalCCTA_MatchedTo_ImageCASX_v1/volumes'
SEGS=ROOT/'Cache/ImageCAS_X_Pretrained_v1/ImageCAS-X_dataset.zip_extracted/ImageCAS-X_dataset/segmentations'
DEV_FILE=BASE/'internal_dev_80_frozen.txt'
DEV_SHA='ef2bd3bdbe7dc9cbf4639e62d80dceb53108ac836acff64617e0f17b4f29be4f'
MODEL_SHA='f789eede621816433240232455d7a7862c7eafc69a979a50938e73349a6796b9'
HEADS={'LM':(1,0),'LAD':(2,1),'LCX':(3,2),'RCA':(9,3)}
PILOT=('11','142','203','305','370','455','562','633','708','776','853','976')
HOLDOUT_N=16
COARSE_STEP=(3,3,2)
FRANGI_SIGMA=(0.9,1.6,2.5)
HU_MIN, HU_MAX=80.,1600.
CANDIDATES=('uniform_soft15','lm_ridge_only','lm_ridge_union_soft15','lm_soft25','hu80')
PRIMARY='lm_ridge_only'
GATE=dict(variant=PRIMARY,min_retention_each_artery=.90,min_raw_pred_removed=.85,
          min_eligible_cases_per_artery=5)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(p):return json.loads(Path(p).read_text())

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def native_mask(path,ref):
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    m=nib.load(str(path))
    if m.shape!=ref.shape or not np.allclose(m.affine,ref.affine,atol=1e-3):
        m=resample_from_to(m,(ref.shape,ref.affine),order=0)
    return np.asarray(m.dataobj,dtype=np.uint8)>0

def crop_bbox(mask,spacing,pad_mm=38):
    import numpy as np
    pts=np.where(mask)
    assert len(pts[0]) and pts[0].size>0,'Empty CT-only heart mask'
    pad=np.ceil(pad_mm/np.array(spacing)).astype(int)+3
    lo=np.maximum(0,np.array([int(x.min()) for x in pts])-pad)
    hi=np.minimum(mask.shape,np.array([int(x.max())+1 for x in pts])+pad)
    return tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))

def roi_from_heart_aorta(heart,aorta,spacing):
    from scipy import ndimage as ndi
    import numpy as np
    # Explicit physical distances, same V23 rule.
    dh=ndi.distance_transform_edt(~heart,sampling=spacing)
    da=ndi.distance_transform_edt(~aorta,sampling=spacing)
    roi=(dh<=10.) | ((da<=8.)&(dh<=22.))
    del dh,da
    return roi

def vesselness_coarse(hu):
    """Coarse bright ridge response; never thresholded alone for V25 candidates."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.filters import frangi
    coarse=np.asarray(hu[::3,::3,::2],dtype=np.float32)
    coarse=np.clip((coarse-80.)/720.,0,1)
    f=frangi(coarse,sigmas=FRANGI_SIGMA,black_ridges=False,mode='reflect')
    f=np.nan_to_num(np.asarray(f,dtype=np.float32),nan=0,posinf=0,neginf=0)
    expanded=ndi.zoom(f,zoom=tuple(hu.shape[i]/f.shape[i] for i in range(3)),order=1)
    expanded=expanded[tuple(slice(0,n) for n in hu.shape)]
    if expanded.shape!=hu.shape:raise RuntimeError('Frangi resampling shape mismatch')
    return expanded

def soft_score(hu,frangi_response):
    import numpy as np
    from scipy.ndimage import uniform_filter
    high=np.clip((hu-60.)/420.,0.,1.)
    tubular=np.clip(frangi_response/0.035,0.,1.)
    attenuation_plausible=((hu>=100.)&(hu<=1400.)).astype(np.float32)
    local_density=uniform_filter(attenuation_plausible,size=(7,7,5),mode='nearest')
    nonbulk=np.clip((0.60-local_density)/0.50,0.,1.)
    result=np.asarray(.45*high+.35*tubular+.20*nonbulk,dtype=np.float32)
    return result

def fixed_budget_rank(candidate,score,k):
    """Top-K deterministic selection only from existing V22 head / CT ROI/HU proposal."""
    import numpy as np
    idx=np.flatnonzero(candidate)
    k=max(0,min(int(k),len(idx)))
    out=np.zeros(candidate.shape,dtype=bool)
    if k==len(idx):out.ravel()[idx]=True
    elif k>0:
        vals=score.ravel()[idx]
        # np.lexsort handles equal scores by native voxel index for reproducibility.
        if k<len(idx)//2:
            partial=np.argpartition(vals,len(vals)-k)[-k:]
            # Explicit deterministic tie behavior at the boundary.
            boundary=float(np.min(vals[partial]))
            above=idx[vals>boundary]
            equal=idx[vals==boundary]
            remain=k-len(above)
            out.ravel()[above]=True
            out.ravel()[equal[:remain]]=True
        else:
            order=np.lexsort((idx,-vals))
            out.ravel()[idx[order[:k]]]=True
    assert int(out.sum())==k
    return out

def topology_metrics(mask,spacing):
    import numpy as np
    from scipy import ndimage as ndi
    if not mask.any():return dict(n_components_ge_10mm3=0,largest_component_fraction=0.,n_components_all=0)
    labels,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,1))
    c=np.bincount(labels.ravel())[1:]
    voxmm=float(np.prod(spacing))
    ret=dict(n_components_all=int(n),n_components_ge_10mm3=int(np.count_nonzero(c*voxmm>=10)),
             largest_component_fraction=float(c.max()/mask.sum()))
    del labels,c
    return ret

def heldout_ids(ids):
    """Selection depends only on frozen IDs + a V26-specific salt, never labels or metrics."""
    pool=sorted(set(ids)-set(PILOT),key=int)
    assert len(pool)==68
    selected=sorted(pool,key=lambda c:(hashlib.sha256((VERSION+'|untouched-dev80|'+c).encode()).hexdigest(),int(c)))[:HOLDOUT_N]
    assert not set(selected)&set(PILOT)
    return tuple(selected)

def plan(ids):
    return dict(version=VERSION,frozen_dev_sha256=DEV_SHA,model_sha256=MODEL_SHA,
        pilot_case_ids=list(PILOT),independent_development_case_ids=list(heldout_ids(ids)),
        selection_method='first 16 SHA256(version|untouched-dev80|case_id) ranks excluding pilot; no labels consulted',
        no_test_reads=True,no_training=True,no_reference_guided_proposals=True,
        reference_only_after_candidate_masks_frozen=True,
        frozen_v22='evaluation_v22_full_volume',frozen_v23='evaluation_v23_heart_aorta_roi',
        frozen_v24='evaluation_v24_vessel_candidates',frozen_v25='evaluation_v25_soft_vessel_candidates',
        candidate_variants=list(CANDIDATES),primary_variant=PRIMARY,
        pilot_diagnostics_are_not_independent=True,
        root_proximity_is_a_proxy_not_confirmed_origin=True,
        annotated_positive_recall_does_not_establish_specificity=True,
        candidate_generation=dict(roi_margin_mm=10,aorta_supplement_distance_mm=8,
              hu_soft_bounds=[80,1600],frangi_coarse_step=list(COARSE_STEP),
              frangi_sigmas_coarse=list(FRANGI_SIGMA),soft15_weights=[.45,.35,.20],
              lm_ridge_frangi_native_threshold=.02,lm_ridge_dilation_mm=1.3,
              lm_ridge_hu_bounds=[120,1400],
              LM_variant1='ridge supported native voxels intersecting ROI10 and HU120-1400',
              LM_variant2='union of ridge supported and V25 soft15 selection',
              LM_variant3='soft25 top-ranked LM voxels; remaining heads soft15'),
        gate=GATE,labels_unknown_outside_annotations=True,no_auto_model_promotion=True)

def preflight():
    ids=[x.strip() for x in DEV_FILE.read_text().splitlines() if x.strip()]
    assert len(ids)==len(set(ids))==80 and set(PILOT)<=set(ids)
    digest=hashlib.sha256(''.join(i+'\n' for i in sorted(ids,key=int)).encode()).hexdigest()
    assert digest==DEV_SHA,'STOP DEV80 list does not match frozen signature'
    m22=read_json(V22/'whole_volume_summary_v22.json')
    assert m22['complete'] and m22['cases_complete']==80 and m22['model_checkpoint_sha256']==MODEL_SHA and m22['frozen_dev_sha256']==DEV_SHA
    m23=read_json(V23/'summary_v23.json')
    assert m23['n_complete']==80 and m23['stage']=='full_complete' and m23['test_reads']==0
    m24=read_json(V24/'summary_v24.json')
    assert m24['n_cases_complete']==12 and not m24['pilot_gate']['passed']
    m25=read_json(V25/'summary_v25.json')
    assert m25['n_complete']==12 and not m25['pilot_gate']['passed']
    m25plan=read_json(V25/'manifest_v25.json')
    assert m25plan['dev_ids_sha256']==DEV_SHA and m25plan['model_checkpoint_sha256']==MODEL_SHA
    assert tuple(m25plan['pilot_case_ids'])==PILOT
    verified=read_json(BASE/'sparse_topology_results_v0_3/frozen_positive_reference_manifest_v0_3.json')
    assert verified['status']=='FROZEN_POSITIVE_REFERENCE_QA_ONLY' and verified['sealed_test_accessed'] is False
    selected=tuple(PILOT)+heldout_ids(ids)
    for cid in selected:
        for p in (CT_ROOT/f'{cid}.img.nii.gz',SEGS/f'{cid}.coronary.nii.gz',
                  V22/'prediction_masks'/f'{cid}.v22_4head_native_raw.nii.gz',
                  V22/'per_case'/f'{cid}.json',
                  V23/'totalseg_masks'/cid/'heart.nii.gz',
                  V23/'totalseg_masks'/cid/'aorta.nii.gz',
                  V23/'per_case'/f'{cid}.json'):
            if not p.is_file():raise FileNotFoundError(p)
    OUT.mkdir(parents=True,exist_ok=True)
    m=OUT/'manifest_v26.json';manifest=plan(ids)
    if m.exists():
        if read_json(m)!=manifest:raise RuntimeError('STOP V26 plan changed. Never mix experiment manifests')
    else:write_json(m,manifest)
    print('V26 PREFLIGHT PASS: development pilot',len(PILOT),' independent development evaluation',len(heldout_ids(ids)))
    print('Frozen independent case IDs:',','.join(heldout_ids(ids)))
    return ids

def roi_and_da(heart,aorta,spacing):
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    dh=distance_transform_edt(~heart,sampling=spacing).astype(np.float32)
    da=distance_transform_edt(~aorta,sampling=spacing).astype(np.float32)
    roi=(dh<=10.) | ((da<=8.) & (dh<=22.))
    del dh
    return roi,da

def lm_ridge_support(frangi_response,hu,roi,spacing):
    """CT-only hard bright-ridge support. A hypothesis inspired by V24 but not bit-identical.
    Threshold Frangi on linearly resampled native grid, then physically dilate by <=1.3 mm.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation
    core=(frangi_response>=.02)
    rad=1.3
    widths=[int(np.ceil(rad/s)) for s in spacing]
    axes=np.ogrid[tuple(slice(-w,w+1) for w in widths)]
    footprint=sum((v*spacing[i])**2 for i,v in enumerate(axes))<=rad**2+1e-7
    support=binary_dilation(core,structure=footprint)
    return support & roi & (hu>=120.) & (hu<=1400.)

def select_variants(bits,roi,hu,soft,frangi,spacing,old):
    """All proposal masks are determined without reference segmentations."""
    import numpy as np
    masks={k:{} for k in CANDIDATES}
    for artery,(lab,j) in HEADS.items():
        raw=(bits&(1<<j))!=0
        base=raw&roi
        plausible=base&(hu>=HU_MIN)&(hu<=HU_MAX)
        nraw=int(old['per_artery'][artery]['raw_prediction_voxels'])
        soft15=fixed_budget_rank(plausible,soft,math.floor(.15*nraw))
        for name in CANDIDATES:
            masks[name][artery]=soft15
        masks['hu80'][artery]=plausible
        if artery=='LM':
            ridge=lm_ridge_support(frangi,hu,roi,spacing)&raw
            masks['lm_ridge_only'][artery]=ridge
            masks['lm_ridge_union_soft15'][artery]=ridge|soft15
            masks['lm_soft25'][artery]=fixed_budget_rank(plausible,soft,math.floor(.25*nraw))
    return masks

def safe_stats(x):
    import numpy as np
    arr=np.asarray(x,dtype=np.float64)
    if not len(arr):return {'n':0,'p10':None,'median':None,'p90':None}
    return {'n':int(len(arr)),'p10':float(np.percentile(arr,10)),'median':float(np.median(arr)),
            'p90':float(np.percentile(arr,90))}

def lm_diagnostic(v22_lm,uniform_lm,ridge_lm,roi,hu,fr,soft,da,labels,spacing):
    """Post-hoc only: label coordinates never feed select_variants or any mask."""
    import numpy as np
    from scipy.spatial import cKDTree
    known=(labels==HEADS['LM'][0])&v22_lm
    kept=known&uniform_lm
    lost=known&(~uniform_lm)
    rescued=lost&ridge_lm
    nearby=(labels==2)|(labels==3)
    kdtree=None
    if nearby.any():
        coords=np.argwhere(nearby)
        if len(coords)>50000:coords=coords[::max(1,len(coords)//50000)]
        kdtree=cKDTree(coords*np.asarray(spacing))
    def near_bif(mask):
        pts=np.argwhere(mask)
        if kdtree is None or not len(pts):return safe_stats([])
        distances=kdtree.query(pts*np.asarray(spacing),k=1,workers=1)[0]
        return safe_stats(distances)
    def group(mask):
        return {'hu':safe_stats(hu[mask]),'frangi':safe_stats(fr[mask]),
                'soft_score':safe_stats(soft[mask]),'aorta_distance_mm':safe_stats(da[mask]),
                'distance_to_annotated_LAD_or_LCX_mm':near_bif(mask),
                'fraction_within_5mm_aorta':float(np.mean(da[mask]<=5)) if mask.any() else None}
    # Distance to LAD/LCX labels is *not* confirmation of the LM bifurcation.
    return {'known_v22_LM_hits':int(known.sum()),'soft15_kept':int(kept.sum()),
            'soft15_lost':int(lost.sum()),'lost_rescued_by_ct_ridge':int(rescued.sum()),
            'lost_kept_neither_method':int((lost&(~ridge_lm)).sum()),
            'kept_soft15':group(kept),'lost_soft15':group(lost),
            'lost_rescued_by_ridge':group(rescued),
            'reference_only_posthoc':True,
            'aorta_distance_proxy_not_true_origin':True}

def per_case(cid,group,force=False):
    import numpy as np
    import nibabel as nib
    path=OUT/'per_case'/f'{cid}.json'
    mask_dest=OUT/'candidate_masks'/f'{cid}.v26_primary_4head_native.nii.gz'
    src={'ct':CT_ROOT/f'{cid}.img.nii.gz','labels':SEGS/f'{cid}.coronary.nii.gz',
         'v22':V22/'prediction_masks'/f'{cid}.v22_4head_native_raw.nii.gz',
         'heart':V23/'totalseg_masks'/cid/'heart.nii.gz',
         'aorta':V23/'totalseg_masks'/cid/'aorta.nii.gz'}
    checks={k:sha(p) for k,p in src.items()}
    if (not force) and path.exists() and mask_dest.exists():
        x=read_json(path)
        if (x.get('version')==VERSION and x.get('source_sha256')==checks and
            x.get('primary_mask_sha256')==sha(mask_dest) and x.get('cohort_group')==group):
            return x,'REUSED'
    ct=nib.load(str(src['ct']));p=nib.load(str(src['v22']));lab=nib.load(str(src['labels']))
    if not(ct.shape==p.shape==lab.shape and np.allclose(ct.affine,p.affine,atol=1e-3) and
        np.allclose(ct.affine,lab.affine,atol=1e-3)):
        raise RuntimeError(f'Source CT/raw prediction/label affine mismatch {cid}')
    sp=tuple(float(x) for x in ct.header.get_zooms()[:3]);voxel_mm3=float(np.prod(sp))
    heart=native_mask(src['heart'],ct)
    aorta=native_mask(src['aorta'],ct)
    crop=crop_bbox(heart,sp)
    roi,da=roi_and_da(heart[crop],aorta[crop],sp)
    del heart,aorta
    hu=np.asarray(ct.dataobj[crop],dtype=np.float32)
    bits=np.asarray(p.dataobj[crop],dtype=np.uint8)
    if hu.shape!=roi.shape or bits.shape!=roi.shape or np.any(bits>15) or not np.isfinite(hu).all():
        raise RuntimeError('Malformed cropped image/data '+cid)
    fr=vesselness_coarse(hu)
    scores=soft_score(hu,fr)
    old=read_json(V22/'per_case'/f'{cid}.json')
    selection=select_variants(bits,roi,hu,scores,fr,sp,old)
    # Only here can annotations be read; all proposal masks have already been finalized.
    truth=np.asarray(lab.dataobj[crop],dtype=np.uint8)
    eligible={a:bool(old['per_artery'][a]['reference_positive_eligible']) for a in HEADS}
    metrics={}
    native_bits=np.zeros(hu.shape,np.uint8)
    for variant in CANDIDATES:
        vrows={}
        for artery,(label,j) in HEADS.items():
            mask=selection[variant][artery]
            n=int(mask.sum());total=int(old['per_artery'][artery]['raw_prediction_voxels'])
            pos=truth==label
            hits=int((mask&pos).sum()) if eligible[artery] else None
            rawhits=int(old['per_artery'][artery]['annotated_positive_hits'] or 0) if eligible[artery] else None
            conflicting=np.isin(truth,[k for a,(k,_) in HEADS.items() if a!=artery and eligible[a]])
            row=dict(raw_prediction_voxels=total,prediction_voxels=n,
                     volume_mm3=n*voxel_mm3,
                     removed_vs_v22=1-n/max(1,total),
                     eligible_reference=eligible[artery],
                     v22_positive_hits=rawhits,
                     retained_reference_hits=hits,
                     positive_retention=(hits/rawhits if hits is not None and rawhits else None),
                     hits_other_known_major_labels=int((mask&conflicting).sum()) if eligible[artery] else None,
                     low_HU_voxels=int((mask&(hu < -300)).sum()))
            if variant==PRIMARY:
                row.update(topology_metrics(mask,sp))
                native_bits[mask]|=np.uint8(1<<j)
            vrows[artery]=row
        metrics[variant]=vrows
    diagnostics=(lm_diagnostic((bits&1)!=0,
        selection['uniform_soft15']['LM'],selection['lm_ridge_only']['LM'],roi,hu,fr,scores,da,truth,sp)
        if group=='DEVELOPMENT_PILOT' and eligible['LM'] else None)
    native=np.zeros(ct.shape,np.uint8);native[crop]=native_bits
    mask_dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=mask_dest.with_name(mask_dest.name+'.partial.nii.gz')
    nib.save(nib.Nifti1Image(native,ct.affine),str(tmp))
    os.replace(tmp,mask_dest)
    result=dict(version=VERSION,case_id=cid,cohort_group=group,source_sha256=checks,
           primary_mask_sha256=sha(mask_dest),shape=list(ct.shape),spacing_mm=sp,
           variants=metrics,lm_annotation_diagnostic=diagnostics,
           frozen_model_checksum=MODEL_SHA,
           reference_only_posthoc=True,unknown_outside_reference=True,
           candidates_not_validated_coronary=True,
           no_training=True,no_test_reads=True,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(path,result)
    del hu,bits,roi,da,fr,scores,selection,truth,native,native_bits
    gc.collect()
    return result,'NEW'

def aggregate(ids):
    import numpy as np
    cases=[read_json(OUT/'per_case'/f'{cid}.json') for cid in ids if (OUT/'per_case'/f'{cid}.json').exists()]
    a=dict(n_expected=len(ids),n_complete=len(cases),missing=[c for c in ids if not (OUT/'per_case'/f'{c}.json').exists()],variants={})
    for variant in CANDIDATES:
        totals={h:dict(raw=0,pred=0,rawhits=0,hits=0,n_eligible=0,components=[],known_other=0)
                for h in HEADS}
        for case in cases:
            for h in HEADS:
                row=case['variants'][variant][h];t=totals[h]
                t['raw']+=row['raw_prediction_voxels'];t['pred']+=row['prediction_voxels']
                if row['eligible_reference']:
                    t['rawhits']+=row['v22_positive_hits'] or 0
                    t['hits']+=row['retained_reference_hits'] or 0
                    t['known_other']+=row['hits_other_known_major_labels'] or 0
                    t['n_eligible']+=1
                if 'n_components_ge_10mm3' in row:t['components'].append(row['n_components_ge_10mm3'])
        raw=sum(t['raw'] for t in totals.values());ret=sum(t['pred'] for t in totals.values())
        a['variants'][variant]=dict(raw_prediction_voxels=raw,remaining_prediction_voxels=ret,
            weighted_removed=1-ret/max(1,raw),per_artery={h:dict(
                 eligible_cases=t['n_eligible'],v22_hit_retention=t['hits']/max(1,t['rawhits']),
                 raw_pred_removed=1-t['pred']/max(1,t['raw']),remaining_voxels=t['pred'],
                 known_other_hits=t['known_other'],
                 median_components_ge_10mm3=(float(np.median(t['components'])) if t['components'] else None))
                 for h,t in totals.items()})
    return a

def gate(hold):
    if hold['n_complete']!=HOLDOUT_N:return dict(complete=False,passed=False,reason='holdout incomplete')
    row=hold['variants'][PRIMARY]
    eligible={h:d['eligible_cases'] for h,d in row['per_artery'].items()}
    retained={h:d['v22_hit_retention'] for h,d in row['per_artery'].items()}
    passed=(all(n>=GATE['min_eligible_cases_per_artery'] for n in eligible.values()) and
          all(x>=GATE['min_retention_each_artery'] for x in retained.values()) and
          row['weighted_removed']>=GATE['min_raw_pred_removed'])
    return dict(complete=True,passed=passed,criteria=GATE,eligible_cases=eligible,
                per_artery_retention=retained,weighted_raw_removed=row['weighted_removed'],
                interpretation='Candidate-screen pilot criteria; not specificity/clinical anatomy')


def summarize_lm_diagnostics():
    import numpy as np
    rows=[]
    for cid in PILOT:
        p=OUT/'per_case'/f'{cid}.json'
        if p.exists():
            d=read_json(p).get('lm_annotation_diagnostic')
            if d is not None:rows.append(dict(case_id=cid,**d))
    if not rows:return dict(eligible_cases=0)
    raw=sum(x['known_v22_LM_hits'] for x in rows)
    lost=sum(x['soft15_lost'] for x in rows)
    rescued=sum(x['lost_rescued_by_ct_ridge'] for x in rows)
    return dict(eligible_cases=len(rows),total_known_LM_hits=raw,
        soft15_LM_lost_hits=lost,soft15_LM_lost_fraction=lost/max(1,raw),
        ridge_rescued_LM_hits=rescued,ridge_rescue_fraction_of_soft15_loss=rescued/max(1,lost),
        per_case=[dict(case_id=r['case_id'],known=r['known_v22_LM_hits'],
            soft15_lost=r['soft15_lost'],ridge_rescued=r['lost_rescued_by_ct_ridge'],
            lost_median_Frangi=r['lost_soft15']['frangi']['median'],
            lost_median_HU=r['lost_soft15']['hu']['median'],
            lost_median_aorta_mm=r['lost_soft15']['aorta_distance_mm']['median'],
            lost_median_distance_to_LAD_LCX_mm=r['lost_soft15']['distance_to_annotated_LAD_or_LCX_mm']['median'])
            for r in rows],
        warning='Distance to labeled LAD/LCX and aorta are diagnostic proxies, not proven coronary junctions')

def write_report(ids,stage):
    pilot=aggregate(PILOT)
    hold=aggregate(heldout_ids(ids))
    whole=aggregate(tuple(PILOT)+heldout_ids(ids))
    summary=dict(version=VERSION,stage=stage,development_pilot=pilot,
                 independent_development_evaluation=hold,combined_descriptive=whole,
                 lm_diagnostic_pilot=summarize_lm_diagnostics(),
                 independent_gate=gate(hold),no_training=True,test_reads=0,
                 test_set_sealed=True,no_labels_used_for_candidate_generation=True)
    write_json(OUT/'summary_v26.json',summary)
    rows=[]
    for title,cohort in [('Prior pilot (NOT independent)',pilot),('Additional 16 development cases',hold)]:
        for variant,v in cohort['variants'].items():
            vals=''.join(f'<td>{v["per_artery"][h]["v22_hit_retention"]:.1%}</td>' for h in HEADS)
            rows.append(f'<tr><td>{html.escape(title)}</td><td>{html.escape(variant)}</td><td>{v["weighted_removed"]:.1%}</td>{vals}</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>OpenPlaque V26 LM holdout</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:25px auto;padding:10px}
table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child,td:nth-child(2){text-align:left}pre{white-space:pre-wrap}</style>
<h1>OpenPlaque V26 — LM ridge-support diagnostic and independent-DEV80 screen</h1>
<p>Research only; sparse reference annotations contain known positives, not verified negatives. No clinical artery identity or origin established.</p>'''
    page+=f'<p>Stage: {html.escape(stage)}. Pilot: {pilot["n_complete"]}/12. Additional development cases: {hold["n_complete"]}/{HOLDOUT_N}. <b>Evaluation gate:</b> {"PASS" if summary["independent_gate"]["passed"] else "FAIL/NOT COMPLETE"}.</p>'
    page+='<table><tr><th>Group</th><th>Candidate</th><th>Raw predictions removed</th>'+''.join(f'<th>{h} hits retained</th>' for h in HEADS)+'</tr>'+''.join(rows)+'</table>'
    diag=summary['lm_diagnostic_pilot']
    page+=f'<h2>Pilot LM failure localization</h2><p>Known LM hits lost with uniform soft15: {diag.get("soft15_LM_lost_hits",0)}; rescued by CT ridge support: {diag.get("ridge_rescued_LM_hits",0)}. Aorta and bifurcation proximity remain diagnostic proxies only.</p>'
    page+='<h2>Machine-readable summary</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'
    p=OUT/'OPENPLAQUE_V26_LM_HOLDOUT_REPORT.html'
    tmp=p.with_name(p.name+'.partial');tmp.write_text(page,encoding='utf8');os.replace(tmp,p)
    return summary

def run(force=False):
    ids=preflight()
    for group,group_ids in [('DEVELOPMENT_PILOT',PILOT),('INDEPENDENT_DEV16',heldout_ids(ids))]:
        for i,cid in enumerate(group_ids,1):
            try:
                _,state=per_case(cid,group,force=force)
                print(f'V26 {group} {i}/{len(group_ids)} {cid}: {state}',flush=True)
                if i%4==0:
                    write_report(ids,group+'_IN_PROGRESS')
            except Exception as e:
                failures=OUT/'failures_v26.json'
                existing=read_json(failures) if failures.exists() else []
                existing.append(dict(case=cid,group=group,error=str(e),traceback=traceback.format_exc()[-2400:]))
                write_json(failures,existing)
                write_json(OUT/'run_status_v26.json',dict(status='ERROR',case=cid,group=group,error=str(e)))
                raise
    report=write_report(ids,'EVALUATION_FINISHED')
    write_json(OUT/'run_status_v26.json',dict(status='COMPLETE_16_NEW_DEV_CASES',
              pilot_cases=12,independent_dev_cases=HOLDOUT_N,
              independent_gate=report['independent_gate'],test_reads=0))
    print('V26 HELDOUT GATE',json.dumps(report['independent_gate'],indent=2),flush=True)
    print('V26 stored',OUT)

def selftest():
    import numpy as np
    from scipy.ndimage import gaussian_filter
    cases=[str(i) for i in range(1,81)]
    pilot=('1','2')
    ranked=sorted(set(cases)-set(pilot),key=lambda c:(hashlib.sha256((VERSION+'|untouched-dev80|'+c).encode()).hexdigest(),int(c)))
    assert len(ranked)==78 and ranked==sorted(ranked,key=lambda c:(hashlib.sha256((VERSION+'|untouched-dev80|'+c).encode()).hexdigest(),int(c)))
    image=np.zeros((20,20,18),np.float32);image[6:14,6:14,6:13]=175
    image=gaussian_filter(image,1)
    vessel=vesselness_coarse(image)
    sc=soft_score(image,vessel)
    roi=np.ones(image.shape,bool)
    bits=np.ones(image.shape,np.uint8)
    old=dict(per_artery={h:dict(raw_prediction_voxels=image.size) for h in HEADS})
    masks=select_variants(bits,roi,image,sc,vessel,(.5,.5,.8),old)
    assert set(masks)==set(CANDIDATES)
    for method in CANDIDATES:
        for h in HEADS:assert masks[method][h].shape==image.shape
    assert (masks['lm_ridge_union_soft15']['LM']|masks['uniform_soft15']['LM']).sum()==masks['lm_ridge_union_soft15']['LM'].sum()
    assert np.all(masks['lm_ridge_only']['LM']<=masks['lm_ridge_union_soft15']['LM'])
    assert all(np.array_equal(masks['uniform_soft15'][h],masks['lm_ridge_only'][h]) for h in ('LAD','LCX','RCA'))
    points=lm_diagnostic(np.ones(image.shape,bool),np.zeros(image.shape,bool),np.ones(image.shape,bool),roi,image,vessel,sc,np.ones(image.shape),np.ones(image.shape,np.uint8),(.5,.5,.8))
    assert points['soft15_lost']==image.size and points['lost_rescued_by_ct_ridge']==image.size
    print('PASS: frozen holdout algorithm, immutable methods, LM-only branch, score shape, posthoc diagnostic')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('test','preflight','run','report'));p.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.stage=='test':selftest();return
    if args.stage=='preflight':preflight();return
    if args.stage=='run':run(force=args.force);return
    if args.stage=='report':
        ids=preflight();r=write_report(ids,'MANUAL_REPORT');print(r['independent_gate']);return

if __name__=='__main__':main()
