"""OpenPlaque V25: CT-only soft tubularity ranking of frozen V22 four-head candidates.

Research diagnostic. Reference annotations ONLY for retrospective assessment. No verified negatives,
no coronary identity promotion, no training, no sealed test data. No label-guided candidates.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'openplaque-v25-dev80-soft-ranked-tubularity-20261014-1'
ROOT = Path('/content/drive/MyDrive/OpenPlaque')
BASE = ROOT / 'OpenPlaque_Evaluation_Dataset_v0_2'
V22 = BASE/'evaluation_v22_full_volume'
V23 = BASE/'evaluation_v23_heart_aorta_roi'
V24 = BASE/'evaluation_v24_vessel_candidates'
OUT = BASE/'evaluation_v25_soft_vessel_candidates'
CT_ROOT = ROOT/'ExternalData/ImageCAS_OriginalCCTA_MatchedTo_ImageCASX_v1/volumes'
SEGS = ROOT/'Cache/ImageCAS_X_Pretrained_v1/ImageCAS-X_dataset.zip_extracted/ImageCAS-X_dataset/segmentations'
DEV_FILE = BASE/'internal_dev_80_frozen.txt'
DEV_SHA = 'ef2bd3bdbe7dc9cbf4639e62d80dceb53108ac836acff64617e0f17b4f29be4f'
MODEL_SHA = 'f789eede621816433240232455d7a7862c7eafc69a979a50938e73349a6796b9'
HEADS = {'LM':(1,0),'LAD':(2,1),'LCX':(3,2),'RCA':(9,3)}
PILOT = ('11','142','203','305','370','455','562','633','708','776','853','976')
COARSE_STEP = (3,3,2)
FRANGI_SIGMA = (0.9,1.6,2.5)
HU_MIN, HU_MAX = 80., 1600.
SOFT_BUDGETS = {'soft20':0.20,'soft15':0.15,'soft10':0.10,'soft15_no_tube':0.15,'soft15_context':0.15}
PILOT_GATE = {'variant':'soft15','retention_each_artery':0.90,'raw_prediction_removed':0.85}


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

def plan():
    return dict(version=VERSION,dev_ids_sha256=DEV_SHA,model_checkpoint_sha256=MODEL_SHA,
        pilot_case_ids=list(PILOT),source_v22='evaluation_v22_full_volume',
        source_v23='evaluation_v23_heart_aorta_roi',source_v24='evaluation_v24_vessel_candidates',
        ct_only_candidate_generation=True,reference_only_for_posthoc_metrics=True,
        no_training=True,no_test_access=True,unknown_voxels_are_not_negatives=True,
        anatomical_margin_mm=10.0,anatomical_aorta_proximity_mm=8.0,
        hu_soft_bounds=[HU_MIN,HU_MAX],coarse_step=list(COARSE_STEP),
        frangi_sigmas_coarse_vox=list(FRANGI_SIGMA),
        score_weights={'attenuation':.45,'coarse_frangi':.35,'nonbulk_local_context':.20},
        score_details='HU / 420 from 60HU, Frangi / 0.035, density(7,7,5) penalty; clipped 0..1',
        candidate_budgets_vs_original_v22=dict(SOFT_BUDGETS),pilot_gate=PILOT_GATE,
        old_v24_hard_gate_unchanged=True,only_pilot_v24_baselines_independent_of_v25=False,
        component_cutoff_mm3=10.,aorta_is_a_proxy_not_confirmed_ostium=True)

def preflight():
    assert DEV_FILE.exists(),f'Missing DEV80 split: {DEV_FILE}'
    ids=[x.strip() for x in DEV_FILE.read_text().splitlines() if x.strip()]
    assert len(ids)==len(set(ids))==80 and set(PILOT).issubset(set(ids)),'Frozen split invalid'
    digest=hashlib.sha256((''.join(i+'\n' for i in sorted(ids,key=int))).encode()).hexdigest()
    assert digest==DEV_SHA,'Frozen DEV80 hash mismatch'
    m22=read_json(V22/'whole_volume_summary_v22.json')
    assert m22['complete'] and m22['cases_complete']==80 and m22['model_checkpoint_sha256']==MODEL_SHA and m22['frozen_dev_sha256']==DEV_SHA,'Frozen V22 provenance mismatch'
    m23=read_json(V23/'summary_v23.json')
    assert m23['n_complete']==80 and m23['test_reads']==0 and m23['stage']=='full_complete','V23 anatomy is not complete'
    m24=read_json(V24/'summary_v24.json')
    assert m24['n_cases_complete']==12 and m24['pilot_gate']['passed'] is False,'V24 pilot provenance unexpected'
    m24plan=read_json(V24/'manifest_v24.json')
    assert set(m24plan['pilot_case_ids'])==set(PILOT),'V24 pilot IDs changed'
    frozen=read_json(BASE/'sparse_topology_results_v0_3/frozen_positive_reference_manifest_v0_3.json')
    assert frozen['status']=='FROZEN_POSITIVE_REFERENCE_QA_ONLY' and frozen['sealed_test_accessed'] is False
    for cid in ids:
        for f in (CT_ROOT/f'{cid}.img.nii.gz',
                  V22/'prediction_masks'/f'{cid}.v22_4head_native_raw.nii.gz',
                  V22/'per_case'/f'{cid}.json',
                  V23/'totalseg_masks'/cid/'heart.nii.gz',
                  V23/'totalseg_masks'/cid/'aorta.nii.gz'):
            if not f.is_file():raise FileNotFoundError(str(f))
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest_v25.json'; expected=plan()
    if manifest.exists() and read_json(manifest)!=expected:raise RuntimeError('Do not mix different V25 plans')
    if not manifest.exists():write_json(manifest,expected)
    print('PREFLIGHT PASS, frozen 80-case split; 12-case V24 pilot available; sealed test untouched')
    return ids

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

def case_result(cid,save_mask):
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt
    report=OUT/'per_case'/f'{cid}.json'
    prediction=V22/'prediction_masks'/f'{cid}.v22_4head_native_raw.nii.gz'
    heartpath=V23/'totalseg_masks'/cid/'heart.nii.gz'
    aortapath=V23/'totalseg_masks'/cid/'aorta.nii.gz'
    ctpath=CT_ROOT/f'{cid}.img.nii.gz'
    refpath=SEGS/f'{cid}.coronary.nii.gz'
    needed=(prediction,heartpath,aortapath,ctpath,refpath)
    for f in needed:
        if not f.is_file():raise FileNotFoundError(str(f))
    fp={name:sha(path) for name,path in zip(('v22','heart','aorta','ct','reference'),needed)}
    mask_out=OUT/'candidate_masks'/f'{cid}.v25_soft15_bitheads_native.nii.gz'
    if report.exists():
        existing=read_json(report)
        if existing.get('version')==VERSION and existing.get('input_sha256')==fp and mask_out.is_file() and sha(mask_out)==existing['candidate_mask_sha256']:
            return existing,'REUSED'
    ctnii=nib.load(str(ctpath));prednii=nib.load(str(prediction));refnii=nib.load(str(refpath))
    if ctnii.shape!=prednii.shape or ctnii.shape!=refnii.shape or not np.allclose(ctnii.affine,prednii.affine,atol=1e-3) or not np.allclose(ctnii.affine,refnii.affine,atol=1e-3):
        raise RuntimeError('CT/prediction/reference geometry mismatch: '+cid)
    sp=tuple(float(x) for x in ctnii.header.get_zooms()[:3]);voxel_mm3=float(np.prod(sp))
    heart=native_mask(heartpath,ctnii)
    aorta=native_mask(aortapath,ctnii)
    crop=crop_bbox(heart,sp)
    hc=heart[crop];ac=aorta[crop]
    del heart,aorta
    roi=roi_from_heart_aorta(hc,ac,sp)
    del hc,ac
    bits=np.asarray(prednii.dataobj[crop],dtype=np.uint8)
    assert bits.shape==roi.shape and not np.any(bits>15),'Raw 4-head bitmask corrupt'
    hu=np.asarray(ctnii.dataobj[crop],dtype=np.float32)
    if not np.isfinite(hu).all():raise RuntimeError('Nonfinite CCTA')
    fr=vesselness_coarse(hu)
    score=soft_score(hu,fr)
    from scipy.ndimage import gaussian_filter
    fr_scaled=np.clip(fr/0.035,0.,1.)
    score_nov=np.asarray(score-.35*fr_scaled,dtype=np.float32)
    score_context=np.asarray(score-.10*fr_scaled+.10*gaussian_filter(fr_scaled,sigma=(1.5,1.5,1.)),dtype=np.float32)
    coarse_diag=dict(frangi_max=float(fr.max()),frangi_p99=float(np.quantile(fr[roi],0.99)) if roi.any() else 0.,
       frangi_nonzero_fraction=float((fr[roi]>0).mean()) if roi.any() else 0.)
    del fr,fr_scaled
    eligible={}
    old=read_json(V22/'per_case'/f'{cid}.json')
    for name in HEADS:
        eligible[name]=bool(old['per_artery'][name]['reference_positive_eligible'])
    # Labels are loaded ONLY AFTER all CT-derived scores and ROIs have been built.
    labels=np.asarray(refnii.dataobj[crop],dtype=np.uint8)
    outputs={key:np.zeros(hu.shape,np.uint8) if key=='soft15' else None for key in ('soft15',)}
    variants={key:{} for key in ('roi10','hu80',*SOFT_BUDGETS)}
    attr={}
    for artery,(label,j) in HEADS.items():
        raw=int(old['per_artery'][artery]['raw_prediction_voxels'])
        p=(bits&(1<<j))!=0
        base=p&roi
        plausible=base&(hu>=HU_MIN)&(hu<=HU_MAX)
        masks={'roi10':base,'hu80':plausible}
        for variant,budget in SOFT_BUDGETS.items():
            variant_score=score_nov if variant=='soft15_no_tube' else (score_context if variant=='soft15_context' else score)
            masks[variant]=fixed_budget_rank(plausible,variant_score,math.floor(raw*budget))
        pos=(labels==label)
        raw_hits=int(old['per_artery'][artery]['annotated_positive_hits'] or 0)
        reference_eligible=eligible[artery]
        losses=dict(raw_positive_hits=raw_hits,
            roi10_hits=int((masks['roi10']&pos).sum()) if reference_eligible else None,
            hu80_hits=int((masks['hu80']&pos).sum()) if reference_eligible else None,
            frangi_soft15_hits=int((masks['soft15']&pos).sum()) if reference_eligible else None,
            v24_hu10_hits=None,v24_tube_lo_hits=None)
        v24file=V24/'per_case'/f'{cid}.json'
        if v24file.is_file():
            oldv24=read_json(v24file).get('per_variant',{})
            for v,k in [('hu10','v24_hu10_hits'),('tube_lo','v24_tube_lo_hits')]:
                a=oldv24.get(v,{}).get('per_artery',{}).get(artery,{})
                losses[k]=a.get('reference_positive_hits')
                if losses[k] is None:losses[k]=a.get('reference_positive_hits_retained')
        attr[artery]=losses
        for key,m in masks.items():
            d={'prediction_voxels':int(m.sum()),'prediction_mm3':int(m.sum())*voxel_mm3,
               'fraction_raw_prediction_removed':1-int(m.sum())/max(1,raw),
               'raw_prediction_voxels':raw,
               'hit_retention_vs_v22':(int((m&pos).sum())/raw_hits if reference_eligible and raw_hits else None),
               'known_positive_hits':int((m&pos).sum()) if reference_eligible else None,
               'known_positive_count':int(pos.sum()) if reference_eligible else None,
               'known_positive_recall':(int((m&pos).sum())/max(1,int(pos.sum())) if reference_eligible else None),
               'hits_on_other_known_arteries':(int((m&np.isin(labels,[lab for n,(lab,_) in HEADS.items() if n!=artery and eligible[n]])).sum()) if reference_eligible else None),
               'low_hu_predicted':int((m&(hu<-300)).sum()),
               'reference_eligible':reference_eligible}
            if key=='soft15':
                d.update(topology_metrics(m,sp))
                outputs['soft15'][m] |= (1<<j)
            variants[key][artery]=d
    if save_mask:
        full=np.zeros(ctnii.shape,dtype=np.uint8)
        full[crop]=outputs['soft15']
        mask_out.parent.mkdir(parents=True,exist_ok=True)
        temp=mask_out.with_name(mask_out.name+'.tmp.nii.gz')
        nib.save(nib.Nifti1Image(full,ctnii.affine),str(temp))
        os.replace(temp,mask_out)
        candidate_sha=sha(mask_out)
    else: candidate_sha=None
    result={'version':VERSION,'case_id':cid,'input_sha256':fp,'candidate_mask_sha256':candidate_sha,
            'crop_shape':list(hu.shape),'spacing_mm':sp,'coarse_frangi':coarse_diag,
            'variants':variants,'loss_attribution':attr,'reference_only_posthoc':True,
            'no_new_training':True,'no_test_reads':True,'candidate_not_anatomical_truth':True,
            'finished_utc':datetime.now(timezone.utc).isoformat()}
    write_json(report,result)
    return result,'NEW'

def aggregate(ids):
    present=[read_json(OUT/'per_case'/f'{cid}.json') for cid in ids if (OUT/'per_case'/f'{cid}.json').is_file()]
    agg={'version':VERSION,'n_expected':len(ids),'n_complete':len(present),'missing':[cid for cid in ids if not (OUT/'per_case'/f'{cid}.json').is_file()],
         'variants':{},'test_reads':0,'training':False,'unknown_not_verified_negative':True}
    for variant in ('roi10','hu80',*SOFT_BUDGETS):
        totals={artery:{'raw_voxels':0,'predicted_voxels':0,'raw_hits':0,'retained_hits':0,'eligible_cases':0,
                       'component_counts':[]} for artery in HEADS}
        for case in present:
            old=read_json(V22/'per_case'/f"{case['case_id']}.json")
            for artery in HEADS:
                row=case['variants'][variant][artery];t=totals[artery]
                t['raw_voxels']+=row['raw_prediction_voxels'];t['predicted_voxels']+=row['prediction_voxels']
                if row['reference_eligible']:
                    t['eligible_cases']+=1;t['raw_hits']+=int(old['per_artery'][artery]['annotated_positive_hits'] or 0)
                    t['retained_hits']+=int(row['known_positive_hits'] or 0)
                if 'n_components_ge_10mm3' in row:t['component_counts'].append(row['n_components_ge_10mm3'])
        total_raw=sum(t['raw_voxels'] for t in totals.values())
        total_pred=sum(t['predicted_voxels'] for t in totals.values())
        per_artery={}
        for artery,t in totals.items():
            import numpy as np
            per_artery[artery]={'eligible_cases':t['eligible_cases'],
                'v22_hit_retention':t['retained_hits']/max(1,t['raw_hits']),
                'raw_prediction_removed':1-t['predicted_voxels']/max(1,t['raw_voxels']),
                'raw_predicted_voxels':t['raw_voxels'],'retained_predicted_voxels':t['predicted_voxels'],
                'median_components_ge_10mm3':float(np.median(t['component_counts'])) if t['component_counts'] else None}
        agg['variants'][variant]={'weighted_raw_pred_removed':1-total_pred/max(1,total_raw),'per_artery':per_artery}
    # Attribution uses exactly V24's frozen 12 pilot measurements; do not imply causal
    # separation between resampling, native image quality, or Frangi sigma effects.
    aggregate_losses={}
    for artery in HEADS:
        totals={k:0 for k in ('raw','v24_roi','v24_hu','v24_frangi','v24_shape','v25_hu80','v25_soft15')}
        for case in present:
            v24path=V24/'per_case'/f"{case['case_id']}.json"
            if not v24path.exists():continue
            v24=read_json(v24path)
            def value(v):
                d=v24.get('per_variant',{}).get(v,{}).get('per_artery',{}).get(artery,{})
                return d.get('reference_positive_hits') if d.get('reference_class_eligible') else None
            raw=read_json(V22/'per_case'/f"{case['case_id']}.json")['per_artery'][artery]
            if not raw['reference_positive_eligible']:continue
            vals={'raw':raw['annotated_positive_hits'], 'v24_roi':value('roi10'),
                  'v24_hu':value('hu10'),'v24_frangi':value('tube_lo'),
                  'v24_shape':value('tube_lo_shape'),
                  'v25_hu80':case['variants']['hu80'][artery]['known_positive_hits'],
                  'v25_soft15':case['variants']['soft15'][artery]['known_positive_hits']}
            if all(v is not None for v in vals.values()):
                for k,v in vals.items():totals[k]+=int(v)
        if totals['raw']:
            aggregate_losses[artery]={'V24_roi_loss_vs_V22':1-totals['v24_roi']/totals['raw'],
               'V24_HU120_loss_increment_fraction_of_V22_hits':(totals['v24_roi']-totals['v24_hu'])/totals['raw'],
               'V24_Frangi_hard_loss_increment_fraction_of_V22_hits':(totals['v24_hu']-totals['v24_frangi'])/totals['raw'],
               'V24_shape_loss_increment_fraction_of_V22_hits':(totals['v24_frangi']-totals['v24_shape'])/totals['raw'],
               'V25_HU80_hit_retention':totals['v25_hu80']/totals['raw'],
               'V25_soft15_hit_retention':totals['v25_soft15']/totals['raw'],
               'cases_from_saved_V24_pilot_only':min(len(present),12)}
    agg['paired_V24_pilot_loss_attribution']=aggregate_losses
    return agg

def pilot_gate(summary):
    if summary['n_complete']!=12:return {'complete':False,'passed':False}
    row=summary['variants'][PILOT_GATE['variant']]
    per={a:d['v22_hit_retention'] for a,d in row['per_artery'].items()}
    passed=all(v>=PILOT_GATE['retention_each_artery'] for v in per.values()) and row['weighted_raw_pred_removed']>=PILOT_GATE['raw_prediction_removed']
    return {'complete':True,'passed':passed,'per_artery':per,'weighted_raw_removed':row['weighted_raw_pred_removed'],
        'requirements':PILOT_GATE,'next_step':'CONTINUE_DEV80' if passed else 'HALT_AFTER_PILOT_NO_MODEL_PROMOTION'}

def report(ids,stage):
    summary=aggregate(ids)
    pilot=aggregate(PILOT)
    summary['pilot_gate']=pilot_gate(pilot)
    summary['stage']=stage
    write_json(OUT/'summary_v25.json',summary)
    table=[]
    for name,variant in summary['variants'].items():
        cells=''.join(f"<td>{variant['per_artery'][a]['v22_hit_retention']:.1%}</td>" for a in HEADS)
        table.append(f"<tr><td>{name}</td><td>{variant['weighted_raw_pred_removed']:.1%}</td>{cells}</tr>")
    g=summary['pilot_gate']
    html='''<!doctype html><meta charset="utf-8"><title>OpenPlaque V25 soft vessel candidates</title>
<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:24px auto;padding:10px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style>
<h1>OpenPlaque V25 — soft vessel-candidate ranking</h1>
<p>Research diagnostic only: reference positives incomplete; background unknown; aortic proximity is not coronary origin.</p>'''
    html+=f'<h2>{summary["n_complete"]}/{summary["n_expected"]} cases — {stage}</h2>'
    html+=f'<p><b>12-case pilot:</b> {"PASS" if g["passed"] else "FAIL"}; predeclared soft15 gate 90% hit retention per artery and ≥85% V22 voxel reduction.</p>'
    html+='<table><tr><th>Variant</th><th>V22 prediction removed</th>'+''.join('<th>'+n+' known hits retained</th>' for n in HEADS)+'</tr>'+''.join(table)+'</table>'
    html+='<h3>Machine-readable summary</h3><pre>'+json.dumps(summary,indent=2)+'</pre>'
    path=OUT/'OPENPLAQUE_V25_SOFT_VESSEL_REPORT.html'
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(html);os.replace(tmp,path)
    return summary

def run(full_after_pilot):
    ids=preflight()
    errors=[]
    for i,cid in enumerate(PILOT,1):
        try:
            _,state=case_result(cid,save_mask=True);print(f'PILOT {i}/12 {cid}: {state}',flush=True)
        except Exception as e:
            errors.append({'case':cid,'error':str(e),'traceback':traceback.format_exc()[-2500:]})
            write_json(OUT/'failures_v25.json',errors)
            write_json(OUT/'run_status_v25.json',{'status':'ERROR','last_id':cid,'error':str(e)})
            raise
    res=report(ids,'pilot_finished');gate=res['pilot_gate']
    write_json(OUT/'run_status_v25.json',{'status':'PILOT_GATE_PASSED' if gate['passed'] else 'PILOT_GATE_FAILED_FULL_NOT_RUN',
                                         'pilot_gate':gate,'n_complete':res['n_complete']})
    print('V25 PILOT GATE',json.dumps(gate,indent=2),flush=True)
    if not gate['passed'] or not full_after_pilot:return
    for i,cid in enumerate(ids,1):
        if cid in PILOT:continue
        try:
            _,state=case_result(cid,save_mask=True);print(f'FULL {i}/80 {cid}: {state}',flush=True)
            if i%4==0:report(ids,'full_in_progress')
        except Exception as e:
            errors.append({'case':cid,'error':str(e),'traceback':traceback.format_exc()[-2500:]})
            write_json(OUT/'failures_v25.json',errors)
            write_json(OUT/'run_status_v25.json',{'status':'ERROR_FULL','last_id':cid,'error':str(e)})
            raise
    res=report(ids,'full_complete')
    write_json(OUT/'run_status_v25.json',{'status':'COMPLETE_80_RESEARCH_ONLY','n_complete':res['n_complete'],'pilot_gate':gate})

def selftest():
    import numpy as np
    arr=np.zeros((13,14,10),bool);arr[1:3,2:4,1:3]=True;arr[9:11,8:10,7:9]=True
    sp=(.5,.5,1.)
    assert topology_metrics(arr,sp)['n_components_all']==2
    sc=np.arange(arr.size,dtype=np.float32).reshape(arr.shape)
    for k in (0,1,4,8,11,16):
        r=fixed_budget_rank(arr,sc,k)
        assert int(r.sum())==min(k,int(arr.sum())) and not np.any(r&~arr)
    tie=fixed_budget_rank(np.ones((3,4,2),bool),np.ones((3,4,2),np.float32),5)
    assert np.flatnonzero(tie).tolist()==list(range(5))
    vol=np.zeros((15,16,12),np.float32)
    vol[4:11,5:10,4:9]=150
    from scipy.ndimage import gaussian_filter
    hu=gaussian_filter(vol,1)
    v=vesselness_coarse(hu)
    assert v.shape==hu.shape and np.isfinite(v).all()
    s=soft_score(hu,v)
    assert s.shape==hu.shape and s.min()>=0 and s.max()<=1.00001
    heart=vol>80;aorta=vol>140
    roi=roi_from_heart_aorta(heart,aorta,sp)
    assert roi[heart].all()
    print('PASS: deterministic rank, 3D components, coarse vesselness geometry, scores, CT-only ROI')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('test','preflight','run','report'))
    p.add_argument('--full-after-pilot',action='store_true')
    args=p.parse_args()
    if args.stage=='test':selftest();return
    if args.stage=='preflight':preflight();return
    if args.stage=='run':run(args.full_after_pilot);return
    if args.stage=='report':
        ids=preflight();x=report(ids,'manual_report');print('V25',x['n_complete'],'/',x['n_expected']);return

if __name__=='__main__':main()
