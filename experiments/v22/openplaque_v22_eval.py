#!/usr/bin/env python3
"""OpenPlaque V22. Full FOV native 80 DEV inference of frozen V21, not training.
Positive-reference research only. No test, expert claims, label writes or clinical accuracy claims.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, html, json, os, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION='openplaque-v22-native-full-volume-v21-dev80-20261012-1'
ROOT=Path('/content/drive/MyDrive/OpenPlaque')
FROZEN=ROOT/'OpenPlaque_Evaluation_Dataset_v0_2'
DATA=ROOT/'Cache/ImageCAS_X_Pretrained_v1/ImageCAS-X_dataset.zip_extracted/ImageCAS-X_dataset'
CT_DIR=ROOT/'ExternalData/ImageCAS_OriginalCCTA_MatchedTo_ImageCASX_v1/volumes'
V7=ROOT/'MT_ImageCASX_Anatomy_Centerline_v1/data_adjudication_v7'
V21=FROZEN/'training_v21_fresh_control'
OUT=FROZEN/'evaluation_v22_full_volume'
CHECKPOINT=V21/'research_checkpoints/resume_latest.pt'
CHECKPOINT_SHA='f789eede621816433240232455d7a7862c7eafc69a979a50938e73349a6796b9'
V21_PLAN='8751dfda1923a57eb08c6c4bfb456622d5d756c667a0ca8278954c8bb96379da'
V21_SELECTION='dc9138356836d237c643b4ad200dbd7ee7d2374253431951772ee4bdee024078'
QA_DEV_SHA='ef2bd3bdbe7dc9cbf4639e62d80dceb53108ac836acff64617e0f17b4f29be4f'
V7_PROPOSAL_SHA='bcfa0542402e772c42b9f89d8c6c52fcf4376ee935e3f8ee74f6c9b8de401c54'
PRIMARY={'LM':1,'LAD':2,'LCX':3,'RCA':9}
PATCH=(64,64,40)
STRIDE=(48,48,30)
THRESHOLD=.5
MAJOR_CC_MM3=10.0


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()

def savejson(path,x):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(tmp,path)

def readjson(path):return json.loads(Path(path).read_text())

def ids(path):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip()]
    if len(lines)!=len(set(lines)) or not all(i.isdecimal() for i in lines):raise RuntimeError('Split has duplicate/invalid IDs: '+str(path))
    return lines

def network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Block(nn.Module):
        def __init__(self,i,o):
            super().__init__();self.seq=nn.Sequential(nn.Conv3d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.ReLU(inplace=True),
                                                      nn.Conv3d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.ReLU(inplace=True))
        def forward(self,x):return self.seq(x)
    class MiniUNet(nn.Module):
        def __init__(self):
            super().__init__();self.b1=Block(1,12);self.b2=Block(12,24);self.b3=Block(24,48)
            self.up2=Block(72,24);self.up1=Block(36,12);self.head=nn.Conv3d(12,4,1)
        def forward(self,x):
            a=self.b1(x);b=self.b2(F.max_pool3d(a,2));c=self.b3(F.max_pool3d(b,2))
            d=self.up2(torch.cat([F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],dim=1))
            e=self.up1(torch.cat([F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a],dim=1))
            return self.head(e)
    return MiniUNet()

def starts(n,p,s):
    if n<p:raise ValueError('CT axis too small for native trained patch')
    seq=list(range(0,n-p+1,s))
    if seq[-1]!=n-p:seq.append(n-p)
    return seq

def weights():
    import numpy as np
    axis=[]
    for n in PATCH:
        v=np.arange(n,dtype='float32')
        axis.append(np.asarray(.18+.82*np.sin(np.pi*(v+.5)/n)**2,dtype='float32'))
    return axis[0][:,None,None]*axis[1][None,:,None]*axis[2][None,None,:]

def preflight(load_model=True):
    import torch
    tr=ids(FROZEN/'train_480_frozen.txt')
    dev=ids(FROZEN/'internal_dev_80_frozen.txt')
    orig_tr=ids(DATA/'filelist/train.txt')
    orig_val=ids(DATA/'filelist/val.txt')
    if len(tr)!=480 or len(dev)!=80 or len(orig_tr)!=560 or len(orig_val)!=80:raise RuntimeError('Split count mismatch')
    if set(tr)&set(dev) or set(tr+dev)!=set(orig_tr) or set(tr+dev)&set(orig_val):raise RuntimeError('Split leakage')
    dev_sha=hashlib.sha256(''.join(x+'\n' for x in sorted(dev,key=int)).encode()).hexdigest()
    qa=readjson(FROZEN/'sparse_topology_results_v0_3/frozen_positive_reference_manifest_v0_3.json')
    if dev_sha!=QA_DEV_SHA or dev_sha!=qa['frozen_dev_sha256'] or qa['status']!='FROZEN_POSITIVE_REFERENCE_QA_ONLY' or qa['audited_n']!=640:
        raise RuntimeError('Unexpected or incomplete v0.3 frozen QA')
    if qa['sealed_test_accessed'] or qa['source_masks_modified'] or qa['verified_negative_voxels']!=0:
        raise RuntimeError('Safety provenance mismatch')
    if sha(V7/'case_class_training_proposal_v7.csv')!=V7_PROPOSAL_SHA:raise RuntimeError('V7 case-class screen was changed')
    status=readjson(V21/'train_status_v9.json')
    if status['status']!='RESEARCH_BASELINE_FINISHED' or status['completed_epochs']!=6 or status['plan_id']!=V21_PLAN:
        raise RuntimeError('V21 not frozen/completed')
    csha=sha(CHECKPOINT)
    if csha!=CHECKPOINT_SHA:raise RuntimeError('V21 checkpoint has changed; safe stop')
    # Checkpoint originates solely from the user's own verified prior V21 run.
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if ck['plan_id']!=V21_PLAN or ck['selection_hash']!=V21_SELECTION or ck['completed_epochs']!=6:
        raise RuntimeError('V21 checkpoint internal provenance mismatch')
    if ck.get('unverified_weak_labels') is not True or ck.get('NOT_ANATOMICAL_GROUND_TRUTH') is not True:
        raise RuntimeError('Unverified-supervision disclaimer missing in checkpoint')
    model=network();model.load_state_dict(ck['model'],strict=True);model.eval()
    assert sum(p.numel() for p in model.parameters())==199384
    with (V7/'case_class_training_proposal_v7.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    screen={}
    for r in rows:
        if r['split']=='train' and r['class_name'] in PRIMARY and r['case_id'] in dev:
            screen.setdefault(str(int(r['case_id'])),{})[r['class_name']]=r
    if set(screen)!=set(dev) or any(set(v)!=set(PRIMARY) for v in screen.values()):
        raise RuntimeError('V7 screening plan missing DEV80 rows for four heads')
    allow={cid:{n:(screen[cid][n]['automated_screen_status']=='SCREENED_CANDIDATE_NOT_APPROVED' and screen[cid][n]['annotated_positive']=='True') for n in PRIMARY} for cid in dev}
    manifest={'version':VERSION,'model_checkpoint_sha256':csha,'checkpoint_steps':ck['steps'],
              'v21_plan_id':V21_PLAN,'v21_selection_hash':V21_SELECTION,'dev_ids_sha256':dev_sha,
              'v7_proposal_sha256':V7_PROPOSAL_SHA,'n_cases':80,'patch':list(PATCH),'stride':list(STRIDE),
              'blend':'0.18+0.82*sin(pi*(x+0.5)/n)^2 separable','threshold':THRESHOLD,
              'per_head_sigmoid_no_argmax_for_raw_masks':True,'native_full_field_of_view':True,
              'model_input_HU':'clip((HU-300)/400,-1.5,1.5)','no_reference_crop':True,
              'source_data_read_only':True,'sealed_test_accessed':False,'reference_mask_used_only_for_metrics':True,
              'annotation_derived_centerlines_not_independent':True,'unknown_unannotated_not_verified_negative':True,
              'zero_model_training':True,'full_coronary_ground_truth':False,
              'comparison_scope':'V21 DEV80 only; V9 original 560-trained model is not independent of DEV80'}
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/'manifest_v22.json'
    if path.exists():
        if readjson(path)!=manifest:raise RuntimeError('V22 output manifest differs, do not mix results')
    else:savejson(path,manifest)
    return model,dev,allow,manifest

def inference(model,ct,batch_size=4):
    import numpy as np, torch
    if not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime to perform full FOV inference')
    shape=tuple(map(int,ct.shape))
    xs,ys,zs=(starts(shape[i],PATCH[i],STRIDE[i]) for i in range(3))
    total=len(xs)*len(ys)*len(zs)
    img=np.clip((ct-300.)/400.,-1.5,1.5).astype('float16')
    acc=np.zeros((4,*shape),dtype='float32')
    den=np.zeros(shape,dtype='float32')
    w=weights()
    # Use generator instead of materializing all tile coordinates.
    coords=((x,y,z) for z in zs for y in ys for x in xs)
    from itertools import islice
    model.to('cuda').eval()
    patches=0
    with torch.inference_mode():
        while True:
            loc=list(islice(coords,batch_size))
            if not loc:break
            x=np.stack([img[a:a+PATCH[0],b:b+PATCH[1],c:c+PATCH[2]] for a,b,c in loc]).astype('float32')
            if x.shape[1:]!=PATCH:raise RuntimeError('Invalid native sliding window')
            pr=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x[:,None])).to('cuda'))).cpu().numpy()
            for (a,b,c),q in zip(loc,pr):
                region=(slice(a,a+PATCH[0]),slice(b,b+PATCH[1]),slice(c,c+PATCH[2]))
                acc[(slice(None),*region)]+=q*w[None,:,:,:]
                den[region]+=w
            patches+=len(loc)
            if patches%240<batch_size:print(f'   tiles {patches}/{total}',flush=True)
    if not np.isfinite(acc).all() or den.min()<=0:raise RuntimeError('Coverage or finite inference assertion failed')
    del img
    return acc,den,{'tiles':total,'tile_grid_xyz':[len(xs),len(ys),len(zs)],'native_shape_xyz':list(shape),'no_reference_crop':True}

def binarize(acc,den):
    import numpy as np
    bits=np.zeros(den.shape,dtype='uint8')
    for j in range(4):
        # No argmax: four independent heads, preserved as a compact 4-bit volume.
        bits |= ((acc[j]/den >= THRESHOLD).astype('uint8')<<j)
    return bits

def components(bitmask,voxel_mm3):
    import numpy as np
    from scipy.ndimage import label,generate_binary_structure
    # Compute on a tight predicted ROI, preventing unbounded connected-component RAM where possible.
    nvox=int(np.count_nonzero(bitmask))
    if not nvox:return {'n_6connected_all':0,'n_6connected_ge_10mm3':0,'largest_component_fraction':0.,'largest_component_mm3':0.}
    # Projection-based tight bounds avoid building large int64 coordinate lists.
    present=(np.any(bitmask,axis=(1,2)),np.any(bitmask,axis=(0,2)),np.any(bitmask,axis=(0,1)))
    sl=tuple(slice(int(np.flatnonzero(v)[0]),int(np.flatnonzero(v)[-1])+1) for v in present)
    roi=bitmask[sl]
    lab,n=label(roi,structure=generate_binary_structure(3,1))
    cc=np.bincount(lab.ravel());cc[0]=0
    big=cc*voxel_mm3>=MAJOR_CC_MM3
    result={'n_6connected_all':int(n),'n_6connected_ge_10mm3':int(big.sum()),
            'largest_component_fraction':float(cc.max()/nvox),
            'largest_component_mm3':float(cc.max()*voxel_mm3)}
    del lab,roi,cc
    return result

def metrics(case_id,seg,ct,bits,allow,spacing,tiles):
    import numpy as np
    voxel=float(np.prod(spacing));summary={};flags=[]
    any_pred=(bits>0);raw_pred_n=int(any_pred.sum());multihit=np.zeros(seg.shape,dtype='bool')
    for maskbit in (3,5,6,7,9,10,11,12,13,14,15):multihit|=bits==maskbit
    multilabel_n=int(multihit.sum())
    for j,(name,lid) in enumerate(PRIMARY.items()):
        pr=(bits&(1<<j))!=0
        pn=int(pr.sum());annot=(seg==lid)
        ref_n=int(annot.sum());eligible=bool(allow[name]);seen=int(np.count_nonzero(pr&annot))
        others=np.isin(seg,[lv for n,lv in PRIMARY.items() if n!=name and allow[n]])
        other_count=int(others.sum());other_hits=int(np.count_nonzero(pr&others))
        unknown=int(np.count_nonzero(pr&(seg==0)))
        lowhu=int(np.count_nonzero(pr&(ct<-300)))
        cc=components(pr,voxel)
        rec={'reference_positive_eligible':eligible,'reference_positive_voxels':ref_n if eligible else None,
             'annotated_positive_hits':seen if eligible else None,
             'conditional_positive_recall':seen/ref_n if eligible and ref_n else None,
             'other_known_major_annotated_voxels':other_count if eligible else None,
             'pred_hits_on_other_major':other_hits if eligible else None,
             'conditional_other_major_overlap':other_hits/other_count if eligible and other_count else None,
             'raw_prediction_voxels':pn,'raw_prediction_mm3':pn*voxel,
             'prediction_in_unannotated_background_unknown_voxels':unknown,
             'fraction_prediction_unannotated_unknown_not_fp':unknown/pn if pn else None,
             'predicted_below_minus300_HU_voxels':lowhu,
             'fraction_predicted_below_minus300_HU_descriptive':lowhu/pn if pn else None,
             'components':cc}
        summary[name]=rec
        if pn*voxel>40000:flags.append(f'{name}:very_large_predicted_volume_unverified')
        if pn and lowhu/pn>.15:flags.append(f'{name}:substantial_lowHU_prediction_artifact_screen')
        if cc['n_6connected_ge_10mm3']>=2:flags.append(f'{name}:multiple_large_prediction_components')
        if eligible and ref_n and seen/ref_n<.8:flags.append(f'{name}:low_known_positive_recall')
    return {'case_id':str(case_id),'version':VERSION,'model_checkpoint_sha256':CHECKPOINT_SHA,
            'native_volume_shape':list(map(int,seg.shape)),'spacing_mm_xyz':list(map(float,spacing)),
            'inference':tiles,'per_artery':summary,'raw_any_prediction_voxels':raw_pred_n,
            'raw_multilabel_conflict_voxels':multilabel_n,'raw_multilabel_conflict_fraction':multilabel_n/raw_pred_n if raw_pred_n else 0.,
            'qc_flags_hypothesis_only':flags,
            'limitations':['Full-volume unannotated tissue is not verified negative','Reference masks may be incomplete',
                           'No expert adjudication, clinical accuracy, or ostial identity asserted',
                           'Only eligible V7 reference classes are scored for recall/confusion']}

def case(case_id,model,allow,manifest,batch_size=4,force=False):
    import numpy as np,nibabel as nib
    rec=OUT/'per_case'/f'{case_id}.json'
    output=OUT/'prediction_masks'/f'{case_id}.v22_4head_native_raw.nii.gz'
    if not force and rec.is_file() and output.is_file():
        old=readjson(rec)
        if old.get('model_checkpoint_sha256')==CHECKPOINT_SHA and old.get('prediction_mask_sha256')==sha(output):
            return old,'REUSED'
    ct_path=CT_DIR/f'{case_id}.img.nii.gz'
    seg_path=DATA/'segmentations'/f'{case_id}.coronary.nii.gz'
    if not ct_path.is_file() or not seg_path.is_file():raise FileNotFoundError(f'CT/seg missing: {case_id}')
    ni=nib.load(ct_path);ref=nib.load(seg_path)
    if ni.shape!=ref.shape or not np.allclose(ni.affine,ref.affine,rtol=0,atol=.01):
        raise RuntimeError(f'Geometry mismatch for {case_id}')
    ct=np.asarray(ni.dataobj,dtype=np.float32);seg=np.asarray(ref.dataobj,dtype=np.uint8)
    if not np.isfinite(ct).all():raise RuntimeError('Nonfinite CT')
    if any(i<.1 or i>2.0 for i in ni.header.get_zooms()[:3]):raise RuntimeError('Abnormal CT spacing')
    print(f'V22 case {case_id}: native volume {ct.shape}, full field of view',flush=True)
    acc,den,tiles=inference(model,ct,batch_size=batch_size)
    bits=binarize(acc,den)
    del acc,den;gc.collect()
    result=metrics(case_id,seg,ct,bits,allow,ni.header.get_zooms()[:3],tiles)
    output.parent.mkdir(parents=True,exist_ok=True)
    nimg=nib.Nifti1Image(bits,ni.affine)
    nimg.header.set_xyzt_units('mm')
    nimg.header.set_zooms(tuple(map(float,ni.header.get_zooms()[:3])))
    nimg.header['descrip']=b'OpenPlaque V22 research 4-bit raw LM LAD LCX RCA; incomplete GT'
    temporary=output.with_name(output.stem+'.tmp.nii.gz')
    nib.save(nimg,str(temporary));os.replace(temporary,output)
    result['prediction_mask_sha256']=sha(output)
    result['prediction_mask_relative_path']=str(output.relative_to(OUT))
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    savejson(rec,result)
    del ct,seg,bits,ni,ref,nimg
    gc.collect()
    return result,'NEW'

def summarize(dev,manifest):
    records=[];missing=[]
    for cid in dev:
        p=OUT/'per_case'/f'{cid}.json';m=OUT/'prediction_masks'/f'{cid}.v22_4head_native_raw.nii.gz'
        if not p.is_file() or not m.is_file():missing.append(cid);continue
        try:
            r=readjson(p)
            if r['model_checkpoint_sha256']!=CHECKPOINT_SHA or sha(m)!=r['prediction_mask_sha256']:
                missing.append(cid);continue
            records.append(r)
        except Exception:missing.append(cid)
    agg={name:{'cases_eligible':0,'positive_voxels':0,'positive_hits':0,'other_annotated_voxels':0,
               'cross_major_hits':0,'prediction_voxels':0,'unannotated_unknown_voxels':0,'below_minus300_hu_prediction_voxels':0,
               'cases_multiple_major_components':0} for name in PRIMARY}
    queue=[];qc_flags=Counter()
    for r in records:
        flags=list(r['qc_flags_hypothesis_only'])
        if r['raw_multilabel_conflict_fraction']>.025:flags.append('many_multihead_overlaps')
        for name,z in r['per_artery'].items():
            s=agg[name]
            s['prediction_voxels']+=z['raw_prediction_voxels']
            s['unannotated_unknown_voxels']+=z['prediction_in_unannotated_background_unknown_voxels']
            s['below_minus300_hu_prediction_voxels']+=z['predicted_below_minus300_HU_voxels']
            s['cases_multiple_major_components']+=int(z['components']['n_6connected_ge_10mm3']>=2)
            if z['reference_positive_eligible']:
                s['cases_eligible']+=1;s['positive_voxels']+=z['reference_positive_voxels']
                s['positive_hits']+=z['annotated_positive_hits']
                s['other_annotated_voxels']+=z['other_known_major_annotated_voxels']
                s['cross_major_hits']+=z['pred_hits_on_other_major']
        for flag in set(flags):qc_flags[flag]+=1
        queue.append({'case_id':r['case_id'],'number_qc_flags':len(set(flags)),
                      'multihead_conflict_fraction':r['raw_multilabel_conflict_fraction'],
                      'flags':'|'.join(sorted(set(flags)))})
    queue.sort(key=lambda v:(-v['number_qc_flags'],int(v['case_id'])))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'automated_attention_queue_v22.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','number_qc_flags','multihead_conflict_fraction','flags']);w.writeheader();w.writerows(queue)
    for s in agg.values():
        s['conditional_whole_FOV_positive_recall']=s['positive_hits']/s['positive_voxels'] if s['positive_voxels'] else None
        s['conditional_cross_major_overlap']=s['cross_major_hits']/s['other_annotated_voxels'] if s['other_annotated_voxels'] else None
        s['fraction_predicted_unannotated_unknown']=s['unannotated_unknown_voxels']/s['prediction_voxels'] if s['prediction_voxels'] else None
        s['fraction_predicted_lowHU_descriptive']=s['below_minus300_hu_prediction_voxels']/s['prediction_voxels'] if s['prediction_voxels'] else None
    summary={'version':VERSION,'complete':len(records)==80,'cases_complete':len(records),'cases_expected':80,
             'missing_ids':missing,'frozen_dev_sha256':QA_DEV_SHA,'model_checkpoint_sha256':CHECKPOINT_SHA,
             'model_used_only_for_inference':True,'per_artery_reference_conditioned_metrics':agg,
             'flag_counts_descriptive':dict(qc_flags),'no_test_reads':True,'no_reference_edits':True,
             'no_verified_negatives':True,'whole_FOV_metrics_not_unbiased_specificity':True}
    savejson(OUT/'whole_volume_summary_v22.json',summary)
    def fmt(x):return 'N/A' if x is None else f'{x:.4f}'
    table=''.join('<tr><th>'+html.escape(name)+'</th><td>'+str(z['cases_eligible'])+'</td><td>'+fmt(z['conditional_whole_FOV_positive_recall'])+'</td><td>'+fmt(z['conditional_cross_major_overlap'])+'</td><td>'+fmt(z['fraction_predicted_unannotated_unknown'])+'</td><td>'+fmt(z['fraction_predicted_lowHU_descriptive'])+'</td></tr>' for name,z in agg.items())
    page='''<!doctype html><html><meta charset="utf-8"><title>OpenPlaque V22 whole FOV research QA</title>
    <style>body{font:15px/1.5 system-ui;max-width:1100px;padding:24px;margin:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;text-align:left}aside{background:#fff1de;padding:14px}pre{white-space:pre-wrap}</style>
    <h1>OpenPlaque V22 — frozen V21 full FOV evaluation</h1><aside>Research-only reference-conditioned metrics. Unannotated tissue is UNKNOWN, not verified negative. No independent expert or clinical validation. Sealed test unused.</aside>'''
    page+=f'<p>Completed <strong>{len(records)}/80</strong> frozen development CT scans; no training.</p><table><tr><th>Artery</th><th>Reference eligible cases</th><th>Positive recall</th><th>Other-known-major overlap</th><th>Predicted in unannotated tissue (unknown)</th><th>Predicted below -300 HU</th></tr>{table}</table>'
    page+='<h2>Hypothesis-only flags</h2><pre>'+html.escape(json.dumps(dict(qc_flags),indent=2))+'</pre><p>See <code>per_case/</code>, <code>prediction_masks/</code> and <code>automated_attention_queue_v22.csv</code>. Do not compare these outputs to V9 as an independent cohort; V9 previously trained on all 560.</p></html>'
    tmp=OUT/'OPENPLAQUE_V22_WHOLE_VOLUME_REPORT.html.tmp';tmp.write_text(page);os.replace(tmp,OUT/'OPENPLAQUE_V22_WHOLE_VOLUME_REPORT.html')
    print(f'V22 {len(records)}/80 complete, {len(missing)} missing. Output: {OUT}',flush=True)
    return summary

def unit_test():
    import numpy as np, torch
    w=weights();assert w.shape==PATCH and w.min()>0
    assert starts(231,40,30)[-1]==191 and starts(512,64,48)[-1]==448
    coverage=np.zeros((70,72,46),dtype='uint8')
    for x in starts(70,64,48):
        for y in starts(72,64,48):
            for z in starts(46,40,30):coverage[x:x+64,y:y+64,z:z+40]+=1
    assert coverage.min()>0
    a=np.zeros((4,2,2,2),dtype='float32');a[0]=.7;a[1]=.6
    b=binarize(a,np.ones((2,2,2),dtype='float32'));assert np.all(b==3),'sigmoid head overlap must not be argmax suppressed'
    n=network();assert sum(p.numel() for p in n.parameters())==199384
    with torch.no_grad():q=n(torch.zeros((1,1,*PATCH)))
    assert q.shape==(1,4,*PATCH)
    mask=np.zeros((22,22,22),bool);mask[1:5,1:5,1:5]=True;mask[12:16,12:16,12:16]=True
    cc=components(mask,.5);assert cc['n_6connected_ge_10mm3']==2 and cc['n_6connected_all']==2
    print('PASS: architecture, native full FOV coverage, positive blending, raw independent sigmoid heads, topology')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('test','preflight','case','all','report'))
    p.add_argument('--case-id',type=str,default=None)
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.stage=='test':unit_test();return
    model,dev,allow,manifest=preflight()
    if args.stage=='preflight':print('PASS V22 frozen V21 checkpoint/splits/V7 screens; DEV80, zero test access.');return
    if args.stage=='report':summarize(dev,manifest);return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('A Colab T4/L4 GPU is required for complete native full FOV inference')
    if args.batch_size<1 or args.batch_size>8:raise RuntimeError('Batch 1..8 required')
    chosen=([args.case_id] if args.stage=='case' else sorted(dev,key=int))
    if any(cid not in set(dev) for cid in chosen):raise RuntimeError('STOP requested case not in V21 DEV80')
    if args.max_cases:chosen=chosen[:args.max_cases]
    statuses=Counter();errors=[]
    for i,cid in enumerate(chosen,1):
        try:
            result,state=case(cid,model,allow[cid],manifest,args.batch_size,args.force)
            statuses[state]+=1
        except Exception as e:
            errors.append({'case_id':cid,'exception':type(e).__name__,'message':str(e)[:600],'traceback':traceback.format_exc()[-2000:]})
            print('V22 failed case',cid,str(e),flush=True)
            if args.stage=='case':raise
        savejson(OUT/'progress_v22.json',{'attempted':i,'selected_this_run':len(chosen),'new_vs_reused':dict(statuses),'errors':errors,
                                          'completed_prior_cases_reusable':True,'last_id':cid})
        if i%5==0:summarize(dev,manifest)
    summarize(dev,manifest)
    if errors:raise RuntimeError(f'V22 {len(errors)} case errors: inspect progress_v22.json')
    print('V22 completed selected cases; new/reused',dict(statuses))

if __name__=='__main__':main()