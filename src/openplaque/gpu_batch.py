from __future__ import annotations
import json, os, shutil, subprocess, traceback, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


def _run(cmd):
    print('RUN:', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def reconstruct_source(cache_dir, out='/content/source_ccta.nii.gz'):
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
    d = np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]], float)
    img.SetDirection(tuple(d.ravel()))
    sitk.WriteImage(img, out)
    return Path(out)


def setup_nnunet(model_zip):
    os.environ['nnUNet_raw']='/content/nnUNet_raw'
    os.environ['nnUNet_preprocessed']='/content/nnUNet_preprocessed'
    os.environ['nnUNet_results']='/content/nnUNet_results'
    for k in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
    if not any(Path(os.environ['nnUNet_results']).rglob('checkpoint_final.pth')):
        with zipfile.ZipFile(model_zip) as z:
            z.extractall(os.environ['nnUNet_results'])
    ckpts = sorted(Path(os.environ['nnUNet_results']).rglob('checkpoint_final.pth'))
    if len(ckpts) < 5:
        raise RuntimeError(f'Expected 5 folds, found {len(ckpts)}')


def run_plaque(cfg, root):
    from openplaque.study import OpenPlaqueStudy
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    setup_nnunet(cfg['MODEL_ZIP'])
    study = OpenPlaqueStudy(cfg['STUDY_ZIP'], extract_root='/content/full_dicom_gpu_batch')
    work = Path('/content/plaque_batch'); work.mkdir(exist_ok=True)
    cases={}; errors=[]; results={}
    for vessel in cfg['VESSELS']:
        image, volume, _ = study.load_series(cfg['SERIES'][vessel])
        cases[vessel]=(image, volume)
        inp=work/vessel/'input'; inp.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(inp/f'{vessel}_0000.nii.gz'))
        results[vessel]=[]
        for fold in cfg['FOLDS']:
            dst=root/vessel/f'fold_{fold}'/f'{vessel}.nii.gz'; dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if cfg['REUSE_PLAQUE_PREDICTIONS'] and dst.exists():
                    print('reuse', vessel, fold)
                else:
                    local=work/vessel/f'fold_{fold}'
                    if local.exists(): shutil.rmtree(local)
                    local.mkdir(parents=True)
                    _run(['nnUNetv2_predict','-i',inp,'-o',local,'-d','Dataset001_CCTA_DHM','-c','3d_fullres','-f',fold])
                    shutil.copy2(local/f'{vessel}.nii.gz', dst)
                results[vessel].append((fold,dst))
            except Exception as e:
                errors.append({'vessel':vessel,'fold':fold,'error':str(e)})
                print(traceback.format_exc())
                if not cfg['KEEP_GOING_ON_ERROR']: raise
    rows=[]; meta={}
    for vessel in cfg['VESSELS']:
        avail=[(f,p) for f,p in results[vessel] if p.exists()]
        if len(avail)<3:
            meta[vessel]={'folds':[f for f,_ in avail],'consensus':False}; continue
        imgs=[sitk.ReadImage(str(p)) for _,p in avail]
        a=np.stack([sitk.GetArrayFromImage(x).astype(np.uint8) for x in imgs])
        counts=np.stack([(a==lab).sum(0) for lab in (0,1,2)])
        cons=np.argmax(counts,0).astype(np.uint8); dis=1-counts.max(0).astype(np.float32)/len(avail)
        ci=sitk.GetImageFromArray(cons); ci.CopyInformation(imgs[0])
        di=sitk.GetImageFromArray(dis); di.CopyInformation(imgs[0])
        sitk.WriteImage(ci,str(root/vessel/f'{vessel}_5fold_consensus.nii.gz'))
        sitk.WriteImage(di,str(root/vessel/f'{vessel}_5fold_disagreement.nii.gz'))
        vv=float(np.prod(imgs[0].GetSpacing()))
        for (f,_),x in zip(avail,a): rows.append({'vessel':vessel,'fold':f,'plaque_volume_mm3':float((x==2).sum()*vv)})
        rows.append({'vessel':vessel,'fold':'consensus','plaque_volume_mm3':float((cons==2).sum()*vv),'mean_disagreement':float(dis.mean()),'n_folds':len(avail)})
        meta[vessel]={'folds':[f for f,_ in avail],'consensus':True}
    df=pd.DataFrame(rows); df.to_csv(root/'plaque_5fold_summary.csv',index=False)
    good=[v for v in cfg['VESSELS'] if (root/v/f'{v}_5fold_consensus.nii.gz').exists()]
    if good:
        fig,ax=plt.subplots(len(good),3,figsize=(13,5*len(good))); ax=np.atleast_2d(ax)
        for r,v in enumerate(good):
            vol=cases[v][1]; cons=sitk.GetArrayFromImage(sitk.ReadImage(str(root/v/f'{v}_5fold_consensus.nii.gz'))); dis=sitk.GetArrayFromImage(sitk.ReadImage(str(root/v/f'{v}_5fold_disagreement.nii.gz'))); z=int(np.argmax((cons==2).sum((1,2))))
            ax[r,0].imshow(vol[z],cmap='gray',vmin=-200,vmax=800); ax[r,1].imshow(vol[z],cmap='gray',vmin=-200,vmax=800); ax[r,1].imshow(cons[z]==2,alpha=.55); ax[r,2].imshow(dis[z],vmin=0,vmax=.8)
            for q in ax[r]: q.axis('off')
        fig.tight_layout(); fig.savefig(root/'plaque_ensemble_qc.png',dpi=160); plt.close(fig)
    state='COMPLETE' if not errors and all(len(results[v])==5 for v in cfg['VESSELS']) else 'PARTIAL'
    summary={'status':state,'model':'Dataset001_CCTA_DHM','vessels':cfg['VESSELS'],'consensus':meta,'errors':errors}; _json(root/'summary.json',summary)
    html=root/'OPENPLAQUE_GPU_PLAQUE_5FOLD_ENSEMBLE_REPORT.html'; html.write_text('<html><body><h1>Plaque 5-fold ensemble</h1>'+df.to_html(index=False)+'</body></html>')
    zf=root/'OPENPLAQUE_GPU_PLAQUE_5FOLD_ENSEMBLE_REPORT_BACK.zip'
    with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [html,root/'summary.json',root/'plaque_5fold_summary.csv',root/'plaque_ensemble_qc.png']:
            if p.exists(): z.write(p,p.name)
    return summary


def _totalseg(source, out, task, reuse=True, probabilities=False, license_number='', preview=False):
    out=Path(out); done=out/'_SUCCESS'
    if reuse and done.exists(): return out
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd=['TotalSegmentator','-i',source,'-o',out,'-ta',task,'--device','gpu']
    if preview: cmd += ['--preview']
    if license_number: cmd += ['-l',license_number]
    if probabilities: cmd += ['--save_probabilities',out/f'{out.name}_probabilities.npz']
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        if probabilities:
            shutil.rmtree(out); out.mkdir(parents=True)
            _run(['TotalSegmentator','-i',source,'-o',out,'-ta',task,'--device','gpu'])
        else: raise
    done.write_text('done'); return out


def run_coronary(cfg, source, root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); errors=[]
    license_number=cfg.get('TOTALSEG_LICENSE','').strip()
    if not license_number:
        summary={'status':'SKIPPED_LICENSE','required_tasks':['coronary_arteries','coronary_arteries_LEGACY'],'errors':[]}
        _json(root/'summary.json',summary); return summary
    cur=leg=None
    try: cur=_totalseg(source,root/'current','coronary_arteries',cfg['REUSE_CORONARY_CURRENT'],cfg['SAVE_CORONARY_PROBABILITIES'],license_number=license_number)
    except Exception as e: errors.append({'task':'current','error':str(e)}); print(traceback.format_exc())
    try: leg=_totalseg(source,root/'legacy','coronary_arteries_LEGACY',cfg['REUSE_CORONARY_LEGACY'],cfg['SAVE_CORONARY_PROBABILITIES'],license_number=license_number)
    except Exception as e: errors.append({'task':'legacy','error':str(e)}); print(traceback.format_exc())
    rows=[]; dice=None
    def pick(folder):
        if folder is None: return None
        p=Path(folder)/'coronary_arteries.nii.gz'
        if p.exists(): return p
        f=list(Path(folder).rglob('*coronary*.nii.gz')); return f[0] if len(f)==1 else None
    cp,lp=pick(cur),pick(leg)
    if cp: ci=sitk.ReadImage(str(cp)); ca=sitk.GetArrayFromImage(ci)>0; vv=float(np.prod(ci.GetSpacing())); rows.append({'mask':'current','volume_mm3':float(ca.sum()*vv)})
    if lp: li=sitk.ReadImage(str(lp)); la=sitk.GetArrayFromImage(li)>0; vv=float(np.prod(li.GetSpacing())); rows.append({'mask':'legacy','volume_mm3':float(la.sum()*vv)})
    if cp and lp:
        inter=ca&la; union=ca|la; dis=ca^la; dice=float(2*inter.sum()/max(ca.sum()+la.sum(),1))
        for name,x in [('intersection',inter),('union',union),('disagreement',dis)]:
            im=sitk.GetImageFromArray(x.astype(np.uint8)); im.CopyInformation(ci); sitk.WriteImage(im,str(root/f'coronary_{name}.nii.gz')); rows.append({'mask':name,'volume_mm3':float(x.sum()*vv)})
    df=pd.DataFrame(rows); df.to_csv(root/'coronary_ensemble_summary.csv',index=False)
    state='COMPLETE' if cp and lp and not errors else ('PARTIAL' if cp or lp else 'ERROR')
    summary={'status':state,'dice_current_vs_legacy':dice,'errors':errors}; _json(root/'summary.json',summary)
    html=root/'OPENPLAQUE_GPU_CORONARY_ARTERY_ENSEMBLE_REPORT.html'; html.write_text('<html><body><h1>Coronary artery ensemble</h1>'+df.to_html(index=False)+'</body></html>')
    zf=root/'OPENPLAQUE_GPU_CORONARY_ARTERY_ENSEMBLE_REPORT_BACK.zip'
    with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [html,root/'summary.json',root/'coronary_ensemble_summary.csv',root/'coronary_intersection.nii.gz',root/'coronary_union.nii.gz',root/'coronary_disagreement.nii.gz']:
            if p.exists(): z.write(p,p.name)
    return summary


def run_heart(cfg, source, root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); errors=[]; total=sinus=ch=None
    try: total=_totalseg(source,root/'total','total',cfg['REUSE_HEART_TOTAL'],preview=True)
    except Exception as e: errors.append({'task':'total','error':str(e)}); print(traceback.format_exc())
    license_number=cfg.get('TOTALSEG_LICENSE','').strip()
    if license_number:
        try: sinus=_totalseg(source,root/'aortic_sinuses','aortic_sinuses',cfg['REUSE_AORTIC_SINUSES'],license_number=license_number)
        except Exception as e: errors.append({'task':'aortic_sinuses','error':str(e)}); print(traceback.format_exc())
    else:
        print('Skipping licensed aortic_sinuses (no TotalSegmentator license supplied).')
    if cfg['RUN_HIGHRES_CHAMBERS_IF_LICENSED'] and license_number:
        try: ch=_totalseg(source,root/'heartchambers_highres','heartchambers_highres',True,license_number=license_number)
        except Exception as e: errors.append({'task':'heartchambers_highres','error':str(e)}); print(traceback.format_exc())
    rows=[]
    for folder in [x for x in [total,sinus,ch] if x]:
        for p in Path(folder).glob('*.nii.gz'):
            try:
                im=sitk.ReadImage(str(p)); a=sitk.GetArrayFromImage(im)>0
                rows.append({'source':Path(folder).name,'structure':p.name.replace('.nii.gz',''),'volume_ml':float(a.sum()*np.prod(im.GetSpacing())/1000)})
            except Exception: pass
    df=pd.DataFrame(rows); df.to_csv(root/'whole_heart_structure_volumes.csv',index=False)
    selected=root/'selected_context_masks'; selected.mkdir(exist_ok=True)
    wanted={'aorta','heart_myocardium','heart_atrium_left','heart_ventricle_left','heart_atrium_right','heart_ventricle_right','pulmonary_artery','right_coronary_cusp','left_coronary_cusp','non_coronary_cusp','myocardium','atrium_left','ventricle_left','atrium_right','ventricle_right'}
    for folder in [x for x in [total,sinus,ch] if x]:
        for p in Path(folder).glob('*.nii.gz'):
            if p.name.replace('.nii.gz','') in wanted: shutil.copy2(p,selected/f'{Path(folder).name}__{p.name}')
    state='COMPLETE' if total and sinus and not errors else ('PARTIAL' if total or sinus or ch else 'ERROR')
    summary={'status':state,'tasks_run':[n for n,x in [('total',total),('aortic_sinuses',sinus),('heartchambers_highres',ch)] if x],'errors':errors}; _json(root/'summary.json',summary)
    html=root/'OPENPLAQUE_GPU_WHOLE_HEART_CONTEXT_REPORT.html'; html.write_text('<html><body><h1>Whole-heart context</h1>'+df.head(120).to_html(index=False)+'</body></html>')
    zf=root/'OPENPLAQUE_GPU_WHOLE_HEART_CONTEXT_REPORT_BACK.zip'
    with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [html,root/'summary.json',root/'whole_heart_structure_volumes.csv']:
            if p.exists(): z.write(p,p.name)
        for p in selected.glob('*.nii.gz'): z.write(p,'selected_context_masks/'+p.name)
    return summary


def run_all(cfg):
    batch=Path(cfg['BATCH_ROOT']); batch.mkdir(parents=True,exist_ok=True)
    status={'status':'RUNNING','phases':{},'errors':[]}; _json(batch/'batch_status.json',status)
    def phase(name, fn):
        try:
            status['phases'][name]={'state':'RUNNING'}; _json(batch/'batch_status.json',status)
            s=fn(); status['phases'][name]=s; _json(batch/'batch_status.json',status); return s
        except Exception as e:
            status['phases'][name]={'status':'ERROR','error':str(e)}; status['errors'].append({'phase':name,'error':str(e),'traceback':traceback.format_exc()[-5000:]}); _json(batch/'batch_status.json',status)
            print(traceback.format_exc())
            if not cfg['KEEP_GOING_ON_ERROR']: raise
    if cfg['RUN_PLAQUE_ENSEMBLE']: phase('plaque',lambda:run_plaque(cfg,cfg['PLAQUE_ROOT']))
    source=None
    if cfg['RUN_CORONARY_ENSEMBLE'] or cfg['RUN_WHOLE_HEART']:
        try: source=reconstruct_source(cfg['SOURCE_CACHE'])
        except Exception as e:
            status['errors'].append({'phase':'source_reconstruction','error':str(e)}); _json(batch/'batch_status.json',status); print(traceback.format_exc())
            if not cfg['KEEP_GOING_ON_ERROR']: raise
    if cfg['RUN_CORONARY_ENSEMBLE'] and source: phase('coronary',lambda:run_coronary(cfg,source,cfg['CORONARY_ROOT']))
    if cfg['RUN_WHOLE_HEART'] and source: phase('heart',lambda:run_heart(cfg,source,cfg['HEART_ROOT']))
    status['status']='COMPLETE_WITH_ERRORS' if status['errors'] or any(v.get('status') in ('ERROR','PARTIAL') for v in status['phases'].values()) else 'COMPLETE'; _json(batch/'batch_status.json',status)
    rows=[{'phase':k,**v} for k,v in status['phases'].items()]; df=pd.DataFrame(rows); df.to_csv(batch/'batch_phase_summary.csv',index=False)
    html=batch/'OPENPLAQUE_GPU_BATCH_REPORT.html'; html.write_text('<html><body><h1>OpenPlaque unattended GPU batch</h1>'+df.to_html(index=False)+'</body></html>')
    zf=batch/'OPENPLAQUE_GPU_BATCH_REPORT_BACK.zip'
    with zipfile.ZipFile(zf,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [html,batch/'batch_status.json',batch/'batch_phase_summary.csv']:
            if p.exists(): z.write(p,p.name)
        for p in [Path(cfg['PLAQUE_ROOT'])/'OPENPLAQUE_GPU_PLAQUE_5FOLD_ENSEMBLE_REPORT_BACK.zip',Path(cfg['CORONARY_ROOT'])/'OPENPLAQUE_GPU_CORONARY_ARTERY_ENSEMBLE_REPORT_BACK.zip',Path(cfg['HEART_ROOT'])/'OPENPLAQUE_GPU_WHOLE_HEART_CONTEXT_REPORT_BACK.zip']:
            if p.exists(): z.write(p,p.name)
    return status, zf
