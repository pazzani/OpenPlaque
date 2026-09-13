from __future__ import annotations
import pickle, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .artery_detection import detect_artery_series
from .boundary import refine_plaque_mask
from .cache_utils import CACHE_VERSION, hash_array, load_json, save_json
from .cache_visuals import CacheVisualMixin
from .cpr_tracking import path_tube
from .segmentation import segment_vessel
from .study import OpenPlaqueStudy
from .tracking_report import FALLBACK_SERIES, VESSELS, _ensure_model, _load_or_compute_plaque, _save_mask
from .tracking_workflow import TrackingWorkflow

class CacheControlledTrackingWorkflow(CacheVisualMixin, TrackingWorkflow):
    COMPONENTS=('series_selection','plaque_masks','tracking','roadmaps','pcat_figures','dashboard','report_package')
    def __init__(self,root='/content/drive/MyDrive/OpenPlaque',reuse=None):
        super().__init__(root); self.reuse={k:True for k in self.COMPONENTS}; self.reuse.update({k:bool(v) for k,v in (reuse or {}).items()}); self.cache_root=self.root/'Cache'/'Image_Driven_Tracking_Cache_Controlled'; self.cache_root.mkdir(parents=True,exist_ok=True); self.provenance=[]; self._done=set()
    def _record(self,c,a,p,note=''):
        self.provenance.append({'component':c,'reuse_requested':self.reuse[c],'action':a,'path':str(p),'note':note}); pd.DataFrame(self.provenance).to_csv(self.out/'cache_provenance.csv',index=False)
    def _paths(self):
        return {'series_selection':self.cache_root/'series_selection.json','plaque_masks':self.cache_root/'plaque','tracking':self.cache_root/'tracking.pkl','roadmaps':self.cache_root/'roadmaps_meta.json','pcat_figures':self.cache_root/'pcat','dashboard':self.cache_root/'dashboard_meta.json','report_package':self.cache_root/'report_meta.json'}
    def cache_status(self):
        p=self._paths(); e={'series_selection':p['series_selection'].exists(),'plaque_masks':all((p['plaque_masks']/f'{v}_canonical_refined.nii.gz').exists() for v in VESSELS),'tracking':p['tracking'].exists(),'roadmaps':p['roadmaps'].exists() and (self.out/'02_straightened_coronary_roadmaps.png').exists(),'pcat_figures':(p['pcat_figures']/'cross.png').exists() and (p['pcat_figures']/'ribbon.png').exists(),'dashboard':p['dashboard'].exists() and (self.out/'05_summary_dashboard.png').exists(),'report_package':p['report_package'].exists() and (self.out/'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip').exists()}
        return pd.DataFrame([{'component':k,'reuse':self.reuse[k],'cache_exists':e[k],'planned_action':'reuse' if self.reuse[k] and e[k] else 'recompute_and_cache','cache_path':str(p[k])} for k in self.COMPONENTS])
    def prepare_inputs(self):
        if 'series_selection' in self._done:return self.series_map
        drive=self.root/'Full_DICOM.zip'; local=Path('/content/Full_DICOM.zip');
        if not drive.exists():raise FileNotFoundError(drive)
        if not local.exists() or local.stat().st_size!=drive.stat().st_size:shutil.copyfile(drive,local)
        self.study=OpenPlaqueStudy(str(local),extract_root='/content/full_dicom_tracking_cache_controlled'); cache=self._paths()['series_selection']; loaded=False
        if self.reuse['series_selection'] and cache.exists():
            try:
                m=load_json(cache); assert int(m['dicom_size'])==int(drive.stat().st_size); self.series_map={k:int(v) for k,v in m['series_map'].items()}; [self.study.load_series(self.series_map[v]) for v in VESSELS]; loaded=True; self._record('series_selection','reused',cache)
            except Exception as e:self._record('series_selection','cache_invalid',cache,repr(e))
        if not loaded:
            self.series_map,_=detect_artery_series(self.study,fallback_series=FALLBACK_SERIES,return_candidates=True); self.series_map={k:int(v) for k,v in self.series_map.items()}; save_json({'dicom_size':int(drive.stat().st_size),'series_map':self.series_map},cache); self._record('series_selection','recomputed_and_cached',cache)
        m=self.root/'Combined_TPV_PCAT_All_Metrics_v2'; self.tpv=pd.read_csv(m/'tpv_metrics_by_vessel_v2.csv'); self.pcat=pd.read_csv(m/'pcat_canonical_primary_v2.csv'); self.ps=pd.read_csv(m/'pcat_circular_sensitivity_v2.csv'); self._done.add('series_selection'); return self.series_map
    def load_cached_plaque(self):
        if 'plaque_masks' in self._done:return self.cache_qc
        if self.study is None:self.prepare_inputs()
        cache=self._paths()['plaque_masks']; cache.mkdir(parents=True,exist_ok=True)
        if self.reuse['plaque_masks']:
            self.data,self.cache_qc=_load_or_compute_plaque(self.root,self.study,self.series_map,self.tpv); [_save_mask(self.data[v]['plaque'],self.data[v]['image'],cache/f'{v}_canonical_refined.nii.gz') for v in VESSELS]; self._record('plaque_masks','reused_or_filled_missing',cache)
        else:
            _ensure_model(self.root); rows=[]; data={}
            for v in VESSELS:
                image,volume,_=self.study.load_series(self.series_map[v]); r=segment_vessel(image,volume,v); q=refine_plaque_mask(volume=r.volume,mask=r.mask,spacing=r.mask_image.GetSpacing(),remove_small=True,min_component_voxels=10,trim_lumen_adjacent=True,lumen_distance_voxels=1,erode_core=False,high_hu_threshold=None,low_hu_threshold=None); plaque=q.refined_mask==2; path=cache/f'{v}_canonical_refined.nii.gz'; _save_mask(plaque,image,path); data[v]={'image':image,'volume':np.asarray(volume),'plaque':plaque}; rows.append({'vessel':v,'source':'forced_recompute','path':str(path)})
            self.data=data; self.cache_qc=pd.DataFrame(rows); self._record('plaque_masks','recomputed_and_cached',cache)
        self.cache_qc.to_csv(self.out/'cache_qc.csv',index=False); self._done.add('plaque_masks'); return self.cache_qc
    def _compact(self,d):
        keys=('frame','image_score','path_length_mm','mean_path_hu','mean_evidence','intensity_fit','threshold_percentile','plaque_tube_voxels','plaque_frame_voxels','path'); return {k:d[k] for k in keys}
    def _inflate(self,v,d):
        o=dict(d); o['path']=np.asarray(o['path'],float); s=self.data[v]['image'].GetSpacing(); o['tube'],o['path_mask']=path_tube(o['path'],self.data[v]['volume'][int(o['frame'])].shape,(float(s[1]),float(s[0])),radius_mm=5.0); return o
    def track_coronaries(self):
        if 'tracking' in self._done:return self.tracking_qc
        if self.data is None:self.load_cached_plaque()
        cache=self._paths()['tracking']; sig={v:hash_array(np.asarray(self.data[v]['plaque'],np.uint8)) for v in VESSELS}
        if self.reuse['tracking'] and cache.exists():
            try:
                with cache.open('rb') as f:p=pickle.load(f)
                if p['version']!=CACHE_VERSION or p['series_map']!=self.series_map or p['plaque_signature']!=sig:raise ValueError('tracking dependency changed')
                self.all_candidates={v:[self._inflate(v,x) for x in p['all_candidates'][v]] for v in VESSELS}; self.selected={v:self._inflate(v,p['selected'][v]) for v in VESSELS}; self.tracking_qc=pd.DataFrame(p['tracking_qc']); self.tracking_qc.to_csv(self.out/'tracking_qc.csv',index=False); self._record('tracking','reused',cache); self._done.add('tracking'); return self.tracking_qc
            except Exception as e:self._record('tracking','cache_invalid',cache,repr(e))
        q=super().track_coronaries(); p={'version':CACHE_VERSION,'series_map':self.series_map,'plaque_signature':sig,'all_candidates':{v:[self._compact(x) for x in self.all_candidates[v]] for v in VESSELS},'selected':{v:self._compact(self.selected[v]) for v in VESSELS},'tracking_qc':q.to_dict(orient='records')}; cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open('wb') as f:pickle.dump(p,f,pickle.HIGHEST_PROTOCOL)
        self._record('tracking','recomputed_and_cached',cache); self._done.add('tracking'); return q
