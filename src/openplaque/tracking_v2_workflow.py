from __future__ import annotations

import base64, hashlib, json, math, os, pickle, shutil, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from .artery_detection import detect_artery_series
from .boundary import refine_plaque_mask
from .cpr_tracker_v2 import frame_candidates, path_tube, qc_status, select_global_candidate, straighten
from .segmentation import segment_vessel
from .study import OpenPlaqueStudy

VESSELS=("LAD","RCA","LCX")
FALLBACK_SERIES={"RCA":1035,"LCX":1039,"LAD":1043}
TRACKER_VERSION="openpath-v2.1"

def _save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2),encoding="utf-8")
def _load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def _hash_bytes(b): return hashlib.sha256(b).hexdigest()
def _hash_array(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes()); return h.hexdigest()
def _hash_file(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def _save_mask(mask,reference_image,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); img=sitk.GetImageFromArray(np.asarray(mask,np.uint8)); img.CopyInformation(reference_image); sitk.WriteImage(img,str(path))
def _load_valid_plaque_mask(path,reference_image,expected_mm3=None):
    path=Path(path)
    if not path.exists(): return None
    try:
        img=sitk.ReadImage(str(path))
        if img.GetSize()!=reference_image.GetSize() or not np.allclose(img.GetSpacing(),reference_image.GetSpacing(),atol=1e-5): return None
        arr=sitk.GetArrayFromImage(img); mask=(arr==2) if np.nanmax(arr)>1 else (arr>0)
        if expected_mm3 is not None:
            voxel=float(np.prod(reference_image.GetSpacing())); actual=float(mask.sum()*voxel)
            if abs(actual-float(expected_mm3))>max(.75,1.1*voxel): return None
        return mask
    except Exception: return None
def _ensure_model(root):
    os.environ["nnUNet_raw"]="/content/nnUNet_raw"; os.environ["nnUNet_preprocessed"]="/content/nnUNet_preprocessed"; os.environ["nnUNet_results"]="/content/nnUNet_results"
    for d in (os.environ["nnUNet_raw"],os.environ["nnUNet_preprocessed"],os.environ["nnUNet_results"]): Path(d).mkdir(parents=True,exist_ok=True)
    target=Path("/content/nnUNet_results/Dataset001_CCTA_DHM")
    if target.exists(): return
    z=Path(root)/"models"/"Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip"
    if not z.exists(): raise FileNotFoundError(z)
    with zipfile.ZipFile(z) as f: f.extractall("/content/nnUNet_results")

class TrackingV2Workflow:
    COMPONENTS=("series_selection","plaque_masks","tracking","candidate_figure","roadmaps","pcat_figures","dashboard","report_package")
    def __init__(self,root="/content/drive/MyDrive/OpenPlaque",reuse=None):
        self.root=Path(root); self.out=self.root/"Image_Driven_Coronary_Tracking_v2_Report"; self.out.mkdir(parents=True,exist_ok=True); self.cache_root=self.root/"Cache"/"Image_Driven_Tracking_v2"; self.cache_root.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS};
        if reuse:
            unknown=set(reuse)-set(self.COMPONENTS)
            if unknown: raise ValueError(f"Unknown reuse components: {sorted(unknown)}")
            self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.study=self.series_map=self.tpv=self.pcat=self.ps=self.data=self.cache_qc=self.all_candidates=self.selected=self.tracking_qc=self.along_df=None; self.provenance=[]; self._done=set()
    def _record(self,c,a,p,note=""):
        self.provenance.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":note}); pd.DataFrame(self.provenance).to_csv(self.out/"cache_provenance.csv",index=False)
    def _paths(self):
        return {"series_selection":self.cache_root/"series_selection.json","plaque_masks":self.cache_root/"plaque","tracking":self.cache_root/"tracking.pkl","candidate_figure":self.cache_root/"candidate_figure.json","roadmaps":self.cache_root/"roadmaps.json","pcat_figures":self.cache_root/"pcat","dashboard":self.cache_root/"dashboard.json","report_package":self.cache_root/"report.json"}
    def _plaque_sources(self,v):
        v2=self._paths()["plaque_masks"]/f"{v}_canonical_refined.nii.gz"; prior=self.root/"Cache"/"Image_Driven_Tracking_Cache_Controlled"/"plaque"/f"{v}_canonical_refined.nii.gz"; legacy=[]
        for d in (self.root/"Segmentations",self.root/"segmentations"): legacy += [d/f"{v}_refined_plaque_segmentation.nii.gz",d/f"{v}_volume_refined_plaque_segmentation.nii.gz"]
        return [v2,prior]+legacy
    def cache_status(self):
        p=self._paths(); e={"series_selection":p["series_selection"].exists(),"plaque_masks":all(any(x.exists() for x in self._plaque_sources(v)) for v in VESSELS),"tracking":p["tracking"].exists(),"candidate_figure":p["candidate_figure"].exists() and (self.out/"01_tracking_candidates_v2.png").exists(),"roadmaps":p["roadmaps"].exists() and (self.out/"02_straightened_coronary_roadmaps_v2.png").exists(),"pcat_figures":(p["pcat_figures"]/"cross.png").exists() and (p["pcat_figures"]/"ribbon.png").exists(),"dashboard":p["dashboard"].exists() and (self.out/"05_summary_dashboard_v2.png").exists(),"report_package":p["report_package"].exists() and (self.out/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_V2_REPORT_BACK.zip").exists()}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_available":e[k],"planned_action":"reuse" if self.reuse[k] and e[k] else "recompute_and_cache","cache_path":str(p[k])} for k in self.COMPONENTS])
    def _load_metrics(self):
        m=self.root/"Combined_TPV_PCAT_All_Metrics_v2"; self.tpv=pd.read_csv(m/"tpv_metrics_by_vessel_v2.csv"); self.pcat=pd.read_csv(m/"pcat_canonical_primary_v2.csv"); self.ps=pd.read_csv(m/"pcat_circular_sensitivity_v2.csv")
    def prepare_inputs(self):
        if "series_selection" in self._done: return self.series_map
        drive=self.root/"Full_DICOM.zip"; local=Path("/content/Full_DICOM.zip")
        if not drive.exists(): raise FileNotFoundError(drive)
        if not local.exists() or local.stat().st_size!=drive.stat().st_size: shutil.copyfile(drive,local)
        self.study=OpenPlaqueStudy(str(local),extract_root="/content/full_dicom_tracking_v2"); cache=self._paths()["series_selection"]; loaded=False
        if self.reuse["series_selection"] and cache.exists():
            try:
                d=_load_json(cache)
                if int(d["dicom_size"])!=int(drive.stat().st_size): raise ValueError("DICOM size changed")
                self.series_map={k:int(v) for k,v in d["series_map"].items()}; [self.study.load_series(self.series_map[v]) for v in VESSELS]; loaded=True; self._record("series_selection","reused",cache)
            except Exception as e: self._record("series_selection","cache_invalid",cache,repr(e))
        if not loaded:
            self.series_map,_=detect_artery_series(self.study,fallback_series=FALLBACK_SERIES,return_candidates=True); self.series_map={k:int(v) for k,v in self.series_map.items()}; _save_json({"dicom_size":int(drive.stat().st_size),"series_map":self.series_map},cache); self._record("series_selection","recomputed_and_cached",cache)
        self._load_metrics(); self._done.add("series_selection"); return self.series_map
    def load_plaque_masks(self):
        if "plaque_masks" in self._done: return self.cache_qc
        if self.study is None: self.prepare_inputs()
        cache=self._paths()["plaque_masks"]; cache.mkdir(parents=True,exist_ok=True); rows=[]; data={}; force=not self.reuse["plaque_masks"]
        if force: _ensure_model(self.root)
        for v in VESSELS:
            image,volume,_=self.study.load_series(self.series_map[v]); expected=float(self.tpv.loc[self.tpv.vessel==v,"canonical_refined_tpv_mm3"].iloc[0]); mask=used=source=None
            if not force:
                for p in self._plaque_sources(v):
                    mask=_load_valid_plaque_mask(p,image,expected)
                    if mask is not None: used=p; source="cache"; break
            if mask is None:
                _ensure_model(self.root); r=segment_vessel(image,volume,v); q=refine_plaque_mask(volume=r.volume,mask=r.mask,spacing=r.mask_image.GetSpacing(),remove_small=True,min_component_voxels=10,trim_lumen_adjacent=True,lumen_distance_voxels=1,erode_core=False,high_hu_threshold=None,low_hu_threshold=None); mask=q.refined_mask==2; source="forced_recompute" if force else "cache_miss_recompute"; used=cache/f"{v}_canonical_refined.nii.gz"; _save_mask(mask,image,used)
            else:
                normalized=cache/f"{v}_canonical_refined.nii.gz"
                if Path(used)!=normalized: _save_mask(mask,image,normalized)
                used=normalized
            data[v]={"image":image,"volume":np.asarray(volume),"plaque":np.asarray(mask,bool)}; rows.append({"vessel":v,"source":source,"path":str(used),"expected_tpv_mm3":expected,"plaque_voxels":int(mask.sum())})
        self.data=data; self.cache_qc=pd.DataFrame(rows); self.cache_qc.to_csv(self.out/"cache_qc.csv",index=False); self._record("plaque_masks","recomputed_and_cached" if force else "reused_or_filled_missing",cache); self._done.add("plaque_masks"); return self.cache_qc
    def _tracking_signature(self): return {"version":TRACKER_VERSION,"series_map":self.series_map,"plaque":{v:_hash_array(self.data[v]["plaque"].astype(np.uint8)) for v in VESSELS}}
    def _compact_candidate(self,d):
        keys=("frame","image_score","selection_score","path_length_mm","endpoint_distance_mm","sinuosity","mean_path_hu","mean_evidence","mean_vesselness","mean_dog","total_turn_deg","threshold_percentile","graph_endpoints","graph_junctions","cycle_rank","plaque_frame_voxels","plaque_tube_voxels","plaque_capture_fraction","path"); return {k:d[k] for k in keys if k in d}
    def _inflate_candidate(self,v,d):
        o=dict(d); o["path"]=np.asarray(o["path"],float); sp=self.data[v]["image"].GetSpacing(); shape=self.data[v]["volume"][int(o["frame"])].shape; o["tube"],o["path_mask"]=path_tube(o["path"],shape,(float(sp[1]),float(sp[0])),radius_mm=5.0); return o
    def track_coronaries(self):
        if "tracking" in self._done: return self.tracking_qc
        if self.data is None: self.load_plaque_masks()
        cache=self._paths()["tracking"]; sig=self._tracking_signature()
        if self.reuse["tracking"] and cache.exists():
            try:
                with cache.open("rb") as f: p=pickle.load(f)
                if p["signature"]!=sig: raise ValueError("tracking dependencies or algorithm version changed")
                self.all_candidates={v:[self._inflate_candidate(v,x) for x in p["all_candidates"][v]] for v in VESSELS}; self.selected={v:(self._inflate_candidate(v,p["selected"][v]) if p["selected"][v] else None) for v in VESSELS}; self.tracking_qc=pd.DataFrame(p["tracking_qc"]); self.tracking_qc.to_csv(self.out/"tracking_qc.csv",index=False); self._record("tracking","reused",cache); self._done.add("tracking"); return self.tracking_qc
            except Exception as e: self._record("tracking","cache_invalid",cache,repr(e))
        allc={}; selected={}; rows=[]
        for v in VESSELS:
            d=self.data[v]; sp=d["image"].GetSpacing(); spyx=(float(sp[1]),float(sp[0])); cand=[]
            for z in range(d["volume"].shape[0]):
                img=np.asarray(d["volume"][z],float)
                for c in frame_candidates(img,spyx,plaque2d=d["plaque"][z]): c["frame"]=int(z); cand.append(c)
            win,_=select_global_candidate(cand); cand=sorted(cand,key=lambda x:x.get("selection_score",-np.inf),reverse=True) if cand else []; allc[v]=cand; selected[v]=win; total=int(d["plaque"].sum()); status,reason=qc_status(win,total)
            if win is None: rows.append({"vessel":v,"selected_frame":np.nan,"path_length_mm":np.nan,"endpoint_distance_mm":np.nan,"sinuosity":np.nan,"mean_path_hu":np.nan,"mean_vesselness":np.nan,"mean_dog":np.nan,"cycle_rank":np.nan,"plaque_voxels_in_selected_frame":0,"plaque_voxels_within_5mm_tube":0,"tube_capture_pct_of_selected_frame":0.0,"total_canonical_plaque_voxels":total,"tracking_status":status,"qc_reason":reason})
            else: rows.append({"vessel":v,"selected_frame":int(win["frame"]),"path_length_mm":win["path_length_mm"],"endpoint_distance_mm":win["endpoint_distance_mm"],"sinuosity":win["sinuosity"],"mean_path_hu":win["mean_path_hu"],"mean_vesselness":win["mean_vesselness"],"mean_dog":win["mean_dog"],"cycle_rank":win["cycle_rank"],"plaque_voxels_in_selected_frame":win["plaque_frame_voxels"],"plaque_voxels_within_5mm_tube":win["plaque_tube_voxels"],"tube_capture_pct_of_selected_frame":100*win["plaque_capture_fraction"],"total_canonical_plaque_voxels":total,"tracking_status":status,"qc_reason":reason})
        self.all_candidates=allc; self.selected=selected; self.tracking_qc=pd.DataFrame(rows); self.tracking_qc.to_csv(self.out/"tracking_qc.csv",index=False); payload={"signature":sig,"all_candidates":{v:[self._compact_candidate(x) for x in allc[v]] for v in VESSELS},"selected":{v:(self._compact_candidate(selected[v]) if selected[v] else None) for v in VESSELS},"tracking_qc":self.tracking_qc.to_dict(orient="records")}; cache.parent.mkdir(parents=True,exist_ok=True)
        with cache.open("wb") as f: pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
        self._record("tracking","recomputed_and_cached",cache); self._done.add("tracking"); return self.tracking_qc
    def _track_hash(self):
        if self.tracking_qc is None: self.track_coronaries()
        parts=[TRACKER_VERSION]
        for v in VESSELS:
            c=self.selected[v]; parts.append(f"{v}:none" if c is None else f"{v}:{c['frame']}:{_hash_array(np.asarray(c['path'],np.float32))}")
        return _hash_bytes("|".join(parts).encode())
    def plot_candidates(self):
        if "candidate_figure" in self._done: return self.out/"01_tracking_candidates_v2.png"
        if self.tracking_qc is None: self.track_coronaries()
        out=self.out/"01_tracking_candidates_v2.png"; meta=self._paths()["candidate_figure"]; sig=self._track_hash()
        if self.reuse["candidate_figure"] and out.exists() and meta.exists():
            try:
                if _load_json(meta)["tracking_signature"]==sig: self._record("candidate_figure","reused",out); self._done.add("candidate_figure"); return out
            except Exception: pass
        fig,axs=plt.subplots(3,3,figsize=(15,14))
        for r,v in enumerate(VESSELS):
            cands=self.all_candidates[v][:3]; vol=self.data[v]["volume"]; plaque3=self.data[v]["plaque"]
            for col in range(3):
                ax=axs[r,col]
                if col>=len(cands): ax.text(.5,.5,"No plausible candidate",ha="center",va="center"); ax.axis("off"); continue
                c=cands[col]; z=c["frame"]; img=np.asarray(vol[z],float); ax.imshow(img,cmap="gray",vmin=-150,vmax=750); p=c["path"]; ax.plot(p[:,1],p[:,0],linewidth=1.8); plaque=plaque3[z]&c["tube"]
                if np.any(plaque): ax.imshow(np.ma.masked_where(~plaque,plaque),cmap="autumn",alpha=.70,interpolation="nearest")
                ax.set_title(f"{v} candidate {col+1}: frame {z}\n{c['path_length_mm']:.1f} mm; sinuosity {c['sinuosity']:.2f}; plaque {c['plaque_tube_voxels']}/{c['plaque_frame_voxels']}"); ax.axis("off")
        fig.suptitle("V2 open-path coronary candidates — long chamber loops rejected",fontsize=15); plt.tight_layout(rect=[0,0,1,.97]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); _save_json({"tracking_signature":sig},meta); self._record("candidate_figure","recomputed_and_cached",out); self._done.add("candidate_figure"); return out
    def plot_roadmaps(self):
        if "roadmaps" in self._done: return self.out/"02_straightened_coronary_roadmaps_v2.png"
        if self.tracking_qc is None: self.track_coronaries()
        out=self.out/"02_straightened_coronary_roadmaps_v2.png"; csv=self.out/"plaque_along_tracked_path_v2.csv"; meta=self._paths()["roadmaps"]; sig=self._track_hash()
        if self.reuse["roadmaps"] and out.exists() and csv.exists() and meta.exists():
            try:
                if _load_json(meta)["tracking_signature"]==sig: self.along_df=pd.read_csv(csv); self._record("roadmaps","reused",out); self._done.add("roadmaps"); return out
            except Exception: pass
        fig,axs=plt.subplots(3,2,figsize=(18,14),gridspec_kw={"width_ratios":[1,1.6]}); along=[]
        for r,v in enumerate(VESSELS):
            c=self.selected[v]
            if c is None:
                for ax in axs[r]: ax.text(.5,.5,f"{v}: no plausible path",ha="center",va="center"); ax.axis("off")
                continue
            d=self.data[v]; z=c["frame"]; img=np.asarray(d["volume"][z],float); plaque=d["plaque"][z]&c["tube"]; p=c["path"]; ax=axs[r,0]; ax.imshow(img,cmap="gray",vmin=-150,vmax=750); ax.plot(p[:,1],p[:,0],linewidth=2)
            if np.any(plaque): ax.imshow(np.ma.masked_where(~plaque,plaque),cmap="autumn",alpha=.70)
            pad=35; y0=max(0,int(np.floor(p[:,0].min()))-pad); y1=min(img.shape[0],int(np.ceil(p[:,0].max()))+pad+1); x0=max(0,int(np.floor(p[:,1].min()))-pad); x1=min(img.shape[1],int(np.ceil(p[:,1].max()))+pad+1); ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); status=self.tracking_qc.loc[self.tracking_qc.vessel==v,"tracking_status"].iloc[0]; ax.set_title(f"{v} selected CPR frame {z} — QC {status}\nopen path + plaque within 5 mm"); ax.axis("off")
            sp=d["image"].GetSpacing(); st=straighten(img,plaque,p,(float(sp[1]),float(sp[0]))); ax2=axs[r,1]
            if st is None: ax2.text(.5,.5,"Straightening failed",ha="center",va="center"); ax2.axis("off"); continue
            s,off,strip,pstrip=st; ax2.imshow(strip,cmap="gray",vmin=-150,vmax=750,aspect="auto",origin="lower",extent=[s[0],s[-1],off[0],off[-1]])
            if np.any(pstrip): ax2.imshow(np.ma.masked_where(~pstrip,pstrip),cmap="autumn",alpha=.70,aspect="auto",origin="lower",extent=[s[0],s[-1],off[0],off[-1]],interpolation="nearest")
            ax2.axhline(0,linewidth=1); ax2.set_xlabel("Distance along tracked CPR path (mm)"); ax2.set_ylabel("Transverse distance (mm)"); ax2.set_title(f"{v} straightened vessel-centered display"); bins=np.arange(0,max(1,math.ceil(s[-1]))+1,1.0); ci=int(np.argmin(np.abs(off)))
            for a,b in zip(bins[:-1],bins[1:]):
                jj=(s>=a)&(s<b); along.append({"vessel":v,"start_mm":a,"end_mm":b,"display_plaque_pixels":int(pstrip[:,jj].sum()) if np.any(jj) else 0,"mean_centerline_hu":float(np.nanmean(strip[ci,jj])) if np.any(jj) else np.nan,"tracking_status":status})
        fig.suptitle("OpenPlaque V2 straightened coronary plaque roadmaps — visualization only",fontsize=16); plt.tight_layout(rect=[0,0,1,.97]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); self.along_df=pd.DataFrame(along); self.along_df.to_csv(csv,index=False); _save_json({"tracking_signature":sig},meta); self._record("roadmaps","recomputed_and_cached",out); self._done.add("roadmaps"); return out
    def _generate_pcat(self,cross,ribbon):
        if self.study is None: self.prepare_inputs()
        base=self.root/"PCAT_RCA_10_50"; cl=pd.read_csv(base/"rca_centerline_smoothed_zyx.csv"); rad=pd.read_csv(base/"pcat_local_radius_profile.csv"); aorta_path=next((p for p in [self.root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz",self.root/"TotalSegmentator_Validation_v2"/"aorta_series7_totalseg.nii.gz"] if p.exists()),None); long_path=next((p for p in [self.root/"Combined_TPV_PCAT_All_Metrics_v2"/"pcat_canonical_longitudinal_v2.csv",self.root/"PCAT_RCA_10_50_Reproducibility_Lock"/"pcat_canonical_primary_longitudinal.csv"] if p.exists()),None)
        if aorta_path is None or long_path is None: raise FileNotFoundError("Missing frozen RCA PCAT inputs")
        source_img,ct,_=self.study.load_series(7); ct=np.asarray(ct,float); sp=np.asarray(source_img.GetSpacing(),float)[::-1]; arc=cl.arc_mm.to_numpy(float); pts=cl[["z","y","x"]].to_numpy(float); pts_mm=pts*sp; lumen=np.interp(arc,rad.arc_mm.to_numpy(float),rad.lumen_radius_mm.to_numpy(float)); ai=sitk.ReadImage(str(aorta_path))
        if ai.GetSize()!=source_img.GetSize() or not np.allclose(ai.GetSpacing(),source_img.GetSpacing()): ai=sitk.Resample(ai,source_img,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        aorta=sitk.GetArrayFromImage(ai).astype(float)
        def basis(t):
            t=t/np.linalg.norm(t); ref=np.array([1.,0.,0.]) if abs(t[0])<.85 else np.array([0.,1.,0.]); u=np.cross(t,ref); u/=np.linalg.norm(u); v=np.cross(t,u); v/=np.linalg.norm(v); return u,v
        def plane(target,half=8.,pix=.2):
            i=int(np.argmin(np.abs(arc-target))); i0=max(0,i-2); i1=min(len(arc)-1,i+2); u,v=basis(pts_mm[i1]-pts_mm[i0]); c=np.arange(-half,half+1e-9,pix); U,V=np.meshgrid(c,c,indexing="xy"); pos=pts_mm[i][None,None,:]+U[...,None]*u+V[...,None]*v; vox=(pos/sp).reshape(-1,3).T; img=map_coordinates(ct,vox,order=1,mode="nearest").reshape(U.shape); am=map_coordinates(aorta,vox,order=0,mode="nearest").reshape(U.shape)>.5; rr=np.sqrt(U**2+V**2); lum=float(lumen[i]); outer=lum+.75; shell=3*outer; fat=(rr>outer)&(rr<=shell)&(~am)&(img>=-190)&(img<=-30); return float(arc[i]),img,fat,lum,outer,shell,[-half,half,-half,half]
        targets=[10,20,30,40,50]; planes=[plane(t) for t in targets]; fig,axs=plt.subplots(1,5,figsize=(20,4.5)); im=None
        for ax,d,t in zip(axs,planes,targets):
            a,img,fat,lum,outer,shell,extent=d; ax.imshow(img,cmap="gray",vmin=-200,vmax=800,extent=extent,origin="lower"); im=ax.imshow(np.ma.masked_where(~fat,img),cmap="coolwarm",vmin=-120,vmax=-60,alpha=.8,extent=extent,origin="lower")
            for rr,ls in [(lum,"-"),(outer,"--"),(shell,":")]: ax.add_patch(plt.Circle((0,0),rr,fill=False,linestyle=ls,linewidth=1.7))
            ax.plot(0,0,"+",markersize=9); ax.set_title(f"RCA {t} mm\nactual {a:.1f} mm"); ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_aspect("equal")
        axs[0].set_ylabel("mm"); fig.suptitle("RCA artery-centered PCAT: solid=lumen, dashed=outer wall, dotted=shell edge",fontsize=13); fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.78,pad=.02).set_label("PCAT attenuation (HU)"); cross.parent.mkdir(parents=True,exist_ok=True); plt.savefig(cross,dpi=180,bbox_inches="tight"); plt.close(fig); d=pd.read_csv(long_path); x=(d.arc_start_mm.to_numpy(float)+d.arc_end_mm.to_numpy(float))/2; y=d.mean_hu.to_numpy(float); fig,ax=plt.subplots(figsize=(12,3.2)); sc=ax.scatter(x,np.zeros_like(x),c=y,cmap="coolwarm",vmin=-110,vmax=-75,s=260,marker="s"); ax.set_xlim(10,50); ax.set_yticks([]); ax.set_xlabel("Distance from RCA ostium (mm)"); ax.set_title("RCA longitudinal PCAT attenuation ribbon"); fig.colorbar(sc,ax=ax,pad=.02).set_label("Mean HU per 1-mm segment"); plt.tight_layout(); plt.savefig(ribbon,dpi=180,bbox_inches="tight"); plt.close(fig)
    def pcat_figures(self):
        if "pcat_figures" in self._done: return {"cross_sections":self.out/"03_rca_pcat_cross_sections_v2.png","ribbon":self.out/"04_rca_longitudinal_pcat_ribbon_v2.png"}
        cache=self._paths()["pcat_figures"]; cache.mkdir(parents=True,exist_ok=True); cross=cache/"cross.png"; ribbon=cache/"ribbon.png"; action=None
        if self.reuse["pcat_figures"] and cross.exists() and ribbon.exists(): action="reused"
        elif self.reuse["pcat_figures"]:
            previous=[self.root/"Image_Driven_Coronary_Tracking_Report",self.root/"Coronary_CPR_Roadmap_Report",self.root/"User_Friendly_Plaque_PCAT_Report_v3",self.root/"User_Friendly_Plaque_PCAT_Report_v2"]
            pc=next((d/n for d in previous for n in ("03_rca_pcat_cross_sections.png","02_rca_pcat_cross_sections.png","02_rca_pcat_cross_sections_v3.png","02_rca_pcat_cross_sections_v2.png") if (d/n).exists()),None); pr=next((d/n for d in previous for n in ("04_rca_longitudinal_pcat_ribbon.png","03_rca_longitudinal_pcat_ribbon.png","03_rca_longitudinal_pcat_ribbon_v3.png","03_rca_longitudinal_pcat_ribbon_v2.png") if (d/n).exists()),None)
            if pc is not None and pr is not None: shutil.copyfile(pc,cross); shutil.copyfile(pr,ribbon); action="imported_validated_cache"
        if action is None: self._generate_pcat(cross,ribbon); action="recomputed_and_cached"
        oc=self.out/"03_rca_pcat_cross_sections_v2.png"; orib=self.out/"04_rca_longitudinal_pcat_ribbon_v2.png"; shutil.copyfile(cross,oc); shutil.copyfile(ribbon,orib); self._record("pcat_figures",action,cache); self._done.add("pcat_figures"); return {"cross_sections":oc,"ribbon":orib}
    def plot_dashboard(self):
        if "dashboard" in self._done: return self.out/"05_summary_dashboard_v2.png"
        if self.tracking_qc is None: self.track_coronaries()
        out=self.out/"05_summary_dashboard_v2.png"; summary_csv=self.out/"tracking_report_summary_v2.csv"; meta=self._paths()["dashboard"]; sig=_hash_bytes((self._track_hash()+"|"+_hash_bytes(self.tpv.to_csv(index=False).encode())+"|"+_hash_bytes(self.pcat.to_csv(index=False).encode())+"|"+_hash_bytes(self.ps.to_csv(index=False).encode())).encode())
        if self.reuse["dashboard"] and out.exists() and summary_csv.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"]==sig: self._record("dashboard","reused",out); self._done.add("dashboard"); return out
            except Exception: pass
        total=self.tpv[self.tpv.vessel=="TOTAL"].iloc[0]; primary=self.pcat.iloc[0]; cmin=float(self.ps.pcat_mean_hu.min()); cmax=float(self.ps.pcat_mean_hu.max()); directional=-94.345186; fmin=min(cmin,directional); fmax=max(cmax,directional); fig=plt.figure(figsize=(14,9)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15]); ax1=fig.add_subplot(gs[0,0]); vv=self.tpv[self.tpv.vessel!="TOTAL"]; ax1.bar(vv.vessel,vv.canonical_refined_tpv_mm3); ax1.set_ylabel("Refined TPV (mm³)"); ax1.set_title("Canonical plaque volume by artery"); ax2=fig.add_subplot(gs[0,1]); ax2.axis("off"); ax2.text(.03,.95,f"Total refined TPV: {total.canonical_refined_tpv_mm3:.0f} mm³\nRaw TPV: {total.raw_tpv_mm3:.0f} mm³\nTPV sensitivity: {total.sensitivity_min_mm3:.0f}–{total.sensitivity_max_mm3:.0f} mm³\n\nRCA 10–50 mm PCAT: {primary.pcat_mean_hu:.2f} HU\nCircular-margin range: {cmin:.2f} to {cmax:.2f} HU\nFull tested geometry: {fmin:.2f} to {fmax:.2f} HU",va="top",fontsize=13); ax3=fig.add_subplot(gs[1,:]); ax3.axis("off"); cols=["vessel","selected_frame","path_length_mm","sinuosity","plaque_voxels_within_5mm_tube","tube_capture_pct_of_selected_frame","tracking_status"]; tab=self.tracking_qc[cols].copy()
        for c in ("path_length_mm","sinuosity","tube_capture_pct_of_selected_frame"): tab[c]=tab[c].map(lambda x:"" if pd.isna(x) else f"{x:.1f}")
        table=ax3.table(cellText=tab.values,colLabels=["Vessel","CPR frame","Length mm","Sinuosity","Plaque px","Capture %","QC"],loc="center",cellLoc="center"); table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1,1.6); ax3.set_title("V2 tracking QC — open path, coronary-scale length, plaque consistency",pad=12); fig.suptitle("OpenPlaque V2 summary: quantitative endpoints + stricter tracking QC",fontsize=16); plt.tight_layout(rect=[0,0,1,.96]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); pd.DataFrame([{"canonical_total_tpv_mm3":float(total.canonical_refined_tpv_mm3),"raw_total_tpv_mm3":float(total.raw_tpv_mm3),"tpv_sensitivity_min_mm3":float(total.sensitivity_min_mm3),"tpv_sensitivity_max_mm3":float(total.sensitivity_max_mm3),"rca_pcat_mean_hu":float(primary.pcat_mean_hu),"pcat_full_geometry_min_hu":fmin,"pcat_full_geometry_max_hu":fmax,"tracker_version":TRACKER_VERSION}]).to_csv(summary_csv,index=False); _save_json({"signature":sig},meta); self._record("dashboard","recomputed_and_cached",out); self._done.add("dashboard"); return out
    def package_report(self):
        if "report_package" in self._done: return self.out/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_V2_REPORT_BACK.zip"
        self.plot_candidates(); self.plot_roadmaps(); self.pcat_figures(); self.plot_dashboard(); html_path=self.out/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_V2_REPORT.html"; zip_path=self.out/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_V2_REPORT_BACK.zip"; meta=self._paths()["report_package"]; names=["01_tracking_candidates_v2.png","02_straightened_coronary_roadmaps_v2.png","03_rca_pcat_cross_sections_v2.png","04_rca_longitudinal_pcat_ribbon_v2.png","05_summary_dashboard_v2.png","tracking_qc.csv","cache_qc.csv","cache_provenance.csv","plaque_along_tracked_path_v2.csv","tracking_report_summary_v2.csv"]; sig=_hash_bytes("|".join(f"{n}:{_hash_file(self.out/n) if (self.out/n).exists() else 'missing'}" for n in names).encode())
        if self.reuse["report_package"] and zip_path.exists() and html_path.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"]==sig: self._record("report_package","reused",zip_path); self._done.add("report_package"); return zip_path
            except Exception: pass
        def tag(n):
            p=self.out/n
            if not p.exists(): return ""
            return f'<h2>{n}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%;height:auto">'
        html="<html><head><meta charset='utf-8'><title>OpenPlaque V2 image-driven coronary tracking</title></head><body><h1>OpenPlaque — V2 image-driven coronary tracking</h1><p><b>Research use only.</b> V2 rejects cyclic and non-coronary-scale paths. Canonical TPV is unchanged. Plaque CPR views and source-volume PCAT are not spatially co-registered.</p>"+"".join(tag(n) for n in names[:5])+"<h2>Tracking QC</h2>"+self.tracking_qc.to_html(index=False)+"<h2>Cache provenance</h2>"+pd.DataFrame(self.provenance).to_html(index=False)+"</body></html>"; html_path.write_text(html,encoding="utf-8"); files=names+[html_path.name]
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
            for n in files:
                p=self.out/n
                if p.exists(): z.write(p,arcname=n)
        _save_json({"signature":sig},meta); self._record("report_package","recomputed_and_cached",zip_path); self._done.add("report_package"); return zip_path
