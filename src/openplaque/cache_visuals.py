from __future__ import annotations
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from .cache_utils import hash_array, hash_bytes, hash_df, load_json, save_json
from .tracking_report import VESSELS

class CacheVisualMixin:
    def _track_signature(self):
        if self.tracking_qc is None: self.track_coronaries()
        text='|'.join(f"{v}:{int(self.selected[v]['frame'])}:{hash_array(np.asarray(self.selected[v]['path'],np.float32))}" for v in VESSELS)
        return hash_bytes(text.encode())

    def plot_straightened_roadmaps(self):
        if 'roadmaps' in self._done: return self.out/'02_straightened_coronary_roadmaps.png'
        path=self.out/'02_straightened_coronary_roadmaps.png'; csv=self.out/'plaque_along_tracked_path.csv'; meta=self._paths()['roadmaps']; sig=self._track_signature()
        if self.reuse['roadmaps'] and path.exists() and csv.exists() and meta.exists():
            try:
                if load_json(meta)['tracking_signature']==sig:
                    self.along_df=pd.read_csv(csv); self._record('roadmaps','reused',path); self._done.add('roadmaps'); return path
            except Exception: pass
        out=super().plot_straightened_roadmaps(); save_json({'tracking_signature':sig},meta); self._record('roadmaps','recomputed_and_cached',out); self._done.add('roadmaps'); return out

    def _generate_pcat(self,cross,ribbon):
        if self.study is None: self.prepare_inputs()
        base=self.root/'PCAT_RCA_10_50'; cl=pd.read_csv(base/'rca_centerline_smoothed_zyx.csv'); rad=pd.read_csv(base/'pcat_local_radius_profile.csv')
        aorta_path=next((p for p in [self.root/'RCA_Ostium_TotalSegmentator'/'aorta_series7_totalseg.nii.gz',self.root/'TotalSegmentator_Validation_v2'/'aorta_series7_totalseg.nii.gz'] if p.exists()),None)
        long_path=next((p for p in [self.root/'Combined_TPV_PCAT_All_Metrics_v2'/'pcat_canonical_longitudinal_v2.csv',self.root/'PCAT_RCA_10_50_Reproducibility_Lock'/'pcat_canonical_primary_longitudinal.csv'] if p.exists()),None)
        if aorta_path is None or long_path is None: raise FileNotFoundError('Missing frozen RCA PCAT inputs')
        source_img,ct,_=self.study.load_series(7); ct=np.asarray(ct,float); sp=np.asarray(source_img.GetSpacing(),float)[::-1]
        arc=cl.arc_mm.to_numpy(float); pts=cl[['z','y','x']].to_numpy(float); pts_mm=pts*sp; lumen=np.interp(arc,rad.arc_mm.to_numpy(float),rad.lumen_radius_mm.to_numpy(float))
        ai=sitk.ReadImage(str(aorta_path))
        if ai.GetSize()!=source_img.GetSize() or not np.allclose(ai.GetSpacing(),source_img.GetSpacing()): ai=sitk.Resample(ai,source_img,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        aorta=sitk.GetArrayFromImage(ai).astype(float)
        def basis(t):
            t=t/np.linalg.norm(t); ref=np.array([1.,0.,0.]) if abs(t[0])<.85 else np.array([0.,1.,0.]); u=np.cross(t,ref); u/=np.linalg.norm(u); v=np.cross(t,u); v/=np.linalg.norm(v); return u,v
        def plane(target,half=8.,pix=.2):
            i=int(np.argmin(np.abs(arc-target))); i0=max(0,i-2); i1=min(len(arc)-1,i+2); u,v=basis(pts_mm[i1]-pts_mm[i0]); c=np.arange(-half,half+1e-9,pix); U,V=np.meshgrid(c,c,indexing='xy')
            pos=pts_mm[i][None,None,:]+U[...,None]*u+V[...,None]*v; vox=(pos/sp).reshape(-1,3).T; img=map_coordinates(ct,vox,order=1,mode='nearest').reshape(U.shape); am=map_coordinates(aorta,vox,order=0,mode='nearest').reshape(U.shape)>.5
            rr=np.sqrt(U**2+V**2); lum=float(lumen[i]); outer=lum+.75; shell=3*outer; fat=(rr>outer)&(rr<=shell)&(~am)&(img>=-190)&(img<=-30); return float(arc[i]),img,fat,lum,outer,shell,[-half,half,-half,half]
        targets=[10,20,30,40,50]; planes=[plane(t) for t in targets]; fig,axs=plt.subplots(1,5,figsize=(20,4.5)); im=None
        for ax,d,t in zip(axs,planes,targets):
            a,img,fat,lum,outer,shell,extent=d; ax.imshow(img,cmap='gray',vmin=-200,vmax=800,extent=extent,origin='lower'); im=ax.imshow(np.ma.masked_where(~fat,img),cmap='coolwarm',vmin=-120,vmax=-60,alpha=.8,extent=extent,origin='lower')
            for rr,ls in [(lum,'-'),(outer,'--'),(shell,':')]: ax.add_patch(plt.Circle((0,0),rr,fill=False,linestyle=ls,linewidth=1.7))
            ax.plot(0,0,'+',markersize=9); ax.set_title(f'RCA {t} mm\nactual {a:.1f} mm'); ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_aspect('equal')
        axs[0].set_ylabel('mm'); fig.suptitle('RCA artery-centered PCAT: solid=lumen, dashed=outer wall, dotted=shell edge',fontsize=13); fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.78,pad=.02).set_label('PCAT attenuation (HU)'); cross.parent.mkdir(parents=True,exist_ok=True); plt.savefig(cross,dpi=180,bbox_inches='tight'); plt.close(fig)
        d=pd.read_csv(long_path); x=(d.arc_start_mm.to_numpy(float)+d.arc_end_mm.to_numpy(float))/2; y=d.mean_hu.to_numpy(float); fig,ax=plt.subplots(figsize=(12,3.2)); sc=ax.scatter(x,np.zeros_like(x),c=y,cmap='coolwarm',vmin=-110,vmax=-75,s=260,marker='s'); ax.set_xlim(10,50); ax.set_yticks([]); ax.set_xlabel('Distance from RCA ostium (mm)'); ax.set_title('RCA longitudinal PCAT attenuation ribbon'); fig.colorbar(sc,ax=ax,pad=.02).set_label('Mean HU per 1-mm segment'); plt.tight_layout(); plt.savefig(ribbon,dpi=180,bbox_inches='tight'); plt.close(fig)

    def reuse_pcat_outputs(self):
        if 'pcat_figures' in self._done: return {'cross_sections':self.out/'03_rca_pcat_cross_sections.png','ribbon':self.out/'04_rca_longitudinal_pcat_ribbon.png'}
        cache=self._paths()['pcat_figures']; cache.mkdir(parents=True,exist_ok=True); cross=cache/'cross.png'; ribbon=cache/'ribbon.png'
        if self.reuse['pcat_figures'] and cross.exists() and ribbon.exists(): action='reused'
        else: self._generate_pcat(cross,ribbon); action='recomputed_and_cached'
        out_cross=self.out/'03_rca_pcat_cross_sections.png'; out_ribbon=self.out/'04_rca_longitudinal_pcat_ribbon.png'; shutil.copyfile(cross,out_cross); shutil.copyfile(ribbon,out_ribbon); self._record('pcat_figures',action,cache); self._done.add('pcat_figures'); return {'cross_sections':out_cross,'ribbon':out_ribbon}

    def _dashboard_signature(self):
        if self.tracking_qc is None: self.track_coronaries()
        return hash_bytes('|'.join([hash_df(self.tracking_qc),hash_df(self.tpv),hash_df(self.pcat),hash_df(self.ps)]).encode())

    def plot_dashboard(self):
        if 'dashboard' in self._done: return self.out/'05_summary_dashboard.png'
        meta=self._paths()['dashboard']; img=self.out/'05_summary_dashboard.png'; summary=self.out/'tracking_report_summary.csv'; sig=self._dashboard_signature()
        if self.reuse['dashboard'] and meta.exists() and img.exists() and summary.exists():
            try:
                if load_json(meta)['signature']==sig: self._record('dashboard','reused',img); self._done.add('dashboard'); return img
            except Exception: pass
        out=super().plot_dashboard(); save_json({'signature':sig},meta); self._record('dashboard','recomputed_and_cached',out); self._done.add('dashboard'); return out

    def _report_signature(self):
        names=['01_tracking_candidates.png','02_straightened_coronary_roadmaps.png','03_rca_pcat_cross_sections.png','04_rca_longitudinal_pcat_ribbon.png','05_summary_dashboard.png','tracking_qc.csv','cache_qc.csv','plaque_along_tracked_path.csv','tracking_report_summary.csv']; bits=[]
        for n in names:
            p=self.out/n; bits.append(f'{n}:{p.stat().st_size}:{p.stat().st_mtime_ns}' if p.exists() else f'{n}:missing')
        return hash_bytes('|'.join(bits).encode())

    def package_report(self):
        self.reuse_pcat_outputs(); self.plot_dashboard();
        if self.along_df is None: self.plot_straightened_roadmaps()
        meta=self._paths()['report_package']; z=self.out/'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip'; html=self.out/'OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html'; sig=self._report_signature()
        if self.reuse['report_package'] and meta.exists() and z.exists() and html.exists():
            try:
                if load_json(meta)['signature']==sig: self._record('report_package','reused',z); return z
            except Exception: pass
        out=super().package_report(); save_json({'signature':sig},meta); self._record('report_package','recomputed_and_cached',out); return out
