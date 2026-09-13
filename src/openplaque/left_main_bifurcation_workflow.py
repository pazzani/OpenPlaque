from __future__ import annotations

import base64, hashlib, json, shutil, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from .study import OpenPlaqueStudy
from .left_main_coronary import (
    ALGORITHM_VERSION, arc_mm, resample_path, source_to_ds, ds_to_source,
    physical_lps_from_zyx, build_downsampled_evidence, detect_left_ostia,
    calibrate_from_rca, score_left_main_candidates, select_branch_pair_from_trunk,
    orthogonal_plane, serial_lumen_qc,
)

SOURCE_SERIES = 7


def _jwrite(obj, path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def _jread(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def _sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def _save_path(path_zyx, spacing_zyx, image, out):
    p=resample_path(path_zyx,spacing_zyx,0.6); s=arc_mm(p,spacing_zyx); ph=physical_lps_from_zyx(image,p)
    df=pd.DataFrame({"arc_mm":s,"z":p[:,0],"y":p[:,1],"x":p[:,2],
                     "lps_x_mm":ph[:,0],"lps_y_mm":ph[:,1],"lps_z_mm":ph[:,2]})
    Path(out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); return p

def _load_path(path): return pd.read_csv(path)[["z","y","x"]].to_numpy(float)


class LeftMainBifurcationWorkflow:
    COMPONENTS=("source_evidence","rca_calibration","left_ostium","left_main","bifurcation_branches","qc_figures","report_package")
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root=Path(root); self.out=self.root/"Left_Main_Bifurcation_Report"; self.out.mkdir(parents=True,exist_ok=True)
        self.cache=self.root/"Cache"/"Left_Main_Bifurcation_v1"; self.cache.mkdir(parents=True,exist_ok=True)
        self.reuse={k:True for k in self.COMPONENTS}
        if reuse:
            unknown=set(reuse)-set(self.COMPONENTS)
            if unknown: raise ValueError(f"Unknown reuse flags: {sorted(unknown)}")
            self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.study=self.image=self.ct=self.aorta=self.spacing_zyx=self.evd=None
        self.rca_path=self.rca_qc_df=self.rca_cal=None
        self.ostia_df=None; self.left_main_path=None; self.left_main_qc_df=None; self.left_main_summary=None
        self.branch_pair=None; self.paths={}; self.branch_qc={}; self.provenance=[]; self._done=set()

    def _record(self,c,a,p,note=""):
        self.provenance.append({"component":c,"reuse_requested":self.reuse[c],"action":a,"path":str(p),"note":note})
        pd.DataFrame(self.provenance).to_csv(self.out/"cache_provenance.csv",index=False)

    def _paths(self):
        return {"evidence":self.cache/"source_evidence.npz","evidence_meta":self.cache/"source_evidence_meta.json",
                "rca_cal":self.cache/"rca_calibration.csv","rca_summary":self.cache/"rca_calibration.json",
                "ostia":self.cache/"left_ostium_candidates.csv","left_main":self.cache/"left_main_centerline.csv",
                "left_main_candidates":self.cache/"left_main_candidates.csv","left_main_qc":self.cache/"left_main_qc.csv",
                "left_main_summary":self.cache/"left_main_summary.json","lad":self.cache/"LAD_preliminary_centerline.csv",
                "lcx":self.cache/"LCX_preliminary_centerline.csv","branches_meta":self.cache/"branches_meta.json",
                "lad_qc":self.cache/"LAD_branch_qc.csv","lcx_qc":self.cache/"LCX_branch_qc.csv",
                "branch_candidates":self.cache/"branch_route_candidates.csv","qc_meta":self.cache/"qc_figures.json",
                "report_meta":self.cache/"report.json"}

    def cache_status(self):
        p=self._paths(); fig_names=("01_left_main_source_mips.png","02_left_main_cross_sections.png","03_bifurcation_branches.png","04_branch_cross_sections.png")
        states={"source_evidence":p["evidence"].exists(),"rca_calibration":p["rca_cal"].exists() and p["rca_summary"].exists(),
                "left_ostium":p["ostia"].exists(),"left_main":p["left_main"].exists() and p["left_main_summary"].exists(),
                "bifurcation_branches":p["lad"].exists() and p["lcx"].exists() and p["branches_meta"].exists(),
                "qc_figures":p["qc_meta"].exists() and all((self.out/n).exists() for n in fig_names),
                "report_package":(self.out/"OPENPLAQUE_LEFT_MAIN_BIFURCATION_REPORT_BACK.zip").exists() and p["report_meta"].exists()}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_available":states[k],
                              "planned_action":"reuse" if self.reuse[k] and states[k] else "recompute_and_cache"} for k in self.COMPONENTS])

    def prepare_source(self):
        if self.image is not None: return self.image,self.ct
        dz=self.root/"Full_DICOM.zip"; lz=Path("/content/Full_DICOM.zip")
        if not dz.exists(): raise FileNotFoundError(dz)
        if not lz.exists() or lz.stat().st_size!=dz.stat().st_size: shutil.copyfile(dz,lz)
        self.study=OpenPlaqueStudy(str(lz),extract_root="/content/full_dicom_left_main")
        self.image,self.ct,_=self.study.load_series(SOURCE_SERIES); self.ct=np.asarray(self.ct,np.float32)
        self.spacing_zyx=np.asarray(self.image.GetSpacing(),float)[::-1]
        ap=self.root/"RCA_Ostium_TotalSegmentator"/"aorta_series7_totalseg.nii.gz"
        if not ap.exists(): raise FileNotFoundError(ap)
        ai=sitk.ReadImage(str(ap))
        if ai.GetSize()!=self.image.GetSize() or not np.allclose(ai.GetSpacing(),self.image.GetSpacing()):
            ai=sitk.Resample(ai,self.image,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        self.aorta=sitk.GetArrayFromImage(ai)>0
        rca=self.root/"PCAT_RCA_10_50"/"rca_centerline_smoothed_zyx.csv"
        if not rca.exists(): raise FileNotFoundError(rca)
        self.rca_path=pd.read_csv(rca)[["z","y","x"]].to_numpy(float)
        return self.image,self.ct

    def prepare_evidence(self):
        if "source_evidence" in self._done:return self.evd
        self.prepare_source(); p=self._paths(); loaded=False
        if self.reuse["source_evidence"] and p["evidence"].exists():
            try:
                z=np.load(p["evidence"],allow_pickle=False); self.evd={k:z[k] for k in z.files}; loaded=True
                self._record("source_evidence","reused",p["evidence"])
            except Exception as e:self._record("source_evidence","cache_invalid",p["evidence"],repr(e))
        if not loaded and self.reuse["source_evidence"]:
            prior=self.root/"Cache"/"Source_Volume_Coronary_Centerlines"/"source_evidence.npz"
            if prior.exists():
                try:
                    z=np.load(prior,allow_pickle=False); d={k:z[k] for k in z.files}
                    if "spacing_zyx" not in d or len(d["spacing_zyx"])!=3: raise ValueError("bad prior evidence")
                    np.savez_compressed(p["evidence"],**d); self.evd=d; loaded=True
                    self._record("source_evidence","imported_prior_validated_cache",prior)
                except Exception as e:self._record("source_evidence","prior_cache_invalid",prior,repr(e))
        if not loaded:
            self.evd=build_downsampled_evidence(self.ct,self.aorta,self.spacing_zyx,self.rca_path[0],half_mm=(82,96,96),target_mm=1.0)
            np.savez_compressed(p["evidence"],**{k:(v.astype(np.uint8) if v.dtype==bool else v) for k,v in self.evd.items()})
            self._record("source_evidence","recomputed_and_cached",p["evidence"])
        for k in ("lo_source_zyx","hi_source_zyx"): self.evd[k]=np.asarray(self.evd[k]).astype(int)
        self._done.add("source_evidence"); return self.evd

    def calibrate_rca(self):
        if "rca_calibration" in self._done:return self.rca_cal
        self.prepare_source(); p=self._paths()
        if self.reuse["rca_calibration"] and p["rca_cal"].exists() and p["rca_summary"].exists():
            self.rca_qc_df=pd.read_csv(p["rca_cal"]); self.rca_cal=_jread(p["rca_summary"]); action="reused"
        else:
            self.rca_qc_df,self.rca_cal=calibrate_from_rca(self.rca_path,self.ct,self.spacing_zyx)
            self.rca_qc_df.to_csv(p["rca_cal"],index=False); _jwrite(self.rca_cal,p["rca_summary"]); action="recomputed_and_cached"
        self._record("rca_calibration",action,p["rca_cal"]); self._done.add("rca_calibration"); return self.rca_cal

    def detect_left_ostium(self):
        if "left_ostium" in self._done:return self.ostia_df
        self.prepare_evidence(); p=self._paths()
        if self.reuse["left_ostium"] and p["ostia"].exists(): self.ostia_df=pd.read_csv(p["ostia"]); action="reused"
        else:
            rca_ds=source_to_ds(self.rca_path[0],self.evd["lo_source_zyx"],self.evd["zoom_zyx"])
            self.ostia_df=detect_left_ostia(self.evd,rca_ds,topn=18); self.ostia_df.to_csv(p["ostia"],index=False); action="recomputed_and_cached"
        self._record("left_ostium",action,p["ostia"]); self._done.add("left_ostium"); return self.ostia_df

    def build_left_main(self):
        if "left_main" in self._done:return self.left_main_path
        self.prepare_evidence(); self.calibrate_rca(); self.detect_left_ostium(); p=self._paths()
        if self.reuse["left_main"] and p["left_main"].exists() and p["left_main_summary"].exists() and p["left_main_qc"].exists():
            self.left_main_path=_load_path(p["left_main"]); self.left_main_summary=_jread(p["left_main_summary"]); self.left_main_qc_df=pd.read_csv(p["left_main_qc"]); action="reused"
        else:
            cand,table=score_left_main_candidates(self.evd,self.ostia_df,self.ct,self.spacing_zyx,self.rca_cal,
                self.evd["lo_source_zyx"],self.evd["zoom_zyx"],max_ostia=10)
            table.to_csv(p["left_main_candidates"],index=False)
            best=cand[0]; self.left_main_path=_save_path(best["path_source"],self.spacing_zyx,self.image,p["left_main"])
            self.left_main_qc_df,self.left_main_summary=serial_lumen_qc(self.left_main_path,self.ct,self.spacing_zyx,self.rca_cal,n_samples=12,label_name="LEFT_MAIN")
            self.left_main_qc_df.to_csv(p["left_main_qc"],index=False)
            self.left_main_summary["combined_score"]=float(best["summary"]["combined_score"])
            self.left_main_summary["ostium_rank"]=int(best["summary"]["ostium_rank"])
            pf=self.left_main_summary["plane_pass_fraction"]; ms=self.left_main_summary["median_plane_score"]; L=self.left_main_summary["length_mm"]
            self.left_main_summary["status"]="OK" if (5<=L<=30 and pf>=0.70 and ms>=0.55) else ("REVIEW" if pf>=0.50 and ms>=0.45 else "FAIL")
            _jwrite(self.left_main_summary,p["left_main_summary"]); action="recomputed_and_cached"
        self._record("left_main",action,p["left_main"]); self._done.add("left_main"); return self.left_main_path

    def build_bifurcation_branches(self):
        if "bifurcation_branches" in self._done:return self.paths
        self.build_left_main(); p=self._paths()
        if self.reuse["bifurcation_branches"] and p["lad"].exists() and p["lcx"].exists() and p["branches_meta"].exists():
            self.paths={"LAD":_load_path(p["lad"]),"LCX":_load_path(p["lcx"])}; self.branch_pair=_jread(p["branches_meta"])
            self.branch_qc={"LAD":pd.read_csv(p["lad_qc"]),"LCX":pd.read_csv(p["lcx_qc"])}; action="reused"
        else:
            pair,routes=select_branch_pair_from_trunk(self.evd,self.left_main_path,self.ct,self.spacing_zyx,self.rca_cal,
                self.image,self.evd["lo_source_zyx"],self.evd["zoom_zyx"],max_routes=24)
            route_rows=[]
            for i,r in enumerate(routes):
                q=r["qc_summary"]; route_rows.append({"rank":i+1,"branch_score":r["branch_score"],"graph_score":r["graph_score"],
                    "length_mm":r["length_mm"],"mean_support":r["mean_support"],"p10_support":r["p10_support"],
                    "serial_median_plane_score":q["median_plane_score"],"serial_plane_pass_fraction":q["plane_pass_fraction"],
                    "serial_median_radius_mm":q["median_radius_mm"]})
            pd.DataFrame(route_rows).to_csv(p["branch_candidates"],index=False)
            if pair is None: raise RuntimeError("No LAD/LCX branch pair passed the staged bifurcation constraints")
            self.paths={}
            meta={k:v for k,v in pair.items() if k not in ("LAD","LCX")}; meta["algorithm"]=ALGORITHM_VERSION
            for v in ("LAD","LCX"):
                path=pair[v]["path_source"]; self.paths[v]=_save_path(path,self.spacing_zyx,self.image,p[v.lower()])
                qdf,qsum=serial_lumen_qc(self.paths[v],self.ct,self.spacing_zyx,self.rca_cal,n_samples=12,label_name=v)
                qdf.to_csv(p[v.lower()+"_qc"],index=False); self.branch_qc[v]=qdf
                meta[v+"_qc"]=qsum
                meta[v+"_status"]="OK" if qsum["plane_pass_fraction"]>=0.65 and qsum["median_plane_score"]>=0.55 else ("REVIEW" if qsum["plane_pass_fraction"]>=0.45 else "FAIL")
            self.branch_pair=meta; _jwrite(meta,p["branches_meta"]); action="recomputed_and_cached"
        self._record("bifurcation_branches",action,p["branches_meta"]); self._done.add("bifurcation_branches"); return self.paths

    def summary_table(self):
        self.build_bifurcation_branches(); rows=[{"segment":"RCA calibration",**{k:v for k,v in self.rca_cal.items() if k!="label"},"status":"REFERENCE"},
            {"segment":"Left main",**{k:v for k,v in self.left_main_summary.items() if k not in ("label","status")},"status":self.left_main_summary.get("status","")}]
        for v in ("LAD","LCX"):
            q=self.branch_pair.get(v+"_qc",{}); rows.append({"segment":v+" preliminary",**{k:val for k,val in q.items() if k!="label"},"status":self.branch_pair.get(v+"_status","")})
        df=pd.DataFrame(rows); df.to_csv(self.out/"segment_qc_summary.csv",index=False); return df

    def _plane_for(self,path,frac,half=6.0,pix=0.18):
        p=resample_path(path,self.spacing_zyx,0.45); s=arc_mm(p,self.spacing_zyx); i=int(np.clip(round(frac*(len(p)-1)),0,len(p)-1)); i0=max(0,i-3); i1=min(len(p)-1,i+3)
        t=(p[i1]-p[i0])*self.spacing_zyx; im,c=orthogonal_plane(self.ct,p[i],t,self.spacing_zyx,half,pix); return im,c,float(s[i])

    def _plot_mips(self):
        paths={"RCA":self.rca_path,"LEFT MAIN":self.left_main_path,**self.paths}; P=np.vstack(list(paths.values())); pad=np.ceil(25/self.spacing_zyx).astype(int)
        lo=np.maximum(0,np.floor(P.min(0)).astype(int)-pad); hi=np.minimum(np.asarray(self.ct.shape),np.ceil(P.max(0)).astype(int)+pad+1); roi=self.ct[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        fig,axs=plt.subplots(1,3,figsize=(18,6)); ims=[np.max(roi,0),np.max(roi,1),np.max(roi,2)]; titles=["Axial MIP (y-x)","Coronal MIP (z-x)","Sagittal MIP (z-y)"]
        for ax,im,t in zip(axs,ims,titles): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(t); ax.axis("off")
        for name,path in paths.items():
            q=path-lo; axs[0].plot(q[:,2],q[:,1],linewidth=2,label=name); axs[1].plot(q[:,2],q[:,0],linewidth=2,label=name); axs[2].plot(q[:,1],q[:,0],linewidth=2,label=name)
        for ax in axs: ax.legend(loc="best")
        fig.suptitle("Source CCTA: frozen RCA, staged left main, and preliminary bifurcation branches",fontsize=15); out=self.out/"01_left_main_source_mips.png"
        plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); return out

    def _plot_left_main_cross(self):
        fig,axs=plt.subplots(2,6,figsize=(18,6)); specs=[("RCA reference",self.rca_path), ("LEFT MAIN",self.left_main_path)]
        for r,(name,path) in enumerate(specs):
            for c,frac in enumerate(np.linspace(.08,.92,6)):
                im,x,s=self._plane_for(path,float(frac)); ext=[x[0],x[-1],x[0],x[-1]]; ax=axs[r,c]; ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=ext,origin="lower"); ax.plot(0,0,"+",markersize=9); ax.set_aspect("equal"); ax.set_title(f"{name}\n{s:.1f} mm")
        fig.suptitle("Serial orthogonal lumen QC: RCA calibration versus proposed left main",fontsize=15); out=self.out/"02_left_main_cross_sections.png"
        plt.tight_layout(rect=[0,0,1,.94]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); return out

    def _plot_bifurcation(self):
        fig,axs=plt.subplots(1,3,figsize=(18,6)); paths={"LEFT MAIN":self.left_main_path,**self.paths}; P=np.vstack(list(paths.values())); pad=np.ceil(18/self.spacing_zyx).astype(int); lo=np.maximum(0,np.floor(P.min(0)).astype(int)-pad); hi=np.minimum(np.asarray(self.ct.shape),np.ceil(P.max(0)).astype(int)+pad+1); roi=self.ct[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        ims=[np.max(roi,0),np.max(roi,1),np.max(roi,2)]; titles=["Axial","Coronal","Sagittal"]
        for ax,im,t in zip(axs,ims,titles): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(t); ax.axis("off")
        for name,path in paths.items():
            q=path-lo; axs[0].plot(q[:,2],q[:,1],linewidth=2.2,label=name); axs[1].plot(q[:,2],q[:,0],linewidth=2.2,label=name); axs[2].plot(q[:,1],q[:,0],linewidth=2.2,label=name)
        for ax in axs: ax.legend(loc="best")
        fig.suptitle("Left-main distal end and preliminary LAD/LCX bifurcation hypotheses",fontsize=15); out=self.out/"03_bifurcation_branches.png"
        plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); return out

    def _plot_branch_cross(self):
        fig,axs=plt.subplots(2,6,figsize=(18,6))
        for r,v in enumerate(("LAD","LCX")):
            for c,frac in enumerate(np.linspace(.08,.92,6)):
                im,x,s=self._plane_for(self.paths[v],float(frac)); ext=[x[0],x[-1],x[0],x[-1]]; ax=axs[r,c]; ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=ext,origin="lower"); ax.plot(0,0,"+",markersize=9); ax.set_aspect("equal"); ax.set_title(f"{v} prelim\n{s:.1f} mm")
        fig.suptitle("Serial orthogonal QC of preliminary distal branches — do not accept without visual review",fontsize=15); out=self.out/"04_branch_cross_sections.png"
        plt.tight_layout(rect=[0,0,1,.94]); plt.savefig(out,dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig); return out

    def plot_qc(self):
        if "qc_figures" in self._done:return [self.out/f"0{i}_{n}.png" for i,n in []]
        self.build_bifurcation_branches(); self.summary_table(); p=self._paths(); names=["01_left_main_source_mips.png","02_left_main_cross_sections.png","03_bifurcation_branches.png","04_branch_cross_sections.png"]
        if self.reuse["qc_figures"] and p["qc_meta"].exists() and all((self.out/n).exists() for n in names): action="reused"
        else:
            self._plot_mips(); self._plot_left_main_cross(); self._plot_bifurcation(); self._plot_branch_cross(); _jwrite({"algorithm":ALGORITHM_VERSION},p["qc_meta"]); action="recomputed_and_cached"
        self._record("qc_figures",action,self.out); self._done.add("qc_figures"); return [self.out/n for n in names]

    def package_report(self):
        if "report_package" in self._done:return self.out/"OPENPLAQUE_LEFT_MAIN_BIFURCATION_REPORT_BACK.zip"
        self.plot_qc(); p=self._paths(); zip_path=self.out/"OPENPLAQUE_LEFT_MAIN_BIFURCATION_REPORT_BACK.zip"; html=self.out/"OPENPLAQUE_LEFT_MAIN_BIFURCATION_REPORT.html"
        copy_map={p["rca_cal"]:"rca_calibration.csv",p["ostia"]:"left_ostium_candidates.csv",p["left_main"]:"left_main_centerline.csv",p["left_main_candidates"]:"left_main_candidates.csv",p["left_main_qc"]:"left_main_qc.csv",p["lad"]:"LAD_preliminary_centerline.csv",p["lcx"]:"LCX_preliminary_centerline.csv",p["lad_qc"]:"LAD_branch_qc.csv",p["lcx_qc"]:"LCX_branch_qc.csv",p["branch_candidates"]:"branch_route_candidates.csv"}
        for src,name in copy_map.items():
            if src.exists(): shutil.copyfile(src,self.out/name)
        names=["01_left_main_source_mips.png","02_left_main_cross_sections.png","03_bifurcation_branches.png","04_branch_cross_sections.png","segment_qc_summary.csv","cache_provenance.csv"]+list(copy_map.values())
        def img(n):
            f=self.out/n
            if not f.exists():return ""
            return f'<h2>{n}</h2><img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}">'
        summary=pd.read_csv(self.out/"segment_qc_summary.csv")
        html.write_text("<html><head><meta charset='utf-8'><title>OpenPlaque left main</title></head><body><h1>OpenPlaque — staged left-main and bifurcation experiment</h1><p><b>Research use only.</b> Frozen RCA is the calibration reference. Left main must show a coronary-sized centered lumen in serial orthogonal sections before distal branches are trusted. LAD/LCX are preliminary hypotheses.</p>"+"".join(img(n) for n in names if n.endswith(".png"))+"<h2>Segment QC</h2>"+summary.to_html(index=False)+"</body></html>",encoding="utf-8")
        names.append(html.name)
        sig=hashlib.sha256("|".join(f"{n}:{_sha(self.out/n) if (self.out/n).exists() else 'missing'}" for n in names).encode()).hexdigest()
        if self.reuse["report_package"] and zip_path.exists() and p["report_meta"].exists():
            try:
                if _jread(p["report_meta"])["signature"]==sig: self._record("report_package","reused",zip_path); self._done.add("report_package"); return zip_path
            except Exception: pass
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
            for n in names:
                f=self.out/n
                if f.exists(): z.write(f,arcname=n)
        _jwrite({"signature":sig,"algorithm":ALGORITHM_VERSION},p["report_meta"]); self._record("report_package","recomputed_and_cached",zip_path); self._done.add("report_package"); return zip_path
