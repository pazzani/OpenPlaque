from __future__ import annotations

import base64, gc, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .left_main_coronary import arc_mm, resample_path, serial_lumen_qc, orthogonal_plane, source_to_ds, ds_to_source
from .left_main_memory_safe import detect_left_ostia, trace_routes

ALGORITHM_VERSION = "left-main-only-v1.2-ultra-low-memory"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class LeftMainLowMemoryWorkflow:
    """Left-main-only stage using cached 1-mm evidence; never loads the full source CCTA."""
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=True):
        self.root = Path(root)
        self.out = self.root / "Left_Main_Low_Memory_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "Cache" / "Left_Main_Low_Memory_v12"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.reuse = bool(reuse)
        self.evd = None
        self.rca_source = None
        self.rca_ds = None
        self.rca_cal = None
        self.ostia = None
        self.candidates = None
        self.best = None

    def paths(self):
        return {
            "evidence": self.root / "Cache" / "Source_Volume_Coronary_Centerlines" / "source_evidence.npz",
            "rca": self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv",
            "ostia": self.cache / "left_ostium_candidates.csv",
            "candidates": self.cache / "left_main_candidates_lowres.csv",
            "best_ds": self.cache / "left_main_best_ds.csv",
            "best_source": self.cache / "left_main_best_source_zyx.csv",
            "summary": self.cache / "left_main_lowres_summary.json",
        }

    def load_cached_inputs(self):
        p = self.paths()
        if not p["evidence"].exists(): raise FileNotFoundError(p["evidence"])
        if not p["rca"].exists(): raise FileNotFoundError(p["rca"])
        z = np.load(p["evidence"], allow_pickle=False)
        needed = ("ct","aorta","support","lumen_radius_mm","dist_aorta_mm","cost","lo_source_zyx","zoom_zyx","spacing_zyx")
        self.evd = {k: z[k] for k in needed}
        self.evd["aorta"] = self.evd["aorta"] > 0
        self.evd["lo_source_zyx"] = np.asarray(self.evd["lo_source_zyx"]).astype(int)
        self.rca_source = pd.read_csv(p["rca"])[["z","y","x"]].to_numpy(float)
        self.rca_ds = source_to_ds(self.rca_source, self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
        self.rca_qc, self.rca_cal = serial_lumen_qc(self.rca_ds, self.evd["ct"], self.evd["spacing_zyx"], None, n_samples=12, label_name="RCA_1mm_reference")
        return self.rca_cal

    def find_ostia(self):
        p = self.paths()
        if self.evd is None: self.load_cached_inputs()
        if self.reuse and p["ostia"].exists():
            self.ostia = pd.read_csv(p["ostia"])
        else:
            self.ostia = detect_left_ostia(self.evd, self.rca_ds[0], topn=12, preselect=500)
            self.ostia.to_csv(p["ostia"], index=False)
        return self.ostia

    def find_left_main(self):
        p = self.paths()
        if self.evd is None: self.load_cached_inputs()
        if self.ostia is None: self.find_ostia()
        if self.reuse and all(p[k].exists() for k in ("candidates","best_ds","best_source","summary")):
            self.candidates = pd.read_csv(p["candidates"])
            ds = pd.read_csv(p["best_ds"])[["z","y","x"]].to_numpy(float)
            src = pd.read_csv(p["best_source"])[["z","y","x"]].to_numpy(float)
            self.best = {"path_ds": ds, "path_source": src, "summary": _load_json(p["summary"])}
            return self.best

        rows=[]; kept=[]
        for _,o in self.ostia.head(8).iterrows():
            seed=np.array([o.z,o.y,o.x],float); heading=np.array([o.dir_z,o.dir_y,o.dir_x],float)
            routes=trace_routes(self.evd, seed, heading, min_len=6, max_len=28, max_routes=8, min_forward=2.5, support_percentile=72)
            for r in routes:
                qdf,qsum=serial_lumen_qc(r["path"], self.evd["ct"], self.evd["spacing_zyx"], self.rca_cal, n_samples=9, label_name="left_main_1mm")
                length_q=float(np.exp(-0.5*((r["length_mm"]-14.0)/7.0)**2))
                score=0.48*r["graph_score"]+0.30*qsum["median_plane_score"]+0.17*qsum["plane_pass_fraction"]+0.05*length_q
                rec={"ostium_rank":int(o["rank"]),"combined_score":float(score),"length_mm":float(r["length_mm"]),
                     "graph_score":float(r["graph_score"]),"median_plane_score":float(qsum["median_plane_score"]),
                     "plane_pass_fraction":float(qsum["plane_pass_fraction"]),"median_radius_mm":float(qsum.get("median_radius_mm",np.nan)),
                     "median_offset_mm":float(qsum.get("median_offset_mm",np.nan)),"median_center_hu":float(qsum.get("median_center_hu",np.nan))}
                rows.append(rec); kept.append((score,r,qdf,qsum))
            gc.collect()
        if not kept: raise RuntimeError("No short left-main candidates found")
        order=np.argsort([x[0] for x in kept])[::-1]
        kept=[kept[i] for i in order]
        self.candidates=pd.DataFrame(rows).sort_values("combined_score",ascending=False).reset_index(drop=True)
        self.candidates.insert(0,"rank",np.arange(1,len(self.candidates)+1)); self.candidates.to_csv(p["candidates"],index=False)
        _,r,qdf,qsum=kept[0]
        ds=resample_path(r["path"], self.evd["spacing_zyx"], 0.6)
        src=ds_to_source(ds, self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
        pd.DataFrame(ds,columns=["z","y","x"]).to_csv(p["best_ds"],index=False)
        pd.DataFrame(src,columns=["z","y","x"]).to_csv(p["best_source"],index=False)
        status = "PASS_LOWRES" if (5 <= qsum["length_mm"] <= 30 and qsum["plane_pass_fraction"] >= 0.70 and qsum["median_plane_score"] >= 0.52) else ("REVIEW_LOWRES" if qsum["plane_pass_fraction"] >= 0.50 else "FAIL_LOWRES")
        summary={**qsum,"status":status,"algorithm":ALGORITHM_VERSION,"combined_score":float(kept[0][0]),"note":"Low-resolution gate only; source-resolution QC is required before acceptance."}
        _save_json(summary,p["summary"])
        qdf.to_csv(self.out/"left_main_lowres_serial_qc.csv",index=False)
        self.best={"path_ds":ds,"path_source":src,"summary":summary}
        return self.best

    def make_figures(self):
        if self.best is None: self.find_left_main()
        ct=self.evd["ct"]; aorta=self.evd["aorta"]; p=self.best["path_ds"]
        lo=np.maximum(0,np.floor(p.min(axis=0)-25).astype(int)); hi=np.minimum(np.asarray(ct.shape),np.ceil(p.max(axis=0)+25).astype(int)+1)
        roi=ct[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        fig,axs=plt.subplots(1,3,figsize=(16,5))
        projs=[np.max(roi,axis=0),np.max(roi,axis=1),np.max(roi,axis=2)]
        pp=p-lo
        for ax,im in zip(axs,projs): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.axis("off")
        axs[0].plot(pp[:,2],pp[:,1],linewidth=2); axs[1].plot(pp[:,2],pp[:,0],linewidth=2); axs[2].plot(pp[:,1],pp[:,0],linewidth=2)
        fig.suptitle("Left-main candidate on cached ~1 mm source evidence")
        f1=self.out/"01_left_main_lowres_mips.png"; plt.tight_layout(); plt.savefig(f1,dpi=170,bbox_inches="tight"); plt.close(fig)

        fig,axs=plt.subplots(2,6,figsize=(15,5.5))
        for row,(name,path) in enumerate((("RCA ref",self.rca_ds),("Left main",p))):
            rp=resample_path(path,self.evd["spacing_zyx"],0.5); s=arc_mm(rp,self.evd["spacing_zyx"])
            for j,frac in enumerate(np.linspace(.1,.9,6)):
                i=int(np.argmin(np.abs(s-frac*s[-1]))); i0=max(0,i-3); i1=min(len(rp)-1,i+3); t=(rp[i1]-rp[i0])*self.evd["spacing_zyx"]
                im,c=orthogonal_plane(ct,rp[i],t,self.evd["spacing_zyx"],half_mm=5,pix_mm=.25)
                axs[row,j].imshow(im,cmap="gray",vmin=-100,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); axs[row,j].plot(0,0,"+"); axs[row,j].set_title(f"{name} {frac:.1f}"); axs[row,j].axis("off")
        fig.suptitle("Cached-evidence orthogonal sections: RCA calibration vs proposed left main")
        f2=self.out/"02_left_main_lowres_cross_sections.png"; plt.tight_layout(); plt.savefig(f2,dpi=170,bbox_inches="tight"); plt.close(fig)
        return [f1,f2]

    def package(self):
        figs=self.make_figures(); p=self.paths()
        for src,name in ((p["ostia"],"left_ostium_candidates.csv"),(p["candidates"],"left_main_candidates_lowres.csv"),(p["best_source"],"left_main_best_source_zyx.csv"),(p["summary"],"left_main_lowres_summary.json")):
            if src.exists(): (self.out/name).write_bytes(src.read_bytes())
        html=self.out/"OPENPLAQUE_LEFT_MAIN_LOW_MEMORY_REPORT.html"
        def tag(fp): return f'<h2>{fp.name}</h2><img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode()}">'
        html.write_text("<html><body><h1>OpenPlaque left-main low-memory gate</h1><p>This report intentionally uses only cached ~1 mm evidence. It does not load the full source CCTA. A passing result is preliminary and must undergo a separate source-resolution QC step.</p>"+"".join(tag(f) for f in figs)+"<pre>"+json.dumps(self.best["summary"],indent=2)+"</pre></body></html>",encoding="utf-8")
        files=[*figs,self.out/"left_main_lowres_serial_qc.csv",self.out/"left_ostium_candidates.csv",self.out/"left_main_candidates_lowres.csv",self.out/"left_main_best_source_zyx.csv",self.out/"left_main_lowres_summary.json",html]
        zpath=self.out/"OPENPLAQUE_LEFT_MAIN_LOW_MEMORY_REPORT_BACK.zip"
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for f in files:
                if f.exists(): z.write(f,arcname=f.name)
        return zpath
