from __future__ import annotations
import json, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import rca_source_space_plaque_geometry_v1 as g

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="rca-source-space-plaque-quantification-v1.0"
OUTPUT_DIRNAME="RCA_Source_Space_Plaque_Quantification_v1"
SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
RCA_CENTERLINE=Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
ACCEPTED=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/accepted_anatomy.csv")
PRIOR_PROFILE=Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
PCAT_PROFILE=Path("PCAT_RCA_10_50_Reproducibility_Lock/pcat_canonical_primary_longitudinal.csv")
MIN_STATION_QC_FRACTION=.90; MIN_PROFILE_CORRELATION=.85
STATUS_QC_FAIL="RCA_SOURCE_SPACE_PLAQUE_PROXY_GEOMETRY_QC_FAILED"
STATUS_UNSTABLE="RCA_SOURCE_SPACE_PLAQUE_PROXY_SENSITIVITY_UNSTABLE"
STATUS_PASS="RCA_SOURCE_SPACE_PLAQUE_PROXY_ROBUSTNESS_PASS"


def _read_json(p): return json.loads(Path(p).read_text())
def _write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding="utf-8")
def _req(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def _sensitivity(stations):
    rows=[]; profiles={}
    for m in g.SHELL_THICKNESSES_MM:
        d=stations[np.isclose(stations.shell_thickness_mm,m)]; p=g.profile_1mm(stations,m); profiles[m]=p
        row={"shell_thickness_mm":m,"station_qc_fraction":float(d.station_qc_pass.mean())}
        for c in ("total_plaque_proxy_mm3","low_attenuation_mm3","noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3","fatlike_excluded_mm3"):
            row[c]=float(d[c].sum(skipna=True))
        rows.append(row)
    s=pd.DataFrame(rows); nom=profiles[g.NOMINAL_SHELL_MM].set_index("arc_start_mm")["total_plaque_proxy_mm3"]; cors=[]
    for m,p in profiles.items():
        y=p.set_index("arc_start_mm")["total_plaque_proxy_mm3"].reindex(nom.index); ok=nom.notna()&y.notna()
        rho=float(spearmanr(nom[ok],y[ok]).statistic) if ok.sum()>=5 else np.nan
        s.loc[np.isclose(s.shell_thickness_mm,m),"profile_spearman_vs_nominal"]=rho
        if not np.isclose(m,g.NOMINAL_SHELL_MM) and np.isfinite(rho): cors.append(rho)
    return s,float(min(cors)) if cors else np.nan

def _prior_validation(profile,root):
    p=root/PRIOR_PROFILE
    if not p.exists() or profile.empty:return {"available":False}
    prior=pd.read_csv(p); arc_col=next((c for c in prior.columns if c in ("arc_start_mm","source_arc_start_mm","arc_mm")),None)
    nums=[c for c in prior.select_dtypes(include=[np.number]).columns if c!=arc_col]
    pref=[c for c in nums if any(k in c.lower() for k in ("vote","plaque","support","confidence"))]; metric=pref[0] if pref else (nums[0] if nums else None)
    if arc_col is None or metric is None:return {"available":True,"usable":False,"columns":list(prior.columns)}
    x=profile[["arc_start_mm","total_plaque_proxy_mm3"]].merge(prior[[arc_col,metric]].rename(columns={arc_col:"arc_start_mm",metric:"prior_metric"}),on="arc_start_mm",how="inner")
    ok=np.isfinite(x.total_plaque_proxy_mm3)&np.isfinite(x.prior_metric); rho=float(spearmanr(x.loc[ok,"total_plaque_proxy_mm3"],x.loc[ok,"prior_metric"]).statistic) if ok.sum()>=5 else np.nan
    return {"available":True,"usable":bool(ok.sum()>=5),"metric":metric,"matched_bins":int(ok.sum()),"spearman_rho":rho}

def _pcat(profile,root):
    p=root/PCAT_PROFILE
    if not p.exists() or profile.empty:return pd.DataFrame()
    return profile.merge(pd.read_csv(p),on=["arc_start_mm","arc_end_mm"],how="inner")

def _plot_profile(profile,out):
    if profile.empty:return
    x=profile.arc_start_mm+.5; plt.figure(figsize=(12,6))
    for c,label in [("total_plaque_proxy_mm3","total plaque proxy"),("calcified_mm3","calcified"),("noncalcified_mm3","noncalcified"),("low_attenuation_mm3","low attenuation (-30 to 30 HU)")]: plt.plot(x,profile[c],label=label)
    plt.xlabel("RCA source-centerline arc (mm)"); plt.ylabel("Resampled source-space volume per 1-mm bin (mm³)"); plt.title("RCA source-space plaque-composition proxy"); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()

def _plot_qc(geom,src,centers,tangents,nominal,out):
    good=nominal[nominal.station_qc_pass].copy()
    if good.empty:return
    ids=list(good.nlargest(4,"total_plaque_proxy_mm3").station_index.astype(int))+list(good.nsmallest(2,"total_plaque_proxy_mm3").station_index.astype(int)); ids=list(dict.fromkeys(ids))[:6]
    fig,axes=plt.subplots(2,3,figsize=(12,8)); axes=axes.ravel()
    for ax in axes:ax.axis("off")
    for ax,i in zip(axes,ids):
        row=nominal[nominal.station_index==i].iloc[0]; im,q=g.plane_image(geom,src,centers[i],tangents[i]); ax.imshow(im,cmap="gray",vmin=-100,vmax=800,extent=[q[0],q[-1],q[-1],q[0]])
        r=float(row.lumen_radius_median_mm); ax.add_patch(plt.Circle((0,0),r,fill=False,linewidth=1.2)); ax.add_patch(plt.Circle((0,0),r+g.NOMINAL_SHELL_MM,fill=False,linewidth=1.2,linestyle="--")); ax.scatter([0],[0],s=8)
        ax.set_title(f"arc {row.arc_mm:.1f} mm | proxy {row.total_plaque_proxy_mm3:.2f} mm³"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Automatically selected RCA orthogonal source-CCTA QC"); fig.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)

def synthetic_self_test(): return g.synthetic_self_test()

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})
    master=_read_json(_req(root/MASTER)); accepted=pd.read_csv(_req(root/ACCEPTED)); r=accepted[accepted.structure.astype(str).str.upper()=="RCA"]
    if r.empty or str(r.iloc[0].status).upper()!="ACCEPTED": raise RuntimeError("Frozen master does not mark RCA ACCEPTED")
    geom,src,voxel_volume=g.load_source(root/SOURCE_CACHE); rca=g.load_path(_req(root/RCA_CENTERLINE)); centers,arcs=g.resample(rca); tang=np.gradient(centers,axis=0); tang/=np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-9); weights=g.arc_weights(arcs)
    center_hu=g.sample(geom,src,centers)
    if float(np.mean(center_hu>=200))<.95: raise RuntimeError("Canonical RCA source support control failed")
    rows=[]
    for i,(c,t,a,ds) in enumerate(zip(centers,tang,arcs,weights)):
        L=g.lumen(geom,src,c,t); base={k:v for k,v in L.items() if k not in ("radii","theta","u","v")}
        for shell in g.SHELL_THICKNESSES_MM:
            if L["station_qc_pass"]: comp=g.integrate_shell(geom,src,c,L,shell,ds)
            else: comp={k:np.nan for k in ("shell_volume_mm3","fatlike_excluded_mm3","low_attenuation_raw_lt30_mm3","low_attenuation_mm3","noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3","total_plaque_proxy_mm3","fatlike_fraction","plaque_proxy_fraction_of_shell","plaque_burden_proxy","shell_mean_hu")}
            rows.append({"station_index":i,"arc_mm":float(a),"integration_ds_mm":float(ds),"shell_thickness_mm":float(shell),**base,**comp})
    stations=pd.DataFrame(rows); stations.to_csv(out/"RCA_source_space_station_quantification.csv",index=False)
    nominal=stations[np.isclose(stations.shell_thickness_mm,g.NOMINAL_SHELL_MM)].copy(); qc=float(nominal.station_qc_pass.mean()); profile=g.profile_1mm(stations); profile.to_csv(out/"RCA_source_space_plaque_profile_1mm.csv",index=False)
    sens,min_corr=_sensitivity(stations); sens.to_csv(out/"RCA_shell_sensitivity.csv",index=False); prior=_prior_validation(profile,root); fusion=_pcat(profile,root)
    if not fusion.empty:fusion.to_csv(out/"RCA_source_space_plaque_PCAT_fusion_10_50.csv",index=False)
    cols=("low_attenuation_mm3","noncalcified_mm3","mixed_intermediate_mm3","calcified_mm3","total_plaque_proxy_mm3","fatlike_excluded_mm3"); totals={c:float(nominal[c].sum(skipna=True)) for c in cols}
    status=STATUS_QC_FAIL if qc<MIN_STATION_QC_FRACTION else (STATUS_UNSTABLE if (not np.isfinite(min_corr) or min_corr<MIN_PROFILE_CORRELATION) else STATUS_PASS)
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"master_modified":False,"rca_centerline_length_mm":float(g.arc(rca)[-1]),"source_voxel_volume_mm3":voxel_volume,"station_count":int(len(nominal)),"station_qc_fraction":qc,"nominal_shell_thickness_mm":g.NOMINAL_SHELL_MM,"shell_sensitivity_min_profile_spearman":min_corr,"nominal_totals_mm3":totals,"prior_longitudinal_plaque_validation":prior,"pcat_fusion_available":bool(not fusion.empty),"hu_bins":{"fatlike_excluded":"< -30","low_attenuation":"-30 to <30","noncalcified":"30 to <130","mixed_intermediate":"130 to <350","calcified":">=350","raw_low_attenuation_diagnostic":"<30"},"is_validated_clinical_tpv":False,"scientific_boundary":"Physical Series-7 source-space resampling yields mm^3 plaque-composition proxy volumes around the accepted RCA. The outer wall is represented by a sensitivity-tested shell beyond a directional lumen boundary, not an independently validated outer-wall segmentation; values are not validated clinical TPV."}
    _write_json(out/"summary.json",summary); _write_json(out/"input_provenance.json",{"source_cache":str(root/SOURCE_CACHE),"rca_centerline":str(root/RCA_CENTERLINE),"master":str(root/MASTER),"prior_profile":str(root/PRIOR_PROFILE),"pcat_profile":str(root/PCAT_PROFILE)})
    _plot_profile(profile,out/"01_RCA_source_space_plaque_profile.png"); _plot_qc(geom,src,centers,tang,nominal,out/"02_RCA_orthogonal_source_QC.png")
    plt.figure(figsize=(9,5)); plt.plot(sens.shell_thickness_mm,sens.total_plaque_proxy_mm3,marker="o"); plt.xlabel("Wall-shell thickness (mm)"); plt.ylabel("Total plaque proxy volume (mm³)"); plt.title("RCA shell-thickness sensitivity"); plt.tight_layout(); plt.savefig(out/"03_RCA_shell_sensitivity.png",dpi=160); plt.close()
    report=out/"OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_QUANTIFICATION_REPORT.html"; report.write_text("<html><body><h1>OpenPlaque RCA Source-Space Plaque Quantification v1</h1>"+f"<p><b>Status:</b> {status}</p><p>Station QC: {qc:.3f}; minimum profile Spearman across shell sensitivities: {min_corr:.3f}.</p>"+"<p><b>Research boundary:</b> physical mm³ source-space plaque-composition proxy, not validated clinical TPV. Outer wall is sensitivity-tested rather than independently segmented.</p>"+"<h2>Nominal totals</h2><pre>"+json.dumps(totals,indent=2)+"</pre><h2>Sensitivity</h2>"+sens.to_html(index=False)+"<h2>Prior longitudinal comparison</h2><pre>"+json.dumps(prior,indent=2)+"</pre>"+'<img src="01_RCA_source_space_plaque_profile.png" style="max-width:100%"><img src="02_RCA_orthogonal_source_QC.png" style="max-width:100%"><img src="03_RCA_shell_sensitivity.png" style="max-width:100%"></body></html>',encoding="utf-8")
    _write_json(out/"run_state.json",{"status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE}); zpath=out/"OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_QUANTIFICATION_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=zpath:z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}
