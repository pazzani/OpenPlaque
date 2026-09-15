from __future__ import annotations

"""Authoritative OpenPlaque integrated research/QC report v2.

Master Coronary Anatomy Baseline v2 controls vessel eligibility. RCA and the
24.997-mm-class LAD are accepted; LM and LCX are unresolved. The older 57.276-mm LAD
is registration provenance only. Research use only.
"""

import json, math, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
OUTPUT_DIRNAME = "Final_Integrated_Research_Report_v2"
SCIENTIFIC_STATUS = "AUTHORITATIVE_RCA_LAD_WITH_LONGITUDINAL_PLAQUE_ONLY"


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_dir(root, dirname):
    root = Path(root); direct = root / dirname
    if direct.is_dir(): return direct
    hits = sorted([p for p in root.rglob(dirname) if p.is_dir()], key=lambda p:(len(str(p)),str(p)))
    if not hits: raise FileNotFoundError(f"Could not find required folder {dirname} under {root}")
    return hits[0]


def _require_file(root, name):
    root = Path(root); direct = root / name
    if direct.exists(): return direct
    hits = sorted([p for p in root.rglob(name) if p.is_file()], key=lambda p:(len(str(p)),str(p)))
    if not hits: raise FileNotFoundError(f"Could not find required file {name} under {root}")
    return hits[0]


def _validate_master_anatomy(summary):
    if summary.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError(f"Unexpected master anatomy status {summary.get('status')!r}")
    a = summary.get("accepted", {}); rca=float(a.get("RCA_length_mm",math.nan)); lad=float(a.get("LAD_length_mm",math.nan))
    if not 52.0 <= rca <= 52.8: raise RuntimeError(f"Unexpected authoritative RCA length {rca}")
    if not 24.8 <= lad <= 25.2: raise RuntimeError(f"Unexpected authoritative LAD length {lad}")
    unresolved=set(summary.get("unresolved",[]))
    if not {"LM","LCX"}.issubset(unresolved): raise RuntimeError(f"LM/LCX must remain unresolved; got {sorted(unresolved)}")
    if "24.997" not in str(summary.get("downstream_analysis",{}).get("LAD","")):
        raise RuntimeError("Master anatomy does not constrain LAD downstream analysis to 24.997-mm class")
    return {"RCA_length_mm":rca,"LAD_length_mm":lad,"unresolved":sorted(unresolved)}


def _validate_lad_profile(summary, mapping, master_lad_mm):
    if summary.get("status") != "LONGITUDINAL_PLAQUE_PROFILE_COMPLETE_NONVOLUMETRIC":
        raise RuntimeError(f"Unexpected LAD plaque status {summary.get('status')!r}")
    lad=float(summary["accepted_lad_length_mm"])
    if abs(lad-master_lad_mm)>0.02: raise RuntimeError(f"LAD plaque length {lad:.6f} != master LAD {master_lad_mm:.6f}")
    if bool(summary.get("anatomical_plaque_volume_mm3_reported",True)): raise RuntimeError("LAD profile unexpectedly reports anatomical plaque volume")
    if summary.get("registration_status") != "PARTIAL_LONGITUDINAL_MAPPING_ONLY": raise RuntimeError("LAD profile must remain longitudinal-only")
    if float(mapping.get("arc_correlation",0))<0.999 or float(mapping.get("max_centerline_separation_mm",99))>0.60:
        raise RuntimeError("Accepted-LAD registration mapping failed conservative validation")
    return lad


def _best_interval(df, level):
    q=df[df.confidence_level.eq(level)].copy()
    if q.empty: return None
    r=q.sort_values(["mapped_native_voxels_vote_ge3","duration_mm"],ascending=False).iloc[0]
    return {k:float(getattr(r,k)) for k in ["arc_start_mm","arc_end_mm","duration_mm","mapped_native_voxels_vote_ge3","mapped_native_voxels_vote_ge4","mapped_native_voxels_vote_5"]}


def _lad_zone_summary(zones):
    out=[]
    for r in zones.itertuples(index=False):
        out.append({"support_zone_id":int(r.support_zone_id),"arc_start_mm":float(r.arc_start_mm),"arc_end_mm":float(r.arc_end_mm),"longitudinal_span_mm":float(r.longitudinal_span_mm),"peak_support_pixels":int(r.peak_support_pixels),"peak_rotation_fraction":float(r.peak_rotation_fraction),"mean_plaque_disagreement":float(r.mean_plaque_disagreement)})
    return out


def _plot_anatomy(out, master):
    vals=[float(master["accepted"]["RCA_length_mm"]),float(master["accepted"]["LAD_length_mm"])]
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.barh(["RCA","LAD"],vals)
    for i,v in enumerate(vals): ax.text(v+.7,i,f"{v:.2f} mm",va="center")
    ax.set_xlim(0,max(vals)*1.18); ax.set_xlabel("Accepted source-CCTA centerline length (mm)"); ax.set_title("Authoritative downstream coronary anatomy — Master Baseline v2"); ax.text(.99,.05,"LM unresolved · LCX unresolved",transform=ax.transAxes,ha="right"); ax.grid(axis="x",alpha=.2); fig.tight_layout()
    fp=Path(out)/"01_authoritative_anatomy_status.png"; fig.savefig(fp,dpi=190); plt.close(fig); return fp


def _plot_plaque(out, plaque):
    q=plaque.set_index("vessel").loc[["RCA","LAD"]]; cols=["majority_3plus_mm3","high_4plus_mm3","strict_5of5_mm3"]; x=np.arange(3); w=.35
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(x-w/2,[q.loc["RCA",c] for c in cols],w,label="RCA"); ax.bar(x+w/2,[q.loc["LAD",c] for c in cols],w,label="LAD"); ax.set_xticks(x,["Majority ≥3/5","High ≥4/5","Strict 5/5"]); ax.set_ylabel("Native curved-series vote volume (mm³)"); ax.set_title("5-fold plaque-model confidence (not source-space TPV)"); ax.legend(); ax.grid(axis="y",alpha=.2); fig.tight_layout()
    fp=Path(out)/"02_native_plaque_confidence.png"; fig.savefig(fp,dpi=190); plt.close(fig); return fp


def _plot_rca(out, fusion):
    x=fusion.arc_start_mm.to_numpy(float)+.5; fig,ax1=plt.subplots(figsize=(11,5.3)); ax1.plot(x,fusion.mapped_native_voxels_vote_ge3,label="Plaque ≥3/5"); ax1.plot(x,fusion.mapped_native_voxels_vote_ge4,label="Plaque ≥4/5"); ax1.set_xlim(10,50); ax1.set_xlabel("Accepted RCA arc (mm)"); ax1.set_ylabel("Mapped native plaque-support count"); ax2=ax1.twinx(); ax2.plot(x,fusion.pcat_mean_hu,linestyle="--",label="PCAT mean HU"); ax2.set_ylabel("OpenPlaque PCAT attenuation (HU)"); ax1.set_title("RCA — validated longitudinal plaque support vs locked PCAT, 10–50 mm"); lines=ax1.get_lines()+ax2.get_lines(); ax1.legend(lines,[l.get_label() for l in lines],loc="best"); ax1.grid(axis="x",alpha=.15); fig.tight_layout()
    fp=Path(out)/"03_RCA_longitudinal_plaque_PCAT.png"; fig.savefig(fp,dpi=190); plt.close(fig); return fp


def _plot_lad(out, bins):
    x=bins.arc_bin_start_mm.to_numpy(float)+.5; fig,axs=plt.subplots(2,1,figsize=(11,7.4),sharex=True); axs[0].bar(x,bins.total_curved_stack_plaque_support_pixels,width=.85); axs[0].set_ylabel("Curved-stack support\n(non-volumetric)"); axs[0].set_title("LAD — validated 24.997-mm longitudinal plaque profile")
    cols=["low_attenuation_fraction","noncalcified_fraction","mixed_intermediate_fraction","calcified_fraction"]; labels=["<30 HU","30–130 HU","130–350 HU","≥350 HU"]; base=np.zeros(len(bins))
    for c,lab in zip(cols,labels): y=bins[c].fillna(0).to_numpy(float); axs[1].bar(x,y,bottom=base,width=.85,label=lab); base+=y
    axs[1].set_ylim(0,1.02); axs[1].set_ylabel("Composition fraction\nof support"); axs[1].set_xlabel("Accepted LAD arc (mm)"); axs[1].legend(ncol=4,fontsize=8)
    for ax in axs: ax.axvspan(17.6259,21.4842,alpha=.10); ax.grid(axis="x",alpha=.12)
    fig.tight_layout(); fp=Path(out)/"04_LAD_validated_longitudinal_plaque.png"; fig.savefig(fp,dpi=190); plt.close(fig); return fp


def _plot_hierarchy(out):
    labels=["RCA anatomy","LAD anatomy","RCA PCAT","RCA plaque position","LAD plaque position","LM","LCX","3-D source plaque TPV"]; states=["ACCEPTED","ACCEPTED","LOCKED","LONGITUDINAL","LONGITUDINAL","UNRESOLVED","UNRESOLVED","NOT VALIDATED"]
    fig,ax=plt.subplots(figsize=(10,5.6)); ax.axis("off"); t=ax.table(cellText=list(zip(labels,states)),colLabels=["Domain","Current status"],loc="center",cellLoc="left",colLoc="left"); t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.5); ax.set_title("OpenPlaque evidence hierarchy after v2 reconciliation",pad=18); fig.tight_layout(); fp=Path(out)/"05_evidence_hierarchy.png"; fig.savefig(fp,dpi=190,bbox_inches="tight"); plt.close(fig); return fp


def _html_table(df, cols=None):
    q=df.copy()
    if cols is not None: q=q[cols]
    return q.to_html(index=False,border=0,classes="data")


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    drive_root=Path(drive_root); out=Path(output_root or drive_root/OUTPUT_DIRNAME); out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","baseline_commit":BASELINE,"scientific_status":SCIENTIFIC_STATUS})
    master_root=_find_dir(drive_root,"Master_Coronary_Anatomy_Baseline_v2"); atlas_root=_find_dir(drive_root,"Plaque_Ensemble_Confidence_Atlas_v1"); pcat_root=_find_dir(drive_root,"PCAT_RCA_10_50_Reproducibility_Lock"); fusion_root=_find_dir(drive_root,"Longitudinal_Plaque_PCAT_Fusion_v1"); lad_root=_find_dir(drive_root,"LAD_Validated_Longitudinal_Plaque_Profile_v1")
    master=_read_json(_require_file(master_root,"master_anatomy_summary.json")); valid=_validate_master_anatomy(master); accepted=pd.read_csv(_require_file(master_root,"accepted_anatomy.csv")); evidence=pd.read_csv(_require_file(master_root,"evidence_ledger.csv")); plaque=pd.read_csv(_require_file(atlas_root,"plaque_vote_confidence_summary.csv")); pcat=pd.read_csv(_require_file(pcat_root,"pcat_canonical_primary.csv")); pcat_row=pcat.iloc[0]
    fusion_summary=_read_json(_require_file(fusion_root,"fusion_summary.json")); rca_intervals=pd.read_csv(_require_file(fusion_root,"RCA_longitudinal_plaque_intervals.csv")); rca_fusion=pd.read_csv(_require_file(fusion_root,"RCA_pcat_plaque_longitudinal_fusion_10_50.csv"))
    if fusion_summary.get("status") != "COMPLETE_LONGITUDINAL_RESEARCH_FUSION": raise RuntimeError("RCA longitudinal fusion is not complete")
    lad_summary=_read_json(_require_file(lad_root,"summary.json")); lad_mapping=_read_json(_require_file(lad_root,"mapping_validation_summary.json")); lad_bins=pd.read_csv(_require_file(lad_root,"lad_longitudinal_plaque_profile_1mm_bins.csv")); lad_zones=pd.read_csv(_require_file(lad_root,"lad_longitudinal_plaque_support_zones.csv")); _validate_lad_profile(lad_summary,lad_mapping,valid["LAD_length_mm"])
    plaque_by=plaque.set_index("vessel"); overlap=fusion_summary["rca_overlap_summary"]
    summary={"status":"COMPLETE_FINAL_INTEGRATED_RESEARCH_REPORT_V2","scientific_status":SCIENTIFIC_STATUS,"baseline_commit":BASELINE,"authoritative_anatomy":{"RCA":{"length_mm":valid["RCA_length_mm"],"status":"ACCEPTED"},"LAD":{"length_mm":valid["LAD_length_mm"],"status":"ACCEPTED","frozen_core_mm":float(master["accepted"]["LAD_frozen_core_mm"]),"validated_proximal_addition_mm":float(master["accepted"]["LAD_validated_proximal_addition_mm"])},"LM":{"status":"UNRESOLVED","quantification_allowed":False},"LCX":{"status":"UNRESOLVED","quantification_allowed":False}},"native_curved_plaque_confidence":{v:{k:float(plaque_by.loc[v,k]) for k in ["majority_3plus_mm3","high_4plus_mm3","strict_5of5_mm3","any_fold_mm3"]} for v in ["RCA","LAD"]},"rca_pcat_lock":{"window_mm":[10,50],"wall_margin_mm":float(pcat_row.wall_margin_mm),"mean_hu":float(pcat_row.pcat_mean_hu),"median_hu":float(pcat_row.pcat_median_hu),"sd_hu":float(pcat_row.pcat_sd_hu),"fat_volume_ml":float(pcat_row.fat_volume_ml),"shell_volume_ml":float(pcat_row.shell_volume_ml),"fat_fraction":float(pcat_row.fat_fraction)},"rca_longitudinal":{"dominant_majority_interval":_best_interval(rca_intervals,"majority_3plus"),"dominant_high_interval":_best_interval(rca_intervals,"high_4plus"),"dominant_strict_interval":_best_interval(rca_intervals,"strict_5of5"),"pcat_window_bins":int(overlap["rca_pcat_bins"]),"majority_bins_in_pcat_window":int(overlap["rca_bins_majority_3plus_signal"]),"high_bins_in_pcat_window":int(overlap["rca_bins_high_4plus_signal"]),"strict_bins_in_pcat_window":int(overlap["rca_bins_strict_5of5_signal"]),"pcat_mean_all_10_50_fat_weighted_hu":float(overlap["pcat_mean_hu_all_10_50_fat_weighted"]),"pcat_mean_majority_bins_fat_weighted_hu":float(overlap["pcat_mean_hu_majority_bins_fat_weighted"])},"lad_longitudinal":{"accepted_length_mm":float(lad_summary["accepted_lad_length_mm"]),"support_zones":_lad_zone_summary(lad_zones),"consensus_support_pixels_total":int(lad_summary["consensus_plaque_support_pixels_total"]),"composition_support_distribution":lad_summary["composition_support_distribution"],"mapping_median_separation_mm":float(lad_mapping["median_centerline_separation_mm"]),"mapping_max_separation_mm":float(lad_mapping["max_centerline_separation_mm"]),"mapping_arc_correlation":float(lad_mapping["arc_correlation"]),"old_registration_centerline_window_mm":[float(lad_mapping["accepted_old_canonical_arc_min_mm"]),float(lad_mapping["accepted_old_canonical_arc_max_mm"])]},"superseded_downstream_claims":["57.276-mm LAD retained as registration provenance only, not authoritative quantitative LAD.","Old LAD plaque intervals outside old-canonical arc 32.845–57.276 mm excluded from authoritative LAD interpretation.","Historical LM/LCX assignments retired; LM and LCX unresolved."],"explicit_non_claims":["No source-space plaque TPV from native curved-series vote volumes.","No circumferential/radial plaque localization or 3-D source plaque mask.","No LAD plaque-PCAT association.","OpenPlaque PCAT attenuation is not Caristo FAI-Score."]}
    _write_json(out/"final_integrated_summary_v2.json",summary)
    provenance=pd.DataFrame([["Coronary anatomy","Master_Coronary_Anatomy_Baseline_v2","AUTHORITATIVE","Supersedes 57.276-mm LAD for downstream quantification"],["Native plaque ensemble","Plaque_Ensemble_Confidence_Atlas_v1","RETAIN","Model confidence only; not source-space TPV"],["RCA PCAT","PCAT_RCA_10_50_Reproducibility_Lock","LOCKED","Retained unchanged"],["RCA longitudinal plaque","Longitudinal_Plaque_PCAT_Fusion_v1","RETAIN RCA ONLY","Validated 1-D localization"],["LAD longitudinal plaque","LAD_Validated_Longitudinal_Plaque_Profile_v1","AUTHORITATIVE LAD","Replaces old 57-mm LAD plaque section"],["LM/LCX","Master_Coronary_Anatomy_Baseline_v2","UNRESOLVED","No vessel-specific quantification"]],columns=["domain","source","policy","note"]); provenance.to_csv(out/"evidence_provenance_v2.csv",index=False); accepted.to_csv(out/"accepted_anatomy_v2.csv",index=False); evidence.to_csv(out/"master_evidence_ledger_v2.csv",index=False); lad_bins.to_csv(out/"LAD_validated_plaque_profile_1mm.csv",index=False); lad_zones.to_csv(out/"LAD_validated_plaque_support_zones.csv",index=False); rca_intervals.to_csv(out/"RCA_validated_plaque_intervals.csv",index=False)
    figs=[_plot_anatomy(out,master),_plot_plaque(out,plaque),_plot_rca(out,rca_fusion),_plot_lad(out,lad_bins),_plot_hierarchy(out)]
    comp=lad_summary["composition_support_distribution"]; z=lad_zones.iloc[0]; css="body{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#17202a;line-height:1.45}h2{border-bottom:1px solid #ddd;padding-bottom:5px;margin-top:30px}.warning{background:#fff4e5;border-left:5px solid #d68910;padding:12px}.good{background:#eafaf1;border-left:5px solid #239b56;padding:12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.card{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}.big{font-size:1.5em;font-weight:bold}table.data{border-collapse:collapse;width:100%;font-size:.9em}table.data th,table.data td{border-bottom:1px solid #eee;padding:6px;text-align:right}table.data th:first-child,table.data td:first-child{text-align:left}img{max-width:100%;height:auto}.muted{color:#566573}"
    html=out/"OPENPLAQUE_FINAL_INTEGRATED_RESEARCH_REPORT_V2.html"; html.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenPlaque Final Integrated Research Report v2</title><style>{css}</style></head><body><h1>OpenPlaque — Integrated Coronary Research Report v2</h1><p class='muted'>Master Anatomy v2 · native 5-fold plaque confidence · validated 1-D plaque localization · locked RCA PCAT</p><div class='warning'><b>Research use only.</b> Master Anatomy v2 controls vessel eligibility. RCA and the 24.997-mm LAD are accepted; LM and LCX remain unresolved. Native plaque vote volumes are not source-space TPV.</div><h2>Executive summary</h2><div class='grid'><div class='card'><div class='big'>{valid['RCA_length_mm']:.2f} mm</div><b>Accepted RCA</b></div><div class='card'><div class='big'>{valid['LAD_length_mm']:.3f} mm</div><b>Accepted LAD</b></div><div class='card'><div class='big'>{float(pcat_row.pcat_mean_hu):.1f} HU</div><b>Locked RCA PCAT</b><br>10–50 mm</div><div class='card'><div class='big'>{len(lad_zones)}</div><b>Validated LAD plaque zone</b><br>{z.arc_start_mm:.2f}–{z.arc_end_mm:.2f} mm</div></div><div class='good' style='margin-top:14px'><b>Reconciliation:</b> the old 57.276-mm LAD is retained only as registration provenance. The authoritative quantitative LAD is 24.997 mm. The validated LAD plaque signal is localized to {z.arc_start_mm:.2f}–{z.arc_end_mm:.2f} mm on that accepted segment.</div><h2>1. Authoritative coronary anatomy</h2><p>Master Baseline v2 accepts RCA and LAD only. LM and LCX remain unresolved and are excluded from vessel-specific plaque/PCAT quantification.</p>{_html_table(accepted)}<img src='{figs[0].name}'><h2>2. Native plaque-model confidence</h2><p>These values quantify 5-fold agreement in native curved-series model coordinates. They are not anatomical source-space plaque volumes.</p>{_html_table(plaque[plaque.vessel.isin(['RCA','LAD'])],[ 'vessel','majority_3plus_mm3','high_4plus_mm3','strict_5of5_mm3','any_fold_mm3'])}<img src='{figs[1].name}'><h2>3. RCA plaque + PCAT</h2><p>The dominant reproducible RCA plaque is proximal (about 0–5 mm; strict core about 1–4 mm), mostly before the locked 10–50 mm PCAT window. Within that window only 11–12 mm reaches majority/high confidence; this single overlap bin is descriptive only.</p>{_html_table(rca_intervals)}<img src='{figs[2].name}'><h2>4. LAD — validated 24.997-mm plaque profile</h2><p>Only the later validated LAD longitudinal profile is promoted here. One support zone is present at {z.arc_start_mm:.3f}–{z.arc_end_mm:.3f} mm (span {z.longitudinal_span_mm:.3f} mm). Total curved-stack consensus support is {int(lad_summary['consensus_plaque_support_pixels_total'])}; this is non-volumetric. Support composition: low attenuation {100*comp['low_attenuation']['fraction']:.1f}%, noncalcified {100*comp['noncalcified']['fraction']:.1f}%, mixed/intermediate {100*comp['mixed_intermediate']['fraction']:.1f}%, calcified {100*comp['calcified']['fraction']:.1f}%.</p>{_html_table(lad_zones)}<img src='{figs[3].name}'><h2>5. Evidence hierarchy and superseded claims</h2><p>The 57.276-mm LAD and older longitudinal LAD intervals remain provenance for registration, but are not current downstream quantitative anatomy. Historical LM/LCX assignments remain retired.</p>{_html_table(provenance)}<img src='{figs[4].name}'><h2>6. Explicit limits</h2><ul><li>No source-space plaque TPV from the curved-series plaque model.</li><li>No circumferential/radial or 3-D source-space plaque localization.</li><li>No LAD PCAT association.</li><li>LM and LCX unresolved; no vessel-specific quantitative claims.</li><li>OpenPlaque PCAT attenuation is not Caristo FAI-Score.</li></ul><h2>Research/QC provenance</h2><p>Frozen baseline <code>{BASELINE}</code>. See <code>evidence_provenance_v2.csv</code> and <code>final_integrated_summary_v2.json</code>.</p></body></html>""",encoding="utf-8")
    zf=out/"OPENPLAQUE_FINAL_INTEGRATED_RESEARCH_REPORT_V2_BACK.zip"; state={"status":"COMPLETE","scientific_status":SCIENTIFIC_STATUS,"baseline_commit":BASELINE,"output_dir":str(out),"report":str(html),"zip":str(zf)}; _write_json(out/"run_state.json",state)
    with zipfile.ZipFile(zf,"w",zipfile.ZIP_DEFLATED) as zzip:
        for p in sorted(out.iterdir()):
            if p.is_file() and p!=zf: zzip.write(p,p.name)
    return {"summary":summary,**state}


def synthetic_v2_self_test():
    master={"status":"CORONARY_ANATOMY_BASELINE_V2_FROZEN","accepted":{"RCA_length_mm":52.365,"LAD_length_mm":24.997},"unresolved":["LM","LCX"],"downstream_analysis":{"LAD":"allowed only on accepted 24.997-mm-class combined LAD centerline"}}; v=_validate_master_anatomy(master); zones=pd.DataFrame([{"support_zone_id":1,"arc_start_mm":17.6,"arc_end_mm":21.5,"longitudinal_span_mm":3.9,"peak_support_pixels":9,"peak_rotation_fraction":.083,"mean_plaque_disagreement":.327}]); return {"passed":bool(abs(v["LAD_length_mm"]-24.997)<1e-6 and _lad_zone_summary(zones)[0]["peak_support_pixels"]==9),"algorithm":"final-integrated-research-report-v2"}
