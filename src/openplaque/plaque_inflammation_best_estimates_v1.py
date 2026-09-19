from __future__ import annotations

import json, math, zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="plaque-inflammation-best-estimates-v1.0"
OUTPUT_DIRNAME="Plaque_Inflammation_Best_Estimates_v1"

LEGACY_PLAQUE_CSV=Path("UCLA_Plaque_Type_Estimates/best_estimate_plaque_types_by_artery.csv")
RCA_PCAT_CSV=Path("RCA_Plaque_PCAT_Research_Lock_v1/RCA_locked_research_plaque_PCAT_profile_10_50.csv")
LAD_PCAT_CSV=Path("LAD_Source_Space_PCAT_Feasibility_v1/LAD_PCAT_segment_summary.csv")
LCX_PCAT_CSV=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/LCX_OM_PCAT_primary_summary.csv")

CONFIRM2_TPV_STAGES=[
    (0.0,0.0,"0 mm3"),
    (0.0,250.0,">0-250 mm3"),
    (250.0,750.0,">250-750 mm3"),
    (750.0,np.inf,">750 mm3"),
]
CONFIRM2_NCPV_STAGES=[
    (0.0,20.0,"0-20 mm3"),
    (20.0,125.0,">20-125 mm3"),
    (125.0,375.0,">125-375 mm3"),
    (375.0,np.inf,">375 mm3"),
]
CARISTO_FAT_MIN_HU=-190.0
CARISTO_FAT_MAX_HU=-30.0
CARISTO_SEGMENT_MM=40.0

LITERATURE=[
    {
      "short_name":"CONFIRM2 whole-heart plaque 2026",
      "citation":"van Rosendael AR et al. Am J Prev Cardiol. 2026; article 101712.",
      "doi":"10.1016/j.ajpc.2026.101712",
      "use":"TPV/NCPV whole-coronary burden and TPV staging",
      "implemented":"TPV stage 0, >0-250, >250-750, >750 mm3; NCPV stage 0-20, >20-125, >125-375, >375 mm3"
    },
    {
      "short_name":"SCCT/NASCI plaque consensus 2021",
      "citation":"Shaw LJ, Blankstein R, Bax JJ, et al. J Cardiovasc Comput Tomogr. 2021;15:93-109.",
      "doi":"10.1016/j.jcct.2020.11.002",
      "use":"structured plaque reporting and high-risk plaque terminology",
      "implemented":"TPV composition plus LAP; positive remodeling/spotty calcium/napkin-ring are not claimed without validated measurements"
    },
    {
      "short_name":"ORFAN 2024",
      "citation":"Chan K et al. Lancet. 2024;403:2606-2618.",
      "doi":"10.1016/S0140-6736(24)00596-8",
      "use":"three-vessel coronary inflammation context",
      "implemented":"RCA/LAD/LCX direct PCAT attenuation reported separately; no proprietary FAI-Score fabricated"
    },
    {
      "short_name":"Standardised coronary inflammation 2021",
      "citation":"Oikonomou EK et al. Cardiovasc Res. 2021;117:2677-2690.",
      "doi":"10.1093/cvr/cvab286",
      "use":"FAI/FAI-Score framework",
      "implemented":"raw PCAT attenuation window -190 to -30 HU; Caristo FAI-Score explicitly unavailable because proprietary technical/anatomic/biologic adjustment is not reproduced"
    },
]


def _req(p: Path)->Path:
    p=Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,default=str,allow_nan=True),encoding="utf-8")


def _stage(v: float, stages)->str:
    if not np.isfinite(v):
        return "NA"
    if v==0:
        return stages[0][2]
    for lo,hi,label in stages[1:]:
        if v>lo and v<=hi:
            return label
    return stages[-1][2]


def _safe(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def build_plaque_estimates(legacy: pd.DataFrame)->pd.DataFrame:
    x=legacy.copy()
    x["artery_norm"]=x["artery"].astype(str).str.strip().str.upper().replace({"LEFT MAIN":"LM"})
    wanted={"LAD","RCA","LCX","LM"}
    rows=[]
    for vessel in ("LAD","RCA","LCX","LM"):
        if vessel=="LM":
            g=x[x.artery_norm=="LM"]
        else:
            g=x[x.artery_norm==vessel]
        if g.empty:
            raise RuntimeError(f"Missing {vessel} in legacy plaque CSV")
        r=g.iloc[0]
        vv=_safe(r.get("voxel_volume_mm3"))
        lap=_safe(r.get("low_attenuation_mm3"))
        nc30_130=_safe(r.get("noncalcified_mm3"))
        mixed130_350=_safe(r.get("mixed_intermediate_mm3"))
        cpv=_safe(r.get("calcified_mm3"))
        tpv=_safe(r.get("total_plaque_volume_proxy_mm3"))
        strict=_safe(r.get("strict_nnunet_core_volume_mm3"))
        candidate_vox=_safe(r.get("candidate_region_voxels"))

        if vessel=="LM":
            # The 54 mm3 LM value is a clinical calcium-volume anchor, not a
            # complete contrast-CCTA total plaque segmentation.
            lower=tpv
            upper=np.nan
            provenance="clinical LM calcium-volume anchor + Series-7 ROI cross-check"
            confidence="moderate for calcified plaque; low for total plaque"
            limitation="Noncalcified LM plaque not quantified; 54 mm3 is a floor/anchor rather than a complete Cleerly-like TPV."
        else:
            lower=strict
            upper=(strict+candidate_vox*vv) if np.isfinite(candidate_vox) and np.isfinite(vv) else np.nan
            provenance="legacy MM-DHM/Q3D plaque proxy rescaled to Series-7 source voxel volume"
            confidence={"RCA":"moderate","LAD":"low-moderate","LCX":"low"}[vessel]
            limitation={
                "RCA":"Absolute TPV remains a research proxy; independent source-space plaque-excess localization is positive but is not a clinical TPV.",
                "LAD":"Source-space plaque transfer did not validate absolute plaque volume; point estimate remains legacy-derived.",
                "LCX":"Clinical LCX identity is unresolved; point estimate derives from the vendor CX series and should be treated as low-confidence."
            }[vessel]

        # For the legacy four-bin classifier, Cleerly/CONFIRM2-like NCPV is
        # reconstructed as all non-calcified bins below 350 HU. LM is different:
        # the 54 mm3 value is a calcium-only anchor, so LM noncalcified plaque is
        # unknown rather than zero.
        if vessel=="LM":
            lap=np.nan
            nc30_130=np.nan
            mixed130_350=np.nan
            ncpv=np.nan
        else:
            ncpv=sum(v for v in (lap,nc30_130,mixed130_350) if np.isfinite(v))
        rows.append({
            "vessel":vessel,
            "tpv_best_estimate_mm3":tpv,
            "tpv_strict_or_known_lower_mm3":lower,
            "tpv_candidate_envelope_upper_mm3":upper,
            "ncpv_best_estimate_mm3":ncpv,
            "lap_best_estimate_mm3":lap,
            "noncalcified_30_130_mm3":nc30_130,
            "fibrous_intermediate_130_350_mm3":mixed130_350,
            "calcified_plaque_volume_mm3":cpv,
            "ncp_fraction_of_tpv":ncpv/tpv if np.isfinite(ncpv) and np.isfinite(tpv) and tpv>0 else np.nan,
            "lap_fraction_of_tpv":lap/tpv if np.isfinite(lap) and np.isfinite(tpv) and tpv>0 else np.nan,
            "confirm2_tpv_stage":_stage(tpv,CONFIRM2_TPV_STAGES),
            "confirm2_ncpv_stage":_stage(ncpv,CONFIRM2_NCPV_STAGES),
            "pav_percent":np.nan,
            "pav_status":"NA: no defensible source-space outer-vessel volume for this fused endpoint",
            "lap_gt_2mm3":(bool(lap>2.0) if np.isfinite(lap) else np.nan),
            "high_risk_plaque_status":"not adjudicable: positive remodeling is not validated in this endpoint",
            "absolute_volume_confidence":confidence,
            "provenance":provenance,
            "limitation":limitation,
        })
    return pd.DataFrame(rows)


def build_whole_heart(plaque: pd.DataFrame)->pd.DataFrame:
    # This is a major-vessel aggregate (LAD/RCA/LCX/LM). CONFIRM2 quantified all
    # segments/branches >=1.5 mm, so this row must not be called a literal whole-coronary-tree TPV.
    tpv=float(plaque.tpv_best_estimate_mm3.sum())
    lower=float(plaque.tpv_strict_or_known_lower_mm3.sum())
    upper_known=float(plaque.loc[plaque.vessel!="LM","tpv_candidate_envelope_upper_mm3"].sum()+plaque.loc[plaque.vessel=="LM","tpv_best_estimate_mm3"].sum())
    known_non_lm=plaque[plaque.vessel!="LM"]
    ncp_known=float(known_non_lm.ncpv_best_estimate_mm3.sum())
    lap_known=float(known_non_lm.lap_best_estimate_mm3.sum())
    cp=float(plaque.calcified_plaque_volume_mm3.sum())
    return pd.DataFrame([{
        "aggregate":"MAJOR_VESSEL_TOTAL_LAD_RCA_LCX_LM",
        "tpv_best_estimate_mm3":tpv,
        "tpv_strict_or_known_lower_mm3":lower,
        "tpv_candidate_envelope_upper_known_mm3":upper_known,
        "ncpv_known_lower_mm3":ncp_known,
        "lap_known_lower_mm3":lap_known,
        "calcified_plaque_volume_mm3":cp,
        "confirm2_tpv_stage_best_estimate":_stage(tpv,CONFIRM2_TPV_STAGES),
        "confirm2_tpv_stage_lower":_stage(lower,CONFIRM2_TPV_STAGES),
        "confirm2_tpv_stage_upper_known":_stage(upper_known,CONFIRM2_TPV_STAGES),
        "confirm2_ncpv_stage_known_lower":_stage(ncp_known,CONFIRM2_NCPV_STAGES),
        "whole_coronary_tree_equivalence":"NO",
        "reason_not_literal_whole_heart":"Side branches >=1.5 mm are not comprehensively quantified and LM noncalcified plaque is unknown.",
    }])


def _weighted_mean(values,weights):
    v=np.asarray(values,float); w=np.asarray(weights,float)
    ok=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not ok.any(): return np.nan
    return float(np.average(v[ok],weights=w[ok]))


def build_inflammation_estimates(rca: pd.DataFrame, lad: pd.DataFrame, lcx: pd.DataFrame)->pd.DataFrame:
    # RCA: locked 10-50 mm profile, exactly aligned to the canonical raw PCAT segment definition.
    rca_hu=_weighted_mean(rca["mean_hu"],rca["fat_voxels"])
    rca_fat=int(pd.to_numeric(rca["fat_voxels"],errors="coerce").fillna(0).sum())

    lad_frozen=lad[lad["segment"].astype(str)=="frozen_LAD"]
    if lad_frozen.empty: raise RuntimeError("Missing frozen_LAD PCAT summary")
    lr=lad_frozen.iloc[0]

    c6=lcx[lcx["vessel"].astype(str)=="C6"]
    c7=lcx[lcx["vessel"].astype(str)=="C7"]
    if c6.empty: raise RuntimeError("Missing C6 PCAT summary")
    c6r=c6.iloc[0]

    rows=[
      {
        "vessel":"RCA",
        "pcat_mean_hu_best_estimate":rca_hu,
        "pcat_segment_start_mm":10.0,
        "pcat_segment_end_mm":50.0,
        "pcat_segment_length_mm":40.0,
        "standard_target_length_mm":40.0,
        "segment_coverage_fraction":1.0,
        "fat_voxels":rca_fat,
        "caristo_comparability":"best raw-PCAT match",
        "fai_score":"NA",
        "fai_score_reason":"Caristo FAI-Score is proprietary and adjusts raw FAI for technical, anatomical, biological, age and sex factors.",
        "confidence":"high for direct raw PCAT attenuation; not a clinical FAI-Score",
        "provenance":"RCA_Plaque_PCAT_Research_Lock_v1, source-space 10-50 mm",
      },
      {
        "vessel":"LAD",
        "pcat_mean_hu_best_estimate":_safe(lr["pcat_mean_hu"]),
        "pcat_segment_start_mm":_safe(lr["frozen_arc_start_mm"]),
        "pcat_segment_end_mm":_safe(lr["frozen_arc_end_mm"]),
        "pcat_segment_length_mm":_safe(lr["segment_length_mm"]),
        "standard_target_length_mm":40.0,
        "segment_coverage_fraction":_safe(lr["segment_length_mm"])/40.0,
        "fat_voxels":_safe(lr["fat_voxels"]),
        "caristo_comparability":"partial proximal-segment raw PCAT proxy",
        "fai_score":"NA",
        "fai_score_reason":"Caristo FAI-Score is proprietary; available validated LAD PCAT coverage is shorter than the standardized 40 mm segment.",
        "confidence":"moderate for direct raw PCAT attenuation; partial segment only",
        "provenance":"LAD_Source_Space_PCAT_Feasibility_v1 frozen_LAD",
      },
      {
        "vessel":"LCX",
        "pcat_mean_hu_best_estimate":_safe(c6r["pcat_mean_hu"]),
        "pcat_segment_start_mm":_safe(c6r["segment_arc_start_mm"]),
        "pcat_segment_end_mm":_safe(c6r["segment_arc_end_mm"]),
        "pcat_segment_length_mm":_safe(c6r["segment_length_mm"]),
        "standard_target_length_mm":40.0,
        "segment_coverage_fraction":_safe(c6r["segment_length_mm"])/40.0,
        "fat_voxels":_safe(c6r["fat_voxels"]),
        "caristo_comparability":"short structural LCX-like parent proxy",
        "fai_score":"NA",
        "fai_score_reason":"Clinical LCX identity and standardized 40 mm proximal coverage are unresolved; Caristo FAI-Score is proprietary.",
        "confidence":"low for clinical LCX FAI equivalence; good direct attenuation precision for C6",
        "provenance":"LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1 C6",
      },
      {
        "vessel":"LM",
        "pcat_mean_hu_best_estimate":np.nan,
        "pcat_segment_start_mm":np.nan,
        "pcat_segment_end_mm":np.nan,
        "pcat_segment_length_mm":np.nan,
        "standard_target_length_mm":np.nan,
        "segment_coverage_fraction":np.nan,
        "fat_voxels":np.nan,
        "caristo_comparability":"not standardized",
        "fai_score":"NA",
        "fai_score_reason":"Validated Caristo/ORFAN FAI-Score is reported for RCA, LAD and LCX, not LM.",
        "confidence":"not estimated",
        "provenance":"none",
      },
    ]
    out=pd.DataFrame(rows)
    # Keep the OM-like daughter as a descriptive alternate, not a second clinical LCX estimate.
    if not c7.empty:
        c7r=c7.iloc[0]
        out.loc[out.vessel=="LCX","alternate_structural_branch"] = (
            f"C7/OM-like daughter direct PCAT {_safe(c7r['pcat_mean_hu']):.2f} HU over "
            f"{_safe(c7r['segment_length_mm']):.1f} mm"
        )
    return out


def plot_plaque(plaque: pd.DataFrame,out: Path):
    order=["LAD","RCA","LCX","LM"]
    d=plaque.set_index("vessel").loc[order]
    fig,ax=plt.subplots(figsize=(9,5))
    bottom=np.zeros(len(d))
    for col,label in [
        ("lap_best_estimate_mm3","LAP <30 HU proxy"),
        ("noncalcified_30_130_mm3","30-130 HU"),
        ("fibrous_intermediate_130_350_mm3","130-350 HU"),
        ("calcified_plaque_volume_mm3",">=350 HU / LM calcium anchor"),
    ]:
        vals=d[col].fillna(0).to_numpy(float)
        ax.bar(order,vals,bottom=bottom,label=label)
        bottom+=vals
    ax.set_ylabel("Best-estimate plaque volume (mm3)")
    ax.set_title("OpenPlaque best-estimate plaque composition")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"01_plaque_composition_best_estimate.png",dpi=170); plt.close(fig)


def plot_uncertainty(plaque: pd.DataFrame,out: Path):
    d=plaque.set_index("vessel").loc[["LAD","RCA","LCX","LM"]]
    x=np.arange(len(d)); best=d.tpv_best_estimate_mm3.to_numpy(float); low=d.tpv_strict_or_known_lower_mm3.to_numpy(float)
    upper=d.tpv_candidate_envelope_upper_mm3.to_numpy(float)
    loerr=np.maximum(best-low,0)
    hierr=np.where(np.isfinite(upper),np.maximum(upper-best,0),0)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.errorbar(x,best,yerr=np.vstack([loerr,hierr]),fmt="o",capsize=5)
    ax.set_xticks(x,["LAD","RCA","LCX","LM"])
    ax.set_ylabel("TPV proxy (mm3)")
    ax.set_title("Best estimate with strict lower / candidate-envelope upper")
    ax.text(3,best[3]," LM upper unknown",va="center")
    fig.tight_layout(); fig.savefig(out/"02_plaque_uncertainty.png",dpi=170); plt.close(fig)


def plot_pcat(infl: pd.DataFrame,out: Path):
    d=infl[infl.vessel.isin(["RCA","LAD","LCX"])].copy()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(d.vessel,d.pcat_mean_hu_best_estimate)
    ax.set_ylabel("Direct PCAT mean attenuation (HU)")
    ax.set_title("Raw PCAT attenuation — not proprietary Caristo FAI-Score")
    for i,r in d.reset_index(drop=True).iterrows():
        ax.text(i,float(r.pcat_mean_hu_best_estimate),f" {float(r.segment_coverage_fraction)*100:.0f}% of 40 mm",ha="center",va="bottom")
    fig.tight_layout(); fig.savefig(out/"03_inflammation_raw_pcat.png",dpi=170); plt.close(fig)


def write_report(out: Path, plaque: pd.DataFrame, whole: pd.DataFrame, infl: pd.DataFrame, lit: pd.DataFrame):
    def table(df):
        return df.to_html(index=False,float_format=lambda x:f"{x:.2f}" if isinstance(x,(float,np.floating)) and np.isfinite(x) else "")
    total=whole.iloc[0]
    html=f"""<html><head><meta charset='utf-8'><style>
body{{font-family:Arial,sans-serif;margin:32px;max-width:1400px}} table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}} th{{background:#f2f2f2}}
.warn{{padding:12px;background:#fff3cd;border:1px solid #e0c36c;border-radius:6px}}
.good{{padding:12px;background:#eef7ee;border:1px solid #9bbf9b;border-radius:6px}}
</style></head><body>
<h1>OpenPlaque Plaque + Coronary Inflammation Best Estimates v1</h1>
<p class='warn'><b>Research endpoint, not a clinical Cleerly or Caristo report.</b> Absolute plaque volumes are fused research proxies. Caristo FAI-Score and CaRi-Heart risk are proprietary outputs and are not reproduced.</p>
<h2>Headline</h2>
<p>Major-vessel TPV best-estimate proxy: <b>{total.tpv_best_estimate_mm3:.1f} mm3</b>, corresponding to CONFIRM2 stage <b>{total.confirm2_tpv_stage_best_estimate}</b>. The strict/known lower aggregate is {total.tpv_strict_or_known_lower_mm3:.1f} mm3 and the known candidate-envelope aggregate is {total.tpv_candidate_envelope_upper_known_mm3:.1f} mm3. This is not literal whole-coronary-tree TPV because side branches are incomplete and LM noncalcified plaque is unmeasured.</p>
<h2>Plaque by vessel</h2>{table(plaque)}
<img src='01_plaque_composition_best_estimate.png' width='900'>
<img src='02_plaque_uncertainty.png' width='900'>
<h2>Major-vessel aggregate</h2>{table(whole)}
<h2>Inflammation / PCAT</h2>
<p>Raw PCAT is the mean attenuation of pericoronary adipose voxels; the literature-standard adipose window is -190 to -30 HU. RCA 10-50 mm is the closest match to the validated raw-PCAT protocol. LAD and LCX coverage limitations are explicit below.</p>
{table(infl)}
<img src='03_inflammation_raw_pcat.png' width='850'>
<h2>Literature alignment</h2>{table(lit)}
<h2>Interpretive boundaries</h2>
<ul>
<li>CONFIRM2/Cleerly AI-QCT quantifies all coronary segments/branches of adequate caliber; this OpenPlaque total currently covers the four named major-vessel territories only.</li>
<li>PAV is intentionally not estimated because a defensible source-space outer-vessel volume is not available across all territories.</li>
<li>LAP volume is reported, but high-risk plaque is not called because positive remodeling is not validated here.</li>
<li>LM 54 mm3 is a clinical calcium-volume anchor, not a complete CCTA plaque segmentation.</li>
<li>Direct PCAT attenuation is not Caristo FAI-Score. No age/sex/tube-voltage/anatomic standardization or CaRi-Heart risk is fabricated.</li>
</ul>
</body></html>"""
    (out/"OPENPLAQUE_PLAQUE_INFLAMMATION_BEST_ESTIMATES_V1_REPORT.html").write_text(html,encoding="utf-8")


def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{"status":"RUNNING","algorithm":ALGORITHM,"baseline":BASELINE})

    legacy=pd.read_csv(_req(root/LEGACY_PLAQUE_CSV))
    rca=pd.read_csv(_req(root/RCA_PCAT_CSV))
    lad=pd.read_csv(_req(root/LAD_PCAT_CSV))
    lcx=pd.read_csv(_req(root/LCX_PCAT_CSV))

    plaque=build_plaque_estimates(legacy)
    whole=build_whole_heart(plaque)
    infl=build_inflammation_estimates(rca,lad,lcx)
    lit=pd.DataFrame(LITERATURE)

    plaque.to_csv(out/"plaque_best_estimates_by_vessel.csv",index=False)
    whole.to_csv(out/"major_vessel_aggregate.csv",index=False)
    infl.to_csv(out/"inflammation_best_estimates_by_vessel.csv",index=False)
    lit.to_csv(out/"literature_alignment.csv",index=False)

    plot_plaque(plaque,out); plot_uncertainty(plaque,out); plot_pcat(infl,out)
    write_report(out,plaque,whole,infl,lit)

    summary={
      "status":"COMPLETE",
      "algorithm":ALGORITHM,
      "baseline_commit":BASELINE,
      "headline_major_vessel_tpv_best_estimate_mm3":float(whole.iloc[0].tpv_best_estimate_mm3),
      "headline_confirm2_tpv_stage":whole.iloc[0].confirm2_tpv_stage_best_estimate,
      "plaque_by_vessel":plaque.to_dict("records"),
      "inflammation_by_vessel":infl.to_dict("records"),
      "scientific_boundaries":{
        "whole_heart":"Major-vessel LAD/RCA/LCX/LM aggregate only; not literal Cleerly whole-coronary-tree TPV.",
        "pav":"Not reported without defensible source-space outer-vessel volume.",
        "hrp":"LAP reported; HRP not adjudicated without validated positive remodeling.",
        "fai":"Direct PCAT attenuation only. Proprietary Caristo FAI-Score and CaRi-Heart risk are not reproduced.",
        "lm":"54 mm3 is a clinical calcium-volume anchor; noncalcified LM plaque is unknown."
      }
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM,"result_status":"PLAQUE_INFLAMMATION_BEST_ESTIMATES_COMPLETE"})

    archive=out/"OPENPLAQUE_PLAQUE_INFLAMMATION_BEST_ESTIMATES_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p!=archive: z.write(p,p.name)
    return summary


def synthetic_self_test():
    d=pd.DataFrame([
      {"artery":"LAD","voxel_volume_mm3":0.04,"low_attenuation_mm3":2.0,"noncalcified_mm3":3.0,"mixed_intermediate_mm3":4.0,"calcified_mm3":5.0,"strict_nnunet_core_volume_mm3":6.0,"candidate_region_voxels":100,"total_plaque_volume_proxy_mm3":14.0},
      {"artery":"RCA","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":2.0,"mixed_intermediate_mm3":3.0,"calcified_mm3":4.0,"strict_nnunet_core_volume_mm3":5.0,"candidate_region_voxels":50,"total_plaque_volume_proxy_mm3":10.0},
      {"artery":"LCX","voxel_volume_mm3":0.04,"low_attenuation_mm3":1.0,"noncalcified_mm3":1.0,"mixed_intermediate_mm3":1.0,"calcified_mm3":1.0,"strict_nnunet_core_volume_mm3":2.0,"candidate_region_voxels":25,"total_plaque_volume_proxy_mm3":4.0},
      {"artery":"Left main","voxel_volume_mm3":np.nan,"low_attenuation_mm3":0.0,"noncalcified_mm3":0.0,"mixed_intermediate_mm3":0.0,"calcified_mm3":54.0,"strict_nnunet_core_volume_mm3":0.0,"candidate_region_voxels":0,"total_plaque_volume_proxy_mm3":54.0},
    ])
    p=build_plaque_estimates(d)
    assert np.isclose(p[p.vessel=="LAD"].iloc[0].ncpv_best_estimate_mm3,9.0)
    assert p[p.vessel=="LM"].iloc[0].confirm2_tpv_stage==">0-250 mm3"
    w=build_whole_heart(p).iloc[0]
    assert np.isclose(w.tpv_best_estimate_mm3,82.0)
    return {"ok":True}
