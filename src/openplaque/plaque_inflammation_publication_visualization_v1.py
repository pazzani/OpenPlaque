from __future__ import annotations

import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGORITHM="plaque-inflammation-publication-visualization-v1.0"
OUTPUT_DIRNAME="Plaque_Inflammation_Publication_Visualization_v1"

RCA_LONG=Path("RCA_Plaque_PCAT_Research_Lock_v1/RCA_locked_research_plaque_PCAT_profile_10_50.csv")
LAD_LONG=Path("LAD_Source_Space_PCAT_Feasibility_v1/frozen_LAD_PCAT_longitudinal.csv")
LCX_LONG=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/C6_PCAT_longitudinal_primary.csv")
C7_LONG=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/C7_PCAT_longitudinal_primary.csv")
RCA_RADIAL=Path("Combined_TPV_PCAT_All_Metrics_v2/pcat_canonical_radial_v2.csv")

BANDS=[
    (-190.0,-150.0,"Very low attenuation\n-190 to <-150 HU"),
    (-150.0,-110.0,"Low attenuation\n-150 to <-110 HU"),
    (-110.0,-70.0,"Intermediate\n-110 to <-70 HU"),
    (-70.0,-30.0,"Higher attenuation\n-70 to -30 HU"),
]

def _req(p):
    p=Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _write_json(p,x):
    Path(p).write_text(json.dumps(x,indent=2,default=str,allow_nan=True),encoding="utf-8")

def _weighted_mean(df):
    x=pd.to_numeric(df["mean_hu"],errors="coerce")
    w=pd.to_numeric(df["fat_voxels"],errors="coerce")
    ok=x.notna() & w.notna() & (w>0)
    return float(np.average(x[ok],weights=w[ok])) if ok.any() else np.nan

def _normalize_long(df,vessel):
    d=df.copy()
    for c in ["arc_start_mm","arc_end_mm","fat_voxels","mean_hu"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d[np.isfinite(d.mean_hu) & np.isfinite(d.fat_voxels) & (d.fat_voxels>0)].copy()
    d["vessel"]=vessel
    d["local_start_mm"]=d.arc_start_mm-d.arc_start_mm.min()
    d["local_mid_mm"]=(d.arc_start_mm+d.arc_end_mm)/2-d.arc_start_mm.min()
    return d

def build_longitudinal_tables(root):
    rca=_normalize_long(pd.read_csv(_req(root/RCA_LONG)),"RCA")
    lad=_normalize_long(pd.read_csv(_req(root/LAD_LONG)),"LAD")
    lcx=_normalize_long(pd.read_csv(_req(root/LCX_LONG)),"LCX")
    c7=_normalize_long(pd.read_csv(_req(root/C7_LONG)),"C7/OM-like alternate")
    return rca,lad,lcx,c7

def build_band_table(long_tables):
    rows=[]
    for vessel,df in long_tables.items():
        total=float(df.fat_voxels.sum())
        for lo,hi,label in BANDS:
            if hi==-30:
                m=(df.mean_hu>=lo)&(df.mean_hu<=hi)
            else:
                m=(df.mean_hu>=lo)&(df.mean_hu<hi)
            w=float(df.loc[m,"fat_voxels"].sum())
            rows.append({
                "vessel":vessel,
                "band":label,
                "band_lo_hu":lo,
                "band_hi_hu":hi,
                "fat_voxels_weight":w,
                "fraction_of_fat_voxels":w/total if total>0 else np.nan,
                "method":"fat-voxel-weighted classification of local longitudinal-bin mean HU",
            })
    return pd.DataFrame(rows)

def _plaque_chart(plaque,out):
    order=["LAD","RCA","LCX","LM"]
    d=plaque.copy()
    d["vessel"]=d["vessel"].astype(str).str.upper()
    d=d.set_index("vessel").reindex(order).reset_index()
    comps=[
        ("lap_best_estimate_mm3","Necrotic core / LAP\n-30 to <30 HU"),
        ("noncalcified_30_130_mm3","Fibro-fatty\n30 to <130 HU"),
        ("fibrous_intermediate_130_350_mm3","Fibrous/intermediate\n130 to <350 HU"),
        ("calcified_plaque_volume_mm3","Dense calcium\n>=350 HU"),
    ]
    fig=plt.figure(figsize=(13.5,8))
    ax=fig.add_axes([0.08,0.12,0.58,0.78])
    lg=fig.add_axes([0.69,0.14,0.29,0.72]); lg.axis("off")
    x=np.arange(len(order)); bottom=np.zeros(len(order)); totals=np.zeros(len(order))
    handles=[]
    for col,label in comps:
        vals=pd.to_numeric(d[col],errors="coerce").fillna(0).to_numpy(float)
        b=ax.bar(x,vals,bottom=bottom,width=.52,edgecolor="black",linewidth=1,label=label)
        handles.append(b[0])
        for i,(v,btm) in enumerate(zip(vals,bottom)):
            if v>=1:
                ax.text(i,btm+v/2,f"{v:.1f}",ha="center",va="center",fontsize=10)
        bottom+=vals; totals+=vals
    for i,t in enumerate(totals):
        if t>0:
            suffix="*" if order[i]=="LM" else ""
            ax.text(i,t+max(totals)*.012,f"Total {t:.1f}{suffix}",ha="center",va="bottom",fontweight="bold")
    ax.set_xticks(x,order); ax.set_xlabel("Artery"); ax.set_ylabel("Plaque volume (mm³)")
    ax.set_title("Best-estimate plaque composition by artery")
    ax.grid(axis="y",alpha=.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lg.text(0,1,"Plaque attenuation category\n(HU threshold and descriptive interpretation)",va="top",fontsize=14,fontweight="bold")
    y=.86
    notes=[
        "Very low-density / lipid-rich-like plaque",
        "Lower-density noncalcified plaque",
        "Higher-density noncalcified plaque",
        "Densely calcified plaque",
    ]
    for h,(_,label),note in zip(handles,comps,notes):
        lg.legend([h],[label],loc="upper left",bbox_to_anchor=(-.02,y),frameon=False,fontsize=11)
        lg.text(.02,y-.08,note,transform=lg.transAxes,fontsize=10)
        y-=.19
    lg.text(0,.06,"* LM is a 54 mm³ calcium-volume anchor only.\nLM noncalcified plaque is unknown and is not imputed.",fontsize=10)
    p=out/"01_plaque_composition_publication_style.png"
    fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p

def _pcat_mean_bar(long_tables,out):
    vessels=["RCA","LAD","LCX"]
    means=[_weighted_mean(long_tables[v]) for v in vessels]
    lengths=[float(long_tables[v].arc_end_mm.max()-long_tables[v].arc_start_mm.min()) for v in vessels]
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.bar(vessels,means,edgecolor="black")
    ax.set_ylabel("Direct PCAT mean attenuation (HU)")
    ax.set_title("Raw PCAT attenuation by vessel\n(not proprietary Caristo FAI-Score)")
    ax.grid(axis="y",alpha=.15)
    for b,m,l in zip(bars,means,lengths):
        ax.text(b.get_x()+b.get_width()/2,m,f"{m:.1f} HU\n{l:.1f} mm available",ha="center",va="bottom")
    p=out/"02_inflammation_mean_pcat_bar.png"
    fig.tight_layout(); fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p

def _pcat_band_bar(band,out):
    order=["RCA","LAD","LCX"]
    labels=[b[2] for b in BANDS]
    pivot=band.pivot(index="vessel",columns="band",values="fraction_of_fat_voxels").reindex(order)[labels]
    fig=plt.figure(figsize=(13,7))
    ax=fig.add_axes([0.08,0.12,0.57,0.78])
    lg=fig.add_axes([0.69,0.14,0.29,0.72]); lg.axis("off")
    x=np.arange(len(order)); bottom=np.zeros(len(order)); handles=[]
    for label in labels:
        vals=pivot[label].fillna(0).to_numpy(float)*100
        b=ax.bar(x,vals,bottom=bottom,width=.55,edgecolor="black",linewidth=1,label=label)
        handles.append(b[0])
        for i,(v,btm) in enumerate(zip(vals,bottom)):
            if v>=5:
                ax.text(i,btm+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=10)
        bottom+=vals
    ax.set_xticks(x,order); ax.set_ylim(0,100)
    ax.set_ylabel("Share of PCAT fat voxels (%)")
    ax.set_title("Local PCAT attenuation-band decomposition\nweighted by fat voxels in measured longitudinal bins")
    ax.grid(axis="y",alpha=.15); ax.spines[["top","right"]].set_visible(False)
    lg.text(0,1,"Local mean PCAT band",va="top",fontsize=14,fontweight="bold")
    lg.legend(handles,labels,loc="upper left",bbox_to_anchor=(-.02,.91),frameon=False,fontsize=10)
    lg.text(0,.28,"Interpretation",fontsize=11,fontweight="bold")
    lg.text(0,.23,"Progressively less-negative attenuation is\nin the inflammation-associated direction.",fontsize=10)
    lg.text(0,.10,"This classifies 1-mm local mean PCAT bins,\nweighted by their fat-voxel counts. It is not\na voxel-level histologic decomposition and is\nnot Caristo FAI-Score.",fontsize=9)
    p=out/"03_inflammation_attenuation_band_decomposition.png"
    fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p

def _longitudinal_profiles(long_tables,out):
    fig,ax=plt.subplots(figsize=(11,5.5))
    for vessel in ["RCA","LAD","LCX"]:
        d=long_tables[vessel]
        ax.plot(d.local_mid_mm,d.mean_hu,marker="o",markersize=3,label=vessel)
    ax.set_xlabel("Distance along available measured segment (mm)")
    ax.set_ylabel("Local PCAT mean attenuation (HU)")
    ax.set_title("Longitudinal PCAT profiles")
    ax.grid(alpha=.15); ax.legend()
    p=out/"04_inflammation_longitudinal_profiles.png"
    fig.tight_layout(); fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p

def _heatmap(long_tables,out):
    order=["RCA","LAD","LCX"]
    n=40
    mat=np.full((3,n),np.nan)
    for i,v in enumerate(order):
        vals=long_tables[v].sort_values("local_mid_mm").mean_hu.to_numpy(float)
        mat[i,:min(n,len(vals))]=vals[:n]
    fig,ax=plt.subplots(figsize=(12,3.2))
    im=ax.imshow(mat,aspect="auto",vmin=-115,vmax=-60)
    ax.set_yticks(range(3),order)
    ax.set_xticks(np.arange(0,40,5),[f"{x}-{x+1}" for x in range(0,40,5)])
    ax.set_xlabel("Successive 1-mm local PCAT bins; blank = not measured")
    ax.set_title("Coverage-aware longitudinal PCAT heatmap")
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("Local mean HU")
    p=out/"05_inflammation_longitudinal_heatmap.png"
    fig.tight_layout(); fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p

def _rca_radial(root,out):
    d=pd.read_csv(_req(root/RCA_RADIAL))
    for c in ["radial_start_mm","radial_end_mm","fat_voxels","mean_hu"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d[np.isfinite(d.mean_hu)].copy()
    d["radial_mid_mm"]=(d.radial_start_mm+d.radial_end_mm)/2
    fig,ax=plt.subplots(figsize=(8.5,5))
    ax.plot(d.radial_mid_mm,d.mean_hu,marker="o")
    ax.set_xlabel("Radial distance in cached RCA PCAT profile (mm)")
    ax.set_ylabel("PCAT mean attenuation (HU)")
    ax.set_title("RCA radial PCAT attenuation gradient")
    ax.grid(alpha=.15)
    p=out/"06_rca_radial_pcat_gradient.png"
    fig.tight_layout(); fig.savefig(p,dpi=190,bbox_inches="tight"); plt.close(fig)
    return p,d

def write_report(out,plaque,aggregate,infl,band,radial):
    imgs=[
        "01_plaque_composition_publication_style.png",
        "02_inflammation_mean_pcat_bar.png",
        "03_inflammation_attenuation_band_decomposition.png",
        "04_inflammation_longitudinal_profiles.png",
        "05_inflammation_longitudinal_heatmap.png",
        "06_rca_radial_pcat_gradient.png",
    ]
    body="".join(f"<h2>{Path(x).stem}</h2><img src='{x}' width='1100'>" for x in imgs)
    html=f"""<html><head><meta charset='utf-8'><style>
body{{font-family:Arial,sans-serif;margin:30px;max-width:1400px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #ddd;padding:6px}}
th{{background:#f2f2f2}} .note{{background:#fff8e1;border:1px solid #e0c36c;padding:12px}}
</style></head><body>
<h1>OpenPlaque Plaque + Inflammation Publication Visualization v1</h1>
<p class='note'><b>Research use only.</b> Plaque volumes are OpenPlaque best-estimate proxies. Direct PCAT attenuation is not proprietary Caristo FAI-Score. The PCAT stacked bars classify local 1-mm mean-HU bins weighted by their fat-voxel counts; they are not voxel-level tissue classes.</p>
<h2>Plaque table</h2>{plaque.to_html(index=False)}
<h2>Major-vessel aggregate</h2>{aggregate.to_html(index=False)}
<h2>Inflammation summary</h2>{infl.to_html(index=False)}
<h2>PCAT attenuation bands</h2>{band.to_html(index=False)}
{body}
</body></html>"""
    p=out/"OPENPLAQUE_PLAQUE_INFLAMMATION_PUBLICATION_VISUALIZATION_V1_REPORT.html"
    p.write_text(html,encoding="utf-8")
    return p

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None,endpoint_dir=None):
    root=Path(drive_root)
    out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    endpoint=Path(endpoint_dir) if endpoint_dir else out/"endpoint"
    endpoint.mkdir(parents=True,exist_ok=True)

    from openplaque.plaque_inflammation_best_estimates_v1 import run as endpoint_run
    endpoint_summary=endpoint_run(drive_root=str(root),output_dir=str(endpoint))

    plaque=pd.read_csv(endpoint/"plaque_best_estimates_by_vessel.csv")
    aggregate=pd.read_csv(endpoint/"major_vessel_aggregate.csv")
    infl=pd.read_csv(endpoint/"inflammation_best_estimates_by_vessel.csv")

    rca,lad,lcx,c7=build_longitudinal_tables(root)
    long_tables={"RCA":rca,"LAD":lad,"LCX":lcx}
    all_long=pd.concat([rca,lad,lcx,c7],ignore_index=True)
    all_long.to_csv(out/"pcat_longitudinal_bins_used.csv",index=False)

    band=build_band_table(long_tables)
    band.to_csv(out/"pcat_local_mean_band_decomposition.csv",index=False)

    figs=[]
    figs.append(_plaque_chart(plaque,out))
    figs.append(_pcat_mean_bar(long_tables,out))
    figs.append(_pcat_band_bar(band,out))
    figs.append(_longitudinal_profiles(long_tables,out))
    figs.append(_heatmap(long_tables,out))
    radial_fig,radial=_rca_radial(root,out)
    figs.append(radial_fig)
    radial.to_csv(out/"rca_radial_pcat_profile_used.csv",index=False)

    report=write_report(out,plaque,aggregate,infl,band,radial)
    summary={
        "status":"COMPLETE",
        "algorithm":ALGORITHM,
        "endpoint_summary_status":endpoint_summary.get("status"),
        "figures":[p.name for p in figs],
        "report":report.name,
        "pcat_band_method":"fat-voxel-weighted classification of local longitudinal-bin mean HU; not voxel-level tissue decomposition",
        "scientific_boundaries":{
            "plaque":"OpenPlaque research best-estimate proxy, not Cleerly output.",
            "pcat":"Direct raw PCAT attenuation, not proprietary Caristo FAI-Score.",
            "lm_inflammation":"Not standardized/not estimated.",
            "lcx":"C6 structural LCX-like parent is primary; C7/OM-like daughter retained only as alternate longitudinal data.",
        }
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"run_state.json",{"status":"COMPLETE","algorithm":ALGORITHM})

    archive=out/"OPENPLAQUE_PLAQUE_INFLAMMATION_PUBLICATION_VISUALIZATION_V1_RESULTS.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p!=archive:
                z.write(p,p.relative_to(out))
    return summary

def synthetic_self_test():
    d=pd.DataFrame({
        "arc_start_mm":[0,1,2,3],
        "arc_end_mm":[1,2,3,4],
        "fat_voxels":[10,20,30,40],
        "mean_hu":[-160,-130,-90,-50],
    })
    n=_normalize_long(d,"X")
    b=build_band_table({"X":n})
    assert np.isclose(b.fraction_of_fat_voxels.sum(),1.0)
    assert len(b)==4
    return {"ok":True}
