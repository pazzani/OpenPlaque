from __future__ import annotations

import json, math, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="plaque-inflammation-final-visualization-v2.0"
OUTPUT_DIRNAME="Plaque_Inflammation_Final_Visualization_v2"

SOURCE_CACHE=Path("Cache/Secondary_3D_Vesselness_Topology_v1")
AORTA=Path("RCA_Ostium_TotalSegmentator/aorta_series7_totalseg.nii.gz")

RCA_CL=Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
RCA_RAD=Path("PCAT_RCA_10_50/pcat_local_radius_profile.csv")
RCA_LOCK_SUMMARY=Path("RCA_Plaque_PCAT_Research_Lock_v1/summary.json")
RCA_LONG=Path("RCA_Plaque_PCAT_Research_Lock_v1/RCA_locked_research_plaque_PCAT_profile_10_50.csv")

LAD_GEOM=Path("LAD_Source_Space_PCAT_Feasibility_v1/LAD_PCAT_source_geometry.csv")
LAD_SUMMARY=Path("LAD_Source_Space_PCAT_Feasibility_v1/summary.json")
LAD_LONG=Path("LAD_Source_Space_PCAT_Feasibility_v1/frozen_LAD_PCAT_longitudinal.csv")

LCX_FREEZE=Path("LCX_Structural_Source_QC_Freeze_v1/LCX_structural_dense_source_QC.csv")
LCX_SUMMARY=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/summary.json")
LCX_LONG=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/C6_PCAT_longitudinal_primary.csv")
C7_LONG=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/C7_PCAT_longitudinal_primary.csv")

FAT_RANGE=(-190.0,-30.0)
WALL_MARGIN_MM=0.75
RCA_RANGE=(10.0,50.0)
LAD_RANGE=(1.0,24.0)
LCX_BIFURCATION_EXCLUSION_MM=1.0

MEAN_TOL_HU=0.10
FAT_VOXEL_REL_TOL=0.002

PCAT_BANDS=[
    (-190.0,-150.0,"Very low attenuation\n-190 to <-150 HU"),
    (-150.0,-110.0,"Low attenuation\n-150 to <-110 HU"),
    (-110.0,-70.0,"Intermediate attenuation\n-110 to <-70 HU"),
    (-70.0,-30.0,"Higher attenuation\n-70 to -30 HU"),
]


def _req(p):
    p=Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p,x):
    Path(p).write_text(json.dumps(x,indent=2,default=str,allow_nan=True),encoding="utf-8")


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx=np.asarray(meta["spacing_zyx"],float)
        self.spacing_xyz=self.spacing_zyx[::-1]
        self.origin=np.asarray(meta["positions_lps_mm"][0],float)
        iop=np.asarray(meta["image_orientation_patient"],float)
        row,col=iop[:3],iop[3:]
        slc=np.cross(row,col)
        self.D=np.array([[row[0],col[0],slc[0]],[row[1],col[1],slc[1]],[row[2],col[2],slc[2]]],float)
        self.invD=np.linalg.inv(self.D)

    def xyz_to_zyx(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float))
        xyz=((pts-self.origin)@self.invD.T)/self.spacing_xyz
        return xyz[:,::-1]

    def zyx_to_xyz(self,pts):
        pts=np.atleast_2d(np.asarray(pts,float))
        return self.origin+(pts[:,::-1]*self.spacing_xyz)@self.D.T


def _load_source(root):
    cache=root/SOURCE_CACHE
    src=np.load(_req(cache/"series7_int16.npy"),mmap_mode="r")
    meta=_read_json(cache/"series7_int16.json")
    geom=SourceGeometry(meta)
    return geom,src,meta


def _sitk_reference(shape_zyx,geom):
    ref=sitk.Image([int(x) for x in shape_zyx[::-1]],sitk.sitkUInt8)
    ref.SetSpacing(tuple(float(x) for x in geom.spacing_xyz))
    ref.SetOrigin(tuple(float(x) for x in geom.origin))
    ref.SetDirection(tuple(float(x) for x in geom.D.ravel()))
    return ref


def _load_aorta(root,src,geom):
    p=_req(root/AORTA)
    im=sitk.ReadImage(str(p))
    ref=_sitk_reference(src.shape,geom)
    same=(
        im.GetSize()==ref.GetSize()
        and np.allclose(im.GetSpacing(),ref.GetSpacing())
        and np.allclose(im.GetOrigin(),ref.GetOrigin())
        and np.allclose(im.GetDirection(),ref.GetDirection())
    )
    if not same:
        im=sitk.Resample(im,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    a=sitk.GetArrayFromImage(im)>0
    if a.shape!=src.shape:
        raise RuntimeError(f"Aorta shape mismatch {a.shape} vs {src.shape}")
    return a


def _generic_lps_voxels(geom,src,aorta,segment,arc_col,radius_col,max_margin=WALL_MARGIN_MM):
    centers=segment[["lps_x_mm","lps_y_mm","lps_z_mm"]].to_numpy(float)
    arcs=segment[arc_col].to_numpy(float)
    radii=segment[radius_col].to_numpy(float)
    max_outer=float(np.max(radii+max_margin))
    pad_mm=3.0*max_outer+2.0
    zyx=geom.xyz_to_zyx(centers)
    lo=np.floor(np.min(zyx,axis=0)-pad_mm/geom.spacing_zyx).astype(int)
    hi=np.ceil(np.max(zyx,axis=0)+pad_mm/geom.spacing_zyx).astype(int)+1
    lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(src.shape))
    crop=np.asarray(src[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]])
    zz,yy,xx=np.indices(crop.shape)
    gzyx=np.stack([zz+lo[0],yy+lo[1],xx+lo[2]],axis=-1).reshape(-1,3).astype(float)
    gxyz=geom.zyx_to_xyz(gzyx)
    tree=cKDTree(centers)
    dist,idx=tree.query(gxyz,k=1,workers=-1)
    hu=crop.reshape(-1).astype(float)
    af=aorta[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]].reshape(-1)
    outer=radii[idx]+WALL_MARGIN_MM
    shell=(dist>outer)&(dist<=3.0*outer)&(~af)
    fat=shell&(hu>=FAT_RANGE[0])&(hu<=FAT_RANGE[1])
    vals=hu[fat]
    return vals,{
        "arc_start_mm":float(arcs.min()),
        "arc_end_mm":float(arcs.max()),
        "station_count":int(len(segment)),
        "fat_voxels":int(len(vals)),
        "pcat_mean_hu":float(vals.mean()),
        "pcat_median_hu":float(np.median(vals)),
        "pcat_sd_hu":float(vals.std()),
    }


def _rca_voxels(root,src,aorta,geom):
    cl=pd.read_csv(_req(root/RCA_CL))
    rad=pd.read_csv(_req(root/RCA_RAD))
    arc=cl.arc_mm.to_numpy(float)
    pts_zyx=cl[["z","y","x"]].to_numpy(float)
    pts_mm=pts_zyx*geom.spacing_zyx
    lumen=np.interp(arc,rad.arc_mm.to_numpy(float),rad.lumen_radius_mm.to_numpy(float))
    keep=(arc>=RCA_RANGE[0])&(arc<=RCA_RANGE[1])
    arcs=arc[keep]; pts=pts_zyx[keep]; ptsmm=pts_mm[keep]; radii=lumen[keep]
    max_outer=float(np.max(radii+WALL_MARGIN_MM))
    pad_mm=3.0*max_outer+3.0
    lo=np.floor(np.min(pts,axis=0)-pad_mm/geom.spacing_zyx).astype(int)
    hi=np.ceil(np.max(pts,axis=0)+pad_mm/geom.spacing_zyx).astype(int)+1
    lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(src.shape))
    crop=np.asarray(src[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]])
    zz,yy,xx=np.indices(crop.shape)
    gzyx=np.stack([zz+lo[0],yy+lo[1],xx+lo[2]],axis=-1).reshape(-1,3).astype(float)
    gmm=gzyx*geom.spacing_zyx
    tree=cKDTree(ptsmm)
    dist,idx=tree.query(gmm,k=1,workers=-1)
    hu=crop.reshape(-1).astype(float)
    af=aorta[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]].reshape(-1)
    nearest_arc=arcs[idx]
    outer=radii[idx]+WALL_MARGIN_MM
    # IMPORTANT: the locked research endpoint is defined by the 40 one-mm
    # longitudinal bins [10,11), ... [49,50).  The canonical whole-shell
    # sampler also contains voxels assigned exactly to the 50-mm endpoint
    # station; those endpoint-cap voxels are intentionally excluded here.
    shell=(
        (dist>outer)
        &(dist<=3.0*outer)
        &(nearest_arc>=RCA_RANGE[0])
        &(nearest_arc<RCA_RANGE[1])
        &(~af)
    )
    fat=shell&(hu>=FAT_RANGE[0])&(hu<=FAT_RANGE[1])
    vals=hu[fat]
    radial_out=(dist-outer)[fat]
    return vals,{
        "arc_start_mm":RCA_RANGE[0],"arc_end_mm":RCA_RANGE[1],
        "station_count":int(len(arcs)),"fat_voxels":int(len(vals)),
        "pcat_mean_hu":float(vals.mean()),"pcat_median_hu":float(np.median(vals)),"pcat_sd_hu":float(vals.std()),
        "endpoint_convention":"half-open [10,50) mm to match locked 40-bin research endpoint"
    },radial_out


def _lad_segment(root):
    d=pd.read_csv(_req(root/LAD_GEOM))
    return d[
        (d.frozen_arc_mm>=LAD_RANGE[0])
        & (d.frozen_arc_mm<=LAD_RANGE[1])
        & d.station_qc_pass.astype(bool)
    ].copy().sort_values("frozen_arc_mm")


def _lcx_segment(root):
    d=pd.read_csv(_req(root/LCX_FREEZE))
    d=d[(d.vessel=="C6")&(d.post_split_arc_mm>=LCX_BIFURCATION_EXCLUSION_MM)].copy().sort_values("post_split_arc_mm")
    passed=d.station_qc_pass.astype(bool).to_numpy()
    fail=np.flatnonzero(~passed)
    if len(fail):
        d=d.iloc[:int(fail[0])].copy()
    if len(d)<2:
        raise RuntimeError("No usable C6 LCX-like PCAT segment")
    return d


def _locked_expectations(root):
    r=_read_json(root/RCA_LOCK_SUMMARY)["pcat"]
    l=_read_json(root/LAD_SUMMARY)["segments"]["frozen_LAD"]
    x=_read_json(root/LCX_SUMMARY)["pcat_primary"]["C6"]
    return {
        "RCA":{"mean":float(r["fat_voxel_weighted_mean_hu"]),"fat_voxels":int(r["fat_voxels_total"]),"length":40.0},
        "LAD":{"mean":float(l["pcat_mean_hu"]),"fat_voxels":int(l["fat_voxels"]),"length":float(l["segment_length_mm"])},
        "LCX":{"mean":float(x["pcat_mean_hu"]),"fat_voxels":int(x["fat_voxels"]),"length":float(x["segment_length_mm"])},
    }


def compute_voxel_distributions(root):
    geom,src,_=_load_source(root)
    aorta=_load_aorta(root,src,geom)
    vals={}
    meta={}
    vals["RCA"],meta["RCA"],rca_radial_out=_rca_voxels(root,src,aorta,geom)
    lad=_lad_segment(root)
    vals["LAD"],meta["LAD"]=_generic_lps_voxels(geom,src,aorta,lad,"frozen_arc_mm","lumen_radius_median_mm")
    lcx=_lcx_segment(root)
    vals["LCX"],meta["LCX"]=_generic_lps_voxels(geom,src,aorta,lcx,"post_split_arc_mm","lumen_radius_median_mm")

    expected=_locked_expectations(root)
    rows=[]
    for v in ("RCA","LAD","LCX"):
        obs=meta[v]; exp=expected[v]
        mean_delta=float(obs["pcat_mean_hu"]-exp["mean"])
        vox_delta=int(obs["fat_voxels"]-exp["fat_voxels"])
        rel=abs(vox_delta)/max(exp["fat_voxels"],1)
        passed=bool(abs(mean_delta)<=MEAN_TOL_HU and rel<=FAT_VOXEL_REL_TOL)
        rows.append({
            "vessel":v,
            "locked_mean_hu":exp["mean"],
            "recomputed_mean_hu":obs["pcat_mean_hu"],
            "mean_delta_hu":mean_delta,
            "locked_fat_voxels":exp["fat_voxels"],
            "recomputed_fat_voxels":obs["fat_voxels"],
            "fat_voxel_delta":vox_delta,
            "fat_voxel_relative_delta":rel,
            "locked_segment_length_mm":exp["length"],
            "voxel_distribution_validation_pass":passed,
        })
    validation=pd.DataFrame(rows)
    if not validation.voxel_distribution_validation_pass.all():
        raise RuntimeError("Voxel-level PCAT reconstruction failed validation against locked endpoints:\n"+validation.to_string(index=False))
    extras={"RCA_radial_out_mm":rca_radial_out}
    return vals,validation,expected,extras


def build_voxel_bands(vals):
    rows=[]
    for vessel,a in vals.items():
        n=len(a)
        for lo,hi,label in PCAT_BANDS:
            m=(a>=lo)&(a<(hi if hi!=-30 else hi+1e-9))
            rows.append({
                "vessel":vessel,"band":label,"band_lo_hu":lo,"band_hi_hu":hi,
                "fat_voxels":int(m.sum()),"fraction_of_fat_voxels":float(m.sum()/n),
            })
    return pd.DataFrame(rows)


def _plaque_chart(plaque,out):
    order=["LAD","RCA","LCX","LM"]
    d=plaque.copy(); d["vessel"]=d.vessel.str.upper(); d=d.set_index("vessel").reindex(order)
    comps=[
        ("lap_best_estimate_mm3","Necrotic core / LAP (-30 to <30 HU)","#d73027"),
        ("noncalcified_30_130_mm3","Fibro-fatty (30 to <130 HU)","#fdae61"),
        ("fibrous_intermediate_130_350_mm3","Fibrous/intermediate (130 to <350 HU)","#3288bd"),
        ("calcified_plaque_volume_mm3","Dense calcium (>=350 HU)","#f7f7f7"),
    ]
    fig,ax=plt.subplots(figsize=(12.5,7.5))
    x=np.arange(len(order)); bottom=np.zeros(len(order)); handles=[]
    for col,label,color in comps:
        val=pd.to_numeric(d[col],errors="coerce").fillna(0).to_numpy(float)
        b=ax.bar(x,val,bottom=bottom,width=.52,edgecolor="black",linewidth=1,color=color)
        handles.append(Patch(facecolor=color,edgecolor="black",label=label))
        for i,(v,bt) in enumerate(zip(val,bottom)):
            if v>=1:
                ax.text(i,bt+v/2,f"{v:.1f}",ha="center",va="center",fontsize=10)
        bottom+=val
    for i,t in enumerate(bottom):
        if t>0:
            ax.text(i,t+2.0,f"Total {t:.1f}"+("*" if order[i]=="LM" else ""),ha="center",va="bottom",fontweight="bold")
    ax.set_xticks(x,order); ax.set_ylabel("Plaque volume (mm³)"); ax.set_xlabel("Artery")
    ax.set_title("Best-estimate plaque composition by artery")
    ax.grid(axis="y",alpha=.15); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1.02,.98),frameon=False,title="Plaque attenuation category")
    ax.text(1.02,.30,"* LM is a 54 mm³ calcium-volume anchor only.\nLM noncalcified plaque is unknown and not imputed.",
            transform=ax.transAxes,fontsize=10,va="top")
    p=out/"01_plaque_composition_corrected.png"
    fig.tight_layout(); fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p


def _mean_pcat_chart(expected,out):
    vessels=["RCA","LAD","LCX"]
    means=[expected[v]["mean"] for v in vessels]
    lengths=[expected[v]["length"] for v in vessels]
    fig,ax=plt.subplots(figsize=(8,5.5))
    bars=ax.bar(vessels,means,edgecolor="black")
    ax.set_ylabel("Direct PCAT mean attenuation (HU)")
    ax.set_title("Locked raw PCAT attenuation by vessel\n(not proprietary Caristo FAI-Score)")
    ax.grid(axis="y",alpha=.15)
    for b,m,l in zip(bars,means,lengths):
        ax.text(b.get_x()+b.get_width()/2,m,f"{m:.1f} HU\n{l:.2f} mm",ha="center",va="bottom")
    p=out/"02_inflammation_mean_pcat_corrected.png"
    fig.tight_layout(); fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p


def _voxel_band_chart(band,out):
    order=["RCA","LAD","LCX"]; labels=[x[2] for x in PCAT_BANDS]
    pivot=band.pivot(index="vessel",columns="band",values="fraction_of_fat_voxels").reindex(order)[labels]
    fig,ax=plt.subplots(figsize=(12.5,7))
    x=np.arange(3); bottom=np.zeros(3)
    handles=[]
    for label in labels:
        val=pivot[label].to_numpy(float)*100
        b=ax.bar(x,val,bottom=bottom,width=.55,edgecolor="black",linewidth=1,label=label)
        handles.append(b[0])
        for i,(v,bt) in enumerate(zip(val,bottom)):
            if v>=4:
                ax.text(i,bt+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=10)
        bottom+=val
    ax.set_xticks(x,order); ax.set_ylim(0,100); ax.set_ylabel("Share of actual PCAT fat voxels (%)")
    ax.set_title("Voxel-level PCAT attenuation decomposition")
    ax.grid(axis="y",alpha=.15); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=handles,labels=labels,loc="upper left",bbox_to_anchor=(1.02,.98),frameon=False,title="Actual voxel HU")
    ax.text(1.02,.31,"Less-negative PCAT attenuation is in the\ninflammation-associated direction.\nThese are attenuation bands, not histologic classes\nand not Caristo FAI-Score.",
            transform=ax.transAxes,fontsize=10,va="top")
    p=out/"03_inflammation_voxel_hu_decomposition.png"
    fig.tight_layout(); fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p


def _voxel_histogram(vals,out):
    fig,ax=plt.subplots(figsize=(10,5.8))
    bins=np.arange(-190,-29,5)
    for v in ("RCA","LAD","LCX"):
        ax.hist(vals[v],bins=bins,density=True,histtype="step",linewidth=2,label=v)
    for x in (-150,-110,-70):
        ax.axvline(x,linestyle="--",linewidth=.8,alpha=.5)
    ax.set_xlim(-190,-30); ax.set_xlabel("PCAT voxel attenuation (HU)"); ax.set_ylabel("Density")
    ax.set_title("Source-space PCAT voxel attenuation distributions")
    ax.legend(); ax.grid(alpha=.12)
    p=out/"04_inflammation_voxel_hu_histograms.png"
    fig.tight_layout(); fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p


def _longitudinal(root,out):
    dfs={}
    for v,p in [("RCA",RCA_LONG),("LAD",LAD_LONG),("LCX",LCX_LONG)]:
        d=pd.read_csv(_req(root/p))
        for c in ("arc_start_mm","arc_end_mm","mean_hu","fat_voxels"):
            d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d[np.isfinite(d.mean_hu)].copy()
        d["local_mid_mm"]=(d.arc_start_mm+d.arc_end_mm)/2-d.arc_start_mm.min()
        dfs[v]=d
    fig,ax=plt.subplots(figsize=(11,5.5))
    for v,d in dfs.items():
        ax.plot(d.local_mid_mm,d.mean_hu,marker="o",markersize=3,label=v)
    ax.set_xlabel("Distance along available measured segment (mm)"); ax.set_ylabel("Local PCAT mean HU")
    ax.set_title("Longitudinal PCAT profiles"); ax.legend(); ax.grid(alpha=.15)
    p1=out/"05_inflammation_longitudinal_profiles.png"
    fig.tight_layout(); fig.savefig(p1,dpi=200,bbox_inches="tight"); plt.close(fig)

    mat=np.full((3,40),np.nan)
    for i,v in enumerate(("RCA","LAD","LCX")):
        a=dfs[v].sort_values("local_mid_mm").mean_hu.to_numpy(float)
        mat[i,:min(40,len(a))]=a[:40]
    fig,ax=plt.subplots(figsize=(12,3.2))
    im=ax.imshow(mat,aspect="auto",vmin=-115,vmax=-60)
    ax.set_yticks(range(3),["RCA","LAD","LCX"]); ax.set_xlabel("Successive 1-mm local bins; blank = not measured")
    ax.set_title("Coverage-aware longitudinal PCAT heatmap")
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("Local mean HU")
    p2=out/"06_inflammation_longitudinal_heatmap.png"
    fig.tight_layout(); fig.savefig(p2,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p1,p2,dfs


def _radial_locked(rca_vals,radial_out,out):
    vals=np.asarray(rca_vals,float)
    radial_out=np.asarray(radial_out,float)
    rows=[]
    for b in np.arange(0.0,6.0,0.5):
        m=(radial_out>=b)&(radial_out<b+0.5)
        v=vals[m]
        rows.append({
            "radial_start_mm":float(b),
            "radial_end_mm":float(b+0.5),
            "fat_voxels":int(len(v)),
            "mean_hu":float(np.mean(v)) if len(v) else np.nan,
        })
    d=pd.DataFrame(rows)
    q=d[np.isfinite(d.mean_hu)].copy()
    mid=(q.radial_start_mm+q.radial_end_mm)/2
    fig,ax=plt.subplots(figsize=(8.5,5))
    ax.plot(mid,q.mean_hu,marker="o")
    ax.set_xlabel("Distance outward from modeled RCA interface (mm)")
    ax.set_ylabel("PCAT mean HU")
    ax.set_title("Locked RCA radial PCAT gradient — exploratory geometry QC")
    ax.grid(alpha=.15)
    ax.text(.02,.02,"Uses the same half-open [10,50) mm voxel population as the locked RCA endpoint.\nDescriptive only; not a validated clinical inflammation gradient.",
            transform=ax.transAxes,fontsize=9,va="bottom")
    p=out/"07_rca_radial_pcat_gradient_descriptive.png"
    fig.tight_layout(); fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig)
    return p,d

def _report(out,plaque,aggregate,infl,validation,bands):
    imgs=[
        "01_plaque_composition_corrected.png",
        "02_inflammation_mean_pcat_corrected.png",
        "03_inflammation_voxel_hu_decomposition.png",
        "04_inflammation_voxel_hu_histograms.png",
        "05_inflammation_longitudinal_profiles.png",
        "06_inflammation_longitudinal_heatmap.png",
        "07_rca_radial_pcat_gradient_descriptive.png",
    ]
    html=f"""<html><head><meta charset='utf-8'><style>
body{{font-family:Arial,sans-serif;margin:30px;max-width:1450px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #ddd;padding:6px}}
th{{background:#f2f2f2}} .note{{padding:12px;background:#fff8e1;border:1px solid #d6b656}}
.pass{{padding:10px;background:#eef8ee;border:1px solid #9bbd9b}}
</style></head><body>
<h1>OpenPlaque Plaque + Inflammation Final Visualization v2</h1>
<p class='note'><b>Research use only.</b> Plaque values are OpenPlaque best-estimate proxies, not Cleerly outputs. Direct PCAT attenuation is not proprietary Caristo FAI-Score. LM inflammation is not standardized.</p>
<p class='pass'><b>Voxel-distribution validation:</b> source-space PCAT voxel reconstructions reproduced the locked RCA/LAD/C6 endpoints within prespecified tolerances.</p>
<h2>Plaque estimates</h2>{plaque.to_html(index=False)}
<h2>Major-vessel aggregate</h2>{aggregate.to_html(index=False)}
<h2>Inflammation endpoint</h2>{infl.to_html(index=False)}
<h2>Voxel reconstruction validation</h2>{validation.to_html(index=False)}
<h2>Voxel HU bands</h2>{bands.to_html(index=False)}
{''.join(f"<h2>{Path(x).stem}</h2><img src='{x}' width='1150'>" for x in imgs)}
</body></html>"""
    p=out/"OPENPLAQUE_PLAQUE_INFLAMMATION_FINAL_VISUALIZATION_V2_REPORT.html"
    p.write_text(html,encoding="utf-8")
    return p


def _endpoint_cache_ready(endpoint):
    required=[
        "plaque_best_estimates_by_vessel.csv",
        "major_vessel_aggregate.csv",
        "inflammation_best_estimates_by_vessel.csv",
        "summary.json",
        "run_state.json",
    ]
    if not all((endpoint/x).is_file() for x in required):
        return False
    try:
        state=_read_json(endpoint/"run_state.json")
        return state.get("status")=="COMPLETE"
    except Exception:
        return False


def _load_or_build_endpoint(root,out,use_cached_endpoint_values):
    endpoint=out/"endpoint"
    endpoint.mkdir(parents=True,exist_ok=True)
    reused=bool(use_cached_endpoint_values and _endpoint_cache_ready(endpoint))
    if not reused:
        from openplaque.plaque_inflammation_best_estimates_v1 import run as endpoint_run
        endpoint_run(drive_root=str(root),output_dir=str(endpoint))
    plaque=pd.read_csv(endpoint/"plaque_best_estimates_by_vessel.csv")
    aggregate=pd.read_csv(endpoint/"major_vessel_aggregate.csv")
    infl=pd.read_csv(endpoint/"inflammation_best_estimates_by_vessel.csv")
    return endpoint,plaque,aggregate,infl,reused


def _validate_cached_voxels(root,vals):
    expected=_locked_expectations(root)
    rows=[]
    for v in ("RCA","LAD","LCX"):
        if v not in vals:
            raise RuntimeError(f"Cached PCAT voxel archive is missing {v}")
        a=np.asarray(vals[v],float)
        obs_mean=float(np.mean(a))
        obs_n=int(len(a))
        exp=expected[v]
        mean_delta=float(obs_mean-exp["mean"])
        vox_delta=int(obs_n-exp["fat_voxels"])
        rel=abs(vox_delta)/max(exp["fat_voxels"],1)
        passed=bool(abs(mean_delta)<=MEAN_TOL_HU and rel<=FAT_VOXEL_REL_TOL)
        rows.append({
            "vessel":v,
            "locked_mean_hu":exp["mean"],
            "recomputed_mean_hu":obs_mean,
            "mean_delta_hu":mean_delta,
            "locked_fat_voxels":exp["fat_voxels"],
            "recomputed_fat_voxels":obs_n,
            "fat_voxel_delta":vox_delta,
            "fat_voxel_relative_delta":rel,
            "locked_segment_length_mm":exp["length"],
            "voxel_distribution_validation_pass":passed,
        })
    validation=pd.DataFrame(rows)
    if not validation.voxel_distribution_validation_pass.all():
        raise RuntimeError(
            "Cached voxel-level PCAT values no longer match locked endpoints:\n"
            +validation.to_string(index=False)
        )
    return validation,expected


def _load_or_build_voxel_cache(root,out,use_cached_pcat_voxels):
    values_path=out/"pcat_voxel_values.npz"
    extras_path=out/"pcat_voxel_geometry_extras.npz"
    can_reuse=bool(
        use_cached_pcat_voxels
        and values_path.is_file()
        and extras_path.is_file()
    )
    if can_reuse:
        with np.load(values_path) as z:
            vals={v:np.asarray(z[v],float) for v in ("RCA","LAD","LCX")}
        with np.load(extras_path) as z:
            extras={"RCA_radial_out_mm":np.asarray(z["RCA_radial_out_mm"],float)}
        validation,expected=_validate_cached_voxels(root,vals)
        if len(extras["RCA_radial_out_mm"])!=len(vals["RCA"]):
            raise RuntimeError("Cached RCA radial-distance array does not match cached RCA voxel count")
        reused=True
    else:
        vals,validation,expected,extras=compute_voxel_distributions(root)
        np.savez_compressed(values_path,**vals)
        np.savez_compressed(extras_path,**extras)
        reused=False
    validation.to_csv(out/"pcat_voxel_reconstruction_validation.csv",index=False)
    return vals,validation,expected,extras,reused


def run(
    drive_root="/content/drive/MyDrive/OpenPlaque",
    output_dir=None,
    use_cached_endpoint_values=True,
    use_cached_pcat_voxels=True,
):
    root=Path(drive_root)
    out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write_json(out/"run_state.json",{
        "status":"RUNNING",
        "algorithm":ALGORITHM,
        "baseline":BASELINE,
        "use_cached_endpoint_values":bool(use_cached_endpoint_values),
        "use_cached_pcat_voxels":bool(use_cached_pcat_voxels),
    })

    endpoint,plaque,aggregate,infl,endpoint_cache_reused=_load_or_build_endpoint(
        root,out,bool(use_cached_endpoint_values)
    )

    vals,validation,expected,extras,pcat_voxel_cache_reused=_load_or_build_voxel_cache(
        root,out,bool(use_cached_pcat_voxels)
    )
    bands=build_voxel_bands(vals)
    bands.to_csv(out/"pcat_voxel_hu_band_decomposition.csv",index=False)

    figures=[]
    figures.append(_plaque_chart(plaque,out))
    figures.append(_mean_pcat_chart(expected,out))
    figures.append(_voxel_band_chart(bands,out))
    figures.append(_voxel_histogram(vals,out))
    p1,p2,longitudinal=_longitudinal(root,out); figures.extend([p1,p2])
    p3,radial=_radial_locked(vals["RCA"],extras["RCA_radial_out_mm"],out); figures.append(p3)

    all_long=[]
    for v,d in longitudinal.items():
        q=d.copy(); q["vessel"]=v; all_long.append(q)
    pd.concat(all_long,ignore_index=True).to_csv(out/"pcat_longitudinal_profiles_used.csv",index=False)
    radial.to_csv(out/"rca_locked_radial_profile_used.csv",index=False)

    report=_report(out,plaque,aggregate,infl,validation,bands)
    summary={
        "status":"COMPLETE","algorithm":ALGORITHM,"baseline_commit":BASELINE,
        "figures":[p.name for p in figures],
        "voxel_validation_pass":bool(validation.voxel_distribution_validation_pass.all()),
        "voxel_validation":validation.to_dict("records"),
        "cache_usage":{
            "endpoint_values_requested":bool(use_cached_endpoint_values),
            "endpoint_values_reused":bool(endpoint_cache_reused),
            "pcat_voxels_requested":bool(use_cached_pcat_voxels),
            "pcat_voxels_reused":bool(pcat_voxel_cache_reused),
        },
        "scientific_boundaries":{
            "plaque":"OpenPlaque research best-estimate proxy; not Cleerly output.",
            "pcat":"Direct -190 to -30 HU PCAT; not proprietary Caristo FAI-Score.",
            "voxel_bands":"True source-space PCAT fat-voxel HU distribution; descriptive attenuation bands, not histology.",
            "rca_endpoint_convention":"Half-open [10,50) mm, matching the locked 40 longitudinal one-mm bins and excluding endpoint-cap voxels assigned exactly to 50 mm.",
            "lm":"LM plaque 54 mm3 is calcium anchor; LM inflammation not standardized.",
            "lcx":"C6 LCX-like structural parent used as current low-confidence clinical-LCX proxy."
        }
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"run_state.json",{
        "status":"COMPLETE",
        "algorithm":ALGORITHM,
        "result_status":"PLAQUE_INFLAMMATION_FINAL_VISUALIZATION_COMPLETE",
        "endpoint_values_reused":bool(endpoint_cache_reused),
        "pcat_voxels_reused":bool(pcat_voxel_cache_reused),
    })

    zpath=out/"OPENPLAQUE_PLAQUE_INFLAMMATION_FINAL_VISUALIZATION_V2_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file() and p!=zpath:
                z.write(p,p.relative_to(out))
    return summary


def synthetic_self_test():
    vals={"RCA":np.array([-170.,-130.,-90.,-50.]),"LAD":np.array([-160.,-120.,-80.,-60.]),"LCX":np.array([-155.,-115.,-75.,-45.])}
    b=build_voxel_bands(vals)
    assert len(b)==12
    for v in vals:
        assert np.isclose(b[b.vessel==v].fraction_of_fat_voxels.sum(),1.0)
    return {"ok":True}
