from __future__ import annotations

import base64
import math
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from .artery_detection import detect_artery_series
from .cpr_tracking import path_tube, straighten, track_frame
from .study import OpenPlaqueStudy

VESSELS = ("LAD", "RCA", "LCX")
FALLBACK_SERIES = {"RCA": 1035, "LCX": 1039, "LAD": 1043}


def _plaque_mask_from_nifti(path: Path, reference_image, expected_mm3: float | None = None):
    """Load a cached plaque mask only if geometry and expected volume agree."""
    if not path.exists():
        return None
    img = sitk.ReadImage(str(path))
    if img.GetSize() != reference_image.GetSize():
        return None
    if not np.allclose(img.GetSpacing(), reference_image.GetSpacing(), atol=1e-5):
        return None
    arr = sitk.GetArrayFromImage(img)
    mask = arr > 0
    if expected_mm3 is not None:
        voxel_mm3 = float(np.prod(reference_image.GetSpacing()))
        actual_mm3 = float(mask.sum() * voxel_mm3)
        # Exact segmentation caches should agree to within about one voxel.
        if abs(actual_mm3 - float(expected_mm3)) > max(0.75, 1.1 * voxel_mm3):
            return None
    return mask


def _save_mask(mask: np.ndarray, reference_image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(path))


def _ensure_model(ROOT: Path):
    os.environ["nnUNet_raw"] = "/content/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/content/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "/content/nnUNet_results"
    for d in (os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    target = Path("/content/nnUNet_results/Dataset001_CCTA_DHM")
    if target.exists():
        return
    model_zip = ROOT / "models" / "Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip"
    if not model_zip.exists():
        raise FileNotFoundError(model_zip)
    with zipfile.ZipFile(model_zip) as z:
        z.extractall("/content/nnUNet_results")


def _load_or_compute_plaque(ROOT: Path, study, series_map, metrics: pd.DataFrame):
    """Reuse exact canonical plaque masks; only run nnU-Net on a cache miss."""
    cache_dir = ROOT / "Cache" / "canonical_plaque_v2"
    legacy_dirs = [ROOT / "Segmentations", ROOT / "segmentations"]
    result = {}
    cache_rows = []

    for vessel in VESSELS:
        image, volume, _ = study.load_series(series_map[vessel])
        expected = float(metrics.loc[metrics.vessel == vessel, "canonical_refined_tpv_mm3"].iloc[0])
        candidates = [cache_dir / f"{vessel}_canonical_refined.nii.gz"]
        for d in legacy_dirs:
            candidates.extend([
                d / f"{vessel}_refined_plaque_segmentation.nii.gz",
                d / f"{vessel}_volume_refined_plaque_segmentation.nii.gz",
            ])
        mask = None
        used = None
        for p in candidates:
            mask = _plaque_mask_from_nifti(p, image, expected_mm3=expected)
            if mask is not None:
                used = p
                break

        if mask is None:
            print(f"{vessel}: no validated canonical cache; computing segmentation once.")
            _ensure_model(ROOT)
            from .boundary import refine_plaque_mask
            from .segmentation import segment_vessel

            report = segment_vessel(image, volume, vessel)
            refined = refine_plaque_mask(
                volume=report.volume,
                mask=report.mask,
                spacing=report.mask_image.GetSpacing(),
                remove_small=True,
                min_component_voxels=10,
                trim_lumen_adjacent=True,
                lumen_distance_voxels=1,
                erode_core=False,
                high_hu_threshold=None,
                low_hu_threshold=None,
            )
            mask = refined.refined_mask == 2
            used = cache_dir / f"{vessel}_canonical_refined.nii.gz"
            _save_mask(mask, image, used)
            source = "computed_and_cached"
        else:
            source = "cache"
            print(f"{vessel}: reused cached plaque mask: {used}")

        result[vessel] = {"image": image, "volume": np.asarray(volume), "plaque": mask}
        cache_rows.append({"vessel": vessel, "source": source, "path": str(used)})

    return result, pd.DataFrame(cache_rows)


def _copy_first(existing_dirs, names, dest):
    for d in existing_dirs:
        for name in names:
            p = d / name
            if p.exists():
                shutil.copyfile(p, dest)
                return p
    return None


def run_tracking_report(root: str | Path = "/content/drive/MyDrive/OpenPlaque"):
    ROOT = Path(root)
    OUT = ROOT / "Image_Driven_Coronary_Tracking_Report"
    OUT.mkdir(parents=True, exist_ok=True)

    drive_zip = ROOT / "Full_DICOM.zip"
    local_zip = Path("/content/Full_DICOM.zip")
    if not drive_zip.exists():
        raise FileNotFoundError(drive_zip)
    if not local_zip.exists() or local_zip.stat().st_size != drive_zip.stat().st_size:
        shutil.copyfile(drive_zip, local_zip)

    # Do not delete the extraction directory: repeated cells in one runtime reuse it.
    study = OpenPlaqueStudy(str(local_zip), extract_root="/content/full_dicom_tracking")
    series_map, _ = detect_artery_series(study, fallback_series=FALLBACK_SERIES, return_candidates=True)

    metrics_dir = ROOT / "Combined_TPV_PCAT_All_Metrics_v2"
    tpv = pd.read_csv(metrics_dir / "tpv_metrics_by_vessel_v2.csv")
    pcat = pd.read_csv(metrics_dir / "pcat_canonical_primary_v2.csv")
    ps = pd.read_csv(metrics_dir / "pcat_circular_sensitivity_v2.csv")

    data, cache_qc = _load_or_compute_plaque(ROOT, study, series_map, tpv)
    cache_qc.to_csv(OUT / "cache_qc.csv", index=False)
    print(cache_qc.to_string(index=False))

    all_candidates, selected, qc_rows = {}, {}, []
    for vessel in VESSELS:
        dset = data[vessel]
        image, volume, plaque3 = dset["image"], dset["volume"], dset["plaque"]
        spacing = image.GetSpacing()
        sp_yx = (float(spacing[1]), float(spacing[0]))
        rows = []
        for z in range(volume.shape[0]):
            img = np.asarray(volume[z], float)
            tr = track_frame(img, sp_yx)
            if tr is None:
                continue
            tube, path_mask = path_tube(tr["path"], img.shape, sp_yx, radius_mm=5.0)
            tr.update(
                frame=int(z), tube=tube, path_mask=path_mask,
                plaque_tube_voxels=int((plaque3[z] & tube).sum()),
                plaque_frame_voxels=int(plaque3[z].sum()),
            )
            rows.append(tr)
        if not rows:
            raise RuntimeError(f"No image-driven path found for {vessel}")
        rows.sort(key=lambda x: x["image_score"], reverse=True)
        top = rows[: min(5, len(rows))]
        max_image = max(x["image_score"] for x in top)
        max_plaque = max(x["plaque_tube_voxels"] for x in top)
        for x in top:
            x["selection_score"] = x["image_score"] / max(max_image, 1e-9) + 0.10 * (
                x["plaque_tube_voxels"] / max(max_plaque, 1) if max_plaque else 0.0
            )
        win = max(top, key=lambda x: x["selection_score"])
        all_candidates[vessel], selected[vessel] = rows, win
        qc_rows.append({
            "vessel": vessel,
            "selected_frame": win["frame"],
            "image_score": win["image_score"],
            "selection_score": win["selection_score"],
            "path_length_mm": win["path_length_mm"],
            "mean_path_hu": win["mean_path_hu"],
            "mean_evidence": win["mean_evidence"],
            "threshold_percentile": win["threshold_percentile"],
            "plaque_voxels_in_selected_frame": win["plaque_frame_voxels"],
            "plaque_voxels_within_5mm_tube": win["plaque_tube_voxels"],
            "tube_capture_pct_of_selected_frame": 100 * win["plaque_tube_voxels"] / max(1, win["plaque_frame_voxels"]),
            "tracking_status": "OK" if win["path_length_mm"] >= 25 and 100 <= win["mean_path_hu"] <= 800 else "REVIEW",
        })
    tracking_qc = pd.DataFrame(qc_rows)
    tracking_qc.to_csv(OUT / "tracking_qc.csv", index=False)
    print(tracking_qc.to_string(index=False))

    fig, axs = plt.subplots(3, 3, figsize=(15, 14))
    for row, vessel in enumerate(VESSELS):
        volume, plaque3 = data[vessel]["volume"], data[vessel]["plaque"]
        for col, d in enumerate(all_candidates[vessel][:3]):
            ax = axs[row, col]
            z = d["frame"]
            img = np.asarray(volume[z], float)
            ax.imshow(img, cmap="gray", vmin=-150, vmax=750)
            p = d["path"]
            ax.plot(p[:, 1], p[:, 0], linewidth=1.8)
            plaque = plaque3[z] & d["tube"]
            if np.any(plaque):
                ax.imshow(np.ma.masked_where(~plaque, plaque), cmap="autumn", alpha=.65, interpolation="nearest")
            ax.set_title(f"{vessel} candidate {col+1}: frame {z}\npath {d['path_length_mm']:.1f} mm; HU {d['mean_path_hu']:.0f}; plaque {d['plaque_tube_voxels']} px")
            ax.axis("off")
    fig.suptitle("Image-driven coronary tracking candidates — path from CT image, not vessel mask", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, .97])
    plt.savefig(OUT / "01_tracking_candidates.png", dpi=180, bbox_inches="tight")
    plt.show(); plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(18, 14), gridspec_kw={"width_ratios": [1, 1.65]})
    along_rows = []
    for row, vessel in enumerate(VESSELS):
        d = selected[vessel]
        image, volume, plaque3 = data[vessel]["image"], data[vessel]["volume"], data[vessel]["plaque"]
        z = d["frame"]; img = np.asarray(volume[z], float); plaque = plaque3[z] & d["tube"]; p = d["path"]
        ax = axs[row, 0]; ax.imshow(img, cmap="gray", vmin=-150, vmax=750); ax.plot(p[:, 1], p[:, 0], linewidth=2)
        if np.any(plaque):
            ax.imshow(np.ma.masked_where(~plaque, plaque), cmap="autumn", alpha=.65, interpolation="nearest")
        y0=max(0,int(np.floor(p[:,0].min()))-45); y1=min(img.shape[0],int(np.ceil(p[:,0].max()))+46)
        x0=max(0,int(np.floor(p[:,1].min()))-45); x1=min(img.shape[1],int(np.ceil(p[:,1].max()))+46)
        ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.set_title(f"{vessel} selected CPR frame {z}\ntracked path + plaque within 5 mm"); ax.axis("off")

        spacing=image.GetSpacing(); sp_yx=(float(spacing[1]),float(spacing[0])); st=straighten(img, plaque, p, sp_yx)
        ax2=axs[row,1]
        if st is None:
            ax2.text(.5,.5,"Straightening failed",ha="center",va="center"); ax2.axis("off"); continue
        s,off,strip,plaque_strip=st
        ax2.imshow(strip,cmap="gray",vmin=-150,vmax=750,aspect="auto",origin="lower",extent=[s[0],s[-1],off[0],off[-1]])
        if np.any(plaque_strip):
            ax2.imshow(np.ma.masked_where(~plaque_strip,plaque_strip),cmap="autumn",alpha=.70,aspect="auto",origin="lower",extent=[s[0],s[-1],off[0],off[-1]],interpolation="nearest")
        ax2.axhline(0,linewidth=1); ax2.set_xlabel("Distance along tracked CPR path (mm)"); ax2.set_ylabel("Transverse distance (mm)"); ax2.set_title(f"{vessel} straightened vessel-centered display")
        bins=np.arange(0,max(1,math.ceil(s[-1]))+1,1.0); center_idx=int(np.argmin(np.abs(off)))
        for a,b in zip(bins[:-1],bins[1:]):
            jj=(s>=a)&(s<b)
            along_rows.append({"vessel":vessel,"start_mm":a,"end_mm":b,"display_plaque_pixels":int(plaque_strip[:,jj].sum()) if np.any(jj) else 0,"mean_centerline_hu":float(np.nanmean(strip[center_idx,jj])) if np.any(jj) else np.nan})
    fig.suptitle("OpenPlaque image-driven straightened coronary plaque roadmaps — visualization only",fontsize=16)
    plt.tight_layout(rect=[0,0,1,.97]); plt.savefig(OUT/"02_straightened_coronary_roadmaps.png",dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig)
    pd.DataFrame(along_rows).to_csv(OUT/"plaque_along_tracked_path.csv",index=False)

    previous_dirs=[ROOT/"Coronary_CPR_Roadmap_Report",ROOT/"User_Friendly_Plaque_PCAT_Report_v3",ROOT/"User_Friendly_Plaque_PCAT_Report_v2"]
    _copy_first(previous_dirs,("02_rca_pcat_cross_sections.png","02_rca_pcat_cross_sections_v3.png","02_rca_pcat_cross_sections_v2.png"),OUT/"03_rca_pcat_cross_sections.png")
    _copy_first(previous_dirs,("03_rca_longitudinal_pcat_ribbon.png","03_rca_longitudinal_pcat_ribbon_v3.png","03_rca_longitudinal_pcat_ribbon_v2.png"),OUT/"04_rca_longitudinal_pcat_ribbon.png")

    total=tpv[tpv.vessel=="TOTAL"].iloc[0]; primary=pcat.iloc[0]; cmin=float(ps.pcat_mean_hu.min()); cmax=float(ps.pcat_mean_hu.max()); directional=-94.345186; fmin=min(cmin,directional); fmax=max(cmax,directional)
    fig=plt.figure(figsize=(14,9)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15]); ax1=fig.add_subplot(gs[0,0]); vv=tpv[tpv.vessel!="TOTAL"]
    ax1.bar(vv.vessel,vv.canonical_refined_tpv_mm3); ax1.set_ylabel("Refined TPV (mm³)"); ax1.set_title("Canonical plaque volume by artery")
    ax2=fig.add_subplot(gs[0,1]); ax2.axis("off"); ax2.text(.03,.95,f"Total refined TPV: {total.canonical_refined_tpv_mm3:.0f} mm³\nRaw TPV: {total.raw_tpv_mm3:.0f} mm³\nTPV sensitivity: {total.sensitivity_min_mm3:.0f}–{total.sensitivity_max_mm3:.0f} mm³\n\nRCA 10–50 mm PCAT: {primary.pcat_mean_hu:.2f} HU\nCircular-margin range: {cmin:.2f} to {cmax:.2f} HU\nFull tested geometry: {fmin:.2f} to {fmax:.2f} HU",va="top",fontsize=13)
    ax3=fig.add_subplot(gs[1,:]); ax3.axis("off"); cols=["vessel","selected_frame","path_length_mm","mean_path_hu","plaque_voxels_within_5mm_tube","tracking_status"]; table_df=tracking_qc[cols].copy(); table_df["path_length_mm"]=table_df["path_length_mm"].map(lambda x:f"{x:.1f}"); table_df["mean_path_hu"]=table_df["mean_path_hu"].map(lambda x:f"{x:.0f}")
    t=ax3.table(cellText=table_df.values,colLabels=["Vessel","CPR frame","Tracked length mm","Mean path HU","Displayed plaque px","QC"],loc="center",cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1,1.6); ax3.set_title("Image-driven CPR tracking QC — display only",pad=12)
    fig.suptitle("OpenPlaque summary: quantitative endpoints + image-driven visualization QC",fontsize=16); plt.tight_layout(rect=[0,0,1,.96]); plt.savefig(OUT/"05_summary_dashboard.png",dpi=180,bbox_inches="tight"); plt.show(); plt.close(fig)

    summary=pd.DataFrame([{"canonical_total_tpv_mm3":float(total.canonical_refined_tpv_mm3),"raw_total_tpv_mm3":float(total.raw_tpv_mm3),"tpv_sensitivity_min_mm3":float(total.sensitivity_min_mm3),"tpv_sensitivity_max_mm3":float(total.sensitivity_max_mm3),"rca_pcat_mean_hu":float(primary.pcat_mean_hu),"pcat_full_geometry_min_hu":fmin,"pcat_full_geometry_max_hu":fmax}]); summary.to_csv(OUT/"tracking_report_summary.csv",index=False)

    imgs=["01_tracking_candidates.png","02_straightened_coronary_roadmaps.png","03_rca_pcat_cross_sections.png","04_rca_longitudinal_pcat_ribbon.png","05_summary_dashboard.png"]
    def img_tag(name):
        p=OUT/name
        if not p.exists(): return ""
        return f'<h2>{name}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%;height:auto">'
    html="<html><head><meta charset='utf-8'><title>OpenPlaque image-driven coronary tracking</title></head><body><h1>OpenPlaque — image-driven coronary tracking</h1><p><b>Research use only.</b> Tracking is derived from CPR image intensity/vesselness. Canonical TPV is unchanged. CPR plaque views and source-volume PCAT are not spatially co-registered.</p>"+"".join(img_tag(x) for x in imgs)+"<h2>Tracking QC</h2>"+tracking_qc.to_html(index=False)+"<h2>Cache provenance</h2>"+cache_qc.to_html(index=False)+"</body></html>"
    (OUT/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html").write_text(html,encoding="utf-8")
    zip_path=OUT/"OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip"; files=["tracking_qc.csv","cache_qc.csv","plaque_along_tracked_path.csv","tracking_report_summary.csv","OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html"]+[x for x in imgs if (OUT/x).exists()]
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(OUT/name,arcname=name)
    print("Report ZIP:",zip_path)
    print("Drive search: https://drive.google.com/drive/u/0/search?q=OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip")
    return {"tracking_qc":tracking_qc,"cache_qc":cache_qc,"output_dir":OUT,"zip_path":zip_path}
