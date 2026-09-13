from __future__ import annotations

import base64
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artery_detection import detect_artery_series
from .cpr_tracking import path_tube, straighten, track_frame
from .study import OpenPlaqueStudy
from .tracking_report import FALLBACK_SERIES, VESSELS, _copy_first, _load_or_compute_plaque


class TrackingWorkflow:
    """Stateful, stepwise workflow for the Colab image-driven tracking notebook.

    Each public method corresponds to one notebook step. Expensive state is kept in
    memory between cells and canonical plaque masks are loaded from validated cache
    whenever possible.
    """

    def __init__(self, root: str | Path = "/content/drive/MyDrive/OpenPlaque"):
        self.root = Path(root)
        self.out = self.root / "Image_Driven_Coronary_Tracking_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.study = None
        self.series_map = None
        self.tpv = None
        self.pcat = None
        self.ps = None
        self.data = None
        self.cache_qc = None
        self.all_candidates = None
        self.selected = None
        self.tracking_qc = None
        self.along_df = None

    def prepare_inputs(self):
        """Prepare DICOM access and load previously computed quantitative metrics."""
        drive_zip = self.root / "Full_DICOM.zip"
        local_zip = Path("/content/Full_DICOM.zip")
        if not drive_zip.exists():
            raise FileNotFoundError(drive_zip)
        if not local_zip.exists() or local_zip.stat().st_size != drive_zip.stat().st_size:
            shutil.copyfile(drive_zip, local_zip)

        # Keep the extraction directory so repeated notebook cells reuse it.
        self.study = OpenPlaqueStudy(str(local_zip), extract_root="/content/full_dicom_tracking")
        self.series_map, _ = detect_artery_series(
            self.study, fallback_series=FALLBACK_SERIES, return_candidates=True
        )

        metrics_dir = self.root / "Combined_TPV_PCAT_All_Metrics_v2"
        self.tpv = pd.read_csv(metrics_dir / "tpv_metrics_by_vessel_v2.csv")
        self.pcat = pd.read_csv(metrics_dir / "pcat_canonical_primary_v2.csv")
        self.ps = pd.read_csv(metrics_dir / "pcat_circular_sensitivity_v2.csv")
        return self.series_map

    def load_cached_plaque(self):
        """Load validated plaque caches; run nnU-Net only for a genuine cache miss."""
        if self.study is None:
            self.prepare_inputs()
        self.data, self.cache_qc = _load_or_compute_plaque(
            self.root, self.study, self.series_map, self.tpv
        )
        self.cache_qc.to_csv(self.out / "cache_qc.csv", index=False)
        return self.cache_qc

    def track_coronaries(self):
        """Run image-driven path tracking on all CPR rotations and select one per vessel."""
        if self.data is None:
            self.load_cached_plaque()

        all_candidates, selected, qc_rows = {}, {}, []
        for vessel in VESSELS:
            dset = self.data[vessel]
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
                    frame=int(z),
                    tube=tube,
                    path_mask=path_mask,
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
            qc_rows.append(
                {
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
                    "tube_capture_pct_of_selected_frame": 100
                    * win["plaque_tube_voxels"]
                    / max(1, win["plaque_frame_voxels"]),
                    "tracking_status": "OK"
                    if win["path_length_mm"] >= 25 and 100 <= win["mean_path_hu"] <= 800
                    else "REVIEW",
                }
            )

        self.all_candidates = all_candidates
        self.selected = selected
        self.tracking_qc = pd.DataFrame(qc_rows)
        self.tracking_qc.to_csv(self.out / "tracking_qc.csv", index=False)
        return self.tracking_qc

    def plot_candidates(self):
        """Show the top three image-driven candidate tracks for each artery."""
        if self.all_candidates is None:
            self.track_coronaries()

        fig, axs = plt.subplots(3, 3, figsize=(15, 14))
        for row, vessel in enumerate(VESSELS):
            volume = self.data[vessel]["volume"]
            plaque3 = self.data[vessel]["plaque"]
            for col in range(3):
                ax = axs[row, col]
                cands = self.all_candidates[vessel]
                if col >= len(cands):
                    ax.axis("off")
                    continue
                d = cands[col]
                z = d["frame"]
                img = np.asarray(volume[z], float)
                ax.imshow(img, cmap="gray", vmin=-150, vmax=750)
                p = d["path"]
                ax.plot(p[:, 1], p[:, 0], linewidth=1.8)
                plaque = plaque3[z] & d["tube"]
                if np.any(plaque):
                    ax.imshow(
                        np.ma.masked_where(~plaque, plaque),
                        cmap="autumn",
                        alpha=.65,
                        interpolation="nearest",
                    )
                ax.set_title(
                    f"{vessel} candidate {col + 1}: frame {z}\n"
                    f"path {d['path_length_mm']:.1f} mm; HU {d['mean_path_hu']:.0f}; "
                    f"plaque {d['plaque_tube_voxels']} px"
                )
                ax.axis("off")

        fig.suptitle(
            "Image-driven coronary tracking candidates — path from CT image, not vessel mask",
            fontsize=15,
        )
        plt.tight_layout(rect=[0, 0, 1, .97])
        path = self.out / "01_tracking_candidates.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return path

    def plot_straightened_roadmaps(self):
        """Create selected CPR views and straightened vessel-centered plaque strips."""
        if self.selected is None:
            self.track_coronaries()

        fig, axs = plt.subplots(
            3, 2, figsize=(18, 14), gridspec_kw={"width_ratios": [1, 1.65]}
        )
        along_rows = []
        for row, vessel in enumerate(VESSELS):
            d = self.selected[vessel]
            image = self.data[vessel]["image"]
            volume = self.data[vessel]["volume"]
            plaque3 = self.data[vessel]["plaque"]
            z = d["frame"]
            img = np.asarray(volume[z], float)
            plaque = plaque3[z] & d["tube"]
            p = d["path"]

            ax = axs[row, 0]
            ax.imshow(img, cmap="gray", vmin=-150, vmax=750)
            ax.plot(p[:, 1], p[:, 0], linewidth=2)
            if np.any(plaque):
                ax.imshow(
                    np.ma.masked_where(~plaque, plaque),
                    cmap="autumn",
                    alpha=.65,
                    interpolation="nearest",
                )
            y0 = max(0, int(np.floor(p[:, 0].min())) - 45)
            y1 = min(img.shape[0], int(np.ceil(p[:, 0].max())) + 46)
            x0 = max(0, int(np.floor(p[:, 1].min())) - 45)
            x1 = min(img.shape[1], int(np.ceil(p[:, 1].max())) + 46)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_title(f"{vessel} selected CPR frame {z}\ntracked path + plaque within 5 mm")
            ax.axis("off")

            spacing = image.GetSpacing()
            sp_yx = (float(spacing[1]), float(spacing[0]))
            st = straighten(img, plaque, p, sp_yx)
            ax2 = axs[row, 1]
            if st is None:
                ax2.text(.5, .5, "Straightening failed", ha="center", va="center")
                ax2.axis("off")
                continue

            s, off, strip, plaque_strip = st
            ax2.imshow(
                strip,
                cmap="gray",
                vmin=-150,
                vmax=750,
                aspect="auto",
                origin="lower",
                extent=[s[0], s[-1], off[0], off[-1]],
            )
            if np.any(plaque_strip):
                ax2.imshow(
                    np.ma.masked_where(~plaque_strip, plaque_strip),
                    cmap="autumn",
                    alpha=.70,
                    aspect="auto",
                    origin="lower",
                    extent=[s[0], s[-1], off[0], off[-1]],
                    interpolation="nearest",
                )
            ax2.axhline(0, linewidth=1)
            ax2.set_xlabel("Distance along tracked CPR path (mm)")
            ax2.set_ylabel("Transverse distance (mm)")
            ax2.set_title(f"{vessel} straightened vessel-centered display")

            bins = np.arange(0, max(1, math.ceil(s[-1])) + 1, 1.0)
            center_idx = int(np.argmin(np.abs(off)))
            for a, b in zip(bins[:-1], bins[1:]):
                jj = (s >= a) & (s < b)
                along_rows.append(
                    {
                        "vessel": vessel,
                        "start_mm": a,
                        "end_mm": b,
                        "display_plaque_pixels": int(plaque_strip[:, jj].sum())
                        if np.any(jj)
                        else 0,
                        "mean_centerline_hu": float(np.nanmean(strip[center_idx, jj]))
                        if np.any(jj)
                        else np.nan,
                    }
                )

        fig.suptitle(
            "OpenPlaque image-driven straightened coronary plaque roadmaps — visualization only",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, .97])
        path = self.out / "02_straightened_coronary_roadmaps.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        self.along_df = pd.DataFrame(along_rows)
        self.along_df.to_csv(self.out / "plaque_along_tracked_path.csv", index=False)
        return path

    def reuse_pcat_outputs(self):
        """Copy the already validated PCAT figures instead of recomputing PCAT."""
        previous_dirs = [
            self.root / "Coronary_CPR_Roadmap_Report",
            self.root / "User_Friendly_Plaque_PCAT_Report_v3",
            self.root / "User_Friendly_Plaque_PCAT_Report_v2",
        ]
        cross = _copy_first(
            previous_dirs,
            (
                "02_rca_pcat_cross_sections.png",
                "02_rca_pcat_cross_sections_v3.png",
                "02_rca_pcat_cross_sections_v2.png",
            ),
            self.out / "03_rca_pcat_cross_sections.png",
        )
        ribbon = _copy_first(
            previous_dirs,
            (
                "03_rca_longitudinal_pcat_ribbon.png",
                "03_rca_longitudinal_pcat_ribbon_v3.png",
                "03_rca_longitudinal_pcat_ribbon_v2.png",
            ),
            self.out / "04_rca_longitudinal_pcat_ribbon.png",
        )
        return {"cross_sections": cross, "ribbon": ribbon}

    def plot_dashboard(self):
        """Build the final summary dashboard from cached quantitative endpoints and tracking QC."""
        if self.tracking_qc is None:
            self.track_coronaries()

        total = self.tpv[self.tpv.vessel == "TOTAL"].iloc[0]
        primary = self.pcat.iloc[0]
        cmin = float(self.ps.pcat_mean_hu.min())
        cmax = float(self.ps.pcat_mean_hu.max())
        directional = -94.345186
        fmin = min(cmin, directional)
        fmax = max(cmax, directional)

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        ax1 = fig.add_subplot(gs[0, 0])
        vv = self.tpv[self.tpv.vessel != "TOTAL"]
        ax1.bar(vv.vessel, vv.canonical_refined_tpv_mm3)
        ax1.set_ylabel("Refined TPV (mm³)")
        ax1.set_title("Canonical plaque volume by artery")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis("off")
        ax2.text(
            .03,
            .95,
            f"Total refined TPV: {total.canonical_refined_tpv_mm3:.0f} mm³\n"
            f"Raw TPV: {total.raw_tpv_mm3:.0f} mm³\n"
            f"TPV sensitivity: {total.sensitivity_min_mm3:.0f}–{total.sensitivity_max_mm3:.0f} mm³\n\n"
            f"RCA 10–50 mm PCAT: {primary.pcat_mean_hu:.2f} HU\n"
            f"Circular-margin range: {cmin:.2f} to {cmax:.2f} HU\n"
            f"Full tested geometry: {fmin:.2f} to {fmax:.2f} HU",
            va="top",
            fontsize=13,
        )

        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")
        cols = [
            "vessel",
            "selected_frame",
            "path_length_mm",
            "mean_path_hu",
            "plaque_voxels_within_5mm_tube",
            "tracking_status",
        ]
        table_df = self.tracking_qc[cols].copy()
        table_df["path_length_mm"] = table_df["path_length_mm"].map(lambda x: f"{x:.1f}")
        table_df["mean_path_hu"] = table_df["mean_path_hu"].map(lambda x: f"{x:.0f}")
        table = ax3.table(
            cellText=table_df.values,
            colLabels=[
                "Vessel",
                "CPR frame",
                "Tracked length mm",
                "Mean path HU",
                "Displayed plaque px",
                "QC",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)
        ax3.set_title("Image-driven CPR tracking QC — display only", pad=12)

        fig.suptitle(
            "OpenPlaque summary: quantitative endpoints + image-driven visualization QC",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, .96])
        path = self.out / "05_summary_dashboard.png"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        summary = pd.DataFrame(
            [
                {
                    "canonical_total_tpv_mm3": float(total.canonical_refined_tpv_mm3),
                    "raw_total_tpv_mm3": float(total.raw_tpv_mm3),
                    "tpv_sensitivity_min_mm3": float(total.sensitivity_min_mm3),
                    "tpv_sensitivity_max_mm3": float(total.sensitivity_max_mm3),
                    "rca_pcat_mean_hu": float(primary.pcat_mean_hu),
                    "pcat_full_geometry_min_hu": fmin,
                    "pcat_full_geometry_max_hu": fmax,
                }
            ]
        )
        summary.to_csv(self.out / "tracking_report_summary.csv", index=False)
        return path

    def package_report(self):
        """Create self-contained HTML and ZIP report-back bundle."""
        if self.tracking_qc is None:
            self.track_coronaries()
        if self.cache_qc is None:
            self.load_cached_plaque()

        imgs = [
            "01_tracking_candidates.png",
            "02_straightened_coronary_roadmaps.png",
            "03_rca_pcat_cross_sections.png",
            "04_rca_longitudinal_pcat_ribbon.png",
            "05_summary_dashboard.png",
        ]

        def img_tag(name):
            p = self.out / name
            if not p.exists():
                return ""
            encoded = base64.b64encode(p.read_bytes()).decode()
            return (
                f'<h2>{name}</h2><img src="data:image/png;base64,{encoded}" '
                'style="max-width:100%;height:auto">'
            )

        html = (
            "<html><head><meta charset='utf-8'><title>OpenPlaque image-driven coronary tracking</title></head><body>"
            "<h1>OpenPlaque — image-driven coronary tracking</h1>"
            "<p><b>Research use only.</b> Tracking is derived from CPR image intensity/vesselness. "
            "Canonical TPV is unchanged. CPR plaque views and source-volume PCAT are not spatially co-registered.</p>"
            + "".join(img_tag(x) for x in imgs)
            + "<h2>Tracking QC</h2>"
            + self.tracking_qc.to_html(index=False)
            + "<h2>Cache provenance</h2>"
            + self.cache_qc.to_html(index=False)
            + "</body></html>"
        )
        html_path = self.out / "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html"
        html_path.write_text(html, encoding="utf-8")

        zip_path = self.out / "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip"
        files = [
            "tracking_qc.csv",
            "cache_qc.csv",
            "plaque_along_tracked_path.csv",
            "tracking_report_summary.csv",
            "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html",
        ] + [x for x in imgs if (self.out / x).exists()]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                p = self.out / name
                if p.exists():
                    z.write(p, arcname=name)
        return zip_path
