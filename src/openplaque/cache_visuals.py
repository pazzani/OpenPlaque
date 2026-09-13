from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from .cache_utils import hash_array, hash_bytes, hash_df, load_json, save_json
from .tracking_report import VESSELS


class CacheVisualMixin:
    def _track_signature(self):
        if self.tracking_qc is None:
            self.track_coronaries()
        text = "|".join(
            f"{v}:{int(self.selected[v]['frame'])}:"
            f"{hash_array(np.asarray(self.selected[v]['path'], np.float32))}"
            for v in VESSELS
        )
        return hash_bytes(text.encode())

    def plot_straightened_roadmaps(self):
        if "roadmaps" in self._done:
            return self.out / "02_straightened_coronary_roadmaps.png"

        path = self.out / "02_straightened_coronary_roadmaps.png"
        csv_path = self.out / "plaque_along_tracked_path.csv"
        meta = self._paths()["roadmaps"]
        signature = self._track_signature()

        if (
            self.reuse["roadmaps"]
            and path.exists()
            and csv_path.exists()
            and meta.exists()
        ):
            try:
                if load_json(meta)["tracking_signature"] == signature:
                    self.along_df = pd.read_csv(csv_path)
                    self._record("roadmaps", "reused", path)
                    self._done.add("roadmaps")
                    return path
            except Exception:
                pass

        output = super().plot_straightened_roadmaps()
        save_json({"tracking_signature": signature}, meta)
        self._record("roadmaps", "recomputed_and_cached", output)
        self._done.add("roadmaps")
        return output

    def _generate_pcat(self, cross, ribbon):
        if self.study is None:
            self.prepare_inputs()

        base = self.root / "PCAT_RCA_10_50"
        cl = pd.read_csv(base / "rca_centerline_smoothed_zyx.csv")
        rad = pd.read_csv(base / "pcat_local_radius_profile.csv")
        aorta_path = next(
            (
                p
                for p in [
                    self.root
                    / "RCA_Ostium_TotalSegmentator"
                    / "aorta_series7_totalseg.nii.gz",
                    self.root
                    / "TotalSegmentator_Validation_v2"
                    / "aorta_series7_totalseg.nii.gz",
                ]
                if p.exists()
            ),
            None,
        )
        longitudinal_path = next(
            (
                p
                for p in [
                    self.root
                    / "Combined_TPV_PCAT_All_Metrics_v2"
                    / "pcat_canonical_longitudinal_v2.csv",
                    self.root
                    / "PCAT_RCA_10_50_Reproducibility_Lock"
                    / "pcat_canonical_primary_longitudinal.csv",
                ]
                if p.exists()
            ),
            None,
        )
        if aorta_path is None or longitudinal_path is None:
            raise FileNotFoundError("Missing frozen RCA PCAT inputs")

        source_img, ct, _ = self.study.load_series(7)
        ct = np.asarray(ct, float)
        sp = np.asarray(source_img.GetSpacing(), float)[::-1]
        arc = cl.arc_mm.to_numpy(float)
        pts = cl[["z", "y", "x"]].to_numpy(float)
        pts_mm = pts * sp
        lumen = np.interp(
            arc, rad.arc_mm.to_numpy(float), rad.lumen_radius_mm.to_numpy(float)
        )

        ai = sitk.ReadImage(str(aorta_path))
        if ai.GetSize() != source_img.GetSize() or not np.allclose(
            ai.GetSpacing(), source_img.GetSpacing()
        ):
            ai = sitk.Resample(
                ai,
                source_img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        aorta = sitk.GetArrayFromImage(ai).astype(float)

        def basis(tangent):
            tangent = tangent / np.linalg.norm(tangent)
            ref = (
                np.array([1.0, 0.0, 0.0])
                if abs(tangent[0]) < 0.85
                else np.array([0.0, 1.0, 0.0])
            )
            u = np.cross(tangent, ref)
            u /= np.linalg.norm(u)
            v = np.cross(tangent, u)
            v /= np.linalg.norm(v)
            return u, v

        def plane(target, half=8.0, pix=0.2):
            i = int(np.argmin(np.abs(arc - target)))
            i0 = max(0, i - 2)
            i1 = min(len(arc) - 1, i + 2)
            u, v = basis(pts_mm[i1] - pts_mm[i0])
            coord = np.arange(-half, half + 1e-9, pix)
            U, V = np.meshgrid(coord, coord, indexing="xy")
            pos = pts_mm[i][None, None, :] + U[..., None] * u + V[..., None] * v
            vox = (pos / sp).reshape(-1, 3).T
            img = map_coordinates(ct, vox, order=1, mode="nearest").reshape(U.shape)
            am = (
                map_coordinates(aorta, vox, order=0, mode="nearest").reshape(U.shape)
                > 0.5
            )
            rr = np.sqrt(U**2 + V**2)
            lum = float(lumen[i])
            outer = lum + 0.75
            shell = 3 * outer
            fat = (
                (rr > outer)
                & (rr <= shell)
                & (~am)
                & (img >= -190)
                & (img <= -30)
            )
            return float(arc[i]), img, fat, lum, outer, shell, [-half, half, -half, half]

        targets = [10, 20, 30, 40, 50]
        planes = [plane(target) for target in targets]
        fig, axs = plt.subplots(1, 5, figsize=(20, 4.5))
        im = None
        for ax, result, target in zip(axs, planes, targets):
            actual, img, fat, lum, outer, shell, extent = result
            ax.imshow(
                img, cmap="gray", vmin=-200, vmax=800, extent=extent, origin="lower"
            )
            im = ax.imshow(
                np.ma.masked_where(~fat, img),
                cmap="coolwarm",
                vmin=-120,
                vmax=-60,
                alpha=0.8,
                extent=extent,
                origin="lower",
            )
            for radius, linestyle in [(lum, "-"), (outer, "--"), (shell, ":")]:
                ax.add_patch(
                    plt.Circle(
                        (0, 0),
                        radius,
                        fill=False,
                        linestyle=linestyle,
                        linewidth=1.7,
                    )
                )
            ax.plot(0, 0, "+", markersize=9)
            ax.set_title(f"RCA {target} mm\nactual {actual:.1f} mm")
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_aspect("equal")
        axs[0].set_ylabel("mm")
        fig.suptitle(
            "RCA artery-centered PCAT: solid=lumen, dashed=outer wall, dotted=shell edge",
            fontsize=13,
        )
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.78, pad=0.02).set_label(
            "PCAT attenuation (HU)"
        )
        cross.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cross, dpi=180, bbox_inches="tight")
        plt.close(fig)

        long_df = pd.read_csv(longitudinal_path)
        x = (
            long_df.arc_start_mm.to_numpy(float)
            + long_df.arc_end_mm.to_numpy(float)
        ) / 2
        y = long_df.mean_hu.to_numpy(float)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        sc = ax.scatter(
            x,
            np.zeros_like(x),
            c=y,
            cmap="coolwarm",
            vmin=-110,
            vmax=-75,
            s=260,
            marker="s",
        )
        ax.set_xlim(10, 50)
        ax.set_yticks([])
        ax.set_xlabel("Distance from RCA ostium (mm)")
        ax.set_title("RCA longitudinal PCAT attenuation ribbon")
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("Mean HU per 1-mm segment")
        plt.tight_layout()
        plt.savefig(ribbon, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def reuse_pcat_outputs(self):
        if "pcat_figures" in self._done:
            return {
                "cross_sections": self.out / "03_rca_pcat_cross_sections.png",
                "ribbon": self.out / "04_rca_longitudinal_pcat_ribbon.png",
            }

        cache = self._paths()["pcat_figures"]
        cache.mkdir(parents=True, exist_ok=True)
        cross = cache / "cross.png"
        ribbon = cache / "ribbon.png"

        if self.reuse["pcat_figures"] and cross.exists() and ribbon.exists():
            action = "reused"
        else:
            self._generate_pcat(cross, ribbon)
            action = "recomputed_and_cached"

        out_cross = self.out / "03_rca_pcat_cross_sections.png"
        out_ribbon = self.out / "04_rca_longitudinal_pcat_ribbon.png"
        shutil.copyfile(cross, out_cross)
        shutil.copyfile(ribbon, out_ribbon)
        self._record("pcat_figures", action, cache)
        self._done.add("pcat_figures")
        return {"cross_sections": out_cross, "ribbon": out_ribbon}

    def _dashboard_signature(self):
        if self.tracking_qc is None:
            self.track_coronaries()
        return hash_bytes(
            "|".join(
                [
                    hash_df(self.tracking_qc),
                    hash_df(self.tpv),
                    hash_df(self.pcat),
                    hash_df(self.ps),
                ]
            ).encode()
        )

    def plot_dashboard(self):
        if "dashboard" in self._done:
            return self.out / "05_summary_dashboard.png"

        meta = self._paths()["dashboard"]
        img = self.out / "05_summary_dashboard.png"
        summary = self.out / "tracking_report_summary.csv"
        signature = self._dashboard_signature()

        if (
            self.reuse["dashboard"]
            and meta.exists()
            and img.exists()
            and summary.exists()
        ):
            try:
                if load_json(meta)["signature"] == signature:
                    self._record("dashboard", "reused", img)
                    self._done.add("dashboard")
                    return img
            except Exception:
                pass

        output = super().plot_dashboard()
        save_json({"signature": signature}, meta)
        self._record("dashboard", "recomputed_and_cached", output)
        self._done.add("dashboard")
        return output

    def _report_signature(self):
        names = [
            "01_tracking_candidates.png",
            "02_straightened_coronary_roadmaps.png",
            "03_rca_pcat_cross_sections.png",
            "04_rca_longitudinal_pcat_ribbon.png",
            "05_summary_dashboard.png",
            "tracking_qc.csv",
            "cache_qc.csv",
            "plaque_along_tracked_path.csv",
            "tracking_report_summary.csv",
        ]
        bits = []
        for name in names:
            path = self.out / name
            if path.exists():
                bits.append(f"{name}:{hash_bytes(path.read_bytes())}")
            else:
                bits.append(f"{name}:missing")
        return hash_bytes("|".join(bits).encode())

    def package_report(self):
        if not (self.out / "01_tracking_candidates.png").exists():
            self.plot_candidates()
        if self.along_df is None:
            self.plot_straightened_roadmaps()
        self.reuse_pcat_outputs()
        self.plot_dashboard()

        meta = self._paths()["report_package"]
        zip_path = self.out / "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip"
        html = self.out / "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT.html"
        signature = self._report_signature()

        if (
            self.reuse["report_package"]
            and meta.exists()
            and zip_path.exists()
            and html.exists()
        ):
            try:
                if load_json(meta)["signature"] == signature:
                    self._record("report_package", "reused", zip_path)
                    return zip_path
            except Exception:
                pass

        output = super().package_report()
        save_json({"signature": signature}, meta)
        self._record("report_package", "recomputed_and_cached", output)
        return output
