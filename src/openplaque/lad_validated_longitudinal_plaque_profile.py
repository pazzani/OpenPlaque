from __future__ import annotations

"""Longitudinal plaque characterization restricted to the validated LAD segment.

This workflow intentionally does NOT report anatomical plaque volume. The available
LAD curved series is a non-spatial rotation stack and canonical registration supports
only longitudinal mapping, not reliable angular/source-space mapping. Therefore all
outputs are longitudinal consensus-support metrics within the independently accepted
24.996653-mm LAD segment.

Research use only. Not clinically validated. Not for diagnosis.
"""

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree

from openplaque.study import OpenPlaqueStudy

ALGORITHM_VERSION = "lad-validated-longitudinal-plaque-profile-v1.0"
LAD_SERIES = 1043
ACCEPTED_LAD_TARGET_MM = 24.996653027945402
HU_THRESHOLDS = {
    "low_attenuation": (-np.inf, 30.0),
    "noncalcified": (30.0, 130.0),
    "mixed_intermediate": (130.0, 350.0),
    "calcified": (350.0, np.inf),
}


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_write(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _find_one(root: Path, name: str):
    hits = list(Path(root).rglob(name))
    if not hits:
        return None
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _arc_zyx(points, spacing_zyx):
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.zeros(len(p), float)
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def _source_spacing(root: Path):
    candidates = [
        root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1" / "series7_geometry.json",
        root / "Cache" / "Coronary_Anatomy_Reconciliation_v1" / "series7_dicom_geometry.json",
        root / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.json",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            meta = _json_read(fp)
            sp = meta.get("spacing_zyx")
            if sp is not None and len(sp) == 3:
                return np.asarray(sp, float), fp
        except Exception:
            pass
    raise FileNotFoundError("Could not resolve verified Series-7 spacing metadata")


def build_consensus_from_fold_masks(root: Path):
    """Build LAD consensus/disagreement from five existing fold masks only."""
    out_root = root / "GPU_Plaque_5Fold_Ensemble_v1" / "LAD"
    fold_masks = [out_root / f"fold_{fold}" / "LAD.nii.gz" for fold in range(5)]
    if not all(p.exists() for p in fold_masks):
        return None, None
    imgs = [sitk.ReadImage(str(p)) for p in fold_masks]
    arr = np.stack([sitk.GetArrayFromImage(x).astype(np.uint8) for x in imgs], axis=0)
    counts = np.stack([(arr == lab).sum(axis=0) for lab in (0, 1, 2)], axis=0)
    consensus = np.argmax(counts, axis=0).astype(np.uint8)
    disagreement = 1.0 - counts.max(axis=0).astype(np.float32) / float(arr.shape[0])
    cimg = sitk.GetImageFromArray(consensus); cimg.CopyInformation(imgs[0])
    dimg = sitk.GetImageFromArray(disagreement.astype(np.float32)); dimg.CopyInformation(imgs[0])
    cfp = out_root / "LAD_5fold_consensus.nii.gz"
    dfp = out_root / "LAD_5fold_disagreement.nii.gz"
    sitk.WriteImage(cimg, str(cfp)); sitk.WriteImage(dimg, str(dfp))
    return cfp, dfp


class LADValidatedLongitudinalPlaqueProfileWorkflow:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque"):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LAD_Validated_Longitudinal_Plaque_Profile_v1"
        self.out = self.root / "LAD_Validated_Longitudinal_Plaque_Profile_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spacing = None
        self.current = self.old = None
        self.current_arc = self.old_arc = None
        self.registration = None
        self.overlap = None
        self.curved_volume = None
        self.consensus = self.disagreement = None
        self.pixel_profile = self.bin_profile = self.support_zones = None
        self.summary = None

    def resolve_inputs(self):
        self.spacing, spacing_fp = _source_spacing(self.root)
        current_fp = _find_one(self.root / "LAD_Frozen_Proximal_Reacquisition_Report", "combined_lad_centerline.csv")
        if current_fp is None:
            current_fp = _find_one(self.root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1", "combined_lad_centerline.csv")
        old_fp = _find_one(self.root / "Canonical_Source_Coronary_Centerlines_v1", "LAD_canonical_source_centerline.csv")
        reg_fp = _find_one(self.root / "Curved_Plaque_to_Source_Registration_Canonical_v1" / "LAD", "registration_model.json")
        if current_fp is None or old_fp is None or reg_fp is None:
            raise FileNotFoundError(f"Missing required LAD inputs: current={current_fp}, canonical={old_fp}, registration={reg_fp}")
        current_df = pd.read_csv(current_fp); old_df = pd.read_csv(old_fp)
        for name, df in [("current", current_df), ("canonical", old_df)]:
            if not {"z", "y", "x"}.issubset(df.columns):
                raise ValueError(f"{name} centerline lacks z,y,x columns")
        self.current = current_df[["z", "y", "x"]].to_numpy(float)
        self.old = old_df[["z", "y", "x"]].to_numpy(float)
        self.current_arc = _arc_zyx(self.current, self.spacing)
        self.old_arc = _arc_zyx(self.old, self.spacing)
        self.registration = _json_read(reg_fp)
        if abs(float(self.current_arc[-1]) - ACCEPTED_LAD_TARGET_MM) > 0.08:
            raise RuntimeError(f"Current LAD length {self.current_arc[-1]:.6f} mm does not match frozen baseline")
        if self.registration.get("status") != "PARTIAL_LONGITUDINAL_MAPPING_ONLY":
            raise RuntimeError(f"Unexpected registration status {self.registration.get('status')!r}")
        lf = self.registration.get("longitudinal_fit", {})
        if float(lf.get("score", 0.0)) < 0.65 or float(lf.get("intensity_corr", 0.0)) < 0.75:
            raise RuntimeError("Longitudinal registration is not strong enough for plaque profiling")

        cfp = self.root / "GPU_Plaque_5Fold_Ensemble_v1" / "LAD" / "LAD_5fold_consensus.nii.gz"
        dfp = self.root / "GPU_Plaque_5Fold_Ensemble_v1" / "LAD" / "LAD_5fold_disagreement.nii.gz"
        if not cfp.exists() or not dfp.exists():
            built = build_consensus_from_fold_masks(self.root)
            if built[0] is not None:
                cfp, dfp = built
        if not cfp.exists() or not dfp.exists():
            raise FileNotFoundError("LAD five-fold consensus/disagreement not found. Run the cached five-fold prediction cell first.")

        study_zip = self.root / "Full_DICOM.zip"
        study = OpenPlaqueStudy(str(study_zip), extract_root="/content/full_dicom_lad_longitudinal_plaque")
        _, volume, _ = study.load_series(LAD_SERIES)
        self.curved_volume = np.asarray(volume)
        self.consensus = sitk.GetArrayFromImage(sitk.ReadImage(str(cfp))).astype(np.uint8)
        self.disagreement = sitk.GetArrayFromImage(sitk.ReadImage(str(dfp))).astype(np.float32)
        if self.consensus.shape != self.curved_volume.shape or self.disagreement.shape != self.curved_volume.shape:
            raise RuntimeError(f"Curved image/mask shapes differ: {self.curved_volume.shape}, {self.consensus.shape}, {self.disagreement.shape}")
        provenance = {
            "algorithm": ALGORITHM_VERSION,
            "current_accepted_lad": str(current_fp),
            "current_accepted_lad_mm": float(self.current_arc[-1]),
            "older_registration_centerline": str(old_fp),
            "older_registration_centerline_mm": float(self.old_arc[-1]),
            "registration_model": str(reg_fp),
            "registration_status": self.registration.get("status"),
            "ensemble_consensus": str(cfp),
            "ensemble_disagreement": str(dfp),
            "curved_series": LAD_SERIES,
            "series7_spacing_metadata": str(spacing_fp),
            "anatomical_volume_mm3_permitted": False,
        }
        _json_write(provenance, self.cache / "input_provenance.json")
        return provenance

    def validate_centerline_overlap(self):
        if self.current is None:
            self.resolve_inputs()
        tree = cKDTree(self.old * self.spacing[None, :])
        dist, idx = tree.query(self.current * self.spacing[None, :])
        matched_old_arc = self.old_arc[idx]
        corr = float(np.corrcoef(self.current_arc, matched_old_arc)[0, 1])
        median_d = float(np.median(dist)); max_d = float(np.max(dist)); p95 = float(np.percentile(dist, 95))
        if median_d > 0.15 or max_d > 0.60 or corr < 0.995:
            raise RuntimeError(f"Final LAD/registration-centerline alignment failed: median={median_d:.3f}, max={max_d:.3f}, corr={corr:.5f}")
        pairs = pd.DataFrame({"old_arc_mm": matched_old_arc, "accepted_arc_mm": self.current_arc, "distance_mm": dist})
        grouped = pairs.groupby("old_arc_mm", as_index=False).agg(accepted_arc_mm=("accepted_arc_mm", "mean"), distance_mm=("distance_mm", "min")).sort_values("old_arc_mm")
        self._old_map = grouped.old_arc_mm.to_numpy(float)
        self._current_map = grouped.accepted_arc_mm.to_numpy(float)
        lf = self.registration["longitudinal_fit"]
        start = int(lf["start_px"]); npx = int(lf["n_pixels"]); flip = bool(lf.get("long_flip", False))
        long_pixels = np.arange(start, start + npx, dtype=int)
        frac = (long_pixels - start) / max(npx - 1, 1)
        if flip: frac = 1.0 - frac
        mapped_old_arc = float(self.old_arc[0]) + frac * float(self.old_arc[-1] - self.old_arc[0])
        omin, omax = float(grouped.old_arc_mm.min()), float(grouped.old_arc_mm.max())
        inside = (mapped_old_arc >= omin - 0.35) & (mapped_old_arc <= omax + 0.35)
        selected_pixels = long_pixels[inside]
        selected_old_arc = mapped_old_arc[inside]
        selected_current_arc = np.interp(np.clip(selected_old_arc, self._old_map[0], self._old_map[-1]), self._old_map, self._current_map)
        selected_current_arc = np.clip(selected_current_arc, 0.0, float(self.current_arc[-1]))
        if len(selected_pixels) < 30:
            raise RuntimeError(f"Only {len(selected_pixels)} curved longitudinal pixels map to accepted LAD")
        self.overlap = pd.DataFrame({"curved_long_pixel": selected_pixels, "old_canonical_arc_mm": selected_old_arc, "accepted_lad_arc_mm": selected_current_arc})
        self.overlap.to_csv(self.cache / "accepted_lad_curved_longitudinal_mapping.csv", index=False)
        pairs.to_csv(self.cache / "accepted_vs_old_centerline_point_mapping.csv", index=False)
        summary = {
            "accepted_lad_length_mm": float(self.current_arc[-1]),
            "old_canonical_length_mm": float(self.old_arc[-1]),
            "median_centerline_separation_mm": median_d,
            "p95_centerline_separation_mm": p95,
            "max_centerline_separation_mm": max_d,
            "arc_correlation": corr,
            "accepted_old_canonical_arc_min_mm": omin,
            "accepted_old_canonical_arc_max_mm": omax,
            "curved_long_pixel_min": int(selected_pixels.min()),
            "curved_long_pixel_max": int(selected_pixels.max()),
            "curved_long_pixels": int(len(selected_pixels)),
            "registration_start_px": start,
            "registration_n_pixels": npx,
            "registration_long_flip": flip,
        }
        _json_write(summary, self.cache / "mapping_validation_summary.json")
        return summary

    def _oriented(self):
        axis = int(self.registration["longitudinal_fit"]["long_axis"])
        if axis == 2:
            return self.curved_volume, self.consensus, self.disagreement
        if axis == 1:
            return np.transpose(self.curved_volume, (0, 2, 1)), np.transpose(self.consensus, (0, 2, 1)), np.transpose(self.disagreement, (0, 2, 1))
        raise ValueError(f"Unsupported long_axis {axis}")

    def build_profile(self):
        if self.overlap is None:
            self.validate_centerline_overlap()
        vol, mask, dis = self._oriented(); nrot, nrad, nlong = vol.shape
        rows = []
        for r in self.overlap.itertuples(index=False):
            x = int(r.curved_long_pixel)
            if not (0 <= x < nlong): continue
            plaque = mask[:, :, x] == 2
            vals = vol[:, :, x][plaque]; dvals = dis[:, :, x][plaque]
            support = int(plaque.sum()); rotations = int(np.any(plaque, axis=1).sum())
            rec = {
                "curved_long_pixel": x,
                "accepted_lad_arc_mm": float(r.accepted_lad_arc_mm),
                "old_canonical_arc_mm": float(r.old_canonical_arc_mm),
                "consensus_plaque_support_pixels": support,
                "plaque_support_fraction_of_rotation_stack": float(support/(nrot*nrad)),
                "rotations_with_consensus_plaque": rotations,
                "rotation_fraction_with_consensus_plaque": float(rotations/nrot),
                "mean_fold_disagreement_in_plaque": float(np.mean(dvals)) if dvals.size else np.nan,
                "mean_consensus_confidence_in_plaque": float(1.0-np.mean(dvals)) if dvals.size else np.nan,
                "mean_plaque_hu_curved_stack": float(np.mean(vals)) if vals.size else np.nan,
                "median_plaque_hu_curved_stack": float(np.median(vals)) if vals.size else np.nan,
            }
            total = max(int(vals.size), 1)
            for name, (lo, hi) in HU_THRESHOLDS.items():
                count = int(np.sum((vals >= lo) & (vals < hi))) if vals.size else 0
                rec[f"{name}_support_pixels"] = count
                rec[f"{name}_fraction"] = float(count/total) if vals.size else 0.0
            rows.append(rec)
        self.pixel_profile = pd.DataFrame(rows).sort_values("accepted_lad_arc_mm").reset_index(drop=True)
        self.pixel_profile.to_csv(self.cache / "lad_longitudinal_plaque_profile_by_pixel.csv", index=False)
        q = self.pixel_profile.copy(); q["arc_bin_mm"] = np.floor(q.accepted_lad_arc_mm).astype(int)
        agg = []
        for b, g in q.groupby("arc_bin_mm"):
            gp = g[g.consensus_plaque_support_pixels > 0]
            totals = {name: int(g[f"{name}_support_pixels"].sum()) for name in HU_THRESHOLDS}; nplaque = sum(totals.values())
            rec = {
                "arc_bin_start_mm": int(b), "arc_bin_end_mm": min(float(b+1), float(self.current_arc[-1])),
                "n_curved_long_pixels": int(len(g)), "mean_support_pixels": float(g.consensus_plaque_support_pixels.mean()),
                "max_support_pixels": int(g.consensus_plaque_support_pixels.max()), "mean_rotation_fraction": float(g.rotation_fraction_with_consensus_plaque.mean()),
                "max_rotation_fraction": float(g.rotation_fraction_with_consensus_plaque.max()),
                "mean_plaque_disagreement": float(gp.mean_fold_disagreement_in_plaque.mean()) if len(gp) else np.nan,
                "total_curved_stack_plaque_support_pixels": int(nplaque),
            }
            for name in HU_THRESHOLDS:
                rec[f"{name}_support_pixels"] = totals[name]; rec[f"{name}_fraction"] = float(totals[name]/nplaque) if nplaque else 0.0
            agg.append(rec)
        self.bin_profile = pd.DataFrame(agg); self.bin_profile.to_csv(self.cache / "lad_longitudinal_plaque_profile_1mm_bins.csv", index=False)
        positive = self.pixel_profile.consensus_plaque_support_pixels.to_numpy() > 0
        zones=[]; i=0; zid=1
        while i < len(positive):
            if not positive[i]: i+=1; continue
            j=i
            while j+1 < len(positive) and positive[j+1]: j+=1
            g=self.pixel_profile.iloc[i:j+1]
            zones.append({"support_zone_id":zid,"arc_start_mm":float(g.accepted_lad_arc_mm.min()),"arc_end_mm":float(g.accepted_lad_arc_mm.max()),"longitudinal_span_mm":float(g.accepted_lad_arc_mm.max()-g.accepted_lad_arc_mm.min()),"n_long_pixels":int(len(g)),"peak_support_pixels":int(g.consensus_plaque_support_pixels.max()),"peak_rotation_fraction":float(g.rotation_fraction_with_consensus_plaque.max()),"mean_plaque_disagreement":float(g.mean_fold_disagreement_in_plaque.mean())})
            zid+=1; i=j+1
        self.support_zones=pd.DataFrame(zones); self.support_zones.to_csv(self.cache / "lad_longitudinal_plaque_support_zones.csv", index=False)
        total_support=int(self.pixel_profile.consensus_plaque_support_pixels.sum()); composition={}
        for name in HU_THRESHOLDS:
            c=int(self.pixel_profile[f"{name}_support_pixels"].sum()); composition[name]={"support_pixels":c,"fraction":float(c/total_support) if total_support else 0.0}
        self.summary={
            "algorithm":ALGORITHM_VERSION,"status":"LONGITUDINAL_PLAQUE_PROFILE_COMPLETE_NONVOLUMETRIC",
            "accepted_lad_length_mm":float(self.current_arc[-1]),"mapped_curved_long_pixel_min":int(self.overlap.curved_long_pixel.min()),"mapped_curved_long_pixel_max":int(self.overlap.curved_long_pixel.max()),"mapped_curved_long_pixel_count":int(len(self.overlap)),
            "consensus_plaque_support_pixels_total":total_support,"longitudinal_support_zone_count":int(len(self.support_zones)),"composition_support_distribution":composition,
            "anatomical_plaque_volume_mm3_reported":False,"anatomical_plaque_area_mm2_reported":False,
            "reason_volume_not_reported":"LAD curved DICOM is a nonspatial/rotation stack and angular registration failed conservative validation; only longitudinal localization is supported.",
            "registration_status":self.registration.get("status"),"longitudinal_registration_score":float(self.registration["longitudinal_fit"]["score"]),"angular_median_ncc":float(self.registration["angular_fit"]["median_ncc"]),"angular_p25_ncc":float(self.registration["angular_fit"]["p25_ncc"]),
        }
        _json_write(self.summary,self.cache/"summary.json"); return self.summary

    def make_figures(self):
        if self.summary is None: self.build_profile()
        names=[]; pairs=pd.read_csv(self.cache/"accepted_vs_old_centerline_point_mapping.csv")
        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(pairs.accepted_arc_mm,pairs.old_arc_mm,marker="."); ax.set_xlabel("Accepted LAD arc (mm)"); ax.set_ylabel("Matched older canonical LAD arc (mm)"); ax.set_title("Accepted LAD embedded in older longitudinal-registration centerline"); ax.grid(alpha=.25)
        fp=self.out/"01_accepted_lad_registration_overlap.png"; fig.tight_layout(); fig.savefig(fp,dpi=170); plt.close(fig); names.append(fp.name)
        q=self.pixel_profile; fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
        axs[0].plot(q.accepted_lad_arc_mm,q.consensus_plaque_support_pixels,marker="o"); axs[0].set_ylabel("Consensus plaque\nsupport pixels")
        axs[1].plot(q.accepted_lad_arc_mm,q.rotation_fraction_with_consensus_plaque,marker="o"); axs[1].set_ylabel("Rotation fraction\nwith plaque")
        axs[2].plot(q.accepted_lad_arc_mm,q.mean_fold_disagreement_in_plaque,marker="o"); axs[2].set_ylabel("Mean fold\ndisagreement"); axs[2].set_xlabel("Accepted LAD arc (mm)"); fig.suptitle("Validated-LAD longitudinal plaque support (non-volumetric)")
        fp=self.out/"02_longitudinal_plaque_support_uncertainty.png"; fig.tight_layout(); fig.savefig(fp,dpi=170); plt.close(fig); names.append(fp.name)
        fig,ax=plt.subplots(figsize=(12,5)); x=q.accepted_lad_arc_mm.to_numpy(float); ys=[q[f"{name}_fraction"].to_numpy(float) for name in HU_THRESHOLDS]; ax.stackplot(x,*ys,labels=list(HU_THRESHOLDS)); ax.set_ylim(0,1); ax.set_xlabel("Accepted LAD arc (mm)"); ax.set_ylabel("Fraction of consensus plaque support pixels"); ax.set_title("Curved-stack attenuation composition along accepted LAD"); ax.legend(loc="upper right",fontsize=8)
        fp=self.out/"03_longitudinal_attenuation_composition.png"; fig.tight_layout(); fig.savefig(fp,dpi=170); plt.close(fig); names.append(fp.name)
        _,mask,_=self._oriented(); xs=self.overlap.curved_long_pixel.to_numpy(int); rot=np.stack([(mask[:,:,x]==2).sum(axis=1) for x in xs],axis=1)
        fig,ax=plt.subplots(figsize=(13,6)); im=ax.imshow(rot,aspect="auto",origin="lower",interpolation="nearest",extent=[float(q.accepted_lad_arc_mm.min()),float(q.accepted_lad_arc_mm.max()),0,rot.shape[0]]); ax.set_xlabel("Accepted LAD arc (mm)"); ax.set_ylabel("Curved rotation index"); ax.set_title("Consensus plaque support across curved rotations"); fig.colorbar(im,ax=ax,label="radial pixels labeled plaque")
        fp=self.out/"04_rotation_by_arc_plaque_support.png"; fig.tight_layout(); fig.savefig(fp,dpi=170); plt.close(fig); names.append(fp.name)
        b=self.bin_profile; fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True); axs[0].bar(b.arc_bin_start_mm,b.total_curved_stack_plaque_support_pixels,width=.85); axs[0].set_ylabel("Plaque support pixels\n(curved stack)"); axs[1].plot(b.arc_bin_start_mm,b.mean_plaque_disagreement,marker="o"); axs[1].set_ylabel("Mean disagreement"); axs[1].set_xlabel("Accepted LAD arc bin start (mm)"); fig.suptitle("Validated-LAD plaque support by 1-mm longitudinal bin")
        fp=self.out/"05_plaque_support_1mm_bins.png"; fig.tight_layout(); fig.savefig(fp,dpi=170); plt.close(fig); names.append(fp.name)
        _json_write(names,self.out/"figure_manifest.json"); return names

    def build_report(self):
        if self.summary is None: self.build_profile()
        names=self.make_figures(); mapping=_json_read(self.cache/"mapping_validation_summary.json")
        zones=self.support_zones.to_html(index=False,float_format=lambda x:f"{x:.3f}") if len(self.support_zones) else "<p>No consensus plaque-support zones within accepted LAD.</p>"
        html=f"""<html><head><meta charset='utf-8'><title>OpenPlaque LAD Validated Longitudinal Plaque Profile</title><style>body{{font-family:Arial;max-width:1500px;margin:24px auto;padding:0 18px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:5px;border-bottom:1px solid #ddd;text-align:left}}.warn{{background:#fff4d6;padding:12px;border-left:4px solid #cc8a00}}</style></head><body><h1>OpenPlaque — Validated LAD Longitudinal Plaque Profile</h1><div class='warn'><b>Non-volumetric result.</b> The LAD curved series is a nonspatial rotation stack. Longitudinal registration is supported, but angular/source-space registration failed conservative validation. This report therefore does <b>not</b> report anatomical plaque volume (mm³) or area (mm²).</div><h2>Summary</h2><pre>{json.dumps(self.summary,indent=2)}</pre><h2>Accepted-LAD mapping validation</h2><pre>{json.dumps(mapping,indent=2)}</pre><h2>Longitudinal plaque-support zones</h2>{zones}<h2>1-mm longitudinal profile</h2>{self.bin_profile.to_html(index=False,float_format=lambda x:f'{x:.3f}')}<h2>Interpretation limits</h2><p>All counts and attenuation fractions are support statistics in the curved rotation stack. Repeated rotations are not independent anatomical voxels. These values may localize and compare plaque-model support along the accepted LAD, but they must not be interpreted as TPV, plaque area, or physical lesion volume.</p><h2>Figures</h2>{''.join(f'<h3>{n}</h3><img src="{n}">' for n in names)}</body></html>"""
        report=self.out/"OPENPLAQUE_LAD_VALIDATED_LONGITUDINAL_PLAQUE_PROFILE_REPORT.html"; report.write_text(html,encoding="utf-8"); return report

    def package(self):
        report=self.build_report(); zp=self.out/"OPENPLAQUE_LAD_VALIDATED_LONGITUDINAL_PLAQUE_PROFILE_REPORT_BACK.zip"
        with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
            for fp in sorted(self.out.iterdir()):
                if fp.is_file() and fp!=zp: z.write(fp,arcname=fp.name)
            for fp in sorted(self.cache.iterdir()):
                if fp.is_file() and fp.suffix in (".csv",".json"): z.write(fp,arcname=fp.name)
        return report,zp


def synthetic_longitudinal_plaque_profile_self_test():
    start,npx,L=196,90,57.27599578748678; pix=np.array([start,start+npx-1]); frac=(pix-start)/(npx-1); arc=frac*L
    return {"passed":bool(np.allclose(arc,[0.0,L])),"algorithm":ALGORITHM_VERSION,"mapping_endpoint_arc_mm":arc.tolist()}
