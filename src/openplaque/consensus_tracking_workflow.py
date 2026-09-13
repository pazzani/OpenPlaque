from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pickle
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from .artery_detection import detect_artery_series
from .boundary import refine_plaque_mask
from .cpr_consensus import (
    TRACKER_VERSION,
    build_consensus,
    choose_best_rotation,
    consensus_candidates,
    path_tube,
    qc_status,
    straighten,
)
from .segmentation import segment_vessel
from .study import OpenPlaqueStudy

VESSELS = ("LAD", "RCA", "LCX")
FALLBACK_SERIES = {"RCA": 1035, "LCX": 1039, "LAD": 1043}


def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _save_mask(mask, reference_image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.asarray(mask, np.uint8))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(path))


def _load_valid_plaque(path, reference_image, expected_mm3):
    path = Path(path)
    if not path.exists():
        return None
    try:
        img = sitk.ReadImage(str(path))
        if img.GetSize() != reference_image.GetSize():
            return None
        if not np.allclose(img.GetSpacing(), reference_image.GetSpacing(), atol=1e-5):
            return None
        arr = sitk.GetArrayFromImage(img)
        mask = (arr == 2) if np.nanmax(arr) > 1 else (arr > 0)
        voxel = float(np.prod(reference_image.GetSpacing()))
        actual = float(mask.sum() * voxel)
        if abs(actual - float(expected_mm3)) > max(0.75, 1.1 * voxel):
            return None
        return mask
    except Exception:
        return None


def _ensure_model(root):
    os.environ["nnUNet_raw"] = "/content/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/content/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "/content/nnUNet_results"
    for d in (os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]):
        Path(d).mkdir(parents=True, exist_ok=True)
    target = Path("/content/nnUNet_results/Dataset001_CCTA_DHM")
    if target.exists():
        return
    model_zip = Path(root) / "models" / "Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip"
    if not model_zip.exists():
        raise FileNotFoundError(model_zip)
    with zipfile.ZipFile(model_zip) as z:
        z.extractall("/content/nnUNet_results")


class ConsensusTrackingWorkflow:
    COMPONENTS = (
        "series_selection",
        "plaque_masks",
        "consensus_evidence",
        "tracking",
        "candidate_figure",
        "roadmaps",
        "pcat_figures",
        "dashboard",
        "report_package",
    )

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.out = self.root / "Cross_Rotation_Consensus_Tracking_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache_root = self.root / "Cache" / "Cross_Rotation_Consensus_v3"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            unknown = set(reuse) - set(self.COMPONENTS)
            if unknown:
                raise ValueError(f"Unknown reuse components: {sorted(unknown)}")
            self.reuse.update({k: bool(v) for k, v in reuse.items()})

        self.study = None
        self.series_map = None
        self.tpv = self.pcat = self.ps = None
        self.data = None
        self.cache_qc = None
        self.consensus = None
        self.candidates = None
        self.selected = None
        self.rotation_rankings = None
        self.tracking_qc = None
        self.along_df = None
        self.provenance = []
        self._done = set()

    def _record(self, component, action, path, note=""):
        self.provenance.append({
            "component": component,
            "reuse_requested": self.reuse[component],
            "action": action,
            "path": str(path),
            "note": note,
        })
        pd.DataFrame(self.provenance).to_csv(self.out / "cache_provenance.csv", index=False)

    def _paths(self):
        return {
            "series_selection": self.cache_root / "series_selection.json",
            "plaque_masks": self.cache_root / "plaque",
            "consensus_evidence": self.cache_root / "consensus",
            "tracking": self.cache_root / "tracking.pkl",
            "candidate_figure": self.cache_root / "candidate_figure.json",
            "roadmaps": self.cache_root / "roadmaps.json",
            "pcat_figures": self.cache_root / "pcat",
            "dashboard": self.cache_root / "dashboard.json",
            "report_package": self.cache_root / "report.json",
        }

    def _plaque_sources(self, vessel):
        own = self._paths()["plaque_masks"] / f"{vessel}_canonical_refined.nii.gz"
        v2 = self.root / "Cache" / "Image_Driven_Tracking_v2" / "plaque" / f"{vessel}_canonical_refined.nii.gz"
        prior = self.root / "Cache" / "Image_Driven_Tracking_Cache_Controlled" / "plaque" / f"{vessel}_canonical_refined.nii.gz"
        legacy = []
        for d in (self.root / "Segmentations", self.root / "segmentations"):
            legacy += [
                d / f"{vessel}_refined_plaque_segmentation.nii.gz",
                d / f"{vessel}_volume_refined_plaque_segmentation.nii.gz",
            ]
        return [own, v2, prior] + legacy

    def cache_status(self):
        p = self._paths()
        exists = {
            "series_selection": p["series_selection"].exists(),
            "plaque_masks": all(any(x.exists() for x in self._plaque_sources(v)) for v in VESSELS),
            "consensus_evidence": all((p["consensus_evidence"] / f"{v}_consensus.npz").exists() for v in VESSELS),
            "tracking": p["tracking"].exists(),
            "candidate_figure": p["candidate_figure"].exists() and (self.out / "01_consensus_candidates.png").exists(),
            "roadmaps": p["roadmaps"].exists() and (self.out / "02_consensus_coronary_roadmaps.png").exists(),
            "pcat_figures": (p["pcat_figures"] / "cross.png").exists() and (p["pcat_figures"] / "ribbon.png").exists(),
            "dashboard": p["dashboard"].exists() and (self.out / "05_summary_dashboard.png").exists(),
            "report_package": p["report_package"].exists() and (self.out / "OPENPLAQUE_CROSS_ROTATION_CONSENSUS_REPORT_BACK.zip").exists(),
        }
        return pd.DataFrame([
            {
                "component": k,
                "reuse": self.reuse[k],
                "cache_available": exists[k],
                "planned_action": "reuse" if self.reuse[k] and exists[k] else "recompute_and_cache",
                "cache_path": str(p[k]),
            }
            for k in self.COMPONENTS
        ])

    def _load_metrics(self):
        m = self.root / "Combined_TPV_PCAT_All_Metrics_v2"
        self.tpv = pd.read_csv(m / "tpv_metrics_by_vessel_v2.csv")
        self.pcat = pd.read_csv(m / "pcat_canonical_primary_v2.csv")
        self.ps = pd.read_csv(m / "pcat_circular_sensitivity_v2.csv")

    def prepare_inputs(self):
        if "series_selection" in self._done:
            return self.series_map
        drive_zip = self.root / "Full_DICOM.zip"
        local_zip = Path("/content/Full_DICOM.zip")
        if not drive_zip.exists():
            raise FileNotFoundError(drive_zip)
        if not local_zip.exists() or local_zip.stat().st_size != drive_zip.stat().st_size:
            shutil.copyfile(drive_zip, local_zip)
        self.study = OpenPlaqueStudy(str(local_zip), extract_root="/content/full_dicom_consensus_v3")
        cache = self._paths()["series_selection"]
        loaded = False
        if self.reuse["series_selection"] and cache.exists():
            try:
                d = _load_json(cache)
                if int(d["dicom_size"]) != int(drive_zip.stat().st_size):
                    raise ValueError("DICOM size changed")
                self.series_map = {k: int(v) for k, v in d["series_map"].items()}
                for v in VESSELS:
                    self.study.load_series(self.series_map[v])
                loaded = True
                self._record("series_selection", "reused", cache)
            except Exception as e:
                self._record("series_selection", "cache_invalid", cache, repr(e))
        if not loaded:
            self.series_map, _ = detect_artery_series(self.study, fallback_series=FALLBACK_SERIES, return_candidates=True)
            self.series_map = {k: int(v) for k, v in self.series_map.items()}
            _save_json({"dicom_size": int(drive_zip.stat().st_size), "series_map": self.series_map}, cache)
            self._record("series_selection", "recomputed_and_cached", cache)
        self._load_metrics()
        self._done.add("series_selection")
        return self.series_map

    def load_plaque_masks(self):
        if "plaque_masks" in self._done:
            return self.cache_qc
        if self.study is None:
            self.prepare_inputs()
        cache_dir = self._paths()["plaque_masks"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        force = not self.reuse["plaque_masks"]
        if force:
            _ensure_model(self.root)
        data, rows = {}, []
        for vessel in VESSELS:
            image, volume, _ = self.study.load_series(self.series_map[vessel])
            expected = float(self.tpv.loc[self.tpv.vessel == vessel, "canonical_refined_tpv_mm3"].iloc[0])
            mask = None
            used = None
            source = None
            if not force:
                for p in self._plaque_sources(vessel):
                    mask = _load_valid_plaque(p, image, expected)
                    if mask is not None:
                        used, source = p, "cache"
                        break
            if mask is None:
                _ensure_model(self.root)
                r = segment_vessel(image, volume, vessel)
                q = refine_plaque_mask(
                    volume=r.volume,
                    mask=r.mask,
                    spacing=r.mask_image.GetSpacing(),
                    remove_small=True,
                    min_component_voxels=10,
                    trim_lumen_adjacent=True,
                    lumen_distance_voxels=1,
                    erode_core=False,
                    high_hu_threshold=None,
                    low_hu_threshold=None,
                )
                mask = q.refined_mask == 2
                source = "forced_recompute" if force else "cache_miss_recompute"
            normalized = cache_dir / f"{vessel}_canonical_refined.nii.gz"
            if used != normalized or not normalized.exists():
                _save_mask(mask, image, normalized)
            data[vessel] = {"image": image, "volume": np.asarray(volume), "plaque": np.asarray(mask, bool)}
            rows.append({
                "vessel": vessel,
                "source": source,
                "path": str(normalized),
                "expected_tpv_mm3": expected,
                "plaque_voxels": int(mask.sum()),
            })
        self.data = data
        self.cache_qc = pd.DataFrame(rows)
        self.cache_qc.to_csv(self.out / "cache_qc.csv", index=False)
        self._record("plaque_masks", "recomputed_and_cached" if force else "reused_or_filled_missing", cache_dir)
        self._done.add("plaque_masks")
        return self.cache_qc

    def build_consensus_evidence(self):
        if "consensus_evidence" in self._done:
            return self.consensus
        if self.data is None:
            self.load_plaque_masks()
        cache_dir = self._paths()["consensus_evidence"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for vessel in VESSELS:
            cache = cache_dir / f"{vessel}_consensus.npz"
            loaded = False
            if self.reuse["consensus_evidence"] and cache.exists():
                try:
                    z = np.load(cache)
                    if str(z["tracker_version"].item()) != TRACKER_VERSION:
                        raise ValueError("tracker version changed")
                    if int(z["series_number"].item()) != int(self.series_map[vessel]):
                        raise ValueError("series changed")
                    result[vessel] = {k: np.asarray(z[k]) for k in ("consensus", "support", "median", "q25", "valid_fraction")}
                    loaded = True
                except Exception:
                    loaded = False
            if not loaded:
                d = build_consensus(self.data[vessel]["volume"])
                np.savez_compressed(
                    cache,
                    tracker_version=np.array(TRACKER_VERSION),
                    series_number=np.array(int(self.series_map[vessel])),
                    **d,
                )
                result[vessel] = d
        self.consensus = result
        self._record("consensus_evidence", "reused" if self.reuse["consensus_evidence"] and all((cache_dir / f"{v}_consensus.npz").exists() for v in VESSELS) else "recomputed_and_cached", cache_dir)
        self._done.add("consensus_evidence")
        return self.consensus

    def _tracking_signature(self):
        if self.consensus is None:
            self.build_consensus_evidence()
        return {
            "version": TRACKER_VERSION,
            "series_map": self.series_map,
            "consensus": {v: _hash_array(self.consensus[v]["consensus"].astype(np.float32)) for v in VESSELS},
            "plaque": {v: _hash_array(self.data[v]["plaque"].astype(np.uint8)) for v in VESSELS},
        }

    def _compact(self, d):
        if d is None:
            return None
        out = dict(d)
        if "path" in out:
            out["path"] = np.asarray(out["path"], np.float32)
        out.pop("tube", None)
        out.pop("path_mask", None)
        return out

    def _inflate(self, vessel, d):
        if d is None:
            return None
        out = dict(d)
        out["path"] = np.asarray(out["path"], float)
        sp = self.data[vessel]["image"].GetSpacing()
        shape = self.data[vessel]["volume"][0].shape
        out["tube"], out["path_mask"] = path_tube(out["path"], shape, (float(sp[1]), float(sp[0])), 5.0)
        return out

    def track_consensus_paths(self):
        if "tracking" in self._done:
            return self.tracking_qc
        if self.consensus is None:
            self.build_consensus_evidence()
        cache = self._paths()["tracking"]
        sig = self._tracking_signature()
        if self.reuse["tracking"] and cache.exists():
            try:
                with cache.open("rb") as f:
                    p = pickle.load(f)
                if p["signature"] != sig:
                    raise ValueError("tracking dependency changed")
                self.candidates = {v: [self._inflate(v, x) for x in p["candidates"][v]] for v in VESSELS}
                self.selected = {v: self._inflate(v, p["selected"][v]) for v in VESSELS}
                self.rotation_rankings = p["rotation_rankings"]
                self.tracking_qc = pd.DataFrame(p["tracking_qc"])
                self.tracking_qc.to_csv(self.out / "tracking_qc.csv", index=False)
                self._record("tracking", "reused", cache)
                self._done.add("tracking")
                return self.tracking_qc
            except Exception as e:
                self._record("tracking", "cache_invalid", cache, repr(e))

        candidates, selected, rankings, rows = {}, {}, {}, []
        for vessel in VESSELS:
            d = self.data[vessel]
            sp = d["image"].GetSpacing()
            sp_yx = (float(sp[1]), float(sp[0]))
            cands = consensus_candidates(self.consensus[vessel], sp_yx)
            candidates[vessel] = cands
            win = cands[0] if cands else None
            if win is None:
                selected[vessel] = None
                rankings[vessel] = []
                rows.append({
                    "vessel": vessel,
                    "selected_frame": np.nan,
                    "path_length_mm": np.nan,
                    "sinuosity": np.nan,
                    "mean_consensus": np.nan,
                    "mean_cross_rotation_support": np.nan,
                    "mean_path_hu": np.nan,
                    "core_minus_ring_hu": np.nan,
                    "plaque_voxels_in_selected_frame": 0,
                    "plaque_voxels_within_5mm_tube": 0,
                    "plaque_capture_pct_of_selected_frame": 0.0,
                    "tracking_status": "FAIL",
                    "qc_reason": "no persistent open coronary-scale consensus path",
                })
                continue
            best_rotation, rank = choose_best_rotation(d["volume"], win["path"], sp_yx)
            rankings[vessel] = rank
            tube, pm = path_tube(win["path"], d["volume"][0].shape, sp_yx, 5.0)
            win = dict(win)
            win.update(best_rotation)
            win["tube"], win["path_mask"] = tube, pm
            z = int(best_rotation["frame"])
            plaque_frame = int(d["plaque"][z].sum())
            plaque_tube = int((d["plaque"][z] & tube).sum())
            win["plaque_frame_voxels"] = plaque_frame
            win["plaque_tube_voxels"] = plaque_tube
            win["plaque_capture_fraction"] = plaque_tube / max(plaque_frame, 1) if plaque_frame else 0.0
            selected[vessel] = win
            status, reason = qc_status(win, best_rotation)
            if plaque_frame > 0 and plaque_tube == 0:
                reason = (reason + "; " if reason else "") + "selected-frame plaque mask does not intersect path (informational; plaque mask is not tracker validation)"
                if status == "OK":
                    status = "REVIEW"
            rows.append({
                "vessel": vessel,
                "selected_frame": z,
                "path_length_mm": win["path_length_mm"],
                "endpoint_distance_mm": win["endpoint_distance_mm"],
                "sinuosity": win["sinuosity"],
                "mean_consensus": win["mean_consensus"],
                "mean_cross_rotation_support": win["mean_support"],
                "mean_path_hu": best_rotation["mean_path_hu"],
                "plausible_hu_fraction": best_rotation["plausible_hu_fraction"],
                "core_minus_ring_hu": best_rotation["core_minus_ring_hu"],
                "plaque_voxels_in_selected_frame": plaque_frame,
                "plaque_voxels_within_5mm_tube": plaque_tube,
                "plaque_capture_pct_of_selected_frame": 100 * win["plaque_capture_fraction"],
                "tracking_status": status,
                "qc_reason": reason,
            })

        self.candidates, self.selected, self.rotation_rankings = candidates, selected, rankings
        self.tracking_qc = pd.DataFrame(rows)
        self.tracking_qc.to_csv(self.out / "tracking_qc.csv", index=False)
        payload = {
            "signature": sig,
            "candidates": {v: [self._compact(x) for x in candidates[v]] for v in VESSELS},
            "selected": {v: self._compact(selected[v]) for v in VESSELS},
            "rotation_rankings": rankings,
            "tracking_qc": self.tracking_qc.to_dict(orient="records"),
        }
        with cache.open("wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
        self._record("tracking", "recomputed_and_cached", cache)
        self._done.add("tracking")
        return self.tracking_qc

    def _track_hash(self):
        if self.tracking_qc is None:
            self.track_consensus_paths()
        parts = [TRACKER_VERSION]
        for v in VESSELS:
            c = self.selected[v]
            parts.append(f"{v}:none" if c is None else f"{v}:{int(c['frame'])}:{_hash_array(np.asarray(c['path'], np.float32))}")
        return _hash_text("|".join(parts))

    def plot_candidates(self):
        if "candidate_figure" in self._done:
            return self.out / "01_consensus_candidates.png"
        if self.tracking_qc is None:
            self.track_consensus_paths()
        out = self.out / "01_consensus_candidates.png"
        meta = self._paths()["candidate_figure"]
        sig = self._track_hash()
        if self.reuse["candidate_figure"] and out.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"] == sig:
                    self._record("candidate_figure", "reused", out)
                    self._done.add("candidate_figure")
                    return out
            except Exception:
                pass

        fig, axs = plt.subplots(3, 3, figsize=(15, 14))
        for r, vessel in enumerate(VESSELS):
            cons = self.consensus[vessel]["consensus"]
            support = self.consensus[vessel]["support"]
            cands = self.candidates[vessel][:3]
            for col in range(3):
                ax = axs[r, col]
                ax.imshow(cons, cmap="magma", vmin=0, vmax=max(0.6, float(np.percentile(cons[cons > 0], 99)) if np.any(cons > 0) else 1))
                if col >= len(cands):
                    ax.text(.5, .5, "No candidate", transform=ax.transAxes, ha="center", va="center")
                    ax.axis("off")
                    continue
                c = cands[col]
                p = c["path"]
                ax.plot(p[:, 1], p[:, 0], linewidth=2)
                tag = "SELECTED" if col == 0 else f"alternate {col}"
                ax.set_title(
                    f"{vessel} {tag}\n{c['path_length_mm']:.1f} mm; sin {c['sinuosity']:.2f}; "
                    f"support {c['mean_support']:.2f}"
                )
                ax.axis("off")
        fig.suptitle("Cross-rotation consensus candidates — persistent evidence only, plaque not used for path generation", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, .97])
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)
        _save_json({"signature": sig}, meta)
        self._record("candidate_figure", "recomputed_and_cached", out)
        self._done.add("candidate_figure")
        return out

    def plot_roadmaps(self):
        if "roadmaps" in self._done:
            return self.out / "02_consensus_coronary_roadmaps.png"
        if self.tracking_qc is None:
            self.track_consensus_paths()
        out = self.out / "02_consensus_coronary_roadmaps.png"
        csv = self.out / "plaque_along_consensus_path.csv"
        meta = self._paths()["roadmaps"]
        sig = self._track_hash()
        if self.reuse["roadmaps"] and out.exists() and csv.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"] == sig:
                    self.along_df = pd.read_csv(csv)
                    self._record("roadmaps", "reused", out)
                    self._done.add("roadmaps")
                    return out
            except Exception:
                pass

        fig, axs = plt.subplots(3, 2, figsize=(18, 14), gridspec_kw={"width_ratios": [1, 1.6]})
        along = []
        for r, vessel in enumerate(VESSELS):
            c = self.selected[vessel]
            status = self.tracking_qc.loc[self.tracking_qc.vessel == vessel, "tracking_status"].iloc[0]
            if c is None:
                for ax in axs[r]:
                    ax.text(.5, .5, f"{vessel}: no consensus path", ha="center", va="center")
                    ax.axis("off")
                continue
            d = self.data[vessel]
            z = int(c["frame"])
            img = np.asarray(d["volume"][z], float)
            plaque = d["plaque"][z] & c["tube"]
            p = c["path"]
            ax = axs[r, 0]
            ax.imshow(img, cmap="gray", vmin=-150, vmax=750)
            ax.plot(p[:, 1], p[:, 0], linewidth=2)
            if np.any(plaque):
                ax.imshow(np.ma.masked_where(~plaque, plaque), cmap="autumn", alpha=.70, interpolation="nearest")
            pad = 32
            y0 = max(0, int(np.floor(p[:, 0].min())) - pad); y1 = min(img.shape[0], int(np.ceil(p[:, 0].max())) + pad + 1)
            x0 = max(0, int(np.floor(p[:, 1].min())) - pad); x1 = min(img.shape[1], int(np.ceil(p[:, 1].max())) + pad + 1)
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            ax.set_title(f"{vessel} best rotation {z} — QC {status}\nfixed cross-rotation consensus path")
            ax.axis("off")

            sp = d["image"].GetSpacing(); sp_yx = (float(sp[1]), float(sp[0]))
            st = straighten(img, plaque, p, sp_yx)
            ax2 = axs[r, 1]
            if st is None:
                ax2.text(.5, .5, "Straightening failed", ha="center", va="center"); ax2.axis("off"); continue
            s, off, strip, pstrip = st
            ax2.imshow(strip, cmap="gray", vmin=-150, vmax=750, aspect="auto", origin="lower", extent=[s[0], s[-1], off[0], off[-1]])
            if np.any(pstrip):
                ax2.imshow(np.ma.masked_where(~pstrip, pstrip), cmap="autumn", alpha=.70, aspect="auto", origin="lower", extent=[s[0], s[-1], off[0], off[-1]], interpolation="nearest")
            ax2.axhline(0, linewidth=1)
            ax2.set_xlabel("Distance along consensus path (mm)")
            ax2.set_ylabel("Transverse distance (mm)")
            ax2.set_title(f"{vessel} straightened consensus path")
            bins = np.arange(0, max(1, math.ceil(s[-1])) + 1, 1.0)
            ci = int(np.argmin(np.abs(off)))
            for a, b in zip(bins[:-1], bins[1:]):
                jj = (s >= a) & (s < b)
                along.append({
                    "vessel": vessel,
                    "start_mm": a,
                    "end_mm": b,
                    "display_plaque_pixels": int(pstrip[:, jj].sum()) if np.any(jj) else 0,
                    "mean_centerline_hu": float(np.nanmean(strip[ci, jj])) if np.any(jj) else np.nan,
                    "tracking_status": status,
                })
        fig.suptitle("OpenPlaque cross-rotation consensus coronary roadmaps — visualization only", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, .97])
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)
        self.along_df = pd.DataFrame(along)
        self.along_df.to_csv(csv, index=False)
        _save_json({"signature": sig}, meta)
        self._record("roadmaps", "recomputed_and_cached", out)
        self._done.add("roadmaps")
        return out

    def _generate_pcat(self, cross, ribbon):
        if self.study is None:
            self.prepare_inputs()
        base = self.root / "PCAT_RCA_10_50"
        cl = pd.read_csv(base / "rca_centerline_smoothed_zyx.csv")
        rad = pd.read_csv(base / "pcat_local_radius_profile.csv")
        aorta_path = next((p for p in [
            self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz",
            self.root / "TotalSegmentator_Validation_v2" / "aorta_series7_totalseg.nii.gz",
        ] if p.exists()), None)
        long_path = next((p for p in [
            self.root / "Combined_TPV_PCAT_All_Metrics_v2" / "pcat_canonical_longitudinal_v2.csv",
            self.root / "PCAT_RCA_10_50_Reproducibility_Lock" / "pcat_canonical_primary_longitudinal.csv",
        ] if p.exists()), None)
        if aorta_path is None or long_path is None:
            raise FileNotFoundError("Missing frozen RCA PCAT inputs")
        source_img, ct, _ = self.study.load_series(7)
        ct = np.asarray(ct, float)
        sp = np.asarray(source_img.GetSpacing(), float)[::-1]
        arc = cl.arc_mm.to_numpy(float)
        pts = cl[["z", "y", "x"]].to_numpy(float)
        pts_mm = pts * sp
        lumen = np.interp(arc, rad.arc_mm.to_numpy(float), rad.lumen_radius_mm.to_numpy(float))
        ai = sitk.ReadImage(str(aorta_path))
        if ai.GetSize() != source_img.GetSize() or not np.allclose(ai.GetSpacing(), source_img.GetSpacing()):
            ai = sitk.Resample(ai, source_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        aorta = sitk.GetArrayFromImage(ai).astype(float)

        def basis(t):
            t = t / np.linalg.norm(t)
            ref = np.array([1., 0., 0.]) if abs(t[0]) < .85 else np.array([0., 1., 0.])
            u = np.cross(t, ref); u /= np.linalg.norm(u)
            v = np.cross(t, u); v /= np.linalg.norm(v)
            return u, v

        def plane(target, half=8., pix=.2):
            i = int(np.argmin(np.abs(arc - target))); i0 = max(0, i - 2); i1 = min(len(arc) - 1, i + 2)
            u, v = basis(pts_mm[i1] - pts_mm[i0])
            c = np.arange(-half, half + 1e-9, pix); U, V = np.meshgrid(c, c, indexing="xy")
            pos = pts_mm[i][None, None, :] + U[..., None] * u + V[..., None] * v
            vox = (pos / sp).reshape(-1, 3).T
            img = map_coordinates(ct, vox, order=1, mode="nearest").reshape(U.shape)
            am = map_coordinates(aorta, vox, order=0, mode="nearest").reshape(U.shape) > .5
            rr = np.sqrt(U ** 2 + V ** 2); lum = float(lumen[i]); outer = lum + .75; shell = 3 * outer
            fat = (rr > outer) & (rr <= shell) & (~am) & (img >= -190) & (img <= -30)
            return float(arc[i]), img, fat, lum, outer, shell, [-half, half, -half, half]

        targets = [10, 20, 30, 40, 50]; planes = [plane(t) for t in targets]
        fig, axs = plt.subplots(1, 5, figsize=(20, 4.5)); im = None
        for ax, d, t in zip(axs, planes, targets):
            a, img, fat, lum, outer, shell, extent = d
            ax.imshow(img, cmap="gray", vmin=-200, vmax=800, extent=extent, origin="lower")
            im = ax.imshow(np.ma.masked_where(~fat, img), cmap="coolwarm", vmin=-120, vmax=-60, alpha=.8, extent=extent, origin="lower")
            for rr, ls in [(lum, "-"), (outer, "--"), (shell, ":")]:
                ax.add_patch(plt.Circle((0, 0), rr, fill=False, linestyle=ls, linewidth=1.7))
            ax.plot(0, 0, "+", markersize=9); ax.set_title(f"RCA {t} mm\nactual {a:.1f} mm"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect("equal")
        axs[0].set_ylabel("mm")
        fig.suptitle("RCA artery-centered PCAT: solid=lumen, dashed=outer wall, dotted=shell edge", fontsize=13)
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=.78, pad=.02).set_label("OpenPlaque PCAT Attenuation (HU)")
        cross.parent.mkdir(parents=True, exist_ok=True); plt.savefig(cross, dpi=180, bbox_inches="tight"); plt.close(fig)

        d = pd.read_csv(long_path); x = (d.arc_start_mm.to_numpy(float) + d.arc_end_mm.to_numpy(float)) / 2; y = d.mean_hu.to_numpy(float)
        fig, ax = plt.subplots(figsize=(12, 3.2)); sc = ax.scatter(x, np.zeros_like(x), c=y, cmap="coolwarm", vmin=-110, vmax=-75, s=260, marker="s")
        ax.set_xlim(10, 50); ax.set_yticks([]); ax.set_xlabel("Distance from RCA ostium (mm)"); ax.set_title("RCA longitudinal OpenPlaque PCAT Attenuation ribbon")
        fig.colorbar(sc, ax=ax, pad=.02).set_label("Mean HU per 1-mm segment"); plt.tight_layout(); plt.savefig(ribbon, dpi=180, bbox_inches="tight"); plt.close(fig)

    def pcat_figures(self):
        if "pcat_figures" in self._done:
            return {"cross_sections": self.out / "03_rca_pcat_cross_sections.png", "ribbon": self.out / "04_rca_longitudinal_pcat_ribbon.png"}
        cache = self._paths()["pcat_figures"]; cache.mkdir(parents=True, exist_ok=True)
        cross, ribbon = cache / "cross.png", cache / "ribbon.png"
        action = None
        if self.reuse["pcat_figures"] and cross.exists() and ribbon.exists():
            action = "reused"
        elif self.reuse["pcat_figures"]:
            previous = [
                self.root / "Image_Driven_Coronary_Tracking_v2_Report",
                self.root / "Image_Driven_Coronary_Tracking_Report",
                self.root / "Coronary_CPR_Roadmap_Report",
                self.root / "User_Friendly_Plaque_PCAT_Report_v3",
            ]
            pc = next((d / n for d in previous for n in ("03_rca_pcat_cross_sections_v2.png", "03_rca_pcat_cross_sections.png", "02_rca_pcat_cross_sections.png", "02_rca_pcat_cross_sections_v3.png") if (d / n).exists()), None)
            pr = next((d / n for d in previous for n in ("04_rca_longitudinal_pcat_ribbon_v2.png", "04_rca_longitudinal_pcat_ribbon.png", "03_rca_longitudinal_pcat_ribbon.png", "03_rca_longitudinal_pcat_ribbon_v3.png") if (d / n).exists()), None)
            if pc is not None and pr is not None:
                shutil.copyfile(pc, cross); shutil.copyfile(pr, ribbon); action = "imported_validated_cache"
        if action is None:
            self._generate_pcat(cross, ribbon); action = "recomputed_and_cached"
        oc, orib = self.out / "03_rca_pcat_cross_sections.png", self.out / "04_rca_longitudinal_pcat_ribbon.png"
        shutil.copyfile(cross, oc); shutil.copyfile(ribbon, orib)
        self._record("pcat_figures", action, cache); self._done.add("pcat_figures")
        return {"cross_sections": oc, "ribbon": orib}

    def plot_dashboard(self):
        if "dashboard" in self._done:
            return self.out / "05_summary_dashboard.png"
        if self.tracking_qc is None:
            self.track_consensus_paths()
        out = self.out / "05_summary_dashboard.png"; summary_csv = self.out / "tracking_report_summary.csv"; meta = self._paths()["dashboard"]
        sig = _hash_text(self._track_hash() + self.tpv.to_csv(index=False) + self.pcat.to_csv(index=False) + self.ps.to_csv(index=False))
        if self.reuse["dashboard"] and out.exists() and summary_csv.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"] == sig:
                    self._record("dashboard", "reused", out); self._done.add("dashboard"); return out
            except Exception:
                pass
        total = self.tpv[self.tpv.vessel == "TOTAL"].iloc[0]; primary = self.pcat.iloc[0]
        cmin, cmax = float(self.ps.pcat_mean_hu.min()), float(self.ps.pcat_mean_hu.max()); directional = -94.345186; fmin, fmax = min(cmin, directional), max(cmax, directional)
        fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        ax1 = fig.add_subplot(gs[0, 0]); vv = self.tpv[self.tpv.vessel != "TOTAL"]
        ax1.bar(vv.vessel, vv.canonical_refined_tpv_mm3); ax1.set_ylabel("Refined TPV (mm³)"); ax1.set_title("Canonical plaque volume by artery")
        ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
        ax2.text(.03, .95, f"Total refined TPV: {total.canonical_refined_tpv_mm3:.0f} mm³\nRaw TPV: {total.raw_tpv_mm3:.0f} mm³\nTPV sensitivity: {total.sensitivity_min_mm3:.0f}–{total.sensitivity_max_mm3:.0f} mm³\n\nRCA 10–50 mm OpenPlaque PCAT Attenuation: {primary.pcat_mean_hu:.2f} HU\nCircular-margin range: {cmin:.2f} to {cmax:.2f} HU\nFull tested geometry: {fmin:.2f} to {fmax:.2f} HU", va="top", fontsize=12.5)
        ax3 = fig.add_subplot(gs[1, :]); ax3.axis("off")
        cols = ["vessel", "selected_frame", "path_length_mm", "sinuosity", "mean_cross_rotation_support", "tracking_status"]
        tab = self.tracking_qc[cols].copy()
        for c in ("path_length_mm", "sinuosity", "mean_cross_rotation_support"):
            tab[c] = tab[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        table = ax3.table(cellText=tab.values, colLabels=["Vessel", "Best rotation", "Length mm", "Sinuosity", "Persistence", "QC"], loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1, 1.6)
        ax3.set_title("Cross-rotation consensus tracking QC — plaque is not used to generate the path", pad=12)
        fig.suptitle("OpenPlaque summary: validated quantitative endpoints + consensus tracking QC", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, .96]); plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); plt.close(fig)
        pd.DataFrame([{
            "canonical_total_tpv_mm3": float(total.canonical_refined_tpv_mm3),
            "raw_total_tpv_mm3": float(total.raw_tpv_mm3),
            "tpv_sensitivity_min_mm3": float(total.sensitivity_min_mm3),
            "tpv_sensitivity_max_mm3": float(total.sensitivity_max_mm3),
            "rca_pcat_mean_hu": float(primary.pcat_mean_hu),
            "pcat_full_geometry_min_hu": fmin,
            "pcat_full_geometry_max_hu": fmax,
            "tracker_version": TRACKER_VERSION,
        }]).to_csv(summary_csv, index=False)
        _save_json({"signature": sig}, meta); self._record("dashboard", "recomputed_and_cached", out); self._done.add("dashboard")
        return out

    def package_report(self):
        if "report_package" in self._done:
            return self.out / "OPENPLAQUE_CROSS_ROTATION_CONSENSUS_REPORT_BACK.zip"
        self.plot_candidates(); self.plot_roadmaps(); self.pcat_figures(); self.plot_dashboard()
        html_path = self.out / "OPENPLAQUE_CROSS_ROTATION_CONSENSUS_REPORT.html"
        zip_path = self.out / "OPENPLAQUE_CROSS_ROTATION_CONSENSUS_REPORT_BACK.zip"
        meta = self._paths()["report_package"]
        names = [
            "01_consensus_candidates.png",
            "02_consensus_coronary_roadmaps.png",
            "03_rca_pcat_cross_sections.png",
            "04_rca_longitudinal_pcat_ribbon.png",
            "05_summary_dashboard.png",
            "tracking_qc.csv",
            "cache_qc.csv",
            "cache_provenance.csv",
            "plaque_along_consensus_path.csv",
            "tracking_report_summary.csv",
        ]
        sig = _hash_text("|".join(f"{n}:{_hash_file(self.out / n) if (self.out / n).exists() else 'missing'}" for n in names))
        if self.reuse["report_package"] and zip_path.exists() and html_path.exists() and meta.exists():
            try:
                if _load_json(meta)["signature"] == sig:
                    self._record("report_package", "reused", zip_path); self._done.add("report_package"); return zip_path
            except Exception:
                pass

        def tag(name):
            p = self.out / name
            if not p.exists():
                return ""
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<h2>{name}</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto">'

        imgs = names[:5]
        html = (
            "<html><head><meta charset='utf-8'><title>OpenPlaque cross-rotation consensus tracking</title></head><body>"
            "<h1>OpenPlaque — cross-rotation consensus tracking</h1>"
            "<p><b>Research use only.</b> The coronary path is generated from evidence persistent across CPR rotations. Plaque is displayed after tracking and is not used to create the path. Canonical TPV is unchanged. CPR plaque views and source-volume PCAT are not spatially co-registered.</p>"
            + "".join(tag(x) for x in imgs)
            + "<h2>Tracking QC</h2>" + self.tracking_qc.to_html(index=False)
            + "<h2>Cache provenance</h2>" + pd.DataFrame(self.provenance).to_html(index=False)
            + "</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")
        bundle = names + [html_path.name]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for name in bundle:
                p = self.out / name
                if p.exists():
                    z.write(p, arcname=name)
        _save_json({"signature": sig}, meta)
        self._record("report_package", "recomputed_and_cached", zip_path); self._done.add("report_package")
        return zip_path
