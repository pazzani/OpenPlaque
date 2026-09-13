from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .artery_detection import detect_artery_series
from .boundary import refine_plaque_mask
from .cache_utils import CACHE_VERSION, hash_array, load_json, save_json
from .cache_visuals import CacheVisualMixin
from .cpr_tracking import path_tube
from .segmentation import segment_vessel
from .study import OpenPlaqueStudy
from .tracking_report import (
    FALLBACK_SERIES,
    VESSELS,
    _ensure_model,
    _plaque_mask_from_nifti,
    _save_mask,
)
from .tracking_workflow import TrackingWorkflow


class CacheControlledTrackingWorkflow(CacheVisualMixin, TrackingWorkflow):
    """Image-driven workflow with explicit, user-controlled persistent caches.

    For each component:
      * reuse=True + valid cache -> load cache
      * reuse=True + no valid cache -> recompute and cache
      * reuse=False -> force recompute and overwrite cache

    In-memory results are always reused within the current notebook run.
    """

    COMPONENTS = (
        "series_selection",
        "plaque_masks",
        "tracking",
        "roadmaps",
        "pcat_figures",
        "dashboard",
        "report_package",
    )

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root)
        self.reuse = {name: True for name in self.COMPONENTS}
        self.reuse.update({k: bool(v) for k, v in (reuse or {}).items()})
        unknown = set(self.reuse) - set(self.COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown cache controls: {sorted(unknown)}")

        self.cache_root = self.root / "Cache" / "Image_Driven_Tracking_Cache_Controlled"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.provenance = []
        self._done = set()

    def _record(self, component, action, path, note=""):
        self.provenance.append(
            {
                "component": component,
                "reuse_requested": self.reuse[component],
                "action": action,
                "path": str(path),
                "note": note,
            }
        )
        pd.DataFrame(self.provenance).to_csv(
            self.out / "cache_provenance.csv", index=False
        )

    def _paths(self):
        return {
            "series_selection": self.cache_root / "series_selection.json",
            "plaque_masks": self.cache_root / "plaque",
            "tracking": self.cache_root / "tracking.pkl",
            "roadmaps": self.cache_root / "roadmaps_meta.json",
            "pcat_figures": self.cache_root / "pcat",
            "dashboard": self.cache_root / "dashboard_meta.json",
            "report_package": self.cache_root / "report_meta.json",
        }

    def _plaque_candidates(self, vessel):
        dedicated = self._paths()["plaque_masks"] / f"{vessel}_canonical_refined.nii.gz"
        candidates = [
            dedicated,
            self.root / "Cache" / "canonical_plaque_v2" / f"{vessel}_canonical_refined.nii.gz",
        ]
        for folder in (self.root / "Segmentations", self.root / "segmentations"):
            candidates.extend(
                [
                    folder / f"{vessel}_refined_plaque_segmentation.nii.gz",
                    folder / f"{vessel}_volume_refined_plaque_segmentation.nii.gz",
                ]
            )
        return candidates

    def cache_status(self):
        p = self._paths()
        exists = {
            "series_selection": p["series_selection"].exists(),
            "plaque_masks": all(
                any(candidate.exists() for candidate in self._plaque_candidates(v))
                for v in VESSELS
            ),
            "tracking": p["tracking"].exists(),
            "roadmaps": p["roadmaps"].exists()
            and (self.out / "02_straightened_coronary_roadmaps.png").exists(),
            "pcat_figures": (p["pcat_figures"] / "cross.png").exists()
            and (p["pcat_figures"] / "ribbon.png").exists(),
            "dashboard": p["dashboard"].exists()
            and (self.out / "05_summary_dashboard.png").exists(),
            "report_package": p["report_package"].exists()
            and (self.out / "OPENPLAQUE_IMAGE_DRIVEN_TRACKING_REPORT_BACK.zip").exists(),
        }
        return pd.DataFrame(
            [
                {
                    "component": name,
                    "reuse": self.reuse[name],
                    "cache_exists": exists[name],
                    "planned_action": (
                        "reuse_if_valid"
                        if self.reuse[name] and exists[name]
                        else "recompute_and_cache"
                    ),
                    "cache_path": str(p[name]),
                }
                for name in self.COMPONENTS
            ]
        )

    def prepare_inputs(self):
        if "series_selection" in self._done:
            return self.series_map

        drive_zip = self.root / "Full_DICOM.zip"
        local_zip = Path("/content/Full_DICOM.zip")
        if not drive_zip.exists():
            raise FileNotFoundError(drive_zip)
        if not local_zip.exists() or local_zip.stat().st_size != drive_zip.stat().st_size:
            shutil.copyfile(drive_zip, local_zip)

        self.study = OpenPlaqueStudy(
            str(local_zip), extract_root="/content/full_dicom_tracking_cache_controlled"
        )
        cache = self._paths()["series_selection"]
        loaded = False

        if self.reuse["series_selection"] and cache.exists():
            try:
                manifest = load_json(cache)
                if int(manifest["dicom_size"]) != int(drive_zip.stat().st_size):
                    raise ValueError("DICOM ZIP size changed")
                self.series_map = {k: int(v) for k, v in manifest["series_map"].items()}
                for vessel in VESSELS:
                    self.study.load_series(self.series_map[vessel])
                loaded = True
                self._record("series_selection", "reused", cache)
            except Exception as exc:
                self._record("series_selection", "cache_invalid", cache, repr(exc))

        if not loaded:
            self.series_map, _ = detect_artery_series(
                self.study, fallback_series=FALLBACK_SERIES, return_candidates=True
            )
            self.series_map = {k: int(v) for k, v in self.series_map.items()}
            save_json(
                {
                    "dicom_size": int(drive_zip.stat().st_size),
                    "series_map": self.series_map,
                },
                cache,
            )
            self._record("series_selection", "recomputed_and_cached", cache)

        metrics_dir = self.root / "Combined_TPV_PCAT_All_Metrics_v2"
        self.tpv = pd.read_csv(metrics_dir / "tpv_metrics_by_vessel_v2.csv")
        self.pcat = pd.read_csv(metrics_dir / "pcat_canonical_primary_v2.csv")
        self.ps = pd.read_csv(metrics_dir / "pcat_circular_sensitivity_v2.csv")
        self._done.add("series_selection")
        return self.series_map

    def _compute_plaque(self, vessel, image, volume):
        _ensure_model(self.root)
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
        return refined.refined_mask == 2

    def load_cached_plaque(self):
        if "plaque_masks" in self._done:
            return self.cache_qc
        if self.study is None:
            self.prepare_inputs()

        dedicated_dir = self._paths()["plaque_masks"]
        dedicated_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        rows = []

        for vessel in VESSELS:
            image, volume, _ = self.study.load_series(self.series_map[vessel])
            expected = float(
                self.tpv.loc[
                    self.tpv.vessel == vessel, "canonical_refined_tpv_mm3"
                ].iloc[0]
            )
            mask = None
            source_path = None

            if self.reuse["plaque_masks"]:
                for candidate in self._plaque_candidates(vessel):
                    mask = _plaque_mask_from_nifti(
                        candidate, image, expected_mm3=expected
                    )
                    if mask is not None:
                        source_path = candidate
                        break

            dedicated = dedicated_dir / f"{vessel}_canonical_refined.nii.gz"
            if mask is None:
                mask = self._compute_plaque(vessel, image, volume)
                _save_mask(mask, image, dedicated)
                source = "forced_recompute" if not self.reuse["plaque_masks"] else "cache_miss_recompute"
                source_path = dedicated
            else:
                source = "cache"
                if source_path != dedicated:
                    _save_mask(mask, image, dedicated)

            data[vessel] = {
                "image": image,
                "volume": np.asarray(volume),
                "plaque": mask,
            }
            rows.append(
                {
                    "vessel": vessel,
                    "source": source,
                    "path": str(source_path),
                    "normalized_cache": str(dedicated),
                }
            )

        self.data = data
        self.cache_qc = pd.DataFrame(rows)
        self.cache_qc.to_csv(self.out / "cache_qc.csv", index=False)
        action = (
            "reused_or_filled_missing"
            if self.reuse["plaque_masks"]
            else "recomputed_and_cached"
        )
        self._record("plaque_masks", action, dedicated_dir)
        self._done.add("plaque_masks")
        return self.cache_qc

    def _compact(self, record):
        keys = (
            "frame",
            "image_score",
            "path_length_mm",
            "mean_path_hu",
            "mean_evidence",
            "intensity_fit",
            "threshold_percentile",
            "plaque_tube_voxels",
            "plaque_frame_voxels",
            "path",
        )
        return {k: record[k] for k in keys}

    def _inflate(self, vessel, record):
        out = dict(record)
        out["path"] = np.asarray(out["path"], float)
        spacing = self.data[vessel]["image"].GetSpacing()
        sp_yx = (float(spacing[1]), float(spacing[0]))
        shape = self.data[vessel]["volume"][int(out["frame"])].shape
        out["tube"], out["path_mask"] = path_tube(
            out["path"], shape, sp_yx, radius_mm=5.0
        )
        return out

    def track_coronaries(self):
        if "tracking" in self._done:
            return self.tracking_qc
        if self.data is None:
            self.load_cached_plaque()

        cache = self._paths()["tracking"]
        plaque_signature = {
            vessel: hash_array(np.asarray(self.data[vessel]["plaque"], np.uint8))
            for vessel in VESSELS
        }

        if self.reuse["tracking"] and cache.exists():
            try:
                with cache.open("rb") as handle:
                    payload = pickle.load(handle)
                if payload["version"] != CACHE_VERSION:
                    raise ValueError("tracking cache version changed")
                if payload["series_map"] != self.series_map:
                    raise ValueError("series selection changed")
                if payload["plaque_signature"] != plaque_signature:
                    raise ValueError("plaque dependency changed")

                self.all_candidates = {
                    vessel: [
                        self._inflate(vessel, item)
                        for item in payload["all_candidates"][vessel]
                    ]
                    for vessel in VESSELS
                }
                self.selected = {
                    vessel: self._inflate(vessel, payload["selected"][vessel])
                    for vessel in VESSELS
                }
                self.tracking_qc = pd.DataFrame(payload["tracking_qc"])
                self.tracking_qc.to_csv(self.out / "tracking_qc.csv", index=False)
                self._record("tracking", "reused", cache)
                self._done.add("tracking")
                return self.tracking_qc
            except Exception as exc:
                self._record("tracking", "cache_invalid", cache, repr(exc))

        qc = super().track_coronaries()
        payload = {
            "version": CACHE_VERSION,
            "series_map": self.series_map,
            "plaque_signature": plaque_signature,
            "all_candidates": {
                vessel: [self._compact(x) for x in self.all_candidates[vessel]]
                for vessel in VESSELS
            },
            "selected": {
                vessel: self._compact(self.selected[vessel]) for vessel in VESSELS
            },
            "tracking_qc": qc.to_dict(orient="records"),
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
        self._record("tracking", "recomputed_and_cached", cache)
        self._done.add("tracking")
        return qc
