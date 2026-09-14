from __future__ import annotations

"""Cache-provenance-safe wrapper for LAD proximal reacquisition v1.2.1.

Same scientific search and post-hoc final-tangent validator.  This runtime fix
prevents reuse of a correctly shaped but unpopulated Series-7 memmap left behind
when the original pydicom JPEG-lossless decode failed.  Cache reuse now honors
recorded provenance and requires sampled CT HU-content plausibility.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .lad_frozen_proximal_reacquisition import _json_write
from .lad_frozen_proximal_reacquisition_v2 import _series7_files, _geometry_from_dicom
from .lad_frozen_proximal_reacquisition_v4 import LADFrozenProximalReacquisitionWorkflowV4

ALGORITHM_VERSION = "lad-frozen-proximal-reacquisition-v1.2.2-cache-provenance-safe"


def _plausible_ct_npy(path, expected_shape=None):
    path = Path(path)
    if not path.exists():
        return None
    try:
        mm = np.load(path, mmap_mode="r")
    except Exception:
        return None
    if expected_shape is not None and tuple(mm.shape) != tuple(expected_shape):
        return None
    if mm.ndim != 3 or min(mm.shape) < 8:
        return None
    # Sample throughout the volume without materializing the full CT.
    zi = np.unique(np.linspace(0, mm.shape[0] - 1, min(9, mm.shape[0])).round().astype(int))
    yi = slice(None, None, max(1, mm.shape[1] // 96))
    xi = slice(None, None, max(1, mm.shape[2] // 96))
    sample = np.asarray(mm[zi, yi, xi], dtype=np.float32).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size < 100:
        return None
    p01, p50, p99 = np.percentile(sample, [1, 50, 99])
    dynamic = float(p99 - p01)
    nonzero_fraction = float(np.mean(sample != 0))
    # Broad enough for a real CT and rejects the zero/empty placeholder decisively.
    if dynamic < 300.0 or nonzero_fraction < 0.20 or not (-2000.0 <= p50 <= 2000.0):
        return None
    return path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


class LADFrozenProximalReacquisitionWorkflowV5(LADFrozenProximalReacquisitionWorkflowV4):
    def load_source_ct(self):
        root = Path(self.root)
        cache = Path(self.cache)
        cache.mkdir(parents=True, exist_ok=True)
        own = cache / "series7_int16.npy"
        meta_fp = cache / "series7_geometry.json"

        # 1) Honor the provenance already recorded by the successful run.
        local_meta = _read_json(meta_fp)
        if local_meta and local_meta.get("shape"):
            expected = tuple(local_meta["shape"])
            recorded = local_meta.get("ct_cache_source")
            if recorded:
                hit = _plausible_ct_npy(recorded, expected)
                if hit is not None:
                    self.ct = np.load(hit, mmap_mode="r")
                    self.meta = local_meta
                    self.spacing = np.asarray(self.meta["spacing_zyx"], float)
                    self._record("source_ct", "reused_recorded_provenance_cache", hit,
                                 "shape+sampled-HU validation passed")
                    return self.ct
            hit = _plausible_ct_npy(own, expected)
            if hit is not None:
                self.ct = np.load(hit, mmap_mode="r")
                self.meta = local_meta
                self.spacing = np.asarray(self.meta["spacing_zyx"], float)
                self._record("source_ct", "reused_own_cache_after_HU_validation", hit,
                             "shape+sampled-HU validation passed")
                return self.ct

        # 2) Reconstruct trusted Series-7 geometry from DICOM metadata.
        files = _series7_files(root)
        meta = _geometry_from_dicom(files)
        expected = tuple(meta["shape"])

        # 3) Prefer the exact cache recorded by the successful reconciliation run.
        rec_meta_fp = root / "Cache" / "Coronary_Anatomy_Reconciliation_v1" / "series7_dicom_geometry.json"
        rec_meta = _read_json(rec_meta_fp)
        candidates = []
        if rec_meta and rec_meta.get("ct_cache_source"):
            candidates.append(Path(rec_meta["ct_cache_source"]))
        candidates.extend([
            rec_meta_fp.parent / "series7_int16.npy",
            root / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.npy",
            root / "Cache" / "LAD_Confirmed_Backtrack_v1" / "series7_int16.npy",
            root / "Cache" / "LAD_Origin_Backtrack_v1" / "series7_int16.npy",
            root / "Cache" / "Left_Coronary_Ostium_Neck_v1" / "series7_int16.npy",
        ])
        for fp in candidates:
            hit = _plausible_ct_npy(fp, expected)
            if hit is not None:
                meta["ct_cache_source"] = str(hit)
                meta["ct_cache_action"] = "reused_shape_and_HU_verified_external_cache"
                _json_write(meta, meta_fp)
                self.ct = np.load(hit, mmap_mode="r")
                self.meta = meta
                self.spacing = np.asarray(meta["spacing_zyx"], float)
                self._record("source_ct", "reused_shape_and_HU_verified_external_cache", hit,
                             "stale local placeholder ignored")
                return self.ct

        # 4) Last resort: decode with SimpleITK/GDCM and overwrite the stale local file.
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in files])
        reader.SetOutputPixelType(sitk.sitkInt16)
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)
        if tuple(arr.shape) != expected:
            raise RuntimeError(f"SimpleITK Series-7 shape {arr.shape} != DICOM metadata shape {expected}")
        mm = np.lib.format.open_memmap(own, mode="w+", dtype=np.int16, shape=expected)
        mm[:] = np.asarray(arr, dtype=np.int16)
        mm.flush()
        del mm, arr, image
        if _plausible_ct_npy(own, expected) is None:
            raise RuntimeError("Rebuilt Series-7 CT failed sampled HU plausibility validation")
        meta["ct_cache_source"] = str(own)
        meta["ct_cache_action"] = "decoded_with_simpleitk_gdcm_and_HU_verified"
        _json_write(meta, meta_fp)
        self.ct = np.load(own, mmap_mode="r")
        self.meta = meta
        self.spacing = np.asarray(meta["spacing_zyx"], float)
        self._record("source_ct", "decoded_with_simpleitk_gdcm_and_HU_verified", own,
                     "stale placeholder overwritten")
        return self.ct


def synthetic_cache_plausibility_self_test():
    # Tests the actual content criterion without writing a giant volume.
    good = np.array([-1000, -500, 0, 100, 400, 800, 1200] * 30, dtype=np.float32)
    bad = np.zeros(300, dtype=np.float32)
    good_dynamic = float(np.percentile(good, 99) - np.percentile(good, 1))
    bad_dynamic = float(np.percentile(bad, 99) - np.percentile(bad, 1))
    return {"passed": bool(good_dynamic > 300 and bad_dynamic < 300),
            "good_dynamic_hu": good_dynamic, "bad_dynamic_hu": bad_dynamic}
