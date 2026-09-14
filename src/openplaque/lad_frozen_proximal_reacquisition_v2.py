from __future__ import annotations

"""JPEG-lossless-safe source loading for frozen-LAD proximal reacquisition.

Scientific tracking logic is inherited unchanged from v1.0.  This wrapper only
changes source-volume loading.  It first reuses the CT cache explicitly recorded
by the completed coronary-reconciliation geometry metadata.  If no compatible
cache is available, Series 7 is decoded by SimpleITK/GDCM rather than pydicom's
pixel_array, avoiding an optional pylibjpeg/gdcm Python decoder dependency.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .lad_frozen_proximal_reacquisition import (
    LADFrozenProximalReacquisitionWorkflow,
    _series7_files,
    _geometry_from_dicom,
    _json_write,
)

ALGORITHM_VERSION = "lad-frozen-proximal-reacquisition-v1.1-sitk-decode"


def _compatible_npy(path, expected_shape):
    path = Path(path)
    if not path.exists():
        return None
    try:
        mm = np.load(path, mmap_mode="r")
    except Exception:
        return None
    return path if tuple(mm.shape) == tuple(expected_shape) else None


def _reconciliation_recorded_cache(root, expected_shape):
    """Return the CT cache named by the successful reconciliation geometry JSON."""
    meta_fp = Path(root) / "Cache" / "Coronary_Anatomy_Reconciliation_v1" / "series7_dicom_geometry.json"
    if not meta_fp.exists():
        return None
    try:
        meta = json.loads(meta_fp.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = []
    if meta.get("ct_cache_source"):
        candidates.append(Path(meta["ct_cache_source"]))
    candidates.append(meta_fp.parent / "series7_int16.npy")
    for fp in candidates:
        hit = _compatible_npy(fp, expected_shape)
        if hit is not None:
            return hit
    return None


def _load_or_build_source_ct_sitk(root, cache):
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    own = cache / "series7_int16.npy"
    meta_fp = cache / "series7_geometry.json"

    if own.exists() and meta_fp.exists():
        try:
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
            hit = _compatible_npy(own, meta.get("shape", ()))
            if hit is not None and "positions_lps_mm" in meta and "image_orientation_patient" in meta:
                return np.load(hit, mmap_mode="r"), meta, "reused_own_cache"
        except Exception:
            pass

    files = _series7_files(root)
    meta = _geometry_from_dicom(files)
    expected = tuple(meta["shape"])

    # Highest priority: exact cache used by the successful reconciliation run.
    hit = _reconciliation_recorded_cache(root, expected)
    if hit is not None:
        meta["ct_cache_source"] = str(hit)
        meta["ct_cache_action"] = "reused_reconciliation_recorded_cache"
        _json_write(meta, meta_fp)
        return np.load(hit, mmap_mode="r"), meta, f"reused_reconciliation_recorded_cache:{hit}"

    # Then try known compatible OpenPlaque caches.
    candidates = [
        root / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.npy",
        root / "Cache" / "LAD_Confirmed_Backtrack_v1" / "series7_int16.npy",
        root / "Cache" / "LAD_Origin_Backtrack_v1" / "series7_int16.npy",
        root / "Cache" / "Left_Coronary_Ostium_Neck_v1" / "series7_int16.npy",
    ]
    for fp in candidates:
        hit = _compatible_npy(fp, expected)
        if hit is not None:
            meta["ct_cache_source"] = str(hit)
            meta["ct_cache_action"] = "reused_shape_verified_cache"
            _json_write(meta, meta_fp)
            return np.load(hit, mmap_mode="r"), meta, f"reused_shape_verified_cache:{hit}"

    # Last resort: let SimpleITK/GDCM decode the JPEG-lossless DICOM series.
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

    meta["ct_cache_source"] = str(own)
    meta["ct_cache_action"] = "decoded_with_simpleitk_gdcm"
    _json_write(meta, meta_fp)
    return np.load(own, mmap_mode="r"), meta, "decoded_with_simpleitk_gdcm"


class LADFrozenProximalReacquisitionWorkflowV2(LADFrozenProximalReacquisitionWorkflow):
    def load_source_ct(self):
        self.ct, self.meta, action = _load_or_build_source_ct_sitk(self.root, self.cache)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", action, self.cache / "series7_geometry.json")
        return self.ct
