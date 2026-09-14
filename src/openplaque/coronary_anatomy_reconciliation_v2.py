from __future__ import annotations

"""Input-geometry-safe wrapper for coronary anatomy reconciliation.

Scientific reconciliation logic is inherited unchanged from v1.0.  This wrapper
only fixes source-Series-7 provenance: when no prior cache JSON contains full
DICOM geometry, it reconstructs ImagePositionPatient / ImageOrientationPatient
from Full_DICOM.zip, verifies the cached CT shape against that series, and writes
an explicit reusable geometry cache.
"""

import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from .study import OpenPlaqueStudy
from .coronary_anatomy_reconciliation import (
    CoronaryAnatomyReconciliationWorkflow,
    _first_existing,
    synthetic_reconciliation_self_test,
)

ALGORITHM_VERSION = "coronary-anatomy-reconciliation-v1.1-dicom-geometry-safe"
SOURCE_SERIES = 7


class CoronaryAnatomyReconciliationWorkflowV2(CoronaryAnatomyReconciliationWorkflow):
    def _series7_files(self):
        root = self.root
        src = root / "Full_DICOM.zip"
        if not src.exists():
            raise FileNotFoundError(f"Required source archive not found: {src}")

        local = Path("/content/Full_DICOM_reconciliation.zip")
        if (not local.exists()) or local.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, local)

        extract_root = "/content/full_dicom_coronary_reconciliation"
        shutil.rmtree(extract_root, ignore_errors=True)
        study = OpenPlaqueStudy(str(local), extract_root=extract_root)
        matches = [s for s in study.series if s.get("series_number") == SOURCE_SERIES]
        if not matches:
            raise RuntimeError("Source CCTA series 7 not found in Full_DICOM.zip")

        reader = sitk.ImageSeriesReader()
        files = list(reader.GetGDCMSeriesFileNames(matches[0]["folder"], matches[0]["uid"]))
        if not files:
            raise RuntimeError("No DICOM files resolved for source CCTA series 7")
        return files

    def _build_geometry_meta(self, files):
        ds0 = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        positions = []
        for fp in files:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            positions.append([float(x) for x in ds.ImagePositionPatient])

        ps = [float(x) for x in ds0.PixelSpacing]
        pos = np.asarray(positions, float)
        if len(pos) > 1:
            dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        else:
            dz = float(getattr(ds0, "SliceThickness", 1.0))

        orient = [float(x) for x in ds0.ImageOrientationPatient]
        return {
            "shape": [len(files), int(ds0.Rows), int(ds0.Columns)],
            "spacing_zyx": [dz, ps[0], ps[1]],
            "positions_lps_mm": positions,
            "image_orientation_patient": orient,
            "source": "reconstructed directly from Full_DICOM.zip series 7",
        }

    def _candidate_ct_caches(self):
        r = self.root
        return [
            r / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.npy",
            r / "Cache" / "LAD_Confirmed_Backtrack_v1" / "series7_int16.npy",
            r / "Cache" / "LAD_Origin_Backtrack_v1" / "series7_int16.npy",
            r / "Cache" / "Left_Coronary_Ostium_Neck_v1" / "series7_int16.npy",
            self.cache / "series7_int16.npy",
        ]

    def _ensure_source_series7(self):
        geometry_fp = self.cache / "series7_dicom_geometry.json"
        own_ct = self.cache / "series7_int16.npy"

        # If our own reconciliation cache is already complete, reuse it directly.
        if own_ct.exists() and geometry_fp.exists():
            meta = json.loads(geometry_fp.read_text())
            mm = np.load(own_ct, mmap_mode="r")
            if tuple(meta.get("shape", [])) == tuple(mm.shape) and \
               "positions_lps_mm" in meta and "image_orientation_patient" in meta:
                return own_ct, geometry_fp

        files = self._series7_files()
        meta = self._build_geometry_meta(files)
        expected_shape = tuple(meta["shape"])

        # Prefer an already materialized int16 Series-7 volume, but verify shape.
        chosen = None
        for fp in self._candidate_ct_caches():
            if not fp.exists():
                continue
            try:
                mm = np.load(fp, mmap_mode="r")
            except Exception:
                continue
            if tuple(mm.shape) == expected_shape:
                chosen = fp
                break

        if chosen is None:
            ds0 = pydicom.dcmread(files[0], force=True)
            mm = np.lib.format.open_memmap(
                own_ct, mode="w+", dtype=np.int16, shape=expected_shape
            )
            for i, fp in enumerate(files):
                ds = pydicom.dcmread(fp, force=True)
                arr = ds.pixel_array.astype(np.float32, copy=False)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                hu = arr * slope + intercept
                mm[i] = np.rint(np.clip(hu, -32768, 32767)).astype(np.int16)
                if i % 100 == 0:
                    mm.flush()
            mm.flush()
            del mm
            gc.collect()
            chosen = own_ct
            meta["ct_cache_action"] = "rebuilt_from_full_dicom"
        else:
            meta["ct_cache_action"] = "reused_shape_verified_cache"
            meta["ct_cache_source"] = str(chosen)

        geometry_fp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return chosen, geometry_fp

    def _resolve_inputs(self):
        r = self.root
        files = {
            "RCA_VALIDATED": _first_existing([
                r / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv",
                r / "PCAT_RCA_10_50_Reproducibility_Lock" / "rca_centerline_smoothed_zyx.csv",
            ]),
            "LAD_FROZEN": _first_existing([
                r / "Cache" / "LAD_Confirmed_Backtrack_v1" / "frozen_lad_centerline.csv",
                r / "LAD_Confirmed_Backtrack_Report" / "frozen_lad_centerline.csv",
            ]),
            "TRUNK_CANDIDATE": _first_existing([
                r / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv",
                r / "Cache" / "LAD_Takeoff_Confirmation_v1" / "trunk_centerline.csv",
            ]),
            "SECONDARY_CANDIDATE": _first_existing([
                r / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv",
                r / "Cache" / "Secondary_Branch_Lateral_Divergence_v1" / "branch_centerline.csv",
            ]),
        }
        missing = [k for k, p in files.items() if p is None]
        if missing:
            raise FileNotFoundError("Missing reconciliation input(s): " + ", ".join(missing))

        ct, meta = self._ensure_source_series7()
        return files, ct, meta
