from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "alternate-series-left-coronary-origin-validation-v1.0"
OUTPUT_DIRNAME = "Alternate_Series_Left_Coronary_Origin_Validation_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
AORTA_MASK = Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres/aorta.nii.gz")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"

MAX_PRESELECT = 8
MAX_ALTERNATE_SERIES = 4
PATH_TOLERANCE_MM = 2.0
AORTA_CONTACT_TOLERANCE_MM = 1.5
MIN_RCA_SUPPORT = 0.60
MIN_LAD_SUPPORT = 0.60
MIN_C6_SUPPORT = 0.50
MIN_C7_SUPPORT_DESCRIPTIVE = 0.50
MAX_REGISTRATION_TRANSLATION_MM = 6.0
RCA_CONTACT_RADIUS_MM = 5.0
MIN_DISTINCT_FROM_RCA_MM = 8.0
MAX_LEFT_ANCHOR_TO_CONTACT_MM = 25.0
MIN_CONTACT_VOXELS = 10
CROSS_SERIES_CONTACT_CONCORDANCE_MM = 5.0
ROOT_ROI_HALF_MM = 32.0

STATUS_CONCORDANT = "ALTERNATE_SERIES_LEFT_OSTIUM_CONCORDANT"
STATUS_SINGLE = "SINGLE_ALTERNATE_SERIES_LEFT_OSTIUM_CANDIDATE"
STATUS_DISCORDANT = "ALTERNATE_SERIES_LEFT_CONTACTS_DISCORDANT"
STATUS_NEGATIVE = "NO_DISTINCT_LEFT_OSTIUM_ACROSS_ALTERNATE_SERIES"
STATUS_INSUFFICIENT = "INSUFFICIENT_INTERPRETABLE_ALTERNATE_SERIES"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def discover_dicom_root(drive_root="/content/drive/MyDrive/OpenPlaque", explicit=None):
    """Resolve the same-exam DICOM study without assuming it lives under OpenPlaque."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit DICOM root does not exist: {p}")

    root = Path(drive_root)
    mydrive = root.parent
    candidates = [
        mydrive / "CCTA" / "DICOM" / "3221",
        mydrive / "DICOM" / "3221",
        root / "DICOM" / "3221",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate the CCTA DICOM study. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return int(default)


def _text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v)


def _load_lps_csv(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _resample_polyline(p, step_mm=0.5):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy()
    q = np.arange(0.0, a[-1] + 1e-9, float(step_mm))
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


@dataclass
class Geometry:
    spacing_xyz: np.ndarray
    origin: np.ndarray
    direction: np.ndarray

    @property
    def spacing_zyx(self):
        return self.spacing_xyz[::-1]

    def xyz_to_zyx_float(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        inv = np.linalg.inv(self.direction)
        xyz = ((pts - self.origin) @ inv.T) / self.spacing_xyz
        return xyz[:, ::-1]

    def zyx_to_xyz(self, zyx):
        zyx = np.atleast_2d(np.asarray(zyx, float))
        xyz = zyx[:, ::-1] * self.spacing_xyz
        return self.origin + xyz @ self.direction.T


def _source_geometry(meta):
    spacing_zyx = np.asarray(meta["spacing_zyx"], float)
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.column_stack([row, col, slc])
    return Geometry(
        spacing_xyz=spacing_zyx[::-1],
        origin=np.asarray(meta["positions_lps_mm"][0], float),
        direction=D,
    )


def _source_reference(root):
    root = Path(root)
    cache = root / SOURCE_CACHE
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = _source_geometry(meta)
    img = sitk.GetImageFromArray(np.asarray(arr))
    img.SetSpacing(tuple(float(v) for v in geom.spacing_xyz))
    img.SetOrigin(tuple(float(v) for v in geom.origin))
    img.SetDirection(tuple(float(v) for v in geom.direction.ravel()))
    return img, geom, tuple(int(v) for v in arr.shape), meta


def _iter_leaf_dirs(root):
    root = Path(root)
    for d, _, files in os.walk(root):
        if files:
            yield Path(d), [Path(d) / f for f in files]


def _read_header(path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _natural_sort_key(path):
    import re
    s = Path(path).name
    parts = re.split(r"(\\d+)", s)
    return [int(x) if x.isdigit() else x for x in parts]


def _slice_spacing_from_headers(paths, iop):
    if len(paths) < 2:
        return np.nan
    ordered = sorted(paths, key=_natural_sort_key)
    try:
        ds0 = _read_header(ordered[0])
        sbs = _safe_float(getattr(ds0, "SpacingBetweenSlices", np.nan))
        if np.isfinite(sbs) and abs(sbs) > 1e-4:
            return float(abs(sbs))
    except Exception:
        pass

    row = np.asarray(iop[:3], float)
    col = np.asarray(iop[3:], float)
    normal = np.cross(row, col)
    # Consecutive files are important: the previous linspace sampling multiplied
    # the apparent spacing by the sampling stride (e.g. 0.3 mm -> 2.4 mm).
    sample = ordered[: min(len(ordered), 160)]
    proj = []
    for p in sample:
        try:
            ds = _read_header(p)
            pos = np.asarray(ds.ImagePositionPatient, float)
            proj.append(float(pos @ normal))
        except Exception:
            continue
    if len(proj) < 2:
        return np.nan
    dif = np.diff(np.sort(np.unique(np.round(proj, 6))))
    dif = np.abs(dif[dif > 1e-4])
    if not len(dif):
        return np.nan
    # Use the smallest stable quartile, robust to skipped positions/multiphase repeats.
    q = np.percentile(dif, 25)
    near = dif[dif <= max(q * 1.5, q + 1e-3)]
    return float(np.median(near if len(near) else dif))


def scan_dicom_series(dicom_root):
    rows = []
    for folder, files in _iter_leaf_dirs(dicom_root):
        files = sorted(files)
        ds = None
        first_path = None
        for p in files[: min(12, len(files))]:
            try:
                cand = _read_header(p)
                if hasattr(cand, "SeriesInstanceUID") and hasattr(cand, "Rows"):
                    ds, first_path = cand, p
                    break
            except Exception:
                continue
        if ds is None:
            continue
        ps = list(getattr(ds, "PixelSpacing", [np.nan, np.nan]))
        iop = list(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))
        spacing_z = _slice_spacing_from_headers(files, iop)
        if not np.isfinite(spacing_z):
            spacing_z = _safe_float(getattr(ds, "SpacingBetweenSlices", np.nan))
        if not np.isfinite(spacing_z):
            spacing_z = _safe_float(getattr(ds, "SliceThickness", np.nan))
        rows.append({
            "folder": str(folder),
            "first_file": str(first_path),
            "series_uid": _text(getattr(ds, "SeriesInstanceUID", "")),
            "study_uid": _text(getattr(ds, "StudyInstanceUID", "")),
            "frame_uid": _text(getattr(ds, "FrameOfReferenceUID", "")),
            "series_number": _safe_int(getattr(ds, "SeriesNumber", -1)),
            "series_description": _text(getattr(ds, "SeriesDescription", "")),
            "protocol_name": _text(getattr(ds, "ProtocolName", "")),
            "modality": _text(getattr(ds, "Modality", "")),
            "image_type": _text(getattr(ds, "ImageType", "")),
            "kernel": _text(getattr(ds, "ConvolutionKernel", "")),
            "rows": _safe_int(getattr(ds, "Rows", -1)),
            "cols": _safe_int(getattr(ds, "Columns", -1)),
            "n_files": int(len(files)),
            "pixel_spacing_y_mm": _safe_float(ps[0] if len(ps) > 0 else np.nan),
            "pixel_spacing_x_mm": _safe_float(ps[1] if len(ps) > 1 else np.nan),
            "slice_spacing_mm": spacing_z,
            "slice_thickness_mm": _safe_float(getattr(ds, "SliceThickness", np.nan)),
            "trigger_time_ms": _safe_float(getattr(ds, "TriggerTime", np.nan)),
            "nominal_phase_pct": _safe_float(getattr(ds, "NominalPercentageOfCardiacPhase", np.nan)),
            "temporal_position": _safe_int(getattr(ds, "TemporalPositionIdentifier", -1)),
            "acquisition_time": _text(getattr(ds, "AcquisitionTime", "")),
            "reconstruction_diameter_mm": _safe_float(getattr(ds, "ReconstructionDiameter", np.nan)),
            "kvp": _safe_float(getattr(ds, "KVP", np.nan)),
        })
    if not rows:
        raise RuntimeError(f"No readable CT DICOM series found under {dicom_root}")
    return pd.DataFrame(rows).drop_duplicates(subset=["series_uid", "folder"]).reset_index(drop=True)


def _metadata_score(row):
    txt = " ".join([str(row.series_description), str(row.protocol_name), str(row.image_type)]).lower()
    score = 0.0
    if str(row.modality).upper() == "CT":
        score += 2.0
    if row.rows >= 512 and row.cols >= 512:
        score += 2.0
    if row.n_files >= 200:
        score += 2.0
    if np.isfinite(row.pixel_spacing_x_mm) and max(row.pixel_spacing_x_mm, row.pixel_spacing_y_mm) <= 0.60:
        score += 2.0
    if np.isfinite(row.slice_spacing_mm) and row.slice_spacing_mm <= 0.80:
        score += 2.0
    if "original" in txt and "primary" in txt:
        score += 1.0
    if any(k in txt for k in ("coronary", "ccta", "cardiac", "heart")):
        score += 2.0
    if any(k in txt for k in ("calcium", "score", "scout", "localizer", "topogram", "bolus", "smartprep", "noncontrast", "non contrast")):
        score -= 8.0
    if any(k in txt for k in ("mpr", "mip", "vr", "cpr")):
        score -= 2.0
    return float(score)


def identify_reference_series(inventory, source_meta):
    z, y, x = source_meta["shape"]
    sy, sx = source_meta["spacing_zyx"][1], source_meta["spacing_zyx"][2]
    sz = source_meta["spacing_zyx"][0]
    d = inventory.copy()
    d["reference_distance"] = (
        np.abs(d.n_files - z) / max(z, 1)
        + np.abs(d.rows - y) / max(y, 1)
        + np.abs(d.cols - x) / max(x, 1)
        + np.abs(pd.to_numeric(d.pixel_spacing_x_mm, errors="coerce") - sx).fillna(10)
        + np.abs(pd.to_numeric(d.pixel_spacing_y_mm, errors="coerce") - sy).fillna(10)
        + np.abs(pd.to_numeric(d.slice_spacing_mm, errors="coerce") - sz).fillna(10)
    )
    d["series7_bonus"] = (d.series_number == 7).astype(float) * 3.0
    d["reference_rank_score"] = d.reference_distance - d.series7_bonus
    return d.sort_values(["reference_rank_score", "reference_distance"]).iloc[0]


def _series_files(folder):
    return sorted([p for p in Path(folder).iterdir() if p.is_file()], key=_natural_sort_key)


def _header_phase_value(ds):
    """Return the strongest standard cardiac-phase identifier available."""
    for name in (
        "NominalPercentageOfCardiacPhase",
        "TriggerTime",
        "TemporalPositionIdentifier",
        "AcquisitionNumber",
    ):
        v = getattr(ds, name, None)
        if v not in (None, ""):
            try:
                return name, float(v)
            except Exception:
                return name, str(v)
    return None, None


def split_multiphase_series(folder, min_slices=180):
    """Split one same-UID multiphase folder into physical 3-D phase volumes.

    Prefer standard cardiac phase tags. If those are absent, infer phase rank from
    repeated ImagePositionPatient locations and InstanceNumber. This reads headers
    only and does not decompress pixels.
    """
    files = _series_files(folder)
    recs = []
    first = None
    for p in files:
        try:
            ds = _read_header(p)
            if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
                continue
            if first is None:
                first = ds
            iop = np.asarray(ds.ImageOrientationPatient, float)
            normal = np.cross(iop[:3], iop[3:])
            pos = np.asarray(ds.ImagePositionPatient, float)
            proj = float(pos @ normal)
            pname, pval = _header_phase_value(ds)
            recs.append({
                "path": p,
                "proj": proj,
                "instance": _safe_int(getattr(ds, "InstanceNumber", -1)),
                "phase_name": pname,
                "phase_value": pval,
            })
        except Exception:
            continue
    if len(recs) < 2 * min_slices:
        return []

    # 1) Standard phase tags: accept only 2..20 reasonably populated groups.
    for pname in ("NominalPercentageOfCardiacPhase", "TriggerTime",
                  "TemporalPositionIdentifier", "AcquisitionNumber"):
        vals = [r["phase_value"] for r in recs if r["phase_name"] == pname]
        uniq = sorted(set(vals), key=lambda x: str(x))
        if 2 <= len(uniq) <= 20:
            groups = []
            for v in uniq:
                rr = [r for r in recs if r["phase_name"] == pname and r["phase_value"] == v]
                # De-duplicate physical positions, keeping earliest InstanceNumber.
                bypos = {}
                for r in sorted(rr, key=lambda x: (x["proj"], x["instance"])):
                    bypos.setdefault(round(r["proj"], 4), r)
                rr = list(bypos.values())
                if len(rr) >= min_slices:
                    groups.append({
                        "phase_label": f"{pname}:{v}",
                        "phase_source": pname,
                        "paths": [r["path"] for r in sorted(rr, key=lambda x: x["proj"])],
                    })
            if len(groups) >= 2:
                return groups

    # 2) Generic repeated-position inference. At each physical z, sort repeated
    # frames by InstanceNumber; the within-z rank defines the phase.
    bypos = {}
    for r in recs:
        bypos.setdefault(round(r["proj"], 3), []).append(r)
    counts = np.array([len(v) for v in bypos.values()], int)
    repeated = counts[counts >= 2]
    if not len(repeated):
        return []
    nphase = int(round(float(np.median(repeated))))
    if not (2 <= nphase <= 20):
        return []

    rank_groups = [[] for _ in range(nphase)]
    for _, rr in sorted(bypos.items()):
        rr = sorted(rr, key=lambda x: x["instance"])
        if len(rr) < nphase:
            continue
        # If there are extra duplicates, distribute only the first nphase stable ranks.
        for k in range(nphase):
            rank_groups[k].append(rr[k])

    groups = []
    for k, rr in enumerate(rank_groups):
        if len(rr) >= min_slices:
            groups.append({
                "phase_label": f"inferred_phase_{k:02d}_of_{nphase:02d}",
                "phase_source": "repeated_position_rank",
                "paths": [r["path"] for r in sorted(rr, key=lambda x: x["proj"])],
            })
    return groups


def load_dicom_paths(paths):
    records = []
    first = None
    for p in sorted([Path(x) for x in paths], key=_natural_sort_key):
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if not hasattr(ds, "PixelData") or not hasattr(ds, "ImagePositionPatient"):
                continue
            if first is None:
                first = ds
            records.append((p, ds))
        except Exception:
            continue
    if not records:
        raise RuntimeError("No pixel-bearing DICOM slices in supplied path set")

    iop = np.asarray(first.ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    rec = []
    for p, ds in records:
        pos = np.asarray(ds.ImagePositionPatient, float)
        rec.append((float(pos @ normal), p, ds, pos))
    rec.sort(key=lambda x: x[0])

    # De-duplicate identical physical positions defensively.
    dedup = []
    seen = set()
    for item in rec:
        key = round(item[0], 4)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    rec = dedup

    positions = np.stack([r[3] for r in rec])
    proj = np.asarray([r[0] for r in rec])
    dz = np.abs(np.diff(proj))
    dz = dz[dz > 1e-4]
    sz = float(np.median(dz)) if len(dz) else _safe_float(
        getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0)), 1.0
    )
    ps = np.asarray(first.PixelSpacing, float)
    sy, sx = float(ps[0]), float(ps[1])

    vol = []
    for _, _, ds, _ in rec:
        try:
            a = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                "DICOM pixel decompression failed. The Colab must install "
                "pylibjpeg and pylibjpeg-libjpeg before preparation. Original error: "
                + str(e)
            ) from e
        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        vol.append(a * slope + intercept)
    arr = np.stack(vol).astype(np.float32)

    D = np.column_stack([row, col, normal])
    geom = Geometry(
        spacing_xyz=np.asarray([sx, sy, sz], float),
        origin=positions[0].astype(float),
        direction=D,
    )
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(v) for v in geom.spacing_xyz))
    img.SetOrigin(tuple(float(v) for v in geom.origin))
    img.SetDirection(tuple(float(v) for v in geom.direction.ravel()))
    return img, arr, geom


def load_dicom_series(folder):
    return load_dicom_paths(_series_files(folder))


def _sample_array(geom, arr, pts, cval=np.nan):
    z = geom.xyz_to_zyx_float(pts)
    return map_coordinates(arr, z.T, order=1, mode="constant", cval=float(cval))


def _path_brightness(geom, arr, paths):
    vals = []
    coverage = []
    shape = np.asarray(arr.shape)
    for p in paths:
        q = _resample_polyline(p, 2.0)
        z = geom.xyz_to_zyx_float(q)
        inside = np.all((z >= 0) & (z < shape[None, :]), axis=1)
        coverage.append(float(inside.mean()) if len(inside) else 0.0)
        if inside.any():
            v = _sample_array(geom, arr, q[inside])
            vals.extend(v[np.isfinite(v)].tolist())
    return {
        "path_coverage_fraction": float(np.mean(coverage)) if coverage else 0.0,
        "path_median_hu": float(np.median(vals)) if vals else np.nan,
        "path_p10_hu": float(np.percentile(vals, 10)) if vals else np.nan,
    }


def _aorta_distance_and_anchors(root, ref_img, geom, shape, lad, rca):
    aimg = sitk.ReadImage(str(_req(Path(root) / AORTA_MASK)))
    same = (
        tuple(aimg.GetSize()) == tuple(ref_img.GetSize())
        and np.allclose(aimg.GetSpacing(), ref_img.GetSpacing(), atol=1e-4)
        and np.allclose(aimg.GetOrigin(), ref_img.GetOrigin(), atol=1e-3)
        and np.allclose(aimg.GetDirection(), ref_img.GetDirection(), atol=1e-4)
    )
    if not same:
        aimg = sitk.Resample(aimg, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    aorta = sitk.GetArrayFromImage(aimg) > 0
    dt = ndi.distance_transform_edt(~aorta, sampling=geom.spacing_zyx)

    def endpoint_near(path):
        pts = np.asarray([path[0], path[-1]], float)
        z = np.rint(geom.xyz_to_zyx_float(pts)).astype(int)
        ds = []
        for q in z:
            if np.all((q >= 0) & (q < np.asarray(shape))):
                ds.append(float(dt[tuple(q)]))
            else:
                ds.append(np.inf)
        return pts[int(np.argmin(ds))], float(min(ds))

    lad_anchor, lad_d = endpoint_near(lad)
    rca_root, rca_d = endpoint_near(rca)
    return aorta, lad_anchor, rca_root, {
        "LAD_endpoint_aorta_distance_mm": lad_d,
        "RCA_endpoint_aorta_distance_mm": rca_d,
    }


def _crop_fixed_roi(ref_img, center_lps, half_mm=ROOT_ROI_HALF_MM):
    center = np.asarray(center_lps, float)
    idx = np.asarray(ref_img.TransformPhysicalPointToContinuousIndex(tuple(center)), float)
    spacing = np.asarray(ref_img.GetSpacing(), float)
    size = np.maximum(16, np.ceil(2 * half_mm / spacing).astype(int))
    start = np.floor(idx - size / 2).astype(int)
    full = np.asarray(ref_img.GetSize(), int)
    start = np.maximum(start, 0)
    size = np.minimum(size, full - start)
    return sitk.RegionOfInterest(
        ref_img,
        [int(v) for v in size],
        [int(v) for v in start],
    )


def register_translation(ref_img, moving_img, root_center_lps):
    fixed = _crop_fixed_roi(ref_img, root_center_lps)
    fixed_f = sitk.Cast(sitk.Clamp(fixed, lowerBound=-200, upperBound=1000), sitk.sitkFloat32)
    moving_f = sitk.Cast(sitk.Clamp(moving_img, lowerBound=-200, upperBound=1000), sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.12, seed=17)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.05,
        numberOfIterations=60,
        gradientMagnitudeTolerance=1e-5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.TranslationTransform(3), inPlace=False)
    try:
        out = reg.Execute(fixed_f, moving_f)
        pars = np.asarray(out.GetParameters(), float)
        mag = float(np.linalg.norm(pars[:3]))
        return out, {
            "registration_success": True,
            "translation_x_mm": float(pars[0]),
            "translation_y_mm": float(pars[1]),
            "translation_z_mm": float(pars[2]),
            "translation_magnitude_mm": mag,
            "metric_value": float(reg.GetMetricValue()),
            "registration_pass": mag <= MAX_REGISTRATION_TRANSLATION_MM,
        }
    except Exception as e:
        return sitk.TranslationTransform(3), {
            "registration_success": False,
            "translation_x_mm": 0.0,
            "translation_y_mm": 0.0,
            "translation_z_mm": 0.0,
            "translation_magnitude_mm": 0.0,
            "metric_value": np.nan,
            "registration_pass": False,
            "registration_error": str(e),
        }


def _write_mha(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(False)
    writer.Execute(img)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write image {path}")
    return path


def prepare(
    drive_root="/content/drive/MyDrive/OpenPlaque",
    dicom_root=None,
    local_workdir="/content/openplaque_alternate_series",
    output_dir=None,
):
    root = Path(drive_root)
    dicom_root = discover_dicom_root(drive_root, explicit=dicom_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    local = Path(local_workdir)
    local.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {
        "status": "PREPARING",
        "algorithm": ALGORITHM,
        "baseline": BASELINE,
    })

    master = _read_json(root / MASTER)
    if master.get("status") != EXPECTED_MASTER_STATUS:
        raise RuntimeError(f"Frozen master prerequisite failed: {master.get('status')}")

    ref_img, ref_geom, ref_shape, source_meta = _source_reference(root)
    lad = _load_lps_csv(root / LAD_PATH)
    rca = _load_lps_csv(root / RCA_PATH)
    c6 = _load_lps_csv(root / C6_PATH)
    c7 = _load_lps_csv(root / C7_PATH)
    _, left_anchor, rca_root, anchor_meta = _aorta_distance_and_anchors(
        root, ref_img, ref_geom, ref_shape, lad, rca
    )
    root_center = 0.5 * (left_anchor + rca_root)

    inv = scan_dicom_series(dicom_root)
    inv["metadata_score"] = inv.apply(_metadata_score, axis=1)
    ref_row = identify_reference_series(inv, source_meta)
    inv["is_reference_series7"] = (
        inv.series_uid.eq(ref_row.series_uid) & inv.folder.eq(ref_row.folder)
    )
    inv.to_csv(out / "series_inventory.csv", index=False)

    cand = inv[~inv.is_reference_series7].copy()
    cand = cand[cand.modality.str.upper().eq("CT")].copy()
    cand = cand.sort_values(
        ["metadata_score", "n_files"], ascending=[False, False]
    ).head(MAX_PRESELECT).copy()

    brightness_rows = []
    loaded_cache = {}
    for _, r in cand.iterrows():
        try:
            img, arr, geom = load_dicom_series(r.folder)
            b = _path_brightness(geom, arr, [lad, rca])
            final = float(
                r.metadata_score
                + 2.0 * b["path_coverage_fraction"]
                + np.clip((b["path_median_hu"] - 100.0) / 100.0, 0.0, 4.0)
            )
            brightness_rows.append({
                "series_uid": r.series_uid,
                "folder": r.folder,
                **b,
                "final_score": final,
                "load_ok": True,
            })
            loaded_cache[str(r.series_uid)] = img
        except Exception as e:
            brightness_rows.append({
                "series_uid": r.series_uid,
                "folder": r.folder,
                "path_coverage_fraction": 0.0,
                "path_median_hu": np.nan,
                "path_p10_hu": np.nan,
                "final_score": -999.0,
                "load_ok": False,
                "load_error": str(e),
            })

    br = pd.DataFrame(brightness_rows)
    ranking = cand.merge(br, on=["series_uid", "folder"], how="left")
    ranking["phase_key"] = ranking.apply(
        lambda r:
            f"phase:{r.nominal_phase_pct:.1f}" if np.isfinite(r.nominal_phase_pct)
            else f"trigger:{r.trigger_time_ms:.1f}" if np.isfinite(r.trigger_time_ms)
            else f"time:{r.acquisition_time}" if str(r.acquisition_time)
            else f"desc:{r.series_description}",
        axis=1,
    )
    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)

    selected_idx = []
    seen_signatures = set()
    for i, r in ranking.iterrows():
        sig = (r.phase_key, str(r.kernel), str(r.series_description))
        if sig in seen_signatures and len(selected_idx) < 2:
            continue
        if (
            bool(r.load_ok)
            and r.path_coverage_fraction >= 0.75
            and np.isfinite(r.path_median_hu)
            and r.path_median_hu >= 150
        ):
            selected_idx.append(i)
            seen_signatures.add(sig)
        if len(selected_idx) >= MAX_ALTERNATE_SERIES:
            break

    if len(selected_idx) < min(2, MAX_ALTERNATE_SERIES):
        for i, r in ranking.iterrows():
            if (
                i not in selected_idx
                and bool(r.load_ok)
                and r.path_coverage_fraction >= 0.60
            ):
                selected_idx.append(i)
            if len(selected_idx) >= MAX_ALTERNATE_SERIES:
                break

    ranking["selected"] = False
    if selected_idx:
        ranking.loc[selected_idx, "selected"] = True
    ranking.to_csv(out / "candidate_ranking.csv", index=False)
    selected = ranking[ranking.selected].copy()
    if selected.empty:
        raise RuntimeError("No alternate CT series passed automatic preselection")

    input_dir = local / "model_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    ref_id = "series7_reference"
    _write_mha(ref_img, input_dir / f"{ref_id}.img.mha")

    reg_rows = []
    selected_records = [{
        "scan_id": ref_id,
        "role": "reference",
        "series_uid": str(ref_row.series_uid),
        "series_number": int(ref_row.series_number),
        "folder": str(ref_row.folder),
        "transform_file": None,
    }]

    for rank_i, (_, r) in enumerate(selected.iterrows(), 1):
        uid = str(r.series_uid)
        img = loaded_cache.get(uid)
        if img is None:
            img, _, _ = load_dicom_series(r.folder)
        scan_id = f"alt_{rank_i:02d}_s{int(r.series_number) if int(r.series_number) >= 0 else rank_i}"
        input_file = input_dir / f"{scan_id}.img.mha"
        _write_mha(img, input_file)
        tx, reg_info = register_translation(ref_img, img, root_center)
        tfm = local / f"{scan_id}_to_series7.tfm"
        sitk.WriteTransform(tx, str(tfm))
        reg_rows.append({
            "scan_id": scan_id,
            "series_uid": uid,
            "series_number": int(r.series_number),
            "series_description": str(r.series_description),
            "phase_key": str(r.phase_key),
            **reg_info,
        })
        selected_records.append({
            "scan_id": scan_id,
            "role": "alternate",
            "series_uid": uid,
            "series_number": int(r.series_number),
            "series_description": str(r.series_description),
            "protocol_name": str(r.protocol_name),
            "kernel": str(r.kernel),
            "phase_key": str(r.phase_key),
            "folder": str(r.folder),
            "input_file": str(input_file),
            "transform_file": str(tfm),
            "registration": reg_info,
            "path_median_hu": float(r.path_median_hu),
            "path_coverage_fraction": float(r.path_coverage_fraction),
            "final_score": float(r.final_score),
        })
    pd.DataFrame(reg_rows).to_csv(out / "registration_summary.csv", index=False)

    prep = {
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "dicom_root": str(dicom_root),
        "reference_series": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in ref_row.to_dict().items()
        },
        "left_anchor_lps_mm": [float(v) for v in left_anchor],
        "rca_root_lps_mm": [float(v) for v in rca_root],
        "root_center_lps_mm": [float(v) for v in root_center],
        "anchor_meta": anchor_meta,
        "selected_series": selected_records,
        "model_input_dir": str(input_dir),
    }
    _write_json(out / "preparation.json", prep)
    _write_json(out / "input_provenance.json", {
        "dicom_root": str(dicom_root),
        "series7_source_cache": str(root / SOURCE_CACHE),
        "aorta_mask": str(root / AORTA_MASK),
        "LAD": str(root / LAD_PATH),
        "RCA": str(root / RCA_PATH),
        "C6": str(root / C6_PATH),
        "C7": str(root / C7_PATH),
        "master": str(root / MASTER),
    })
    _write_json(out / "run_state.json", {
        "status": "PREPARED",
        "algorithm": ALGORITHM,
        "baseline": BASELINE,
        "selected_alternate_series": len(selected_records) - 1,
    })
    return prep


def _mask_on_reference(pred_path, ref_img, tx=None):
    img = sitk.ReadImage(str(_req(pred_path)))
    tx = tx if tx is not None else sitk.Transform()
    r = sitk.Resample(
        img, ref_img, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    return sitk.GetArrayFromImage(r) > 0


def _bbox(mask, pad=8):
    pts = np.argwhere(mask)
    if not len(pts):
        return tuple(slice(0, s) for s in mask.shape), np.zeros(3, int)
    lo = np.maximum(pts.min(axis=0) - pad, 0)
    hi = np.minimum(pts.max(axis=0) + pad + 1, np.asarray(mask.shape))
    return tuple(slice(int(lo[k]), int(hi[k])) for k in range(3)), lo


def _sparse_components(mask, geom, aorta):
    sl, offset = _bbox(mask, pad=24)
    m = mask[sl]
    a = aorta[sl]
    labels, n = ndi.label(m, structure=np.ones((3, 3, 3), np.uint8))
    da = ndi.distance_transform_edt(~a, sampling=geom.spacing_zyx)
    contact = m & (da <= AORTA_CONTACT_TOLERANCE_MM)
    clab, nc = ndi.label(contact, structure=np.ones((3, 3, 3), np.uint8))
    return sl, offset, labels, int(n), clab, int(nc)


def _mask_points_tree(mask, geom, labels_crop, offset, sl):
    pts = np.argwhere(mask[sl])
    if not len(pts):
        return None, np.zeros(0, int)
    global_zyx = pts + offset[None, :]
    xyz = geom.zyx_to_xyz(global_zyx)
    labs = labels_crop[tuple(pts.T)]
    return cKDTree(xyz), labs


def _path_support_tree(path, tree, labs, tol=PATH_TOLERANCE_MM):
    q = _resample_polyline(path, 0.5)
    if tree is None:
        return {
            "n_samples": int(len(q)),
            "support_fraction": 0.0,
            "median_distance_mm": np.inf,
            "p90_distance_mm": np.inf,
            "supported_component_ids": [],
            "dominant_component": 0,
        }
    d, idx = tree.query(q, k=1)
    ok = d <= tol
    near_labs = labs[idx[ok]] if ok.any() else np.zeros(0, int)
    ids, cnt = np.unique(near_labs[near_labs > 0], return_counts=True)
    dom = int(ids[np.argmax(cnt)]) if len(ids) else 0
    return {
        "n_samples": int(len(q)),
        "support_fraction": float(ok.mean()),
        "median_distance_mm": float(np.median(d)),
        "p90_distance_mm": float(np.percentile(d, 90)),
        "supported_component_ids": [int(x) for x in ids],
        "dominant_component": dom,
    }


def _contact_rows(clab, labels, offset, geom, rca_root, left_anchor, scan_id):
    rows = []
    for cid in range(1, int(clab.max()) + 1):
        pts = np.argwhere(clab == cid)
        if not len(pts):
            continue
        global_zyx = pts + offset[None, :]
        xyz = geom.zyx_to_xyz(global_zyx)
        centroid = xyz.mean(axis=0)
        compvals = labels[tuple(pts.T)]
        compvals = compvals[compvals > 0]
        comp = int(np.bincount(compvals).argmax()) if len(compvals) else 0
        rows.append({
            "scan_id": scan_id,
            "contact_cluster": int(cid),
            "contact_voxels": int(len(pts)),
            "component": comp,
            "centroid_lps_x_mm": float(centroid[0]),
            "centroid_lps_y_mm": float(centroid[1]),
            "centroid_lps_z_mm": float(centroid[2]),
            "distance_to_RCA_root_mm": float(np.linalg.norm(centroid - rca_root)),
            "distance_to_left_anchor_mm": float(np.linalg.norm(centroid - left_anchor)),
        })
    return rows


def _load_aorta(root, ref_img):
    aimg = sitk.ReadImage(str(_req(Path(root) / AORTA_MASK)))
    same = (
        tuple(aimg.GetSize()) == tuple(ref_img.GetSize())
        and np.allclose(aimg.GetSpacing(), ref_img.GetSpacing(), atol=1e-4)
        and np.allclose(aimg.GetOrigin(), ref_img.GetOrigin(), atol=1e-3)
        and np.allclose(aimg.GetDirection(), ref_img.GetDirection(), atol=1e-4)
    )
    if not same:
        aimg = sitk.Resample(
            aimg, ref_img, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
    return sitk.GetArrayFromImage(aimg) > 0


def _orth_view(arr, geom, point_lps, half_mm=22.0):
    zyx = np.rint(geom.xyz_to_zyx_float([point_lps])[0]).astype(int)
    rz = max(4, int(round(half_mm / geom.spacing_zyx[0])))
    ry = max(4, int(round(half_mm / geom.spacing_zyx[1])))
    rx = max(4, int(round(half_mm / geom.spacing_zyx[2])))
    z, y, x = zyx
    z = int(np.clip(z, 0, arr.shape[0] - 1))
    y = int(np.clip(y, 0, arr.shape[1] - 1))
    x = int(np.clip(x, 0, arr.shape[2] - 1))
    axial = arr[
        z,
        max(0, y-ry):min(arr.shape[1], y+ry+1),
        max(0, x-rx):min(arr.shape[2], x+rx+1),
    ]
    coronal = arr[
        max(0, z-rz):min(arr.shape[0], z+rz+1),
        y,
        max(0, x-rx):min(arr.shape[2], x+rx+1),
    ]
    sagittal = arr[
        max(0, z-rz):min(arr.shape[0], z+rz+1),
        max(0, y-ry):min(arr.shape[1], y+ry+1),
        x,
    ]
    return axial, coronal, sagittal


def _plot_root_qc(scan_id, ct_ref, pred_ref, aorta, geom, left_anchor, rca_root, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (name, pt) in enumerate((
        ("left anchor", left_anchor),
        ("RCA root", rca_root),
    )):
        ct_views = _orth_view(ct_ref, geom, pt)
        p_views = _orth_view(pred_ref.astype(float), geom, pt)
        a_views = _orth_view(aorta.astype(float), geom, pt)
        titles = ("axial", "coronal", "sagittal")
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(ct_views[col], cmap="gray", vmin=-100, vmax=900)
            try:
                ax.contour(a_views[col], levels=[0.5], linewidths=1.0)
                ax.contour(p_views[col], levels=[0.5], linewidths=1.0)
            except Exception:
                pass
            ax.set_title(f"{scan_id} | {name} | {titles[col]}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=165)
    plt.close(fig)


def _plot_contacts(df, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if not df.empty:
        for sid, g in df.groupby("scan_id"):
            ax.scatter(
                g.distance_to_RCA_root_mm,
                g.distance_to_left_anchor_mm,
                s=np.clip(g.contact_voxels, 15, 180),
                label=sid,
                alpha=.7,
            )
    ax.axvline(RCA_CONTACT_RADIUS_MM, linestyle="--", linewidth=1)
    ax.axvline(MIN_DISTINCT_FROM_RCA_MM, linestyle=":", linewidth=1)
    ax.axhline(MAX_LEFT_ANCHOR_TO_CONTACT_MM, linestyle="--", linewidth=1)
    ax.set_xlabel("Aortic-contact centroid distance to RCA root (mm)")
    ax.set_ylabel("Distance to left LAD anchor (mm)")
    ax.set_title("Alternate-series aortic contact clusters")
    if not df.empty:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def analyze(
    predictions_dir,
    drive_root="/content/drive/MyDrive/OpenPlaque",
    local_workdir="/content/openplaque_alternate_series",
    output_dir=None,
):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    local = Path(local_workdir)
    prep = _read_json(out / "preparation.json")
    ref_img, geom, _, _ = _source_reference(root)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    aorta = _load_aorta(root, ref_img)
    lad = _load_lps_csv(root / LAD_PATH)
    rca = _load_lps_csv(root / RCA_PATH)
    c6 = _load_lps_csv(root / C6_PATH)
    c7 = _load_lps_csv(root / C7_PATH)
    left_anchor = np.asarray(prep["left_anchor_lps_mm"], float)
    rca_root = np.asarray(prep["rca_root_lps_mm"], float)

    path_rows = []
    contact_rows = []
    gate_rows = []
    series_results = []

    for rec in prep["selected_series"]:
        sid = rec["scan_id"]
        pred_path = Path(predictions_dir) / f"{sid}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(str(pred_path))
        tx = (
            sitk.Transform()
            if rec["role"] == "reference"
            else sitk.ReadTransform(str(rec["transform_file"]))
        )
        pred = _mask_on_reference(pred_path, ref_img, tx)
        sl, offset, labels, ncomp, clab, ncontacts = _sparse_components(
            pred, geom, aorta
        )
        tree, lab_near = _mask_points_tree(pred, geom, labels, offset, sl)
        paths = {"LAD": lad, "RCA": rca, "C6": c6, "C7": c7}
        supports = {
            k: _path_support_tree(v, tree, lab_near)
            for k, v in paths.items()
        }
        for path_name, s in supports.items():
            path_rows.append({
                "scan_id": sid,
                "role": rec["role"],
                "path": path_name,
                **s,
            })

        cr = _contact_rows(
            clab, labels, offset, geom, rca_root, left_anchor, sid
        )
        contact_rows.extend(cr)
        cdf = pd.DataFrame(cr)
        shared_lad_c6 = sorted(
            set(supports["LAD"]["supported_component_ids"])
            & set(supports["C6"]["supported_component_ids"])
        )
        rca_comps = set(supports["RCA"]["supported_component_ids"])

        if cdf.empty:
            rca_contact = False
            left_candidates = []
        else:
            rca_contact = bool((
                (cdf.distance_to_RCA_root_mm <= RCA_CONTACT_RADIUS_MM)
                & cdf.component.isin(rca_comps)
            ).any())
            left_candidates = cdf[
                (cdf.contact_voxels >= MIN_CONTACT_VOXELS)
                & (cdf.distance_to_RCA_root_mm >= MIN_DISTINCT_FROM_RCA_MM)
                & (cdf.distance_to_left_anchor_mm <= MAX_LEFT_ANCHOR_TO_CONTACT_MM)
                & cdf.component.isin(shared_lad_c6)
            ].to_dict("records")

        reg_pass = (
            True
            if rec["role"] == "reference"
            else bool(rec.get("registration", {}).get("registration_pass", False))
        )
        gates = {
            "registration_pass": reg_pass,
            "RCA_support_control": supports["RCA"]["support_fraction"] >= MIN_RCA_SUPPORT,
            "RCA_distinct_aortic_contact_control": rca_contact,
            "LAD_support": supports["LAD"]["support_fraction"] >= MIN_LAD_SUPPORT,
            "C6_support": supports["C6"]["support_fraction"] >= MIN_C6_SUPPORT,
            "LAD_C6_share_component": bool(shared_lad_c6),
            "distinct_left_aortic_contact": bool(left_candidates),
        }
        interpretable = bool(
            gates["registration_pass"]
            and gates["RCA_support_control"]
            and gates["RCA_distinct_aortic_contact_control"]
        )
        positive = bool(
            interpretable
            and gates["LAD_support"]
            and gates["C6_support"]
            and gates["LAD_C6_share_component"]
            and gates["distinct_left_aortic_contact"]
        )
        for gate, val in gates.items():
            gate_rows.append({
                "scan_id": sid,
                "role": rec["role"],
                "gate": gate,
                "pass": bool(val),
            })

        series_results.append({
            "scan_id": sid,
            "role": rec["role"],
            "series_number": rec.get("series_number"),
            "series_description": rec.get("series_description", ""),
            "phase_key": rec.get("phase_key", ""),
            "interpretable": interpretable,
            "left_contact_positive": positive,
            "shared_LAD_C6_components": shared_lad_c6,
            "left_contact_candidates": left_candidates,
            "prediction_components": ncomp,
            "contact_clusters": ncontacts,
            "supports": supports,
            "registration": rec.get("registration", {}),
        })

        if rec["role"] == "reference":
            ct_ref = ref_arr
        else:
            moving = sitk.ReadImage(
                str(local / "model_inputs" / f"{sid}.img.mha")
            )
            ct_img = sitk.Resample(
                moving, ref_img, tx,
                sitk.sitkLinear, -1024.0, sitk.sitkFloat32
            )
            ct_ref = sitk.GetArrayFromImage(ct_img)

        _plot_root_qc(
            sid, ct_ref, pred, aorta, geom,
            left_anchor, rca_root,
            out / f"QC_{sid}_root_planes.png",
        )

    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(out / "path_support_by_series.csv", index=False)
    contact_df = pd.DataFrame(contact_rows)
    contact_df.to_csv(out / "aorta_contact_clusters.csv", index=False)
    gates_df = pd.DataFrame(gate_rows)
    gates_df.to_csv(out / "gates_by_series.csv", index=False)
    series_df = pd.DataFrame([
        {
            k: v for k, v in r.items()
            if k not in ("supports", "left_contact_candidates", "registration")
        }
        for r in series_results
    ])
    series_df.to_csv(out / "series_decisions.csv", index=False)
    _plot_contacts(contact_df, out / "01_aortic_contact_cluster_distances.png")

    alt = [
        r for r in series_results
        if r["role"] == "alternate" and r["interpretable"]
    ]
    pos = [r for r in alt if r["left_contact_positive"]]
    centroids = []
    for r in pos:
        for c in r["left_contact_candidates"]:
            centroids.append((
                r["scan_id"],
                np.array([
                    c["centroid_lps_x_mm"],
                    c["centroid_lps_y_mm"],
                    c["centroid_lps_z_mm"],
                ], float),
            ))

    concordant = False
    concordance_pair = None
    if len({x[0] for x in centroids}) >= 2:
        best = np.inf
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if centroids[i][0] == centroids[j][0]:
                    continue
                d = float(np.linalg.norm(centroids[i][1] - centroids[j][1]))
                if d < best:
                    best = d
                    concordance_pair = [
                        centroids[i][0],
                        centroids[j][0],
                        d,
                    ]
        concordant = bool(best <= CROSS_SERIES_CONTACT_CONCORDANCE_MM)

    if concordant:
        status = STATUS_CONCORDANT
    elif len(pos) >= 2:
        status = STATUS_DISCORDANT
    elif len(pos) == 1:
        status = STATUS_SINGLE
    elif len(alt) >= 2:
        status = STATUS_NEGATIVE
    else:
        status = STATUS_INSUFFICIENT

    decision = {
        "status": status,
        "interpretable_alternate_series": len(alt),
        "positive_alternate_series": [r["scan_id"] for r in pos],
        "cross_series_contact_concordant": concordant,
        "concordance_pair": concordance_pair,
        "clinical_LM_identity_established": False,
        "clinical_LCX_OM_identity_established": False,
        "master_anatomy_modified": False,
        "reopen_left_coronary_review": bool(
            status in (STATUS_CONCORDANT, STATUS_SINGLE)
        ),
    }
    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "prespecified_gates": {
            "PATH_TOLERANCE_MM": PATH_TOLERANCE_MM,
            "AORTA_CONTACT_TOLERANCE_MM": AORTA_CONTACT_TOLERANCE_MM,
            "MIN_RCA_SUPPORT": MIN_RCA_SUPPORT,
            "MIN_LAD_SUPPORT": MIN_LAD_SUPPORT,
            "MIN_C6_SUPPORT": MIN_C6_SUPPORT,
            "MAX_REGISTRATION_TRANSLATION_MM": MAX_REGISTRATION_TRANSLATION_MM,
            "RCA_CONTACT_RADIUS_MM": RCA_CONTACT_RADIUS_MM,
            "MIN_DISTINCT_FROM_RCA_MM": MIN_DISTINCT_FROM_RCA_MM,
            "MAX_LEFT_ANCHOR_TO_CONTACT_MM": MAX_LEFT_ANCHOR_TO_CONTACT_MM,
            "MIN_CONTACT_VOXELS": MIN_CONTACT_VOXELS,
            "CROSS_SERIES_CONTACT_CONCORDANCE_MM": CROSS_SERIES_CONTACT_CONCORDANCE_MM,
        },
        "left_anchor_lps_mm": prep["left_anchor_lps_mm"],
        "rca_root_lps_mm": prep["rca_root_lps_mm"],
        "series_results": series_results,
        "decision": decision,
        "scientific_boundary": (
            "Alternate same-exam series are independent reconstruction/phase evidence "
            "but not a clinical vessel label. Distinct left aortic contact requires "
            "separation from the RCA ostium. Frozen master remains unchanged."
        ),
    }
    _write_json(out / "decision.json", decision)
    _write_json(out / "summary.json", summary)

    qc_imgs = sorted(out.glob("QC_*_root_planes.png"))
    rows_html = (
        series_df.to_html(index=False)
        if not series_df.empty
        else "<p>No series results.</p>"
    )
    report = out / "OPENPLAQUE_ALTERNATE_SERIES_LEFT_CORONARY_ORIGIN_VALIDATION_V1_REPORT.html"
    imgs = "".join(
        f"<h3>{p.stem}</h3><img src='{p.name}' width='1000'>"
        for p in qc_imgs
    )
    report.write_text(
        "<html><body><h1>OpenPlaque Alternate-Series Left Coronary Origin Validation v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Interpretable alternate series: {len(alt)}; positive: {len(pos)}.</p>"
        "<p>Distinct left aortic contact is explicitly separated from the RCA ostium. "
        "Frozen master unchanged.</p>"
        "<h2>Series decisions</h2>" + rows_html +
        "<h2>Aortic-contact QC</h2>"
        "<img src='01_aortic_contact_cluster_distances.png' width='900'>" +
        imgs +
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out / "run_state.json", {
        "status": "COMPLETE",
        "algorithm": ALGORITHM,
        "result_status": status,
    })
    archive = out / "OPENPLAQUE_ALTERNATE_SERIES_LEFT_CORONARY_ORIGIN_VALIDATION_V1_RESULTS.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            if p.is_file() and p != archive:
                zf.write(p, p.name)
    return summary


def synthetic_self_test():
    rca = np.array([0.0, 0.0, 0.0])
    left = np.array([18.0, 0.0, 0.0])
    c_rca = np.array([1.0, 0.0, 0.0])
    c_left = np.array([18.5, 0.5, 0.0])
    assert np.linalg.norm(c_rca - rca) <= RCA_CONTACT_RADIUS_MM
    assert np.linalg.norm(c_left - rca) >= MIN_DISTINCT_FROM_RCA_MM
    assert np.linalg.norm(c_left - left) <= MAX_LEFT_ANCHOR_TO_CONTACT_MM
    p = np.array([[0., 0., 0.], [2., 0., 0.]])
    q = _resample_polyline(p, .5)
    assert len(q) == 5
    assert np.isclose(_arc(q)[-1], 2.0)
    return {"ok": True, "resampled_points": len(q)}
