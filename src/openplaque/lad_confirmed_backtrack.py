from __future__ import annotations

"""Backtrack proximally from a previously convincing LAD segment.

This experiment is intentionally narrow.  It does not search the whole heart for
an LAD, and it does not attempt LCX tracking.  It consumes the distal/proximal-LAD
segment that looked convincing in the preceding LAD-origin experiment, rechecks it
using the *previously validated* RCA calibration, freezes the convincing segment,
and then performs a short source-resolution beam search proximally.

The TotalSegmentator aorta mask is used only as an anatomical constraint (inside /
distance-to-aorta); it is not used as a coronary segmentation.

Research use only.
"""

import base64
import gc
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from .study import OpenPlaqueStudy

ALGORITHM_VERSION = "lad-confirmed-segment-backtrack-v1.0"
SOURCE_SERIES = 7

# This calibration is the strong source-resolution RCA calibration from the
# earlier staged-left-main run.  It is a subject-specific positive reference,
# not manually labelled ground truth.  We prefer the saved JSON when it passes
# the same sanity envelope; otherwise we fall back to these pinned values rather
# than recomputing a broken calibration.
PINNED_VALIDATED_RCA = {
    "median_radius_mm": 1.616906,
    "median_center_hu": 551.061478,
    "median_offset_mm": 0.317649,
    "median_circularity": 1.2,
    "median_core_minus_ring_hu": 503.91744,
    "median_plane_score": 0.932688,
    "plane_pass_fraction": 0.916667,
    "source": "validated staged-left-main RCA calibration",
}


def _json_write(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        return float("nan")


def _ram(label):
    print(f"{label}: RSS {_rss_gb():.2f} GB")


def arc_mm(path_zyx, spacing_zyx):
    p = np.asarray(path_zyx, float)
    if len(p) == 0:
        return np.zeros(0, float)
    if len(p) == 1:
        return np.zeros(1, float)
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def resample_path(path_zyx, spacing_zyx, step_mm=0.45):
    p = np.asarray(path_zyx, float)
    if len(p) < 2:
        return p.copy()
    s = arc_mm(p, spacing_zyx)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] <= step_mm:
        return p.copy()
    q = np.arange(0.0, s[-1] + 0.5 * step_mm, step_mm)
    q[-1] = min(q[-1], s[-1])
    return np.column_stack([np.interp(q, s, p[:, j]) for j in range(3)])


def _series7_files(root, extract_root="/content/full_dicom_lad_confirmed_backtrack"):
    root = Path(root)
    src = root / "Full_DICOM.zip"
    if not src.exists():
        raise FileNotFoundError(src)
    local = Path("/content/Full_DICOM.zip")
    if not local.exists() or local.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, local)
    study = OpenPlaqueStudy(str(local), extract_root=extract_root)
    match = [s for s in study.series if s["series_number"] == SOURCE_SERIES]
    if not match:
        raise RuntimeError("Source CCTA series 7 not found")
    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(match[0]["folder"], match[0]["uid"]))
    if not files:
        raise RuntimeError("No DICOM files for source CCTA series 7")
    return files


def stream_source_ct_to_memmap(root, out_path, reuse_sources=()):
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        mm = np.load(out_path, mmap_mode="r")
        meta = _json_read(meta_path)
        if tuple(meta.get("shape", [])) == tuple(mm.shape):
            return mm, meta, "reused_own_cache"

    for candidate in reuse_sources:
        candidate = Path(candidate)
        cmeta = candidate.with_suffix(".json")
        if candidate.exists() and cmeta.exists():
            mm = np.load(candidate, mmap_mode="r")
            meta = _json_read(cmeta)
            if tuple(meta.get("shape", [])) == tuple(mm.shape):
                return mm, meta, f"imported_validated_cache:{candidate}"

    files = _series7_files(root)
    ds0 = pydicom.dcmread(files[0], force=True)
    n, rows, cols = len(files), int(ds0.Rows), int(ds0.Columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16, shape=(n, rows, cols))
    positions = []
    for i, fp in enumerate(files):
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        mm[i] = np.rint(np.clip(arr * slope + intercept, -32768, 32767)).astype(np.int16)
        positions.append([float(x) for x in getattr(ds, "ImagePositionPatient", (0, 0, i))])
        if i % 100 == 0:
            mm.flush()
    mm.flush()
    ps = [float(x) for x in ds0.PixelSpacing]
    pos = np.asarray(positions, float)
    dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else float(getattr(ds0, "SliceThickness", 1.0))
    meta = {
        "shape": [n, rows, cols],
        "spacing_zyx": [dz, ps[0], ps[1]],
        "positions_lps_mm": positions,
        "image_orientation_patient": [float(x) for x in getattr(ds0, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0))],
    }
    _json_write(meta, meta_path)
    del mm
    gc.collect()
    return np.load(out_path, mmap_mode="r"), meta, "recomputed_and_cached"


def source_zyx_to_lps(meta, zyx):
    p = np.atleast_2d(np.asarray(zyx, float))
    positions = np.asarray(meta["positions_lps_mm"], float)
    orient = np.asarray(meta["image_orientation_patient"], float)
    row_cos, col_cos = orient[:3], orient[3:]
    sp = np.asarray(meta["spacing_zyx"], float)
    if len(positions) > 1:
        slice_vec = (positions[-1] - positions[0]) / max(len(positions) - 1, 1)
    else:
        slice_vec = np.cross(row_cos, col_cos) * sp[0]
    origin = positions[0]
    out = origin[None, :] + p[:, 0, None] * slice_vec[None, :]
    out += p[:, 2, None] * sp[2] * row_cos[None, :]
    out += p[:, 1, None] * sp[1] * col_cos[None, :]
    return out


def _orth_basis(tangent_zyx_mm):
    t = np.asarray(tangent_zyx_mm, float)
    t /= max(np.linalg.norm(t), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u)
    v /= max(np.linalg.norm(v), 1e-9)
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=5.5, pix_mm=0.18):
    u, v = _orth_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, [vox[..., 0], vox[..., 1], vox[..., 2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def plane_lumen_metrics(im, c, lo_hu=120.0, hi_hu=1100.0):
    pix = float(abs(c[1] - c[0]))
    bright = (im >= lo_hu) & (im <= hi_hu)
    lab, _ = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    cy = cx = len(c) // 2
    chosen = int(lab[cy, cx])
    if chosen == 0:
        yy, xx = np.nonzero(bright)
        if len(yy):
            dist = np.hypot(c[yy], c[xx])
            j = int(np.argmin(dist))
            if dist[j] <= 0.85:
                chosen = int(lab[yy[j], xx[j]])
    comp = lab == chosen if chosen > 0 else np.zeros_like(bright)
    area_px = int(comp.sum())
    radius = math.sqrt(area_px * pix * pix / math.pi) if area_px else float("nan")
    if area_px:
        yy, xx = np.nonzero(comp)
        ym, xm = float(np.mean(c[yy])), float(np.mean(c[xx]))
        offset = float(math.hypot(ym, xm))
        er = ndi.binary_erosion(comp, structure=np.ones((3, 3), bool))
        perimeter_px = max(1, int((comp & ~er).sum()))
        perimeter_mm = perimeter_px * pix
        circularity = float(np.clip(4 * math.pi * (area_px * pix * pix) / max(perimeter_mm * perimeter_mm, 1e-8), 0, 1.2))
    else:
        offset = circularity = float("nan")
    U, V = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(U, V)
    center_hu = float(np.median(im[R <= 0.7]))
    core_hu = float(np.median(im[R <= 1.0]))
    ring_hu = float(np.median(im[(R >= 2.5) & (R <= 4.0)]))
    return {
        "radius_mm": radius,
        "centroid_offset_mm": offset,
        "circularity": circularity,
        "center_hu": center_hu,
        "core_minus_ring_hu": core_hu - ring_hu,
    }


def score_plane(metrics, rca_cal):
    rr = float(rca_cal["median_radius_mm"])
    rh = float(rca_cal["median_center_hu"])
    r = metrics["radius_mm"]
    off = metrics["centroid_offset_mm"]
    circ = metrics["circularity"]
    radius_score = float(np.exp(-0.5 * ((r - 1.05 * rr) / max(0.55 * rr, 0.65)) ** 2)) if np.isfinite(r) else 0.0
    offset_score = float(np.exp(-0.5 * (off / 0.72) ** 2)) if np.isfinite(off) else 0.0
    circ_score = float(np.clip(circ / 0.55, 0, 1)) if np.isfinite(circ) else 0.0
    hu_score = float(np.exp(-0.5 * ((metrics["center_hu"] - rh) / 330.0) ** 2))
    contrast_score = float(1 / (1 + np.exp(-(metrics["core_minus_ring_hu"] - 20.0) / 75.0)))
    score = 0.33 * radius_score + 0.30 * offset_score + 0.17 * circ_score + 0.11 * hu_score + 0.09 * contrast_score
    hard_pass = bool(
        np.isfinite(r)
        and 0.62 * rr <= r <= min(3.20, 1.90 * rr + 0.15)
        and np.isfinite(off) and off <= 1.15
        and np.isfinite(circ) and circ >= 0.20
        and 120 <= metrics["center_hu"] <= 1100
    )
    soft_pass = bool(
        np.isfinite(r)
        and 0.50 * rr <= r <= 3.35
        and np.isfinite(off) and off <= 1.75
        and np.isfinite(circ) and circ >= 0.13
        and 100 <= metrics["center_hu"] <= 1150
        and score >= 0.52
    )
    return float(score), hard_pass, soft_pass


def serial_qc(path, ct, spacing, rca_cal, step_sample_mm=2.0, label="path"):
    p = resample_path(path, spacing, 0.35)
    s = arc_mm(p, spacing)
    if len(p) < 4:
        return pd.DataFrame(), {"label": label, "length_mm": float(s[-1]) if len(s) else 0.0, "plane_pass_fraction": 0.0, "median_plane_score": 0.0}
    sample_s = np.arange(min(0.8, 0.12 * s[-1]), s[-1] + 1e-6, step_sample_mm)
    if len(sample_s) == 0 or sample_s[-1] < s[-1] - 0.6:
        sample_s = np.r_[sample_s, s[-1]]
    rows = []
    for ss in sample_s:
        i = int(np.argmin(abs(s - ss)))
        i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
        t = (p[i1] - p[i0]) * np.asarray(spacing, float)
        if np.linalg.norm(t) < 1e-7:
            continue
        im, c = orthogonal_plane(ct, p[i], t, spacing)
        m = plane_lumen_metrics(im, c)
        sc, hp, spass = score_plane(m, rca_cal)
        rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_score": sc, "plane_pass": hp, "soft_pass": spass})
    df = pd.DataFrame(rows)
    summary = {
        "label": label,
        "length_mm": float(s[-1]),
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "soft_pass_fraction": float(df.soft_pass.mean()) if len(df) else 0.0,
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else float("nan"),
        "median_offset_mm": float(df.centroid_offset_mm.median()) if len(df) else float("nan"),
        "median_circularity": float(df.circularity.median()) if len(df) else float("nan"),
        "median_center_hu": float(df.center_hu.median()) if len(df) else float("nan"),
    }
    return df, summary


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def _cone_directions(base_dir, root_dir, max_angle_deg=24.0, n_az=8):
    base = _unit(0.82 * _unit(base_dir) + 0.18 * _unit(root_dir))
    u, v = _orth_basis(base)
    dirs = [base]
    for ang_deg in (10.0, max_angle_deg):
        a = math.radians(ang_deg)
        for k in range(n_az):
            ph = 2 * math.pi * k / n_az
            d = math.cos(a) * base + math.sin(a) * (math.cos(ph) * u + math.sin(ph) * v)
            dirs.append(_unit(d))
    return dirs


def _nearest_sample(arr, point_local):
    p = np.rint(point_local).astype(int)
    if np.any(p < 0) or np.any(p >= np.asarray(arr.shape)):
        return None
    return arr[tuple(p)]


class LADConfirmedBacktrackWorkflow:
    COMPONENTS = ("source_ct", "rca_calibration", "frozen_lad", "aorta_constraint", "backtrack", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LAD_Confirmed_Backtrack_v1"
        self.out = self.root / "LAD_Confirmed_Backtrack_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.rca_cal = None
        self.prior_lad = self.frozen_lad = None
        self.frozen_qc = self.frozen_summary = None
        self.rca_root = None
        self.aorta_lo = self.aorta_crop = self.aorta_dist = None
        self.backtrack_path = self.backtrack_qc = self.backtrack_summary = None
        self.combined_lad = None
        self.origin = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {
            "source_ct": "series7_int16.npy",
            "rca_calibration": "validated_rca_calibration.json",
            "frozen_lad": "frozen_lad_centerline.csv",
            "aorta_constraint": "aorta_constraint.npz",
            "backtrack": "backtrack_summary.json",
            "figures": "figures.done",
            "report": "report.done",
        }
        return pd.DataFrame([{"component": c, "reuse": self.reuse[c], "cache_exists": (self.cache / names[c]).exists()} for c in self.COMPONENTS])

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        priors = [
            self.root / "Cache" / "LAD_Origin_Backtrack_v1" / "series7_int16.npy",
            self.root / "Cache" / "Left_Coronary_Ostium_Neck_v1" / "series7_int16.npy",
            self.root / "Cache" / "Left_Main_Bifurcation_v1" / "series7_int16.npy",
        ]
        if not self.reuse["source_ct"]:
            for fp in (own, own.with_suffix(".json")):
                if fp.exists():
                    fp.unlink()
            priors = []
        self.ct, self.meta, action = stream_source_ct_to_memmap(self.root, own, reuse_sources=priors)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        src = own if action in ("reused_own_cache", "recomputed_and_cached") else action.split(":", 1)[-1]
        self._record("source_ct", action, src)
        _ram("After source CT")
        return self.ct

    def load_validated_rca_calibration(self):
        fp = self.cache / "validated_rca_calibration.json"
        if self.reuse["rca_calibration"] and fp.exists():
            self.rca_cal = _json_read(fp)
            self._record("rca_calibration", "reused", fp)
            return self.rca_cal

        candidates = [
            self.root / "Cache" / "Left_Main_Bifurcation_v1" / "rca_calibration.json",
            self.root / "Left_Main_Bifurcation_Report" / "rca_calibration.json",
        ]
        chosen = None
        for c in candidates:
            if not c.exists():
                continue
            try:
                x = _json_read(c)
            except Exception:
                continue
            radius = float(x.get("median_radius_mm", np.nan))
            passes = float(x.get("plane_pass_fraction", np.nan))
            score = float(x.get("median_plane_score", np.nan))
            if 1.0 <= radius <= 2.3 and passes >= 0.75 and score >= 0.70:
                chosen = dict(x)
                chosen["source"] = str(c)
                break
        if chosen is None:
            chosen = dict(PINNED_VALIDATED_RCA)
            action = "pinned_validated_fallback"
        else:
            action = "imported_prior_validated_calibration"
        self.rca_cal = chosen
        _json_write(chosen, fp)
        self._record("rca_calibration", action, fp, f"radius={chosen['median_radius_mm']:.3f}, pass={chosen['plane_pass_fraction']:.3f}")
        return chosen

    def freeze_confirmed_lad(self, start_arc_mm=32.5, end_arc_mm=55.5):
        pf = self.cache / "frozen_lad_centerline.csv"
        qf = self.cache / "frozen_lad_qc.csv"
        sf = self.cache / "frozen_lad_summary.json"
        if self.ct is None:
            self.load_source_ct()
        if self.rca_cal is None:
            self.load_validated_rca_calibration()
        if self.reuse["frozen_lad"] and pf.exists() and qf.exists() and sf.exists():
            self.frozen_lad = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.frozen_qc = pd.read_csv(qf)
            self.frozen_summary = _json_read(sf)
            self._record("frozen_lad", "reused", pf)
            return self.frozen_summary

        prior_fp = self.root / "Cache" / "LAD_Origin_Backtrack_v1" / "best_lad_centerline.csv"
        if not prior_fp.exists():
            raise FileNotFoundError(f"Required prior LAD path not found: {prior_fp}")
        prior = pd.read_csv(prior_fp)
        self.prior_lad = prior[["z", "y", "x"]].to_numpy(float)
        s = prior["arc_mm"].to_numpy(float) if "arc_mm" in prior.columns else arc_mm(self.prior_lad, self.spacing)
        m = (s >= float(start_arc_mm)) & (s <= float(end_arc_mm))
        if int(m.sum()) < 5:
            raise RuntimeError("Prior LAD path does not contain enough points in requested 32.5-55.5 mm freeze interval")
        seg = self.prior_lad[m]
        seg = resample_path(seg, self.spacing, 0.35)
        qdf, qsum = serial_qc(seg, self.ct, self.spacing, self.rca_cal, step_sample_mm=1.6, label="FROZEN_LAD")
        accepted = bool(qsum["plane_pass_fraction"] >= 0.70 and qsum["median_plane_score"] >= 0.78 and qsum["median_offset_mm"] <= 0.85)
        qsum.update({"accepted": accepted, "source_prior_path": str(prior_fp), "requested_prior_arc_interval_mm": [float(start_arc_mm), float(end_arc_mm)]})
        self.frozen_lad, self.frozen_qc, self.frozen_summary = seg, qdf, qsum
        pd.DataFrame({"arc_mm": arc_mm(seg, self.spacing), "z": seg[:, 0], "y": seg[:, 1], "x": seg[:, 2]}).to_csv(pf, index=False)
        qdf.to_csv(qf, index=False)
        _json_write(qsum, sf)
        self._record("frozen_lad", "recomputed_and_cached", pf, f"accepted={accepted}, pass={qsum['plane_pass_fraction']:.3f}")
        return qsum

    def build_aorta_constraint(self, pad_mm=24.0):
        fp = self.cache / "aorta_constraint.npz"
        if self.frozen_lad is None:
            self.freeze_confirmed_lad()
        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        self.rca_root = pd.read_csv(rca_fp)[["z", "y", "x"]].to_numpy(float)[0]
        if self.reuse["aorta_constraint"] and fp.exists():
            z = np.load(fp, allow_pickle=False)
            self.aorta_lo = z["lo"].astype(int)
            self.aorta_crop = z["aorta"].astype(bool)
            self.aorta_dist = z["dist"].astype(np.float32)
            self._record("aorta_constraint", "reused", fp)
            return self.aorta_dist

        aorta_fp = self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz"
        aimg = sitk.ReadImage(str(aorta_fp))
        full = sitk.GetArrayFromImage(aimg).astype(bool)
        if tuple(full.shape) != tuple(self.ct.shape):
            raise RuntimeError(f"Aorta mask shape {full.shape} != source CT {self.ct.shape}")
        pts = np.vstack([self.frozen_lad, self.rca_root[None, :]])
        pad = np.ceil(float(pad_mm) / self.spacing).astype(int)
        lo = np.maximum(0, np.floor(pts.min(axis=0)).astype(int) - pad)
        hi = np.minimum(np.asarray(full.shape), np.ceil(pts.max(axis=0)).astype(int) + pad + 1)
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        crop = np.asarray(full[sl], dtype=bool)
        dist = ndi.distance_transform_edt(~crop, sampling=self.spacing).astype(np.float32)
        np.savez_compressed(fp, lo=lo.astype(np.int32), aorta=crop.astype(np.uint8), dist=dist)
        self.aorta_lo, self.aorta_crop, self.aorta_dist = lo, crop, dist
        del full
        gc.collect()
        self._record("aorta_constraint", "recomputed_and_cached", fp, "TotalSegmentator aorta used only as constraint")
        _ram("After aorta constraint")
        return dist

    def _aorta_distance(self, p_source):
        q = np.asarray(p_source, float) - self.aorta_lo
        if np.any(q < 0) or np.any(q >= np.asarray(self.aorta_dist.shape) - 1):
            return float("inf"), False
        d = float(map_coordinates(self.aorta_dist, q[:, None], order=1, mode="nearest", prefilter=False)[0])
        inside = bool(_nearest_sample(self.aorta_crop, q))
        return d, inside

    def backtrack(self, step_mm=0.80, max_backtrack_mm=24.0, beam_width=5):
        pf = self.cache / "backtrack_centerline.csv"
        qf = self.cache / "backtrack_qc.csv"
        sf = self.cache / "backtrack_summary.json"
        cf = self.cache / "combined_lad_centerline.csv"
        of = self.cache / "lad_origin_estimate.json"
        if self.frozen_lad is None:
            self.freeze_confirmed_lad()
        if self.aorta_dist is None:
            self.build_aorta_constraint()
        if self.rca_cal is None:
            self.load_validated_rca_calibration()
        if self.reuse["backtrack"] and all(x.exists() for x in (pf, qf, sf, cf, of)):
            self.backtrack_path = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.backtrack_qc = pd.read_csv(qf)
            self.backtrack_summary = _json_read(sf)
            self.combined_lad = pd.read_csv(cf)[["z", "y", "x"]].to_numpy(float)
            self.origin = _json_read(of)
            self._record("backtrack", "reused", sf)
            return self.backtrack_summary

        if not bool(self.frozen_summary.get("accepted", False)):
            self.backtrack_path = self.frozen_lad[:1].copy()
            self.backtrack_qc = pd.DataFrame()
            self.backtrack_summary = {"status": "FROZEN_SEGMENT_REJECTED", "algorithm": ALGORITHM_VERSION, "message": "Corrected RCA calibration did not validate the proposed 32.5-55.5 mm LAD segment."}
            self.combined_lad = self.frozen_lad.copy()
            self.origin = {"status": "NOT_ESTIMATED"}
            _json_write(self.backtrack_summary, sf); _json_write(self.origin, of)
            self._record("backtrack", "not_run", sf, "frozen LAD failed acceptance gate")
            return self.backtrack_summary

        p = resample_path(self.frozen_lad, self.spacing, 0.35)
        # Frozen segment is inherited in proximal->distal orientation.  The search
        # starts at its proximal end and points opposite the first few-mm tangent.
        anchor = p[0].copy()
        j = min(len(p) - 1, max(3, int(round(3.0 / 0.35))))
        distal_tangent = _unit((p[j] - p[0]) * self.spacing)
        back_dir = -distal_tangent
        root_mm = self.rca_root * self.spacing
        root_dir = _unit(root_mm - anchor * self.spacing)

        init = {"point": anchor, "direction": back_dir, "path": [anchor.copy()], "plane_scores": [], "hard": [], "soft": [], "bad_streak": 0, "objective": 0.0}
        beams = [init]
        completed = []
        nsteps = int(math.ceil(float(max_backtrack_mm) / float(step_mm)))
        shape = np.asarray(self.ct.shape, float)

        for step in range(nsteps):
            proposals = []
            for st in beams:
                cur = st["point"]
                cur_root = float(np.linalg.norm(cur * self.spacing - root_mm))
                local_root_dir = _unit(root_mm - cur * self.spacing)
                for d in _cone_directions(st["direction"], local_root_dir, max_angle_deg=24.0, n_az=8):
                    nxt = cur + (float(step_mm) * d) / self.spacing
                    if np.any(nxt < 2) or np.any(nxt >= shape - 3):
                        continue
                    aorta_d, inside = self._aorta_distance(nxt)
                    if inside or aorta_d < 0.70:
                        continue
                    im, c = orthogonal_plane(self.ct, nxt, d, self.spacing)
                    met = plane_lumen_metrics(im, c)
                    plane_score, hard_pass, soft_pass = score_plane(met, self.rca_cal)
                    if not soft_pass:
                        # Keep at most a short transitional wobble near a branch.
                        if st["bad_streak"] >= 1 or plane_score < 0.46:
                            continue
                    new_root = float(np.linalg.norm(nxt * self.spacing - root_mm))
                    progress = float(np.clip((cur_root - new_root) / max(step_mm, 1e-6), -1.0, 1.0))
                    smooth = float(np.clip(np.dot(_unit(st["direction"]), _unit(d)), -1.0, 1.0))
                    local_obj = 0.68 * plane_score + 0.12 * float(hard_pass) + 0.10 * ((smooth + 1.0) / 2.0) + 0.10 * ((progress + 1.0) / 2.0)
                    bad_streak = 0 if soft_pass else st["bad_streak"] + 1
                    ns = {
                        "point": nxt,
                        "direction": d,
                        "path": st["path"] + [nxt.copy()],
                        "plane_scores": st["plane_scores"] + [plane_score],
                        "hard": st["hard"] + [bool(hard_pass)],
                        "soft": st["soft"] + [bool(soft_pass)],
                        "bad_streak": bad_streak,
                        "objective": st["objective"] + local_obj,
                    }
                    proposals.append(ns)
            if not proposals:
                completed.extend(beams)
                break

            # Rank primarily on mean local objective and continuity length.  NMS
            # prevents the beam from collapsing to five nearly identical points.
            def rank_state(st):
                n = max(1, len(st["plane_scores"]))
                hard_frac = float(np.mean(st["hard"])) if st["hard"] else 0.0
                return st["objective"] / n + 0.12 * hard_frac + 0.004 * n
            proposals.sort(key=rank_state, reverse=True)
            keep = []
            for st in proposals:
                if all(np.linalg.norm((st["point"] - q["point"]) * self.spacing) >= 0.55 for q in keep):
                    keep.append(st)
                if len(keep) >= int(beam_width):
                    break
            completed.extend(beams)
            beams = keep

        completed.extend(beams)
        viable = []
        for st in completed:
            if len(st["path"]) < 4:
                continue
            L = (len(st["path"]) - 1) * float(step_mm)
            hard_frac = float(np.mean(st["hard"])) if st["hard"] else 0.0
            soft_frac = float(np.mean(st["soft"])) if st["soft"] else 0.0
            mean_score = float(np.mean(st["plane_scores"])) if st["plane_scores"] else 0.0
            root_gain = float(np.linalg.norm(anchor * self.spacing - root_mm) - np.linalg.norm(st["point"] * self.spacing - root_mm))
            final_rank = 0.38 * hard_frac + 0.32 * mean_score + 0.17 * min(L / 16.0, 1.0) + 0.08 * soft_frac + 0.05 * np.clip(root_gain / 12.0, -1, 1)
            viable.append((final_rank, L, hard_frac, soft_frac, mean_score, root_gain, st))
        if not viable:
            raise RuntimeError("No proximal backtrack trajectory survived the source-resolution lumen gate")
        viable.sort(key=lambda x: x[0], reverse=True)
        final_rank, L, hard_frac, soft_frac, mean_score, root_gain, best = viable[0]

        # best path is anchor->proximal.  Reverse for proximal->anchor and join to
        # the already frozen proximal->distal segment.
        back = np.asarray(best["path"], float)[::-1]
        back = resample_path(back, self.spacing, 0.35)
        qdf, qsum = serial_qc(back, self.ct, self.spacing, self.rca_cal, step_sample_mm=1.2, label="BACKTRACK")
        frozen = resample_path(self.frozen_lad, self.spacing, 0.35)
        combined = np.vstack([back[:-1], frozen]) if len(back) > 1 else frozen.copy()
        combined = resample_path(combined, self.spacing, 0.35)

        origin_pt = back[0]
        origin_lps = source_zyx_to_lps(self.meta, origin_pt[None, :])[0]
        aorta_d, _ = self._aorta_distance(origin_pt)
        status = "PASS" if (L >= 6.0 and qsum["plane_pass_fraction"] >= 0.65 and qsum["median_plane_score"] >= 0.76) else ("REVIEW" if L >= 3.0 and qsum["soft_pass_fraction"] >= 0.65 else "FAIL")
        self.origin = {
            "status": "ESTIMATED" if status in ("PASS", "REVIEW") else "UNVALIDATED",
            "source_zyx": [float(x) for x in origin_pt],
            "lps_mm": [float(x) for x in origin_lps],
            "distance_to_TotalSegmentator_aorta_mm": float(aorta_d),
            "note": "Proximal endpoint of source-resolution LAD backtrack. This is an estimated LAD takeoff neighborhood, not a manually labelled bifurcation and not an aortic ostium.",
        }
        self.backtrack_path, self.backtrack_qc, self.combined_lad = back, qdf, combined
        self.backtrack_summary = {
            "algorithm": ALGORITHM_VERSION,
            "status": status,
            "backtrack_length_mm": float(L),
            "beam_rank_score": float(final_rank),
            "beam_hard_pass_fraction": float(hard_frac),
            "beam_soft_pass_fraction": float(soft_frac),
            "beam_mean_plane_score": float(mean_score),
            "root_distance_reduction_mm": float(root_gain),
            **{f"serial_{k}": v for k, v in qsum.items() if k != "label"},
            "frozen_segment_pass_fraction": float(self.frozen_summary["plane_pass_fraction"]),
            "frozen_segment_median_plane_score": float(self.frozen_summary["median_plane_score"]),
        }

        pd.DataFrame({"arc_mm": arc_mm(back, self.spacing), "z": back[:, 0], "y": back[:, 1], "x": back[:, 2]}).to_csv(pf, index=False)
        qdf.to_csv(qf, index=False)
        pd.DataFrame({"arc_mm": arc_mm(combined, self.spacing), "z": combined[:, 0], "y": combined[:, 1], "x": combined[:, 2]}).to_csv(cf, index=False)
        _json_write(self.backtrack_summary, sf)
        _json_write(self.origin, of)
        self._record("backtrack", "recomputed_and_cached", sf, f"status={status}, length={L:.1f} mm")
        _ram("After LAD backtrack")
        return self.backtrack_summary

    def _plot_sections(self, path, axs, label, n=8):
        p = resample_path(path, self.spacing, 0.35)
        s = arc_mm(p, self.spacing)
        ss = np.linspace(0.5, max(0.5, s[-1] - 0.4), n)
        for ax, x in zip(axs, ss):
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
            t = (p[i1] - p[i0]) * self.spacing
            im, c = orthogonal_plane(self.ct, p[i], t, self.spacing)
            ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower")
            ax.plot(0, 0, "+", ms=8)
            ax.set_aspect("equal")
            ax.set_title(f"{label}\n{x:.1f} mm")

    def plot_qc(self):
        done = self.cache / "figures.done"
        figs = [self.out / f for f in (
            "01_frozen_lad_corrected_qc.png",
            "02_backtrack_cross_sections.png",
            "03_combined_lad_source_mips.png",
            "04_backtrack_profiles.png",
            "05_origin_neighborhood.png",
            "06_combined_lad_lps_course.png",
        )]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs):
            self._record("figures", "reused", done)
            return figs
        if self.backtrack_summary is None:
            self.backtrack()

        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        self._plot_sections(self.frozen_lad, axs.ravel(), "Frozen LAD", n=8)
        fig.suptitle(f"Frozen 32.5-55.5 mm prior LAD segment with validated RCA calibration — pass {self.frozen_summary['plane_pass_fraction']:.2f}")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[0], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        if self.backtrack_path is not None and len(self.backtrack_path) > 2:
            self._plot_sections(self.backtrack_path, axs.ravel(), "Backtrack", n=8)
        else:
            for ax in axs.ravel(): ax.axis("off")
        fig.suptitle(f"Proximal LAD backtrack — {self.backtrack_summary.get('status')}")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[1], dpi=180, bbox_inches="tight"); plt.close(fig)

        pts = np.vstack([self.combined_lad, self.rca_root[None, :]]) if self.combined_lad is not None else self.frozen_lad
        pad = np.ceil(10.0 / self.spacing).astype(int)
        lo = np.maximum(0, np.floor(pts.min(0)).astype(int) - pad); hi = np.minimum(np.asarray(self.ct.shape), np.ceil(pts.max(0)).astype(int) + pad + 1)
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3)); crop = np.asarray(self.ct[sl], dtype=np.int16)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5)); ims = [crop.max(0), crop.max(1), crop.max(2)]
        for ax, im, title in zip(axs, ims, ("Axial MIP", "Coronal MIP", "Sagittal MIP")):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower"); ax.set_title(title); ax.axis("off")
        if self.combined_lad is not None:
            q = self.combined_lad - lo
            axs[0].plot(q[:, 2], q[:, 1], lw=2); axs[1].plot(q[:, 2], q[:, 0], lw=2); axs[2].plot(q[:, 1], q[:, 0], lw=2)
        rr = self.rca_root - lo
        axs[0].plot(rr[2], rr[1], "x", ms=9); axs[1].plot(rr[2], rr[0], "x", ms=9); axs[2].plot(rr[1], rr[0], "x", ms=9)
        fig.suptitle("Combined source-resolution LAD: backtrack + frozen confirmed segment")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[2], dpi=180, bbox_inches="tight"); plt.close(fig); del crop

        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        if self.backtrack_qc is not None and len(self.backtrack_qc):
            d = self.backtrack_qc
            axs[0].plot(d.arc_mm, d.radius_mm, marker="o"); axs[0].axhline(self.rca_cal["median_radius_mm"], ls="--"); axs[0].set_ylabel("radius mm")
            axs[1].plot(d.arc_mm, d.centroid_offset_mm, marker="o"); axs[1].axhline(1.15, ls="--"); axs[1].set_ylabel("offset mm")
            axs[2].plot(d.arc_mm, d.plane_score, marker="o"); axs[2].axhline(.76, ls="--"); axs[2].set_ylabel("plane score"); axs[2].set_xlabel("proximal-to-distal backtrack arc mm")
        fig.suptitle("Backtrack source-resolution lumen profile"); fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(figs[3], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        if self.combined_lad is not None:
            p = resample_path(self.combined_lad, self.spacing, 0.35); s = arc_mm(p, self.spacing); ss = np.linspace(0, min(10.0, s[-1]), 8)
            for ax, x in zip(axs.ravel(), ss):
                i = int(np.argmin(abs(s - x))); i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4); t = (p[i1] - p[i0]) * self.spacing
                im, c = orthogonal_plane(self.ct, p[i], t, self.spacing); ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower"); ax.plot(0, 0, "+", ms=8); ax.set_aspect("equal"); ax.set_title(f"origin +{x:.1f} mm")
        fig.suptitle("Estimated proximal LAD takeoff neighborhood"); fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[4], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
        if self.combined_lad is not None:
            l = source_zyx_to_lps(self.meta, self.combined_lad); root = source_zyx_to_lps(self.meta, self.rca_root[None, :])[0]
            pairs = [(0, 2, "L-R vs S-I"), (1, 2, "P-A vs S-I"), (0, 1, "L-R vs P-A")]
            for ax, (a, b, title) in zip(axs, pairs):
                ax.plot(l[:, a], l[:, b], ".-"); ax.plot(root[a], root[b], "x", ms=9, label="RCA-root ref"); ax.set_title(title); ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8)
        fig.suptitle("Combined LAD course in patient LPS coordinates"); fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[5], dpi=180, bbox_inches="tight"); plt.close(fig)

        done.write_text(ALGORITHM_VERSION)
        self._record("figures", "recomputed_and_cached", done)
        return figs

    def package(self):
        done = self.cache / "report.done"
        zpath = self.out / "OPENPLAQUE_LAD_CONFIRMED_BACKTRACK_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists():
            self._record("report", "reused", zpath)
            return zpath
        figs = self.plot_qc()
        html = self.out / "OPENPLAQUE_LAD_CONFIRMED_BACKTRACK_REPORT.html"
        def img(fp):
            b64 = base64.b64encode(Path(fp).read_bytes()).decode()
            return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{b64}'>"
        html.write_text(
            "<html><body><h1>OpenPlaque — confirmed LAD segment proximal backtrack</h1>"
            "<p><b>Research use only.</b> The distal/proximal LAD segment from the preceding experiment is revalidated with the previously validated RCA calibration, then frozen. The search proceeds only proximally from that segment. TotalSegmentator contributes only the aorta constraint. No LCX or plaque processing is performed.</p>"
            + "".join(img(f) for f in figs)
            + "<h2>Validated RCA calibration</h2><pre>" + json.dumps(self.rca_cal, indent=2) + "</pre>"
            + "<h2>Frozen LAD</h2><pre>" + json.dumps(self.frozen_summary, indent=2) + "</pre>"
            + "<h2>Backtrack summary</h2><pre>" + json.dumps(self.backtrack_summary, indent=2) + "</pre>"
            + "<h2>Estimated LAD origin/takeoff</h2><pre>" + json.dumps(self.origin, indent=2) + "</pre></body></html>",
            encoding="utf-8",
        )
        names = [f.name for f in figs] + ["cache_provenance.csv", html.name]
        for fp in (
            self.cache / "validated_rca_calibration.json",
            self.cache / "frozen_lad_centerline.csv",
            self.cache / "frozen_lad_qc.csv",
            self.cache / "frozen_lad_summary.json",
            self.cache / "backtrack_centerline.csv",
            self.cache / "backtrack_qc.csv",
            self.cache / "backtrack_summary.json",
            self.cache / "combined_lad_centerline.csv",
            self.cache / "lad_origin_estimate.json",
        ):
            if fp.exists():
                shutil.copyfile(fp, self.out / fp.name)
                names.append(fp.name)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n in dict.fromkeys(names):
                fp = self.out / n
                if fp.exists():
                    z.write(fp, arcname=n)
        done.write_text(ALGORITHM_VERSION)
        self._record("report", "recomputed_and_cached", zpath)
        return zpath
