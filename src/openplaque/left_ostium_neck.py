from __future__ import annotations

"""Diameter-constrained left coronary ostium/neck detection in source CCTA.

This experiment deliberately stops before LAD/LCX tracking.  It searches a small
full-resolution aortic-root crop for *skeleton centerlines* of contrast-filled
structures whose local 3-D radius is coronary-sized.  This avoids proposing rays
through broad chambers: a large chamber can be bright, but its medial skeleton has
a large distance-transform radius and is removed before candidate generation.

Research use only.
"""

import base64
import gc
import heapq
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
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    label as ndi_label,
    map_coordinates,
)
from skimage.morphology import skeletonize

from .study import OpenPlaqueStudy

ALGORITHM_VERSION = "left-ostium-neck-v1.0"
SOURCE_SERIES = 7


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
    if s[-1] <= step_mm:
        return p.copy()
    q = np.arange(0, s[-1] + 0.5 * step_mm, step_mm)
    q[-1] = min(q[-1], s[-1])
    return np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)])


def _series7_files(root, extract_root="/content/full_dicom_ostium_neck"):
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
        raise RuntimeError("No DICOM files for series 7")
    return files


def stream_source_ct_to_memmap(root, out_path, reuse_sources=()):
    """Return a disk-backed int16 source CCTA, importing an existing cache if possible."""
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        return np.load(out_path, mmap_mode="r"), _json_read(meta_path), "reused_own_cache"

    for candidate in reuse_sources:
        candidate = Path(candidate)
        cmeta = candidate.with_suffix(".json")
        if candidate.exists() and cmeta.exists():
            meta = _json_read(cmeta)
            mm = np.load(candidate, mmap_mode="r")
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


def _orth_basis(tangent_zyx_mm):
    t = np.asarray(tangent_zyx_mm, float)
    t /= max(np.linalg.norm(t), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u)
    v /= max(np.linalg.norm(v), 1e-9)
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=6.0, pix_mm=0.18):
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
    lab, nlab = ndi_label(bright, structure=np.ones((3, 3), np.uint8))
    cy = cx = len(c) // 2
    chosen = int(lab[cy, cx])
    if chosen == 0:
        yy, xx = np.nonzero(bright)
        if len(yy):
            dist = np.hypot(c[yy], c[xx])
            j = int(np.argmin(dist))
            if dist[j] <= 1.0:
                chosen = int(lab[yy[j], xx[j]])
    comp = lab == chosen if chosen > 0 else np.zeros_like(bright)
    area_px = int(comp.sum())
    radius = math.sqrt(area_px * pix * pix / math.pi) if area_px else float("nan")
    if area_px:
        yy, xx = np.nonzero(comp)
        ym = float(np.mean(c[yy])); xm = float(np.mean(c[xx]))
        offset = float(math.hypot(ym, xm))
        er = comp.copy()
        if comp.shape[0] > 2 and comp.shape[1] > 2:
            from scipy.ndimage import binary_erosion
            er = binary_erosion(comp, structure=np.ones((3, 3), bool))
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


def serial_qc(path, ct, spacing_zyx, rca_cal=None, n_samples=6, label="candidate"):
    p = resample_path(path, spacing_zyx, 0.4)
    s = arc_mm(p, spacing_zyx)
    if len(p) < 5 or s[-1] < 2.0:
        return pd.DataFrame(), {"label": label, "length_mm": float(s[-1]) if len(s) else 0.0, "median_plane_score": 0.0, "plane_pass_fraction": 0.0}
    start = min(1.5, 0.18 * s[-1])
    stop = max(start, s[-1] - 0.8)
    sample_s = np.linspace(start, stop, n_samples)
    rows = []
    rr = float(rca_cal.get("median_radius_mm", 1.6)) if rca_cal else 1.6
    rh = float(rca_cal.get("median_center_hu", 560.0)) if rca_cal else 560.0
    rmax = min(3.3, 2.05 * rr + 0.15)
    for ss in sample_s:
        i = int(np.argmin(abs(s - ss)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing_zyx, float)
        if np.linalg.norm(t) < 1e-7:
            continue
        im, c = orthogonal_plane(ct, p[i], t, spacing_zyx)
        m = plane_lumen_metrics(im, c)
        r = m["radius_mm"]
        radius_score = float(np.exp(-0.5 * ((r - min(max(rr * 1.25, 1.2), 2.6)) / max(0.65 * rr, 0.7)) ** 2)) if np.isfinite(r) else 0.0
        offset_score = float(np.exp(-0.5 * (m["centroid_offset_mm"] / 0.8) ** 2)) if np.isfinite(m["centroid_offset_mm"]) else 0.0
        circ_score = float(np.clip(m["circularity"] / 0.55, 0, 1)) if np.isfinite(m["circularity"]) else 0.0
        hu_score = float(np.exp(-0.5 * ((m["center_hu"] - rh) / 330.0) ** 2))
        contrast_score = float(1 / (1 + np.exp(-(m["core_minus_ring_hu"] - 25.0) / 80.0)))
        score = 0.31 * radius_score + 0.27 * offset_score + 0.18 * circ_score + 0.13 * hu_score + 0.11 * contrast_score
        passed = bool(
            np.isfinite(r)
            and 0.55 * rr <= r <= rmax
            and np.isfinite(m["centroid_offset_mm"]) and m["centroid_offset_mm"] <= 1.25
            and np.isfinite(m["circularity"]) and m["circularity"] >= 0.20
            and 120 <= m["center_hu"] <= 1100
        )
        rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_score": float(score), "plane_pass": passed})
    df = pd.DataFrame(rows)
    return df, {
        "label": label,
        "length_mm": float(s[-1]),
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else float("nan"),
        "median_offset_mm": float(df.centroid_offset_mm.median()) if len(df) else float("nan"),
        "median_circularity": float(df.circularity.median()) if len(df) else float("nan"),
        "median_center_hu": float(df.center_hu.median()) if len(df) else float("nan"),
        "median_core_minus_ring_hu": float(df.core_minus_ring_hu.median()) if len(df) else float("nan"),
    }


def _aorta_center_yx(mask, z):
    z = int(np.clip(round(z), 0, mask.shape[0] - 1))
    yy, xx = np.nonzero(mask[z])
    if len(yy) == 0:
        for d in range(1, 6):
            for zz in (z - d, z + d):
                if 0 <= zz < mask.shape[0]:
                    yy, xx = np.nonzero(mask[zz])
                    if len(yy):
                        return np.array([float(yy.mean()), float(xx.mean())])
        raise RuntimeError("No aorta voxels near requested z")
    return np.array([float(yy.mean()), float(xx.mean())])


def _angle_deg(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a /= max(np.linalg.norm(a), 1e-9); b /= max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def _component_longest_path(coords, spacing_zyx, seed_index):
    """Dijkstra on a small 26-connected skeleton component; returns seed->farthest path."""
    coords = np.asarray(coords, int)
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    neigh = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]
    sp = np.asarray(spacing_zyx, float)
    dist = np.full(len(coords), np.inf)
    prev = np.full(len(coords), -1, int)
    dist[seed_index] = 0.0
    heap = [(0.0, int(seed_index))]
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]:
            continue
        p = coords[i]
        for off in neigh:
            q = (int(p[0] + off[0]), int(p[1] + off[1]), int(p[2] + off[2]))
            j = lookup.get(q)
            if j is None:
                continue
            w = float(np.linalg.norm(np.asarray(off, float) * sp))
            nd = d + w
            if nd < dist[j]:
                dist[j] = nd; prev[j] = i; heapq.heappush(heap, (nd, j))
    finite = np.where(np.isfinite(dist))[0]
    if not len(finite):
        return coords[[seed_index]], 0.0
    end = int(finite[np.argmax(dist[finite])])
    chain = []
    j = end
    while j >= 0:
        chain.append(j)
        if j == seed_index:
            break
        j = int(prev[j])
    chain = chain[::-1]
    return coords[chain], float(dist[end])


class LeftOstiumNeckWorkflow:
    COMPONENTS = ("source_ct", "root_crop", "rca_calibration", "neck_candidates", "neck_qc", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Left_Coronary_Ostium_Neck_v1"
        self.out = self.root / "Left_Coronary_Ostium_Neck_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.rca = self.rca_cal = self.rca_qc = None
        self.crop_lo = self.root_ct = self.root_aorta = self.dist_aorta = self.radius_map = self.skeleton = None
        self.candidates = []
        self.candidate_table = self.best = self.best_qc = self.best_summary = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        return pd.DataFrame([
            {"component": c, "reuse": self.reuse[c], "cache_exists": (self.cache / {
                "source_ct":"series7_int16.npy", "root_crop":"root_crop.npz", "rca_calibration":"rca_calibration.json",
                "neck_candidates":"neck_candidates.csv", "neck_qc":"best_neck_summary.json", "figures":"figures.done", "report":"report.done"}[c]).exists()}
            for c in self.COMPONENTS
        ])

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        prior = self.root / "Cache" / "Left_Main_Bifurcation_v1" / "series7_int16.npy"
        if not self.reuse["source_ct"]:
            for fp in (own, own.with_suffix(".json")):
                if fp.exists(): fp.unlink()
            reuse_sources = ()
        else:
            reuse_sources = (prior,)
        self.ct, self.meta, action = stream_source_ct_to_memmap(self.root, own, reuse_sources=reuse_sources)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", action, own if action != f"imported_validated_cache:{prior}" else prior)
        _ram("After source CT")
        return self.ct

    def build_root_crop(self):
        fp = self.cache / "root_crop.npz"
        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        self.rca = pd.read_csv(rca_fp)[["z", "y", "x"]].to_numpy(float)
        if self.ct is None: self.load_source_ct()
        if self.reuse["root_crop"] and fp.exists():
            z = np.load(fp, allow_pickle=False)
            self.root_ct = z["ct"]; self.root_aorta = z["aorta"].astype(bool); self.crop_lo = z["crop_lo"].astype(int)
            self._record("root_crop", "reused", fp)
        else:
            aorta_fp = self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz"
            aimg = sitk.ReadImage(str(aorta_fp)); full_aorta = sitk.GetArrayFromImage(aimg).astype(bool)
            if full_aorta.shape != self.ct.shape:
                raise RuntimeError(f"Aorta mask shape {full_aorta.shape} does not match source CT {self.ct.shape}")
            seed = np.asarray(self.rca[0], float)
            half_mm = np.array([18.0, 42.0, 42.0])
            half = np.ceil(half_mm / self.spacing).astype(int)
            lo = np.maximum(0, np.floor(seed).astype(int) - half)
            hi = np.minimum(np.asarray(self.ct.shape), np.ceil(seed).astype(int) + half + 1)
            sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
            self.root_ct = np.asarray(self.ct[sl], dtype=np.int16)
            self.root_aorta = np.asarray(full_aorta[sl], dtype=bool)
            self.crop_lo = lo
            np.savez_compressed(fp, ct=self.root_ct, aorta=self.root_aorta.astype(np.uint8), crop_lo=self.crop_lo)
            del full_aorta; gc.collect()
            self._record("root_crop", "recomputed_and_cached", fp)
        outside = ~self.root_aorta
        self.dist_aorta = distance_transform_edt(outside, sampling=self.spacing).astype(np.float32)
        blood = (self.root_ct >= 120) & (self.root_ct <= 1100) & outside
        blood = binary_closing(blood, structure=np.ones((3, 3, 3), bool), iterations=1)
        self.radius_map = distance_transform_edt(blood, sampling=self.spacing).astype(np.float32)
        self.skeleton = skeletonize(blood).astype(bool)
        _ram("After root crop + skeleton")
        return self.root_ct

    def calibrate_rca(self):
        fpj = self.cache / "rca_calibration.json"; fpc = self.cache / "rca_serial_qc.csv"
        if self.rca is None: self.build_root_crop()
        if self.reuse["rca_calibration"] and fpj.exists() and fpc.exists():
            self.rca_cal = _json_read(fpj); self.rca_qc = pd.read_csv(fpc); self._record("rca_calibration", "reused", fpj); return self.rca_cal
        qdf, _ = serial_qc(self.rca, self.ct, self.spacing, rca_cal=None, n_samples=12, label="RCA_REFERENCE")
        good = qdf[np.isfinite(qdf.radius_mm)]
        self.rca_cal = {
            "median_radius_mm": float(good.radius_mm.median()), "median_center_hu": float(good.center_hu.median()),
            "median_offset_mm": float(good.centroid_offset_mm.median()), "median_circularity": float(good.circularity.median()),
            "median_core_minus_ring_hu": float(good.core_minus_ring_hu.median()), "median_plane_score": float(good.plane_score.median()),
        }
        rr = self.rca_cal["median_radius_mm"]
        qdf["plane_pass"] = (
            qdf.radius_mm.between(0.55 * rr, min(3.3, 2.05 * rr + 0.15)) &
            (qdf.centroid_offset_mm <= 1.25) & (qdf.circularity >= 0.20) & qdf.center_hu.between(120, 1100)
        )
        self.rca_cal["plane_pass_fraction"] = float(qdf.plane_pass.mean())
        self.rca_qc = qdf; qdf.to_csv(fpc, index=False); _json_write(self.rca_cal, fpj)
        self._record("rca_calibration", "recomputed_and_cached", fpj)
        return self.rca_cal

    def find_neck_candidates(self):
        table_fp = self.cache / "neck_candidates.csv"; cdir = self.cache / "candidate_paths"
        if self.root_ct is None: self.build_root_crop()
        if self.rca_cal is None: self.calibrate_rca()
        if self.reuse["neck_candidates"] and table_fp.exists() and cdir.exists():
            tab = pd.read_csv(table_fp); self.candidates = []
            for i, row in tab.iterrows():
                pp = cdir / f"candidate_{i+1:02d}.npy"
                if pp.exists(): self.candidates.append({"path": np.load(pp), "rec": row.to_dict()})
            self.candidate_table = tab; self._record("neck_candidates", "reused", table_fp); return tab

        rr = float(self.rca_cal["median_radius_mm"])
        max_r = min(3.4, 2.1 * rr + 0.2)
        sk = self.skeleton & (self.radius_map >= max(0.45, 0.30 * rr)) & (self.radius_map <= max_r) & (self.dist_aorta <= 28.0)
        rca_local = self.rca[0] - self.crop_lo
        zmm = (np.arange(sk.shape[0]) - rca_local[0]) * self.spacing[0]
        sk &= np.abs(zmm)[:, None, None] <= 14.0
        lab, n = ndi_label(sk, structure=np.ones((3, 3, 3), np.uint8))
        rca_center = _aorta_center_yx(self.root_aorta, rca_local[0])
        rca_vec = (rca_local[1:] - rca_center) * self.spacing[1:]
        rows = []; candidates = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            if len(coords) < 5: continue
            da = self.dist_aorta[tuple(coords.T)]
            if float(da.min()) > 3.0 or float(da.max()) < 5.0: continue
            near = np.where(da <= 3.0)[0]
            if not len(near): continue
            jseed = int(near[np.argmin(da[near])]); seed = coords[jseed]
            d_rca = float(np.linalg.norm((seed - rca_local) * self.spacing))
            if d_rca < 5.0: continue
            try: ctr = _aorta_center_yx(self.root_aorta, seed[0])
            except RuntimeError: continue
            radial = (seed[1:] - ctr) * self.spacing[1:]
            angle = _angle_deg(radial, rca_vec)
            if angle < 45.0: continue
            path_local, geod = _component_longest_path(coords, self.spacing, jseed)
            if geod < 5.0: continue
            path_src = path_local.astype(float) + self.crop_lo[None, :]
            p = resample_path(path_src, self.spacing, 0.4)
            s = arc_mm(p, self.spacing)
            if s[-1] > 13.0:
                p = p[s <= 13.0]
            qdf, qsum = serial_qc(p, self.ct, self.spacing, self.rca_cal, n_samples=6, label=f"NECK_{len(candidates)+1}")
            local_idx = np.rint(p - self.crop_lo[None, :]).astype(int)
            local_idx = np.clip(local_idx, [0,0,0], np.asarray(self.root_ct.shape)-1)
            dp = self.dist_aorta[tuple(local_idx.T)]
            monotonic = float(np.mean(np.diff(dp) >= -0.55)) if len(dp) > 1 else 0.0
            median_skel_radius = float(np.median(self.radius_map[tuple(coords.T)]))
            radius_fit = float(np.exp(-0.5 * ((median_skel_radius - min(1.35 * rr, 2.4)) / max(0.65 * rr, 0.7)) ** 2))
            angle_score = float(np.clip((angle - 45.0) / 95.0, 0, 1))
            extension_score = float(np.clip(float(da.max()) / 10.0, 0, 1))
            score = 0.46*qsum["plane_pass_fraction"] + 0.22*qsum["median_plane_score"] + 0.10*radius_fit + 0.09*monotonic + 0.07*extension_score + 0.06*angle_score
            viable = bool(qsum["plane_pass_fraction"] >= 0.60 and qsum["median_plane_score"] >= 0.52 and qsum["median_radius_mm"] <= max_r)
            rec = {
                "score": float(score), "viable": viable, "component": int(k), "n_skeleton_voxels": int(len(coords)),
                "geodesic_mm": float(geod), "path_length_mm": float(qsum["length_mm"]), "min_dist_aorta_mm": float(da.min()),
                "max_dist_aorta_mm": float(da.max()), "median_skeleton_radius_mm": median_skel_radius, "angle_from_rca_deg": float(angle),
                "distance_from_rca_seed_mm": d_rca, "outward_monotonic_fraction": monotonic,
                **{f"serial_{kk}": vv for kk, vv in qsum.items() if kk != "label"},
            }
            rows.append(rec); candidates.append({"path": p, "qc": qdf, "summary": qsum, "rec": rec})
        order = np.argsort([c["rec"]["score"] for c in candidates])[::-1] if candidates else []
        self.candidates = [candidates[i] for i in order]
        rows = [c["rec"] for c in self.candidates]
        self.candidate_table = pd.DataFrame(rows)
        if len(self.candidate_table): self.candidate_table.insert(0, "rank", np.arange(1, len(self.candidate_table)+1))
        self.candidate_table.to_csv(table_fp, index=False)
        if cdir.exists(): shutil.rmtree(cdir)
        cdir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(self.candidates, 1): np.save(cdir / f"candidate_{i:02d}.npy", c["path"].astype(np.float32))
        self._record("neck_candidates", "recomputed_and_cached", table_fp, f"{len(self.candidates)} diameter-constrained skeleton candidates")
        return self.candidate_table

    def qc_necks(self):
        summary_fp = self.cache / "best_neck_summary.json"; qc_fp = self.cache / "best_neck_serial_qc.csv"; path_fp = self.cache / "best_neck_centerline.csv"
        if self.reuse["neck_qc"] and summary_fp.exists() and qc_fp.exists() and path_fp.exists():
            self.best_summary = _json_read(summary_fp); self.best_qc = pd.read_csv(qc_fp); self.best = pd.read_csv(path_fp)[["z","y","x"]].to_numpy(float)
            self._record("neck_qc", "reused", summary_fp); return self.best_summary
        if self.candidate_table is None: self.find_neck_candidates()
        if not self.candidates:
            self.best_summary = {"status":"NO_CANDIDATE", "algorithm":ALGORITHM_VERSION, "message":"No coronary-sized skeleton component touched the aortic wall and extended >=5 mm."}
            self.best_qc = pd.DataFrame(); self.best = None
        else:
            best = next((c for c in self.candidates if c["rec"]["viable"]), self.candidates[0])
            self.best = best["path"]; self.best_qc = best["qc"]; self.best_summary = dict(best["summary"])
            self.best_summary.update({"candidate_score": float(best["rec"]["score"]), "viable": bool(best["rec"]["viable"]), "algorithm": ALGORITHM_VERSION})
            self.best_summary["status"] = "PASS" if best["rec"]["viable"] else "FAIL"
            pd.DataFrame({"arc_mm":arc_mm(self.best,self.spacing), "z":self.best[:,0], "y":self.best[:,1], "x":self.best[:,2]}).to_csv(path_fp,index=False)
            self.best_qc.to_csv(qc_fp,index=False)
        _json_write(self.best_summary, summary_fp); self._record("neck_qc", "recomputed_and_cached", summary_fp)
        return self.best_summary

    def plot_qc(self):
        done = self.cache / "figures.done"
        figs = [self.out / f for f in ("01_neck_components_mips.png", "02_candidate_cross_sections.png", "03_rca_vs_best_neck.png", "04_radius_profiles.png")]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs):
            self._record("figures", "reused", done); return figs
        if self.best_summary is None: self.qc_necks()
        p = self.best
        # 1: local MIPs with candidates
        fig, axs = plt.subplots(1,3,figsize=(16,5))
        ims = [self.root_ct.max(0), self.root_ct.max(1), self.root_ct.max(2)]
        for ax, im, title in zip(axs, ims, ("Axial MIP","Coronal MIP","Sagittal MIP")):
            ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(title); ax.axis("off")
        for rank,cand in enumerate(self.candidates[:8],1):
            q = cand["path"] - self.crop_lo
            axs[0].plot(q[:,2],q[:,1],lw=2 if rank==1 else 1,alpha=0.9 if rank==1 else 0.55)
            axs[1].plot(q[:,2],q[:,0],lw=2 if rank==1 else 1,alpha=0.9 if rank==1 else 0.55)
            axs[2].plot(q[:,1],q[:,0],lw=2 if rank==1 else 1,alpha=0.9 if rank==1 else 0.55)
        fig.suptitle("Diameter-constrained skeleton candidates at the aortic root")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[0],dpi=180,bbox_inches="tight"); plt.close(fig)

        # 2: top candidate cross-sections
        fig, axs = plt.subplots(4,4,figsize=(11,11)); axs=axs.ravel()
        for row,cand in enumerate(self.candidates[:4]):
            path=cand["path"]; s=arc_mm(path,self.spacing); ss=np.linspace(min(1.5,s[-1]*.2),max(min(1.5,s[-1]*.2),s[-1]-.8),4)
            for col,x in enumerate(ss):
                ax=axs[row*4+col]; i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(path)-1,i+3); t=(path[i1]-path[i0])*self.spacing
                im,c=orthogonal_plane(self.ct,path[i],t,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"#{row+1} {s[i]:.1f} mm")
        for ax in axs[len(self.candidates[:4])*4:]: ax.axis("off")
        fig.suptitle("Top neck candidates — serial orthogonal sections"); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(figs[1],dpi=180,bbox_inches="tight"); plt.close(fig)

        # 3: RCA vs best
        fig, axs = plt.subplots(2,6,figsize=(15,5.5))
        for r,(label,path) in enumerate((("RCA reference",self.rca),("Best left-ostium neck",p))):
            if path is None:
                for ax in axs[r]: ax.axis("off")
                continue
            path=resample_path(path,self.spacing,.4); s=arc_mm(path,self.spacing); ss=np.linspace(min(1.5,s[-1]*.15),max(min(1.5,s[-1]*.15),s[-1]-.8),6)
            for col,x in enumerate(ss):
                i=int(np.argmin(abs(s-x))); i0=max(0,i-3); i1=min(len(path)-1,i+3); t=(path[i1]-path[i0])*self.spacing
                im,c=orthogonal_plane(self.ct,path[i],t,self.spacing); ax=axs[r,col]; ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"{label}\n{s[i]:.1f} mm")
        fig.suptitle(f"RCA reference vs selected left-ostium neck — {self.best_summary.get('status')}"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[2],dpi=180,bbox_inches="tight"); plt.close(fig)

        # 4: radius profiles from serial QC
        fig,ax=plt.subplots(figsize=(8,5))
        if self.rca_qc is not None and len(self.rca_qc): ax.plot(self.rca_qc.arc_mm,self.rca_qc.radius_mm,label="RCA")
        for i,cand in enumerate(self.candidates[:5],1):
            if len(cand["qc"]): ax.plot(cand["qc"].arc_mm,cand["qc"].radius_mm,label=f"Candidate {i}")
        ax.axhline(float(self.rca_cal["median_radius_mm"]),ls="--",label="RCA median")
        ax.set_xlabel("Arc length (mm)"); ax.set_ylabel("Equivalent bright-lumen radius (mm)"); ax.set_title("Coronary-size gate"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figs[3],dpi=180,bbox_inches="tight"); plt.close(fig)
        done.write_text(ALGORITHM_VERSION); self._record("figures", "recomputed_and_cached", done)
        return figs

    def package(self):
        done=self.cache/"report.done"; zpath=self.out/"OPENPLAQUE_LEFT_OSTIUM_NECK_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists(): self._record("report","reused",zpath); return zpath
        figs=self.plot_qc(); html=self.out/"OPENPLAQUE_LEFT_OSTIUM_NECK_REPORT.html"
        def img(fp):
            b64=base64.b64encode(Path(fp).read_bytes()).decode(); return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{b64}'>"
        tab=self.candidate_table if self.candidate_table is not None else pd.DataFrame()
        html.write_text("<html><body><h1>OpenPlaque — left coronary ostium neck detector</h1><p><b>Research use only.</b> No LAD/LCX tracking is attempted. Candidate generation uses diameter-constrained 3-D skeleton components outside the aorta.</p>"+"".join(img(f) for f in figs)+"<h2>Best summary</h2><pre>"+json.dumps(self.best_summary,indent=2)+"</pre><h2>Candidates</h2>"+tab.head(30).to_html(index=False)+"</body></html>",encoding="utf-8")
        names=[f.name for f in figs]+["cache_provenance.csv"]
        for fp in (self.cache/"neck_candidates.csv",self.cache/"best_neck_summary.json",self.cache/"best_neck_serial_qc.csv",self.cache/"best_neck_centerline.csv",self.cache/"rca_calibration.json",self.cache/"rca_serial_qc.csv"):
            if fp.exists(): shutil.copyfile(fp,self.out/fp.name); names.append(fp.name)
        names.append(html.name)
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for n in dict.fromkeys(names):
                fp=self.out/n
                if fp.exists(): z.write(fp,arcname=n)
        done.write_text(ALGORITHM_VERSION); self._record("report","recomputed_and_cached",zpath)
        return zpath
