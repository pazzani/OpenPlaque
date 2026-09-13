from __future__ import annotations

"""Root-directed alternatives from an already validated proximal LAD.

This experiment does not rediscover the LAD and does not trace the LCX. It starts
from the proximal endpoint of the previously validated ~57 mm LAD backbone,
generates a small set of root-directed proximal alternatives in source CCTA,
re-centers each step on an RCA-calibrated coronary-sized lumen component, and
compares alternatives by serial orthogonal-plane QC and progress toward the
TotalSegmentator aorta. A short local branch probe is used only to look for a
possible LAD takeoff/bifurcation neighborhood; it is not an LCX centerline.

Research use only.
"""

import base64
import gc
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

ALGORITHM_VERSION = "lad-takeoff-root-alternatives-v1.0"
PINNED_VALIDATED_RCA = {
    "label": "RCA_reference",
    "length_mm": 52.19779952034923,
    "median_plane_score": 0.9326883846581364,
    "plane_pass_fraction": 0.9166666666666666,
    "median_radius_mm": 1.6169055428727572,
    "median_offset_mm": 0.3176490169454266,
    "median_circularity": 1.2,
    "median_center_hu": 551.061478407865,
    "median_core_minus_ring_hu": 503.91744004980774,
}


def _json_read(path):
    return json.loads(Path(path).read_text())


def _json_write(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) == 0:
        return np.zeros(0, float)
    if len(p) == 1:
        return np.zeros(1, float)
    ds = np.linalg.norm(np.diff(p, axis=0) * np.asarray(spacing, float), axis=1)
    return np.r_[0.0, np.cumsum(ds)]


def resample_path(path, spacing, step_mm=0.35):
    p = np.asarray(path, float)
    if len(p) < 2:
        return p.copy()
    s = arc_mm(p, spacing)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < step_mm:
        return p.copy()
    ss = np.arange(0.0, s[-1], float(step_mm))
    if len(ss) == 0 or ss[-1] < s[-1] - 0.05:
        ss = np.r_[ss, s[-1]]
    out = np.column_stack([np.interp(ss, s, p[:, k]) for k in range(3)])
    return out


def _ram(label):
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"{label}: RSS {rss:.2f} GB")
    except Exception:
        pass


def _dicom_files_from_zip(zip_path, series_number=7):
    import zipfile as _zf
    tmp = Path("/content/full_dicom_takeoff")
    marker = tmp / ".done"
    if not marker.exists():
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with _zf.ZipFile(zip_path) as z:
            z.extractall(tmp)
        marker.write_text("ok")
    files = []
    for fp in tmp.rglob("*"):
        if not fp.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            if int(getattr(ds, "SeriesNumber", -999)) == int(series_number):
                files.append(str(fp))
        except Exception:
            continue
    if not files:
        raise RuntimeError(f"No DICOM files found for source series {series_number}")
    def key(fp):
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))
    files.sort(key=key)
    return files


def stream_source_ct_to_memmap(root, out_path, reuse_sources=()):
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        meta = _json_read(meta_path)
        return np.load(out_path, mmap_mode="r"), meta, "reused_own_cache"
    for src in reuse_sources:
        src = Path(src)
        sm = src.with_suffix(".json")
        if src.exists() and sm.exists():
            shutil.copyfile(src, out_path)
            shutil.copyfile(sm, meta_path)
            return np.load(out_path, mmap_mode="r"), _json_read(meta_path), f"imported_prior_cache:{src}"
    zip_path = Path(root) / "Full_DICOM.zip"
    files = _dicom_files_from_zip(zip_path, 7)
    ds0 = pydicom.dcmread(files[0], force=True)
    n, rows, cols = len(files), int(ds0.Rows), int(ds0.Columns)
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
    slice_vec = (positions[-1] - positions[0]) / max(len(positions) - 1, 1) if len(positions) > 1 else np.cross(row_cos, col_cos) * sp[0]
    out = positions[0][None, :] + p[:, 0, None] * slice_vec[None, :]
    out += p[:, 2, None] * sp[2] * row_cos[None, :]
    out += p[:, 1, None] * sp[1] * col_cos[None, :]
    return out


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_mm, spacing, half_mm=5.5, pix_mm=0.18):
    u, v = _orth_basis(tangent_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, [vox[..., 0], vox[..., 1], vox[..., 2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def _component_metrics(im, c, comp):
    pix = float(abs(c[1] - c[0]))
    yy, xx = np.nonzero(comp)
    if not len(yy):
        return None
    cu, cv = float(np.mean(c[xx])), float(np.mean(c[yy]))
    area = int(comp.sum())
    radius = math.sqrt(area * pix * pix / math.pi)
    er = ndi.binary_erosion(comp, structure=np.ones((3, 3), bool))
    per = max(1, int((comp & ~er).sum())) * pix
    circ = float(np.clip(4 * math.pi * area * pix * pix / max(per * per, 1e-8), 0, 1.2))
    U, V = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(U - cu, V - cv)
    center = float(np.median(im[R <= 0.70]))
    core = float(np.median(im[R <= 1.0]))
    ring = float(np.median(im[(R >= 2.5) & (R <= 4.0)]))
    return {
        "radius_mm": float(radius),
        "circularity": circ,
        "center_hu": center,
        "core_minus_ring_hu": float(core - ring),
        "centroid_u_mm": cu,
        "centroid_v_mm": cv,
        "recenter_shift_mm": float(math.hypot(cu, cv)),
    }


def plane_components(im, c, rca, max_shift_mm=2.5, near_root=False, topn=3):
    rr = float(rca["median_radius_mm"])
    rh = float(rca["median_center_hu"])
    lo_hu = max(170.0, min(260.0, 0.38 * rh))
    lab, nlab = ndi.label((im >= lo_hu) & (im <= 1200.0), structure=np.ones((3, 3), np.uint8))
    out = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, c, comp)
        if m is None:
            continue
        r, sh, circ, hu, con = (float(m[x]) for x in ("radius_mm", "recenter_shift_mm", "circularity", "center_hu", "core_minus_ring_hu"))
        rmax = 3.9 if near_root else 3.25
        if sh > max_shift_mm or not (0.65 <= r <= rmax):
            continue
        rs = math.exp(-0.5 * ((r - 1.10 * rr) / max(0.62 * rr, 0.72)) ** 2)
        ss = math.exp(-0.5 * (sh / 1.05) ** 2)
        cs = float(np.clip(circ / 0.50, 0, 1))
        hs = math.exp(-0.5 * ((hu - rh) / 350.0) ** 2)
        xs = 1.0 / (1.0 + math.exp(-(con - 10.0) / 85.0))
        score = 0.33 * rs + 0.27 * ss + 0.17 * cs + 0.13 * hs + 0.10 * xs
        hard = bool(0.55 * rr <= r <= (3.65 if near_root else min(3.25, 2.0 * rr + 0.1)) and sh <= 1.65 and circ >= 0.16 and 120 <= hu <= 1150 and score >= 0.58)
        soft = bool(0.45 * rr <= r <= rmax and sh <= max_shift_mm and circ >= 0.09 and 100 <= hu <= 1200 and score >= 0.48)
        m.update({"plane_score": float(score), "hard_pass": hard, "soft_pass": soft})
        if soft:
            out.append(m)
    out.sort(key=lambda x: x["plane_score"], reverse=True)
    return out[: int(topn)]


def serial_qc(path, ct, spacing, rca, step_sample_mm=1.1, label="path"):
    p = resample_path(path, spacing, 0.35)
    s = arc_mm(p, spacing)
    rows = []
    if len(p) >= 4:
        ss = np.arange(min(0.45, 0.10 * s[-1]), s[-1] + 1e-6, float(step_sample_mm))
        if len(ss) == 0 or ss[-1] < s[-1] - 0.4:
            ss = np.r_[ss, s[-1]]
        for x in ss:
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
            t = (p[i1] - p[i0]) * np.asarray(spacing, float)
            im, c = orthogonal_plane(ct, p[i], t, spacing)
            comps = plane_components(im, c, rca, max_shift_mm=1.55, near_root=False, topn=1)
            if not comps:
                rows.append({"label": label, "arc_mm": float(s[i]), "radius_mm": np.nan, "circularity": np.nan, "center_hu": np.nan, "core_minus_ring_hu": np.nan, "recenter_shift_mm": np.inf, "plane_score": 0.0, "plane_pass": False, "soft_pass": False})
            else:
                m = comps[0]
                rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_pass": bool(m["hard_pass"]), "soft_pass": bool(m["soft_pass"])})
    df = pd.DataFrame(rows)
    return df, {
        "label": label,
        "length_mm": float(s[-1]) if len(s) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "soft_pass_fraction": float(df.soft_pass.mean()) if len(df) else 0.0,
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else float("nan"),
        "median_recenter_shift_mm": float(df.recenter_shift_mm.replace([np.inf], np.nan).median()) if len(df) else float("nan"),
        "median_circularity": float(df.circularity.median()) if len(df) else float("nan"),
        "median_center_hu": float(df.center_hu.median()) if len(df) else float("nan"),
    }


def _cone(base, angles=(0.0, 12.0, 24.0, 36.0), n_az=8):
    base = _unit(base)
    u, v = _orth_basis(base)
    out = []
    for deg in angles:
        if deg == 0:
            out.append(base)
            continue
        a = math.radians(float(deg))
        for k in range(int(n_az)):
            ph = 2 * math.pi * k / int(n_az)
            out.append(_unit(math.cos(a) * base + math.sin(a) * (math.cos(ph) * u + math.sin(ph) * v)))
    return out


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


class LADTakeoffRootAlternativesWorkflow:
    COMPONENTS = ("source_ct", "validated_lad", "aorta_constraint", "alternatives", "branch_probe", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LAD_Takeoff_Root_Alternatives_v1"
        self.out = self.root / "LAD_Takeoff_Root_Alternatives_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = self.rca_cal = None
        self.lad = self.lad_qc = self.lad_summary = None
        self.aorta_lo = self.aorta_crop = self.aorta_dist = None
        self.alternatives = []
        self.alternative_summary = None
        self.branch_probe = None
        self.takeoff_candidate = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {
            "source_ct": "series7_int16.npy",
            "validated_lad": "validated_lad_summary.json",
            "aorta_constraint": "aorta_constraint.npz",
            "alternatives": "alternative_summary.csv",
            "branch_probe": "branch_probe_summary.csv",
            "figures": "figures.done",
            "report": "report.done",
        }
        return pd.DataFrame([{"component": k, "reuse": self.reuse[k], "cache_exists": (self.cache / v).exists()} for k, v in names.items()])

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        priors = [
            self.root / "Cache" / "LAD_Proximal_Recenter_v1" / "series7_int16.npy",
            self.root / "Cache" / "LAD_Confirmed_Backtrack_v1" / "series7_int16.npy",
            self.root / "Cache" / "LAD_Origin_Backtrack_v1" / "series7_int16.npy",
        ]
        if not self.reuse["source_ct"]:
            for p in (own, own.with_suffix(".json")):
                if p.exists():
                    p.unlink()
            priors = []
        self.ct, self.meta, action = stream_source_ct_to_memmap(self.root, own, priors)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", action, own)
        _ram("After source CT")
        return self.ct

    def validate_lad_backbone(self):
        pf = self.cache / "validated_lad_centerline.csv"
        qf = self.cache / "validated_lad_qc.csv"
        sf = self.cache / "validated_lad_summary.json"
        rf = self.cache / "validated_rca_calibration.json"
        if self.ct is None:
            self.load_source_ct()
        if self.reuse["validated_lad"] and all(p.exists() for p in (pf, qf, sf, rf)):
            self.lad = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.lad_qc = pd.read_csv(qf)
            self.lad_summary = _json_read(sf)
            self.rca_cal = _json_read(rf)
            self._record("validated_lad", "reused", sf)
            return self.lad_summary
        prior = self.root / "Cache" / "LAD_Proximal_Recenter_v1"
        ppath = prior / "combined_lad_centerline.csv"
        if not ppath.exists():
            raise FileNotFoundError(f"Required prior validated LAD not found: {ppath}")
        rpath = prior / "validated_rca_calibration.json"
        x = _json_read(rpath) if rpath.exists() else dict(PINNED_VALIDATED_RCA)
        self.rca_cal = x if 1.0 <= float(x.get("median_radius_mm", np.nan)) <= 2.3 and float(x.get("plane_pass_fraction", 0)) >= 0.75 else dict(PINNED_VALIDATED_RCA)
        _json_write(self.rca_cal, rf)
        d = pd.read_csv(ppath)
        p = resample_path(d[["z", "y", "x"]].to_numpy(float), self.spacing, 0.35)
        qdf, qsum = serial_qc(p, self.ct, self.spacing, self.rca_cal, 1.35, "VALIDATED_LAD_BACKBONE")
        accepted = bool(qsum["length_mm"] >= 45 and qsum["plane_pass_fraction"] >= 0.66 and qsum["median_plane_score"] >= 0.80)
        qsum.update({"accepted": accepted, "source_prior_path": str(ppath)})
        self.lad, self.lad_qc, self.lad_summary = p, qdf, qsum
        pd.DataFrame({"arc_mm": arc_mm(p, self.spacing), "z": p[:, 0], "y": p[:, 1], "x": p[:, 2]}).to_csv(pf, index=False)
        qdf.to_csv(qf, index=False)
        _json_write(qsum, sf)
        self._record("validated_lad", "recomputed_and_cached", sf, f"accepted={accepted}, length={qsum['length_mm']:.1f}, pass={qsum['plane_pass_fraction']:.3f}")
        return qsum

    def build_aorta_constraint(self, pad_mm=36.0):
        fp = self.cache / "aorta_constraint.npz"
        if self.lad is None:
            self.validate_lad_backbone()
        if self.reuse["aorta_constraint"] and fp.exists():
            z = np.load(fp, allow_pickle=False)
            self.aorta_lo = z["lo"].astype(int)
            self.aorta_crop = z["aorta"].astype(bool)
            self.aorta_dist = z["dist"].astype(np.float32)
            self._record("aorta_constraint", "reused", fp)
            return self.aorta_dist
        aorta_fp = self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz"
        full = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_fp))).astype(bool)
        if tuple(full.shape) != tuple(self.ct.shape):
            raise RuntimeError(f"Aorta mask shape {full.shape} != source CT shape {self.ct.shape}")
        anchor = self.lad[0]
        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        rca_root = pd.read_csv(rca_fp)[["z", "y", "x"]].to_numpy(float)[0]
        pts = np.vstack([anchor[None, :], rca_root[None, :]])
        pad = np.ceil(float(pad_mm) / self.spacing).astype(int)
        lo = np.maximum(0, np.floor(pts.min(0)).astype(int) - pad)
        hi = np.minimum(np.asarray(full.shape), np.ceil(pts.max(0)).astype(int) + pad + 1)
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        crop = np.asarray(full[sl], bool)
        dist = ndi.distance_transform_edt(~crop, sampling=self.spacing).astype(np.float32)
        np.savez_compressed(fp, lo=lo.astype(np.int32), aorta=crop.astype(np.uint8), dist=dist)
        self.aorta_lo, self.aorta_crop, self.aorta_dist = lo, crop, dist
        del full
        gc.collect()
        self._record("aorta_constraint", "recomputed_and_cached", fp, "Fresh source-space TotalSegmentator aorta crop around current LAD endpoint")
        _ram("After aorta constraint")
        return dist

    def _aorta_distance(self, p):
        q = np.asarray(p, float) - self.aorta_lo
        if np.any(q < 1) or np.any(q >= np.asarray(self.aorta_dist.shape) - 2):
            return float("inf"), False
        d = float(map_coordinates(self.aorta_dist, q[:, None], order=1, mode="nearest", prefilter=False)[0])
        qi = np.rint(q).astype(int)
        return d, bool(self.aorta_crop[tuple(qi)])

    def _aorta_gradient_dir(self, p):
        q = np.asarray(p, float) - self.aorta_lo
        if np.any(q < 2) or np.any(q >= np.asarray(self.aorta_dist.shape) - 3):
            return None
        vals = []
        for k in range(3):
            qm, qp = q.copy(), q.copy()
            qm[k] -= 1.0
            qp[k] += 1.0
            dm = float(map_coordinates(self.aorta_dist, qm[:, None], order=1, mode="nearest", prefilter=False)[0])
            dp = float(map_coordinates(self.aorta_dist, qp[:, None], order=1, mode="nearest", prefilter=False)[0])
            vals.append((dp - dm) / (2.0 * self.spacing[k]))
        g = np.asarray(vals, float)
        if np.linalg.norm(g) < 1e-5:
            return None
        return _unit(-g)

    def _propose(self, cur, direction, near_root=False):
        step_mm = 0.75
        guess = cur + (step_mm * _unit(direction)) / self.spacing
        if np.any(guess < 2) or np.any(guess >= np.asarray(self.ct.shape, float) - 3):
            return []
        im, c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        comps = plane_components(im, c, self.rca_cal, max_shift_mm=3.0 if near_root else 2.5, near_root=near_root, topn=2)
        u, v = _orth_basis(direction)
        out = []
        for m in comps:
            p = (guess * self.spacing + m["centroid_u_mm"] * u + m["centroid_v_mm"] * v) / self.spacing
            vec = (p - cur) * self.spacing
            L = float(np.linalg.norm(vec))
            if not (0.45 <= L <= (2.6 if near_root else 2.1)):
                continue
            ad, inside = self._aorta_distance(p)
            if inside or ad < 0.55:
                continue
            out.append({"point": p, "direction": _unit(vec), "aorta_distance_mm": ad, "step_length_mm": L, **m})
        return out

    def search_alternatives(self, max_extension_mm=28.0, beam_width=14, n_alternatives=4):
        sf = self.cache / "alternative_summary.csv"
        if self.lad is None:
            self.validate_lad_backbone()
        if self.aorta_dist is None:
            self.build_aorta_constraint()
        if self.reuse["alternatives"] and sf.exists():
            summary = pd.read_csv(sf)
            alts = []
            for _, r in summary.iterrows():
                fp = self.cache / f"alternative_{int(r.alternative_id):02d}_centerline.csv"
                qf = self.cache / f"alternative_{int(r.alternative_id):02d}_qc.csv"
                if fp.exists() and qf.exists():
                    alts.append({"id": int(r.alternative_id), "path": pd.read_csv(fp)[["z", "y", "x"]].to_numpy(float), "qc": pd.read_csv(qf), "summary": r.to_dict()})
            if alts:
                self.alternatives, self.alternative_summary = alts, summary
                self._record("alternatives", "reused", sf)
                return summary
        if not bool(self.lad_summary.get("accepted", False)):
            raise RuntimeError("Prior LAD backbone failed revalidation gate")
        p = resample_path(self.lad, self.spacing, 0.35)
        anchor = p[0].copy()
        j = min(len(p) - 1, max(5, int(round(4.0 / 0.35))))
        init_dir = -_unit((p[j] - p[0]) * self.spacing)
        ad0, _ = self._aorta_distance(anchor)
        init = {"point": anchor, "direction": init_dir, "path": [anchor.copy()], "scores": [], "hard": [], "soft": [], "shifts": [], "aorta": [ad0], "obj": 0.0}
        beams = [init]
        pool = []
        nsteps = int(math.ceil(float(max_extension_mm) / 0.75))
        for step in range(nsteps):
            props = []
            for st in beams:
                cur = st["point"]
                ad = st["aorta"][-1]
                root_dir = self._aorta_gradient_dir(cur)
                bases = [st["direction"]]
                if root_dir is not None:
                    for w in (0.18, 0.35, 0.52, 0.68):
                        bases.append(_unit((1.0 - w) * _unit(st["direction"]) + w * root_dir))
                dirs = []
                for b in bases:
                    dirs.extend(_cone(b, (0, 12, 24, 36 if ad > 10 else 48), 6))
                unique = []
                for d in dirs:
                    if all(_angle_deg(d, q) >= 7.0 for q in unique):
                        unique.append(d)
                near_root = bool(np.isfinite(ad) and ad <= 12.0)
                for d in unique:
                    for r in self._propose(cur, d, near_root):
                        smooth = float(np.clip(np.dot(_unit(st["direction"]), _unit(r["direction"])), -1, 1))
                        prev_ad = st["aorta"][-1]
                        prog = float(np.clip((prev_ad - r["aorta_distance_mm"]) / 1.2, -1, 1)) if np.isfinite(prev_ad) and np.isfinite(r["aorta_distance_mm"]) else 0.0
                        root_weight = 0.18 if near_root else 0.11
                        local = (0.54 - root_weight / 2) * r["plane_score"] + 0.13 * float(r["hard_pass"]) + 0.12 * ((smooth + 1) / 2) + 0.10 * math.exp(-0.5 * (r["recenter_shift_mm"] / 1.05) ** 2) + root_weight * ((prog + 1) / 2)
                        ns = {
                            "point": r["point"], "direction": r["direction"], "path": st["path"] + [r["point"].copy()],
                            "scores": st["scores"] + [r["plane_score"]], "hard": st["hard"] + [r["hard_pass"]], "soft": st["soft"] + [r["soft_pass"]],
                            "shifts": st["shifts"] + [r["recenter_shift_mm"]], "aorta": st["aorta"] + [r["aorta_distance_mm"]], "obj": st["obj"] + local,
                        }
                        props.append(ns)
            pool.extend(beams)
            if not props:
                break
            def rank(st):
                n = max(1, len(st["scores"]))
                L = arc_mm(np.asarray(st["path"]), self.spacing)[-1]
                gain = ad0 - st["aorta"][-1] if np.isfinite(ad0) and np.isfinite(st["aorta"][-1]) else 0.0
                return st["obj"] / n + 0.10 * np.mean(st["hard"]) + 0.005 * min(L, 24) + 0.004 * max(gain, 0)
            props.sort(key=rank, reverse=True)
            keep = []
            for st in props:
                if all(np.linalg.norm((st["point"] - q["point"]) * self.spacing) >= 0.7 or _angle_deg(st["direction"], q["direction"]) >= 13 for q in keep):
                    keep.append(st)
                if len(keep) >= int(beam_width):
                    break
            beams = keep
            if any(np.isfinite(st["aorta"][-1]) and st["aorta"][-1] <= 3.0 and np.mean(st["hard"][-4:]) >= 0.75 for st in beams):
                pool.extend(beams)
                break
        pool.extend(beams)
        cand = []
        for st in pool:
            if len(st["path"]) < 8:
                continue
            raw = np.asarray(st["path"], float)[::-1]
            path = resample_path(raw, self.spacing, 0.35)
            qdf, qsum = serial_qc(path, self.ct, self.spacing, self.rca_cal, 1.1, "ROOT_ALT")
            L = float(qsum["length_mm"])
            if L < 5.0:
                continue
            end_ad, _ = self._aorta_distance(path[0])
            gain = ad0 - end_ad if np.isfinite(ad0) and np.isfinite(end_ad) else 0.0
            quality = 0.55 * qsum["plane_pass_fraction"] + 0.45 * qsum["median_plane_score"]
            balanced = 0.46 * quality + 0.25 * min(max(gain, 0) / 20.0, 1) + 0.18 * min(L / 20.0, 1) + 0.11 * math.exp(-0.5 * (float(qsum["median_recenter_shift_mm"]) / 0.75) ** 2)
            cand.append({"state": st, "path": path, "qc": qdf, "qsum": qsum, "endpoint_aorta_distance_mm": end_ad, "aorta_gain_mm": gain, "quality": quality, "balanced": balanced})
        if not cand:
            raise RuntimeError("No root-directed LAD alternative survived serial source-resolution QC")
        # Multi-objective shortlist: balanced, quality-first, root-first, then distinct backups.
        ordered = []
        for key, rev in (("balanced", True), ("quality", True), ("endpoint_aorta_distance_mm", False), ("aorta_gain_mm", True)):
            for x in sorted(cand, key=lambda z: z[key], reverse=rev):
                if x not in ordered:
                    ordered.append(x)
                    break
        for x in sorted(cand, key=lambda z: z["balanced"], reverse=True):
            if x not in ordered:
                ordered.append(x)
        selected = []
        for x in ordered:
            ep = x["path"][0]
            tan = _unit((x["path"][min(len(x["path"]) - 1, 8)] - x["path"][0]) * self.spacing)
            if all(np.linalg.norm((ep - y["path"][0]) * self.spacing) >= 1.6 or _angle_deg(tan, _unit((y["path"][min(len(y["path"]) - 1, 8)] - y["path"][0]) * self.spacing)) >= 18 for y in selected):
                selected.append(x)
            if len(selected) >= int(n_alternatives):
                break
        if not selected:
            selected = [max(cand, key=lambda z: z["balanced"])]
        rows, alts = [], []
        for i, x in enumerate(selected, 1):
            fp = self.cache / f"alternative_{i:02d}_centerline.csv"
            qf = self.cache / f"alternative_{i:02d}_qc.csv"
            pd.DataFrame({"arc_mm": arc_mm(x["path"], self.spacing), "z": x["path"][:, 0], "y": x["path"][:, 1], "x": x["path"][:, 2]}).to_csv(fp, index=False)
            x["qc"].to_csv(qf, index=False)
            row = {"alternative_id": i, "length_mm": x["qsum"]["length_mm"], "plane_pass_fraction": x["qsum"]["plane_pass_fraction"], "soft_pass_fraction": x["qsum"]["soft_pass_fraction"], "median_plane_score": x["qsum"]["median_plane_score"], "median_radius_mm": x["qsum"]["median_radius_mm"], "median_recenter_shift_mm": x["qsum"]["median_recenter_shift_mm"], "endpoint_aorta_distance_mm": x["endpoint_aorta_distance_mm"], "aorta_distance_reduction_mm": x["aorta_gain_mm"], "quality_score": x["quality"], "balanced_score": x["balanced"]}
            rows.append(row)
            alts.append({"id": i, "path": x["path"], "qc": x["qc"], "summary": row})
        summary = pd.DataFrame(rows)
        summary.to_csv(sf, index=False)
        self.alternatives, self.alternative_summary = alts, summary
        self._record("alternatives", "recomputed_and_cached", sf, f"selected {len(alts)} distinct root-directed alternatives")
        return summary

    def _probe_one_direction(self, point, direction, nsteps=4):
        cur = np.asarray(point, float)
        d = _unit(direction)
        scores = []
        hard = []
        pts = [cur.copy()]
        for _ in range(int(nsteps)):
            props = self._propose(cur, d, near_root=True)
            if not props:
                break
            r = max(props, key=lambda x: x["plane_score"])
            cur = r["point"]
            d = r["direction"]
            pts.append(cur.copy())
            scores.append(r["plane_score"])
            hard.append(bool(r["hard_pass"]))
        if len(pts) < 3:
            return None
        return {"path": np.asarray(pts, float), "length_mm": float(arc_mm(np.asarray(pts), self.spacing)[-1]), "mean_score": float(np.mean(scores)), "hard_fraction": float(np.mean(hard)), "direction": _unit((pts[-1] - pts[0]) * self.spacing)}

    def probe_takeoff(self):
        sf = self.cache / "branch_probe_summary.csv"
        tf = self.cache / "takeoff_candidate.json"
        if not self.alternatives:
            self.search_alternatives()
        if self.reuse["branch_probe"] and sf.exists() and tf.exists():
            self.branch_probe = pd.read_csv(sf)
            self.takeoff_candidate = _json_read(tf)
            self._record("branch_probe", "reused", sf)
            return self.takeoff_candidate
        rows = []
        best = None
        for alt in self.alternatives:
            p = resample_path(alt["path"], self.spacing, 0.35)  # proximal -> prior anchor
            s = arc_mm(p, self.spacing)
            max_probe = min(12.0, float(s[-1]))
            for x in np.arange(0.8, max_probe + 1e-6, 1.6):
                i = int(np.argmin(abs(s - x)))
                i0, i1 = max(0, i - 5), min(len(p) - 1, i + 5)
                lad_tan = _unit((p[i1] - p[i0]) * self.spacing)
                # distal LAD direction is toward increasing arc in p.
                distal_dir = lad_tan
                u, v = _orth_basis(distal_dir)
                dirs = []
                for polar in (45, 65, 90, 115, 135, 155):
                    a = math.radians(polar)
                    for k in range(12):
                        ph = 2 * math.pi * k / 12
                        d = _unit(math.cos(a) * distal_dir + math.sin(a) * (math.cos(ph) * u + math.sin(ph) * v))
                        if _angle_deg(d, distal_dir) >= 35:
                            dirs.append(d)
                probes = []
                for d in dirs:
                    z = self._probe_one_direction(p[i], d, 4)
                    if z is not None and z["length_mm"] >= 1.8 and z["mean_score"] >= 0.68 and z["hard_fraction"] >= 0.50:
                        probes.append(z)
                clusters = []
                for z in sorted(probes, key=lambda q: q["mean_score"], reverse=True):
                    if all(_angle_deg(z["direction"], q["direction"]) >= 32 for q in clusters):
                        clusters.append(z)
                ad, _ = self._aorta_distance(p[i])
                multiplicity = len(clusters)
                best_secondary = clusters[0]["mean_score"] if clusters else 0.0
                evidence = min(multiplicity, 3) / 3.0 * 0.45 + best_secondary * 0.35 + 0.20 * math.exp(-0.5 * ((ad - 10.0) / 7.0) ** 2) if np.isfinite(ad) else min(multiplicity, 3) / 3.0 * 0.45 + best_secondary * 0.35
                row = {"alternative_id": alt["id"], "arc_from_proximal_mm": float(s[i]), "aorta_distance_mm": float(ad), "viable_non_lad_directions": multiplicity, "best_secondary_mean_score": best_secondary, "branch_evidence_score": float(evidence), "z": p[i, 0], "y": p[i, 1], "x": p[i, 2]}
                rows.append(row)
                if best is None or row["branch_evidence_score"] > best["branch_evidence_score"]:
                    best = row.copy()
        df = pd.DataFrame(rows)
        df.to_csv(sf, index=False)
        if best is None:
            candidate = {"status": "NOT_IDENTIFIED", "note": "No point had a usable local branch probe."}
        else:
            strong = best["viable_non_lad_directions"] >= 2 and best["best_secondary_mean_score"] >= 0.72
            lps = source_zyx_to_lps(self.meta, np.array([[best["z"], best["y"], best["x"]]], float))[0]
            candidate = {
                "status": "CANDIDATE" if strong else "UNVALIDATED",
                "alternative_id": int(best["alternative_id"]),
                "arc_from_proximal_mm": float(best["arc_from_proximal_mm"]),
                "source_zyx": [float(best["z"]), float(best["y"]), float(best["x"])],
                "lps_mm": [float(x) for x in lps],
                "distance_to_TotalSegmentator_aorta_mm": float(best["aorta_distance_mm"]),
                "viable_non_lad_directions": int(best["viable_non_lad_directions"]),
                "best_secondary_mean_score": float(best["best_secondary_mean_score"]),
                "branch_evidence_score": float(best["branch_evidence_score"]),
                "note": "Local short-branch evidence only. This is not a manually labeled bifurcation and no LCX centerline was traced.",
            }
        _json_write(candidate, tf)
        self.branch_probe, self.takeoff_candidate = df, candidate
        self._record("branch_probe", "recomputed_and_cached", sf, candidate.get("status", ""))
        return candidate

    def _plot_sections(self, path, axs, label):
        p = resample_path(path, self.spacing, 0.35)
        s = arc_mm(p, self.spacing)
        for ax, x in zip(axs, np.linspace(0.35, max(0.35, s[-1] - 0.25), len(axs))):
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
            im, c = orthogonal_plane(self.ct, p[i], (p[i1] - p[i0]) * self.spacing, self.spacing)
            ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower")
            ax.plot(0, 0, "+", ms=8)
            ax.set_aspect("equal")
            ax.set_title(f"{label}\n{x:.1f} mm")

    def plot_qc(self):
        done = self.cache / "figures.done"
        figs = [self.out / f for f in (
            "01_prior_lad_endpoint_revalidation.png",
            "02_root_alternatives_source_mips.png",
            "03_balanced_alternative_cross_sections.png",
            "04_alternative_tradeoff.png",
            "05_branch_probe_profile.png",
            "06_takeoff_candidate_neighborhood.png",
            "07_combined_lad_lps_course.png",
        )]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs):
            self._record("figures", "reused", done)
            return figs
        if not self.alternatives:
            self.search_alternatives()
        if self.takeoff_candidate is None:
            self.probe_takeoff()
        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        prox = self.lad[: max(8, min(len(self.lad), int(round(14 / 0.35))))]
        self._plot_sections(prox, axs.ravel(), "Prior proximal LAD")
        fig.suptitle("Previously validated LAD proximal endpoint revalidation")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[0], dpi=180, bbox_inches="tight"); plt.close(fig)

        pts = np.vstack([a["path"] for a in self.alternatives] + [self.lad[:1]])
        pad = np.ceil(10 / self.spacing).astype(int)
        lo = np.maximum(0, np.floor(pts.min(0)).astype(int) - pad)
        hi = np.minimum(np.asarray(self.ct.shape), np.ceil(pts.max(0)).astype(int) + pad + 1)
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        crop = np.asarray(self.ct[sl], dtype=np.int16)
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        for ax, im, title in zip(axs, [crop.max(0), crop.max(1), crop.max(2)], ("Axial MIP", "Coronal MIP", "Sagittal MIP")):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower"); ax.set_title(title); ax.axis("off")
        for a in self.alternatives:
            q = a["path"] - lo
            axs[0].plot(q[:, 2], q[:, 1], lw=1.8, label=f"alt {a['id']}")
            axs[1].plot(q[:, 2], q[:, 0], lw=1.8)
            axs[2].plot(q[:, 1], q[:, 0], lw=1.8)
        axs[0].legend(fontsize=8)
        fig.suptitle("Distinct root-directed proximal LAD alternatives")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[1], dpi=180, bbox_inches="tight"); plt.close(fig); del crop

        best = max(self.alternatives, key=lambda a: float(a["summary"]["balanced_score"]))
        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        self._plot_sections(best["path"], axs.ravel(), f"Balanced alt {best['id']}")
        fig.suptitle(f"Balanced root-directed alternative — {best['summary']['length_mm']:.1f} mm, aorta {best['summary']['endpoint_aorta_distance_mm']:.1f} mm")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[2], dpi=180, bbox_inches="tight"); plt.close(fig)

        s = self.alternative_summary
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(s.endpoint_aorta_distance_mm, s.quality_score, s=80 + 8 * s.length_mm)
        for _, r in s.iterrows():
            ax.annotate(f"alt {int(r.alternative_id)}", (r.endpoint_aorta_distance_mm, r.quality_score), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("endpoint distance to TotalSegmentator aorta (mm)")
        ax.set_ylabel("serial lumen quality score")
        ax.set_title("Root progress vs source-resolution lumen quality")
        fig.tight_layout(); fig.savefig(figs[3], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        if self.branch_probe is not None and len(self.branch_probe):
            for aid, g in self.branch_probe.groupby("alternative_id"):
                ax.plot(g.arc_from_proximal_mm, g.branch_evidence_score, marker=".", label=f"alt {int(aid)}")
        ax.set_xlabel("arc from alternative proximal endpoint (mm)")
        ax.set_ylabel("local branch-evidence score")
        ax.set_title("Short local branch probe; no LCX centerline")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs[4], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, axs = plt.subplots(2, 4, figsize=(11, 5.5))
        if self.takeoff_candidate and self.takeoff_candidate.get("status") in ("CANDIDATE", "UNVALIDATED"):
            aid = int(self.takeoff_candidate["alternative_id"])
            alt = next(a for a in self.alternatives if a["id"] == aid)
            p = resample_path(alt["path"], self.spacing, 0.35)
            s0 = arc_mm(p, self.spacing)
            x0 = float(self.takeoff_candidate["arc_from_proximal_mm"])
            vals = np.linspace(max(0, x0 - 4), min(s0[-1], x0 + 4), 8)
            for ax, x in zip(axs.ravel(), vals):
                i = int(np.argmin(abs(s0 - x))); i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
                im, c = orthogonal_plane(self.ct, p[i], (p[i1] - p[i0]) * self.spacing, self.spacing)
                ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower"); ax.plot(0, 0, "+", ms=8); ax.set_aspect("equal"); ax.set_title(f"candidate {x:.1f} mm")
        else:
            for ax in axs.ravel(): ax.axis("off")
        fig.suptitle(f"Takeoff/bifurcation neighborhood — {self.takeoff_candidate.get('status','NOT_IDENTIFIED') if self.takeoff_candidate else 'NOT_IDENTIFIED'}")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[5], dpi=180, bbox_inches="tight"); plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
        best = max(self.alternatives, key=lambda a: float(a["summary"]["balanced_score"]))
        combined = resample_path(np.vstack([best["path"][:-1], self.lad]), self.spacing, 0.35)
        l = source_zyx_to_lps(self.meta, combined)
        for ax, (aa, bb, title) in zip(axs, [(0, 2, "L-R vs S-I"), (1, 2, "P-A vs S-I"), (0, 1, "L-R vs P-A")]):
            ax.plot(l[:, aa], l[:, bb], ".-"); ax.plot(l[0, aa], l[0, bb], "x", ms=9, label="current proximal endpoint"); ax.set_title(title); ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8)
        fig.suptitle("Balanced alternative + validated LAD backbone in patient LPS coordinates")
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(figs[6], dpi=180, bbox_inches="tight"); plt.close(fig)
        done.write_text(ALGORITHM_VERSION)
        self._record("figures", "recomputed_and_cached", done)
        return figs

    def package(self):
        done = self.cache / "report.done"
        zpath = self.out / "OPENPLAQUE_LAD_TAKEOFF_ROOT_ALTERNATIVES_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists():
            self._record("report", "reused", zpath)
            return zpath
        figs = self.plot_qc()
        html = self.out / "OPENPLAQUE_LAD_TAKEOFF_ROOT_ALTERNATIVES_REPORT.html"
        def img(fp):
            return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(Path(fp).read_bytes()).decode()}'>"
        html.write_text(
            "<html><body><h1>OpenPlaque — LAD takeoff root-directed alternatives</h1>"
            "<p><b>Research use only.</b> Starts only from the validated proximal LAD. Several distinct root-directed alternatives are compared by true source-resolution lumen morphology and distance to the TotalSegmentator aorta. The short branch probe is used only as local bifurcation evidence; no LCX centerline is traced.</p>"
            + "".join(img(f) for f in figs)
            + "<h2>Validated LAD backbone</h2><pre>" + json.dumps(self.lad_summary, indent=2) + "</pre>"
            + "<h2>Alternative summary</h2>" + self.alternative_summary.to_html(index=False)
            + "<h2>Takeoff candidate</h2><pre>" + json.dumps(self.takeoff_candidate, indent=2) + "</pre></body></html>", encoding="utf-8")
        names = [f.name for f in figs] + ["cache_provenance.csv", html.name]
        for n in ("validated_rca_calibration.json", "validated_lad_centerline.csv", "validated_lad_qc.csv", "validated_lad_summary.json", "alternative_summary.csv", "branch_probe_summary.csv", "takeoff_candidate.json"):
            p = self.cache / n
            if p.exists():
                shutil.copyfile(p, self.out / n); names.append(n)
        for a in self.alternatives:
            for suffix in ("centerline.csv", "qc.csv"):
                n = f"alternative_{a['id']:02d}_{suffix}"
                p = self.cache / n
                if p.exists():
                    shutil.copyfile(p, self.out / n); names.append(n)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n in dict.fromkeys(names):
                p = self.out / n
                if p.exists():
                    z.write(p, arcname=n)
        done.write_text(ALGORITHM_VERSION)
        self._record("report", "recomputed_and_cached", zpath)
        return zpath
