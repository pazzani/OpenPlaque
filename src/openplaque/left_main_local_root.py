from __future__ import annotations

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

ALGORITHM_VERSION = "left-main-local-root-v1.0"
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
    if len(p) < 2:
        return np.array([0.0])
    d = np.diff(p, axis=0) * np.asarray(spacing_zyx, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def resample_path(path_zyx, spacing_zyx, step_mm=0.5):
    p = np.asarray(path_zyx, float)
    s = arc_mm(p, spacing_zyx)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 2 or s[-1] < step_mm:
        return p
    su = np.arange(0.0, s[-1] + 1e-6, step_mm)
    return np.column_stack([np.interp(su, s, p[:, j]) for j in range(3)])


def _series7_files(root, extract_root="/content/full_dicom_local_root"):
    root = Path(root)
    dz = root / "Full_DICOM.zip"
    if not dz.exists():
        raise FileNotFoundError(dz)
    local_zip = Path("/content/Full_DICOM.zip")
    if not local_zip.exists() or local_zip.stat().st_size != dz.stat().st_size:
        shutil.copyfile(dz, local_zip)
    study = OpenPlaqueStudy(str(local_zip), extract_root=extract_root)
    match = [s for s in study.series if s["series_number"] == SOURCE_SERIES]
    if not match:
        raise RuntimeError("Series 7 not found")
    folder = match[0]["folder"]
    uid = match[0]["uid"]
    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(folder, uid))
    if not files:
        raise RuntimeError("No DICOM files found for series 7")
    return files


def stream_source_ct_to_memmap(root, out_path, persistent_cache=None, prior_cache=None, reuse=True):
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    persistent_cache = Path(persistent_cache) if persistent_cache else None
    persistent_meta = persistent_cache.with_suffix(".json") if persistent_cache else None
    prior_cache = Path(prior_cache) if prior_cache else None
    prior_meta = prior_cache.with_suffix(".json") if prior_cache else None

    candidates = []
    if reuse and persistent_cache is not None:
        candidates.append((persistent_cache, persistent_meta, "reused_local_root_cache"))
    if reuse and prior_cache is not None:
        candidates.append((prior_cache, prior_meta, "imported_prior_lowram_cache"))
    for src, smeta, action in candidates:
        if src.exists() and smeta.exists():
            if not out_path.exists() or out_path.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, out_path)
            shutil.copyfile(smeta, meta_path)
            return np.load(out_path, mmap_mode="r"), _json_read(meta_path), action

    if reuse and out_path.exists() and meta_path.exists():
        return np.load(out_path, mmap_mode="r"), _json_read(meta_path), "reused_runtime_memmap"

    files = _series7_files(root)
    ds0 = pydicom.dcmread(files[0], force=True)
    rows, cols, n = int(ds0.Rows), int(ds0.Columns), len(files)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16, shape=(n, rows, cols))
    positions = []
    for i, fp in enumerate(files):
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        mm[i] = np.rint(np.clip(hu, -32768, 32767)).astype(np.int16)
        positions.append([float(x) for x in getattr(ds, "ImagePositionPatient", (0, 0, i))])
        if i % 100 == 0:
            mm.flush()
    mm.flush()
    ps = [float(x) for x in ds0.PixelSpacing]
    pos = np.asarray(positions, float)
    dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) >= 2 else float(getattr(ds0, "SliceThickness", 1.0))
    meta = {
        "shape": [n, rows, cols],
        "spacing_zyx": [dz, ps[0], ps[1]],
        "image_orientation_patient": [float(x) for x in getattr(ds0, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0))],
        "positions_lps_mm": positions,
    }
    _json_write(meta, meta_path)
    del mm
    gc.collect()
    if persistent_cache is not None:
        persistent_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, persistent_cache)
        shutil.copyfile(meta_path, persistent_meta)
    return np.load(out_path, mmap_mode="r"), meta, "recomputed_and_cached"


def _orthogonal_basis(tangent_zyx_mm):
    t = np.asarray(tangent_zyx_mm, float)
    t /= max(np.linalg.norm(t), 1e-8)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u /= max(np.linalg.norm(u), 1e-8)
    v = np.cross(t, u)
    v /= max(np.linalg.norm(v), 1e-8)
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=5.5, pix_mm=0.18):
    u, v = _orthogonal_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, [vox[..., 0], vox[..., 1], vox[..., 2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def plane_lumen_metrics(im, coords_mm):
    im = np.asarray(im, float)
    c = np.asarray(coords_mm, float)
    h, w = im.shape
    yy, xx = np.mgrid[:h, :w]
    Y, X = c[yy], c[xx]
    R = np.sqrt(X * X + Y * Y)
    center_hu = float(np.median(im[R <= 0.7]))
    threshold = float(np.clip(0.55 * center_hu, 180, 520))
    mask = (im >= threshold) & (im <= 1200) & (R <= 4.5)
    lab, _ = ndi.label(mask)
    central = lab[R <= 0.50]
    central = central[central > 0]
    if central.size == 0:
        return {
            "center_hu": center_hu, "threshold_hu": threshold, "radius_mm": np.nan,
            "centroid_offset_mm": np.inf, "circularity": 0.0, "core_minus_ring_hu": np.nan,
            "component_area_mm2": 0.0,
        }
    vals, counts = np.unique(central, return_counts=True)
    k = int(vals[np.argmax(counts)])
    comp = lab == k
    pix = float(abs(c[1] - c[0])) if len(c) > 1 else 0.18
    area = float(comp.sum() * pix * pix)
    radius = math.sqrt(area / math.pi)
    weights = comp.astype(float)
    cy = float((Y * weights).sum() / max(weights.sum(), 1))
    cx = float((X * weights).sum() / max(weights.sum(), 1))
    offset = float(math.hypot(cx, cy))
    edge = comp & ~ndi.binary_erosion(comp)
    perimeter = float(edge.sum() * pix)
    circularity = float(np.clip(4 * math.pi * area / max(perimeter * perimeter, 1e-6), 0, 1.2))
    core = float(np.mean(im[R <= 0.9]))
    ring_mask = (R >= 2.8) & (R <= 4.6)
    ring = float(np.mean(im[ring_mask])) if np.any(ring_mask) else core
    return {
        "center_hu": center_hu, "threshold_hu": threshold, "radius_mm": radius,
        "centroid_offset_mm": offset, "circularity": circularity,
        "core_minus_ring_hu": core - ring, "component_area_mm2": area,
    }


def _derive_rca_calibration(raw_df):
    good = raw_df[np.isfinite(raw_df["radius_mm"])].copy()
    if len(good) < 6:
        raise RuntimeError("Frozen RCA calibration produced too few finite lumen sections")
    return {
        "median_radius_mm": float(good["radius_mm"].median()),
        "median_center_hu": float(good["center_hu"].median()),
        "median_offset_mm": float(good["centroid_offset_mm"].median()),
        "median_circularity": float(good["circularity"].median()),
        "median_core_minus_ring_hu": float(good["core_minus_ring_hu"].median()),
    }


def _score_plane(m, rca):
    r = float(m["radius_mm"]) if np.isfinite(m["radius_mm"]) else np.nan
    rr = float(rca["median_radius_mm"])
    rh = float(rca["median_center_hu"])
    radius_target = 1.20 * rr
    radius_score = float(np.exp(-0.5 * ((r / max(radius_target, 1e-6) - 1) / 0.45) ** 2)) if np.isfinite(r) else 0.0
    offset_score = float(np.exp(-0.5 * (m["centroid_offset_mm"] / 0.65) ** 2)) if np.isfinite(m["centroid_offset_mm"]) else 0.0
    circ_score = float(np.clip(m["circularity"] / 0.60, 0, 1))
    contrast_score = float(1 / (1 + np.exp(-(m["core_minus_ring_hu"] - 25) / 80))) if np.isfinite(m["core_minus_ring_hu"]) else 0.0
    hu_score = float(np.exp(-0.5 * ((m["center_hu"] - rh) / 300) ** 2))
    score = 0.32 * radius_score + 0.27 * offset_score + 0.18 * circ_score + 0.14 * contrast_score + 0.09 * hu_score
    pass_plane = (
        np.isfinite(r)
        and 0.55 * rr <= r <= min(3.5, 2.0 * rr + 0.15)
        and m["centroid_offset_mm"] <= 1.10
        and m["circularity"] >= 0.28
        and 150 <= m["center_hu"] <= 1100
        and np.isfinite(m["core_minus_ring_hu"])
        and m["core_minus_ring_hu"] >= -20
    )
    return float(score), bool(pass_plane)


def serial_lumen_qc(path_source_zyx, ct, spacing_zyx, rca, n_samples=10, label_name="candidate"):
    p = resample_path(path_source_zyx, spacing_zyx, 0.45)
    s = arc_mm(p, spacing_zyx)
    if len(p) < 5 or s[-1] < 2:
        return pd.DataFrame(), {
            "label": label_name, "length_mm": float(s[-1]) if len(s) else 0.0,
            "median_plane_score": 0.0, "plane_pass_fraction": 0.0,
        }
    ss = np.linspace(min(0.8, 0.08 * s[-1]), max(min(0.8, 0.08 * s[-1]), s[-1] - 0.8), n_samples)
    rows = []
    for x in ss:
        i = int(np.argmin(np.abs(s - x)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing_zyx, float)
        if np.linalg.norm(t) < 1e-6:
            continue
        im, c = orthogonal_plane(ct, p[i], t, spacing_zyx)
        m = plane_lumen_metrics(im, c)
        score, passed = _score_plane(m, rca)
        rows.append({"label": label_name, "arc_mm": float(s[i]), **m, "plane_score": score, "plane_pass": passed})
    df = pd.DataFrame(rows)
    return df, {
        "label": label_name,
        "length_mm": float(s[-1]),
        "median_plane_score": float(df["plane_score"].median()) if len(df) else 0.0,
        "plane_pass_fraction": float(df["plane_pass"].mean()) if len(df) else 0.0,
        "median_radius_mm": float(df["radius_mm"].median()) if len(df) else np.nan,
        "median_offset_mm": float(df["centroid_offset_mm"].median()) if len(df) else np.nan,
        "median_circularity": float(df["circularity"].median()) if len(df) else np.nan,
        "median_center_hu": float(df["center_hu"].median()) if len(df) else np.nan,
        "median_core_minus_ring_hu": float(df["core_minus_ring_hu"].median()) if len(df) else np.nan,
    }


def calibrate_rca(rca_path, ct, spacing_zyx, n_samples=12):
    p = resample_path(rca_path, spacing_zyx, 0.45)
    s = arc_mm(p, spacing_zyx)
    sample_s = np.linspace(min(0.8, s[-1] * 0.04), max(min(0.8, s[-1] * 0.04), s[-1] - 0.8), n_samples)
    rows = []
    for x in sample_s:
        i = int(np.argmin(np.abs(s - x)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing_zyx, float)
        im, c = orthogonal_plane(ct, p[i], t, spacing_zyx)
        rows.append({"label": "RCA_reference", "arc_mm": float(s[i]), **plane_lumen_metrics(im, c)})
    raw = pd.DataFrame(rows)
    rca = _derive_rca_calibration(raw)
    scores, passes = [], []
    for _, row in raw.iterrows():
        score, passed = _score_plane(row, rca)
        scores.append(score)
        passes.append(passed)
    raw["plane_score"] = scores
    raw["plane_pass"] = passes
    rca["median_plane_score"] = float(raw["plane_score"].median())
    rca["plane_pass_fraction"] = float(raw["plane_pass"].mean())
    return raw, rca


def _aorta_center_xy(aorta_crop, z_local):
    z0 = int(np.clip(round(z_local), 0, aorta_crop.shape[0] - 1))
    for dz in range(0, 8):
        for z in {z0 - dz, z0 + dz}:
            if 0 <= z < aorta_crop.shape[0]:
                yy, xx = np.where(aorta_crop[z])
                if len(yy) >= 20:
                    return np.array([float(np.mean(yy)), float(np.mean(xx))])
    raise RuntimeError("Unable to estimate aortic center in root crop")


def _cone_directions(base_dir, angles_deg=(0, 12, 24, 32), azimuths=8):
    b = np.asarray(base_dir, float)
    b /= max(np.linalg.norm(b), 1e-8)
    u, v = _orthogonal_basis(b)
    dirs = [b]
    for a in angles_deg:
        if a == 0:
            continue
        th = math.radians(a)
        for k in range(azimuths):
            ph = 2 * math.pi * k / azimuths
            d = math.cos(th) * b + math.sin(th) * (math.cos(ph) * u + math.sin(ph) * v)
            d /= max(np.linalg.norm(d), 1e-8)
            dirs.append(d)
    return dirs


def _sample_ray_hu(ct, seed, direction_mm, spacing_zyx, distances_mm):
    sp = np.asarray(spacing_zyx, float)
    d = np.asarray(direction_mm, float)
    pts = np.asarray(seed, float)[None, :] + np.asarray(distances_mm)[:, None] * d[None, :] / sp[None, :]
    vals = map_coordinates(ct, [pts[:, 0], pts[:, 1], pts[:, 2]], order=1, mode="nearest", prefilter=False)
    return pts, np.asarray(vals, float)


def _dedupe_rows(rows, spacing_zyx, max_n=16, min_seed_mm=2.0, min_end_mm=4.0):
    sp = np.asarray(spacing_zyx, float)
    kept = []
    for r in sorted(rows, key=lambda q: q["initial_score"], reverse=True):
        seed = np.asarray(r["seed_zyx"], float)
        end = np.asarray(r["path_zyx"][-1], float)
        ok = True
        for k in kept:
            ds = np.linalg.norm((seed - np.asarray(k["seed_zyx"], float)) * sp)
            de = np.linalg.norm((end - np.asarray(k["path_zyx"][-1], float)) * sp)
            if ds < min_seed_mm and de < min_end_mm:
                ok = False
                break
        if ok:
            kept.append(r)
        if len(kept) >= max_n:
            break
    return kept


def detect_local_ostium_rays(ct, aorta_crop, crop_lo, spacing_zyx, rca_seed_source, rca_calibration, topn=16):
    sp = np.asarray(spacing_zyx, float)
    crop_lo = np.asarray(crop_lo, int)
    rca_local = np.asarray(rca_seed_source, float) - crop_lo
    dist = ndi.distance_transform_edt(~aorta_crop, sampling=sp).astype(np.float32)
    ct_crop = np.asarray(ct[
        crop_lo[0]:crop_lo[0] + aorta_crop.shape[0],
        crop_lo[1]:crop_lo[1] + aorta_crop.shape[1],
        crop_lo[2]:crop_lo[2] + aorta_crop.shape[2]
    ], dtype=np.int16)

    z_mm = np.abs((np.arange(aorta_crop.shape[0], dtype=np.float32) - rca_local[0]) * sp[0])
    shell = ((~aorta_crop) & (dist >= 0.6) & (dist <= 3.4) &
             (ct_crop >= 180) & (ct_crop <= 950) & (z_mm[:, None, None] <= 10.0))
    coords = np.argwhere(shell)
    del shell
    if len(coords) == 0:
        raise RuntimeError("No local aortic-shell points for left ostium search")

    rca_ctr = _aorta_center_xy(aorta_crop, rca_local[0])
    rca_rad = np.array([0.0, (rca_local[1] - rca_ctr[0]) * sp[1], (rca_local[2] - rca_ctr[1]) * sp[2]], float)
    rca_rad /= max(np.linalg.norm(rca_rad), 1e-8)

    centers = {}
    for z in np.unique(coords[:, 0]):
        centers[int(z)] = _aorta_center_xy(aorta_crop, int(z))
    cxy = np.vstack([centers[int(z)] for z in coords[:, 0]])
    rad = np.column_stack([
        np.zeros(len(coords)),
        (coords[:, 1] - cxy[:, 0]) * sp[1],
        (coords[:, 2] - cxy[:, 1]) * sp[2],
    ]).astype(float)
    norm = np.linalg.norm(rad, axis=1)
    valid = norm >= 2.0
    rad[valid] /= norm[valid, None]
    opposite = -(rad @ rca_rad)
    valid &= opposite >= 0.20
    coords, rad, opposite = coords[valid], rad[valid], opposite[valid]
    if len(coords) == 0:
        raise RuntimeError("No opposite-side aortic-shell points survive local root screening")

    hu0 = ct_crop[tuple(coords.T)].astype(float)
    dq = dist[tuple(coords.T)].astype(float)
    quick = 0.50 * np.clip((hu0 - 180) / 500, 0, 1) + 0.35 * np.clip((opposite + 0.2) / 1.2, 0, 1) + 0.15 * np.exp(-0.5 * ((dq - 1.6) / 0.8) ** 2)
    order = np.argsort(quick)[::-1]
    seeds = []
    for j in order:
        p = coords[j]
        if all(np.linalg.norm((p - q[0]) * sp) >= 1.8 for q in seeds):
            seeds.append((p, rad[j], float(opposite[j]), float(quick[j])))
        if len(seeds) >= 48:
            break

    rows = []
    ray_dist = np.arange(0.8, 10.01, 0.7)
    probe_dist = (2.5, 5.5, 8.5)
    for seed_local, radial, opp, qscore in seeds:
        seed_source = seed_local.astype(float) + crop_lo
        direction_rows = []
        for d in _cone_directions(radial):
            pts, hu = _sample_ray_hu(ct, seed_source, d, sp, ray_dist)
            plausible = float(np.mean((hu >= 150) & (hu <= 1000)))
            median_hu = float(np.median(hu))
            p10_hu = float(np.percentile(hu, 10))
            hu_cont = 0.65 * plausible + 0.20 * np.clip((p10_hu - 120) / 350, 0, 1) + 0.15 * np.clip((850 - abs(median_hu - 550)) / 850, 0, 1)
            direction_rows.append((hu_cont, d, pts, median_hu, p10_hu, plausible))
        direction_rows.sort(key=lambda x: x[0], reverse=True)
        for hu_cont, d, pts, median_hu, p10_hu, plausible in direction_rows[:3]:
            pm = []
            for mm in probe_dist:
                point = seed_source + mm * d / sp
                im, c = orthogonal_plane(ct, point, d, sp)
                m = plane_lumen_metrics(im, c)
                score, passed = _score_plane(m, rca_calibration)
                pm.append((score, passed, m))
            plane_med = float(np.median([x[0] for x in pm]))
            plane_pass = float(np.mean([x[1] for x in pm]))
            radius_med = float(np.nanmedian([x[2]["radius_mm"] for x in pm]))
            offset_med = float(np.nanmedian([x[2]["centroid_offset_mm"] for x in pm]))
            initial = 0.24 * np.clip((opp + 1) / 2, 0, 1) + 0.12 * qscore + 0.22 * hu_cont + 0.26 * plane_med + 0.16 * plane_pass
            rows.append({
                "seed_zyx": seed_source,
                "direction_mm": np.asarray(d, float),
                "path_zyx": pts,
                "initial_score": float(initial),
                "opposite_rca": float(opp),
                "quick_shell_score": float(qscore),
                "ray_plausible_hu_fraction": plausible,
                "ray_median_hu": median_hu,
                "ray_p10_hu": p10_hu,
                "probe_median_plane_score": plane_med,
                "probe_plane_pass_fraction": plane_pass,
                "probe_median_radius_mm": radius_med,
                "probe_median_offset_mm": offset_med,
            })
    if not rows:
        raise RuntimeError("No local ostium ray hypotheses")
    return _dedupe_rows(rows, sp, max_n=topn)


def _direction_candidates(current_dir, angle_degs=(0, 8, 16), azimuths=8):
    return _cone_directions(current_dir, angles_deg=angle_degs, azimuths=azimuths)


def extend_local_lumen(seed_path, ct, spacing_zyx, rca_calibration, max_total_mm=22.0, step_mm=0.9):
    sp = np.asarray(spacing_zyx, float)
    p = resample_path(seed_path, sp, 0.7)
    if len(p) < 4:
        return p
    s = arc_mm(p, sp)
    keep = s <= min(7.0, s[-1])
    p = p[keep]
    fail_streak = 0
    while arc_mm(p, sp)[-1] < max_total_mm:
        current = p[-1]
        i0 = max(0, len(p) - 4)
        d = (p[-1] - p[i0]) * sp
        d /= max(np.linalg.norm(d), 1e-8)
        choices = []
        for cand_dir in _direction_candidates(d):
            nxt = current + step_mm * cand_dir / sp
            if np.any(nxt < 1) or np.any(nxt >= np.asarray(ct.shape) - 2):
                continue
            hu = float(map_coordinates(ct, nxt[:, None], order=1, mode="nearest", prefilter=False)[0])
            if hu < 100 or hu > 1200:
                continue
            im, c = orthogonal_plane(ct, nxt, cand_dir, sp)
            m = plane_lumen_metrics(im, c)
            plane_score, passed = _score_plane(m, rca_calibration)
            angle = math.degrees(math.acos(np.clip(np.dot(cand_dir, d), -1, 1)))
            hu_score = float(np.exp(-0.5 * ((hu - rca_calibration["median_center_hu"]) / 340) ** 2))
            score = plane_score + 0.12 * hu_score - 0.006 * angle
            choices.append((score, passed, nxt, cand_dir, m))
        if not choices:
            break
        choices.sort(key=lambda x: x[0], reverse=True)
        score, passed, nxt, cand_dir, m = choices[0]
        if score < 0.42:
            fail_streak += 1
        else:
            fail_streak = 0
        p = np.vstack([p, nxt])
        if fail_streak >= 2:
            p = p[:-2]
            break
    return resample_path(p, sp, 0.5)


def best_prefix(path, ct, spacing_zyx, rca_calibration):
    sp = np.asarray(spacing_zyx, float)
    p = resample_path(path, sp, 0.5)
    s = arc_mm(p, sp)
    lengths = np.arange(5.0, min(22.0, s[-1]) + 1e-6, 1.0)
    if len(lengths) == 0:
        qdf, qsum = serial_lumen_qc(p, ct, sp, rca_calibration, n_samples=8, label_name="LEFT_MAIN")
        return p, qdf, qsum, 0.0
    best = None
    for L in lengths:
        pp = p[s <= L + 1e-6]
        qdf, qsum = serial_lumen_qc(pp, ct, sp, rca_calibration, n_samples=9, label_name="LEFT_MAIN")
        length_quality = float(np.exp(-0.5 * ((L - 11.0) / 7.0) ** 2))
        score = 0.50 * qsum["median_plane_score"] + 0.42 * qsum["plane_pass_fraction"] + 0.08 * length_quality
        if best is None or score > best[0]:
            best = (score, pp, qdf, qsum)
    return best[1], best[2], best[3], float(best[0])


def _save_path_csv(path, spacing_zyx, out):
    p = resample_path(path, spacing_zyx, 0.5)
    s = arc_mm(p, spacing_zyx)
    pd.DataFrame({"arc_mm": s, "z": p[:, 0], "y": p[:, 1], "x": p[:, 2]}).to_csv(out, index=False)
    return p


class LeftMainLocalRootWorkflow:
    COMPONENTS = ("source_ct", "root_crop", "rca_calibration", "ostium_candidates", "left_main", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Left_Main_Local_Root_v1"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out = self.root / "Left_Main_Local_Root_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            unknown = set(reuse) - set(self.COMPONENTS)
            if unknown:
                raise ValueError(f"Unknown reuse controls: {sorted(unknown)}")
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.provenance = []
        self.ct = None
        self.ct_meta = None
        self.spacing = None
        self.rca = None
        self.root_ct = None
        self.root_aorta = None
        self.crop_lo = None
        self.rca_qc = None
        self.rca_cal = None
        self.rays = []
        self.best = None
        self.best_qc = None
        self.best_summary = None

    def _record(self, component, action, path, note=""):
        self.provenance.append({
            "component": component, "reuse_requested": self.reuse[component],
            "action": action, "path": str(path), "note": note,
        })
        pd.DataFrame(self.provenance).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        states = {
            "source_ct": (self.cache / "series7_int16.npy").exists() or (self.root / "Cache" / "Left_Main_Bifurcation_v1" / "series7_int16.npy").exists(),
            "root_crop": (self.cache / "root_ct.npy").exists() and (self.cache / "root_aorta.npy").exists() and (self.cache / "root_meta.json").exists(),
            "rca_calibration": (self.cache / "rca_calibration.json").exists() and (self.cache / "rca_serial_qc.csv").exists(),
            "ostium_candidates": (self.cache / "ostium_candidates.csv").exists() and (self.cache / "ostium_paths.npz").exists(),
            "left_main": (self.cache / "left_main_best_summary.json").exists() and (self.cache / "left_main_best_centerline.csv").exists(),
            "figures": all((self.out / n).exists() for n in (
                "01_root_localization.png", "02_ostium_ray_candidates.png",
                "03_left_main_vs_rca_cross_sections.png", "04_left_main_mips.png",
            )),
            "report": (self.out / "OPENPLAQUE_LEFT_MAIN_LOCAL_ROOT_REPORT_BACK.zip").exists(),
        }
        return pd.DataFrame([{
            "component": k, "reuse": self.reuse[k], "cache_available": states[k],
            "planned_action": "reuse" if self.reuse[k] and states[k] else "recompute_and_cache",
        } for k in self.COMPONENTS])

    def load_source_ct(self):
        if self.ct is not None:
            return self.ct
        prior = self.root / "Cache" / "Left_Main_Bifurcation_v1" / "series7_int16.npy"
        persistent = self.cache / "series7_int16.npy"
        self.ct, self.ct_meta, action = stream_source_ct_to_memmap(
            self.root, "/content/openplaque_local_root_series7_int16.npy",
            persistent_cache=persistent, prior_cache=prior, reuse=self.reuse["source_ct"],
        )
        self.spacing = np.asarray(self.ct_meta["spacing_zyx"], float)
        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        self.rca = pd.read_csv(rca_fp)[["z", "y", "x"]].to_numpy(float)
        self._record("source_ct", action, prior if action == "imported_prior_lowram_cache" else persistent)
        _ram("After source CT")
        return self.ct

    def build_root_crop(self):
        if self.root_ct is not None:
            return self.root_ct, self.root_aorta
        self.load_source_ct()
        ct_fp = self.cache / "root_ct.npy"
        ao_fp = self.cache / "root_aorta.npy"
        meta_fp = self.cache / "root_meta.json"
        if self.reuse["root_crop"] and ct_fp.exists() and ao_fp.exists() and meta_fp.exists():
            self.root_ct = np.load(ct_fp, mmap_mode="r")
            self.root_aorta = np.load(ao_fp, mmap_mode="r")
            meta = _json_read(meta_fp)
            self.crop_lo = np.asarray(meta["crop_lo_zyx"], int)
            self._record("root_crop", "reused", ct_fp)
            return self.root_ct, self.root_aorta

        half_mm = np.array([16.0, 38.0, 38.0])
        half = np.ceil(half_mm / self.spacing).astype(int)
        center = np.rint(self.rca[0]).astype(int)
        lo = np.maximum(0, center - half)
        hi = np.minimum(np.asarray(self.ct.shape), center + half + 1)
        aorta_fp = self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz"
        ai = sitk.ReadImage(str(aorta_fp))
        aorta = sitk.GetArrayFromImage(ai) > 0
        if tuple(aorta.shape) != tuple(self.ct.shape):
            raise RuntimeError(f"Aorta/source shape mismatch: {aorta.shape} vs {self.ct.shape}")
        root_ct = np.asarray(self.ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.int16)
        root_aorta = np.asarray(aorta[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.uint8)
        del aorta
        np.save(ct_fp, root_ct)
        np.save(ao_fp, root_aorta)
        _json_write({"crop_lo_zyx": lo.tolist(), "crop_hi_zyx": hi.tolist(), "spacing_zyx": self.spacing.tolist()}, meta_fp)
        del root_ct, root_aorta
        gc.collect()
        self.root_ct = np.load(ct_fp, mmap_mode="r")
        self.root_aorta = np.load(ao_fp, mmap_mode="r") > 0
        self.crop_lo = lo
        self._record("root_crop", "recomputed_and_cached", ct_fp)
        _ram("After root crop")
        return self.root_ct, self.root_aorta

    def calibrate_rca(self):
        self.load_source_ct()
        json_fp = self.cache / "rca_calibration.json"
        csv_fp = self.cache / "rca_serial_qc.csv"
        if self.reuse["rca_calibration"] and json_fp.exists() and csv_fp.exists():
            self.rca_cal = _json_read(json_fp)
            self.rca_qc = pd.read_csv(csv_fp)
            self._record("rca_calibration", "reused", json_fp)
            return self.rca_cal
        self.rca_qc, self.rca_cal = calibrate_rca(self.rca, self.ct, self.spacing, n_samples=12)
        self.rca_qc.to_csv(csv_fp, index=False)
        _json_write(self.rca_cal, json_fp)
        self._record("rca_calibration", "recomputed_and_cached", json_fp)
        return self.rca_cal

    def find_ostium_candidates(self):
        self.build_root_crop()
        self.calibrate_rca()
        table_fp = self.cache / "ostium_candidates.csv"
        paths_fp = self.cache / "ostium_paths.npz"
        if self.reuse["ostium_candidates"] and table_fp.exists() and paths_fp.exists():
            tab = pd.read_csv(table_fp)
            z = np.load(paths_fp, allow_pickle=False)
            self.rays = []
            for i, row in tab.iterrows():
                self.rays.append({
                    **row.to_dict(),
                    "seed_zyx": z[f"seed_{i}"],
                    "direction_mm": z[f"dir_{i}"],
                    "path_zyx": z[f"path_{i}"],
                })
            self._record("ostium_candidates", "reused", table_fp)
            return tab

        rays = detect_local_ostium_rays(
            self.ct, np.asarray(self.root_aorta, bool), self.crop_lo, self.spacing,
            self.rca[0], self.rca_cal, topn=16,
        )
        self.rays = rays
        rows, arrs = [], {}
        for i, r in enumerate(rays):
            rows.append({
                "rank": i + 1, "initial_score": r["initial_score"], "opposite_rca": r["opposite_rca"],
                "quick_shell_score": r["quick_shell_score"],
                "ray_plausible_hu_fraction": r["ray_plausible_hu_fraction"],
                "ray_median_hu": r["ray_median_hu"], "ray_p10_hu": r["ray_p10_hu"],
                "probe_median_plane_score": r["probe_median_plane_score"],
                "probe_plane_pass_fraction": r["probe_plane_pass_fraction"],
                "probe_median_radius_mm": r["probe_median_radius_mm"],
                "probe_median_offset_mm": r["probe_median_offset_mm"],
                "seed_z": r["seed_zyx"][0], "seed_y": r["seed_zyx"][1], "seed_x": r["seed_zyx"][2],
                "dir_z_mm": r["direction_mm"][0], "dir_y_mm": r["direction_mm"][1], "dir_x_mm": r["direction_mm"][2],
            })
            arrs[f"seed_{i}"] = np.asarray(r["seed_zyx"], np.float32)
            arrs[f"dir_{i}"] = np.asarray(r["direction_mm"], np.float32)
            arrs[f"path_{i}"] = np.asarray(r["path_zyx"], np.float32)
        tab = pd.DataFrame(rows)
        tab.to_csv(table_fp, index=False)
        np.savez_compressed(paths_fp, **arrs)
        self._record("ostium_candidates", "recomputed_and_cached", table_fp)
        _ram("After local ostium rays")
        return tab

    def build_left_main(self):
        self.find_ostium_candidates()
        summary_fp = self.cache / "left_main_best_summary.json"
        path_fp = self.cache / "left_main_best_centerline.csv"
        serial_fp = self.cache / "left_main_best_serial_qc.csv"
        cand_fp = self.cache / "left_main_candidate_qc.csv"
        if self.reuse["left_main"] and summary_fp.exists() and path_fp.exists() and serial_fp.exists():
            self.best_summary = _json_read(summary_fp)
            self.best = pd.read_csv(path_fp)[["z", "y", "x"]].to_numpy(float)
            self.best_qc = pd.read_csv(serial_fp)
            self._record("left_main", "reused", summary_fp)
            return self.best

        candidates = []
        rows = []
        for i, r in enumerate(self.rays[:10], 1):
            viable = (float(r["probe_plane_pass_fraction"]) >= 0.50 and
                      float(r["probe_median_plane_score"]) >= 0.48 and
                      float(r["probe_median_radius_mm"]) <= 3.6)
            if viable:
                ext = extend_local_lumen(r["path_zyx"], self.ct, self.spacing, self.rca_cal, max_total_mm=22.0)
            else:
                ext = resample_path(r["path_zyx"], self.spacing, 0.5)
            prefix, qdf, qsum, prefix_score = best_prefix(ext, self.ct, self.spacing, self.rca_cal)
            combined = 0.28 * float(r["initial_score"]) + 0.42 * qsum["median_plane_score"] + 0.26 * qsum["plane_pass_fraction"] + 0.04 * prefix_score
            status = "OK" if (
                viable and 5 <= qsum["length_mm"] <= 22 and qsum["plane_pass_fraction"] >= 0.75
                and qsum["median_plane_score"] >= 0.60 and qsum["median_radius_mm"] <= 3.3
            ) else ("REVIEW" if qsum["plane_pass_fraction"] >= 0.55 and qsum["median_plane_score"] >= 0.52 else "FAIL")
            rec = {
                "candidate": i, "viable_ostium_gate": bool(viable), "combined_score": float(combined),
                "initial_score": float(r["initial_score"]), "status": status, **{k: v for k, v in qsum.items() if k != "label"},
            }
            rows.append(rec)
            candidates.append((combined, status, prefix, qdf, qsum, r))
        candidates.sort(key=lambda x: x[0], reverse=True)
        combined, status, prefix, qdf, qsum, ray = candidates[0]
        self.best = prefix
        self.best_qc = qdf
        self.best_summary = dict(qsum)
        self.best_summary.update({
            "combined_score": float(combined), "status": status,
            "ostium_initial_score": float(ray["initial_score"]),
            "ostium_probe_pass_fraction": float(ray["probe_plane_pass_fraction"]),
            "ostium_probe_median_radius_mm": float(ray["probe_median_radius_mm"]),
            "algorithm": ALGORITHM_VERSION,
        })
        pd.DataFrame(rows).sort_values("combined_score", ascending=False).to_csv(cand_fp, index=False)
        _save_path_csv(self.best, self.spacing, path_fp)
        self.best_qc.to_csv(serial_fp, index=False)
        _json_write(self.best_summary, summary_fp)
        self._record("left_main", "recomputed_and_cached", summary_fp)
        _ram("After left-main local tracking")
        return self.best

    def plot_qc(self):
        self.build_left_main()
        names = [
            "01_root_localization.png", "02_ostium_ray_candidates.png",
            "03_left_main_vs_rca_cross_sections.png", "04_left_main_mips.png",
        ]
        if self.reuse["figures"] and all((self.out / n).exists() for n in names):
            self._record("figures", "reused", self.out)
            return [self.out / n for n in names]

        roi = np.asarray(self.root_ct, dtype=np.int16)
        qrca = self.rca - self.crop_lo
        fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
        ims = [roi.max(axis=0), roi.max(axis=1), roi.max(axis=2)]
        coords_rca = [(qrca[:, 2], qrca[:, 1]), (qrca[:, 2], qrca[:, 0]), (qrca[:, 1], qrca[:, 0])]
        qbest = self.best - self.crop_lo
        coords_best = [(qbest[:, 2], qbest[:, 1]), (qbest[:, 2], qbest[:, 0]), (qbest[:, 1], qbest[:, 0])]
        for ax, im, cr, cb, title in zip(axs, ims, coords_rca, coords_best, ("Axial MIP", "Coronal MIP", "Sagittal MIP")):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower")
            ax.plot(cr[0], cr[1], linewidth=1.5, label="Frozen RCA")
            ax.plot(cb[0], cb[1], linewidth=2.2, label="Left-main candidate")
            ax.set_title(title)
            ax.axis("off")
        axs[0].legend(loc="lower right")
        fig.suptitle(f"Local aortic-root search — {self.best_summary['status']}")
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(self.out / names[0], dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)

        fig, axs = plt.subplots(2, 3, figsize=(15, 9))
        base = roi.max(axis=0)
        for ax, r, i in zip(axs.ravel(), self.rays[:6], range(1, 7)):
            p = np.asarray(r["path_zyx"]) - self.crop_lo
            ax.imshow(base, cmap="gray", vmin=-100, vmax=850, origin="lower")
            ax.plot(p[:, 2], p[:, 1], linewidth=2)
            ax.set_title(f"Ray {i}: score {float(r['initial_score']):.3f}, pass {float(r['probe_plane_pass_fraction']):.2f}, r {float(r['probe_median_radius_mm']):.2f} mm")
            ax.axis("off")
        fig.suptitle("Top local left-ostium ray hypotheses")
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(self.out / names[1], dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)

        if self.rca_qc is None or self.rca_cal is None:
            self.calibrate_rca()
        rca_p = resample_path(self.rca, self.spacing, 0.5)
        lm_p = resample_path(self.best, self.spacing, 0.5)
        sr, sl = arc_mm(rca_p, self.spacing), arc_mm(lm_p, self.spacing)
        fracs = np.linspace(0.08, 0.92, 6)
        fig, axs = plt.subplots(2, 6, figsize=(16, 6))
        for col, frac in enumerate(fracs):
            for row, p, s, label in ((0, rca_p, sr, "RCA"), (1, lm_p, sl, "LM")):
                i = int(np.argmin(np.abs(s - frac * s[-1])))
                i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
                t = (p[i1] - p[i0]) * self.spacing
                im, c = orthogonal_plane(self.ct, p[i], t, self.spacing)
                axs[row, col].imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower")
                axs[row, col].plot(0, 0, "+", markersize=8)
                axs[row, col].set_title(f"{label} {s[i]:.1f} mm")
                axs[row, col].set_aspect("equal")
        fig.suptitle("Frozen RCA reference vs proposed left-main serial lumen")
        fig.tight_layout(rect=[0, 0, 1, .95])
        fig.savefig(self.out / names[2], dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)

        p = self.best
        pad = np.ceil(np.array([10.0, 14.0, 14.0]) / self.spacing).astype(int)
        lo = np.maximum(0, np.floor(p.min(axis=0)).astype(int) - pad)
        hi = np.minimum(np.asarray(self.ct.shape), np.ceil(p.max(axis=0)).astype(int) + pad + 1)
        r = np.asarray(self.ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.int16)
        q = p - lo
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        ims = [r.max(axis=0), r.max(axis=1), r.max(axis=2)]
        coords = [(q[:, 2], q[:, 1]), (q[:, 2], q[:, 0]), (q[:, 1], q[:, 0])]
        for ax, im, xy, title in zip(axs, ims, coords, ("Axial", "Coronal", "Sagittal")):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower")
            ax.plot(xy[0], xy[1], linewidth=2)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Best local left-main candidate — {self.best_summary['status']}")
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(self.out / names[3], dpi=180, bbox_inches="tight")
        plt.show(); plt.close(fig)

        self._record("figures", "recomputed_and_cached", self.out)
        return [self.out / n for n in names]

    def package(self):
        zpath = self.out / "OPENPLAQUE_LEFT_MAIN_LOCAL_ROOT_REPORT_BACK.zip"
        if self.reuse["report"] and zpath.exists():
            self._record("report", "reused", zpath)
            return zpath
        self.plot_qc()
        copy_map = {
            self.cache / "ostium_candidates.csv": "ostium_candidates.csv",
            self.cache / "left_main_candidate_qc.csv": "left_main_candidate_qc.csv",
            self.cache / "left_main_best_centerline.csv": "left_main_best_centerline.csv",
            self.cache / "left_main_best_serial_qc.csv": "left_main_best_serial_qc.csv",
            self.cache / "left_main_best_summary.json": "left_main_best_summary.json",
            self.cache / "rca_serial_qc.csv": "rca_serial_qc.csv",
            self.cache / "rca_calibration.json": "rca_calibration.json",
        }
        for src, name in copy_map.items():
            if src.exists():
                shutil.copyfile(src, self.out / name)
        html = self.out / "OPENPLAQUE_LEFT_MAIN_LOCAL_ROOT_REPORT.html"
        def img(name):
            fp = self.out / name
            return f"<h2>{name}</h2><img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(fp.read_bytes()).decode()}'>"
        cand = pd.read_csv(self.out / "left_main_candidate_qc.csv")
        ost = pd.read_csv(self.out / "ostium_candidates.csv")
        serial = pd.read_csv(self.out / "left_main_best_serial_qc.csv")
        html.write_text(
            "<html><body><h1>OpenPlaque — local aortic-root left-main search</h1>"
            "<p><b>Research use only.</b> This run deliberately does not search LAD/LCX. "
            "A left-main candidate is generated only from a local high-resolution aortic-root search and must pass coronary-sized serial orthogonal lumen QC calibrated to the frozen RCA.</p>"
            + "".join(img(n) for n in ("01_root_localization.png", "02_ostium_ray_candidates.png", "03_left_main_vs_rca_cross_sections.png", "04_left_main_mips.png"))
            + "<h2>Best summary</h2><pre>" + json.dumps(self.best_summary, indent=2) + "</pre>"
            + "<h2>Ostium rays</h2>" + ost.to_html(index=False)
            + "<h2>Left-main candidates</h2>" + cand.to_html(index=False)
            + "<h2>Serial QC</h2>" + serial.to_html(index=False)
            + "</body></html>",
            encoding="utf-8",
        )
        names = [
            "01_root_localization.png", "02_ostium_ray_candidates.png",
            "03_left_main_vs_rca_cross_sections.png", "04_left_main_mips.png",
            "ostium_candidates.csv", "left_main_candidate_qc.csv",
            "left_main_best_centerline.csv", "left_main_best_serial_qc.csv",
            "left_main_best_summary.json", "rca_serial_qc.csv", "rca_calibration.json",
            "cache_provenance.csv", html.name,
        ]
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                fp = self.out / n
                if fp.exists():
                    z.write(fp, arcname=n)
        self._record("report", "recomputed_and_cached", zpath)
        return zpath
