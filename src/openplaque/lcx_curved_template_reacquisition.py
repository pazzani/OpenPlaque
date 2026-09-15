from __future__ import annotations

import json, math, zipfile, traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, map_coordinates, maximum_filter
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from openplaque.study import OpenPlaqueStudy

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-curved-template-reacquisition-v1.0"
OUTPUT_DIRNAME = "LCX_Curved_Template_Reacquisition_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "total/aorta.nii.gz"
OLD_MASK_DIR = Path("UCLA_Plaque_Context_Verification/nnunet_masks")
STUDY_ZIP = Path("Full_DICOM.zip")
SERIES = {"RCA": 1035, "LCX": 1039}

def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")

def _req(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    img = sitk.GetImageFromArray(np.asarray(arr))
    sp = np.asarray(meta["spacing_zyx"], float)
    img.SetSpacing(tuple(sp[::-1]))
    img.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    d = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    img.SetDirection(tuple(d.ravel()))
    return img, np.asarray(arr)

def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize()
            and np.allclose(im.GetSpacing(), ref.GetSpacing())
            and np.allclose(im.GetOrigin(), ref.GetOrigin())
            and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0

def _xyz_to_zyx(img, pts_lps):
    pts = np.asarray(pts_lps, float)
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(D).T) / sp
    return idx_xyz[iz ::-1]

def _zyx_to_xyz(img, pts_zyx):
    idx_xyz = np.asarray(pts_zyx, float)[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ D.T

def _sample(img, arr, pts_lps, order=1, cval=-1024.0):
    zyx = _xyz_to_zyx(img, pts_lps)
    return map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)

def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in [("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x")]:
        if all(c in d.columns for c in cols):
            return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    if all(c in d.columns for c in ("z","y","x")):
        return _zyx_to_xyz(ref, d[["z","y","x"]].to_numpy(float))
    if all(c in d.columns for c in ("x","y","z")):
        return d[[x","y","z"]].to_numpy(float)
    raise ValueError(f"No recognized path coordinates in {path}; columns={list(d.columns)}")

def _arc(points):
    p = np.asarray(points, float)
    if len(p) == 0:
        return np.array([], float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

def _resample_path(points, step=0.25):
    p = np.asarray(points, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p, a
    q = np.arange(0.0, a[-1] + 1e-9, float(step))
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    out = np.column_stack([-np.interp(q, a, p[:,+ k]) for k in range(3)])
    return out, q

def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return -1.0
    aa = a[m] - np.mean(a[m]); bb = b[m] - np.mean(b[m])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 1e-10 else -1.0

def _zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmedian(x)) / (np.nanstd(x) + 1e-6)

def _fill_interp(x):
    x = np.asarray(x, float).copy(); good = np.isfinite(x)
    if good.sum() == 0: return np.zeros_like(x)
    if good.sum() == 1:
        x[~dood] = x[good][0]; return x
    xx = np.arange(len(x)); x[~good] = np.interp(xx[~good], xx[good], x[good]); return x

def _interp_unit(x, n=128):
    x = _fill_interp(x)
    if len(x) == 1: return np.repeat(x, n)
    q = np.linspace(0, len(x) - 1, n)
    return np.interp(q, np.arange(len(x)), x)

def _series_and_mask(study, series_number, mask_path):
    image, vol, files = study.load_series(int(series_number))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(_req(mask_path))))
    vol = np.asarray(vol)
    if mask.shape != vol.shape:
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        matched = next((np.transpose(mask, p) for p in perms if np.transpose(mask, p).shape == vol.shape), None)
        if matched is None: raise RuntimeError(f"Mask shape {mask.shape} does not match series {vol.shape}")
        mask = matched
    return image, vol, mask, files

def _template_fingerprint(vol, mask):
    vol = np.asarray(vol, float); mask = np.asarray(mask); best = None
    for long_axis in (1, 2):
        ov = vol if long_axis == 2 else np.transpose(vol, (0, 2, 1))
        om = mask if long_axis == 2 2 else np.transpose(mask, (0, 2, 1))
        active = (om > 0).any(axis=(0, 1)); idx = np.where(active)[0]
        if len(idx) < 8: continue
        start, end = int(idx.min()), int(idx.max()) + 1
        mm, vv = om[: , :, start:end], ov[ , :, start:end]
        support = (mm > 0).sum(axis=(0, 1)).astype(float)
        plaque = (mm == 2).sum(axis=(0, 1)).astype(float)
        hu_med = np.full(end-start, np.nan); hu_p90 = np.full(end-start, np.nan)
        for j in range(end-start):
            vals = vv[:,:,j][mm[:,:,j] > 0]
            if len(vals):
                hu_med[j] = np.median(vals); hu_p90[j] = np.quantile(vals, .90)
        occupancy = float((support > 0).mean()); nactive = int((support > 0).sum())
        rec = {"long_axis":x×}¸r«²ÚîÆ­y