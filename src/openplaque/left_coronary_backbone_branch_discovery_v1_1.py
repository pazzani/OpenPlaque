from __future__ import annotations

"""Technical v1.1 fix for left-coronary backbone branch discovery.

The v1.0 notebook incorrectly treated ``Cache/Secondary_3D_Vesselness_Topology_v1/vesselness.npy``
as if it were a full Series-7 volume.  That cache is an older cropped ROI (for this case
63x79x59), whereas ``series7_int16.npy`` is the full source CCTA (524x512x512).  Sampling the
cropped array with the full source image transform is invalid.

This wrapper keeps every prospective branch-discovery gate and search parameter from v1.0, but
replaces only the vesselness plumbing: a full-resolution multiscale Frangi field is computed
once in a source-CCTA ROI enclosing the validated label-neutral backbone plus a 14-mm margin.
The 14-mm margin exceeds the 9-mm maximum branch search and preview reach.  All vesselness
sampling is then performed in that ROI's explicit source-voxel coordinates.

Research use only.  No clinical vessel labels or frozen anatomy are changed.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from . import left_coronary_backbone_branch_discovery_v1 as base

ALGORITHM = "left-coronary-backbone-branch-discovery-v1.1-local-source-vesselness"
VESSELNESS_MARGIN_MM = 14.0


def _frangi_3d(vol, spacing_zyx, scales=(0.55, 0.80, 1.10, 1.45)):
    """Bright tubular multiscale vesselness in a source-CCTA ROI."""
    x = np.clip(np.asarray(vol, np.float32), 80.0, 1000.0)
    x = (x - 80.0) / 920.0
    spacing = np.asarray(spacing_zyx, float)
    best = np.zeros_like(x, dtype=np.float32)
    eps = 1e-8

    for sm in scales:
        sig = np.maximum(sm / spacing, 0.55)
        n = float(sm * sm)
        hzz = ndi.gaussian_filter(x, sig, order=(2, 0, 0), mode="nearest") * n / spacing[0] ** 2
        hyy = ndi.gaussian_filter(x, sig, order=(0, 2, 0), mode="nearest") * n / spacing[1] ** 2
        hxx = ndi.gaussian_filter(x, sig, order=(0, 0, 2), mode="nearest") * n / spacing[2] ** 2
        hzy = ndi.gaussian_filter(x, sig, order=(1, 1, 0), mode="nearest") * n / (spacing[0] * spacing[1])
        hzx = ndi.gaussian_filter(x, sig, order=(1, 0, 1), mode="nearest") * n / (spacing[0] * spacing[2])
        hyx = ndi.gaussian_filter(x, sig, order=(0, 1, 1), mode="nearest") * n / (spacing[1] * spacing[2])

        H = np.empty(x.shape + (3, 3), dtype=np.float32)
        H[..., 0, 0] = hzz
        H[..., 1, 1] = hyy
        H[..., 2, 2] = hxx
        H[..., 0, 1] = H[..., 1, 0] = hzy
        H[..., 0, 2] = H[..., 2, 0] = hzx
        H[..., 1, 2] = H[..., 2, 1] = hyx

        vals = np.linalg.eigvalsh(H)
        vals = np.take_along_axis(vals, np.argsort(np.abs(vals), axis=-1), axis=-1)
        l1, l2, l3 = vals[..., 0], vals[..., 1], vals[..., 2]
        ra = np.abs(l2) / (np.abs(l3) + eps)
        rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
        ss = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        nz = ss[ss > 0]
        c = max(float(np.percentile(nz, 90)) * 0.45 if nz.size else 0.05, 1e-4)
        vv = (1.0 - np.exp(-(ra * ra) / 0.5)) * np.exp(-(rb * rb) / 0.5) * (1.0 - np.exp(-(ss * ss) / (2.0 * c * c)))
        vv[(l2 >= 0) | (l3 >= 0)] = 0.0
        best = np.maximum(best, np.nan_to_num(vv).astype(np.float32))
        del H, vals, hzz, hyy, hxx, hzy, hzx, hyx, vv

    return best


class _LazySourceVesselness:
    """Build a local source-space vesselness ROI on the first backbone sample."""

    def __init__(self, src, spacing_zyx, margin_mm=VESSELNESS_MARGIN_MM):
        self.src = src
        self.spacing_zyx = np.asarray(spacing_zyx, float)
        self.margin_mm = float(margin_mm)
        self.lo = None
        self.field = None
        self.roi_shape = None

    def _build(self, ref, anchor_pts):
        z = base._xyz_to_zyx(ref, np.asarray(anchor_pts, float))
        margin_vox = self.margin_mm / self.spacing_zyx
        lo = np.maximum(np.floor(z.min(axis=0) - margin_vox).astype(int), 0)
        hi = np.minimum(np.ceil(z.max(axis=0) + margin_vox).astype(int) + 1, np.asarray(self.src.shape))
        if np.any(hi - lo < 5):
            raise RuntimeError(f"source vesselness ROI is unexpectedly small: lo={lo.tolist()} hi={hi.tolist()}")
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        roi = np.asarray(self.src[sl], dtype=np.float32)
        self.field = _frangi_3d(roi, self.spacing_zyx)
        self.lo = lo.astype(float)
        self.roi_shape = tuple(int(v) for v in roi.shape)

    def sample(self, ref, pts, cval=0.0):
        pts = np.atleast_2d(np.asarray(pts, float))
        if self.field is None:
            # The first call in v1.0 is the complete resampled backbone, which is exactly
            # what we want for defining the one-time ROI.
            self._build(ref, pts)
        z = base._xyz_to_zyx(ref, pts) - self.lo[None, :]
        return map_coordinates(self.field, z.T, order=1, mode="constant", cval=float(cval))


def _source_fixed(cache):
    cache = Path(cache)
    src = np.load(base._req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = base._read_json(cache / "series7_int16.json")
    ref = sitk.GetImageFromArray(np.asarray(src))
    sp_zyx = np.asarray(meta["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp_zyx[::-1]))
    ref.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref, src, _LazySourceVesselness(src, sp_zyx)


_ORIGINAL_SAMPLE_ARRAY = base._sample_array


def _sample_array_fixed(ref, arr, pts, cval=0.0):
    if isinstance(arr, _LazySourceVesselness):
        return arr.sample(ref, pts, cval=cval)
    return _ORIGINAL_SAMPLE_ARRAY(ref, arr, pts, cval=cval)


def synthetic_local_vesselness_self_test():
    vol = np.zeros((13, 15, 17), dtype=np.float32)
    vol[:, 7, 8] = 700.0
    vv = _frangi_3d(vol, np.ones(3, float), scales=(0.8,))
    assert vv.shape == vol.shape
    assert np.all(np.isfinite(vv))
    return {"ok": True, "max_vesselness": float(vv.max())}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    # Patch only the invalid vesselness-coordinate assumption. All discovery/QC logic,
    # thresholds, branch gates, control logic, output names, and frozen-anatomy rules remain v1.0.
    base._source = _source_fixed
    base._sample_array = _sample_array_fixed
    base.ALGORITHM = ALGORITHM
    result = base.run(drive_root=drive_root, output_dir=output_dir)

    # Add explicit provenance so this technical correction is visible in every completed run.
    out = Path(output_dir) if output_dir else Path(drive_root) / base.OUTPUT_DIRNAME
    provenance = {
        "algorithm": ALGORITHM,
        "vesselness_source": "multiscale Frangi recomputed from full-resolution Series-7 source CCTA in a local backbone ROI",
        "vesselness_margin_mm": VESSELNESS_MARGIN_MM,
        "ignored_incompatible_cached_vesselness_shape": True,
        "scientific_change": "none to prospective branch/control/QC gates; coordinate-space implementation fix only",
    }
    (out / "vesselness_coordinate_fix_v1_1.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return result
