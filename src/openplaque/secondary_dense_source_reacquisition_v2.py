from __future__ import annotations

"""Memory-safe wrapper for dense source-CCTA reacquisition.

Scientific search logic and thresholds are unchanged from
secondary_dense_source_reacquisition.py. The only substantive runtime change is that
orthogonal source planes are interpolated directly from the int16 memmap into a small
float32 output plane instead of materializing the entire CT as float64 on every call.
"""

import json
import math

import numpy as np
from scipy import ndimage as ndi

from . import secondary_3d_vesselness_topology as base
from .secondary_dense_source_reacquisition import (
    SecondaryDenseSourceReacquisitionWorkflow as _V1,
    _compact_component,
)

ALGORITHM_VERSION = "secondary-dense-source-reacquisition-v1.1-memmap-safe"


def orthogonal_plane_memmap(ct, point_zyx, tangent_zyx_mm, spacing, half_mm=4.5, pix_mm=0.18):
    """Interpolate one small float32 plane directly from a source CT/memmap."""
    t = base._unit(tangent_zyx_mm)
    u, v = base._orth_basis(t)
    grid = np.arange(-half_mm, half_mm + 1e-9, pix_mm)
    vv, uu = np.meshgrid(grid, grid, indexing="ij")
    center_mm = np.asarray(point_zyx, float) * np.asarray(spacing, float)
    zyx_mm = center_mm[None, None, :] + uu[..., None] * u + vv[..., None] * v
    zyx = zyx_mm / np.asarray(spacing, float)
    im = ndi.map_coordinates(
        ct,
        [zyx[..., 0], zyx[..., 1], zyx[..., 2]],
        output=np.float32,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return im, grid


def synthetic_memmap_plane_self_test():
    z, y, x = np.indices((25, 25, 25))
    vol = (100 + 600 * (((y - 12) ** 2 + (x - 12) ** 2) <= 4)).astype(np.int16)
    im, grid = orthogonal_plane_memmap(
        vol,
        np.array([12.0, 12.0, 12.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.4, 0.4, 0.4]),
        half_mm=3.0,
        pix_mm=0.16,
    )
    return {
        "passed": bool(im.dtype == np.float32 and im.shape == (len(grid), len(grid)) and float(im.max()) > 500),
        "dtype": str(im.dtype),
        "shape": list(im.shape),
        "max_hu": float(im.max()),
    }


class SecondaryDenseSourceReacquisitionWorkflow(_V1):
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse=reuse)
        self.cache = self.root / "Cache" / "Secondary_Dense_Source_Reacquisition_v1_1"
        self.out = self.root / "Secondary_Dense_Source_Reacquisition_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    def load_inputs(self, reacquisition_origin_arc_mm=12.20):
        snap = super().load_inputs(reacquisition_origin_arc_mm=reacquisition_origin_arc_mm)
        snap = dict(snap)
        snap["algorithm"] = ALGORITHM_VERSION
        snap["plane_sampler"] = "direct int16 memmap -> float32 orthogonal plane"
        snap["whole_volume_float_conversion_per_plane"] = False
        base._write_json(snap, self.cache / "input_snapshot.json")
        return snap

    def _calibrate_lumen(self):
        sp = base.resample_path(self.seed, self.spacing, 0.30)
        s = base.arc_mm(sp, self.spacing)
        p = sp[(s >= 10.0) & (s <= 12.7)]
        center = ndi.map_coordinates(
            self.ct,
            [p[:, 0], p[:, 1], p[:, 2]],
            output=np.float32,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        refhu = float(np.median(center))
        thr = float(max(170.0, min(300.0, 0.38 * refhu)))
        rows = []
        for i, pt in enumerate(p):
            i0, i1 = max(0, i - 2), min(len(p) - 1, i + 2)
            t = (p[i1] - p[i0]) * self.spacing
            im, grid = orthogonal_plane_memmap(
                self.ct, pt, t, self.spacing, half_mm=4.2, pix_mm=0.16
            )
            m = _compact_component(im, grid, thr, max_shift_mm=0.80)
            if m is not None:
                rows.append(m)
        if not rows:
            raise RuntimeError("Could not calibrate lumen on accepted secondary branch")
        import pandas as pd

        df = pd.DataFrame(rows)
        good = df[
            (df.radius_mm >= 0.65)
            & (df.radius_mm <= 2.65)
            & (df.centroid_shift_mm <= 0.80)
        ]
        if len(good) < 3:
            good = df
        self.calibration = {
            "reference_center_hu": refhu,
            "bright_threshold_hu": thr,
            "median_radius_mm": float(good.radius_mm.median()),
            "p90_radius_mm": float(good.radius_mm.quantile(0.9)),
            "median_circularity": float(good.circularity.median()),
            "n_planes": int(len(good)),
        }
        base._write_json(self.calibration, self.cache / "lumen_calibration.json")

    def _plane_measure(self, point, direction, half_mm=4.0, pix_mm=0.18):
        im, grid = orthogonal_plane_memmap(
            self.ct,
            point,
            direction,
            self.spacing,
            half_mm=half_mm,
            pix_mm=pix_mm,
        )
        m = _compact_component(
            im,
            grid,
            self.calibration["bright_threshold_hu"],
            max_shift_mm=0.90,
        )
        if m is None:
            return im, None, False, 0.0
        rref = float(self.calibration["median_radius_mm"])
        passed = bool(
            0.65 <= m["radius_mm"] <= 2.65
            and m["centroid_shift_mm"] <= 0.75
            and m["circularity"] >= 0.30
            and m["component_median_hu"]
            >= max(220.0, 0.45 * self.calibration["reference_center_hu"])
        )
        rscore = math.exp(
            -0.5
            * ((m["radius_mm"] - rref) / max(0.55 * rref, 0.70)) ** 2
        )
        sscore = math.exp(-0.5 * (m["centroid_shift_mm"] / 0.55) ** 2)
        cscore = float(
            np.clip(
                m["circularity"] / max(self.calibration["median_circularity"], 0.35),
                0,
                1,
            )
        )
        hscore = float(
            np.clip(
                m["component_median_hu"]
                / max(self.calibration["reference_center_hu"], 1.0),
                0,
                1.2,
            )
            / 1.2
        )
        score = float(0.35 * rscore + 0.35 * sscore + 0.18 * cscore + 0.12 * hscore)
        return im, m, passed, score

    def dense_scan(self, *args, **kwargs):
        hits = super().dense_scan(*args, **kwargs)
        if self.scan_summary is not None:
            self.scan_summary = dict(self.scan_summary)
            self.scan_summary["algorithm"] = ALGORITHM_VERSION
            self.scan_summary["plane_sampler"] = "direct int16 memmap -> float32 orthogonal plane"
            base._write_json(self.scan_summary, self.cache / "dense_scan_summary.json")
        return hits

    def run(self):
        summary = super().run()
        summary = dict(summary)
        summary["algorithm"] = ALGORITHM_VERSION
        summary["plane_sampler"] = "direct int16 memmap -> float32 orthogonal plane"
        self.summary = summary
        base._write_json(summary, self.cache / "summary.json")
        return summary
