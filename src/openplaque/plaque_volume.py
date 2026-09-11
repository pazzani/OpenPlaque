"""Reusable plaque-volume definitions used by OpenPlaque PAV analyses.

This module reproduces the plaque numerator used in
"Best-Estimate Plaque Types by Artery":

* strict nnU-Net plaque core: label 2
* context-expanded low-attenuation, fibrofatty, and fibrous candidates from a
  one-voxel anatomical band around the union of labels 1 and 2
* dense calcium counted only inside the strict nnU-Net plaque core

The result is a research plaque-volume proxy, not a validated clinical
quantitative CCTA measurement.

Research use only. Not clinically validated. Not for diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


@dataclass
class PlaqueVolumeEstimate:
    artery: str
    spacing_xyz_mm: tuple[float, float, float]
    voxel_volume_mm3: float

    strict_core_mask: np.ndarray
    best_estimate_mask: np.ndarray
    context_candidate_mask: np.ndarray

    low_attenuation_mask: np.ndarray
    fibrofatty_mask: np.ndarray
    fibrous_mask: np.ndarray
    dense_calcium_mask: np.ndarray

    strict_nnunet_core_voxels: int
    strict_nnunet_core_volume_mm3: float
    context_selected_candidate_voxels: int
    context_expanded_candidate_volume_mm3: float
    total_plaque_volume_proxy_voxels: int
    total_plaque_volume_proxy_mm3: float

    low_attenuation_voxels: int
    low_attenuation_mm3: float
    fibrofatty_voxels: int
    fibrofatty_mm3: float
    fibrous_voxels: int
    fibrous_mm3: float
    dense_calcium_voxels: int
    dense_calcium_mm3: float

    def to_row(self) -> dict:
        total = self.total_plaque_volume_proxy_voxels
        return {
            "artery": self.artery,
            "voxel_volume_mm3": self.voxel_volume_mm3,
            "strict_nnunet_core_voxels": self.strict_nnunet_core_voxels,
            "strict_nnunet_core_volume_mm3": self.strict_nnunet_core_volume_mm3,
            "context_selected_candidate_voxels": self.context_selected_candidate_voxels,
            "context_expanded_candidate_volume_mm3": self.context_expanded_candidate_volume_mm3,
            "total_plaque_volume_proxy_voxels": self.total_plaque_volume_proxy_voxels,
            "total_plaque_volume_proxy_mm3": self.total_plaque_volume_proxy_mm3,
            "low_attenuation_voxels": self.low_attenuation_voxels,
            "low_attenuation_mm3": self.low_attenuation_mm3,
            "fibrofatty_voxels": self.fibrofatty_voxels,
            "fibrofatty_mm3": self.fibrofatty_mm3,
            "fibrous_voxels": self.fibrous_voxels,
            "fibrous_mm3": self.fibrous_mm3,
            "dense_calcium_voxels": self.dense_calcium_voxels,
            "dense_calcium_mm3": self.dense_calcium_mm3,
            "low_attenuation_fraction": self.low_attenuation_voxels / total if total else 0.0,
            "fibrofatty_fraction": self.fibrofatty_voxels / total if total else 0.0,
            "fibrous_fraction": self.fibrous_voxels / total if total else 0.0,
            "dense_calcium_fraction": self.dense_calcium_voxels / total if total else 0.0,
        }


def voxel_volume_mm3(spacing_xyz_mm) -> float:
    spacing = tuple(float(x) for x in spacing_xyz_mm)
    if len(spacing) != 3:
        raise ValueError("spacing_xyz_mm must contain exactly three values")
    return float(np.prod(spacing))


def _structure(ndim: int, connectivity: int):
    if ndim == 2:
        rank = 1 if connectivity <= 4 else 2
    elif ndim == 3:
        rank = 1 if connectivity <= 6 else 2 if connectivity <= 18 else 3
    else:
        rank = ndim
    return ndi.generate_binary_structure(ndim, rank)


def vessel_context_mask(
    segmentation_mask: np.ndarray,
    vessel_label: int = 1,
    plaque_label: int = 2,
    vessel_dilation_voxels: int = 1,
    connectivity: int = 26,
) -> np.ndarray:
    """Replicate the anatomical search band used by the earlier analysis."""
    mask = np.asarray(segmentation_mask)
    vessel = (mask == vessel_label) | (mask == plaque_label)
    radius = int(vessel_dilation_voxels)
    if radius <= 0:
        return vessel
    return ndi.binary_dilation(
        vessel,
        structure=_structure(mask.ndim, connectivity),
        iterations=radius,
    )


def _range_mask(values: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    out = np.ones(values.shape, dtype=bool)
    if lo is not None:
        out &= values >= float(lo)
    if hi is not None:
        out &= values < float(hi)
    return out


def estimate_best_estimate_plaque_volume(
    artery: str,
    volume: np.ndarray,
    segmentation_mask: np.ndarray,
    spacing_xyz_mm,
    *,
    vessel_label: int = 1,
    plaque_label: int = 2,
    vessel_dilation_voxels: int = 1,
    exclude_vessel_label_for_candidates: bool = True,
    low_attenuation_min_hu: float = -30.0,
    low_attenuation_max_hu: float = 30.0,
    fibrofatty_min_hu: float = 30.0,
    fibrofatty_max_hu: float = 130.0,
    fibrous_min_hu: float = 130.0,
    fibrous_max_hu: float = 350.0,
    dense_calcium_min_hu: float = 350.0,
    connectivity: int = 26,
) -> PlaqueVolumeEstimate:
    """Reproduce the earlier context-expanded TPV proxy and return its masks."""
    volume = np.asarray(volume)
    mask = np.asarray(segmentation_mask)
    if volume.shape != mask.shape:
        raise ValueError("volume and segmentation_mask must have the same shape")
    if volume.ndim != 3:
        raise ValueError("volume and segmentation_mask must be 3-D")

    vv = voxel_volume_mm3(spacing_xyz_mm)
    strict_core = mask == plaque_label

    anatomy_band = vessel_context_mask(
        mask,
        vessel_label=vessel_label,
        plaque_label=plaque_label,
        vessel_dilation_voxels=vessel_dilation_voxels,
        connectivity=connectivity,
    )
    candidate_region = anatomy_band & (mask != plaque_label)
    if exclude_vessel_label_for_candidates:
        candidate_region &= mask != vessel_label

    core_low = strict_core & _range_mask(volume, low_attenuation_min_hu, low_attenuation_max_hu)
    core_ff = strict_core & _range_mask(volume, fibrofatty_min_hu, fibrofatty_max_hu)
    core_fibrous = strict_core & _range_mask(volume, fibrous_min_hu, fibrous_max_hu)
    core_calc = strict_core & _range_mask(volume, dense_calcium_min_hu, None)

    cand_low = candidate_region & _range_mask(volume, low_attenuation_min_hu, low_attenuation_max_hu)
    cand_ff = candidate_region & _range_mask(volume, fibrofatty_min_hu, fibrofatty_max_hu)
    cand_fibrous = candidate_region & _range_mask(volume, fibrous_min_hu, fibrous_max_hu)

    low = core_low | cand_low
    ff = core_ff | cand_ff
    fibrous = core_fibrous | cand_fibrous
    calc = core_calc

    context_selected = cand_low | cand_ff | cand_fibrous
    best = low | ff | fibrous | calc

    def n(m):
        return int(np.sum(m))

    def vol(m):
        return float(n(m) * vv)

    return PlaqueVolumeEstimate(
        artery=str(artery),
        spacing_xyz_mm=tuple(float(x) for x in spacing_xyz_mm),
        voxel_volume_mm3=vv,
        strict_core_mask=strict_core,
        best_estimate_mask=best,
        context_candidate_mask=context_selected,
        low_attenuation_mask=low,
        fibrofatty_mask=ff,
        fibrous_mask=fibrous,
        dense_calcium_mask=calc,
        strict_nnunet_core_voxels=n(strict_core),
        strict_nnunet_core_volume_mm3=vol(strict_core),
        context_selected_candidate_voxels=n(context_selected),
        context_expanded_candidate_volume_mm3=vol(context_selected),
        total_plaque_volume_proxy_voxels=n(best),
        total_plaque_volume_proxy_mm3=vol(best),
        low_attenuation_voxels=n(low),
        low_attenuation_mm3=vol(low),
        fibrofatty_voxels=n(ff),
        fibrofatty_mm3=vol(ff),
        fibrous_voxels=n(fibrous),
        fibrous_mm3=vol(fibrous),
        dense_calcium_voxels=n(calc),
        dense_calcium_mm3=vol(calc),
    )


def segmentation_mask_with_plaque_proxy(
    segmentation_mask: np.ndarray,
    plaque_proxy_mask: np.ndarray,
    plaque_label: int = 2,
) -> np.ndarray:
    """Return a copy whose proxy voxels are labeled plaque for outer-wall seeding."""
    mask = np.asarray(segmentation_mask).copy()
    proxy = np.asarray(plaque_proxy_mask, dtype=bool)
    if mask.shape != proxy.shape:
        raise ValueError("segmentation_mask and plaque_proxy_mask must have the same shape")
    mask[proxy] = plaque_label
    return mask
