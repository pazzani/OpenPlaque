"""Experimental percent atheroma volume (PAV) utilities.

This module is intentionally separate from the existing plaque model.  It uses the
existing lumen/vessel (label 1) and plaque (label 2) masks as seeds and estimates a
candidate outer-vessel envelope from the original CCTA volume.

The outer-wall estimate is a research prototype, not a clinically validated EEM
segmentation.  PAV should only be reported after visual/manual validation of the
outer-wall contours.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class PAVResult:
    plaque_volume_mm3: float
    outer_vessel_volume_mm3: float
    pav_percent: float
    plaque_voxels: int
    outer_vessel_voxels: int
    outer_wall_mask: np.ndarray
    candidate_seed_mask: np.ndarray
    parameters: dict

    def summary(self):
        print("Experimental PAV estimate")
        print(f"Plaque volume:       {self.plaque_volume_mm3:.2f} mm^3")
        print(f"Outer vessel volume: {self.outer_vessel_volume_mm3:.2f} mm^3")
        print(f"PAV:                 {self.pav_percent:.2f}%")
        print("Research prototype only; validate outer-wall contours before interpretation.")


def _spacing_zyx(spacing):
    """Convert SimpleITK-style (x, y, z) spacing to numpy array (z, y, x)."""
    spacing = tuple(float(v) for v in spacing)
    if len(spacing) != 3:
        raise ValueError("spacing must contain exactly three values (x, y, z)")
    return spacing[::-1]


def voxel_volume_mm3(spacing):
    return float(np.prod(tuple(float(v) for v in spacing)))


def _keep_components_touching_seed(candidate, seed, connectivity=26):
    if connectivity <= 6:
        structure = ndi.generate_binary_structure(3, 1)
    elif connectivity <= 18:
        structure = ndi.generate_binary_structure(3, 2)
    else:
        structure = ndi.generate_binary_structure(3, 3)

    labels, n = ndi.label(candidate, structure=structure)
    if n == 0:
        return seed.copy()

    touching = np.unique(labels[seed & (labels > 0)])
    if touching.size == 0:
        return seed.copy()
    return np.isin(labels, touching) | seed


def estimate_outer_wall_candidate(
    volume,
    mask,
    spacing,
    vessel_label=1,
    plaque_label=2,
    max_wall_thickness_mm=2.0,
    fat_threshold_hu=-30.0,
    closing_iterations=1,
    fill_holes=True,
    connectivity=26,
):
    """Estimate a candidate outer-vessel envelope around the current artery mask.

    Method
    ------
    1. Use vessel/lumen + plaque labels as an anatomical seed.
    2. Expand by a physical distance (mm), not a fixed voxel count.
    3. Reject newly added voxels with attenuation below ``fat_threshold_hu``.
       Seed voxels are never removed, so low-attenuation plaque already identified
       by the plaque model is preserved.
    4. Keep only connected regions touching the seed and optionally close/fill them.

    This is deliberately conservative and transparent.  It is meant to generate a
    contour for visual review and later refinement, not to stand in for a validated
    EEM/outer-wall segmentation model.
    """
    volume = np.asarray(volume)
    mask = np.asarray(mask)
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have the same shape")
    if volume.ndim != 3:
        raise ValueError("volume and mask must be 3-D arrays")
    if max_wall_thickness_mm < 0:
        raise ValueError("max_wall_thickness_mm must be non-negative")

    seed = (mask == vessel_label) | (mask == plaque_label)
    if not np.any(seed):
        return np.zeros_like(seed, dtype=bool)

    distance_mm = ndi.distance_transform_edt(~seed, sampling=_spacing_zyx(spacing))
    within_distance = distance_mm <= float(max_wall_thickness_mm)

    # Preserve the existing segmentation exactly; HU filtering only applies to
    # newly proposed outer-wall voxels.
    tissue_candidate = volume >= float(fat_threshold_hu)
    candidate = seed | (within_distance & tissue_candidate)
    candidate = _keep_components_touching_seed(candidate, seed, connectivity=connectivity)

    if closing_iterations and int(closing_iterations) > 0:
        structure = ndi.generate_binary_structure(3, 2)
        candidate = ndi.binary_closing(
            candidate,
            structure=structure,
            iterations=int(closing_iterations),
        ) | seed

    if fill_holes:
        # Fill holes slice-by-slice rather than globally; coronary cross-sections
        # are locally tubular and this is less likely to bridge distant anatomy.
        filled = np.zeros_like(candidate, dtype=bool)
        for z in range(candidate.shape[0]):
            filled[z] = ndi.binary_fill_holes(candidate[z])
        candidate = filled | seed

    return candidate.astype(bool)


def compute_pav(plaque_mask, outer_wall_mask, spacing):
    """Compute PAV = 100 * plaque volume / outer-vessel volume."""
    plaque_mask = np.asarray(plaque_mask, dtype=bool)
    outer_wall_mask = np.asarray(outer_wall_mask, dtype=bool)
    if plaque_mask.shape != outer_wall_mask.shape:
        raise ValueError("plaque_mask and outer_wall_mask must have the same shape")
    if np.any(plaque_mask & ~outer_wall_mask):
        raise ValueError("outer_wall_mask must contain every plaque voxel")

    vv = voxel_volume_mm3(spacing)
    plaque_voxels = int(np.sum(plaque_mask))
    outer_voxels = int(np.sum(outer_wall_mask))
    plaque_volume = plaque_voxels * vv
    outer_volume = outer_voxels * vv
    pav = 100.0 * plaque_volume / outer_volume if outer_volume > 0 else 0.0
    return plaque_volume, outer_volume, pav, plaque_voxels, outer_voxels


def estimate_pav_from_labels(
    volume,
    mask,
    spacing,
    vessel_label=1,
    plaque_label=2,
    max_wall_thickness_mm=2.0,
    fat_threshold_hu=-30.0,
    closing_iterations=1,
    fill_holes=True,
    connectivity=26,
):
    """Estimate a candidate outer wall and compute experimental PAV."""
    outer = estimate_outer_wall_candidate(
        volume=volume,
        mask=mask,
        spacing=spacing,
        vessel_label=vessel_label,
        plaque_label=plaque_label,
        max_wall_thickness_mm=max_wall_thickness_mm,
        fat_threshold_hu=fat_threshold_hu,
        closing_iterations=closing_iterations,
        fill_holes=fill_holes,
        connectivity=connectivity,
    )
    plaque = np.asarray(mask) == plaque_label
    plaque_volume, outer_volume, pav, plaque_voxels, outer_voxels = compute_pav(
        plaque, outer, spacing
    )

    params = dict(
        vessel_label=vessel_label,
        plaque_label=plaque_label,
        max_wall_thickness_mm=max_wall_thickness_mm,
        fat_threshold_hu=fat_threshold_hu,
        closing_iterations=closing_iterations,
        fill_holes=fill_holes,
        connectivity=connectivity,
    )

    return PAVResult(
        plaque_volume_mm3=plaque_volume,
        outer_vessel_volume_mm3=outer_volume,
        pav_percent=pav,
        plaque_voxels=plaque_voxels,
        outer_vessel_voxels=outer_voxels,
        outer_wall_mask=outer,
        candidate_seed_mask=(np.asarray(mask) == vessel_label) | plaque,
        parameters=params,
    )


def show_pav_overlay(volume, mask, outer_wall_mask, z=None, vessel_label=1, plaque_label=2,
                     vmin=-200, vmax=800):
    """Display one axial slice with seed/plaque and candidate outer-wall contours."""
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    volume = np.asarray(volume)
    mask = np.asarray(mask)
    outer_wall_mask = np.asarray(outer_wall_mask, dtype=bool)

    if z is None:
        counts = np.sum(mask == plaque_label, axis=(1, 2))
        if np.any(counts):
            z = int(np.argmax(counts))
        else:
            counts = np.sum(outer_wall_mask, axis=(1, 2))
            z = int(np.argmax(counts))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)

    for contour in find_contours(outer_wall_mask[z].astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, label="candidate outer wall")
    for contour in find_contours((mask[z] == plaque_label).astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, linestyle="--", label="plaque")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right")
    ax.set_title(f"Experimental PAV outer-wall candidate, slice {z}")
    ax.axis("off")
    return fig, ax
