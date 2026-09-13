"""Experimental percent atheroma volume (PAV) utilities.

This module is intentionally separate from the existing plaque model. It uses the
existing vessel/plaque masks as seeds and estimates a candidate outer-vessel
envelope from the original CCTA volume.

The outer-wall estimate is a research prototype, not a clinically validated EEM
segmentation. PAV should only be reported after visual/manual validation of the
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
    """Convert SimpleITK-style (x, y, z) spacing to NumPy order (z, y, x)."""
    spacing = tuple(float(v) for v in spacing)
    if len(spacing) != 3:
        raise ValueError("spacing must contain exactly three values (x, y, z)")
    return spacing[::-1]


def voxel_volume_mm3(spacing):
    return float(np.prod(tuple(float(v) for v in spacing)))


def _structure_2d(connectivity=8):
    return ndi.generate_binary_structure(2, 1 if connectivity <= 4 else 2)


def _keep_2d_components_touching_anchor(candidate_2d, anchor_2d, seed_2d, connectivity=8):
    """Keep local 2-D regions attached to the artery anchor; always preserve seed."""
    candidate_2d = np.asarray(candidate_2d, dtype=bool)
    anchor_2d = np.asarray(anchor_2d, dtype=bool)
    seed_2d = np.asarray(seed_2d, dtype=bool)

    labels, n = ndi.label(candidate_2d, structure=_structure_2d(connectivity))
    if n == 0 or not np.any(anchor_2d):
        return seed_2d.copy()

    touching = np.unique(labels[anchor_2d & (labels > 0)])
    if touching.size == 0:
        return seed_2d.copy()

    return np.isin(labels, touching) | seed_2d


def _slice_local_outer_wall(
    volume,
    seed,
    anchor,
    spacing,
    max_wall_thickness_mm,
    fat_threshold_hu,
    closing_iterations=1,
    fill_holes=True,
    connectivity=8,
):
    """Estimate the candidate wall independently in each axial artery slice.

    New pixels must be close to BOTH the analysis seed and the original artery
    anchor in the same slice. This deliberately avoids 3-D growth into adjacent
    anatomy and suppresses detached regions that only happen to be connected in a
    neighboring slice.
    """
    volume = np.asarray(volume)
    seed = np.asarray(seed, dtype=bool)
    anchor = np.asarray(anchor, dtype=bool)
    out = seed.copy()

    spacing_x, spacing_y, _ = tuple(float(v) for v in spacing)
    sampling_yx = (spacing_y, spacing_x)
    max_distance = float(max_wall_thickness_mm)
    structure = _structure_2d(connectivity)

    for z in range(seed.shape[0]):
        seed_z = seed[z]
        anchor_z = anchor[z]

        if not np.any(seed_z):
            continue
        if not np.any(anchor_z):
            # Preserve plaque/seed voxels, but do not invent an outer wall when
            # the original artery segmentation provides no local anchor.
            out[z] = seed_z
            continue

        distance_to_seed = ndi.distance_transform_edt(~seed_z, sampling=sampling_yx)
        distance_to_anchor = ndi.distance_transform_edt(~anchor_z, sampling=sampling_yx)

        tissue_z = volume[z] >= float(fat_threshold_hu)
        growth = (
            (distance_to_seed <= max_distance)
            & (distance_to_anchor <= max_distance)
            & tissue_z
        )

        local = seed_z | growth
        local = _keep_2d_components_touching_anchor(
            local, anchor_z, seed_z, connectivity=connectivity
        )

        if closing_iterations and int(closing_iterations) > 0:
            local = ndi.binary_closing(
                local,
                structure=structure,
                iterations=int(closing_iterations),
            ) | seed_z

        if fill_holes:
            local = ndi.binary_fill_holes(local) | seed_z

        # Morphology may bridge into an unrelated structure, so reapply the
        # physical and HU support before the final connectivity filter.
        supported = seed_z | (local & growth)
        out[z] = _keep_2d_components_touching_anchor(
            supported, anchor_z, seed_z, connectivity=connectivity
        )

    return out.astype(bool)


def _volume_growth_outer_wall(
    volume,
    seed,
    anchor,
    spacing,
    max_wall_thickness_mm,
    fat_threshold_hu,
):
    """Legacy 3-D growth retained for comparison/debugging."""
    spacing_zyx = _spacing_zyx(spacing)
    ds = ndi.distance_transform_edt(~seed, sampling=spacing_zyx)
    da = ndi.distance_transform_edt(~anchor, sampling=spacing_zyx)
    tissue = np.asarray(volume) >= float(fat_threshold_hu)
    growth = (
        (ds <= float(max_wall_thickness_mm))
        & (da <= float(max_wall_thickness_mm))
        & tissue
    )
    return (seed | growth).astype(bool)


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
    connectivity=8,
    reference_mask=None,
    growth_mode="slice_local",
):
    """Estimate a candidate outer-vessel envelope around the current artery mask.

    Parameters
    ----------
    reference_mask : array-like, optional
        Original nnU-Net artery mask. Supply this when ``mask`` has been augmented
        with a broader plaque proxy. New outer-wall growth remains locally anchored
        to the original artery while all seed/plaque voxels are preserved.
    growth_mode : {"slice_local", "volume"}
        ``slice_local`` is the current default and grows independently in each
        axial slice. ``volume`` preserves the earlier 3-D behavior for comparison.

    Notes
    -----
    This remains a research candidate contour, not a validated outer-wall/EEM
    segmentation.
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

    if reference_mask is None:
        anchor = seed.copy()
    else:
        reference_mask = np.asarray(reference_mask)
        if reference_mask.shape != mask.shape:
            raise ValueError("reference_mask and mask must have the same shape")
        anchor = (
            (reference_mask == vessel_label)
            | (reference_mask == plaque_label)
        )
        if not np.any(anchor):
            raise ValueError("reference_mask has no vessel/plaque anchor voxels")

    mode = str(growth_mode).lower()
    if mode == "slice_local":
        return _slice_local_outer_wall(
            volume=volume,
            seed=seed,
            anchor=anchor,
            spacing=spacing,
            max_wall_thickness_mm=max_wall_thickness_mm,
            fat_threshold_hu=fat_threshold_hu,
            closing_iterations=closing_iterations,
            fill_holes=fill_holes,
            connectivity=connectivity,
        )
    if mode == "volume":
        return _volume_growth_outer_wall(
            volume=volume,
            seed=seed,
            anchor=anchor,
            spacing=spacing,
            max_wall_thickness_mm=max_wall_thickness_mm,
            fat_threshold_hu=fat_threshold_hu,
        )

    raise ValueError("growth_mode must be 'slice_local' or 'volume'")


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
    connectivity=8,
    reference_mask=None,
    growth_mode="slice_local",
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
        reference_mask=reference_mask,
        growth_mode=growth_mode,
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
        reference_mask_supplied=reference_mask is not None,
        growth_mode=growth_mode,
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


def show_pav_overlay(
    volume,
    mask,
    outer_wall_mask,
    z=None,
    vessel_label=1,
    plaque_label=2,
    reference_mask=None,
    vmin=-200,
    vmax=800,
):
    """Display CT with original artery, plaque, and candidate outer-wall contours."""
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    volume = np.asarray(volume)
    mask = np.asarray(mask)
    outer_wall_mask = np.asarray(outer_wall_mask, dtype=bool)

    if reference_mask is None:
        reference_mask = mask
    reference_mask = np.asarray(reference_mask)

    if z is None:
        counts = np.sum(mask == plaque_label, axis=(1, 2))
        if np.any(counts):
            z = int(np.argmax(counts))
        else:
            counts = np.sum(outer_wall_mask, axis=(1, 2))
            z = int(np.argmax(counts))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)

    artery = (reference_mask[z] == vessel_label) | (reference_mask[z] == plaque_label)
    plaque = mask[z] == plaque_label

    layers = [
        (outer_wall_mask[z], "yellow", "-", "candidate outer wall"),
        (artery, "lime", ":", "original artery mask"),
        (plaque, "cyan", "--", "plaque"),
    ]

    for layer, color, linestyle, label in layers:
        first = True
        for contour in find_contours(layer.astype(float), 0.5):
            ax.plot(
                contour[:, 1], contour[:, 0],
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                label=label if first else None,
            )
            first = False

    ax.legend(loc="upper right")
    ax.set_title(f"Experimental slice-local PAV candidate, slice {z}")
    ax.axis("off")
    return fig, ax
