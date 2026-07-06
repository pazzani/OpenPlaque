"""
OpenPlaque boundary refinement.

Experimental post-processing for nnU-Net plaque masks.
Research use only. Not clinically validated.
"""

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class RefinementResult:
    original_mask: np.ndarray
    refined_mask: np.ndarray
    removed_mask: np.ndarray
    spacing: tuple
    method: str
    parameters: dict

    @property
    def voxel_volume_mm3(self):
        return float(np.prod(self.spacing))

    @property
    def original_plaque_voxels(self):
        return int(np.sum(self.original_mask == 2))

    @property
    def refined_plaque_voxels(self):
        return int(np.sum(self.refined_mask == 2))

    @property
    def removed_voxels(self):
        return int(np.sum(self.removed_mask))

    @property
    def original_tpv_mm3(self):
        return self.original_plaque_voxels * self.voxel_volume_mm3

    @property
    def refined_tpv_mm3(self):
        return self.refined_plaque_voxels * self.voxel_volume_mm3

    @property
    def removed_volume_mm3(self):
        return self.removed_voxels * self.voxel_volume_mm3

    def summary(self):
        print("Boundary refinement")
        print("Method:", self.method)
        print("Parameters:", self.parameters)
        print()
        print(f"Original plaque voxels: {self.original_plaque_voxels}")
        print(f"Refined plaque voxels:  {self.refined_plaque_voxels}")
        print(f"Removed voxels:         {self.removed_voxels}")
        print()
        print(f"Original TPV: {self.original_tpv_mm3:.2f} mm³")
        print(f"Refined TPV:  {self.refined_tpv_mm3:.2f} mm³")
        print(f"Removed vol:  {self.removed_volume_mm3:.2f} mm³")

    def show_removed_overlay(self, volume, z=None, vmin=-200, vmax=800):
        import matplotlib.pyplot as plt
        if z is None:
            counts = np.sum(self.removed_mask, axis=(1, 2))
            z = int(np.argmax(counts))
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
        plt.imshow(self.removed_mask[z], alpha=0.6)
        plt.title(f"Removed plaque-boundary voxels, slice {z}")
        plt.axis("off")
        plt.show()

    def show_refined_overlay(self, volume, z=None, vmin=-200, vmax=800):
        import matplotlib.pyplot as plt
        if z is None:
            counts = np.sum(self.refined_mask == 2, axis=(1, 2))
            z = int(np.argmax(counts))
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
        plt.imshow(self.refined_mask[z] == 2, alpha=0.6)
        plt.title(f"Refined plaque mask, slice {z}")
        plt.axis("off")
        plt.show()


def _structure_for_connectivity(connectivity: int = 26):
    if connectivity <= 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity <= 18:
        return ndi.generate_binary_structure(3, 2)
    return ndi.generate_binary_structure(3, 3)


def remove_small_components(mask, min_voxels=10, plaque_label=2, connectivity=26):
    plaque = mask == plaque_label
    labels, _ = ndi.label(plaque, structure=_structure_for_connectivity(connectivity))
    sizes = np.bincount(labels.ravel())
    keep_labels = np.where(sizes >= min_voxels)[0]
    keep_labels = keep_labels[keep_labels != 0]
    keep = np.isin(labels, keep_labels)
    refined = mask.copy()
    refined[plaque & ~keep] = 0
    return refined


def erode_plaque_boundary(mask, iterations=1, plaque_label=2, connectivity=26):
    plaque = mask == plaque_label
    core = ndi.binary_erosion(
        plaque,
        structure=_structure_for_connectivity(connectivity),
        iterations=iterations,
    )
    refined = mask.copy()
    refined[plaque & ~core] = 0
    return refined


def lumen_adjacent_trim(mask, vessel_label=1, plaque_label=2, distance_voxels=1, connectivity=26):
    plaque = mask == plaque_label
    vessel = mask == vessel_label
    vessel_dilated = ndi.binary_dilation(
        vessel,
        structure=_structure_for_connectivity(connectivity),
        iterations=distance_voxels,
    )
    remove = plaque & vessel_dilated
    refined = mask.copy()
    refined[remove] = 0
    return refined


def intensity_trim(volume, mask, plaque_label=2, high_hu_threshold=None, low_hu_threshold=None):
    plaque = mask == plaque_label
    remove = np.zeros_like(plaque, dtype=bool)
    if high_hu_threshold is not None:
        remove |= plaque & (volume > high_hu_threshold)
    if low_hu_threshold is not None:
        remove |= plaque & (volume < low_hu_threshold)
    refined = mask.copy()
    refined[remove] = 0
    return refined


def close_plaque(mask, plaque_label=2, closing_radius_voxels=0, connectivity=26):
    """Morphological closing of plaque label only; preserves other labels."""
    if closing_radius_voxels <= 0:
        return mask
    plaque = mask == plaque_label
    closed = ndi.binary_closing(
        plaque,
        structure=_structure_for_connectivity(connectivity),
        iterations=int(closing_radius_voxels),
    )
    refined = mask.copy()
    # Add closed voxels as plaque only where background currently exists, avoiding label-1 overwrite.
    refined[(closed & ~plaque) & (refined == 0)] = plaque_label
    return refined


def fill_plaque_holes(mask, plaque_label=2):
    """Fill enclosed holes inside plaque components; preserves other labels."""
    plaque = mask == plaque_label
    filled = ndi.binary_fill_holes(plaque)
    refined = mask.copy()
    refined[(filled & ~plaque) & (refined == 0)] = plaque_label
    return refined


def refine_plaque_mask(
    volume,
    mask,
    spacing,
    remove_small=True,
    min_component_voxels=10,
    trim_lumen_adjacent=True,
    lumen_distance_voxels=1,
    erode_core=False,
    erosion_iterations=1,
    high_hu_threshold=None,
    low_hu_threshold=None,
    closing_radius_voxels=0,
    fill_holes=False,
    connectivity=26,
):
    """Apply configurable experimental refinement steps.

    Parameters searched by the tuning notebook include:
    - min_component_voxels
    - lumen_distance_voxels
    - high_hu_threshold
    - low_hu_threshold
    - closing_radius_voxels
    - fill_holes
    - connectivity

    erode_core/erosion_iterations are normally fixed for the main estimate and
    used separately for conservative core/uncertainty estimates.
    """
    original = mask.copy()
    refined = mask.copy()

    params = {
        "remove_small": remove_small,
        "min_component_voxels": min_component_voxels,
        "trim_lumen_adjacent": trim_lumen_adjacent,
        "lumen_distance_voxels": lumen_distance_voxels,
        "erode_core": erode_core,
        "erosion_iterations": erosion_iterations,
        "high_hu_threshold": high_hu_threshold,
        "low_hu_threshold": low_hu_threshold,
        "closing_radius_voxels": closing_radius_voxels,
        "fill_holes": fill_holes,
        "connectivity": connectivity,
    }

    if closing_radius_voxels and int(closing_radius_voxels) > 0:
        refined = close_plaque(refined, plaque_label=2, closing_radius_voxels=int(closing_radius_voxels), connectivity=int(connectivity))

    if fill_holes:
        refined = fill_plaque_holes(refined, plaque_label=2)

    if remove_small:
        refined = remove_small_components(refined, min_voxels=min_component_voxels, plaque_label=2, connectivity=int(connectivity))

    if trim_lumen_adjacent and int(lumen_distance_voxels) > 0:
        refined = lumen_adjacent_trim(refined, vessel_label=1, plaque_label=2, distance_voxels=int(lumen_distance_voxels), connectivity=int(connectivity))

    if high_hu_threshold is not None or low_hu_threshold is not None:
        refined = intensity_trim(volume, refined, plaque_label=2, high_hu_threshold=high_hu_threshold, low_hu_threshold=low_hu_threshold)

    if erode_core:
        refined = erode_plaque_boundary(refined, iterations=int(erosion_iterations), plaque_label=2, connectivity=int(connectivity))

    removed = (original == 2) & (refined != 2)

    return RefinementResult(
        original_mask=original,
        refined_mask=refined,
        removed_mask=removed,
        spacing=spacing,
        method="component_lumen_intensity_morphology_refinement",
        parameters=params,
    )
