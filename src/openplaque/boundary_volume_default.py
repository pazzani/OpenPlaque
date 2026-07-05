"""
OpenPlaque volume-based boundary refinement.

Default parameters are from volume-based tuning:

DEFAULT_VOLUME_REFINEMENT_PARAMS = {'min_component_volume_mm3': 0.0, 'lumen_distance_voxels': 1, 'low_hu_threshold': None, 'high_hu_threshold': None}

Research software only. Not for clinical use.
"""

import os
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


DEFAULT_VOLUME_REFINEMENT_PARAMS = {'min_component_volume_mm3': 0.0, 'lumen_distance_voxels': 1, 'low_hu_threshold': None, 'high_hu_threshold': None}


def voxel_volume_mm3(spacing):
    return float(np.prod(spacing))


def min_volume_to_voxels(min_component_volume_mm3, spacing):
    if min_component_volume_mm3 is None or min_component_volume_mm3 <= 0:
        return 0
    return int(np.ceil(float(min_component_volume_mm3) / voxel_volume_mm3(spacing)))


def remove_small_components_by_volume(mask, spacing, min_component_volume_mm3=0.0, plaque_label=2):
    min_voxels = min_volume_to_voxels(min_component_volume_mm3, spacing)
    if min_voxels <= 0:
        return mask.copy()

    plaque = mask == plaque_label
    labels, n = ndi.label(plaque)
    if n == 0:
        return mask.copy()

    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(plaque, dtype=bool)

    for label_id, size in enumerate(sizes):
        if label_id != 0 and size >= min_voxels:
            keep |= labels == label_id

    refined = mask.copy()
    refined[plaque & ~keep] = 0
    return refined


def lumen_adjacent_trim(mask, vessel_label=1, plaque_label=2, distance_voxels=0):
    if distance_voxels <= 0:
        return mask.copy()

    plaque = mask == plaque_label
    vessel = mask == vessel_label
    vessel_dilated = ndi.binary_dilation(vessel, iterations=distance_voxels)

    refined = mask.copy()
    refined[plaque & vessel_dilated] = 0
    return refined


def intensity_trim(volume, mask, plaque_label=2, low_hu_threshold=None, high_hu_threshold=None):
    plaque = mask == plaque_label
    remove = np.zeros_like(plaque, dtype=bool)

    if low_hu_threshold is not None:
        remove |= plaque & (volume < low_hu_threshold)
    if high_hu_threshold is not None:
        remove |= plaque & (volume > high_hu_threshold)

    refined = mask.copy()
    refined[remove] = 0
    return refined


@dataclass
class VolumeRefinementResult:
    original_mask: np.ndarray
    refined_mask: np.ndarray
    removed_mask: np.ndarray
    spacing: tuple
    parameters: dict

    @property
    def voxel_volume_mm3(self):
        return voxel_volume_mm3(self.spacing)

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
        print("Volume-based boundary refinement")
        print("Parameters:", self.parameters)
        print(f"Original TPV: {self.original_tpv_mm3:.2f} mm³")
        print(f"Refined TPV:  {self.refined_tpv_mm3:.2f} mm³")
        print(f"Removed vol:  {self.removed_volume_mm3:.2f} mm³")
        print(f"Original plaque voxels: {self.original_plaque_voxels}")
        print(f"Refined plaque voxels:  {self.refined_plaque_voxels}")

    def show_refined_overlay(self, volume, z=None, vmin=-200, vmax=800):
        import matplotlib.pyplot as plt
        if z is None:
            counts = np.sum(self.refined_mask == 2, axis=(1, 2))
            z = int(np.argmax(counts))
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
        plt.imshow(self.refined_mask[z] == 2, alpha=0.6)
        plt.axis("off")
        plt.title(f"Refined plaque mask, slice {z}")
        plt.show()

    def show_removed_overlay(self, volume, z=None, vmin=-200, vmax=800):
        import matplotlib.pyplot as plt
        if z is None:
            counts = np.sum(self.removed_mask, axis=(1, 2))
            z = int(np.argmax(counts))
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
        plt.imshow(self.removed_mask[z], alpha=0.6)
        plt.axis("off")
        plt.title(f"Removed plaque voxels, slice {z}")
        plt.show()

    def save_refined_mask(self, reference_image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = sitk.GetImageFromArray(self.refined_mask.astype("uint8"))
        img.CopyInformation(reference_image)
        sitk.WriteImage(img, path)
        print("Saved:", path)


def refine_plaque_mask_volume_based(
    volume,
    mask,
    spacing,
    min_component_volume_mm3=None,
    lumen_distance_voxels=None,
    low_hu_threshold="DEFAULT",
    high_hu_threshold="DEFAULT",
    use_defaults=True,
):
    """
    Apply volume-based boundary refinement.

    Defaults:
      min_component_volume_mm3 = 0.0
      lumen_distance_voxels = 1
      low_hu_threshold = None
      high_hu_threshold = None
    """
    if use_defaults:
        params = DEFAULT_VOLUME_REFINEMENT_PARAMS.copy()
    else:
        params = {
            "min_component_volume_mm3": 0.0,
            "lumen_distance_voxels": 0,
            "low_hu_threshold": None,
            "high_hu_threshold": None,
        }

    if min_component_volume_mm3 is not None:
        params["min_component_volume_mm3"] = min_component_volume_mm3
    if lumen_distance_voxels is not None:
        params["lumen_distance_voxels"] = lumen_distance_voxels
    if low_hu_threshold != "DEFAULT":
        params["low_hu_threshold"] = low_hu_threshold
    if high_hu_threshold != "DEFAULT":
        params["high_hu_threshold"] = high_hu_threshold

    original = mask.copy()
    refined = mask.copy()

    refined = remove_small_components_by_volume(
        refined,
        spacing=spacing,
        min_component_volume_mm3=params["min_component_volume_mm3"],
        plaque_label=2,
    )

    refined = lumen_adjacent_trim(
        refined,
        vessel_label=1,
        plaque_label=2,
        distance_voxels=params["lumen_distance_voxels"],
    )

    refined = intensity_trim(
        volume,
        refined,
        plaque_label=2,
        low_hu_threshold=params["low_hu_threshold"],
        high_hu_threshold=params["high_hu_threshold"],
    )

    removed = (original == 2) & (refined != 2)

    return VolumeRefinementResult(
        original_mask=original,
        refined_mask=refined,
        removed_mask=removed,
        spacing=spacing,
        parameters=params,
    )
