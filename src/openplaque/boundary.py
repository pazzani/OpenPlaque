"""
OpenPlaque boundary refinement.

Experimental post-processing for nnU-Net plaque masks.

Defaults are set from boundary-refinement tuning results supplied in
boundary_refinement_case_results.csv.

Research software only. Not for clinical use.
"""

from dataclasses import dataclass
import os
import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk


DEFAULT_REFINEMENT_PARAMS = {'remove_small': True, 'min_component_voxels': 80, 'trim_lumen_adjacent': False, 'lumen_distance_voxels': 0, 'erode_core': False, 'erosion_iterations': 0, 'low_hu_threshold': None, 'high_hu_threshold': None}


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


def remove_small_components(mask, min_voxels=10, plaque_label=2):
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


def lumen_adjacent_trim(mask, vessel_label=1, plaque_label=2, distance_voxels=1):
    if distance_voxels <= 0:
        return mask.copy()
    plaque = mask == plaque_label
    vessel = mask == vessel_label
    vessel_dilated = ndi.binary_dilation(vessel, iterations=distance_voxels)
    remove = plaque & vessel_dilated
    refined = mask.copy()
    refined[remove] = 0
    return refined


def erode_plaque_boundary(mask, iterations=1, plaque_label=2):
    if iterations <= 0:
        return mask.copy()
    plaque = mask == plaque_label
    core = ndi.binary_erosion(plaque, iterations=iterations)
    refined = mask.copy()
    refined[plaque & ~core] = 0
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


def refine_plaque_mask(
    volume,
    mask,
    spacing,
    remove_small=None,
    min_component_voxels=None,
    trim_lumen_adjacent=None,
    lumen_distance_voxels=None,
    erode_core=None,
    erosion_iterations=None,
    high_hu_threshold="DEFAULT",
    low_hu_threshold="DEFAULT",
    use_defaults=True,
):
    """
    Refine a plaque mask.

    By default, this uses DEFAULT_REFINEMENT_PARAMS tuned on the supplied sample
    dataset. Override any parameter by passing it explicitly.
    """
    if use_defaults:
        params = DEFAULT_REFINEMENT_PARAMS.copy()
    else:
        params = {
            "remove_small": True,
            "min_component_voxels": 0,
            "trim_lumen_adjacent": False,
            "lumen_distance_voxels": 0,
            "erode_core": False,
            "erosion_iterations": 0,
            "high_hu_threshold": None,
            "low_hu_threshold": None,
        }

    overrides = {
        "remove_small": remove_small,
        "min_component_voxels": min_component_voxels,
        "trim_lumen_adjacent": trim_lumen_adjacent,
        "lumen_distance_voxels": lumen_distance_voxels,
        "erode_core": erode_core,
        "erosion_iterations": erosion_iterations,
    }
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    if high_hu_threshold != "DEFAULT":
        params["high_hu_threshold"] = high_hu_threshold
    if low_hu_threshold != "DEFAULT":
        params["low_hu_threshold"] = low_hu_threshold

    original = mask.copy()
    refined = mask.copy()

    if params["remove_small"]:
        refined = remove_small_components(
            refined, min_voxels=params["min_component_voxels"], plaque_label=2
        )

    if params["trim_lumen_adjacent"]:
        refined = lumen_adjacent_trim(
            refined,
            vessel_label=1,
            plaque_label=2,
            distance_voxels=params["lumen_distance_voxels"],
        )

    if params["high_hu_threshold"] is not None or params["low_hu_threshold"] is not None:
        refined = intensity_trim(
            volume,
            refined,
            plaque_label=2,
            high_hu_threshold=params["high_hu_threshold"],
            low_hu_threshold=params["low_hu_threshold"],
        )

    if params["erode_core"]:
        refined = erode_plaque_boundary(
            refined, iterations=params["erosion_iterations"], plaque_label=2
        )

    removed = (original == 2) & (refined != 2)

    return RefinementResult(
        original_mask=original,
        refined_mask=refined,
        removed_mask=removed,
        spacing=spacing,
        method="tuned_default_boundary_refinement",
        parameters=params,
    )
