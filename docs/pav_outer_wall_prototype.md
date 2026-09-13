# Experimental PAV / outer-wall prototype

This branch adds a deliberately isolated research prototype for estimating percent atheroma volume (PAV) from OpenPlaque outputs plus the original CCTA volume.

## Definition

PAV is computed as:

`100 * plaque volume / outer-vessel volume`

The denominator is intended to approximate the volume enclosed by the outer coronary vessel wall. It is **not** the lumen volume and should not be computed as plaque / (label 1 + plaque) unless label 1 is known to represent the full outer-wall compartment.

## Prototype method

`openplaque.pav.estimate_outer_wall_candidate`:

1. Uses existing label 1 (vessel/lumen) and label 2 (plaque) voxels as an anatomical seed.
2. Expands from the seed using physical distance in millimeters.
3. Rejects newly added voxels below a configurable HU threshold to reduce expansion into epicardial fat.
4. Preserves all existing plaque voxels, including low-attenuation plaque.
5. Keeps only components touching the seed and optionally applies mild closing and slice-wise hole filling.

This creates a **candidate outer-wall envelope for visual inspection**. It is not a validated EEM segmentation and should not be interpreted clinically without manual/independent validation.

## Example

```python
import SimpleITK as sitk
from openplaque.pav import estimate_pav_from_labels, show_pav_overlay

ct_img = sitk.ReadImage("artery_ct.nii.gz")
mask_img = sitk.ReadImage("artery_mask.nii.gz")

volume = sitk.GetArrayFromImage(ct_img)
mask = sitk.GetArrayFromImage(mask_img)
spacing = ct_img.GetSpacing()  # (x, y, z)

result = estimate_pav_from_labels(
    volume,
    mask,
    spacing,
    max_wall_thickness_mm=2.0,
    fat_threshold_hu=-30.0,
)

result.summary()
show_pav_overlay(volume, mask, result.outer_wall_mask)
```

## Validation plan

Before using reported PAV values:

- inspect representative proximal, mid, and distal cross-sections for LAD, LCX, and RCA;
- compare candidate contours with manually drawn outer-wall contours on a small validation set;
- tune the physical expansion limit and HU threshold against those manual contours;
- quantify Dice/IoU and outer-vessel-volume error;
- only then calculate vessel-specific and whole-heart PAV.

The current implementation is intentionally simple so that every step can be inspected and replaced as better outer-wall logic becomes available.
