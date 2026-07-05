# Boundary refinement example

```python
from openplaque.boundary import refine_plaque_mask

# Example using an existing SegmentationReport
result = refine_plaque_mask(
    volume=lad_report.volume,
    mask=lad_report.mask,
    spacing=lad_report.mask_image.GetSpacing(),
    remove_small=True,
    min_component_voxels=10,
    trim_lumen_adjacent=True,
    lumen_distance_voxels=1,
    erode_core=False,
)

result.summary()
result.show_refined_overlay(lad_report.volume)
result.show_removed_overlay(lad_report.volume)
```

## More conservative high-confidence plaque core

```python
core = refine_plaque_mask(
    volume=lad_report.volume,
    mask=lad_report.mask,
    spacing=lad_report.mask_image.GetSpacing(),
    remove_small=True,
    min_component_voxels=10,
    trim_lumen_adjacent=True,
    lumen_distance_voxels=1,
    erode_core=True,
    erosion_iterations=1,
)

core.summary()
core.show_refined_overlay(lad_report.volume)
```

## Experimental high-HU trim

Use cautiously on contrast CCTA. High HU may represent calcium or contrast-filled lumen.

```python
trimmed = refine_plaque_mask(
    volume=lad_report.volume,
    mask=lad_report.mask,
    spacing=lad_report.mask_image.GetSpacing(),
    high_hu_threshold=900,
)
```
