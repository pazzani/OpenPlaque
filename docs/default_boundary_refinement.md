# Default Boundary Refinement

Default parameters derived from uploaded `boundary_refinement_case_results.csv`.

```python
DEFAULT_REFINEMENT_PARAMS = {'remove_small': True, 'min_component_voxels': 80, 'trim_lumen_adjacent': False, 'lumen_distance_voxels': 0, 'erode_core': False, 'erosion_iterations': 0, 'low_hu_threshold': None, 'high_hu_threshold': None}
```

Validation summary for selected default:

min_component_voxels             80.000000
lumen_distance_voxels             0.000000
erosion_iterations                0.000000
low_hu_threshold                       NaN
high_hu_threshold                      NaN
mean_dice                         0.753003
median_dice                       0.804545
mean_iou                          0.621273
mean_precision                    0.726027
mean_recall                       0.831407
mean_abs_volume_error_mm3      1179.743590
median_abs_volume_error_mm3     557.000000
mean_score                        0.608462
n_cases                          39.000000
