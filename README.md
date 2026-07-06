# OpenPlaque Repository Additions: Supervised Boundary Parameter Tuning with Expanded Grid

This release replaces 5-fold CV with full labeled-sample parameter evaluation.

## Why no cross-validation?

Boundary refinement is deterministic post-processing after a fixed nnU-Net model. The efficient and appropriate workflow is:

1. Run nnU-Net once per labeled sample case.
2. Cache the predictions locally.
3. Save the predictions as a compressed ZIP archive on Google Drive.
4. On later clean Colab runs, restore predictions from that archive and skip nnU-Net inference.
5. Evaluate every boundary-refinement parameter set on all cached predictions.
4. Compare each refined mask against the expert label.
5. Select the parameter set with the best average supervised metrics.

No bootstrap is included.

## Files

```text
src/openplaque/boundary.py
src/openplaque/boundary_parameter_tuning.py
notebooks/08_Boundary_Parameter_Tuning_CachedPredictions_GitHub.ipynb
docs/boundary_parameter_tuning_cached_predictions.md
tests/test_boundary_parameter_tuning.py
```

## Parameter grid

Default grid:

```python
{
    "min_component_voxels": [1, 5, 10, 25, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None, 700, 850, 1000],
    "low_hu_threshold": [None, -100, -50],
    "closing_radius_voxels": [0, 1, 2],
    "fill_holes": [False, True],
    "min_plaque_length_mm": [0, 1, 2, 3],
    "connectivity": [6, 18, 26],
    "adaptive_hu_thresholds": [False, True],
    "erode_core": [False],
    "erosion_iterations": [1],
}
```

Total: 25,920 parameter combinations.

For a fast smoke test, the notebook also exposes `SMALL_GRID`.

## Outputs

The notebook saves:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/nnunet_prediction_cache.zip
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/all_case_parameter_results.csv
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/parameter_summary.csv
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/best_boundary_parameters.json
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/boundary_parameter_tuning_report.html
```

Research use only. Not clinically validated.


## Prediction archive behavior

The notebook now uses:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/nnunet_prediction_cache.zip
```

Workflow:

1. If local `predictions/` already contains all expected masks, reuse it.
2. Else, if `nnunet_prediction_cache.zip` exists, restore predictions from it.
3. Else, run `nnUNetv2_predict` once per sample case batch.
4. After prediction generation, write/update the compressed archive in Google Drive.

Set `OVERWRITE_PREDICTIONS = True` in the notebook only when intentionally regenerating predictions.
