# OpenPlaque repository additions: run optimized boundary refinement on new data

This package adds files for applying already-selected boundary-refinement parameters to a new CCTA study, such as the UCLA case.

It does **not** tune parameters. It consumes the current tuning JSON format directly:

```json
{
  "final_parameters_selected_on_all_cases": {
    "min_component_voxels": 25,
    "lumen_distance_voxels": 1,
    "high_hu_threshold": null,
    "low_hu_threshold": null,
    "closing_radius_voxels": 1,
    "fill_holes": true,
    "min_plaque_length_mm": 2.0,
    "connectivity": 26,
    "adaptive_hu_thresholds": false,
    "erode_core": false,
    "erosion_iterations": 1
  }
}
```

No extended metadata wrapper is required.

## Files

- `src/openplaque/run_new_data.py` — helpers to load current-format best-parameter JSON, apply optimized refinement, save masks/CSV/HTML.
- `notebooks/10_Run_Optimized_Boundary_On_New_Data_GitHub.ipynb` — clean Colab notebook for UCLA/new-data inference.
- `tests/test_run_new_data.py` — syntax/format smoke tests.

## Expected Drive inputs

```text
/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/best_boundary_parameters_bayesian.json
```

The notebook also checks for `best_boundary_parameters.json` and alternate `Boundary_Tuning` locations.

## Outputs

```text
/content/drive/MyDrive/OpenPlaque/New_Data_Optimized_Boundary/
  new_data_tpv_summary.csv
  new_data_optimized_tpv_report.html
  masks/*.nii.gz
  overlays/*.png
```

Research use only. Not clinically validated. Not for diagnosis or medical decision-making.
