# Run optimized boundary refinement on new data

This workflow applies a previously optimized boundary-refinement parameter set to a new CCTA study.

## Why separate from tuning?

Parameter tuning uses labeled sample data. New patient/study inference should not retune parameters on the target study. It should load the selected parameters and apply them directly.

## JSON format

The code reads the current best-parameters JSON exactly as produced by the tuning notebooks:

```json
{
  "final_parameters_selected_on_all_cases": { ... }
}
```

The parameter dictionary under that key is passed to `refine_plaque_mask` after type normalization.

## New-data pipeline

1. Clone/pull OpenPlaque from GitHub.
2. Mount Google Drive.
3. Extract trained nnU-Net model.
4. Load `best_boundary_parameters_bayesian.json` or `best_boundary_parameters.json`.
5. Load the new DICOM ZIP.
6. Detect LAD/RCA/LCX curved reformats, with UCLA fallback series.
7. Run nnU-Net once per artery.
8. Apply optimized boundary refinement.
9. Save TPV CSV, refined masks, overlays, and HTML report.
