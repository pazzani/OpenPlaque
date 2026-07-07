# OpenPlaque UCLA Plaque Characterization Additions

Repository-ready additions for running optimized boundary refinement and plaque-type quantification on a new CCTA study, including the UCLA `Full_DICOM.zip` workflow.

## Main notebook

```text
notebooks/11_Run_UCLA_Plaque_Characterization_Clean_Colab.ipynb
```

The notebook runs from a fresh Colab runtime and:

1. Mounts Google Drive.
2. Clones/updates OpenPlaque from GitHub.
3. Optionally overlays the files in this ZIP if they are not yet merged into GitHub.
4. Loads the optimized boundary parameters from the **current** JSON format:

```json
{
  "final_parameters_selected_on_all_cases": {
    "min_component_voxels": 50,
    "lumen_distance_voxels": 0,
    "...": "..."
  }
}
```

No extended metadata wrapper is required.

## Inputs expected in Google Drive

```text
/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/best_boundary_parameters.json
```

The notebook also checks these fallback parameter locations:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Tuning/best_boundary_parameters.json
/content/drive/MyDrive/OpenPlaque/best_boundary_parameters.json
```

## Outputs

Written to:

```text
/content/drive/MyDrive/OpenPlaque/UCLA_Plaque_Characterization/
```

Key outputs:

```text
optimized_tpv_summary.csv
plaque_type_composition_summary.csv
plaque_type_lesion_summary.csv
optimized_boundary_tpv_report.html
plaque_characterization_report.html
masks/*_optimized_refined.nii.gz
masks/*_plaque_type_class_map.nii.gz
overlays/*_plaque_type_overlay.png
```

## Plaque type classes

Default HU thresholds:

| Class | HU range |
|---|---:|
| Low attenuation | < 30 |
| Non-calcified | 30 to <130 |
| Mixed/intermediate | 130 to <350 |
| Calcified | >=350 |

These thresholds are configurable in the notebook.

## Files added

```text
src/openplaque/plaque_characterization.py
src/openplaque/run_characterization_new_data.py
notebooks/11_Run_UCLA_Plaque_Characterization_Clean_Colab.ipynb
tests/test_plaque_characterization.py
```

Research use only. Not clinically validated. Not for diagnosis or treatment decisions.
