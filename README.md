# OpenPlaque Repository Additions: Bayesian Boundary Parameter Tuning

This repository-ready package adds an Optuna Bayesian-optimization workflow for OpenPlaque boundary-refinement parameters.

## Main change

Instead of brute-forcing the full expanded grid, this version:

1. runs nnU-Net **once per labeled sample case**,
2. saves predictions to a reusable compressed Google Drive archive,
3. restores predictions from that archive in later clean Colab runs,
4. uses Optuna TPESampler Bayesian/sequential optimization to search downstream boundary-refinement parameters,
5. evaluates each trial on **all labeled sample cases**.

No cross-validation and no bootstrap are used in this version.

## Notebook

```text
notebooks/09_Bayesian_Boundary_Parameter_Tuning_CachedPredictions_GitHub.ipynb
```

The notebook clones/pulls OpenPlaque from GitHub and expects these additions to be committed to the repository.

## Source additions

```text
src/openplaque/boundary.py
src/openplaque/boundary_parameter_tuning.py
```

## Parameters optimized

Bayesian search space:

- `min_component_voxels`: `[1, 5, 10, 25, 50, 100]`
- `lumen_distance_voxels`: integer `0..3`
- `high_hu_threshold`: `[None, 650, 700, 850, 1000, 1200]`
- `low_hu_threshold`: `[None, -150, -100, -50, 0]`
- `closing_radius_voxels`: integer `0..2`
- `fill_holes`: `[False, True]`
- `min_plaque_length_mm`: `[0, 1, 2, 3, 5]`
- `connectivity`: `[6, 18, 26]`
- `adaptive_hu_thresholds`: `[False, True]`
- `erode_core`: fixed `False`
- `erosion_iterations`: fixed `1`

## Outputs

Saved to Google Drive under:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning_Bayesian/
```

Includes:

- `nnunet_prediction_cache.zip`
- `bayesian_trial_case_results.csv`
- `bayesian_trial_summary.csv`
- `best_boundary_parameters_bayesian.json`
- `bayesian_boundary_parameter_tuning_report.html`

## Scoring

The supervised score is:

```text
0.35*Dice + 0.20*IoU + 0.20*(1 - min(abs TPV error fraction, 1)) + 0.15*Precision + 0.10*Recall
```

Research use only. Not clinically validated.
