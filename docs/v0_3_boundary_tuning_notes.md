# OpenPlaque v0.3 boundary tuning notes

This release adds unsupervised parameter tuning for experimental plaque-boundary refinement.

## What is tuned

The tuner evaluates combinations of:

- `min_component_voxels`
- `lumen_distance_voxels`
- `high_hu_threshold`
- `low_hu_threshold`
- `erode_core`
- `erosion_iterations`

## Scoring

Because no manual expert contours are included, the score is heuristic. It favors settings that:

- remove a moderate amount of likely boundary/noise plaque
- avoid deleting most plaque
- avoid doing nothing
- reduce fragmented tiny components
- preserve a dominant connected component
- pass simple HU sanity checks

This is not a clinical validation method. Use visual QC and, ideally, expert annotation.

## Outputs

The tuning notebook writes:

- `boundary_tuning_results.csv`
- `boundary_tuning_results.json`
- `boundary_tuning_report.html`
- `openplaque_tuned_tpv_report.html`
