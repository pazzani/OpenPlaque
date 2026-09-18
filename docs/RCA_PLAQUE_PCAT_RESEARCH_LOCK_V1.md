# RCA Plaque + PCAT Research Lock v1

## Purpose

This is a consolidation experiment, not another plaque-method tuning experiment.

Two more anatomical outer-wall models were tested after the successful fixed-shell/reference-normalized RCA plaque experiment:

- independent per-ray adaptive outer-wall detection;
- globally regularized outer-wall surface estimation.

Both models produced plausible source-CCTA wall geometry but failed to improve plaque specificity. Therefore further source-HU-only outer-wall tuning on this same scan is retired.

The best-performing developmental RCA plaque method is locked as the research benchmark:

- original accepted RCA source geometry;
- 1.0-mm fixed shell outside the directional lumen boundary;
- research HU composition bins;
- robust normal-wall model trained only in the prespecified RCA 20–50 mm reference zone;
- 90th-percentile reference residual normalization;
- evaluation in the disjoint RCA 0–20 mm holdout.

## Required prior statuses

- Frozen master: \`CORONARY_ANATOMY_BASELINE_V2_FROZEN\`.
- Fixed-shell RCA plaque specificity: \`RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS\`.
- Per-ray adaptive wall: \`RCA_ADAPTIVE_OUTER_WALL_SPECIFICITY_FAILED\`.
- Global wall surface: \`RCA_GLOBAL_OUTER_WALL_SURFACE_SPECIFICITY_FAILED\`.

No master anatomy is changed.

## What is locked

The research plaque benchmark is the existing fixed-shell/reference-normalized result. This experiment does not refit it.

The lock records:

- nominal total plaque-excess proxy volume;
- HU composition;
- shell-sensitivity envelope across 0.75, 1.0, 1.25, and 1.5 mm;
- holdout specificity performance;
- longitudinal 1-mm plaque profile;
- top plaque-excess bins.

The established RCA PCAT measurement is fused on the same longitudinal coordinate system:

- RCA arc 10–50 mm;
- wall margin 0.75 mm;
- fat HU window -190 to -30 HU;
- direct mean attenuation, not proprietary FAI.

The report includes fat-voxel-weighted mean PCAT attenuation and the longitudinal plaque + PCAT profile.

## Scientific interpretation

The lock does **not** convert the plaque proxy into clinical TPV.

The locked quantity remains a source-space, reference-normalized excess-wall proxy. It is selected because it outperformed two more anatomically explicit outer-wall models on the predeclared specificity tests, not because it has a validated histologic or expert-contoured outer wall.

PCAT remains a direct source-CCTA attenuation measurement rather than proprietary FAI.

## Method-development boundary

No additional outer-wall threshold/smoothness tuning on this same scan should be used to claim improvement.

Future serious improvement should come from one of:

1. independent expert outer-wall contours;
2. a separately trained vessel-wall segmentation model;
3. an independent scan/dataset for validation.

The locked RCA benchmark can meanwhile be used for research reporting, longitudinal software regression testing, and integrated plaque + PCAT visualization.
