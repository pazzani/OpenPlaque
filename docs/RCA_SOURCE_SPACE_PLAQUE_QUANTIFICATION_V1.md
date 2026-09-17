# RCA Source-Space Plaque Quantification v1

Research-only experiment on the frozen accepted RCA.

## Goal

Move OpenPlaque from nonspatial plaque-vote counts to physical source-CCTA volume measurements in mm³, while being explicit that a validated outer-wall segmentation is not yet available.

## Inputs

- Frozen master anatomy baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- Canonical source-Series-7 RCA centerline from `Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv`.
- Series-7 source CCTA from `Cache/Secondary_3D_Vesselness_Topology_v1`.
- Optional post-hoc comparison to the existing longitudinal plaque-confidence profile and locked RCA PCAT profile.

## Method

The RCA is resampled every 0.5 mm. At each station an orthogonal source-CCTA plane is sampled in 72 directions. A directional lumen radius is estimated from the first sustained drop below a center-HU-derived lumen threshold, with robust angular median filtering and a 0.75-mm directional deviation cap.

Physical wall-shell volume is then integrated in polar coordinates. Four predeclared shell thicknesses are evaluated: 0.75, 1.00, 1.25, and 1.50 mm. The nominal research result uses 1.00 mm; the others are sensitivity analyses.

To reduce perivascular-fat leakage, the composition proxy separates HU < -30 as a `fatlike_excluded` diagnostic. The plaque-proxy bins are:

- low attenuation: -30 to <30 HU
- noncalcified: 30 to <130 HU
- mixed/intermediate: 130 to <350 HU
- calcified: >=350 HU

A raw `<30 HU` volume is also retained so the project can compare directly with its earlier threshold convention.

## Prospective gates

- >=95% of canonical RCA centerline samples must have source HU >=200.
- >=90% of nominal stations must pass lumen-plane geometry QC.
- Longitudinal total-plaque-proxy profile must have Spearman correlation >=0.85 versus every non-nominal shell sensitivity, summarized by the minimum correlation.

Passing status is `RCA_SOURCE_SPACE_PLAQUE_PROXY_ROBUSTNESS_PASS`.

## Scientific boundary

The output is a physical source-space mm³ **plaque-composition proxy**, not validated clinical TPV. The unresolved limitation is the vessel outer wall: v1 uses a sensitivity-tested shell outside an independently estimated lumen boundary rather than a validated patient-specific outer-wall segmentation. The experiment therefore tests whether source-space volume/composition is stable enough to justify the next outer-wall-validation step.

The optional prior longitudinal plaque profile and locked PCAT profile are used only post hoc; they do not guide the source-space quantification.
