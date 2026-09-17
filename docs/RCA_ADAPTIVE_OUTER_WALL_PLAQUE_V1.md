# RCA Adaptive Outer-Wall Plaque v1

## Question

Does replacing the fixed wall shell with an adaptive, source-CCTA-derived outer-wall boundary improve plaque specificity while preserving the validated RCA source geometry?

## Why this is the next experiment

The cross-vessel lumen-frame refinement experiment repaired the LAD source-plane geometry completely, but the combined refinement reduced the RCA majority-plaque AUROC below its prespecified control threshold and did not rescue LAD plaque specificity. That argues against further centerline/frame tuning as the main solution.

The remaining structural limitation is the use of a fixed shell outside the lumen. A fixed shell inevitably contains normal wall plus variable neighboring tissue. This experiment therefore returns to the original accepted RCA source geometry and changes only the **outer-wall model**.

## Prerequisites

- Frozen baseline commit: \`0593b453959f5a353d644267fbeef24b514ef4d7\`.
- Frozen coronary master remains unchanged.
- The prior fixed-shell RCA excess-specificity experiment must have status \`RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS\`.
- The prespecified RCA 20–50 mm reference region must remain free of >=3/5 ensemble plaque voxels and majority-positive bins.

## Adaptive outer-wall detection

For each accepted RCA source station and each of 72 radial rays:

1. The lumen boundary is measured with the same original directional source-CCTA method used by the successful fixed-shell RCA experiment.
2. Source HU values are sampled radially from the lumen boundary to 2.4 mm outside it.
3. The outer-wall detector first looks for a direct transition into perivascular fat (< -30 HU). This is the strongest boundary evidence.
4. Where no direct fat edge exists, a strong negative HU gradient is allowed as a fallback.
5. Missing angular boundary values are filled by circular interpolation and a 5-ray median regularizer.
6. No plaque labels or plaque-vote maps are used to select an outer boundary.

Three prospectively fixed detector variants are evaluated:

- strict: minimum gradient drop 100 HU, post-edge HU <=80, max thickness 1.8 mm;
- nominal: minimum gradient drop 70 HU, post-edge HU <=130, max thickness 2.0 mm;
- liberal: minimum gradient drop 45 HU, post-edge HU <=180, max thickness 2.2 mm.

The nominal variant is the primary analysis.

## Station outer-wall QC

A station passes adaptive-wall QC only if:

- the original lumen QC passes;
- at least 25% of angular rays have direct/fallback outer-edge detections;
- at least 10% of rays show a direct transition to perivascular fat;
- the interpolated wall-thickness IQR is <=1.10 mm.

The overall nominal station-QC fraction must be >=0.80 and the median direct-fat edge fraction must be >=0.25.

## Plaque excess model

The HU composition is measured only between the directional lumen boundary and the adaptive outer wall:

- fat-like excluded: < -30 HU;
- low attenuation: -30 to <30 HU;
- noncalcified: 30 to <130 HU;
- mixed/intermediate: 130 to <350 HU;
- calcified: >=350 HU.

As before, a robust normal-wall composition model is fit only in the prespecified RCA 20–50 mm reference region. The model includes lumen radius, lumen radius squared, center HU, and median adaptive wall thickness. A 90th-percentile reference residual threshold is used to define positive excess. The model is then evaluated in the disjoint 0–20 mm RCA holdout against the established plaque-vote profile.

## Prospective gates

The nominal adaptive-wall result passes specificity only if:

- majority-positive vs vote-free AUROC >=0.80;
- strict 5/5 vs vote-free AUROC >=0.90 when at least two strict bins exist;
- median excess in majority-positive bins is >=2x the vote-free median;
- minimum strict/liberal profile Spearman correlation versus nominal is >=0.80.

If those gates pass and majority AUROC improves by at least 0.02 relative to the successful fixed 1.0-mm shell experiment, status is \`RCA_ADAPTIVE_OUTER_WALL_SPECIFICITY_GAIN\`. If gates pass without that gain, status is \`RCA_ADAPTIVE_OUTER_WALL_PASS_NO_CLEAR_GAIN\`.

## Outputs

The run writes:

- per-station adaptive outer-wall measurements for all three variants;
- nominal per-ray outer-wall diagnostics;
- robust reference-model coefficients;
- 1-mm longitudinal plaque-excess profiles;
- holdout validation tables;
- cross-variant consistency;
- adaptive outer-wall source-CCTA QC figures;
- optional fusion with the locked RCA PCAT profile;
- summary/run-state JSON, HTML report, and ZIP archive.

## Scientific boundary

This remains a developmental single-scan experiment. Although the outer-wall detector itself is label-blind, this RCA has already been used during method development. Passing the holdout gates would support the adaptive-wall method as a better research model, not constitute independent external validation or validated clinical TPV.
