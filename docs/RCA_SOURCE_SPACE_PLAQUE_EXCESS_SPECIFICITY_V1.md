# RCA Source-Space Plaque Excess Specificity v1

## Purpose

The preceding RCA source-space plaque-quantification experiment established stable Series-7 source-space resampling and strong geometric QC, but its fixed outer shell behaved mainly as a **wall/tissue volume proxy**, not yet as plaque-specific clinical TPV. The nominal 1.0-mm shell produced a large background volume along most of the RCA and correlated only weakly with the previously validated longitudinal 5-fold plaque-vote profile.

This experiment tests whether a prespecified **reference-normalized excess-wall metric** can suppress that normal-wall background and recover plaque-localized source-space signal without fitting on the evaluation region.

## Prospective design

- Branch is created directly from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- Prerequisite: `RCA_Source_Space_Plaque_Quantification_v1` must have status `RCA_SOURCE_SPACE_PLAQUE_PROXY_ROBUSTNESS_PASS`.
- Input source-space station measurements are reused; source CCTA is not recomputed.
- The normal-wall reference zone is fixed at RCA arc **20-50 mm**.
- The disjoint evaluation/holdout zone is fixed at **0-20 mm**.
- The reference zone must contain no prior 3+/5-vote voxels and no `majority_3plus_signal` bins; otherwise the experiment stops.
- For each shell thickness (0.75, 1.0, 1.25, 1.5 mm) and each HU component, a robust Huber model predicts normal component fraction from lumen radius, lumen-radius squared, and center HU using only reference-zone stations.
- A one-sided 90th-percentile reference residual is added to the expected fraction. Only source-space volume above that threshold contributes to the **excess** metric.
- Composition bins remain those from v1: low attenuation -30 to <30 HU, noncalcified 30 to <130 HU, mixed/intermediate 130 to <350 HU, calcified >=350 HU. Tissue < -30 HU remains excluded as fatlike background.

## Holdout validation

Within 0-20 mm:

- positive bins: prior `majority_3plus_signal=True`;
- strict positives: prior `strict_5of5_signal=True`;
- negative controls: `mapped_native_vote_sum==0`;
- ambiguous low-vote bins are not used in the majority-vs-negative ROC calculation.

The nominal 1.0-mm-shell excess metric passes only if:

1. majority-vs-vote-free AUROC >= 0.80;
2. strict-5/5-vs-vote-free AUROC >= 0.90;
3. median positive excess / median negative excess >= 2.0;
4. minimum excess-profile Spearman correlation between the nominal shell and the non-nominal shell sensitivities >= 0.80.

## Scientific boundary

This remains a research source-space **plaque excess proxy**. It is explicitly not a validated clinical TPV measurement because the outer coronary wall is still not independently segmented. A pass means that the source-space wall signal has materially improved plaque specificity relative to the raw fixed shell, not that clinical plaque volume is established.

The locked RCA PCAT profile is merged after the plaque metric is frozen, providing a plaque/inflammation longitudinal table without changing the established PCAT method.
