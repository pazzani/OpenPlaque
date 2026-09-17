# Cross-Vessel Lumen Frame Refinement Plaque Rescue v1

## Question

The first RCA-trained plaque-excess transfer to the LAD failed for two distinct reasons: the source-plane lumen geometry had inadequate QC on the LAD, and the transferred plaque-specificity metrics did not reproduce the RCA performance. This experiment asks whether a **label-blind source-geometry refinement** can repair the lumen frames without changing plaque labels, HU bins, shell widths, or specificity gates.

## Prerequisites

- Frozen baseline commit: \`0593b453959f5a353d644267fbeef24b514ef4d7\`.
- RCA excess-specificity status: \`RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS\`.
- Prior LAD transfer status: \`LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATION_FAILED\`.
- Distal LAD extension status: \`LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED\`.
- Frozen master is not modified.

## Why this experiment

The failed LAD transfer had frozen-LAD station QC 0.865, distal-extension QC 0.467, minimum cross-shell Spearman 0.640, majority-vs-vote-free AUROC 0.595, and strict-5/5 AUROC 0.643. The station failures were dominated by the radial p90/p10 lumen-axis proxy, while center HU and radial coverage remained strong. That pattern is consistent with frame/tangent and centering error rather than loss of coronary source signal.

## Prospective method

No plaque labels are used to choose the new frame.

1. Resample the accepted RCA and research LAD paths at 0.5 mm.
2. Replace pointwise finite-difference tangents with a local PCA tangent over ±3 stations (about ±1.5 mm).
3. In each orthogonal plane, test 25 constrained candidate centers: the original center plus 8 directions at radii 0.25, 0.50, and 0.75 mm.
4. Select the center using source-CCTA lumen geometry only: radial boundary coverage, circularity, center HU, and an offset penalty.
5. Retain the same full-resolution radial lumen boundary, HU bins, shell thicknesses, 2.5 axis-proxy QC limit, RCA 20–50 mm reference zone, 90th-percentile residual normalization, and plaque-specificity gates.
6. Recompute the RCA first. The identical refined geometry must preserve the RCA positive control before any LAD inference is allowed.
7. Fit the refined normal-wall model only on the RCA 20–50 mm reference zone and transfer it unchanged to the LAD.

## Gates

RCA control:

- station QC ≥ 0.98;
- majority-positive vs vote-free AUROC ≥ 0.80;
- strict 5/5 vs vote-free AUROC ≥ 0.90 when at least two strict bins exist;
- positive/negative median excess ratio ≥ 2.0;
- minimum cross-shell profile Spearman ≥ 0.80.

LAD developmental transfer:

- frozen-LAD station QC ≥ 0.90;
- confirmed distal-extension station QC ≥ 0.85;
- majority-positive vs vote-free AUROC ≥ 0.80;
- strict 5/5 vs vote-free AUROC ≥ 0.90 when at least two strict bins exist;
- positive/negative median excess ratio ≥ 2.0;
- minimum cross-shell profile Spearman ≥ 0.80.

## Interpretation boundary

The geometry refinement itself is label-blind. However, the LAD plaque labels have already been inspected during development, so any improved LAD agreement in this experiment is **developmental**, not untouched external validation. The output remains a reference-normalized excess-wall research proxy, not independently segmented outer-wall TPV and not validated clinical TPV.

A positive result would justify locking the refined geometry algorithm and seeking validation on an independent scan or truly untouched vessel/dataset. A negative result would indicate that the main limitation is not frame geometry and would redirect development toward vessel-neutral outer-wall modeling rather than further threshold tuning.
