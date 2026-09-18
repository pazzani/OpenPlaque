# Alternate-Series Left Coronary Origin Validation v1

This experiment asks whether another CT series from the same CCTA exam provides genuinely new evidence about the unresolved proximal left-coronary origin.

## Scientific role

The frozen Series 7 anatomy remains the reference. The experiment does **not** retune same-series vesselness or geodesic heuristics.

Instead it:

1. inventories every DICOM series under the same exam;
2. identifies Series 7 from its frozen source geometry;
3. ranks alternate CT series using actual DICOM metadata;
4. verifies that candidate series contain bright contrast-enhanced coronary signal along the frozen LAD/RCA paths;
5. selects up to four informative alternate phases/reconstructions automatically;
6. aligns each alternate series to Series 7 with a translation-only root-region registration;
7. runs the independent ImageCAS-X CAS-Net coronary-lumen model on Series 7 and each selected alternate series;
8. adjudicates aortic contact clusters with the RCA ostium explicitly separated from any left-root candidate.

No sliders, manual coordinates, or post-hoc threshold changes are allowed.

## Fixed baseline

Fresh branch directly from:

`0593b453959f5a353d644267fbeef24b514ef4d7`

The frozen master is never modified by this experiment.

## Candidate-series ranking

Metadata scoring favors:

- CT modality;
- 512x512 source geometry;
- many source slices;
- <=0.6 mm in-plane spacing;
- <=0.8 mm slice spacing;
- ORIGINAL/PRIMARY and coronary/cardiac descriptors.

Scout/localizer, calcium-score, bolus-tracking, noncontrast and derived MPR/MIP/VR/CPR series are penalized.

The top metadata candidates are then actually loaded and sampled along the frozen LAD and RCA. A candidate must cover the paths and show contrast-enhanced lumen signal before it can be selected.

## Prespecified interpretation gates

For each selected alternate series:

- registration translation <= 6 mm;
- RCA model support >= 0.60;
- a distinct RCA aortic-contact cluster must be within 5 mm of the frozen RCA root;
- LAD support >= 0.60;
- C6 support >= 0.50;
- LAD and C6 must share a predicted coronary component;
- a left candidate must be a **separate aortic-contact cluster**, at least 8 mm from the RCA root, within 25 mm of the frozen proximal LAD anchor, with >=10 contact voxels.

A left-root candidate reproduced across two alternate series within 5 mm is called cross-series concordant.

## Possible overall statuses

- `ALTERNATE_SERIES_LEFT_OSTIUM_CONCORDANT`
- `SINGLE_ALTERNATE_SERIES_LEFT_OSTIUM_CANDIDATE`
- `ALTERNATE_SERIES_LEFT_CONTACTS_DISCORDANT`
- `NO_DISTINCT_LEFT_OSTIUM_ACROSS_ALTERNATE_SERIES`
- `INSUFFICIENT_INTERPRETABLE_ALTERNATE_SERIES`

Even a concordant result does not automatically create a clinical LM/LCX/OM label or change the frozen master. It reopens scientific review with genuinely new same-exam phase/reconstruction evidence.

## Outputs

Written to:

`/content/drive/MyDrive/OpenPlaque/Alternate_Series_Left_Coronary_Origin_Validation_v1/`

Key outputs include:

- `series_inventory.csv`
- `candidate_ranking.csv`
- `registration_summary.csv`
- `preparation.json`
- `path_support_by_series.csv`
- `aorta_contact_clusters.csv`
- `gates_by_series.csv`
- `series_decisions.csv`
- automatic root-plane QC PNGs for every analyzed series
- HTML report
- `summary.json`, `decision.json`, `run_state.json`
- ZIP archive

Research use only. Not for clinical diagnosis.
