# Left-coronary backbone branch discovery v1

## Purpose

The preceding through-vessel continuity experiment established that the frozen LAD and the validated proximal/common-trunk segment behave as one continuous source-CCTA vessel and that the prior 30-mm parent candidate substantially re-entered known LAD. This experiment therefore stops using a presumed LM direction and instead asks a label-neutral topologic question: **where are the source-supported side branches along the validated left-coronary backbone?**

## Frozen baseline

This experiment is developed on a fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`. It never modifies the frozen anatomy master.

## Inputs

- Series 7 source CCTA and cached full-volume vesselness from `Cache/Secondary_3D_Vesselness_Topology_v1`.
- Frozen LAD centerline.
- Dense-QC-positive 20-mm continuation.
- C6 source path, used only as the label-neutral parent continuation after continuity has already shown that its common prefix is the same vessel as the 20-mm continuation.
- C7 extended source path, used as the same-method positive branch-control target.
- `Left_Coronary_Through_Vessel_Continuity_v1/summary.json`.

## Algorithm

1. Orient the frozen LAD and validated common-trunk trajectory at their zero-gap junction.
2. Construct a label-neutral backbone as frozen LAD (reversed through the junction) plus C6 continuation. No clinical LAD/LCX/LM label is assigned to this composite.
3. Calibrate the cached vesselness threshold from the accepted backbone itself.
4. At the known C6/C7 split, run the exact same side-branch discovery method used in the blind scan. The control must recover the C7 post-split path with dense source-plane QC and geometric agreement.
5. If the control passes, sample the backbone every 1.5 mm, excluding a ±3-mm neighborhood of the known C6/C7 split.
6. At each seed, preview source-supported directions 35–145 degrees from the local backbone tangent, then perform a 9-mm beam search using source HU, cached vesselness, smoothness, and explicit avoidance of re-entering the backbone after the first 2 mm.
7. Dense orthogonal source-CCTA QC is run on the leading hypotheses. A prospective branch candidate requires:
   - accepted dense-QC arc ≥5 mm;
   - plane-pass fraction ≥0.80;
   - initial branch angle ≥30 degrees;
   - endpoint separation from the backbone ≥3 mm.
8. Valid hypotheses are clustered by junction region so a single branch does not produce multiple duplicate detections.

## Interpretation

Possible statuses:

- `BACKBONE_BRANCH_DISCOVERY_PREREQUISITE_FAILED`
- `BACKBONE_BRANCH_DISCOVERY_C7_CONTROL_FAILED`
- `BACKBONE_BRANCH_DISCOVERY_NO_ADDITIONAL_VALID_BRANCH`
- `BACKBONE_BRANCH_DISCOVERY_ADDITIONAL_SOURCE_BRANCHES_FOUND`

A positive blind result establishes an **unlabeled source-CCTA side-branch hypothesis** only. It does not identify LAD, LM, LCX, diagonal, ramus, or obtuse-marginal anatomy by itself.

## Outputs

- `run_state.json`
- `vesselness_calibration.json`
- `label_neutral_backbone.csv`
- `C7_control.json`
- `C7_control_candidates.csv`
- `C7_control_recovered_path.csv`
- `C7_control_dense_qc.csv`
- `control_search_attrition.csv`
- `blind_branch_search_attrition.csv`
- `blind_branch_hypotheses.csv`
- `additional_branch_candidates.csv`
- up to five `additional_branch_XX_*.csv` paths and dense-QC CSVs
- `01_backbone_branch_geometry.png`
- `02_branch_scan_overview.png`
- `03_best_additional_branch_orthogonal_qc.png` when a blind candidate exists
- `summary.json`
- `OPENPLAQUE_LEFT_CORONARY_BACKBONE_BRANCH_DISCOVERY_REPORT.html`
- `OPENPLAQUE_LEFT_CORONARY_BACKBONE_BRANCH_DISCOVERY_RESULTS.zip`
