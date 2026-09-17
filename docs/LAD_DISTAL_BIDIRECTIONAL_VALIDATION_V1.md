# LAD distal bidirectional validation v1

## Purpose

The preceding endpoint-continuation experiment independently found a 15.2-mm source-CCTA-supported continuation from the frozen LAD distal endpoint. All 16 target hypotheses satisfied the extension gate, but the best path matched only the proximal portion of the earlier blind trajectory. This experiment does not choose between those two forward trajectories. It asks whether the independently traced 15.2-mm segment itself is stable under reverse tracing.

## Branch discipline

This experiment is on `lad-distal-bidirectional-consensus-from-main`, created directly from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`. The prior experiment is used only through persistent Google Drive artifacts. Frozen anatomy is never modified.

## Prerequisite

`LAD_Distal_Endpoint_Continuation_v1/summary.json` must report a passing known-LAD control and a valid forward extension. The accepted prerequisite statuses are the independently reproduced status and the source-supported-different-trajectory status.

## Prospective design

1. Load the frozen LAD and the saved best independent 15.2-mm forward candidate.
2. Recompute local multiscale source-space vesselness from full-resolution Series 7; no incompatible cropped vesselness cache is used.
3. Positive control: from the tested frozen LAD endpoint, point inward and recover at least 5 mm of the known terminal LAD with the same tracer. Control requires >=80% dense-plane QC, >=80% within 2 mm of the known LAD, and median separation <=1.0 mm.
4. Target: start at the far endpoint of the 15.2-mm forward candidate. The initial reverse tangent is obtained from its terminal 4 mm. Candidate coordinates are not used in discovery scoring.
5. Search up to 17 mm with 0.4-mm steps and source HU/vesselness gates only.
6. Dense orthogonal source-CCTA QC uses the same lumen-shape thresholds used in the immediately preceding anatomy experiments.
7. A reverse hypothesis confirms the segment only if it:
   - reaches within 1.5 mm of the tested frozen LAD endpoint,
   - does so after at least 12 mm of reverse travel,
   - has >=80% dense-plane pass fraction through the reaching point,
   - has >=80% of the reverse prefix within 2 mm of the reversed forward candidate,
   - has median separation <=1.0 mm and p90 separation <=2.0 mm,
   - has median tangent alignment >=0.80.
8. At least three separate reverse hypotheses must satisfy all confirmation gates.

## Interpretation

`LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED` means the forward source-supported segment is also recoverable in reverse with multiple source-led hypotheses. This is strong evidence for a real continuous distal coronary segment, but it does not automatically alter the frozen anatomy baseline.

`LAD_DISTAL_BIDIRECTIONAL_NO_CONFIRMING_REVERSE_TRACE` means the forward candidate remains source-QC positive but lacks bidirectional confirmation. Do not promote it.

## Outputs

The run writes `run_state.json`, `summary.json`, vesselness calibration, known-LAD reverse-control outputs, reverse hypotheses and attrition CSVs, the best reverse path and dense QC, bidirectional geometry PNG, orthogonal QC PNG, HTML report, and ZIP archive.

Research use only; no clinical vessel-label assignment is made by this experiment.
