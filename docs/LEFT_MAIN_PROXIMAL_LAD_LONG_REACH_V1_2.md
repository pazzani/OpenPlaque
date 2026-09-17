# Left Main / Proximal LAD Long-Reach v1.2

## Purpose

This experiment revisits the source-CCTA bridge from the independently accepted proximal LAD endpoint to the aortic root after the blind source-ostium positive control was recovered in `Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1`.

The preceding calibrated bridge experiment (`Left_Main_Proximal_LAD_Ostial_Bridge_v1_1`) started the left search at the accepted LAD endpoint nearest the aorta. That endpoint was approximately 36.04 mm from the nearest aortic surface, but the search was capped at 20 mm with an 18 mm acceptance-length ceiling. Therefore the experiment could not reach the aorta even along a straight line. Its `no_path_reached_aorta` result is not interpretable as evidence against a left-coronary bridge.

## Prospective scientific change

The only intended scientific change on the left side is the reachability budget:

- previous maximum left search length: 20 mm
- previous maximum accepted left path length: 18 mm
- v1.2 maximum left search length: 46 mm
- v1.2 maximum accepted left path length: 44 mm
- minimum prospective slack beyond the measured straight-line aortic distance: 6 mm

All source-search evidence and gates remain those of the calibrated v1.1 bridge:

- source-CCTA HU support
- multiscale 3-D vesselness
- current/legacy coronary-mask proximity and support
- monotonically improving distance to the aorta
- endpoint aortic-distance gate
- tortuosity and turn-angle gates
- robust-HU, vesselness, and coronary-mask support gates

The RCA calibrated proximal retrace is rerun unchanged with its original 12 mm search budget, 10 mm acceptance ceiling, and 6 mm inside-path starting point.

## Prerequisite

`Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1/summary.json` must exist, match the frozen baseline commit, and report `RCA_control_pass = true`.

This ensures that the source-CCTA root-discovery machinery has a valid positive control before the longer targeted LAD-to-aorta experiment is interpreted.

## RCA independence

The known RCA does not steer the left-side search. After a source-supported path reaches the aorta, its aortic endpoint is compared post hoc with the known RCA proximal endpoint. A path is not nominated as a left-coronary bridge unless the two aortic endpoints are separated by at least 8 mm.

This is an identity/exclusion check, not a search constraint.

## Interpretation

Possible statuses are:

- `LONG_REACH_RCA_CONTROL_FAILED`: the unchanged calibrated RCA control failed, so the left result is not interpreted.
- `LONG_REACH_LAD_TO_AORTA_NOT_ESTABLISHED`: the feasible 46 mm search did not produce a path satisfying the unchanged source gates.
- `LONG_REACH_PATH_REACHED_AORTA_BUT_RCA_ASSOCIATED`: a source-supported path reached the aorta but terminated too close to the known RCA ostium to nominate an independent left exit.
- `LONG_REACH_LEFT_CORONARY_BRIDGE_REQUIRES_VISUAL_QC`: a source-supported path reached an RCA-independent aortic endpoint and requires visual adjudication.

A positive result does **not** by itself establish clinical left-main identity, prove the LCX/C6 joining point, or modify Master Coronary Anatomy Baseline v2.

## Reproducibility

The experiment is run from a fresh branch rooted directly at frozen baseline commit:

`0593b453959f5a353d644267fbeef24b514ef4d7`

The Colab notebook pins an exact science commit containing the code, tests, and this document before execution.
