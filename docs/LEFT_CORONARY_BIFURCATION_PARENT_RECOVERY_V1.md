# Left coronary bifurcation-derived parent recovery v1

## Purpose

The previous source-led continuation did not discover new proximal anatomy. Topology crosswalk showed that the 10-mm local continuation lies on the established C6/C7 common trunk, and the subsequent 25.2-mm path first follows that common trunk and then re-enters the frozen LAD. This experiment therefore stops treating those paths as an LM-like extension.

Instead, v1 treats the frozen LAD endpoint plus the validated 20-mm continuation as a two-daughter junction and asks whether an independent parent vessel can be recovered from source CCTA.

## Prospective design

- Fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- Frozen master is read-only and never modified.
- First verify that the dense-QC 20-mm continuation geometrically overlaps the established C6/C7 common trunk.
- Derive a parent-search direction from the opposite angular bisector of the two daughter tangents.
- Discovery uses source HU + multiscale 3-D vesselness + turn/loop gates only. Aortic geometry is post-hoc only.
- After 2 mm, proposals within 0.90 mm of either known daughter are rejected so the search cannot simply backtrack onto LAD/common-trunk anatomy.
- Dense orthogonal source-plane QC is performed every 0.4 mm; acceptance requires at least 6 mm and at least 80% plane pass.

## Same-mechanism positive control

At the already-established C6/C7 split, the two post-split daughter tangents are used to derive the parent direction. The exact same source-led parent search must recover the known C6/C7 common trunk. A control candidate must be source-QC accepted and have median distance <=1.5 mm, p90 distance <=2.5 mm, and endpoint distance <=2.5 mm to the known parent trunk.

No target inference is permitted if this control fails.

## Target interpretation

The target search begins at the LAD/common-trunk junction. If a source-QC accepted parent continuation is found, aortic distance and RCA association are evaluated only after discovery. Status may indicate no aortic progress, meaningful progress toward the aorta, or an aortic-reaching candidate. None of these statuses assigns a clinical LM label by itself.

## Outputs

`run_state.json`, `junction_geometry.json`, control attrition/hypotheses/recovery metrics and QC, target attrition/hypotheses, best parent path and dense QC when available, post-hoc aortic-distance profile, geometry/QC PNGs, `summary.json`, HTML report, and result ZIP.
