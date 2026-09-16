# C6 monotonic distal reacquisition

Fresh experiment from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

Purpose: determine whether the established C6 parent continuation can be extended distally without the loop-like behavior seen with unconstrained Dijkstra search.

The source-space beam search is chamber-blind. It advances in short physical-space steps within a forward cone, updates the local tangent, requires increasing endpoint displacement, rejects loop-like returns, and scores only source-CCTA HU, multiscale vesselness, current/legacy coronary support, and directional continuity.

C6 must first rediscover its known terminal ~3 mm as a positive control. A new distal extension must add at least 5 mm, displace at least 4 mm, have tortuosity <=1.8, max turn <=60 degrees, robust HU fraction >=0.90, union coronary-mask support >=0.90, and adequate vesselness.

Only after the C6 path is frozen is the validated direct chamber-interface midpoint metric reapplied against the previously accepted frozen C7 extended path. The existing chamber score-margin threshold of 0.08 is unchanged. No curved-template or prior AV-groove score is used in path selection.

A positive status nominates C6 as LCX-like and C7 as OM-like for visual QC only. It does not modify the frozen Master Coronary Anatomy Baseline and does not establish clinical identity automatically.
