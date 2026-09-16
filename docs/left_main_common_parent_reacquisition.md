# Left Main Common-Parent Reacquisition

This experiment starts from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7` and does not modify Master Coronary Anatomy Baseline v2.1.

Prerequisite: the completed structural adjudication must have all predeclared gates pass, with C6 frozen only as an **LCX-like parent continuation** and C7 as an **OM-like daughter** research label.

The new experiment addresses the remaining LM gap. It automatically identifies the proximal LAD/C6 neighborhood, forms a bifurcation seed, and searches from that seed toward the aortic surface using source-CCTA HU, multiscale vesselness, current/legacy coronary-mask proximity, local directional continuity, loop avoidance, and monotonically non-increasing distance to the aorta.

Before the left-sided search is interpreted, the identical search logic is calibrated by tracing the accepted RCA proximal segment back to the aorta. A failed RCA control blocks any LM interpretation.

A left-sided path is accepted only if it reaches the aortic surface, is 1.5–14 mm long, has tortuosity <=1.8 and max turn <=65 degrees, robust HU support >=0.90, union coronary support >=0.70, adequate vesselness, and at least 1 mm reduction in aortic distance.

A positive result is `LEFT_MAIN_LIKE_COMMON_PARENT_REQUIRES_VISUAL_QC`. It nominates a structural LM-like common parent only. Clinical LM identity remains unresolved until visual QC, and the frozen master anatomy is unchanged.