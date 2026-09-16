# Blind source-CCTA coronary ostium discovery

Fresh experiment from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

Purpose: determine whether source CCTA contains a second coronary-sized ostial exit even though the current/legacy TotalSegmentator coronary masks reach only the RCA at the aortic surface.

Discovery is blind to LAD, C6, and coronary masks. A root crop is defined from the accepted RCA proximal aortic level only to localize the coronary-root search field. Multiscale source-CCTA vesselness is calibrated from the accepted proximal RCA. Bright tubular components that physically touch the aortic surface are traced outward using HU, vesselness, directional continuity, and monotonically increasing distance from the aorta.

All candidate paths use the same predeclared gates: length >=5 mm; tortuosity <=1.8; robust HU fraction >=0.90; p10 vesselness >=0.50 of the RCA-calibrated vesselness threshold; outside-aorta gain >=3 mm; serial RCA-calibrated lumen plane pass fraction >=0.60; median plane score >=0.60; and coronary-scale median radius bounded by the RCA calibration.

After all blind candidates are frozen, the accepted RCA is used only to identify the rediscovered RCA candidate. The positive control requires an accepted path, median distance to known proximal RCA <=1.0 mm, p90 distance <=1.8 mm, and seed distance to the known RCA proximal endpoint <=3.0 mm.

Only if the RCA control passes may a spatially separate accepted candidate be nominated as a second coronary ostial exit. Current/legacy coronary-mask support and distances to LAD/C6 are then computed as post-hoc QC and have zero decision weight.

Positive status is `SOURCE_LEFT_CORONARY_OSTIAL_EXIT_REQUIRES_VISUAL_QC`. A positive result is still not clinical left-main confirmation. Master Coronary Anatomy Baseline v2.1 remains unchanged.
