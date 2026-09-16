# Left coronary ostium mask consensus

Fresh experiment from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

Purpose: discover a second coronary-sized aortic-root exit without using LAD or C6 coordinates in candidate generation.

Method:
- Align current/legacy TotalSegmentator coronary masks and high-resolution aorta to source Series 7 CCTA.
- Find coronary-mask clusters within 2 mm of the aortic surface near the accepted RCA ostial level.
- Identify the RCA interface cluster automatically and require that an outward trace from it reproduces the known proximal RCA.
- Evaluate every other sufficiently separated interface cluster with the same outward source-CCTA tracing machinery.
- Require >=5 mm path length, <=1.8 tortuosity, >=90% robust coronary HU, >=70% union-mask support, >=20% dual-mask support, >=3 mm gain away from the aorta, and >=60% RCA-calibrated serial orthogonal lumen-plane pass rate.
- RCA-calibrated lumen size rejects aortic-scale candidates; the historical failed root candidate had ~4.51 mm radius versus ~1.61 mm median RCA radius.
- LAD/C6 distances are calculated only after candidate selection and are never decision inputs.

A positive result nominates a `LEFT_CORONARY_OSTIAL_EXIT_REQUIRES_VISUAL_QC`. It does not establish clinical LM identity and does not modify Master Coronary Anatomy Baseline v2.1.
