# Left proximal-trunk source-led long extension v1

## Question
The local multidirection experiment recovered 27/29 dense-QC accepted 10-mm continuations from the validated proximal-trunk endpoint, but every accepted hypothesis moved farther from the nearest aortic surface. This experiment asks whether the strongest source-supported branch continues for a longer arc when aortic geometry is excluded from discovery entirely.

## Frozen baseline
Base commit: `0593b453959f5a353d644267fbeef24b514ef4d7` (`CORONARY_ANATOMY_BASELINE_V2_FROZEN`). The experiment never modifies the frozen master.

## Prerequisites
- `Left_Proximal_Trunk_Continuation_QC_v1`: positive 20-mm dense-QC continuation.
- `Left_Proximal_Trunk_Local_Multidirection_v1`: `PROXIMAL_TRUNK_LOCAL_MULTIDIRECTION_CONTINUATION_QC_POSITIVE` and saved `best_local_multidirection_path.csv`.
- Series 7 source CCTA cache, current + legacy coronary masks, high-resolution aorta mask, canonical RCA.

## Prospective design
- Begin at the distal endpoint of the prior best 10-mm local source-supported continuation.
- 28-mm local source field, 25-mm maximum new search arc, 0.40-mm steps, beam width 100.
- Up to six source-supported initial directions within the forward tangent cone.
- Discovery uses only source HU, multiscale vesselness, coronary-mask proximity, turning, loop/self-return protection, and avoidance of the already-known path.
- **No aortic-distance gate or aortic-distance scoring is used during discovery.**
- Dense source-CCTA orthogonal plane QC every 0.40 mm; first three consecutive failures truncate a candidate. Positive continuation requires >=5 mm accepted arc and >=80% pass fraction.
- Canonical RCA source-plane QC is rerun as the positive control.
- Nearest-aorta distance is calculated only after source-led candidates are frozen. A >=2-mm post-hoc return toward the aorta is reported separately; <=0.90 mm nominates an aortic-contact candidate for visual adjudication.

## Interpretation boundary
A positive result establishes only long source-supported vessel continuity beyond the already validated local continuation. It does not establish left-main identity, left-coronary ostial identity, LCX topology, or modify the frozen anatomy baseline.

## Outputs
`run_state.json`, `summary.json`, RCA control CSV, field metadata, search attrition CSV, candidate table, best path CSV, dense plane-QC CSV, post-hoc aortic-distance profile, QC PNGs, HTML report, and ZIP archive.
