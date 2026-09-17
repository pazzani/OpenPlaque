# Left proximal-trunk continuation QC v1

## Motivation

The LAD mask-gate diagnostic showed that removing the hard TotalSegmentator coronary-mask rejection did not change the LAD-to-aorta search. The hard and shadow searches were identical, no proposal was rejected by the mask gate, and both terminated after about 23.6 mm of path while still roughly 28 mm from the aorta.

However, the saved best frontier path is strongly source-supported for most of that interval: it remains centered on a compact contrast-filled lumen through approximately the first 20 mm and only loses vesselness/centering near the terminal portion. This experiment asks a narrower question than left-main discovery: **how much of that continuation is independently defensible as a real proximal vessel segment extending from the accepted LAD endpoint?**

## Prospective method

- Fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- Input candidate: `Left_Main_LAD_Mask_Gate_Diagnostic_v1/hard_mask_best_frontier_path.csv`.
- Dense orthogonal source-CCTA planes every 0.4 mm.
- Fixed plane gates: center HU >= 200; connected bright-lumen radius 0.55–3.2 mm; centroid offset <= 1.10 mm; axis ratio <= 2.2; local lumen-to-ring contrast >= 40 HU.
- Canonical RCA is processed with the same plane-QC code and must pass at least 75% of sampled planes.
- Candidate truncation is at the first run of three consecutive failed planes.
- A candidate continuation is positive only if the retained segment is at least 5 mm and at least 80% of its dense planes pass.

## Scientific boundary

A positive result validates source-CCTA continuity of a proximal vessel segment beyond the currently accepted LAD endpoint. It is called a **proximal-trunk continuation candidate**, not LM. It does not establish the LCX join, does not prove clinical LM identity, and does not modify Master Coronary Anatomy Baseline v2.
