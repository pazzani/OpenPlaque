# RCA source-space plaque candidate validation

This experiment is the first post-synthesis attempt to obtain a physically meaningful plaque-related volume directly from the original source CCTA.

Scientific rules:

- Master Coronary Anatomy Baseline v2 controls vessel eligibility.
- Only the accepted RCA is analyzed.
- The earlier curved-series plaque model is used only to define validated longitudinal target windows (dominant 0–5 mm and the weaker 11–12 mm interval).
- The failed circumferential/angular curved-to-source registration is not used.
- Current + legacy TotalSegmentator coronary masks define a conservative source-space lumen union.
- The TotalSegmentator aorta mask is excluded to reduce proximal ostial blood-pool contamination.
- Candidate voxels must lie outside the coronary union, within a 0.20–2.25 mm periluminal shell, and near the accepted RCA centerline.
- A broad >=350 HU shell is retained only as sensitivity/QC.
- The strict calcific threshold is max(600 HU, source RCA centerline median HU + 75 HU).
- Strict source candidates must form connected 3-D components with at least 0.15 mm3 volume and at least 0.40 mm RCA arc span.
- The primary model-positive RCA window is compared with deterministic plaque-negative control windows.

Reported mm3 values are unique spatial source-CCTA voxels, so they are geometrically physical volumes. They are still labeled **calcific candidate volume**, not total plaque volume (TPV), because a validated outer-wall segmentation is not available and noncalcified plaque is not quantified.

A positive automatic result remains `REQUIRES_VISUAL_QC`; promotion requires review of the automatically selected source-CCTA orthogonal planes.

## Standalone Colab rule

The notebook is built for a completely fresh Colab runtime. Drive mount is the first executable cell, controls are second, the exact branch is freshly cloned, OpenPlaque is installed from that clone with `pip install -e`, import is verified in a separate Python process, tests run, and only then is the workflow executed. It must not depend on state from another notebook.
