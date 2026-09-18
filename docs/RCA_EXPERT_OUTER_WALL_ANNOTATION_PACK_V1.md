# RCA Expert Outer-Wall Annotation Pack v1

## Purpose

The RCA research plaque benchmark is now locked. The best developmental method is the original 1.0-mm fixed-shell, reference-normalized source-space excess proxy. Two more anatomically explicit outer-wall methods failed to improve specificity.

The next scientifically meaningful step is therefore **independent expert outer-wall annotation**, not more threshold tuning on the same scan.

This experiment creates a blinded annotation package from the accepted RCA source CCTA.

## Prerequisites

- Frozen baseline commit: \`0593b453959f5a353d644267fbeef24b514ef4d7\`.
- Frozen coronary master remains unchanged.
- RCA plaque + PCAT research lock status must be \`RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED\`.

## Annotation sampling

The pack contains source-CCTA orthogonal RCA planes from:

- 0 to 50 mm along the accepted RCA centerline;
- 1.0-mm longitudinal spacing;
- 0.15-mm in-plane sampling;
- 10 x 10 mm field of view.

This produces 51 cross-sectional source planes.

## Blinding

The expert package intentionally excludes:

- plaque-vote labels;
- majority/strict plaque labels;
- locked plaque volumes;
- plaque hot-spot locations;
- PCAT attenuation values.

The only supplied segmentation context is the automated research lumen boundary.

## Expert task

For each plane, contour the **outer vessel boundary**.

The required binary output mask represents the full area inside the outer vessel boundary, including the lumen. It is not a ring mask.

A completed mask should be saved as:

\`RCA_outer_wall_expert_mask.npy\`

with the same shape as the provided empty template.

If a plane is not interpretable, the expert should leave that entire slice blank rather than infer a contour from neighboring plaque results or a fixed wall thickness.

## Package contents

- \`RCA_source_HU_planes.npy\`: source HU stack;
- \`RCA_lumen_seed_mask.npy\`: automated lumen context;
- \`RCA_outer_wall_annotation_template.npy\`: empty binary template;
- \`RCA_expert_outer_wall_planes.npz\`: combined geometry/HU package;
- \`RCA_expert_outer_wall_manifest.csv\`: arc and LPS mapping;
- \`annotation_spec.json\`: mask semantics and spacing;
- \`previews/*.png\`: windowed source-plane previews;
- \`README_EXPERT_ANNOTATION.md\`: instructions.

## Validation planned after annotation

Once an independent expert mask is available, the next experiment should:

1. compute expert-defined wall area and wall volume;
2. classify expert-wall pixels into the existing research HU bins;
3. compare the locked 1.0-mm fixed-shell excess proxy with the expert-contoured reference;
4. evaluate longitudinal agreement without changing the locked plaque algorithm;
5. quantify where and why the fixed shell over- or under-estimates the expert wall.

Only after that comparison should any new learned wall-segmentation model be considered.

## Scientific boundary

This experiment produces an annotation package only. It does not create expert labels, modify the frozen anatomy, change the locked RCA plaque result, or claim clinical TPV.
