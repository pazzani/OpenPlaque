# LAD Distal-Reference Plaque Self-Calibration v1

## Question

Can the validated distal LAD continuation provide a label-blind, vessel-specific normal-wall reference that allows a useful source-space plaque-excess estimate in the frozen LAD?

## Why this is the next experiment

The RCA research benchmark is now locked. The RCA-to-LAD transfer experiment failed, and a later frame-refinement experiment showed that LAD source-plane geometry could be rescued to essentially complete QC while plaque specificity still did not transfer adequately.

That leaves a vessel-neutral calibration problem rather than an unresolved LAD anatomy problem.

This experiment therefore does **not** transfer the RCA normal-wall model. Instead, it uses the independently source-confirmed distal LAD continuation as an internal LAD reference region.

## Anatomy prerequisites

- Frozen baseline commit: \`0593b453959f5a353d644267fbeef24b514ef4d7\`.
- Frozen coronary master remains unchanged.
- Frozen LAD source centerline is used as-is.
- The distal LAD extension must retain status \`LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED\`.
- The distal extension is joined only to the frozen LAD arc-0 endpoint.

## Geometry

The experiment uses the same label-blind LAD geometry refinement that previously rescued LAD source-plane QC:

- 0.5-mm longitudinal sampling;
- local PCA tangent over ±3 stations;
- candidate center shifts of 0, 0.25, 0.50, and 0.75 mm in the orthogonal plane;
- candidate selection based only on source-CCTA lumen geometry and center HU;
- 72-ray final lumen boundary;
- final station QC: center HU >=200, valid radial fraction >=0.60, p90/p10 lumen-axis proxy <=2.5.

No plaque labels enter geometry selection.

## Distal reference region

The reference region is fixed prospectively in **frozen-LAD coordinates**:

\`-15.0 mm <= arc < -3.0 mm\`

Negative arc values lie on the validated distal extension. The last 3 mm adjacent to the frozen-LAD join are excluded.

This region is selected from anatomy only. The prior plaque-vote profile is not used to select or fit the reference region.

## Plaque model

The source-space shell model is otherwise unchanged:

- shells: 0.75, 1.0, 1.25, and 1.5 mm;
- nominal shell: 1.0 mm;
- low attenuation: -30 to <30 HU;
- noncalcified: 30 to <130 HU;
- mixed/intermediate: 130 to <350 HU;
- calcified: >=350 HU;
- fat-like voxels < -30 HU excluded.

For each shell and HU component, a robust normal-wall fraction model is fit **only** in the distal reference region using:

- lumen radius;
- lumen radius squared;
- center HU;
- Huber regression;
- 90th-percentile positive reference residual threshold.

The fitted LAD-specific reference model is then applied unchanged to the frozen LAD.

## Prospective gates

Geometry:

- at least 18 valid reference stations;
- distal-reference station QC >=0.95;
- frozen-LAD station QC >=0.95.

Developmental plaque specificity on the frozen LAD:

- at least 4 majority-positive bins;
- at least 5 vote-free negative bins;
- majority-positive vs vote-free AUROC >=0.80;
- strict 5/5 vs vote-free AUROC >=0.90 when at least two strict bins exist;
- positive/negative median excess ratio >=2.0;
- minimum cross-shell profile Spearman correlation >=0.80.

Possible statuses:

- \`LAD_DISTAL_REFERENCE_SELF_CALIBRATION_GEOMETRY_FAILED\`
- \`LAD_DISTAL_REFERENCE_SELF_CALIBRATION_SPECIFICITY_FAILED\`
- \`LAD_DISTAL_REFERENCE_SELF_CALIBRATION_DEVELOPMENTAL_PASS\`

## Interpretation boundary

A pass is **not independent validation**.

The distal reference selection and model fitting are label-blind, but the frozen-LAD plaque-vote labels have already been inspected in prior method-development experiments. They are therefore used only as a developmental benchmark.

A passing result would justify a provisional LAD research plaque proxy for OpenPlaque reporting. It would not constitute validated clinical TPV and would not modify the frozen anatomy master.

## Outputs

The run writes:

- source-space station quantification;
- LAD-specific distal-reference model coefficients;
- excess-plaque station table;
- 1-mm profiles for all shell widths;
- nominal frozen-LAD profile;
- developmental validation bins;
- shell consistency;
- geometry QC;
- automatically selected source-plane QC;
- summary/run-state/provenance JSON;
- HTML report;
- ZIP archive.
