# RCA Expert Outer-Wall Validation v1

## Purpose

This experiment validates the locked RCA plaque research proxy against an independently drawn, blinded expert outer-vessel contour.

It is deliberately **not** another tuning experiment.

The locked RCA plaque algorithm remains unchanged:

- original accepted RCA source geometry;
- 1.0-mm fixed shell;
- RCA 20–50 mm normal-wall reference normalization;
- locked research plaque excess result;
- no changes to plaque HU bins, centerline, shell width, or thresholds.

## Required expert input

The expert annotation pack must first be completed with:

\`RCA_outer_wall_expert_mask.npy\`

The validator searches for that mask in three locations:

1. \`OpenPlaque/RCA_Expert_Outer_Wall_Annotation_Pack_v1/expert_pack/\`
2. \`OpenPlaque/RCA_Expert_Outer_Wall_Annotation_Pack_v1/\`
3. \`OpenPlaque/\`

The mask must:

- match the 51 x 67 x 67 source-plane stack;
- be binary;
- contain the lumen wherever a slice is annotated;
- use all zeros for uninterpretable slices.

## Prespecified mask QC

Validation proceeds only if:

- at least 41 of 51 slices are annotated;
- at least 98% of annotated slices fully contain the supplied lumen seed;
- at least 20 reference-region slices from 20–50 mm are annotated;
- at least 15 holdout-region slices from 0–20 mm are annotated.

These thresholds are mask-completeness checks, not plaque-performance gates.

## Geometry comparison

For every annotated slice, the validator computes:

- expert outer-vessel area;
- expert wall area;
- radial expert outer-vessel radius;
- expert wall thickness relative to the supplied lumen boundary;
- fixed 1.0-mm shell outer-vessel area;
- fixed-shell versus expert outer-vessel Dice and Jaccard;
- fixed-shell wall-area bias;
- ray-level fixed-minus-expert wall-thickness differences.

This directly tests the geometric assumption underlying the locked 1.0-mm shell.

## Expert-wall HU composition

Within the expert-contoured wall, the same research HU ranges are reported:

- fat-like excluded: < -30 HU;
- low attenuation: -30 to <30 HU;
- noncalcified: 30 to <130 HU;
- mixed/intermediate: 130 to <350 HU;
- calcified: >=350 HU.

Raw expert wall volume is anatomical wall volume on the resampled orthogonal planes. It is **not** plaque volume by itself.

## Like-for-like plaque-excess comparison

To compare the locked plaque proxy with the expert-defined wall without changing the locked algorithm, the expert wall is processed with the same prespecified normalization strategy:

- fit normal-wall HU-component fractions only in 20–50 mm;
- robust Huber regression;
- 90th-percentile reference residual threshold;
- evaluate expert-wall excess in the disjoint 0–20 mm holdout.

The resulting quantity is called the **expert-wall reference-normalized excess proxy**. It is not assumed to be clinical TPV.

The validator compares locked versus expert-wall holdout profiles using:

- Spearman correlation;
- mean absolute per-slice difference;
- locked-minus-expert bias;
- holdout total volumes;
- locked/expert total ratio.

No plaque-performance threshold is used to tune or accept the locked algorithm. The comparison is reported descriptively.

## Scientific boundary

The expert contour supplies independent anatomical outer-wall information. However:

- the expert wall does not directly identify plaque versus normal wall;
- reference normalization remains a research model;
- volumes are calculated from resampled source-orthogonal planes;
- no clinical TPV claim is made.

If the locked fixed-shell excess agrees well with the blinded expert-wall proxy, that supports its use as a research approximation. If it disagrees, the disagreement should guide a separately trained wall-segmentation model rather than retuning this scan.
