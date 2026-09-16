# Blind source-CCTA coronary ostium discovery v1.2

## Purpose

v1.2 is a performance-only continuation of the fixed v1.1 blind source-CCTA coronary ostium discovery experiment. It does not relax the prospective scientific thresholds, alter discovery weights, change the 40-component limit, change RCA positive-control criteria, or modify Master Coronary Anatomy Baseline v2.1.

## Why v1.2 exists

The v1.1 Colab successfully passed the prior integer-indexing failure and completed RCA lumen calibration, but the final cell remained in the source-root search for more than 30 minutes without reaching the candidate CSV. Two expensive stages had no progress reporting: multiscale 3-D Frangi vesselness and serial orthogonal-plane QC over many beam-search finalists.

## Execution changes

1. The exact v1.0 Frangi formula is evaluated one scale at a time and cached persistently under `OpenPlaque/Cache/Left_Coronary_Source_Ostium_Discovery_v1_2`. A completed cache is reused only when source-root voxel contents, shape, spacing, scales, and formula identifier all match. Partial results are saved after each scale so an interrupted run can resume at the next scale.
2. The inexpensive non-QC terms of the frozen v1.1 acceptance gate are evaluated before serial orthogonal-plane QC. A beam finalist that already fails any of these terms cannot satisfy the final gate, so its 9-plane QC is skipped. Every finalist that survives those terms still receives the unchanged v1.1 serial QC and final gate.
3. `progress.json` plus console messages identify the current vesselness scale and component-tracing stage.

## Frozen acceptance terms

The pre-QC terms remain:

- length >= 5.0 mm
- tortuosity <= 1.8
- robust HU fraction >= 0.90
- p10 vesselness >= 0.50 × the RCA-calibrated vesselness threshold
- outside-aorta gain >= 3.0 mm

For survivors, the unchanged serial-QC terms remain:

- plane pass fraction >= 0.60
- median plane score >= 0.60
- median radius within the same RCA-calibrated coronary-scale bounds used in v1.1

The unchanged final selection score is used for serial-QC-evaluated finalists. If a component has no finalist that survives the non-QC gate, v1.2 keeps one beam-score representative only for post-hoc diagnostics; it is necessarily marked unaccepted and cannot create a positive scientific result.

## Interpretation

As before, a positive run only nominates a second coronary-sized ostial exit for visual QC. LAD, C6, and current/legacy coronary masks remain post-hoc only and have zero discovery weight. Clinical left-main identity remains unresolved until a prospectively gated connection experiment establishes common-parent topology.
