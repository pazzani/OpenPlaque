# LAD Distal Endpoint Continuation v1

## Motivation

The completed label-neutral backbone branch scan passed its C7 same-method positive control and
found one clustered additional source-supported trajectory. It was seeded at backbone arc
approximately 4 mm, near the distal endpoint of the frozen LAD, and ran about 9.2 mm with a
100% dense-plane pass fraction. Its apparent branch angle was about 151 degrees relative to the
backbone's forward direction.

That geometry is suspicious for *terminal continuation* of the frozen LAD rather than a true
side branch. This experiment prospectively tests that hypothesis.

## Independence rule

The prior blind branch path is **not** used to guide or score the target trace. It is used only:

1. to identify which endpoint of the frozen LAD is the endpoint under test; and
2. after the target path has been frozen, as a post-hoc reproducibility reference.

The target search starts exactly at the frozen distal LAD endpoint. Its only directional prior
is the outgoing tangent computed from the terminal 4 mm of the frozen LAD.

## Source search

- Full-resolution Series-7 CCTA is the source image.
- A local multiscale 3-D Frangi vesselness field is recomputed in a 22-mm source-space margin.
- Search step: 0.40 mm.
- Target search limit: 15 mm.
- Eight source-supported initial directions are selected prospectively from a 40-degree cone.
- Beam width: 72.
- HU gate: 100-1400 HU.
- Vesselness threshold: 20% of the terminal-LAD local p20 calibration value.
- Step-turn limit: 55 degrees.
- After 1.6 mm, proposals within 0.8 mm of the frozen LAD are rejected to prevent re-entry.

## Positive control

Before target inference, the same source-led tracer must recover at least 5 mm of an already
known terminal frozen-LAD segment. The recovered control path must have at least 80% dense-plane
pass rate, at least 80% of its points within 2 mm of the frozen LAD, and median separation at
most 1 mm.

## Target acceptance

A target extension is accepted only when all are true:

- accepted dense-QC arc >= 5 mm;
- dense-plane pass fraction >= 0.80;
- endpoint is >= 3 mm from the frozen LAD.

Dense source-plane QC uses the same lumen morphology thresholds used in recent source-CCTA
continuation experiments.

## Post-hoc reproducibility

After target tracing is complete, the independently accepted target is compared with the
previously discovered beyond-endpoint trajectory. Reproduction requires:

- >= 60% of the independent path within 2 mm of the prior trajectory;
- median separation <= 1.5 mm;
- median tangent alignment >= 0.70.

## Scientific boundary

A positive result establishes a source-supported distal extension candidate of the frozen LAD.
It does **not** automatically modify the frozen master or claim a final clinical segment label.
