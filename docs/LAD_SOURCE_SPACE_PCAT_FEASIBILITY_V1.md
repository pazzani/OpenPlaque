# LAD Source-Space PCAT Feasibility v1

## Purpose

The LAD anatomy is strong, but plaque quantification has not met the prespecified specificity/stability gates. This experiment therefore isolates the inflammation side and asks whether the accepted LAD geometry supports technically stable **direct PCAT attenuation**.

No plaque model is fitted and the frozen anatomy master is not modified.

## Anatomy inputs

- Frozen LAD source centerline: approximately 24.997 mm.
- Independently source-confirmed distal continuation: approximately 15.2 mm.
- Distal continuation prerequisite status: LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED.

The distal continuation remains research-only and is not promoted into the frozen master.

## Source-plane geometry

The experiment uses the same label-blind source-plane refinement that previously brought LAD geometry QC to essentially complete pass rates:

- 0.5-mm longitudinal sampling;
- local PCA-smoothed tangent;
- deterministic center search up to 0.75 mm in the orthogonal plane;
- source-CCTA lumen compactness and center-HU scoring;
- 72-ray final lumen boundary.

No plaque labels enter this process.

## PCAT definition

The direct attenuation definition matches the locked RCA prototype:

- adipose HU window: -190 to -30 HU;
- nominal modeled outer-wall margin: +0.75 mm from local lumen radius;
- radial perivascular extent to 3x the modeled outer radius;
- aligned aorta exclusion when available.

Margin sensitivity is computed at 0.25, 0.50, 0.75, 1.00 and 1.25 mm.

This is not proprietary FAI.

## Research segments

Two segments are reported separately.

### Frozen LAD

Frozen-LAD arc coordinate:

- 1.0 to 24.0 mm.

This excludes approximately 1 mm at each endpoint of the accepted frozen LAD.

### Validated distal continuation

Frozen-LAD arc coordinate:

- -14.0 to -1.0 mm.

Negative coordinates lie on the independently source-confirmed distal continuation. The immediate 1-mm join neighborhood and extreme distal tip are excluded.

## Technical feasibility gates

Frozen LAD:

- segment length >=20 mm;
- source-plane QC fraction >=0.95;
- at least 1,000 nominal PCAT fat voxels;
- >=90% of 1-mm longitudinal bins contain at least 25 fat voxels;
- repeated +0.75-mm computation is bit-identical.

Validated distal continuation:

- segment length >=10 mm;
- source-plane QC fraction >=0.95;
- at least 1,000 nominal PCAT fat voxels;
- >=90% longitudinal-bin coverage;
- repeated +0.75-mm computation is bit-identical.

A positive status is LAD_SOURCE_SPACE_PCAT_FEASIBILITY_COMPLETE.

## Scientific boundary

A positive run means that the LAD geometry can support reproducible research direct-PCAT measurements under the OpenPlaque sampling definition.

It does not:

- validate LAD plaque burden;
- make the distal continuation part of the master anatomy;
- constitute independent clinical validation;
- reproduce proprietary FAI methodology.
