# LCX-like / OM-like Source-Space Composition + PCAT Feasibility v1

## Purpose

The LCX structural source-QC freeze established deterministic research-only structural labels on the exact source-space paths:

- C6 = LCX-like parent continuation
- C7 = OM-like daughter

Clinical LCX/OM identity is still not established, left main remains unresolved, and the frozen master anatomy is not modified.

This experiment asks a narrower question:

**Can the frozen C6/C7 research paths support technically stable source-space peri-luminal HU composition and direct PCAT measurements?**

It is not a plaque-validation experiment.

## Inputs

The experiment reuses:

- frozen Series-7 source CCTA;
- the frozen coronary master status;
- LCX Structural Source-QC Freeze v1 summary;
- LCX structural dense source-QC CSV.

No new vessel search is performed.

## Segment definition

Measurements start 1.0 mm after the C6/C7 split to reduce immediate bifurcation overlap.

For raw peri-luminal shell composition, all source-QC-passing stations after that exclusion are used.

For PCAT, the longest contiguous QC-passing segment is used. This prevents a low-HU terminal endpoint from creating an artificial perivascular region.

C6 and C7 are always reported separately. They must not be added and described as a clinical LCX burden.

## Raw peri-luminal shell composition

At each source-space station, the same 72-ray contrast-lumen boundary method used in prior source-space work is recomputed on the exact frozen center and tangent.

Shell thicknesses:

- 0.75 mm
- 1.00 mm nominal
- 1.25 mm
- 1.50 mm

Research HU categories:

- fat-like excluded: < -30 HU
- low attenuation: -30 to <30 HU
- noncalcified: 30 to <130 HU
- mixed/intermediate: 130 to <350 HU
- calcified: >=350 HU

These are reported as **raw shell HU composition**.

The raw non-fatlike shell volume is explicitly **not** called plaque burden or TPV because there is no validated C6/C7 normal-wall reference model or independently segmented outer wall.

## Direct PCAT

PCAT uses the same direct attenuation definition as the locked RCA prototype:

- adipose attenuation range: -190 to -30 HU
- nominal modeled outer-wall margin: +0.75 mm from local lumen radius
- radial perivascular shell extends to 3x the modeled outer radius, equivalent to one local modeled outer diameter beyond the outer wall
- aorta exclusion is used when an aligned aorta mask is available

Margin sensitivity is measured at:

0.25, 0.50, 0.75, 1.00, and 1.25 mm.

PCAT is reported for C6 and C7 separately.

These are direct attenuation feasibility measurements, not proprietary FAI.

## Technical feasibility gates

Each branch must have:

- at least 5.0 mm of evaluable source-space shell path;
- at least 5.0 mm of contiguous PCAT path;
- at least 500 fat voxels for the nominal +0.75 mm PCAT geometry;
- at least 75% of longitudinal PCAT bins containing at least 25 fat voxels.

These are only technical sampling gates. They do not validate plaque or PCAT as clinical biomarkers.

## Possible statuses

- LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_COMPLETE
- LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_FAILED

## Outputs

The run writes:

- raw shell station quantification;
- raw shell totals and 1-mm profiles;
- shell profile stability;
- C6 and C7 direct PCAT summaries;
- PCAT longitudinal and radial profiles;
- PCAT wall-margin sensitivity;
- four QC figures;
- summary/provenance/run-state JSON;
- HTML report;
- ZIP archive.

## Scientific boundary

A positive run means the exact frozen C6/C7 research paths are technically usable for source-space composition and direct PCAT measurements.

It does not establish clinical LCX/OM labels, does not resolve LM, does not establish validated plaque burden, does not establish clinical TPV, and does not produce proprietary FAI.
