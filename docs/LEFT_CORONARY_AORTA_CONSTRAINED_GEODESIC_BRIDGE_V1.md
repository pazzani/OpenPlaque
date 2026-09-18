# Left-Coronary Aorta-Constrained Geodesic Bridge v1

## Purpose

The current multivessel research summary is complete. RCA is quantitatively locked, LAD anatomy is strong with technically feasible direct PCAT, and C6/C7 research structural labels are frozen with technically feasible direct PCAT. The remaining major anatomy problem is the unresolved proximal left-coronary connection to the aortic root.

Earlier local beam-search and blind ostium-discovery experiments did not establish a second aortic exit. A later parent-continuation experiment produced a source-supported path that moved away from the aorta, and through-vessel continuity showed that the candidate re-entered known LAD rather than establishing a distinct left-main trunk.

This experiment is therefore deliberately different: it asks whether a global source-CCTA geodesic constrained by cardiac chamber masks can connect the accepted LAD endpoint to the aortic-root surface.

## Scientific design

The algorithm uses the following source-space evidence:

- frozen Series-7 source CCTA;
- high-resolution aorta mask;
- high-resolution atrial and ventricular masks;
- pulmonary artery mask;
- frozen RCA source centerline as a positive control;
- frozen LAD source centerline as the target anatomy.

The aortic-root target surface is defined without using the LAD. The known RCA proximal endpoint identifies the vertical/root band, and the search is allowed around the root circumference.

## Positive control first

The same method is first run from a point approximately 10 mm down the known RCA.

For the experiment to be interpretable, both prespecified cost variants must recover a path that:

- remains close to the known proximal RCA;
- reaches the known RCA root neighborhood;
- passes source-plane tubular QC;
- agrees across cost variants.

If the RCA positive control fails, the LAD result is not interpreted.

## Geodesic cost

The method does not use the older whole-root Frangi component detector.

Instead it performs minimum-cost path search directly in source CCTA using:

- source HU;
- distance-to-background within contrast-bright voxels as a local centrality term;
- hard exclusion of the four cardiac chambers;
- hard exclusion of the pulmonary artery;
- hard exclusion of the aortic interior except for the target surface.

Two prespecified bright-tube cost variants are used:

- 140-HU bright threshold;
- 180-HU bright threshold.

The variants are not selected after the result. Agreement between them is part of the acceptance gate.

## LAD bridge gates

After the RCA control passes, the two LAD paths are evaluated for:

- source-plane QC fraction at least 0.65;
- fraction of source center samples at least 180 HU at least 0.70;
- tortuosity at most 1.8;
- symmetric cross-variant median path separation at most 2.0 mm;
- symmetric cross-variant p90 path separation at most 4.0 mm;
- aortic target separation at most 4.0 mm.

A positive status is:

LEFT_CORONARY_AORTA_CONSTRAINED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED

A negative but interpretable status is:

LEFT_CORONARY_AORTA_GEODESIC_NO_STABLE_LAD_ROOT_BRIDGE

If the RCA control fails:

LEFT_CORONARY_AORTA_GEODESIC_RCA_CONTROL_FAILED

## Post-hoc relationship to C6/C7 work

If the optional C6 source path is available, the accepted LAD-to-root bridge is compared post hoc with that path.

This relationship is not used to select or accept the geodesic.

## Scientific boundary

Even a positive result does not establish clinical left-main identity.

A positive result would establish only a source-supported proximal left-coronary bridge candidate from the accepted LAD endpoint to the aortic-root surface.

Clinical left-main identity would still require evidence that this bridge represents the common parent before independently established LAD and LCX/OM branches rather than simply a proximal LAD continuation or another contrast structure.

The frozen master anatomy is never modified by this experiment.

## Outputs

The experiment writes:

- RCA control paths for both cost variants;
- LAD aortic bridge paths for both cost variants;
- source-plane QC CSVs;
- gate table;
- MPR-style source image;
- 3-D geometry figure;
- center-HU QC figure;
- summary JSON;
- run-state JSON;
- provenance JSON;
- HTML report;
- ZIP archive.
