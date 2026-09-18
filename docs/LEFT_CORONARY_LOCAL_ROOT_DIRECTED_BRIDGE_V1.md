# Left-Coronary Local Root-Directed Bridge v1

## Why this experiment exists

The prior global aorta-constrained geodesic experiment passed its RCA positive control but failed to establish a stable LAD-to-root bridge.

The failure was not caused by poor source-plane signal. Both LAD geodesic variants were extremely similar, had 100% plane QC, and remained highly contrast-enhanced. Instead, the paths were approximately 86.8 mm long with tortuosity about 1.84 and visibly retraced the known LAD before taking a long high-HU route toward the RCA-root target.

The prespecified tortuosity gate correctly rejected that solution. The threshold is not relaxed.

This follow-up is conceptually different: a true proximal bridge must leave the proximal LAD endpoint and progress locally toward the nearest aortic surface rather than travel backward along known distal coronary anatomy.

## Design

The same frozen source-CCTA and high-resolution cardiovascular masks are used.

The algorithm:

- orients the frozen RCA and LAD with the endpoint nearest the aorta first;
- defines a local aortic-surface target around the nearest surface point to the search start;
- constrains search to a 12-mm-radius corridor around the straight start-to-local-aorta segment;
- hard-excludes the deep interior of atrial, ventricular, and pulmonary-artery masks, while leaving boundary neighborhoods traversable so mask dilation cannot create an artificial epicardial barrier;
- excludes the aortic interior except for the local target surface;
- excludes already-known downstream coronary centerline beyond the launch neighborhood;
- requires progressive reduction in distance to the aortic surface;
- evaluates two fixed source-HU cost variants, 140 HU and 180 HU.

No manual coordinates, sliders, or result-dependent target selection are used.

## RCA positive control

The method is first launched from approximately 10 mm down the known RCA.

For both variants the recovered root-directed path must:

- pass source-plane tubular QC;
- remain close to the known proximal RCA;
- reach the known RCA root neighborhood;
- have tortuosity at most 1.35;
- make predominantly monotonic progress toward the aortic surface;
- agree with the other cost variant.

If the RCA control fails, the LAD result is not interpreted. If a constrained search has no finite route, that variant is recorded as a failed gate rather than raising an exception.

## LAD gates

For each LAD variant:

- source-plane QC fraction at least 0.65;
- at least 70% of center samples at or above 180 HU;
- tortuosity at most 1.60;
- at least 80% of successive path steps do not increase aortic-surface distance by more than 0.35 mm;
- maximum backtracking from the best prior aortic-surface distance at most 2.0 mm;
- target remains inside the prespecified local aortic-surface neighborhood.

Across the two variants:

- median symmetric path separation at most 2.0 mm;
- p90 separation at most 4.0 mm;
- endpoint separation at most 4.0 mm.

## Interpretation

Possible statuses:

- LEFT_CORONARY_LOCAL_ROOT_DIRECTED_RCA_CONTROL_FAILED
- LEFT_CORONARY_LOCAL_ROOT_DIRECTED_NO_STABLE_BRIDGE
- LEFT_CORONARY_LOCAL_ROOT_DIRECTED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED

A positive result is still not clinical left-main identification.

It would establish only that a reproducible, locally root-directed, source-supported proximal bridge exists from the accepted LAD endpoint to a nearby aortic surface neighborhood.

The next anatomical question after a positive result would be whether that bridge converges with the independently supported C6/C7 left-coronary-like branch system.

The frozen master anatomy is never modified by this experiment.

## Outputs

- RCA local-control paths and QC tables
- LAD local root-directed paths and QC tables
- gate table
- 3-D geometry figure
- aortic-surface progress figure
- center-HU figure
- automatically selected LAD source-plane montage
- summary, provenance and run-state JSON
- HTML report
- ZIP archive
