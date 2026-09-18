# RCA Global Outer-Wall Surface v1

## Question

Can a single explicit outer-wall surface, optimized jointly over longitudinal arc and circumferential angle, improve plaque specificity beyond both the successful fixed-shell proxy and the failed independent per-ray adaptive-wall model?

## Motivation

The prior adaptive-wall experiment succeeded geometrically but failed specificity: nominal outer-wall QC was 0.972 with median direct-fat edge fraction 0.528, yet majority-plaque AUROC was 0.783 and strict 5/5 AUROC 0.733. The independent ray model also showed substantial dependence on strict/nominal/liberal detector settings.

Inspection of the ray diagnostics showed a structural reason: about half of the rays had direct transitions into perivascular fat, while many fallback gradient detections collapsed to the minimum wall thickness. An anatomical wall surface should be spatially coherent around the circumference and along the vessel rather than estimated independently ray by ray.

## Design

This experiment returns to the original accepted RCA source geometry used by the successful fixed-shell method. It does **not** use the later recentered/tangent-refined RCA geometry.

The method builds one cylindrical surface over:

- longitudinal RCA station;
- circumferential angle;
- wall thickness outside the measured lumen boundary.

### Source-derived anchors

At every station and angle, radial source-CCTA HU samples are examined outside the lumen.

Two label-blind anchor types are allowed:

1. **Direct outer-wall anchor:** the earliest convincing transition from wall-like tissue into perivascular fat (< -30 HU).
2. **Weak gradient anchor:** only when a direct fat edge is absent, a high-confidence negative HU transition may provide a low-weight anchor.

Direct anchors receive substantially higher weights than fallback gradient anchors.

### Global surface optimization

The wall thickness surface is obtained by minimizing a quadratic energy containing:

- weighted agreement with direct/weak source anchors;
- circumferential smoothness between neighboring rays;
- longitudinal smoothness between neighboring stations;
- a tiny global stabilizing prior equal to the median direct-anchor wall thickness.

No plaque labels, plaque-vote maps, PCAT values, or holdout outcomes enter the surface optimization.

Three prospectively fixed regularization variants are evaluated:

- flexible: angular 0.40, longitudinal 0.15;
- nominal: angular 0.80, longitudinal 0.35;
- smooth: angular 1.60, longitudinal 0.70.

The nominal variant is primary.

## Surface QC

A nominal station passes if:

- original RCA lumen QC passes;
- direct anchors cover at least 10% of rays;
- direct + weak anchors cover at least 25% of rays;
- weighted surface-to-anchor MAE <=0.60 mm;
- within-station surface-thickness IQR <=0.90 mm.

The primary geometry gate requires:

- overall station surface-QC fraction >=0.85;
- median direct-anchor fraction >=0.25;
- median weighted anchor error <=0.45 mm.

## Plaque model

Plaque HU composition is integrated only between the original directional lumen surface and the optimized global outer-wall surface.

The same research HU bins are retained:

- low attenuation: -30 to <30 HU;
- noncalcified: 30 to <130 HU;
- mixed/intermediate: 130 to <350 HU;
- calcified: >=350 HU.

A robust normal-wall composition model is fitted only in the prespecified RCA 20–50 mm reference region. The prior longitudinal plaque map is checked before execution to confirm that this reference region still contains no >=3/5 plaque-vote voxels and no majority-positive bins.

The fitted model is then evaluated only in the disjoint 0–20 mm holdout.

## Prospective specificity gates

The nominal global-surface model passes only if:

- majority-positive vs vote-free AUROC >=0.80;
- strict 5/5 vs vote-free AUROC >=0.90 when at least two strict bins exist;
- median excess in majority-positive bins is >=2x the vote-free median;
- minimum flexible/smooth profile Spearman correlation versus nominal >=0.80.

A status of \`RCA_GLOBAL_OUTER_WALL_SURFACE_SPECIFICITY_GAIN\` additionally requires majority AUROC improvement >=0.02 relative to the successful fixed 1.0-mm shell method.

## Outputs

The experiment writes:

- station-level global-surface geometry for all regularization variants;
- nominal ray-level anchor and fitted-surface diagnostics;
- reference-model coefficients;
- longitudinal excess-plaque profiles;
- holdout validation tables;
- regularization-sensitivity metrics;
- global surface map;
- source-CCTA overlay QC;
- optional fusion with locked RCA PCAT;
- summary/run-state JSON, HTML report, and ZIP archive.

## Boundary

This is still a developmental single-scan experiment. The global surface is label-blind, but the RCA itself has already been used for method development. A pass would justify freezing this explicit surface algorithm and testing it on an independent scan or untouched dataset; it would not establish validated clinical TPV.
