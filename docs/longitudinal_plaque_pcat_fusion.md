# Longitudinal plaque-confidence + PCAT fusion

This experiment is intentionally one-dimensional.

Inputs:
- `Canonical_Source_Coronary_Centerlines_v1`
- `Curved_Plaque_to_Source_Registration_Canonical_v1`
- `Plaque_Ensemble_Confidence_Atlas_v1`
- `PCAT_RCA_10_50_Reproducibility_Lock`

The native curved-series plaque longitudinal profile is mapped to canonical source-centerline arc using only the validated longitudinal registration parameters (`long_axis`, `start_px`, `n_pixels`, and `long_flip`). The workflow rejects a native-axis mismatch and re-checks the longitudinal score/gradient gates.

RCA plaque support is aggregated into 1-mm canonical source-arc bins and joined to the locked 1-mm RCA PCAT profile from 10–50 mm. LAD plaque support is mapped to canonical source arc but is not fused with PCAT because this experiment has no locked LAD PCAT profile.

The plaque outputs are native curved-series support counts after longitudinal remapping. They must not be interpreted as anatomical source-space plaque volume, TPV, circumferential plaque position, or a 3-D source-space plaque mask.
