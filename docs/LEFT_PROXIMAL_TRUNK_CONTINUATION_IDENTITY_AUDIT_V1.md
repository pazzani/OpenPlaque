# Left Proximal-Trunk Continuation Identity Audit v1

## Question

The earlier dense orthogonal-plane QC experiment accepted 20.0 mm of source-supported vessel beyond the frozen LAD endpoint. Subsequent local and long source-led experiments unexpectedly followed the established C6/C7 common-trunk geometry and then returned onto the frozen LAD. This experiment therefore audits the identity of the original 20-mm accepted continuation itself.

It asks whether that 20-mm path is genuinely new proximal LAD/LM anatomy or whether it is already represented by the established C6/C7 common trunk.

## Prospective design

The experiment is purely geometric and uses already frozen/generated paths. No new vessel search is performed.

The accepted 20-mm continuation is compared bidirectionally with the C6/C7 common trunk (candidate 04 from arc 0 to 20.25 mm) and with the frozen LAD. Paths are resampled at 0.25 mm. The identity gate requires all of:

- minimum continuation-to-common-trunk distance <= 1.0 mm;
- >= 80% of the continuation within 2.0 mm of the common trunk;
- >= 15 mm contiguous continuation span within 2.0 mm;
- median absolute tangent alignment >= 0.90 within the 2-mm band;
- >= 80% reverse common-trunk coverage within 2.0 mm of the continuation;
- absolute continuation-arc to common-trunk-arc mapping correlation >= 0.90;
- one continuation endpoint within 2.0 mm of a frozen LAD endpoint;
- the opposite continuation endpoint within 2.0 mm of the C6/C7 split endpoint;
- those endpoint anchors must be distinct;
- continuation/common-trunk length difference <= 4.0 mm.

A positive result is `VALIDATED_20MM_CONTINUATION_REIDENTIFIED_AS_C6_C7_COMMON_TRUNK`.

## Scientific boundary

A positive result changes interpretation of the investigational continuation only. It does not establish clinical LCX/OM identity, left main, or a coronary ostium, and it does not modify Master Coronary Anatomy Baseline v2.

If the path is reidentified as C6/C7 common-trunk anatomy, the appropriate next LM experiment is to return to the LAD/common-trunk junction and search for a third common-parent direction rather than extending the reidentified branch.

## Outputs

- `run_state.json`
- `summary.json`
- `identity_metrics.csv`
- `identity_gate_matrix.csv`
- `continuation_reference_profiles.csv`
- `endpoint_topology.json`
- `01_continuation_identity_distance_profile.png`
- `02_identity_geometry_projections.png`
- `OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_IDENTITY_AUDIT_REPORT.html`
- `OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_IDENTITY_AUDIT_RESULTS.zip`
