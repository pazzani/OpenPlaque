# LCX Structural Source-QC Freeze v1

## Purpose

The prior LCX structural adjudication established, under all predeclared evidence gates, the following **research structural labels**:

- C6: LCX-like parent continuation
- C7: OM-like daughter

That adjudication intentionally stopped at REQUIRES_VISUAL_QC. Clinical LCX/OM identity remained unestablished, LM remained unresolved, and the frozen master anatomy was not modified.

This experiment replaces subjective manual review with a deterministic dense source-CCTA orthogonal-plane QC step.

## Inputs

- frozen master anatomy baseline;
- Series-7 source CCTA;
- prior LCX Structural Identity Adjudication v1 summary;
- C6 source path from Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv;
- C7 extended source path from LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv.

No new vessel search is performed.

## Dense source-plane QC

Only the post-split segments are evaluated. The first 1.0 mm after the split is excluded from compactness gating because orthogonal planes in an immediate bifurcation zone can legitimately contain both branches.

The remaining paths are sampled every 0.5 mm.

For each station:

- a local PCA-smoothed tangent is computed;
- a source-CCTA orthogonal plane is implied by the tangent;
- center HU is sampled;
- 72 radial rays are traced from the center;
- the lumen boundary is estimated using the same contrast-relative threshold used in prior source-space geometry work;
- radial support, p10/p50/p90 radius and p90/p10 axis proxy are measured.

A station passes when:

- center HU >=200;
- valid radial fraction >=0.60;
- axis proxy <=2.50.

## Vessel-level gates

Each structural branch passes dense source QC only if:

- station pass fraction >=0.85;
- median axis proxy <=2.00;
- p90 axis proxy <=2.50;
- median center HU >=250.

The run automatically displays representative stations plus the worst compactness stations. No sliders or manual station selection are used.

## Freeze decision

A positive result has status:

LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN

This freezes only OpenPlaque **research structural bookkeeping**:

- C6 = LCX-like parent continuation
- C7 = OM-like daughter

It does **not**:

- establish clinical LCX identity;
- establish clinical OM identity;
- resolve the left main;
- modify Master Coronary Anatomy Baseline v2.

A positive result allows research-only source-space quantification on these exact C6/C7 paths while preserving the explicit provisional labels.

## Why this follows the LAD experiment

The LAD distal-reference self-calibration substantially improved developmental plaque discrimination relative to direct RCA-model transfer, but still failed the prespecified strict-plaque and shell-stability gates. Further tuning against already inspected LAD labels would add overfitting risk.

The project therefore moves to the next unresolved anatomy problem rather than relaxing LAD gates.
