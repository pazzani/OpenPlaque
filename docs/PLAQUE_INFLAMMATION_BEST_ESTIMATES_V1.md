# Plaque + Inflammation Best Estimates v1

## Purpose

Create a single research endpoint for the user's UCLA CCTA that is easy to compare with quantitative CCTA publications while preserving OpenPlaque's evidence boundaries.

Fresh branch from frozen main:
`0593b453959f5a353d644267fbeef24b514ef4d7`.

## Plaque outputs

Per vessel: LAD, RCA, LCX, LM.

Reported:
- total plaque volume (TPV) best-estimate proxy
- strict/known lower evidence
- candidate-envelope upper estimate when available
- noncalcified plaque volume (NCPV) reconstructed as all plaque bins below 350 HU
- low attenuation plaque (LAP) proxy
- 30-130 HU noncalcified/fibrofatty proxy
- 130-350 HU fibrous/intermediate proxy
- calcified plaque volume
- CONFIRM2 TPV and NCPV stages

The legacy plaque bins are:
- LAP proxy: -30 to <30 HU
- noncalcified/fibrofatty: 30 to <130 HU
- fibrous/intermediate: 130 to <350 HU
- calcified: >=350 HU

Dense calcium is counted only from the strict plaque core in the legacy method. The point TPV proxy is strict MM-DHM plaque core plus selected vessel-context candidates, rescaled to Series-7 source voxel volume.

### Critical plaque boundaries

- These volumes are **research proxies**, not Cleerly AI-QCT outputs.
- CONFIRM2 quantified the whole coronary tree, including eligible branches; our aggregate covers LAD/RCA/LCX/LM territories and is therefore named a **major-vessel aggregate**, not literal whole-heart TPV.
- PAV is not reported without a defensible source-space outer-vessel volume.
- LAP is reported, but high-risk plaque is not adjudicated because positive remodeling is not validated in this endpoint.
- LM 54 mm3 is a clinical calcium-volume anchor plus Series-7 ROI cross-check. It is not a complete LM total-plaque segmentation.

## CONFIRM2 alignment

van Rosendael et al., American Journal of Preventive Cardiology 2026, article 101712, DOI 10.1016/j.ajpc.2026.101712.

TPV staging:
- 0 mm3
- >0-250 mm3
- >250-750 mm3
- >750 mm3

NCPV staging:
- 0-20 mm3
- >20-125 mm3
- >125-375 mm3
- >375 mm3

The publication's quantitative analysis used an FDA-cleared AI-QCT platform and all coronary segments of adequate caliber. Stage labels in OpenPlaque are therefore contextual comparisons only.

## SCCT/NASCI alignment

Shaw et al. J Cardiovasc Comput Tomogr. 2021;15(2):93-109. DOI 10.1016/j.jcct.2020.11.002.

The endpoint emphasizes plaque presence, burden/composition, LAP, and explicit reporting boundaries. Positive remodeling, napkin-ring sign, spotty calcification and CAD-RADS are not invented when they have not been validated.

## Inflammation outputs

Direct source-space PCAT attenuation is reported for:
- RCA: locked 10-50 mm segment
- LAD: frozen validated available segment
- LCX: C6 structural parent as the current best LCX-like estimate, with C7/OM-like daughter retained as an alternate structural branch
- LM: not standardized / not estimated

The raw PCAT adipose window corresponding to the FAI literature is -190 to -30 HU, with perivascular radial extent tied to vessel diameter.

Caristo/ORFAN standardized FAI-Score is **not reproduced**. Published FAI-Score uses a proprietary algorithm that adjusts raw FAI for technical scan parameters, anatomy/biology, age, and sex. CaRi-Heart risk is also not reproduced.

References:
- Chan et al. Lancet 2024;403:2606-2618. DOI 10.1016/S0140-6736(24)00596-8.
- Oikonomou et al. Cardiovasc Res. 2021;117:2677-2690. DOI 10.1093/cvr/cvab286.
- CRISP-CT methodology: proximal RCA 10-50 mm; proximal LAD and LCX 40 mm; LM not assessed because of variable length.

## Research interpretation

This endpoint is designed for transparent comparison and longitudinal OpenPlaque research. It is not a diagnostic report and should not be represented as output from Cleerly, Caristo, or any FDA-cleared commercial platform.
