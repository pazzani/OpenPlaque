# Final integrated research report v2

This branch is the corrected synthesis layer for OpenPlaque after Master Coronary Anatomy Baseline v2 and the validated LAD longitudinal plaque profile.

Scientific hierarchy:

- Master Anatomy v2 is authoritative for downstream vessel eligibility: RCA accepted, LAD accepted at 24.997 mm, LM unresolved, LCX unresolved.
- The older 57.276-mm LAD is registration provenance only and must not be used as the quantitative LAD.
- Native 5-fold plaque confidence volumes remain curved-series model-coordinate quantities, not source-space TPV.
- The earlier longitudinal plaque/PCAT fusion is retained for RCA only.
- The LAD section is replaced by `LAD_Validated_Longitudinal_Plaque_Profile_v1`, including its validated 17.63–21.48 mm support zone.
- Locked RCA PCAT 10–50 mm remains unchanged.
- No 3-D source-space plaque mask, circumferential plaque localization, LAD PCAT association, LM quantification, LCX quantification, or Caristo FAI-Score claim is made.

The report fixes the v1 packaging bug by writing `run_state.json` with status `COMPLETE` before creating the ZIP.
