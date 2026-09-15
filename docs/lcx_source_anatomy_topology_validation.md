# LCX source-space anatomy/topology validation

This experiment is a fresh branch from the frozen OpenPlaque baseline. It evaluates the five strongest source-path hypotheses nominated by the prior joint three-vessel experiment, but it does **not** use curved-template similarity in the primary anatomy score or anatomy gates.

Primary evidence:

- sustained divergence from the accepted proximal LAD rather than merely sharing the artificial proximal-LAD anchor used by the prior graph search;
- branch angle relative to the accepted LAD;
- current and legacy TotalSegmentator coronary-mask support, including dual-mask agreement;
- sustained distal separation from the accepted LAD and lack of rejoining;
- separation from accepted RCA;
- exit from the aortic mask after divergence;
- source-CCTA contrast support;
- path smoothness and tortuosity.

The prior LCX-template score/margin is retained only as supporting metadata and is excluded from the anatomy score and all anatomy gates.

A passing result is still only an LCX source-path candidate requiring visual and independent anatomical confirmation. Master Coronary Anatomy Baseline v2 keeps LM and LCX unresolved. This experiment does not establish circumferential registration, source-space plaque localization, or plaque volume.
