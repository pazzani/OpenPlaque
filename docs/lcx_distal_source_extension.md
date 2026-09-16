# LCX distal source extension

This experiment follows the unresolved hierarchical LCX alternatives after the first decisive distal bifurcation.

## Scientific question

The hierarchical experiment selected the C9/C6/C7 family over C11/C5, but the next split remained unresolved because C7 had better local AV-groove geometry while C9/C6 had much straighter continuity. This experiment asks a longer-range question: after the split, which alternative continues as a source-supported coronary tube and remains more consistent with the left atrioventricular groove over 10–20 mm?

## Method

- Fresh branch from the frozen OpenPlaque baseline.
- Reuses the source CCTA cache but **does not reuse** the historical secondary-branch vesselness ROI because it is spatially unrelated to the current C7/C9/C6 endpoints.
- Builds two small endpoint-centered source-CCTA crops:
  - C7 (source candidate 7)
  - the common C9/C6 trunk
- Computes multiscale 3-D bright-tube vesselness using the previously validated scale family: 0.55, 0.80, 1.10, 1.45 mm.
- Each seed must first pass a 2.5-mm positive-control reacquisition of its known terminal segment.
- The distal graph search is target-free.
- The historical seed corridor is blocked except for the terminal launch neighborhood so the graph cannot backtrack and escape through a proximal branch.
- Candidate continuation selection within each seed uses **source evidence only**: vesselness, length, current/legacy coronary support, source HU, smoothness, and launch continuity.
- Left-atrium/left-ventricle anatomy is evaluated only after the source-supported path has been selected.
- Old curved-template LCX similarity has zero decision weight.

## Decision

Both alternatives must produce source-supported extensions of at least 10 mm. A difference of at least 0.08 in the longitudinal AV-groove score can nominate one continuation for visual/anatomic QC. Otherwise the result remains ambiguous.

No Master Coronary Anatomy update is performed. LM remains unresolved.
