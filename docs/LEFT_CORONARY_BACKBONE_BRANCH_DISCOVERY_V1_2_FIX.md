# Left-coronary backbone branch discovery v1.2 — C7 control seed robustness

## Why v1.2 exists

v1.1 completed successfully as a technical run but stopped at the required C7 positive control. This is not a negative anatomic result because blind branch discovery never ran.

The nominal control station was 45.2467 mm along the label-neutral backbone. At that station the backbone/C6 control seed and the C7 reference split are separated by about 0.80 mm. Most of that displacement is cross-sectional rather than longitudinal. A single centerline seed is therefore a brittle test of a real bifurcation whose separately reconstructed daughter centerlines are sub-millimeter misregistered.

The failed v1.1 control supports this diagnosis: some rays stayed near C7 for only ~1.5 mm before dense QC failed, while other rays had excellent dense QC but tracked a different bright vessel trajectory. The control therefore failed before any blind inference was allowed.

## Technical change

For every requested backbone station, including the known C7 control and every blind station, v1.2 samples a small deterministic source-space seed neighborhood:

- backbone arc offsets: -0.4, 0, +0.4 mm;
- cross-sectional positions: center plus an eight-point ring at radius 0.8 mm;
- two top preview directions per local start are collected;
- only the same six expensive beam traces used previously are retained globally, with start-point diversity.

Thus the search budget remains bounded while the branch-origin seed is robust to a sub-millimeter discrepancy between independently reconstructed centerlines.

## What does not change

All prospective scientific gates remain exactly those of v1.0/v1.1:

- branch-search length and step size;
- source-HU gate;
- source-space vesselness threshold calibration;
- curvature/loop and backbone re-entry rules;
- minimum accepted branch length;
- dense orthogonal source-plane QC thresholds;
- minimum branch angle;
- endpoint separation;
- C7 reference-overlap control gate;
- no aortic or clinical-label information in discovery/ranking;
- frozen anatomy remains unchanged.

v1.1 source-space vesselness and one-point failed-path guards remain active.

## Interpretation

If the C7 control now passes, the blind scan may be interpreted under the predeclared branch gates. If it still fails, no blind result is interpretable; the branch-discovery method itself needs redesign rather than threshold relaxation.
