# Blind multi-seed source-CCTA ostium recovery v2

## Motivation

The completed v1.2 run finished successfully but failed its required RCA positive control. Five blind root components survived initial component gates, four were traced, and none was accepted. The closest traced path to the known RCA had median distance about 3.83 mm, p90 about 5.77 mm, seed distance about 5.65 mm, plane-pass fraction 1/9, and median serial-plane radius about 2.64 mm. Therefore no conclusion about a second coronary exit is permitted from v1.2.

The failure pattern suggests that a single minimum-distance seed per connected root component is too brittle when vesselness closing joins multiple surface-contact neighborhoods into one component.

## Prospective v2 change

This experiment branches directly from the frozen baseline commit `0593b453959f5a353d644267fbeef24b514ef4d7`.

The source-CCTA HU/vesselness candidate definition, RCA lumen calibration, serial-plane acceptance gates, RCA post-hoc control criteria, and second-exit separation criteria are retained. The prospective change is limited to root-candidate parameterization:

- every connected component may generate up to 10 near-surface seed hypotheses;
- seed points must be within 2.0 mm of the aortic surface;
- seeds are greedily separated by at least 2.5 mm in physical space;
- each seed is traced from several blind source-derived initial directions (local component PCA, aortic-distance gradient, and blends);
- beam paths are sampled only at 6, 9, and 12 mm checkpoints;
- at most three endpoint-distinct candidates per seed/checkpoint undergo serial orthogonal-plane QC.

Known RCA, LAD, and C6 coordinates do not participate in seed generation, beam scoring, candidate acceptance, or blind selection. The known RCA is used only for calibration already present in the prior experiment and for the post-search positive-control test. LAD/C6 are post-hoc only if a second accepted exit survives after the RCA control passes.

## Gates

The RCA positive control is unchanged: the closest blind accepted hypothesis must have median distance to known RCA <=1.0 mm, p90 <=1.8 mm, and seed distance to the known RCA proximal endpoint <=3.0 mm.

A second exit is considered only after that control passes. It must be an independently accepted blind hypothesis, have median distance to known RCA >4.0 mm, and its seed must be at least 8.0 mm from the RCA seed.

A positive v2 result still does not establish clinical left-main identity and does not modify Master Coronary Anatomy Baseline v2.1. It only nominates a source-CCTA root exit for visual QC and a later common-parent connectivity experiment.
