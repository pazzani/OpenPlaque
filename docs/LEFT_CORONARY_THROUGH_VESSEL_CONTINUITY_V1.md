# Left-Coronary Through-Vessel Continuity v1

Research-use experiment from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

## Motivation

The bifurcation-derived parent experiment produced a 30-mm, 100%-dense-QC source path, but it made no aortic progress and visually tracked the frozen LAD. At the putative LAD/common-trunk junction, the two outgoing tangents were almost antiparallel, which is more consistent with one continuous vessel than with two daughter branches.

This experiment tests that interpretation prospectively rather than launching another vessel search.

## Inputs

- frozen accepted LAD centerline;
- dense-QC accepted 20-mm proximal/common-trunk continuation;
- established C6 and extended C7 paths;
- prior bifurcation-parent result and target path;
- Series 7 source CCTA;
- frozen Master Coronary Anatomy Baseline v2.

## Prospective gates

The LAD-to-proximal/common-trunk junction is classified as a source-supported through-vessel only if:

1. endpoint gap <= 0.50 mm;
2. the validated proximal segment reproduces the established C6/C7 common trunk (>=80% within 2 mm and median tangent alignment >=0.80, with minimum distance <=1 mm);
3. straight-through deflection is <=25 degrees;
4. dense orthogonal source-CCTA QC across +/-8 mm of the junction passes >=90% of planes with no three consecutive failures;
5. a real-branch control at the established C6/C7 split reproduces the parent-vs-side-branch pattern (parent <=25 degrees from incoming, side branch >=35 degrees, separation >=25 degrees).

The previous parent candidate is separately cross-walked against the frozen LAD. Re-entry requires minimum distance <=1.5 mm, >=20% of the path within 2 mm, >=5 mm contiguous within 2 mm, and median tangent alignment >=0.75.

## Scientific boundary

A positive result establishes only that the frozen-LAD path and validated common-trunk segment behave as one continuous source-supported vessel at this junction, and/or that the prior parent candidate re-entered known LAD. It does **not** establish or change clinical LAD, LM, LCX, or OM identity. It does not modify the frozen master anatomy.

## Outputs

`summary.json`, `continuity_decision.json`, dense-QC CSV, through-path CSV, C6/C7 split-control JSON, parent-candidate/LAD crosswalk CSV and distance profile, three PNG QC figures, HTML report, `run_state.json`, and a ZIP archive.
