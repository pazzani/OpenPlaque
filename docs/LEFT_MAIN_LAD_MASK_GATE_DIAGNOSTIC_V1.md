# Left-main LAD mask-gate diagnostic v1

## Purpose

The feasible long-reach LAD-to-aorta experiment corrected the previous 20 mm reachability bug by allowing 46 mm of search for an accepted LAD endpoint 36.04 mm from the aortic surface. Despite adequate physical reach, the search still returned `no_path_reached_aorta` and left only four frontier states.

This experiment determines which class of gate is responsible before changing any accepted-anatomy rule.

## Prospective design

The branch is created directly from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

1. Re-run the unchanged calibrated RCA proximal retrace positive control.
2. Reproduce the long-reach left field from the same accepted LAD proximal endpoint.
3. Instrument the exact prior hard-mask beam and count proposal losses from turn, aortic-distance monotonicity, loop, ROI, HU, vesselness, and coronary-mask-proximity gates.
4. Run a shadow beam with exactly one change: proposals more than 1.6 mm from the current/legacy coronary masks are no longer rejected. The same HU range, vesselness threshold, turning constraint, monotonic aortic-distance rule, step size, beam width, and original mask-support score are retained.
5. Save the best frontier/path, mask-distance profile, source orthogonal-plane QC, and hard-vs-shadow aortic-distance curves.

The shadow run is diagnostic only. It cannot update or establish left-main anatomy.

## Interpretation

- `LAD_MASK_GATE_ATTRITION_CONFIRMED_SHADOW_REACHES_AORTA`: the source-supported beam can reach the aorta when the hard TotalSegmentator mask-proximity rejection is removed. The next prospective experiment should validate a source-only bridge with explicit serial lumen QC rather than reinstate the truncated mask as a hard gate.
- `LAD_SOURCE_SUPPORT_FAILS_EVEN_WITH_MASK_GATE_REMOVED`: removing the hard mask gate is insufficient. The next step should inspect source/vesselness attrition, field geometry, or the accepted LAD proximal endpoint rather than relax the mask criterion further.
- `LAD_HARD_MASK_SEARCH_UNEXPECTEDLY_REACHED_AORTA`: the instrumented reproduction contradicts the prior run and should be reconciled before another experiment.

Master Coronary Anatomy Baseline v2 remains unchanged in all cases.
