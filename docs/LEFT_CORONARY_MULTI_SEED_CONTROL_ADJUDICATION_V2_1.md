# Left coronary source-ostium multi-seed RCA-control adjudication v2.1

This is a focused follow-up to `Left_Coronary_Source_Ostium_Multiseed_v2`.

## Why this experiment exists

The v2.0 blind multi-seed run generated 25 surface-seed hypotheses, traced 22, and blindly accepted 2. Both accepted hypotheses belonged to component 305 and were close to the known RCA in post-hoc geometry. However, v2.0 chose the *single geometrically closest traced path* as its RCA-control representative before checking whether that path had passed the frozen blind acceptance gate. The closest traced path (component 305, seed 3) was not blind-accepted, so the positive control was reported as failed even though separate blind-accepted paths existed nearby.

That ordering is a control-adjudication bug, not a reason to change discovery thresholds.

## Prospective v2.1 rule

1. Freeze the v2.0 accepted-hypothesis set exactly as written in `blind_multiseed_root_hypotheses.csv`.
2. Deterministically regenerate only those blind-accepted component/seed hypotheses with the same source CCTA, vesselness cache, beam search, and serial-QC rules.
3. Apply known-RCA geometry only after blind acceptance.
4. The RCA positive control passes if **any** blind-accepted hypothesis satisfies all unchanged control geometry limits:
   - median distance to proximal known RCA <= 1.0 mm
   - p90 distance <= 1.8 mm
   - seed distance to known RCA proximal endpoint <= 3.0 mm
5. After a control representative is established, apply the same v2.0 second-exit exclusions to the remaining blind-accepted hypotheses:
   - median distance to known RCA > 4.0 mm
   - seed separation from the RCA-control seed >= 8.0 mm

No HU threshold, vesselness threshold, beam-search parameter, serial-plane gate, lumen-radius gate, or blind-acceptance rule is relaxed or changed.

## Interpretation

A recovered positive control establishes that the multi-seed blind parameterization can rediscover the known RCA under the frozen acceptance rules. If the control is recovered but no independent accepted exit remains, the experiment concludes that this particular blind aortic-root detector did not establish a second coronary ostial exit. That negative result does not negate the separately established LAD/LCX anatomy; it only constrains this discovery method.

The Master Coronary Anatomy Baseline remains unchanged.