# Left proximal-trunk expanded-field reacquisition v1

This experiment follows the positive dense source-CCTA QC of a 20.0-mm vessel continuation beyond the frozen accepted LAD endpoint.

## Motivation

The prior long-reach LAD→aorta search had adequate path-length budget but its search field was built with only a 7-mm margin. Attrition instrumentation showed a sharp rise in `reject_outside_field` near the same arc where subsequent dense orthogonal-plane QC established a real proximal continuation. The coronary-mask gate itself rejected zero source-passing proposals.

## Prospective change

The search now starts from the independently dense-QC accepted proximal-trunk endpoint, about 29.88 mm from the aortic surface. The source-field margin is expanded from 7 mm to 18 mm and the search budget is 40 mm, giving at least 8 mm of prospective slack over the measured straight-line requirement.

The previous beam rules are retained: 0.40-mm step, beam width 110, same source HU range, same vesselness gate, same 1.6-mm coronary-mask proximity gate except when already near the aorta, same turning rule, and the same monotonic aortic-distance rule. No C6/LCX geometry is used to steer the search.

## Controls and adjudication

The canonical RCA is rerun through the same fixed orthogonal-plane QC as an independent positive control. The best expanded-field path is evaluated by dense orthogonal source-CCTA QC every 0.4 mm. Three consecutive failed planes prospectively truncate the accepted extension; an extension must be at least 5 mm long with at least 80% plane-pass fraction.

If a candidate reaches the aortic surface, dense QC must cover the full path and its endpoint must be at least 8 mm from the known RCA proximal endpoint before it can be nominated for visual adjudication.

## Scientific boundary

A positive result establishes only a source-supported proximal-trunk continuation or proximal-trunk-to-aorta candidate. It does not establish clinical LM identity, does not resolve LCX topology, and does not modify the frozen Master Coronary Anatomy Baseline v2.
