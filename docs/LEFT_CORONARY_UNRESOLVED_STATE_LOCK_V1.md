# Left-Coronary Unresolved-State Research Lock v1

## Purpose

This experiment does not search for another vessel.

It audits the accumulated left-coronary source-CCTA evidence and determines whether the unresolved clinical LM / LCX / OM state should now be locked as unresolved rather than repeatedly retuned on the same scan.

The frozen master anatomy is never modified.

## Why a lock is needed

The current evidence contains a reproducible topology conflict.

The accepted frozen LAD joins the candidate-04/C6 path with essentially zero endpoint gap and near-through-vessel geometry. Independent dense source-CCTA adjudication previously confirmed that continuity and showed that the earlier parent hypothesis re-entered known LAD.

At the same time, several independent attempts to establish a left-coronary origin at the aorta have failed:

- mask-consensus ostial search did not establish a second coronary exit despite a passing RCA interface control;
- blind and multiseed ostium searches did not establish a second exit;
- parent-recovery tracing produced a source-supported continuation but no meaningful progress toward the aorta;
- global aorta-constrained geodesic tracing passed its RCA positive control but found no acceptable LAD-to-root bridge;
- local root-directed geodesic tracing also passed its RCA control but found no finite LAD route under the prespecified local constraints.

The later C6/C7 source-QC freeze validates those exact paths as research structures only. It explicitly does not establish clinical LCX/OM identity and leaves LM unresolved.

## Core lock criteria

The lock requires all of the following:

- frozen master status unchanged;
- LAD-C6 junction gap <= 0.5 mm;
- recomputed through-vessel deflection <= 25 degrees;
- prior dense through-vessel QC pass fraction >= 0.90;
- prior parent recovery showed no meaningful aortic progress;
- mask-consensus experiment did not establish a second ostial exit and had a passing RCA control;
- global geodesic experiment was negative with a passing RCA control;
- local root-directed experiment was negative with a passing RCA control;
- C6/C7 remain research-only structural labels.

The blind and multiseed ostium experiments are retained as supporting negative evidence, but their failed RCA controls prevent them from being treated as decisive negatives.

## Candidate-04 classifier diagnostic

The earlier template classifier score for candidate 04 is reported descriptively.

It is not used as a lock gate because template-score separation was weak and the later structural adjudication used stronger topology/chamber/source evidence.

## Positive lock status

The expected positive result is:

LEFT_CORONARY_UNRESOLVED_STATE_RESEARCH_LOCKED

This means:

- clinical LM remains unresolved;
- clinical LCX/OM identity remains unresolved;
- C6 = LCX-like parent continuation and C7 = OM-like daughter remain research structural descriptors only;
- no additional same-scan heuristic parameter tuning should be performed to try to convert the current ambiguity into a clinical vessel label.

## What could reopen the question

The lock can be revisited if genuinely independent evidence becomes available, such as:

- independent expert coronary annotation;
- a validated trained coronary segmentation / centerline model;
- another reconstruction or cardiac phase that directly shows the proximal left-coronary origin;
- external clinical vessel labels or centerlines.

## Downstream policy

The lock does not invalidate established research measurements.

- RCA research plaque and PCAT remain usable under the existing lock.
- Frozen LAD direct PCAT remains usable under its feasibility boundary.
- C6/C7 measurements may continue only as exact-path research structural measurements.
- Clinical LM quantification remains disabled.
- Clinical LCX/OM quantification remains disabled.

## Outputs

- evidence matrix CSV
- topology-conflict 3-D figure
- evidence-matrix figure
- summary JSON
- decision JSON
- provenance JSON
- HTML report
- ZIP archive

## Scientific boundary

This is a research workflow lock, not a clinical diagnosis.

Its purpose is to prevent same-scan overfitting: repeated heuristic retuning should not be allowed to transform unresolved source topology into an unsupported clinical identity.
