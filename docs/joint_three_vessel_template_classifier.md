# Joint three-vessel curved-template classifier

This experiment is a fresh branch from the frozen OpenPlaque baseline and is deliberately separate from the earlier LCX-only reacquisition experiment.

## Purpose

Use the historical curved-series RCA, LAD and LCX image/mask pairs jointly. RCA and the accepted LAD are calibration vessels. The LCX-labeled historical curved series is treated only as an unlabeled coronary template until independent source-space anatomy supports its identity.

## Method

- Master Coronary Anatomy Baseline v2 remains authoritative.
- RCA and LAD accepted source centerlines are the only positive calibration paths.
- Cached curved RCA/LAD/LCX CT NIfTIs and cached nnU-Net masks provide longitudinal template fingerprints.
- RCA source length calibrates source-mm per curved longitudinal pixel.
- Source paths are represented by orthogonal tube median/p90 HU profiles rather than centerline HU alone.
- Several predeclared feature recipes are evaluated only on RCA/LAD correct-vessel versus wrong-vessel separation.
- The best RCA/LAD recipe is frozen before any LCX candidate ranking.
- Candidate source paths come from the current+legacy TotalSegmentator coronary union near the proximal accepted LAD while excluding the accepted RCA corridor.
- Each candidate receives RCA, LAD and LCX template scores. LCX candidacy requires successful RCA/LAD calibration plus a distinct LCX margin over both known-vessel templates.
- No stale historical LCX source centerline is read.

## Scientific boundary

A passing result is only an LCX source-path candidate requiring independent anatomical/topological QC. It does not establish circumferential curved-series registration, source-space plaque localization, noncalcified plaque volume, total plaque volume, LM identity, or LCX identity by itself.
