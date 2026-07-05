# v0.2 implementation notes

## Automatic artery detection

The detector scores available series metadata for artery aliases and curved-reformat keywords. It supports common OpenPlaque-style study attributes and methods, and can sample DICOM ZIP metadata if the study object exposes a ZIP path.

## TPV uncertainty

The reported interval is intentionally conservative:

- low bound: eroded high-confidence core TPV
- working estimate: refined TPV
- high bound: raw AI TPV

This is not a statistical confidence interval. It is a boundary-sensitivity interval.

## HTML report

The HTML report embeds PNG overlays as base64 data URIs, so the output is portable as a single file.
