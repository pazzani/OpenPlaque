# LCX curved-template source reacquisition

This experiment uses the historical curved-series mask `LCX.nii.gz` from Series 1039 as an **unlabeled coronary template** to help reacquire a source-CCTA path.

Scientific rules:

- Fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- The stale historical `LCX_source_centerline.csv` is never read.
- Master Anatomy v2 must still list LCX as unresolved.
- Candidate source paths are generated only from the current + legacy TotalSegmentator coronary masks near the proximal accepted LAD, with the accepted RCA excluded.
- The old Series-1039 mask contributes only a longitudinal fingerprint: mask support, plaque-support landmarks, and curved-series CT attenuation.
- The same matcher must first recognize the accepted RCA using the old RCA Series-1035 template.
- The LCX template is also scored against accepted RCA and accepted LAD as wrong-vessel controls.
- A template match does **not** establish LCX identity. A passing result is promoted only to `LCX_CURVED_TEMPLATE_MATCHES_SOURCE_PATH_REQU%ITS_ANATOMICAL_QC`.
- No circumferential registration, plaque volume, or source-space lesion localization is claimed.

Outputs include candidate source centerlines, control scores, candidate ranking, automatic source-CCTA orthogonal QC, an HTML report, and a packaged ZIP.
