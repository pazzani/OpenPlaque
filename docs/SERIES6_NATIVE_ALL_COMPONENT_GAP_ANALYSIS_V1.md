# Series 6 Native All-Component Gap Analysis v1

This experiment follows the completed native Series-6 ostium topology result, which found one qualifying native coronary/aorta contact and identified it as the RCA neighborhood.

## Scientific question

Do any **left-associated portions** of the native Series-6 CAS-Net coronary components approach the native aorta closely but stop short of contact?

This is a diagnostic experiment. It does not create a geodesic, synthetic vessel, or segmentation bridge.

## Inputs reused

- native Series-6 CCTA DICOM, BestSyst 32%
- cached Series-6 CAS-Net prediction
- cached native Series-6 TotalSegmentator aorta
- cached Series-7 CAS-Net prediction and frozen Series-7 source aorta for comparison
- frozen LAD, RCA, C6 and C7 research paths

## Path-local component analysis

The root-region Series7→Series6 translation is recomputed only to localize the known research paths in systolic Series 6. For each CAS-Net connected component, component voxels within **4 mm** of each transformed path are treated as a local path-associated subset.

A path/component subset must contain at least **20 voxels**.

For every qualifying subset the experiment reports:

- component/path association size
- exact native closest distance to the aortic surface
- closest model and aorta coordinates
- straight-line source-CCTA HU samples across the gap
- automatic orthogonal source planes through the gap

The aortic distance uses a physical-space KD-tree built from the aortic **surface only**, avoiding a multi-gigabyte whole-volume nearest-point field.

## Prespecified descriptive thresholds

- short gap: <= 5 mm
- gap HU sample spacing: 0.25 mm
- contrast-support heuristic: >=60% of interior gap samples >=150 HU

The contrast heuristic is research-only and does not establish lumen continuity clinically.

## Statuses

- `SERIES6_LEFT_REGION_SHORT_GAP_CONTRAST_SUPPORTED`
- `SERIES6_LEFT_REGION_SHORT_GAP_NO_CONTRAST_SUPPORT`
- `SERIES6_LEFT_REGION_NO_SHORT_GAP`
- `SERIES6_LEFT_REGION_LOCALIZATION_FAILED`
- `SERIES6_ROOT_REGISTRATION_FAILED`

No status automatically establishes clinical LM/LCX/OM identity. Frozen master is never modified.
