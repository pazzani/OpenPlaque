# Vendor Q3D Proximal Origin Audit v1

## Purpose

Audit Siemens-derived coronary Q3D radial curved reformats as **same-exam vendor-derived structural evidence** for proximal coronary origin.

Series are fixed from the CCTA study inventory:

- Series 1035 / folder 32218: RCA Curved Range Radial Q3D(MT)
- Series 1039 / folder 32219: CX Curved Range Radial Q3D(MT)
- Series 1043 / folder 32220: LAD Curved Range Radial Q3D(MT)

Each contains 24 512x512 derived images.

## Critical geometry rule

The 24 Q3D files are treated as radial curved views around a vessel, **not as 24 physical slices of a 3-D volume**.

The prior plaque notebook used these series as fallback artery inputs and substituted Series-7 spacing when the derived-image spacing was implausible. That prior workflow is not used for proximal-origin geometry.

## Layer 1: DICOM provenance and geometry discovery

The experiment writes:

- q3d_geometry_inventory.csv
- q3d_tag_inventory.csv
- q3d_private_tag_inventory.csv
- q3d_source_reference_mapping.csv

It checks for standard PixelSpacing / ImageOrientationPatient / ImagePositionPatient geometry and recursively inventories source-image references and private tags.

No 3-D endpoint coordinate is invented when the curved reformats do not provide a trustworthy standard spatial mapping.

## Layer 2: RCA-calibrated pixel-space origin signature

The radial renderer axis is detected from all 72 images.

For every view:

- an adaptive bright-lumen threshold is chosen
- the vessel corridor is localized near image center
- bright-width and connected bright-component expansion are measured at both ends
- an origin score is computed at each end

Series 1035 RCA is the positive control. The end that most consistently shows the stronger origin signature in RCA defines the renderer's proximal side.

RCA control requires:

- proximal-side consistency >= 0.75
- proximal/distal median score ratio >= 1.20

LAD or CX is called origin-signature positive only if:

- RCA control passes
- same-side consistency >= 0.60
- proximal/distal ratio >= 1.10
- RCA-calibrated origin index >= 0.50

The LAD-vs-CX proximal profile correlation is descriptive only.

## Statuses

- Q3D_BOTH_LEFT_ORIGIN_SIGNATURES_PRESENT
- Q3D_LAD_ORIGIN_SIGNATURE_PRESENT
- Q3D_CX_ORIGIN_SIGNATURE_PRESENT
- Q3D_NO_LEFT_ORIGIN_SIGNATURE
- Q3D_RCA_ORIGIN_CONTROL_FAILED

Even a positive left signature does not establish clinical LM/LCX/OM identity. The frozen master is never modified.


## v1.1 adjudication correction

The first completed run forced every Q3D image onto one global renderer axis. This caused a false RCA-control failure because several radial rotations were horizontal but were measured vertically.

Version 1.1 corrects that implementation error without changing the underlying image thresholding or origin-score definition.

Changes:

- every radial image is measured along its own automatically detected vessel axis
- images with max end-origin score < 0.50 or end-score margin < 0.25 are uninformative rather than negative
- views i and i+12 are treated as repeated radial-angle pairs
- an RCA angle is allowed to orient LAD/CX only when both repeated RCA views are informative and agree on which image end is proximal
- at least 6 valid RCA angle pairs are required
- RCA repeated-pair side reproducibility must be >= 0.75
- corresponding LAD/CX angles are evaluated using the fixed RCA-derived proximal end
- at least 6 target angles are required
- paired raw-pixel correlations are reported as a descriptive repeatability check

The output folder is separated from the original run:
`Vendor_Q3D_Proximal_Origin_Audit_v1_1`.

This remains same-exam vendor-derived evidence only. It cannot establish clinical LM/LCX/OM identity by itself.
