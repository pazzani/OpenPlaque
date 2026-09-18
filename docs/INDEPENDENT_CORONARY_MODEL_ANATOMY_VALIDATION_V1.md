# Independent Coronary Model Anatomy Validation v1

This experiment tests whether an independently trained coronary-lumen model provides new evidence about the unresolved proximal left-coronary anatomy in the UCLA Series 7 CCTA.

## External observer

The initial observer is CAS-Net as released with the ImageCAS-X benchmark (Bransby et al., 2026; Zenodo record 21887809).

ImageCAS-X provides branch-named annotations, but its benchmark dataloader binarises them to lumen versus background for these released segmentation models. Therefore this experiment tests independent lumen continuity/topology only. It cannot by itself establish clinical LM, LCX, or OM identity.

## Frozen baseline

Fresh branch directly from:

`0593b453959f5a353d644267fbeef24b514ef4d7`

The master must remain `CORONARY_ANATOMY_BASELINE_V2_FROZEN`. This experiment never writes to the master.

## Prespecified gates

Fixed before viewing the model output:

- path-to-model tolerance: 1.25 mm
- model/aorta contact tolerance: 1.50 mm
- RCA support >= 0.70
- frozen LAD support >= 0.70
- C6 support >= 0.60
- C7 support >= 0.50 is descriptive only

The RCA is the control. The model is interpretable on this scan only if it supports the RCA and an RCA-supporting connected component reaches the aorta.

With a passing RCA control, the left-coronary result is positive only if the LAD and C6 support gates pass, LAD and C6 share an independent-model connected component, and that same component reaches the aorta.

Statuses:

- `INDEPENDENT_MODEL_RCA_CONTROL_FAILED`
- `INDEPENDENT_MODEL_LEFT_CORONARY_AORTIC_CONTINUITY_POSITIVE`
- `INDEPENDENT_MODEL_NO_LEFT_CORONARY_AORTIC_CONTINUITY`

Do not relax gates after seeing the result.

## Inputs

Persistent OpenPlaque Drive inputs:

- Series 7 source cache
- frozen LAD centerline
- RCA source centerline
- candidate-04/C6 path
- C7 extended path
- high-resolution TotalSegmentator aorta mask
- frozen master summary

The Colab writes a geometry-faithful Series 7 NIfTI, runs the independent ImageCAS-X model, then returns its prediction to original source geometry before OpenPlaque topology analysis.

## Outputs

`/content/drive/MyDrive/OpenPlaque/Independent_Coronary_Model_Anatomy_Validation_v1/`

Includes run state, summary/decision/provenance JSON, path and gate CSVs, QC PNGs, HTML report, and a ZIP archive.

## Scientific boundary

A positive result is independent learned-lumen evidence for a continuous left-coronary route to the aortic root and can justify reopening scientific review of the unresolved origin. It does not automatically modify the master or assign clinical LM/LCX/OM labels.

A negative result is interpretable only if the RCA control passes.

Research use only; not for clinical diagnosis.
