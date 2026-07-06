# Boundary parameter tuning with cached predictions

The nnU-Net model is fixed. Boundary refinement is downstream post-processing. Therefore nnU-Net prediction should be run once per case, not once per parameter combination.

The notebook uses a prediction cache:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/predictions/
```

Set `OVERWRITE_PREDICTIONS = True` only when intentionally regenerating predictions.

## Parameters searched

- `min_component_voxels`: removes plaque components smaller than this number of voxels.
- `lumen_distance_voxels`: removes plaque touching/near the lumen label within this many voxels; `0` disables it.
- `high_hu_threshold`: removes predicted plaque above this HU threshold; `None` disables it.
- `low_hu_threshold`: removes predicted plaque below this HU threshold; `None` disables it.
- `closing_radius_voxels`: morphological closing radius to smooth/fill small breaks in plaque masks.
- `fill_holes`: fills enclosed holes within plaque components.
- `min_plaque_length_mm`: removes short plaque components using approximate axial/centerline extent.
- `connectivity`: 3D connectivity used for components and morphology: 6, 18, or 26.
- `adaptive_hu_thresholds`: uses lumen attenuation to set broad HU trimming bounds.
- `erode_core`: fixed false for main parameter selection; conservative core masks can be generated separately.
- `erosion_iterations`: fixed 1.

## Metrics

Each refined prediction is compared with the expert plaque label using Dice, IoU, precision, recall, and TPV error. The default score is:

```text
0.35*Dice + 0.20*IoU + 0.20*(1 - min(abs TPV error fraction, 1)) + 0.15*Precision + 0.10*Recall
```


## Compressed Drive archive

The notebook also writes a reusable ZIP archive:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Parameter_Tuning/nnunet_prediction_cache.zip
```

On a later clean Colab runtime, if this file exists, the notebook restores `predictions/` from the ZIP and skips nnU-Net inference. This is useful because boundary-parameter tuning is downstream of prediction and does not require rerunning the neural network.
