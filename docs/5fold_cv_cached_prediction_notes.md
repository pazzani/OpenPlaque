# Cached prediction 5-fold CV notes

Boundary refinement is post-processing. Therefore nnU-Net inference should be run once per case, not once per parameter combination.

The notebook uses:

```text
/content/drive/MyDrive/OpenPlaque/CV_Boundary_Tuning/predictions/
```

as the prediction cache. If all expected prediction files are present, `nnUNetv2_predict` is skipped.

Set `OVERWRITE_PREDICTIONS = True` in the notebook only when you intentionally want to regenerate predictions.
