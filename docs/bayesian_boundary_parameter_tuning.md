# Bayesian boundary-parameter tuning

This version replaces exhaustive expanded-grid search with Optuna Bayesian optimization.

## Why

The expanded grid can exceed 10,000 combinations. Because each combination must be evaluated on every labeled artery, exhaustive search can be slow even after nnU-Net predictions are cached. Bayesian optimization samples promising regions of the parameter space and can usually find good parameter sets with far fewer evaluations.

## Prediction cache

nnU-Net inference is independent of boundary-refinement parameters. The notebook therefore runs nnU-Net once per case and saves predictions to:

```text
Boundary_Parameter_Tuning_Bayesian/nnunet_prediction_cache.zip
```

Later clean Colab runs restore this archive and skip inference.

## No cross-validation

All labeled sample cases are used for evaluating each candidate parameter set. This is parameter selection on a fixed deterministic post-processing algorithm, not neural-network training.

## Search parameters

The optimizer searches component size, lumen-distance trimming, HU thresholds, morphology, component length, connectivity, and adaptive HU thresholding. Core erosion remains fixed off for the main TPV estimate.
