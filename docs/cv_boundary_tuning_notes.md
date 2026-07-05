# Cross-validation tuning method

The v0.5 workflow treats each DHM labeled case as an evaluation target. nnU-Net predictions are generated for validation folds, then boundary-refinement parameters are applied to those predictions.

Each refined mask is compared with the expert label using:

- Dice
- IoU
- Precision
- Recall
- TPV error

The default composite score is:

```text
0.35 * Dice
+ 0.20 * IoU
+ 0.20 * (1 - min(abs TPV error fraction, 1))
+ 0.15 * Precision
+ 0.10 * Recall
```

The best parameter set is the one with the highest mean cross-validated score.

## Practical note

The notebook defaults to `MAX_CASES_TOTAL = 30` so the workflow can be tested quickly. Increase this value, or set it to `None`, when running a full parameter-selection experiment.
