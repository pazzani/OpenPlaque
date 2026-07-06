# 5-fold CV method

For every sample case, the workflow compares a refined nnU-Net prediction with the corresponding expert label mask.

Metrics:

- Dice
- IoU
- Precision
- Recall
- TPV error

Composite score:

```text
0.35 * Dice
+ 0.20 * IoU
+ 0.20 * (1 - min(abs TPV error fraction, 1))
+ 0.15 * Precision
+ 0.10 * Recall
```

For each CV fold, parameters are selected using the other four folds and then evaluated on the held-out fold. The final deployable parameters are selected using all labeled sample cases.
