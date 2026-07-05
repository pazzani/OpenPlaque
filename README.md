# OpenPlaque v0.4 Working Boundary Tuning

This release replaces the earlier placeholder boundary-tuning code with executable code.

## What is included

- `src/openplaque/tuning.py` — real grid-search tuning code.
- `notebooks/06_Boundary_Refinement_Parameter_Tuning_STANDALONE_WORKING.ipynb` — fresh-runtime Colab notebook.
- `tests/test_tuning_working.py` — smoke test on synthetic plaque masks.

## Main notebook variables

After running the tuning cell, these are defined:

```python
tuning_results          # pandas DataFrame, one row per vessel/parameter candidate
selected_params         # best global parameter dictionary
best_by_vessel          # best row for LAD/RCA/LCX individually
selected_refinements    # refined masks using selected_params
```

Display the best parameters:

```python
from openplaque.tuning import best_parameters, best_rows_by_vessel

selected_params = best_parameters(tuning_results)
print(selected_params)
display(best_rows_by_vessel(tuning_results))
```

Outputs are saved to:

```text
/content/drive/MyDrive/OpenPlaque/Boundary_Tuning/
```

Research use only. Not for clinical decision-making.
