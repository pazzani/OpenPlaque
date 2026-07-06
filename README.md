# OpenPlaque Repository Additions: 5-Fold CV Boundary Tuning

This ZIP contains files intended to be committed into the OpenPlaque GitHub repository.

It removes the notebook dependency on copying code from a ZIP. After these files are committed, the clean Colab notebook simply clones/pulls OpenPlaque from GitHub and imports `openplaque.cv_boundary_tuning` directly.

## Files to add/update

```text
src/openplaque/boundary.py
src/openplaque/cv_boundary_tuning.py
notebooks/07_5Fold_CV_Boundary_Parameter_Selection_Clean_Colab_GitHub.ipynb
tests/test_cv_boundary_tuning.py
docs/5fold_cv_boundary_tuning_notes.md
```

`src/openplaque/__init__.py` does not need to be modified for the notebook because it imports directly from `openplaque.cv_boundary_tuning`. If you want package-level exports, add:

```python
try:
    from .cv_boundary_tuning import *
except Exception:
    pass
```

## Main notebook

```text
notebooks/07_5Fold_CV_Boundary_Parameter_Selection_Clean_Colab_GitHub.ipynb
```

The notebook:

1. Mounts Google Drive.
2. Clones/pulls OpenPlaque from GitHub.
3. Installs dependencies.
4. Verifies `openplaque.cv_boundary_tuning` is present in the GitHub checkout.
5. Extracts the nnU-Net model from Drive.
6. Downloads or locates the labeled sample dataset.
7. Collects paired `*_0000.nii.gz` images and matching label masks.
8. Generates or reuses nnU-Net predictions.
9. Runs 5-fold supervised parameter-selection cross-validation.
10. Saves CSV/JSON/HTML reports to Drive.

## Expected labeled sample data

Old sample dataset layout:

```text
Sample_Dataset/
  P02_LAD_axial_0000.nii.gz
  P02_LAD_axial.nii.gz
```

or nnU-Net raw layout:

```text
Dataset001_CCTA_DHM/
  imagesTr/*_0000.nii.gz
  labelsTr/*.nii.gz
```

## Outputs

```text
/content/drive/MyDrive/OpenPlaque/CV_Boundary_Tuning/
  predictions/
  cv_all_case_results.csv
  cv_heldout_results.csv
  cv_selected_by_fold.csv
  cv_best_boundary_parameters.json
  cv_report.html
```

## Test

```bash
PYTHONPATH=src pytest -q tests/test_cv_boundary_tuning.py
```

Research use only. Not clinically validated and not for clinical decision-making.
