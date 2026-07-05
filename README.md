# OpenPlaque v0.5 Cross-Validation Boundary Tuning

This release selects boundary-refinement parameters using labeled DHM-style sample data instead of tuning on a single UCLA patient study.

## Main notebook

```text
notebooks/07_CV_Boundary_Parameter_Selection_Clean_Colab.ipynb
```

The notebook is designed for a clean Colab GPU runtime. It:

1. Mounts Google Drive.
2. Extracts this v0.5 ZIP and adds `src/openplaque` to Python path.
3. Installs Python dependencies.
4. Configures nnU-Net folders.
5. Locates/extracts `Dataset001_CCTA_DHM/imagesTr` and `labelsTr` sample data.
6. Extracts nnU-Net model weights if available.
7. Creates K-fold validation splits.
8. Runs nnU-Net prediction for each validation fold.
9. Applies every boundary-refinement parameter set.
10. Compares each refined prediction against the expert label.
11. Selects parameters by cross-validated Dice/IoU/TPV/precision/recall score.
12. Writes CSV, JSON, and HTML reports.

## Expected data layout

```text
/content/drive/MyDrive/OpenPlaque/Dataset001_CCTA_DHM/
  imagesTr/*_0000.nii.gz
  labelsTr/*.nii.gz
```

or a ZIP at:

```text
/content/drive/MyDrive/OpenPlaque/Dataset001_CCTA_DHM.zip
```

Model weights are expected at:

```text
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
```

## Output

```text
/content/drive/MyDrive/OpenPlaque/CV_Boundary_Tuning/
  predictions/
  cv_boundary_tuning_case_results.csv
  cv_boundary_tuning_summary.csv
  cv_best_boundary_parameters.json
  cv_boundary_tuning_report.html
```

## Best parameters

After the notebook runs:

```python
best_cv_params
cv_summary.head(10)
```

The selected parameter JSON can be loaded later and applied to UCLA or any other new patient.

## Important

Research use only. Not clinically validated. Cross-validation improves parameter defensibility, but outputs still require visual QC and expert review.


## GitHub-based clean Colab update

The notebook `notebooks/07_CV_Boundary_Parameter_Selection_Clean_Colab.ipynb` now clones or updates `https://github.com/pazzani/OpenPlaque.git` directly. It does not require `OpenPlaque_v0_5_CV_Boundary_Tuning.zip` inside Colab. Make sure the GitHub repo contains `src/openplaque/cv_tuning.py` and the current `boundary.py` before running.
