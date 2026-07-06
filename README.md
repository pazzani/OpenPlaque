# OpenPlaque v0.6 — 5-Fold CV Boundary Parameter Selection

This release uses labeled sample data to select boundary-refinement parameters by 5-fold cross-validation.

## Main notebook

```text
notebooks/07_5Fold_CV_Boundary_Parameter_Selection_Clean_Colab.ipynb
```

The notebook runs from a clean Colab GPU runtime. It:

1. Mounts Google Drive.
2. Clones/pulls OpenPlaque from GitHub.
3. Installs requirements.
4. Writes the v0.6 CV tuning module into the GitHub checkout, so no ZIP upload is needed inside Colab.
5. Extracts the nnU-Net model weights from Drive.
6. Downloads or locates the labeled sample dataset.
7. Finds image/label pairs such as:

```text
P02_LAD_axial_0000.nii.gz   # CT input image
P02_LAD_axial.nii.gz        # label mask, values 0/1/2
```

8. Runs nnU-Net prediction once for all sample cases, using cached predictions when available.
9. Evaluates every boundary-refinement parameter combination against the labels.
10. Performs true 5-fold parameter-selection CV:
    - choose best parameters on 4 folds
    - test those parameters on the held-out fold
11. Selects final parameters using all sample cases.
12. Saves CSV, JSON, and HTML reports.

## Outputs

```text
/content/drive/MyDrive/OpenPlaque/CV_Boundary_Tuning_v06/
  predictions/
  cv_all_case_parameter_results.csv
  cv_full_dataset_parameter_summary.csv
  cv_fold_assignments.csv
  cv_heldout_selected_parameter_results.csv
  cv_selected_parameters_by_fold.csv
  cv_best_boundary_parameters.json
  cv_boundary_tuning_report.html
```

## Data expected

The notebook first tries to download the old sample dataset Google Drive folder used by `06_BoundaryRefinement_Tuning.ipynb`. It also supports:

```text
/content/drive/MyDrive/OpenPlaque/Sample_Dataset
/content/drive/MyDrive/OpenPlaque/Sample_Dataset.zip
/content/drive/MyDrive/OpenPlaque/Dataset001_CCTA_DHM
/content/drive/MyDrive/OpenPlaque/Dataset001_CCTA_DHM.zip
```

## Model expected

```text
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
```

## Important

Research use only. Not clinically validated. The selected parameters should be visually checked before applying to new patient studies.
