# OpenPlaque v0.3 — Boundary Tuning, Standalone Notebook

This ZIP updates the `06_Boundary_Refinement_Parameter_Tuning.ipynb` notebook so it can run in a fresh Colab/runtime.

## Main notebook

```text
notebooks/06_Boundary_Refinement_Parameter_Tuning.ipynb
```

The notebook now performs the full setup before tuning:

1. Mounts Google Drive.
2. Defines Drive paths for the DICOM study ZIP, nnU-Net model ZIP, and output folder.
3. Clones the OpenPlaque repository.
4. Installs Colab requirements and extra packages.
5. Writes the bundled v0.3 modules into `/content/OpenPlaque/src/openplaque`.
6. Configures `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
7. Verifies GPU with `nvidia-smi`.
8. Extracts the nnU-Net model weights.
9. Loads the full DICOM study.
10. Automatically detects LAD/RCA/LCX curved reformat series.
11. Runs segmentation.
12. Runs boundary-refinement parameter tuning.
13. Saves CSV, JSON, tuning HTML, final TPV HTML report, and tuned NIfTI masks.

Default expected Drive files:

```text
/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
```

Outputs are written to:

```text
/content/drive/MyDrive/OpenPlaque/Segmentations/
```

Research use only. Not for clinical decision-making.
