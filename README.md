# OpenPlaque v0.2: Auto-Series + TPV Uncertainty + HTML Report

Research-use-only OpenPlaque update that adds:

- automatic LAD/RCA/LCX curved-reformat series detection
- TPV uncertainty interval: high-confidence core TPV → raw AI TPV, with refined TPV as the working estimate
- self-contained HTML report with TPV tables and raw/refined/core overlay images
- fixed notebook import order so `detect_artery_series` is defined before use

## Main notebook

`notebooks/05_End_to_End_OpenPlaque_v0_2_AutoSeries_Uncertainty_HTML.ipynb`

## Colab use

1. Upload this ZIP to:

```text
/content/drive/MyDrive/OpenPlaque/OpenPlaque_v0_2_AutoSeries_Uncertainty_HTML.zip
```

or upload it directly to `/content`.

2. Open the notebook and run from the top.

The notebook will clone/update OpenPlaque, then use the bundled v0.2 modules if the ZIP is available.

Expected data files remain:

```text
/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip
/content/drive/MyDrive/OpenPlaque/models/Dataset001_CCTA_DHM-20260703T233210Z-3-001.zip
```

## Output

The notebook writes results to:

```text
/content/drive/MyDrive/OpenPlaque/Segmentations/
```

including:

- raw plaque masks
- refined plaque masks
- `tpv_boundary_refinement_summary.txt`
- `openplaque_tpv_boundary_report.html`

## Notes

The UCLA fallback series are retained only as a fallback:

```python
{"RCA": 1035, "LCX": 1039, "LAD": 1043}
```

If DICOM metadata includes artery names in `SeriesDescription`, `ProtocolName`, or related fields, those are used first.

Not clinically validated. Not for diagnosis or treatment decisions.
