# SegmentationReport example

```python
from openplaque.study import OpenPlaqueStudy
from openplaque.segmentation import segment_vessel

study = OpenPlaqueStudy("/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip")

image_lad, volume_lad, _ = study.load_series(1043)

lad_report = segment_vessel(image_lad, volume_lad, "LAD")

lad_report.summary()
lad_report.show_overlay(label=2)

lad_report.save_mask(
    "/content/drive/MyDrive/OpenPlaque/Segmentations/LAD_plaque_segmentation.nii.gz"
)
```
