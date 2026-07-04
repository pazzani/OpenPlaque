# Example

```python
from openplaque.study import OpenPlaqueStudy

study = OpenPlaqueStudy("/content/drive/MyDrive/OpenPlaque/Full_DICOM.zip")

study.summary()

lad = study.find("LAD")
image, volume, files = study.load_series(lad[0]["series_number"])

print(volume.shape)
```
