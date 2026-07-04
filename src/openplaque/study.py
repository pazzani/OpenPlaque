"""
OpenPlaque Study class
"""

import os
import zipfile
from collections import defaultdict

import pydicom
import SimpleITK as sitk


class OpenPlaqueStudy:
    def __init__(self, zip_path, extract_root="/content/full_dicom"):
        self.zip_path = zip_path
        self.extract_root = extract_root
        self.series = []
        self._extract()
        self._scan()

    def _extract(self):
        if not os.path.exists(self.extract_root):
            with zipfile.ZipFile(self.zip_path) as z:
                z.extractall(self.extract_root)

    def _scan(self):
        groups = defaultdict(list)

        for root, _, files in os.walk(self.extract_root):
            for f in files:
                path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                    groups[ds.SeriesInstanceUID].append(path)
                except Exception:
                    pass

        self.series = []

        for uid, paths in groups.items():
            ds = pydicom.dcmread(paths[0], stop_before_pixels=True, force=True)
            self.series.append({
                "uid": uid,
                "series_number": int(getattr(ds, "SeriesNumber", -1)),
                "description": str(getattr(ds, "SeriesDescription", "")),
                "images": len(paths),
                "folder": os.path.dirname(paths[0]),
            })

        self.series.sort(key=lambda x: x["series_number"])

    def summary(self):
        for s in self.series:
            print(f"{s['series_number']:>5} {s['images']:>5} {s['description']}")

    def find(self, text):
        text = text.lower()
        return [s for s in self.series if text in s["description"].lower()]

    def load_series(self, series_number):
        match = [s for s in self.series if s["series_number"] == series_number]
        if not match:
            raise ValueError(f"Series {series_number} not found")

        folder = match[0]["folder"]

        reader = sitk.ImageSeriesReader()
        sid = reader.GetGDCMSeriesIDs(folder)[0]
        files = reader.GetGDCMSeriesFileNames(folder, sid)
        reader.SetFileNames(files)

        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image)

        return image, volume, files
