"""
OpenPlaque segmentation utilities.

This module provides a higher-level SegmentationReport object instead of
passing plain dictionaries around.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class SegmentationReport:
    name: str
    image: sitk.Image
    mask_image: sitk.Image
    volume: np.ndarray
    mask: np.ndarray
    input_dir: str
    output_dir: str
    model: str = "Dataset001_CCTA_DHM"

    @property
    def spacing(self):
        return self.mask_image.GetSpacing()

    @property
    def voxel_volume_mm3(self):
        return float(np.prod(self.spacing))

    @property
    def plaque_voxels(self):
        return int(np.sum(self.mask == 2))

    @property
    def vessel_voxels(self):
        return int(np.sum(self.mask == 1))

    @property
    def tpv_mm3(self):
        return self.plaque_voxels * self.voxel_volume_mm3

    @property
    def vessel_volume_mm3(self):
        return self.vessel_voxels * self.voxel_volume_mm3

    def label_counts(self):
        labels, counts = np.unique(self.mask, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    def hu_statistics(self):
        hu = self.volume[self.mask == 2]
        if len(hu) == 0:
            return None
        return {
            "count": int(len(hu)),
            "min": float(hu.min()),
            "max": float(hu.max()),
            "mean": float(hu.mean()),
            "median": float(np.median(hu)),
        }

    def summary(self):
        print(f"SegmentationReport: {self.name}")
        print(f"Model: {self.model}")
        print(f"Spacing: {self.spacing}")
        print(f"Voxel volume: {self.voxel_volume_mm3:.4f} mm³")
        print(f"Vessel voxels: {self.vessel_voxels}")
        print(f"Plaque voxels: {self.plaque_voxels}")
        print(f"Vessel volume: {self.vessel_volume_mm3:.2f} mm³")
        print(f"Total plaque volume: {self.tpv_mm3:.2f} mm³")
        print("Labels:", self.label_counts())

    def best_plaque_slice(self):
        counts = np.sum(self.mask == 2, axis=(1, 2))
        return int(np.argmax(counts))

    def show_overlay(self, label=2, z=None, alpha=0.6, vmin=-200, vmax=800):
        import matplotlib.pyplot as plt

        if z is None:
            z = self.best_plaque_slice()

        plt.figure(figsize=(8, 8))
        plt.imshow(self.volume[z], cmap="gray", vmin=vmin, vmax=vmax)
        plt.imshow(self.mask[z] == label, alpha=alpha)
        plt.axis("off")
        plt.title(f"{self.name}: label {label}, slice {z}")
        plt.show()

    def save_mask(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sitk.WriteImage(self.mask_image, path)
        print("Saved:", path)


def export_for_nnunet(image, output_dir, case_name):
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{case_name}_0000.nii.gz")
    sitk.WriteImage(image, outfile)
    return outfile


def run_nnunet(input_dir,
               output_dir,
               dataset="Dataset001_CCTA_DHM",
               configuration="3d_fullres",
               folds=(0, 1, 2, 3, 4)):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", dataset,
        "-c", configuration,
        "-f",
    ] + [str(f) for f in folds]

    subprocess.run(cmd, check=True)


def load_segmentation(output_dir, case_name):
    mask_image = sitk.ReadImage(os.path.join(output_dir, f"{case_name}.nii.gz"))
    mask = sitk.GetArrayFromImage(mask_image)
    return mask_image, mask


def segment_vessel(image,
                   volume,
                   vessel_name,
                   dataset="Dataset001_CCTA_DHM",
                   work_root="/content"):
    """
    Segment one curved coronary artery and return a SegmentationReport.

    Parameters
    ----------
    image:
        SimpleITK image for the curved vessel series.
    volume:
        NumPy array corresponding to image.
    vessel_name:
        Case name, e.g. "LAD", "RCA", "LCX".
    """

    input_dir = os.path.join(work_root, f"{vessel_name}_input")
    output_dir = os.path.join(work_root, f"{vessel_name}_output")

    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

    export_for_nnunet(image, input_dir, vessel_name)
    run_nnunet(input_dir, output_dir, dataset=dataset)

    mask_image, mask = load_segmentation(output_dir, vessel_name)

    return SegmentationReport(
        name=vessel_name,
        image=image,
        mask_image=mask_image,
        volume=volume,
        mask=mask,
        input_dir=input_dir,
        output_dir=output_dir,
        model=dataset,
    )
