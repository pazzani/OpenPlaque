"""
OpenPlaque coronary segmentation helpers.

This module intentionally does NOT vendor or copy third-party model code.
It provides small Colab-friendly helpers for:
  1. exporting a loaded SimpleITK CTA volume to NIfTI,
  2. cloning peer-reviewed/open-source coronary segmentation repositories,
  3. running placeholder inference commands that the user can adapt after
     model weights and repo-specific paths are available.

Research software only. Not for clinical use.
"""

from pathlib import Path
import subprocess
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


THIRD_PARTY_REPOS = {
    "nnunet_coronary": {
        "name": "MM-DHM/nnUNet-Coronary-CTA-Segmentation",
        "url": "https://github.com/MM-DHM/nnUNet-Coronary-CTA-Segmentation.git",
        "purpose": "Coronary CTA segmentation using nnU-Net-style workflow.",
    },
    "nnunet": {
        "name": "MIC-DKFZ/nnUNet",
        "url": "https://github.com/MIC-DKFZ/nnUNet.git",
        "purpose": "General biomedical nnU-Net segmentation framework.",
    },
    "lct_unet": {
        "name": "qianjinmingliang/Coronary-Artery-segmentation-with-LCTUnet",
        "url": "https://github.com/qianjinmingliang/Coronary-Artery-segmentation-with-LCTUnet.git",
        "purpose": "Coronary artery segmentation with local contextual transformer U-Net.",
    },
    "fancnn": {
        "name": "RoelvH97/FanCNN",
        "url": "https://github.com/RoelvH97/FanCNN.git",
        "purpose": "Plaque/lumen mesh segmentation using coronary centerline priors.",
    },
}


def run(cmd, cwd=None):
    print("+", cmd)
    return subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def clone_repo(repo_key, dest_root="/content/third_party"):
    info = THIRD_PARTY_REPOS[repo_key]
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / info["name"].split("/")[-1]

    if dest.exists():
        print(f"Already exists: {dest}")
    else:
        run(f"git clone {info['url']} {dest}")

    return dest


def save_nifti(image, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    print("Saved:", out_path)
    return out_path


def show_mask_overlay(volume, mask, z=None, vmin=-200, vmax=800, alpha=0.35):
    if z is None:
        z = volume.shape[0] // 2

    plt.figure(figsize=(8, 8))
    plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.imshow(mask[z] > 0, alpha=alpha)
    plt.axis("off")
    plt.title(f"Mask overlay z={z}")
    plt.show()


def summarize_mask(mask, spacing):
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    voxels = int(np.sum(mask > 0))
    return {
        "voxels": voxels,
        "voxel_volume_mm3": float(voxel_volume),
        "volume_mm3": float(voxels * voxel_volume),
    }


def make_placeholder_coronary_mask(volume, threshold=250):
    """
    Placeholder only: threshold bright contrast/calcification.

    This is NOT coronary segmentation. It is used only to test downstream
    visualization and volume-measurement plumbing before integrating a real model.
    """
    return volume > threshold
