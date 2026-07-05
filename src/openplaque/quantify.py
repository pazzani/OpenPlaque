"""
OpenPlaque plaque quantification helpers.
"""

import numpy as np


def total_plaque_volume(mask, spacing):
    voxel_volume = float(np.prod(spacing))
    plaque_voxels = int(np.sum(mask == 2))
    return {
        "plaque_voxels": plaque_voxels,
        "voxel_volume_mm3": voxel_volume,
        "tpv_mm3": plaque_voxels * voxel_volume,
    }


def label_counts(mask):
    labels, counts = np.unique(mask, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def hu_statistics(volume, mask):
    hu = volume[mask == 2]
    if len(hu) == 0:
        return None

    return {
        "count": int(len(hu)),
        "min": float(hu.min()),
        "max": float(hu.max()),
        "mean": float(hu.mean()),
        "median": float(np.median(hu)),
    }
