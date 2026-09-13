"""Anatomical growth constraints for the experimental PAV prototype."""

import numpy as np
from scipy import ndimage as ndi


def constrained_growth_support(seed, anchor, spacing_zyx, max_distance_mm, tissue_mask):
    """Return allowed non-seed growth near both the PAV seed and artery anchor."""
    seed = np.asarray(seed, dtype=bool)
    anchor = np.asarray(anchor, dtype=bool)
    tissue_mask = np.asarray(tissue_mask, dtype=bool)
    if seed.shape != anchor.shape or seed.shape != tissue_mask.shape:
        raise ValueError("seed, anchor, and tissue_mask must have the same shape")
    ds = ndi.distance_transform_edt(~seed, sampling=spacing_zyx)
    da = ndi.distance_transform_edt(~anchor, sampling=spacing_zyx)
    return (ds <= float(max_distance_mm)) & (da <= float(max_distance_mm)) & tissue_mask


def keep_slice_growth_touching_anchor(candidate, seed, anchor, connectivity=8):
    """Remove newly grown 2-D components that do not touch the anchor in that slice.

    Seed voxels are always preserved. This suppresses detached outer-wall halos in
    nearby tissue while retaining every plaque/vessel seed voxel.
    """
    candidate = np.asarray(candidate, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    anchor = np.asarray(anchor, dtype=bool)
    if candidate.shape != seed.shape or candidate.shape != anchor.shape:
        raise ValueError("candidate, seed, and anchor must have the same shape")

    structure = ndi.generate_binary_structure(2, 1 if connectivity <= 4 else 2)
    out = seed.copy()
    for z in range(candidate.shape[0]):
        labels, n = ndi.label(candidate[z], structure=structure)
        if n == 0:
            continue
        touching = np.unique(labels[anchor[z] & (labels > 0)])
        if touching.size:
            out[z] |= np.isin(labels, touching)
    return out
