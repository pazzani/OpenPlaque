import numpy as np
import pytest

from openplaque.pav import (
    compute_pav,
    estimate_outer_wall_candidate,
    estimate_pav_from_labels,
    voxel_volume_mm3,
)


def test_voxel_volume_mm3():
    assert voxel_volume_mm3((0.5, 0.5, 1.0)) == pytest.approx(0.25)


def test_compute_pav_known_fraction():
    plaque = np.zeros((2, 2, 2), dtype=bool)
    outer = np.ones((2, 2, 2), dtype=bool)
    plaque.ravel()[:2] = True

    plaque_vol, outer_vol, pav, plaque_voxels, outer_voxels = compute_pav(
        plaque, outer, spacing=(1.0, 1.0, 1.0)
    )

    assert plaque_voxels == 2
    assert outer_voxels == 8
    assert plaque_vol == pytest.approx(2.0)
    assert outer_vol == pytest.approx(8.0)
    assert pav == pytest.approx(25.0)


def test_compute_pav_rejects_plaque_outside_outer_wall():
    plaque = np.zeros((3, 3, 3), dtype=bool)
    outer = np.zeros_like(plaque)
    plaque[1, 1, 1] = True

    with pytest.raises(ValueError):
        compute_pav(plaque, outer, spacing=(1.0, 1.0, 1.0))


def test_slice_local_outer_wall_preserves_seed_and_rejects_fat():
    volume = np.full((7, 7, 7), -100.0, dtype=float)
    mask = np.zeros((7, 7, 7), dtype=np.uint8)

    mask[3, 3, 3] = 1
    mask[3, 3, 4] = 2
    volume[3, 2:5, 2:6] = 50.0

    outer = estimate_outer_wall_candidate(
        volume,
        mask,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.5,
        fat_threshold_hu=-30.0,
        closing_iterations=0,
        fill_holes=False,
        growth_mode="slice_local",
    )

    seed = (mask == 1) | (mask == 2)
    assert np.all(outer[seed])
    assert outer[3, 2, 3]
    assert not outer[0, 0, 0]


def test_reference_mask_stops_proxy_only_outer_wall_halo():
    volume = np.full((5, 9, 9), 50.0, dtype=float)
    reference = np.zeros((5, 9, 9), dtype=np.uint8)
    reference[2, 2, 2] = 1

    augmented = reference.copy()
    augmented[2, 6, 6] = 2

    outer = estimate_outer_wall_candidate(
        volume,
        augmented,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.5,
        closing_iterations=0,
        fill_holes=False,
        reference_mask=reference,
        growth_mode="slice_local",
    )

    assert outer[2, 6, 6]
    assert not outer[2, 6, 5]
    assert outer[2, 2, 3]


def test_slice_local_mode_does_not_grow_into_neighbor_slice():
    volume = np.full((5, 5, 5), 50.0, dtype=float)
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1

    local = estimate_outer_wall_candidate(
        volume,
        mask,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.1,
        closing_iterations=0,
        fill_holes=False,
        growth_mode="slice_local",
    )

    volume_mode = estimate_outer_wall_candidate(
        volume,
        mask,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.1,
        closing_iterations=0,
        fill_holes=False,
        growth_mode="volume",
    )

    assert local[2, 2, 3]
    assert not local[1, 2, 2]
    assert volume_mode[1, 2, 2]


def test_slice_local_mode_requires_local_reference_anchor_for_growth():
    volume = np.full((5, 7, 7), 50.0, dtype=float)
    reference = np.zeros((5, 7, 7), dtype=np.uint8)
    reference[2, 3, 3] = 1

    augmented = reference.copy()
    augmented[1, 3, 3] = 2

    outer = estimate_outer_wall_candidate(
        volume,
        augmented,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.5,
        closing_iterations=0,
        fill_holes=False,
        reference_mask=reference,
        growth_mode="slice_local",
    )

    assert outer[1, 3, 3]
    assert not outer[1, 3, 4]
    assert outer[2, 3, 4]


def test_estimate_pav_from_labels_zero_plaque():
    volume = np.zeros((5, 5, 5), dtype=float)
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1

    result = estimate_pav_from_labels(
        volume,
        mask,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.0,
        closing_iterations=0,
        fill_holes=False,
        growth_mode="slice_local",
    )

    assert result.plaque_voxels == 0
    assert result.outer_vessel_voxels > 0
    assert result.pav_percent == pytest.approx(0.0)
