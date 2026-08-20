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


def test_outer_wall_candidate_preserves_seed_and_rejects_fat():
    volume = np.full((7, 7, 7), -100.0, dtype=float)
    mask = np.zeros((7, 7, 7), dtype=np.uint8)

    # Lumen seed plus one plaque voxel.
    mask[3, 3, 3] = 1
    mask[3, 3, 4] = 2

    # Tissue-valued voxels near the artery should be eligible to enter the
    # candidate envelope; surrounding fat should not.
    volume[2:5, 2:5, 2:6] = 50.0

    outer = estimate_outer_wall_candidate(
        volume,
        mask,
        spacing=(1.0, 1.0, 1.0),
        max_wall_thickness_mm=1.5,
        fat_threshold_hu=-30.0,
        closing_iterations=0,
        fill_holes=False,
    )

    seed = (mask == 1) | (mask == 2)
    assert np.all(outer[seed])
    assert outer[3, 2, 3]
    assert not outer[0, 0, 0]


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
    )

    assert result.plaque_voxels == 0
    assert result.outer_vessel_voxels > 0
    assert result.pav_percent == pytest.approx(0.0)
