import numpy as np

from openplaque.plaque_volume import (
    estimate_best_estimate_plaque_volume,
    segmentation_mask_with_plaque_proxy,
)


def test_best_estimate_matches_expected_core_and_context_counts():
    volume = np.full((5, 5, 5), -200.0, dtype=float)
    mask = np.zeros((5, 5, 5), dtype=np.uint8)

    # One vessel voxel and one strict plaque-core voxel.
    mask[2, 2, 2] = 1
    mask[2, 2, 3] = 2
    volume[2, 2, 3] = 500.0  # dense calcium in strict core

    # Candidate background voxels one dilation step from vessel/plaque union.
    volume[2, 1, 2] = 0.0    # low attenuation
    volume[2, 3, 2] = 80.0   # fibrofatty
    volume[1, 2, 2] = 200.0  # fibrous
    volume[3, 2, 2] = 500.0  # not counted: calcium outside strict core

    r = estimate_best_estimate_plaque_volume(
        'TEST', volume, mask, spacing_xyz_mm=(0.5, 0.5, 1.0)
    )

    assert r.strict_nnunet_core_voxels == 1
    assert r.dense_calcium_voxels == 1
    assert r.low_attenuation_voxels == 1
    assert r.fibrofatty_voxels == 1
    assert r.fibrous_voxels == 1
    assert r.context_selected_candidate_voxels == 3
    assert r.total_plaque_volume_proxy_voxels == 4

    # voxel volume = 0.25 mm^3
    assert r.strict_nnunet_core_volume_mm3 == 0.25
    assert r.total_plaque_volume_proxy_mm3 == 1.0


def test_core_below_minus_30_is_in_strict_core_but_not_best_estimate_proxy():
    volume = np.full((3, 3, 3), -200.0, dtype=float)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 2
    volume[1, 1, 1] = -100.0

    r = estimate_best_estimate_plaque_volume('TEST', volume, mask, (1, 1, 1))
    assert r.strict_nnunet_core_voxels == 1
    assert r.total_plaque_volume_proxy_voxels == 0


def test_vessel_label_is_excluded_from_context_candidates_by_default():
    volume = np.full((3, 3, 3), 80.0, dtype=float)
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1

    r = estimate_best_estimate_plaque_volume('TEST', volume, mask, (1, 1, 1))
    assert not r.best_estimate_mask[1, 1, 1]
    assert r.context_selected_candidate_voxels > 0


def test_proxy_can_be_promoted_into_outer_wall_seed_mask():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    proxy = np.zeros_like(mask, dtype=bool)
    proxy[1, 1, 2] = True

    seeded = segmentation_mask_with_plaque_proxy(mask, proxy)
    assert seeded[1, 1, 1] == 1
    assert seeded[1, 1, 2] == 2
