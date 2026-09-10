import numpy as np
import pytest

from openplaque.centerline import (
    cumulative_length,
    extract_rca_centerline,
    point_at_distance,
)


def _straight_tube(shape=(80, 21, 21), radius=2):
    z, y, x = np.indices(shape)
    return ((y - 10) ** 2 + (x - 10) ** 2 <= radius ** 2) & (z >= 5) & (z <= 70)


def test_cumulative_length_and_interpolation():
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    cum = cumulative_length(pts)
    assert np.allclose(cum, [0.0, 3.0, 7.0])
    assert np.allclose(point_at_distance(pts, cum, 5.0), [3.0, 2.0, 0.0])


def test_straight_tube_centerline_has_expected_length_and_landmarks():
    mask = _straight_tube()
    result = extract_rca_centerline(
        mask,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        ostium_zyx=(5, 10, 10),
        distal_hint_zyx=(70, 10, 10),
    )
    assert result.length_mm == pytest.approx(65.0, abs=2.0)
    assert set(result.landmarks_xyz_mm) == {0.0, 10.0, 50.0}
    assert result.landmarks_xyz_mm[10.0][2] == pytest.approx(15.0, abs=2.0)
    assert result.landmarks_xyz_mm[50.0][2] == pytest.approx(55.0, abs=2.0)


def test_anisotropic_spacing_is_used_for_arc_length():
    mask = _straight_tube()
    result = extract_rca_centerline(
        mask,
        spacing_xyz_mm=(0.5, 0.5, 2.0),
        ostium_zyx=(5, 10, 10),
        distal_hint_zyx=(70, 10, 10),
    )
    assert result.length_mm == pytest.approx(130.0, abs=4.0)
    assert 50.0 in result.landmarks_xyz_mm


def test_branch_uses_distal_hint_to_choose_requested_path():
    mask = _straight_tube(shape=(70, 31, 31), radius=1)
    # Add a side branch from z=35 toward +x.
    mask[35, 10, 10:26] = True
    result = extract_rca_centerline(
        mask,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        ostium_zyx=(5, 10, 10),
        distal_hint_zyx=(65, 10, 10),
    )
    # The chosen endpoint should remain on the main z-directed trunk.
    assert abs(result.endpoint_zyx_voxel[2] - 10) <= 1
    assert result.endpoint_zyx_voxel[0] >= 63


def test_short_centerline_omits_unreachable_50mm_landmark():
    mask = _straight_tube(shape=(35, 21, 21), radius=2)
    result = extract_rca_centerline(
        mask,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        ostium_zyx=(5, 10, 10),
        distal_hint_zyx=(30, 10, 10),
    )
    assert 10.0 in result.landmarks_xyz_mm
    assert 50.0 not in result.landmarks_xyz_mm


def test_rejects_tiny_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    with pytest.raises(ValueError, match="too small"):
        extract_rca_centerline(mask, (1, 1, 1), (5, 5, 5))
