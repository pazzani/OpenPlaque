import numpy as np
import pandas as pd

from openplaque.longitudinal_plaque_pcat_fusion import (
    aggregate_source_bins,
    fuse_rca_pcat,
    map_native_profile_to_source,
)


def _profile(axis=1):
    return pd.DataFrame({
        'native_axis': [axis, axis, axis, axis],
        'slice_index': [10, 11, 12, 13],
        'vote_sum': [0, 5, 9, 0],
        'voxels_vote_ge3': [0, 1, 2, 0],
        'voxels_vote_ge4': [0, 0, 1, 0],
        'voxels_vote_5': [0, 0, 0, 0],
        'mean_vote_among_any_plaque': [np.nan, 1.25, 1.8, np.nan],
    })


def _model(flip=False):
    return {
        'longitudinal_fit': {
            'long_axis': 1,
            'start_px': 10,
            'n_pixels': 4,
            'long_flip': flip,
            'score': 0.8,
            'gradient_corr': 0.7,
        }
    }


def test_forward_longitudinal_mapping():
    mapped = map_native_profile_to_source(_profile(), _model(False), 6.0)
    assert np.allclose(mapped['source_arc_mm'], [0.0, 2.0, 4.0, 6.0])


def test_flipped_longitudinal_mapping():
    mapped = map_native_profile_to_source(_profile(), _model(True), 6.0)
    assert np.allclose(mapped['source_arc_mm'], [0.0, 2.0, 4.0, 6.0])
    assert mapped['slice_index'].tolist() == [13, 12, 11, 10]


def test_axis_mismatch_is_rejected():
    try:
        map_native_profile_to_source(_profile(axis=2), _model(False), 6.0)
    except RuntimeError as e:
        assert 'does not match' in str(e)
    else:
        raise AssertionError('Expected native-axis mismatch to be rejected')


def test_source_bin_aggregation_preserves_support_counts():
    mapped = map_native_profile_to_source(_profile(), _model(False), 6.0)
    bins = aggregate_source_bins(mapped, 6.0)
    assert bins['mapped_native_vote_sum'].sum() == 14
    assert bins['mapped_native_voxels_vote_ge3'].sum() == 3
    assert bins['mapped_native_voxels_vote_ge4'].sum() == 1


def test_rca_pcat_fusion_requires_one_mm_bins():
    plaque = pd.DataFrame({
        'source_bin': [10, 11],
        'arc_start_mm': [10.0, 11.0],
        'arc_end_mm': [11.0, 12.0],
        'native_slices_mapped': [1, 1],
        'mapped_native_vote_sum': [2, 3],
        'mapped_native_voxels_vote_ge3': [1, 0],
        'mapped_native_voxels_vote_ge4': [0, 0],
        'mapped_native_voxels_vote_5': [0, 0],
        'max_native_mean_vote': [2.0, 1.0],
        'mean_native_mean_vote': [2.0, 1.0],
        'any_plaque_signal': [True, True],
        'majority_3plus_signal': [True, False],
        'high_4plus_signal': [False, False],
        'strict_5of5_signal': [False, False],
        'source_arc_sample_min_mm': [10.1, 11.1],
        'source_arc_sample_max_mm': [10.9, 11.9],
    })
    pcat = pd.DataFrame({
        'wall_margin_mm': [0.75, 0.75],
        'arc_start_mm': [10.0, 11.0],
        'arc_end_mm': [11.0, 12.0],
        'fat_voxels': [100, 120],
        'mean_hu': [-90.0, -95.0],
    })
    fused = fuse_rca_pcat(plaque, pcat)
    assert fused['pcat_mean_hu'].tolist() == [-90.0, -95.0]
    assert fused['majority_3plus_signal'].tolist() == [True, False]
