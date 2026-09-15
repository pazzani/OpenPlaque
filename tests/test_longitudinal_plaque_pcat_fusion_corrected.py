import numpy as np

from openplaque.longitudinal_plaque_pcat_fusion_corrected import profile_vote_array_on_registration_axis


def _model(axis):
    return {
        'longitudinal_fit': {
            'long_axis': axis,
            'oriented_shape': [2, 4, 5],
        },
        'dicom_geometry': {
            'instances': 2,
            'rows': 4,
            'cols': 5,
        },
    }


def test_lad_axis2_profile_uses_registration_axis_not_atlas_tie_break():
    vote = np.zeros((2, 4, 5), dtype=np.uint8)
    vote[:, :, 3] = 4
    profile = profile_vote_array_on_registration_axis(vote, _model(2), 'LAD')
    assert profile['native_axis'].unique().tolist() == [2]
    assert len(profile) == 5
    assert int(profile.loc[3, 'voxels_vote_ge4']) == 8
    assert int(profile.loc[1, 'voxels_vote_ge4']) == 0


def test_rca_axis1_profile_remains_axis1():
    vote = np.zeros((2, 4, 5), dtype=np.uint8)
    vote[:, 2, :] = 3
    profile = profile_vote_array_on_registration_axis(vote, _model(1), 'RCA')
    assert profile['native_axis'].unique().tolist() == [1]
    assert len(profile) == 4
    assert int(profile.loc[2, 'voxels_vote_ge3']) == 10


def test_shape_mismatch_is_rejected():
    vote = np.zeros((2, 4, 6), dtype=np.uint8)
    try:
        profile_vote_array_on_registration_axis(vote, _model(2), 'LAD')
    except RuntimeError as exc:
        assert 'shape' in str(exc)
    else:
        raise AssertionError('Expected shape mismatch to be rejected')
