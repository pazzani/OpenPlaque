from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from openplaque import longitudinal_plaque_pcat_fusion as _base


def profile_vote_array_on_registration_axis(
    vote: np.ndarray,
    registration_model: dict,
    vessel: str,
    spacing_zyx=(1.0, 1.0, 1.0),
) -> pd.DataFrame:
    """Build the native plaque-confidence profile on the validated registration axis.

    The confidence-atlas v1 profile picked the largest physical array dimension. For
    24x512x512 curved stacks that makes axes 1 and 2 a tie, and numpy.argmax chooses
    axis 1. That is valid for the RCA in this study, but the validated LAD
    registration identifies axis 2 as the longitudinal direction. This helper uses
    the validated registration axis directly.
    """
    vote = np.asarray(vote)
    if vote.ndim != 3:
        raise RuntimeError(f'{vessel}: plaque vote array must be 3-D, got shape {vote.shape}.')
    if not np.isfinite(vote).all():
        raise RuntimeError(f'{vessel}: plaque vote array contains non-finite values.')
    if float(vote.min()) < 0 or float(vote.max()) > 5:
        raise RuntimeError(f'{vessel}: plaque vote array must contain fold votes in [0,5].')

    lf = registration_model.get('longitudinal_fit', {})
    if 'long_axis' not in lf:
        raise RuntimeError(f'{vessel}: registration model has no longitudinal_fit.long_axis.')
    axis = int(lf['long_axis'])
    if axis not in (0, 1, 2):
        raise RuntimeError(f'{vessel}: invalid registration long_axis={axis}.')

    expected = lf.get('oriented_shape')
    if expected is not None and tuple(int(x) for x in expected) != tuple(vote.shape):
        raise RuntimeError(
            f'{vessel}: plaque vote shape {tuple(vote.shape)} does not match '
            f'registration oriented_shape {tuple(expected)}.'
        )

    dg = registration_model.get('dicom_geometry', {})
    geometry_shape = (
        int(dg.get('instances', vote.shape[0])),
        int(dg.get('rows', vote.shape[1])),
        int(dg.get('cols', vote.shape[2])),
    )
    if geometry_shape != tuple(vote.shape):
        raise RuntimeError(
            f'{vessel}: plaque vote shape {tuple(vote.shape)} does not match '
            f'curved-series DICOM geometry {geometry_shape}.'
        )

    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    if spacing_zyx.shape != (3,) or not np.isfinite(spacing_zyx).all() or np.any(spacing_zyx <= 0):
        raise RuntimeError(f'{vessel}: invalid spacing_zyx={spacing_zyx}.')

    moved = np.moveaxis(vote, axis, 0)
    rows = []
    for i, sl in enumerate(moved):
        active = sl > 0
        rows.append({
            'vessel': vessel,
            'native_axis': axis,
            'slice_index': int(i),
            'native_longitudinal_mm': float(i * spacing_zyx[axis]),
            'vote_sum': int(sl.sum()),
            'voxels_vote_ge3': int((sl >= 3).sum()),
            'voxels_vote_ge4': int((sl >= 4).sum()),
            'voxels_vote_5': int((sl == 5).sum()),
            'mean_vote_among_any_plaque': float(sl[active].mean()) if active.any() else np.nan,
        })
    return pd.DataFrame(rows)


def profile_vote_map_on_registration_axis(vote_path: Path, registration_model: dict, vessel: str) -> pd.DataFrame:
    vote_path = Path(vote_path)
    if not vote_path.exists():
        raise FileNotFoundError(f'{vessel}: missing plaque vote map {vote_path}')
    img = sitk.ReadImage(str(vote_path))
    vote = sitk.GetArrayFromImage(img)
    spacing_zyx = np.asarray(img.GetSpacing(), dtype=float)[::-1]
    return profile_vote_array_on_registration_axis(vote, registration_model, vessel, spacing_zyx)


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'Longitudinal_Plaque_PCAT_Fusion_v1')
    out.mkdir(parents=True, exist_ok=True)

    atlas_root = _base._find_dir_recursive(drive_root, 'Plaque_Ensemble_Confidence_Atlas_v1')
    reg_root = _base._find_dir_recursive(drive_root, 'Curved_Plaque_to_Source_Registration_Canonical_v1')
    derived_root = out / 'axis_reconciled_native_profiles'
    derived_root.mkdir(parents=True, exist_ok=True)

    corrected_profiles = {}
    axis_notes = {
        'reason': (
            'Plaque_Ensemble_Confidence_Atlas_v1 selected its native longitudinal axis by largest physical '
            'array dimension. For 24x512x512 curved stacks, axes 1 and 2 tie, so the atlas defaulted to axis 1. '
            'This fusion instead rebuilds each profile from the saved 0-5 vote map using the validated '
            'canonical registration long_axis for that vessel.'
        ),
        'vessels': {},
    }

    original_require_file = _base._require_file
    for vessel in ('RCA', 'LAD'):
        model = _base._load_registration(reg_root, vessel)
        vote_path = original_require_file(atlas_root, f'{vessel}_plaque_vote_0to5.nii.gz')
        profile = profile_vote_map_on_registration_axis(vote_path, model, vessel)
        profile_path = derived_root / f'{vessel}_native_longitudinal_profile_registration_axis.csv'
        profile.to_csv(profile_path, index=False)
        corrected_profiles[f'{vessel}_native_longitudinal_profile.csv'] = profile_path
        axis_notes['vessels'][vessel] = {
            'registration_long_axis': int(model['longitudinal_fit']['long_axis']),
            'vote_map': str(vote_path),
            'derived_profile': str(profile_path),
            'rows': int(len(profile)),
            'shape_check': 'passed',
        }

    (out / 'axis_reconciliation.json').write_text(json.dumps(axis_notes, indent=2))

    def corrected_require_file(root: Path, name: str) -> Path:
        root = Path(root)
        if name in corrected_profiles:
            try:
                same_root = root.resolve() == atlas_root.resolve()
            except Exception:
                same_root = str(root) == str(atlas_root)
            if same_root:
                return corrected_profiles[name]
        return original_require_file(root, name)

    try:
        _base._require_file = corrected_require_file
        result = _base.run(drive_root=str(drive_root), output_root=str(out))
    finally:
        _base._require_file = original_require_file

    result['axis_reconciliation'] = axis_notes
    return result
