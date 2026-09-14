from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Make the src-layout package importable when pytest is launched directly
# from the repository without installing OpenPlaque as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openplaque.left_coronary_reconciliation import (
    _classification,
    _phys_to_zyx,
    _zyx_to_phys,
)


def _identity_image():
    img = sitk.Image([20, 21, 22], sitk.sitkInt16)
    img.SetSpacing((0.4, 0.5, 0.6))
    img.SetOrigin((10.0, -20.0, 30.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def test_physical_voxel_round_trip():
    img = _identity_image()
    zyx = np.array([[0.0, 0.0, 0.0], [3.5, 4.0, 5.25], [10.0, 11.0, 12.0]])
    xyz = _zyx_to_phys(img, zyx)
    got = _phys_to_zyx(img, xyz)
    assert np.allclose(got, zyx)


def test_classification_accepts_dataframe_without_series_truth_value_error():
    rows = [
        dict(label='RCA', coord_variant='stored_lps', is_source_volume_file=True,
             combined_reconciliation_score=.95, source_lumen_score=.9,
             union_median_distance_mm=.2, intersection_median_distance_mm=.2,
             intersection_inside_fraction=1.0, source_hu_gt200_fraction=.9,
             lumen_like_source_support=True),
        dict(label='LAD', coord_variant='stored_lps', is_source_volume_file=True,
             combined_reconciliation_score=.8, source_lumen_score=.8,
             union_median_distance_mm=1.0, intersection_median_distance_mm=1.2,
             intersection_inside_fraction=.5, source_hu_gt200_fraction=.8,
             lumen_like_source_support=True),
        dict(label='SECONDARY', coord_variant='stored_lps', is_source_volume_file=True,
             combined_reconciliation_score=.7, source_lumen_score=.75,
             union_median_distance_mm=1.2, intersection_median_distance_mm=1.5,
             intersection_inside_fraction=.4, source_hu_gt200_fraction=.7,
             lumen_like_source_support=True),
    ]
    result = _classification(pd.DataFrame(rows))
    assert result['rca_positive_control'] == 'PASS'
    assert result['primary_conclusion'] == 'LEFT_CORONARY_RECONCILED'
