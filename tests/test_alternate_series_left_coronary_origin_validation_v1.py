import numpy as np
import pandas as pd

from openplaque.alternate_series_left_coronary_origin_validation_v1 import (
    _arc,
    _metadata_score,
    _resample_polyline,
    identify_reference_series,
    synthetic_self_test,
)


def test_synthetic_self_test():
    out = synthetic_self_test()
    assert out["ok"] is True
    assert out["resampled_points"] == 5


def test_resample_polyline():
    p = np.array([[0., 0., 0.], [2., 0., 0.]])
    q = _resample_polyline(p, 0.5)
    assert len(q) == 5
    assert np.isclose(_arc(q)[-1], 2.0)


def test_identify_series7_reference_prefers_matching_geometry():
    inv = pd.DataFrame([
        {
            "series_number": 7, "n_files": 524, "rows": 512, "cols": 512,
            "pixel_spacing_x_mm": .3515625, "pixel_spacing_y_mm": .3515625,
            "slice_spacing_mm": .3, "series_uid": "ref", "folder": "a",
        },
        {
            "series_number": 8, "n_files": 500, "rows": 512, "cols": 512,
            "pixel_spacing_x_mm": .4, "pixel_spacing_y_mm": .4,
            "slice_spacing_mm": .5, "series_uid": "other", "folder": "b",
        },
    ])
    meta = {"shape": [524, 512, 512], "spacing_zyx": [.3, .3515625, .3515625]}
    row = identify_reference_series(inv, meta)
    assert row.series_uid == "ref"


def test_metadata_score_penalizes_calcium_score():
    good = pd.Series({
        "series_description": "Coronary CCTA", "protocol_name": "Cardiac",
        "image_type": "ORIGINAL PRIMARY AXIAL", "modality": "CT",
        "rows": 512, "cols": 512, "n_files": 500,
        "pixel_spacing_x_mm": .35, "pixel_spacing_y_mm": .35,
        "slice_spacing_mm": .4,
    })
    bad = good.copy()
    bad["series_description"] = "Calcium Score"
    assert _metadata_score(good) > _metadata_score(bad)
