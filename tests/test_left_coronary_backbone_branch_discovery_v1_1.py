import json

import numpy as np

from openplaque.left_coronary_backbone_branch_discovery_v1_1 import (
    _source_fixed,
    synthetic_local_vesselness_self_test,
)


def test_synthetic_local_vesselness_self_test():
    out = synthetic_local_vesselness_self_test()
    assert out["ok"] is True
    assert out["max_vesselness"] >= 0.0


def test_source_fixed_ignores_incompatible_cached_vesselness(tmp_path):
    src = np.zeros((4, 5, 6), dtype=np.int16)
    np.save(tmp_path / "series7_int16.npy", src)
    # Regression fixture for the exact bug: cached vesselness is a different cropped shape.
    np.save(tmp_path / "vesselness.npy", np.zeros((2, 2, 2), dtype=np.float32))
    meta = {
        "spacing_zyx": [1.0, 1.0, 1.0],
        "positions_lps_mm": [[0.0, 0.0, 0.0]],
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    (tmp_path / "series7_int16.json").write_text(json.dumps(meta), encoding="utf-8")
    ref, loaded, lazy = _source_fixed(tmp_path)
    assert loaded.shape == src.shape
    assert tuple(ref.GetSize()) == (6, 5, 4)
    assert lazy.field is None
