import numpy as np
import pandas as pd

from openplaque.lcx_curved_template_reacquisition import (
    _template_fingerprint, _match_score, _decision, synthetic_lcx_template_self_test
)

def test_template_axis_and_plaque_profile():
    vol = np.zeros((24, 40, 90), dtype=np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[:, 18:22, 10:80] = 1
    mask[:, 16:24, 45:52] = 2
    vol[:] = 100
    vol[mask > 0] = 520
    vol[mask == 2] = 700
    fp = _template_fingerprint(vol, mask)
    assert fp["long_axis"] == 2
    assert fp["plaque"].max() > 0
    assert np.nanmedian(fp["hu_median"]) > 400

def test_match_score_prefers_similar_profile():
    t = {"hu_median": np.linspace(350, 700, 80), "plaque_fraction": np.r_[np.zeros(30), np.ones(10), np.zeros(40)]}
    good = {"hu": np.linspace(355, 695, 100), "local_max_hu": np.r_[np.ones(35)*300, np.ones(15)*700, np.ones(50)*300],
            "robust_fraction": .9, "median_hu": 525.0, "length_mm": 16.0}
    bad = {"hu": np.random.default_rng(4).normal(400, 180, 100), "local_max_hu": np.ones(100)*280,
           "robust_fraction": .5, "median_hu": 400.0, "length_mm": 16.0}
    assert _match_score(t, good)["score"] > _match_score(t, bad)["score"]

def test_decision_and_self_test():
    ranking = pd.DataFrame([{"score": .50, "robust_fraction": .90, "median_hu": 500.0}])
    status = _decision({"score": .60, "intensity_corr": .50}, [{"score": .20}, {"score": .15}], ranking)
    assert status == "LCX_CURVED_TEMPLATE_MATCHES_SOURCE_PATH_REQUIRES_ANATOMICAL_QC"
    assert synthetic_lcx_template_self_test()["passed"]
