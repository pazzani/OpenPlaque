import numpy as np

from openplaque.rca_adaptive_outer_wall_plaque_v1 import (
    BASELINE,
    VARIANTS,
    NOMINAL_VARIANT,
    REFERENCE_ARC,
    HOLDOUT_ARC,
    _auc,
    _detect_outer_wall,
    _fill_circular,
    _specificity_pass,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert NOMINAL_VARIANT == "nominal"
    assert set(VARIANTS) == {"strict", "nominal", "liberal"}
    assert REFERENCE_ARC == (20.0, 50.0)
    assert HOLDOUT_ARC == (0.0, 20.0)
    assert synthetic_self_test()["ok"] is True


def test_circular_fill():
    raw = np.array([0.8, np.nan, np.nan, 1.0, np.nan, 0.9, np.nan, np.nan])
    x = _fill_circular(raw)
    assert np.isfinite(x).all()
    assert x.min() > 0.6
    assert x.max() < 1.2


def test_direct_fat_edge_is_detected():
    off = np.arange(0.05, 2.65, 0.1)
    # 72 rays: wall-like signal followed by a sharp drop to fat near 0.85 mm.
    sm = np.zeros((72, len(off)), float)
    for i in range(72):
        sm[i] = np.where(off < 0.85, 120.0, -80.0)
    out = _detect_outer_wall(off, sm, "nominal")
    assert out["direct_fat_fraction"] > 0.95
    assert np.nanmedian(out["smoothed_thickness_mm"]) < 1.05
    assert np.nanmedian(out["smoothed_thickness_mm"]) > 0.65


def test_gradient_fallback_is_detected():
    off = np.arange(0.05, 2.65, 0.1)
    sm = np.zeros((72, len(off)), float)
    for i in range(72):
        sm[i] = np.where(off < 1.05, 260.0, 100.0)
    out = _detect_outer_wall(off, sm, "nominal")
    assert out["detected_fraction"] > 0.95
    assert np.mean(out["edge_mode"] == 2) > 0.90


def test_auc_and_specificity_gate():
    assert _auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    good = {
        "usable": True,
        "majority_vs_vote_free_auc": 0.82,
        "strict_5of5_bins": 3,
        "strict5_vs_vote_free_auc": 0.94,
        "positive_negative_median_ratio": 2.5,
    }
    assert _specificity_pass(good, 0.85)
    bad = dict(good)
    bad["positive_negative_median_ratio"] = 1.9
    assert not _specificity_pass(bad, 0.85)
