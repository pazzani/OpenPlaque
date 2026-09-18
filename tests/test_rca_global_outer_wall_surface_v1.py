import numpy as np

from openplaque.rca_global_outer_wall_surface_v1 import (
    BASELINE,
    NOMINAL_VARIANT,
    SURFACE_VARIANTS,
    REFERENCE_ARC,
    HOLDOUT_ARC,
    WALL_MIN_MM,
    WALL_MAX_MM,
    _auc,
    _solve_surface,
    _specificity_pass,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert NOMINAL_VARIANT == "nominal"
    assert set(SURFACE_VARIANTS) == {"flexible", "nominal", "smooth"}
    assert REFERENCE_ARC == (20.0, 50.0)
    assert HOLDOUT_ARC == (0.0, 20.0)
    out = synthetic_self_test()
    assert out["ok"] is True
    assert WALL_MIN_MM <= out["surface_min_mm"] <= out["surface_max_mm"] <= WALL_MAX_MM


def test_global_surface_interpolates_missing_rays():
    anchor = np.full((30, 12), np.nan)
    weight = np.zeros((30, 12))
    anchor[:, 0] = 0.7
    anchor[:, 4] = 1.0
    anchor[:, 8] = 0.8
    weight[:, [0, 4, 8]] = 3.0
    surf, prior, err = _solve_surface(anchor, weight, 0.8, 0.35)
    assert surf.shape == anchor.shape
    assert np.isfinite(surf).all()
    assert 0.7 <= prior <= 1.0
    assert np.nanmedian(err) < 0.25
    assert np.nanmax(np.abs(np.diff(surf, axis=1))) < 0.5


def test_more_regularization_is_not_rougher():
    rng = np.random.default_rng(3)
    anchor = np.full((30, 12), np.nan)
    weight = np.zeros((30, 12))
    for s in range(30):
        for a in (0, 3, 6, 9):
            anchor[s, a] = 0.8 + 0.15*np.sin(s/5 + a) + 0.04*rng.normal()
            weight[s, a] = 3.0
    flex, _, _ = _solve_surface(anchor, weight, 0.4, 0.15)
    smooth, _, _ = _solve_surface(anchor, weight, 1.6, 0.70)
    rough_flex = np.mean(np.abs(np.diff(flex, axis=0))) + np.mean(np.abs(flex - np.roll(flex, 1, axis=1)))
    rough_smooth = np.mean(np.abs(np.diff(smooth, axis=0))) + np.mean(np.abs(smooth - np.roll(smooth, 1, axis=1)))
    assert rough_smooth <= rough_flex + 1e-9


def test_auc_and_specificity_gate():
    assert _auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    good = {
        "usable": True,
        "majority_vs_vote_free_auc": 0.82,
        "strict_5of5_bins": 3,
        "strict5_vs_vote_free_auc": 0.94,
        "positive_negative_median_ratio": 2.3,
    }
    assert _specificity_pass(good, 0.85)
    bad = dict(good)
    bad["strict5_vs_vote_free_auc"] = 0.85
    assert not _specificity_pass(bad, 0.85)
