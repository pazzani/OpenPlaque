import numpy as np
import pandas as pd

from openplaque.rca_source_space_plaque_excess_specificity_v1 import (
    COMPONENTS,
    NOMINAL_SHELL,
    REFERENCE_ARC,
    HOLDOUT_ARC,
    _auc,
    _huber_fit,
    synthetic_self_test,
)


def test_auc_perfect_ordering():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.2, 0.8, 1.0])
    assert abs(_auc(y, s) - 1.0) < 1e-12


def test_huber_fit_finite():
    x = np.linspace(1.0, 2.0, 40)
    X = np.column_stack([np.ones_like(x), x, x*x, np.zeros_like(x)])
    y = 0.1 + 0.2*x + 0.01*np.sin(np.arange(len(x)))
    y[5] += 2.0
    beta, resid = _huber_fit(X, y)
    assert beta.shape == (4,)
    assert np.isfinite(beta).all()
    assert np.isfinite(resid).all()


def test_prospective_constants():
    assert COMPONENTS == (
        "low_attenuation_mm3",
        "noncalcified_mm3",
        "mixed_intermediate_mm3",
        "calcified_mm3",
    )
    assert NOMINAL_SHELL == 1.0
    assert REFERENCE_ARC == (20.0, 50.0)
    assert HOLDOUT_ARC == (0.0, 20.0)


def test_synthetic_self_test():
    r = synthetic_self_test()
    assert r["ok"] is True
    assert r["model_rows"] == 4
    assert r["excess_sum_mm3"] > 0
