import numpy as np
import pandas as pd

from openplaque.rca_expert_outer_wall_validation_v1 import (
    BASELINE,
    STATUS_COMPLETE,
    _component_volumes,
    _dice,
    _fixed_outer_mask,
    _integration_weights,
    _jaccard,
    _mask_outer_radii,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_COMPLETE == "RCA_EXPERT_OUTER_WALL_VALIDATION_COMPLETE"
    assert synthetic_self_test()["ok"] is True


def test_integration_weights():
    a = np.array([0., 1., 2., 3.])
    w = _integration_weights(a)
    assert np.allclose(w, [0.5, 1.0, 1.0, 0.5])
    assert np.isclose(w.sum(), 3.0)


def test_fixed_outer_mask_and_radii():
    shape = (67, 67)
    pixel = 0.15
    lumen = np.full(72, 1.2)
    m = _fixed_outer_mask(lumen, pixel, shape, 1.0)
    r = _mask_outer_radii(m, pixel)
    assert np.isfinite(r).all()
    assert 2.0 < np.median(r) < 2.35


def test_overlap_metrics():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[5:15, 5:15] = True
    b[6:16, 5:15] = True
    assert 0 < _dice(a, b) < 1
    assert 0 < _jaccard(a, b) < 1
    assert _dice(a, a) == 1.0
    assert _jaccard(a, a) == 1.0


def test_component_volumes():
    hu = np.array([
        [-100., 0., 50.],
        [150., 400., 50.],
        [0., 150., 400.],
    ])
    wall = np.ones_like(hu, bool)
    v = _component_volumes(hu, wall, pixel_mm=1.0, ds_mm=1.0)
    assert np.isclose(v["fatlike_excluded_mm3"], 1.0)
    assert np.isclose(v["low_attenuation_mm3"], 2.0)
    assert np.isclose(v["noncalcified_mm3"], 2.0)
    assert np.isclose(v["mixed_intermediate_mm3"], 2.0)
    assert np.isclose(v["calcified_mm3"], 2.0)
    assert np.isclose(v["wall_volume_mm3"], 9.0)
