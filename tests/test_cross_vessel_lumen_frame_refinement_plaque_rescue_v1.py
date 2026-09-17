import numpy as np
import pandas as pd

from openplaque.cross_vessel_lumen_frame_refinement_plaque_rescue_v1 import (
    BASELINE, SHELLS, NOMINAL_SHELL, RCA_REFERENCE_ARC,
    CENTER_RING_RADII_MM, CENTER_RING_DIRECTIONS,
    _arc, _auc, _candidate_offsets, _join_frozen_and_extension,
    _passes_specificity, _smooth_tangents, synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert SHELLS == (0.75, 1.0, 1.25, 1.5)
    assert NOMINAL_SHELL == 1.0
    assert RCA_REFERENCE_ARC == (20.0, 50.0)
    out = synthetic_self_test()
    assert out["ok"] is True
    assert out["candidate_centers"] == 1 + len(CENTER_RING_RADII_MM) * CENTER_RING_DIRECTIONS


def test_tangent_smoothing_is_unit_and_oriented():
    p = np.array([
        [0., 0., 0.],
        [1., .1, 0.],
        [2., -.1, 0.],
        [3., .1, 0.],
        [4., 0., 0.],
    ])
    raw, sm = _smooth_tangents(p, half=2)
    assert raw.shape == sm.shape == p.shape
    assert np.allclose(np.linalg.norm(sm, axis=1), 1.0)
    assert np.all(np.sum(raw * sm, axis=1) >= 0)


def test_candidate_offsets_are_bounded():
    offs = _candidate_offsets(np.array([1., 0., 0.]))
    radii = [r for r, _ in offs]
    assert radii[0] == 0.0
    assert max(radii) == max(CENTER_RING_RADII_MM)
    assert len(offs) == 25


def test_auc_and_join():
    assert _auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    frozen = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    ext = np.array([[-2.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
    joined, elen, gap = _join_frozen_and_extension(frozen, ext)
    assert np.isclose(_arc(joined)[-1], 4.0)
    assert np.isclose(elen, 2.0)
    assert gap < 1e-9


def test_specificity_gate():
    good = {
        "usable": True,
        "majority_vs_vote_free_auc": 0.81,
        "strict_5of5_bins": 3,
        "strict5_vs_vote_free_auc": 0.95,
        "positive_negative_median_ratio": 2.5,
    }
    assert _passes_specificity(good, 0.85)
    bad = dict(good)
    bad["majority_vs_vote_free_auc"] = 0.79
    assert not _passes_specificity(bad, 0.85)
