import numpy as np
import pytest
from pathlib import Path
from dataclasses import dataclass

from openplaque.boundary_parameter_tuning import (
    mask_metrics,
    score_metrics,
    sample_boundary_params_from_trial,
    BAYESIAN_SEARCH_SPACE,
)


def test_mask_metrics_smoke():
    true = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros_like(true)
    true[1:3, 1:3, 1:3] = 2
    pred[1:3, 1:3, 1:3] = 2
    m = mask_metrics(pred, true, spacing=(1, 1, 1))
    assert m["dice"] == 1.0
    assert score_metrics(m) > 0.99


def test_bayesian_search_space_declared():
    assert "closing_radius_voxels" in BAYESIAN_SEARCH_SPACE
    assert "min_plaque_length_mm" in BAYESIAN_SEARCH_SPACE
    assert "connectivity" in BAYESIAN_SEARCH_SPACE
    assert "adaptive_hu_thresholds" in BAYESIAN_SEARCH_SPACE
