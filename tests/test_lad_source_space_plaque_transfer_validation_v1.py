import numpy as np
import pandas as pd

from openplaque.lad_source_space_plaque_transfer_validation_v1 import (
    BASELINE, SHELLS, NOMINAL_SHELL, RCA_REFERENCE_ARC,
    MIN_MAJORITY_AUC, MIN_STRICT_AUC, MIN_POS_NEG_RATIO,
    MIN_CROSS_SHELL_SPEARMAN,
    _auc, _arc, _join_frozen_and_extension, _fit_rca_models,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert SHELLS == (0.75, 1.0, 1.25, 1.5)
    assert NOMINAL_SHELL == 1.0
    assert RCA_REFERENCE_ARC == (20.0, 50.0)
    assert MIN_MAJORITY_AUC == 0.80
    assert MIN_STRICT_AUC == 0.90
    assert MIN_POS_NEG_RATIO == 2.00
    assert MIN_CROSS_SHELL_SPEARMAN == 0.80
    assert synthetic_self_test()["ok"] is True


def test_auc_and_join():
    assert _auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    frozen = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    ext = np.array([[-2., 0., 0.], [-1., 0., 0.], [0., 0., 0.]])
    combined, elen, gap = _join_frozen_and_extension(frozen, ext)
    assert np.isclose(elen, 2.0)
    assert gap < 1e-9
    assert np.isclose(_arc(combined)[-1], 4.0)


def test_rca_model_fit_shapes():
    rows = []
    for shell in SHELLS:
        for arc in np.arange(0.0, 52.5, 0.5):
            r = 1.5 + 0.05*np.sin(arc/5)
            hu = 600 + 20*np.cos(arc/7)
            sv = (2*np.pi*r*shell + np.pi*shell*shell)*0.5
            vals = [0.10*sv, 0.25*sv, 0.40*sv, 0.02*sv]
            rows.append({
                "arc_mm": arc,
                "shell_thickness_mm": shell,
                "station_qc_pass": True,
                "lumen_radius_median_mm": r,
                "center_hu": hu,
                "shell_volume_mm3": sv,
                "low_attenuation_mm3": vals[0],
                "noncalcified_mm3": vals[1],
                "mixed_intermediate_mm3": vals[2],
                "calcified_mm3": vals[3],
            })
    models = _fit_rca_models(pd.DataFrame(rows))
    assert len(models) == len(SHELLS) * 4
    assert set(models.component) == {
        "low_attenuation_mm3", "noncalcified_mm3",
        "mixed_intermediate_mm3", "calcified_mm3"
    }
