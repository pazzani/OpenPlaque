import numpy as np
import pandas as pd

from openplaque.rca_plaque_pcat_research_lock_v1 import (
    BASELINE,
    STATUS,
    _component_percentages,
    _pcat_summary,
    _shell_summary,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS == "RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED"
    assert synthetic_self_test()["ok"] is True


def test_shell_summary():
    d = pd.DataFrame({
        "shell_thickness_mm": [0.75, 0.75, 1.0, 1.0],
        "excess_low_attenuation_mm3": [1, 2, 2, 3],
        "excess_noncalcified_mm3": [1, 1, 2, 2],
        "excess_mixed_intermediate_mm3": [0, 0, 1, 0],
        "excess_calcified_mm3": [0, 1, 0, 1],
        "excess_total_plaque_proxy_mm3": [2, 4, 5, 6],
    })
    s = _shell_summary(d)
    assert len(s) == 2
    assert np.isclose(s.loc[np.isclose(s.shell_thickness_mm, 1.0), "excess_total_plaque_proxy_mm3"].iloc[0], 11.0)


def test_component_percentages_sum_to_100():
    p = _component_percentages({
        "excess_low_attenuation_mm3": 2.0,
        "excess_noncalcified_mm3": 3.0,
        "excess_mixed_intermediate_mm3": 1.0,
        "excess_calcified_mm3": 4.0,
        "excess_total_plaque_proxy_mm3": 10.0,
    })
    assert abs(sum(p.values()) - 100.0) < 1e-9


def test_pcat_weighted_mean():
    d = pd.DataFrame({
        "arc_start_mm": [10.0, 11.0],
        "arc_end_mm": [11.0, 12.0],
        "fat_voxels": [100, 300],
        "mean_hu": [-80.0, -100.0],
        "wall_margin_mm": [0.75, 0.75],
    })
    s = _pcat_summary(d)
    assert s["available"] is True
    assert np.isclose(s["fat_voxel_weighted_mean_hu"], -95.0)
    assert s["fat_voxels_total"] == 400
