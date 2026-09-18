import numpy as np
import pandas as pd

from openplaque.lcx_om_source_space_composition_pcat_feasibility_v1 import (
    BASELINE,
    STATUS_PASS,
    PRIMARY_WALL_MARGIN_MM,
    _arc_weights,
    _shell_profile_1mm,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS_PASS == "LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_COMPLETE"
    assert PRIMARY_WALL_MARGIN_MM == 0.75
    assert synthetic_self_test()["ok"] is True


def test_arc_weights():
    a = np.array([1.0, 1.5, 2.0, 2.5])
    w = _arc_weights(a)
    assert np.allclose(w, [0.25, 0.5, 0.5, 0.25])
    assert np.isclose(w.sum(), 1.5)


def test_profile_aggregation():
    rows = []
    for vessel in ("C6", "C7"):
        for arc in (1.0, 1.5, 2.0, 2.5):
            rows.append({
                "vessel": vessel,
                "station_index": 0,
                "post_split_arc_mm": arc,
                "integration_ds_mm": 0.5,
                "shell_thickness_mm": 1.0,
                "recomputed_station_qc_pass": True,
                "shell_volume_mm3": 2.0,
                "fatlike_excluded_mm3": 0.2,
                "low_attenuation_mm3": 0.3,
                "noncalcified_mm3": 0.4,
                "mixed_intermediate_mm3": 0.5,
                "calcified_mm3": 0.6,
                "raw_nonfatlike_shell_mm3": 1.8,
            })
    df = pd.DataFrame(rows)
    p = _shell_profile_1mm(df, 1.0)
    c6 = p[p.vessel == "C6"].sort_values("arc_start_mm")
    assert list(c6.arc_start_mm) == [1.0, 2.0]
    assert np.isclose(c6.iloc[0].raw_nonfatlike_shell_mm3, 3.6)
    assert np.isclose(c6.iloc[1].raw_nonfatlike_shell_mm3, 3.6)
