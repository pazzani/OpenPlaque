import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from openplaque.run_new_data import load_best_boundary_parameters, tpv_summary_rows


def test_load_current_best_parameter_format(tmp_path):
    p = tmp_path / "best_boundary_parameters_bayesian.json"
    p.write_text(json.dumps({
        "final_parameters_selected_on_all_cases": {
            "min_component_voxels": 25,
            "lumen_distance_voxels": 1,
            "high_hu_threshold": None,
            "low_hu_threshold": -100,
            "closing_radius_voxels": 1,
            "fill_holes": True,
            "min_plaque_length_mm": 2.0,
            "connectivity": 26,
            "adaptive_hu_thresholds": False,
            "erode_core": False,
            "erosion_iterations": 1
        },
        "best_mean_score": 0.9
    }))
    params = load_best_boundary_parameters(p)
    assert params["min_component_voxels"] == 25
    assert params["trim_lumen_adjacent"] is True
    assert params["remove_small"] is True
    assert params["fill_holes"] is True


@dataclass
class FakeReport:
    name: str = "LAD"
    tpv_mm3: float = 10.0
    plaque_voxels: int = 10


class FakeRef:
    refined_tpv_mm3 = 8.0
    removed_volume_mm3 = 2.0
    refined_plaque_voxels = 8


def test_tpv_summary_rows():
    rows = tpv_summary_rows([FakeReport()], {"LAD": FakeRef()})
    assert rows[0]["vessel"] == "LAD"
    assert rows[-1]["vessel"] == "TOTAL"
    assert rows[-1]["refined_tpv_mm3"] == 8.0
