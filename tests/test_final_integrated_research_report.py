import pandas as pd

from openplaque.final_integrated_research_report import _best_interval, _contiguous_intervals


def test_contiguous_intervals():
    df = pd.DataFrame({
        "arc_start_mm": [0, 1, 2, 3, 4, 5],
        "arc_end_mm": [1, 2, 3, 4, 5, 6],
        "signal": [False, True, True, False, True, False],
    })
    got = _contiguous_intervals(df, "signal")
    assert got == [
        {"arc_start_mm": 1.0, "arc_end_mm": 3.0, "duration_mm": 2.0},
        {"arc_start_mm": 4.0, "arc_end_mm": 5.0, "duration_mm": 1.0},
    ]


def test_best_interval_prefers_stronger_support_then_duration():
    df = pd.DataFrame({
        "confidence_level": ["majority_3plus", "majority_3plus", "high_4plus"],
        "arc_start_mm": [0.0, 11.0, 1.0],
        "arc_end_mm": [5.0, 12.0, 5.0],
        "duration_mm": [5.0, 1.0, 4.0],
        "mapped_native_voxels_vote_ge3": [57.0, 10.0, 46.0],
        "mapped_native_voxels_vote_ge4": [22.0, 9.0, 22.0],
        "mapped_native_voxels_vote_5": [13.0, 0.0, 13.0],
    })
    got = _best_interval(df, "majority_3plus")
    assert got["arc_start_mm"] == 0.0
    assert got["arc_end_mm"] == 5.0
    assert got["mapped_native_voxels_vote_ge3"] == 57.0
