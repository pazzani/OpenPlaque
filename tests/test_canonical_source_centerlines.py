from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openplaque.canonical_source_centerlines import _candidate_from_summary, _truncate_at_arc


def test_truncate_inserts_exact_cutoff():
    df = pd.DataFrame({
        'arc_mm': [0.0, 5.0, 10.0, 15.0],
        'z': [0.0, 1.0, 2.0, 3.0],
        'y': [0.0, 0.0, 0.0, 0.0],
        'x': [0.0, 0.0, 0.0, 0.0],
        'lps_x_mm': [0.0, 5.0, 10.0, 15.0],
        'lps_y_mm': [0.0, 0.0, 0.0, 0.0],
        'lps_z_mm': [0.0, 0.0, 0.0, 0.0],
    })
    out = _truncate_at_arc(df, 12.2)
    assert np.isclose(out['arc_mm'].iloc[-1], 12.2)
    assert np.isclose(out['lps_x_mm'].iloc[-1], 12.2)


def test_candidate_selection_rejects_stale_long_lad():
    rows = [
        dict(label='LAD', coord_variant='recomputed_from_zyx', length_mm=117.0,
             source_hu_gt200_fraction=1.0, source_hu_median=600.0,
             current_median_distance_mm=58.0, legacy_median_distance_mm=58.0,
             combined_reconciliation_score=.1, path='stale.csv'),
        dict(label='LAD', coord_variant='recomputed_from_zyx', length_mm=57.2,
             source_hu_gt200_fraction=1.0, source_hu_median=580.0,
             current_median_distance_mm=.16, legacy_median_distance_mm=.16,
             combined_reconciliation_score=.98, path='good.csv'),
    ]
    chosen = _candidate_from_summary(pd.DataFrame(rows), 'LAD', 56.5, 58.5)
    assert chosen['path'] == 'good.csv'
