import numpy as np
import pandas as pd
from openplaque.cv_tuning import mask_metrics, score_metrics, aggregate_cv_results, select_best_parameters


def test_mask_metrics_perfect():
    a = np.zeros((5, 5, 5), dtype=np.uint8)
    a[1:3, 1:3, 1:3] = 2
    m = mask_metrics(a, a)
    assert m['dice'] == 1.0
    assert m['iou'] == 1.0
    assert score_metrics(m) == 1.0


def test_aggregate_and_select():
    rows = []
    for fold in [0, 1]:
        for case_id in ['a', 'b']:
            rows.append({
                'fold': fold, 'case_id': case_id, 'candidate_id': 0,
                'min_component_voxels': 1, 'lumen_distance_voxels': 0,
                'high_hu_threshold': None, 'low_hu_threshold': None,
                'erode_core': False, 'erosion_iterations': 1,
                'dice': 0.8, 'iou': 0.7, 'precision': 0.9, 'recall': 0.75,
                'abs_tpv_error_fraction': 0.1, 'tpv_error_mm3': 1.0,
                'score': 0.8,
            })
            rows.append({
                'fold': fold, 'case_id': case_id, 'candidate_id': 1,
                'min_component_voxels': 10, 'lumen_distance_voxels': 1,
                'high_hu_threshold': None, 'low_hu_threshold': None,
                'erode_core': False, 'erosion_iterations': 1,
                'dice': 0.9, 'iou': 0.8, 'precision': 0.9, 'recall': 0.9,
                'abs_tpv_error_fraction': 0.05, 'tpv_error_mm3': 0.5,
                'score': 0.9,
            })
    df = pd.DataFrame(rows)
    summary = aggregate_cv_results(df)
    assert summary.iloc[0]['min_component_voxels'] == 10
    params = select_best_parameters(df)
    assert params['min_component_voxels'] == 10
    assert params['trim_lumen_adjacent'] is True
