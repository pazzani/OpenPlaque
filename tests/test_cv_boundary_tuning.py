import numpy as np
import pandas as pd
from openplaque.cv_boundary_tuning import mask_metrics, score_metrics, aggregate_by_params, select_best_params, run_5fold_parameter_cv


def test_mask_metrics_perfect():
    a = np.zeros((5,5,5), dtype=np.uint8)
    a[1:3,1:3,1:3] = 2
    m = mask_metrics(a, a)
    assert m['dice'] == 1.0
    assert m['iou'] == 1.0
    assert score_metrics(m) == 1.0


def test_cv_selection():
    rows=[]
    for case in ['a','b','c','d','e']:
        for cid, minc, score in [(0,0,0.7),(1,10,0.9)]:
            rows.append({
                'case_id':case,'candidate_id':cid,'min_component_voxels':minc,'lumen_distance_voxels':1,
                'high_hu_threshold':None,'low_hu_threshold':None,'erode_core':False,'erosion_iterations':0,
                'score':score,'dice':score,'iou':score,'precision':score,'recall':score,
                'abs_tpv_error_fraction':1-score,'tpv_error_mm3':0,
            })
    df=pd.DataFrame(rows)
    params=select_best_params(df)
    assert params['min_component_voxels']==10
    assignments=pd.DataFrame({'case_id':['a','b','c','d','e'],'fold':[0,1,2,3,4]})
    held, selected, final = run_5fold_parameter_cv(df, assignments)
    assert len(held)==5
    assert final['min_component_voxels']==10
