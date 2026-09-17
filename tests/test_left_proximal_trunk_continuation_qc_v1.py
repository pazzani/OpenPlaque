import pandas as pd
from openplaque.left_proximal_trunk_continuation_qc_v1 import _truncate_by_sustained_failure, synthetic_truncation_self_test

def test_synthetic_truncation_self_test():
    r=synthetic_truncation_self_test()
    assert r['ok'] is True
    assert r['first_sustained_failure_index']==7
    assert r['accepted_last_index']==6

def test_isolated_failure_does_not_truncate():
    d=pd.DataFrame({'arc_mm':[0,.4,.8,1.2,1.6,2.0,2.4],'plane_pass':[1,1,0,1,1,1,1]})
    r=_truncate_by_sustained_failure(d,3)
    assert r['first_sustained_failure_index'] is None
    assert r['accepted_last_index']==6
