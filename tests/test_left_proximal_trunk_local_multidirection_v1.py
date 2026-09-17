import pandas as pd
import numpy as np
from openplaque import left_proximal_trunk_local_multidirection_v1 as e


def test_constants_and_self_test():
    r=e.synthetic_local_multidirection_self_test()
    assert r['ok']
    assert e.BASELINE=='0593b453959f5a353d644267fbeef24b514ef4d7'
    assert e.ALGORITHM=='left-proximal-trunk-local-multidirection-v1.0'
    assert e.LOCAL_MAX_MM==10.0
    assert e.TOP_INITIAL_DIRS==5
    assert len(e.SEED_OFFSETS_MM)==4


def test_truncate_requires_three_consecutive_failures():
    d=pd.DataFrame({'arc_mm':np.arange(10)*0.4,
                    'plane_pass':[1,1,1,0,1,1,1,0,0,0]})
    r=e._truncate(d)
    assert r['first_sustained_failure_index']==7
    assert r['accepted_last_index']==6
    assert abs(r['accepted_arc_mm']-2.4)<1e-9


def test_local_discovery_does_not_require_aorta_monotonic_gate():
    # The prospective change is explicit in the experiment design: local discovery has no aortic-distance gate.
    assert not hasattr(e, 'AORTA_MONOTONIC_TOLERANCE_MM')
    assert e.MIN_ACCEPTED_EXTENSION_MM==5.0
    assert e.MIN_PLANE_PASS_FRACTION==0.80
