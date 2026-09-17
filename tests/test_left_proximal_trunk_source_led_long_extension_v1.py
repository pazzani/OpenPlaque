import numpy as np
import pandas as pd
from openplaque import left_proximal_trunk_source_led_long_extension_v1 as e


def test_synthetic_long_extension_self_test():
    r=e.synthetic_long_extension_self_test()
    assert r['ok']
    assert r['search_max_mm']==25.0
    assert r['field_margin_mm']==28.0


def test_aortic_geometry_not_used_in_discovery_constants():
    assert e.SEARCH_MAX_MM == 25.0
    assert e.FIELD_MARGIN_MM > e.SEARCH_MAX_MM
    assert e.MEANINGFUL_AORTA_RETURN_MM == 2.0


def test_truncation_three_consecutive_failures():
    d=pd.DataFrame({'arc_mm':np.arange(10)*.4,'plane_pass':[1,1,1,1,0,1,1,0,0,0]})
    r=e._truncate(d)
    assert r['first_sustained_failure_index']==7
    assert r['accepted_last_index']==6
    assert abs(r['accepted_arc_mm']-2.4)<1e-9
