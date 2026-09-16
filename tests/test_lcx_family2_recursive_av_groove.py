import numpy as np
from openplaque.lcx_family2_recursive_av_groove import synthetic_family2_self_test, _union_find_groups

def test_synthetic_family2_self_test():
    out=synthetic_family2_self_test(); assert out['ok']; assert 19.5 <= out['f2_common_mm'] <= 20.5

def test_grouping():
    pts=np.array([[0,0,0],[0.2,0,0],[3,0,0]],float); g=_union_find_groups(pts,1.0); sizes=sorted(len(x) for x in g); assert sizes==[1,2]
