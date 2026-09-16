import numpy as np
from scipy.spatial import cKDTree
from openplaque import left_main_proximal_lad_ostial_bridge as b
from openplaque import left_main_proximal_lad_ostial_bridge_v2 as m


def test_algorithm_and_baseline():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-main-proximal-lad-ostial-bridge-v1.1-calibrated"


def test_endpoint_orientation_chooses_aorta_nearest_end():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])
    tree=cKDTree(np.array([[-.5,0.,0.],[20.,0.,0.]]))
    q,d,f=b._orient_endpoint_nearest_tree(p[::-1],tree)
    assert np.allclose(q[0],[0.,0.,0.])
    assert f == 1
    assert d < 1.0


def test_calibrated_control_self_test():
    r=m.synthetic_calibrated_control_self_test()
    assert r["ok"]
    assert 5.8 <= r["control_start_inside_mm"] <= 6.2


def test_cone_directions_allow_curvature_but_remain_forward():
    t=np.array([1.,0.,0.])
    dirs=b._cone_dirs(t)
    assert len(dirs) > 40
    assert all(np.dot(d,t) >= np.cos(np.deg2rad(60))-1e-8 for d in dirs)
