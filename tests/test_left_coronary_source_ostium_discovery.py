import numpy as np
from openplaque import left_coronary_source_ostium_discovery as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-source-ostium-discovery-v1.0-lowmem"


def test_synthetic_root_component_self_test():
    r = m.synthetic_root_component_self_test()
    assert r["ok"]
    assert r["components"] >= 2


def test_rca_calibrated_lumen_gate_accepts_coronary_scale():
    rca = {"median_radius_mm": 1.61, "median_center_hu": 560.0}
    x = {"radius_mm":1.75,"centroid_offset_mm":0.20,"circularity":0.72,
         "core_minus_ring_hu":180.0,"center_hu":570.0}
    score, passed = m._score_plane(x, rca)
    assert passed
    assert score > 0.70


def test_rca_calibrated_lumen_gate_rejects_aortic_scale():
    rca = {"median_radius_mm": 1.61, "median_center_hu": 560.0}
    x = {"radius_mm":4.50,"centroid_offset_mm":0.05,"circularity":0.90,
         "core_minus_ring_hu":180.0,"center_hu":570.0}
    _, passed = m._score_plane(x, rca)
    assert not passed


def test_component_direction_orients_away_from_aorta():
    shape=(25,25,25)
    outside=np.zeros(shape,float)
    for x in range(shape[2]):
        outside[:,:,x]=max(0,x-12)
    pts=np.array([[12.,12.,x] for x in range(13,20)])
    t=m._component_direction(pts,np.array([12.,12.,13.]),np.ones(3),outside)
    assert t[2] > 0.5


def test_root_crop_is_physical_and_bounded():
    lo,hi,sl=m._root_crop((100,200,300),np.array([50.,100.,150.]),np.array([1.,1.,1.]),18.,45.)
    assert np.all(lo >= 0)
    assert np.all(hi <= np.array([100,200,300]))
    assert (hi-lo)[0] == 37
    assert (hi-lo)[1] == 91
    assert (hi-lo)[2] == 91
