import numpy as np

from openplaque.series6_native_all_component_gap_analysis_v1 import (
    Geometry,
    SHORT_GAP_MM,
    _aorta_surface_tree,
    _line_samples,
    _nearest_gap,
    _resample_path,
    synthetic_self_test,
)


def test_synthetic_self_test():
    assert synthetic_self_test()["ok"] is True


def test_resample_path():
    q=_resample_path(np.array([[0.,0.,0.],[2.,0.,0.]]),.5)
    assert q.shape==(5,3)
    assert np.allclose(q[-1],[2.,0.,0.])


def test_surface_gap_distance():
    g=Geometry(np.array([1.,1.,1.]),np.zeros(3),np.eye(3))
    a=np.zeros((9,9,9),bool)
    a[4:7,4:7,4:7]=True
    tree,z,xyz=_aorta_surface_tree(a,g)
    assoc=np.array([[4,4,1]],int)
    gap=_nearest_gap(assoc,tree,z,xyz,g)
    assert np.isclose(gap["gap_distance_mm"],3.0)
    assert gap["gap_distance_mm"] <= SHORT_GAP_MM


def test_line_samples_endpoints():
    arr=np.linspace(0,300,11).reshape(11,1,1).astype(float)
    g=Geometry(np.array([1.,1.,1.]),np.zeros(3),np.eye(3))
    pts,hu=_line_samples(arr,g,np.array([0.,0.,0.]),np.array([0.,0.,10.]),1.0)
    assert len(hu)==11
    assert np.isclose(hu[0],0)
    assert np.isclose(hu[-1],300)
