import numpy as np
from openplaque.lcx_distal_source_extension import (
    Geometry, _common_prefix, _initial_angle, synthetic_extension_self_test
)

def test_synthetic_extension_self_test():
    r = synthetic_extension_self_test()
    assert r["ok"]
    assert 6.5 <= r["common_prefix_mm"] <= 7.5

def test_geometry_roundtrip_identity():
    g = Geometry(
        origin=np.array([-10.0, -20.0, 30.0]),
        spacing_xyz=np.array([0.4, 0.5, 0.6]),
        direction=np.eye(3),
        shape_zyx=np.array([100, 100, 100]),
    )
    p = np.array([[1.2, -3.4, 42.6], [2.0, 5.0, 50.0]])
    z = g.lps_to_zyx(p)
    q = g.zyx_to_lps(z)
    assert np.allclose(p, q)

def test_common_prefix_hierarchical_pair():
    t=np.linspace(0,12,61)
    p=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)])
    q=p.copy()
    q[t>9,1]=(t[t>9]-9)*0.5
    c,s,d=_common_prefix([p,q],step=0.2,tol_mm=0.3,sustain=3,start_check_mm=2)
    assert 8.5 <= s[-1] <= 9.5

def test_initial_angle():
    p=np.column_stack([np.linspace(0,5,30),np.zeros(30),np.zeros(30)])
    assert _initial_angle(np.array([1.,0,0]),p) < 1.0
