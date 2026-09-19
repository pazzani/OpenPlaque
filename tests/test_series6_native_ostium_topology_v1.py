import numpy as np
from openplaque.series6_native_ostium_topology_v1 import (
    RCA_CONTACT_RADIUS_MM,
    MIN_SECOND_CONTACT_SEPARATION_MM,
    _resample_path,
    synthetic_self_test,
)

def test_synthetic_self_test():
    assert synthetic_self_test()["ok"] is True

def test_resample_path():
    q=_resample_path(np.array([[0.,0.,0.],[2.,0.,0.]]),.5)
    assert q.shape==(5,3)
    assert np.allclose(q[-1],[2.,0.,0.])

def test_contact_geometry():
    r=np.array([0.,0.,0.]); c1=np.array([1.,0.,0.]); c2=np.array([12.,0.,0.])
    assert np.linalg.norm(c1-r)<=RCA_CONTACT_RADIUS_MM
    assert np.linalg.norm(c2-c1)>=MIN_SECOND_CONTACT_SEPARATION_MM
