import numpy as np
from openplaque import lcx_c6_monotonic_distal as m


def test_algorithm_and_baseline():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "lcx-c6-monotonic-distal-v1.0-lowmem"


def test_synthetic_monotonic_geometry():
    r = m.synthetic_monotonic_self_test()
    assert r["ok"]
    assert r["tortuosity"] < 1.1


def test_cone_directions_are_forward():
    t=np.array([1.,0.,0.])
    ds=m._cone_dirs(t)
    assert len(ds) > 10
    assert min(float(np.dot(d,t)) for d in ds) > 0.6
    assert all(abs(np.linalg.norm(d)-1.0) < 1e-6 for d in ds)
