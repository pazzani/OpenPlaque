import numpy as np
from openplaque.lcx_distal_source_extension import Geometry
from openplaque.lcx_distal_source_extension_guarded import _boundary_margin_mm


def test_boundary_margin_centered_identity_geometry():
    g = Geometry(
        origin=np.zeros(3),
        spacing_xyz=np.ones(3),
        direction=np.eye(3),
        shape_zyx=np.array([100, 100, 100]),
    )
    lo = np.array([10, 20, 30])
    shape = np.array([21, 21, 21])
    # local zyx=(10,10,10) corresponds global zyx=(20,30,40),
    # hence LPS xyz=(40,30,20) for identity geometry.
    m = _boundary_margin_mm(np.array([40.0, 30.0, 20.0]), g, lo, shape)
    assert np.isclose(m, 10.0)


def test_boundary_margin_near_edge():
    g = Geometry(
        origin=np.zeros(3),
        spacing_xyz=np.ones(3),
        direction=np.eye(3),
        shape_zyx=np.array([100, 100, 100]),
    )
    lo = np.array([10, 20, 30])
    shape = np.array([21, 21, 21])
    # local zyx=(1,10,10) -> margin 1 mm.
    m = _boundary_margin_mm(np.array([40.0, 30.0, 11.0]), g, lo, shape)
    assert np.isclose(m, 1.0)
