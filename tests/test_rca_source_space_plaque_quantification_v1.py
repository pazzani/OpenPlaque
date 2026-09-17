import numpy as np
from openplaque import rca_source_space_plaque_geometry_v1 as g
from openplaque import rca_source_space_plaque_quantification_v1 as q


def test_synthetic_geometry():
    out=q.synthetic_self_test()
    assert out["ok"] is True
    assert abs(out["circle_area_mm2"]-np.pi*1.5**2)<1e-9


def test_shells_and_nominal():
    assert g.NOMINAL_SHELL_MM in g.SHELL_THICKNESSES_MM
    assert tuple(sorted(g.SHELL_THICKNESSES_MM))==g.SHELL_THICKNESSES_MM
    assert all(x>0 for x in g.SHELL_THICKNESSES_MM)


def test_arc_weights_integrate_length():
    qv=np.arange(0,10.0+1e-9,.5)
    assert abs(g.arc_weights(qv).sum()-10.0)<1e-12


def test_hu_bin_boundaries_are_explicit():
    # Research proxy deliberately separates fat-like contamination below -30 HU.
    vals=np.array([-100.,-30.,29.9,30.,129.9,130.,349.9,350.])
    fat=vals<-30; low=(vals>=-30)&(vals<30); ncp=(vals>=30)&(vals<130); mixed=(vals>=130)&(vals<350); calc=vals>=350
    assert fat.sum()==1 and low.sum()==2 and ncp.sum()==2 and mixed.sum()==2 and calc.sum()==1


def test_status_boundary_not_clinical_tpv():
    assert "PROXY" in q.STATUS_PASS
    assert q.MIN_STATION_QC_FRACTION>=.9
    assert q.MIN_PROFILE_CORRELATION>=.8
