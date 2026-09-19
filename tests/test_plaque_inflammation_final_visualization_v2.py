import numpy as np
import pandas as pd

from openplaque.plaque_inflammation_final_visualization_v2 import (
    PCAT_BANDS,
    build_voxel_bands,
    synthetic_self_test,
)


def test_synthetic_self_test():
    assert synthetic_self_test()["ok"] is True


def test_voxel_bands_are_true_counts():
    vals={
        "RCA":np.array([-170.,-130.,-90.,-50.]),
        "LAD":np.array([-160.,-125.,-85.,-45.]),
        "LCX":np.array([-180.,-140.,-100.,-60.]),
    }
    d=build_voxel_bands(vals)
    assert len(d)==12
    for vessel,a in vals.items():
        g=d[d.vessel==vessel]
        assert int(g.fat_voxels.sum())==len(a)
        assert np.isclose(g.fraction_of_fat_voxels.sum(),1.0)
        assert set(g.fat_voxels)=={1}


def test_band_edges_cover_fat_window_without_overlap():
    edges=[(x[0],x[1]) for x in PCAT_BANDS]
    assert edges[0][0]==-190.0
    assert edges[-1][1]==-30.0
    for a,b in zip(edges[:-1],edges[1:]):
        assert a[1]==b[0]
