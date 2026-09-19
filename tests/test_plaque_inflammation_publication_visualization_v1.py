import numpy as np
import pandas as pd

from openplaque.plaque_inflammation_publication_visualization_v1 import (
    _normalize_long,
    build_band_table,
    synthetic_self_test,
)


def test_synthetic_self_test():
    assert synthetic_self_test()["ok"] is True


def test_weighted_local_mean_bands_sum_to_one():
    d=pd.DataFrame({
        "arc_start_mm":[10,11,12,13],
        "arc_end_mm":[11,12,13,14],
        "fat_voxels":[10,20,30,40],
        "mean_hu":[-160,-130,-90,-50],
    })
    n=_normalize_long(d,"RCA")
    b=build_band_table({"RCA":n})
    assert np.isclose(b.fraction_of_fat_voxels.sum(),1.0)
    assert np.allclose(b.fat_voxels_weight.to_numpy(),[10,20,30,40])


def test_local_coordinates_start_at_zero():
    d=pd.DataFrame({
        "arc_start_mm":[5,6],
        "arc_end_mm":[6,7],
        "fat_voxels":[100,100],
        "mean_hu":[-90,-80],
    })
    n=_normalize_long(d,"LAD")
    assert np.isclose(n.local_start_mm.min(),0.0)
    assert np.isclose(n.local_mid_mm.iloc[0],0.5)
