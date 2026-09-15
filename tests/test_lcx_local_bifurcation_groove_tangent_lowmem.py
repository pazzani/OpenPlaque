import numpy as np
import SimpleITK as sitk

from openplaque.lcx_local_bifurcation_groove_tangent_lowmem import (
    _dist_normal,
    _tree,
    synthetic_local_bifurcation_self_test,
)


def test_synthetic_lowmem_self_test():
    out = synthetic_local_bifurcation_self_test()
    assert out['ok'] is True
    assert 14.5 <= out['consensus_mm'] <= 15.5


def test_sparse_surface_tree(tmp_path):
    a = np.zeros((9, 10, 11), np.uint8)
    a[2:7, 2:8, 2:9] = 1
    im = sitk.GetImageFromArray(a)
    im.SetSpacing((0.7, 0.8, 0.9))
    p = tmp_path / 'mask.nii.gz'
    sitk.WriteImage(im, str(p))
    tree = _tree(p, surface=True, max_points=10000)
    assert tree.n > 0
    d, n, good = _dist_normal(tree, np.array([[8.0, 4.0, 4.0]]))
    assert np.isfinite(d[0])
    assert good[0]
    assert np.isclose(np.linalg.norm(n[0]), 1.0)
