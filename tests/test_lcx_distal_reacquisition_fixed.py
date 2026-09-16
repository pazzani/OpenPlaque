import numpy as np

from openplaque import lcx_distal_reacquisition_fixed as m


def test_reachable_coordinates_unravels_flat_distance_field():
    shape = (3, 4, 5)
    d = np.full(np.prod(shape), np.inf, float)
    for zyx in [(0, 0, 0), (1, 2, 3), (2, 3, 4)]:
        d[np.ravel_multi_index(zyx, shape)] = 0.0
    coords = m.reachable_coordinates(d, shape)
    got = {tuple(x) for x in coords.tolist()}
    assert got == {(0, 0, 0), (1, 2, 3), (2, 3, 4)}
    assert coords.shape[1] == 3


def test_dijkstra_bugfix_returns_roi_shaped_distance_field():
    shape = (4, 5, 6)
    mask = np.ones(shape, bool)
    cost = np.ones(shape, np.float32)
    dist, prev, reached = m._dijkstra_reshaped(
        mask, cost, (1, 1, 1), np.array([1.0, 1.0, 1.0]), max_cost=2.0
    )
    assert dist.shape == shape
    assert prev.ndim == 1
    coords = np.column_stack(np.nonzero(np.isfinite(dist)))
    assert coords.shape[1] == 3
    assert np.all(coords >= 0)
    assert np.all(coords < np.asarray(shape))


def test_scientific_identity_is_unchanged():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "lcx-distal-reacquisition-v1.0-lowmem"
    assert m.PATCH_VERSION == "dijkstra-distance-shape-v1"
