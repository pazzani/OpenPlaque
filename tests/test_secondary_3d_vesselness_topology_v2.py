import numpy as np

from openplaque.secondary_3d_vesselness_topology_v2 import finite_dist_coords


def test_finite_dist_coords_unravels_flat_dijkstra_ids():
    shape = (63, 50, 60)
    dist = np.full(np.prod(shape), np.inf, dtype=np.float32)
    expected = [(0, 0, 0), (10, 20, 30), (62, 49, 59)]
    for p in expected:
        dist[np.ravel_multi_index(p, shape)] = 1.0
    got = finite_dist_coords(dist, shape)
    assert [tuple(x) for x in got.tolist()] == expected
