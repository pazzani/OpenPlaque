from openplaque.lcx_chamber_interface_midpoint_identity import synthetic_interface_self_test


def test_synthetic_interface_self_test():
    r = synthetic_interface_self_test()
    assert r['ok'] is True
    assert r['n_interface_midpoints'] >= 20
    assert r['good_distance'] < r['bad_distance']
