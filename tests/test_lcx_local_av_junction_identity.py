from openplaque.lcx_local_av_junction_identity import synthetic_local_junction_self_test


def test_synthetic_local_junction_self_test():
    result = synthetic_local_junction_self_test()
    assert result['ok'] is True
    assert result['good_distance'] < result['bad_distance']
