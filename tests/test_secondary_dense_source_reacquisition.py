from openplaque.secondary_dense_source_reacquisition import synthetic_dense_source_self_test, _direction_fan


def test_dense_source_self_test():
    out = synthetic_dense_source_self_test()
    assert out['passed']
    assert out['n_coarse_orientations'] == 17


def test_direction_fan_normalized():
    dirs = _direction_fan([0, 1, 0])
    assert len(dirs) == 17
    for d in dirs:
        assert abs(float((d*d).sum()) - 1.0) < 1e-6
