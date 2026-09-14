from openplaque.secondary_distal_reacquisition import synthetic_reacquisition_self_test


def test_synthetic_reacquisition_self_test():
    result = synthetic_reacquisition_self_test()
    assert result['passed'], result
    assert 1.45 <= result['tube_radius_mm'] <= 2.05
    assert result['broad_radius_mm'] > 2.65
    assert result['n_orientation_hypotheses'] == 17
