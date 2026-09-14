from openplaque.master_coronary_anatomy_qc import synthetic_master_qc_self_test


def test_master_coronary_anatomy_qc_self_test():
    out = synthetic_master_qc_self_test()
    assert out['passed']
    assert abs(out['arc_length_mm'] - 2.0) < 1e-6
    assert out['support_classes'][3] == 'local_island_unaccepted'
