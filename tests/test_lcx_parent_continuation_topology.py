from openplaque.lcx_parent_continuation_topology import synthetic_parent_continuation_self_test


def test_synthetic_parent_continuation():
    r = synthetic_parent_continuation_self_test()
    assert r['ok'] is True
    assert r['truncation']['truncation_gate_pass'] is True
    assert 19.5 <= r['common_mm'] <= 20.5
