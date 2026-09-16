from openplaque.lcx_av_groove_locus_identity import synthetic_groove_locus_self_test


def test_synthetic_groove_locus_self_test():
    r = synthetic_groove_locus_self_test()
    assert r['ok'] is True
    assert r['follow_score'] > r['leave_score'] + 0.10
