from openplaque.secondary_terminal_transition import synthetic_transition_self_test


def test_synthetic_transition_detector():
    out = synthetic_transition_self_test()
    assert out['passed']
    assert out['last_supported_arc_mm'] < out['sustained_broadening_onset_arc_mm']
