import numpy as np

from openplaque.left_proximal_trunk_continuation_identity_audit_v1 import (
    association_profile,
    identity_gates,
)


def test_identical_reversed_curve_has_strong_geometric_identity():
    x = np.linspace(0.0, 20.0, 81)
    ref = np.column_stack([x, 0*x, 0*x])
    query = ref[::-1].copy()
    fwd, _ = association_profile(query, ref)
    rev, _ = association_profile(ref, query)
    endpoint = {
        "best_continuation_to_LAD_endpoint_mm": 0.2,
        "best_other_continuation_to_common_split_mm": 0.3,
        "anchors_distinct": True,
    }
    gates = identity_gates(fwd, rev, endpoint, 0.0)
    assert all(gates.values())
    assert fwd["fraction_within_2mm"] == 1.0
    assert fwd["median_tangent_alignment_within_2mm"] > 0.99
    assert abs(fwd["arc_mapping_correlation_within_2mm"]) > 0.99


def test_offset_curve_fails_identity_overlap():
    x = np.linspace(0.0, 20.0, 81)
    ref = np.column_stack([x, 0*x, 0*x])
    query = np.column_stack([x, 5 + 0*x, 0*x])
    fwd, _ = association_profile(query, ref)
    rev, _ = association_profile(ref, query)
    endpoint = {
        "best_continuation_to_LAD_endpoint_mm": 0.2,
        "best_other_continuation_to_common_split_mm": 0.3,
        "anchors_distinct": True,
    }
    gates = identity_gates(fwd, rev, endpoint, 0.0)
    assert not gates["forward_min_distance_le_1mm"]
    assert not gates["forward_fraction_within_2mm_ge_0_80"]
    assert not gates["reverse_fraction_within_2mm_ge_0_80"]


def test_endpoint_anchors_must_be_distinct():
    x = np.linspace(0.0, 20.0, 81)
    p = np.column_stack([x, 0*x, 0*x])
    fwd, _ = association_profile(p, p)
    rev, _ = association_profile(p, p)
    endpoint = {
        "best_continuation_to_LAD_endpoint_mm": 0.1,
        "best_other_continuation_to_common_split_mm": 0.1,
        "anchors_distinct": False,
    }
    gates = identity_gates(fwd, rev, endpoint, 0.0)
    assert not gates["endpoint_anchors_are_distinct"]
