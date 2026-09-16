import json

import numpy as np
import pandas as pd

from openplaque import left_coronary_source_ostium_discovery as base
from openplaque import left_coronary_source_ostium_discovery_v1_2 as m


def test_algorithm_and_baseline_are_frozen():
    assert m.BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert m.ALGORITHM == "left-coronary-source-ostium-discovery-v1.2-performance"


def test_synthetic_root_component_self_test():
    r = m.synthetic_root_component_self_test()
    assert r["ok"]
    assert r["components"] >= 2


def test_non_qc_gate_matches_frozen_v11_terms():
    v_thr = 0.04
    good = {
        "length_mm": 6.0,
        "tortuosity": 1.2,
        "robust_hu_fraction": 0.95,
        "p10_vesselness": 0.03,
        "outside_aorta_gain_mm": 3.5,
    }
    assert m._cheap_path_gate(good, v_thr)
    for key, bad_value in [
        ("length_mm", 4.99),
        ("tortuosity", 1.81),
        ("robust_hu_fraction", 0.89),
        ("p10_vesselness", 0.019),
        ("outside_aorta_gain_mm", 2.99),
    ]:
        q = dict(good)
        q[key] = bad_value
        assert not m._cheap_path_gate(q, v_thr)


def test_serial_gate_matches_frozen_v11_terms():
    rca = {"median_radius_mm": 1.6}
    q = {"plane_pass_fraction": 0.75, "median_plane_score": 0.70, "median_radius_mm": 1.7}
    assert m._serial_gate(q, rca)
    q2 = dict(q)
    q2["plane_pass_fraction"] = 0.59
    assert not m._serial_gate(q2, rca)


def test_cached_frangi_matches_frozen_formula_and_reuses_cache(tmp_path):
    z, y, x = np.indices((11, 11, 11))
    vol = np.full((11, 11, 11), 100.0, np.float32)
    vol[((y - 5) ** 2 + (x - 5) ** 2 <= 2) & (z >= 2) & (z <= 8)] = 600.0
    spacing = np.array([1.0, 1.0, 1.0])
    scales = (0.60,)
    expected = base._frangi_3d(vol, spacing, scales=scales)
    got = m._frangi_3d_cached(vol, spacing, scales=scales, cache_dir=tmp_path, reuse_cache=True)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-7)
    assert (tmp_path / "root_vesselness.npy").exists()
    meta = json.loads((tmp_path / "root_vesselness_meta.json").read_text())
    assert meta["completed_scales"] == 1
    got2 = m._frangi_3d_cached(vol, spacing, scales=scales, cache_dir=tmp_path, reuse_cache=True)
    np.testing.assert_array_equal(np.asarray(got2), expected)


def test_impossible_finalist_skips_serial_qc(monkeypatch):
    outside = np.zeros((9, 9, 9), float)
    comp = {
        "component_id": 1,
        "n_voxels": 3,
        "physical_span_mm": 2.0,
        "points": np.array([[4, 4, 4], [4, 4, 5], [4, 4, 6]], dtype=np.intp),
    }
    fake_path = np.array([[4.0, 4.0, 4.0], [4.0, 4.0, 5.0]])
    fake_state = (1.0, [fake_path[0], fake_path[1]], np.array([0.0, 0.0, 1.0]), 1.0)

    monkeypatch.setattr(base, "_component_direction", lambda *a, **k: np.array([0.0, 0.0, 1.0]))
    monkeypatch.setattr(base, "_beam_source", lambda *a, **k: [fake_state])
    monkeypatch.setattr(
        base,
        "_path_metrics",
        lambda *a, **k: {
            "length_mm": 5.2,
            "tortuosity": 1.1,
            "robust_hu_fraction": 0.80,
            "p10_vesselness": 0.1,
            "outside_aorta_gain_mm": 4.0,
            "median_vesselness": 0.2,
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("serial QC must not run for a path that already cannot pass")

    monkeypatch.setattr(base, "_serial_qc", forbidden)
    p, sm, qdf = m._trace_component(
        comp,
        np.zeros((9, 9, 9), np.float32),
        np.zeros((9, 9, 9), np.float32),
        outside,
        np.ones(3),
        0.02,
        {"median_radius_mm": 1.6, "median_center_hu": 560.0},
        "test",
    )
    assert p is not None
    assert not sm["accepted"]
    assert sm["performance"]["serial_qc_evaluated"] == 0
    assert sm["performance"]["serial_qc_skipped"] == 1
    assert isinstance(qdf, pd.DataFrame) and qdf.empty
