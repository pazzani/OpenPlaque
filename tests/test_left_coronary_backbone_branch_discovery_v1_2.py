import numpy as np

from openplaque.left_coronary_backbone_branch_discovery_v1_2 import (
    _seed_start_candidates,
    _select_preview_rays,
    synthetic_multistart_seed_self_test,
)


def test_synthetic_multistart_seed_self_test():
    out = synthetic_multistart_seed_self_test()
    assert out["ok"] is True
    assert out["candidate_starts"] >= 20
    assert out["max_radial_offset_mm"] >= 0.79


def test_seed_candidates_cover_submillimeter_cross_section_offset():
    x = np.linspace(0.0, 20.0, 81)
    backbone = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    starts = _seed_start_candidates(backbone, 10.0)
    pts = np.asarray([s["start"] for s in starts])
    for target in (
        np.array([10.0, 0.8, 0.0]),
        np.array([10.0, -0.8, 0.0]),
        np.array([10.0, 0.0, 0.8]),
        np.array([10.0, 0.0, -0.8]),
    ):
        assert float(np.min(np.linalg.norm(pts - target, axis=1))) < 0.05


def test_preview_selection_keeps_start_diversity():
    rays = []
    for i in range(8):
        rays.append({
            "preview_score": 10.0 - i,
            "start": np.array([float(i), 0.0, 0.0]),
            "direction": np.array([1.0, 0.0, 0.0]),
        })
    chosen = _select_preview_rays(rays, 6)
    assert len(chosen) == 6
    unique_starts = {tuple(np.round(r["start"], 3)) for r in chosen}
    assert len(unique_starts) >= 4
