import json
from pathlib import Path

from openplaque import curved_plaque_source_registration as _base
from openplaque.curved_plaque_source_registration_canonical import (
    CANONICAL_SCHEMA,
    _canonical_find_one,
    _resolve_canonical_root,
)


def _make_bundle(tmp_path: Path):
    root = tmp_path / 'Canonical_Source_Coronary_Centerlines_v1'
    root.mkdir()
    (root / 'run_state.json').write_text(json.dumps({'status': 'COMPLETE'}))
    (root / 'canonical_manifest.json').write_text(json.dumps({'schema': CANONICAL_SCHEMA}))
    for name in [
        'RCA_canonical_source_centerline.csv',
        'LAD_canonical_source_centerline.csv',
        'SECONDARY_canonical_source_trajectory.csv',
        'SECONDARY_quantitative_compact_lumen_0_12p2mm.csv',
    ]:
        (root / name).write_text('arc_mm,lps_x_mm,lps_y_mm,lps_z_mm\n0,0,0,0\n')
    return root


def test_resolve_complete_canonical_bundle(tmp_path):
    root = _make_bundle(tmp_path)
    assert _resolve_canonical_root(tmp_path) == root


def test_canonical_filename_mapping(tmp_path):
    root = _make_bundle(tmp_path)
    assert _canonical_find_one(root, 'RCA_source_centerline.csv').name == 'RCA_canonical_source_centerline.csv'
    assert _canonical_find_one(root, 'LAD_source_centerline.csv').name == 'LAD_canonical_source_centerline.csv'


def test_filename_mapping_survives_runtime_monkeypatch(tmp_path):
    root = _make_bundle(tmp_path)
    original = _base._find_one
    try:
        _base._find_one = _canonical_find_one
        assert _base._find_one(root, 'RCA_source_centerline.csv').name == 'RCA_canonical_source_centerline.csv'
        assert _base._find_one(root, 'LAD_source_centerline.csv').name == 'LAD_canonical_source_centerline.csv'
    finally:
        _base._find_one = original


def test_reject_incomplete_bundle(tmp_path):
    root = _make_bundle(tmp_path)
    (root / 'run_state.json').write_text(json.dumps({'status': 'STARTED'}))
    try:
        _resolve_canonical_root(tmp_path)
    except RuntimeError as e:
        assert 'not COMPLETE' in str(e)
    else:
        raise AssertionError('Expected incomplete canonical bundle to be rejected')
