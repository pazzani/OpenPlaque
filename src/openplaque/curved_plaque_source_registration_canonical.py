from __future__ import annotations

import json
from pathlib import Path

from openplaque import curved_plaque_source_registration as _base

CANONICAL_DIRNAME = 'Canonical_Source_Coronary_Centerlines_v1'
CANONICAL_SCHEMA = 'openplaque-canonical-source-coronary-centerlines-v1'


def _resolve_canonical_root(drive_root: Path) -> Path:
    root = _base._find_dir_recursive(Path(drive_root), CANONICAL_DIRNAME)
    state_path = root / 'run_state.json'
    manifest_path = root / 'canonical_manifest.json'
    if not state_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('Canonical source-centerline bundle is incomplete: run_state.json/canonical_manifest.json missing.')
    state = json.loads(state_path.read_text())
    if state.get('status') != 'COMPLETE':
        raise RuntimeError(f"Canonical source-centerline bundle status is {state.get('status')!r}, not COMPLETE.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != CANONICAL_SCHEMA:
        raise RuntimeError(f"Unexpected canonical manifest schema: {manifest.get('schema')!r}")
    required = [
        root / 'RCA_canonical_source_centerline.csv',
        root / 'LAD_canonical_source_centerline.csv',
        root / 'SECONDARY_canonical_source_trajectory.csv',
        root / 'SECONDARY_quantitative_compact_lumen_0_12p2mm.csv',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Canonical source-centerline bundle missing required files: ' + ', '.join(missing))
    return root


def _canonical_find_one(root: Path, name: str):
    """Resolve the old RCA/LAD filenames inside the canonical bundle.

    This implementation deliberately does not call ``_base._find_one`` because
    ``run`` temporarily monkey-patches that symbol to this function. Calling it
    through ``_base`` here would therefore recurse indefinitely.
    """
    mapping = {
        'RCA_source_centerline.csv': 'RCA_canonical_source_centerline.csv',
        'LAD_source_centerline.csv': 'LAD_canonical_source_centerline.csv',
    }
    target = mapping.get(name, name)
    hits = list(Path(root).rglob(target))
    if not hits:
        raise FileNotFoundError(f'Could not find {target} under {root}')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None, study_zip=None, source_cache=None, series_map=None):
    drive_root = Path(drive_root)
    canonical_root = _resolve_canonical_root(drive_root)
    output_root = output_root or drive_root / 'Curved_Plaque_to_Source_Registration_Canonical_v1'

    original_find_dir = _base._find_dir_recursive
    original_find_one = _base._find_one

    def patched_find_dir(root, dirname):
        if dirname == 'Source_Volume_Coronary_Centerlines':
            return canonical_root
        return original_find_dir(root, dirname)

    try:
        _base._find_dir_recursive = patched_find_dir
        _base._find_one = _canonical_find_one
        result = _base.run(
            drive_root=str(drive_root),
            output_root=str(output_root),
            study_zip=study_zip,
            source_cache=source_cache,
            series_map=series_map,
        )
        out = Path(result['output_dir'])
        provenance = {
            'canonical_centerline_root': str(canonical_root),
            'canonical_manifest': str(canonical_root / 'canonical_manifest.json'),
            'RCA_centerline': str(canonical_root / 'RCA_canonical_source_centerline.csv'),
            'LAD_centerline': str(canonical_root / 'LAD_canonical_source_centerline.csv'),
            'secondary_quantitative_cutoff_mm': 12.2,
            'note': 'RCA/LAD registration uses only the validated canonical source-centerline bundle. Deprecated LAD_source_centerline.csv and LCX_source_centerline.csv are not eligible inputs.',
        }
        (out / 'canonical_input_provenance.json').write_text(json.dumps(provenance, indent=2))
        return result
    finally:
        _base._find_dir_recursive = original_find_dir
        _base._find_one = original_find_one
