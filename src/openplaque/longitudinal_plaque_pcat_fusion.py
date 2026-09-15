from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = '0593b453959f5a353d644267fbeef24b514ef4d7'
LONGITUDINAL_SCORE_GATE = 0.30
LONGITUDINAL_GRADIENT_GATE = 0.15
ALLOWED_REGISTRATION_STATUSES = {
    'PARTIAL_LONGITUDINAL_MAPPING_ONLY',
    'VALIDATED_CURVED_TO_SOURCE_RESEARCH_MAPPING',
}


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _find_dir_recursive(root: Path, dirname: str) -> Path:
    root = Path(root)
    direct = root / dirname
    if direct.is_dir():
        return direct
    hits = [p for p in root.rglob(dirname) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f'Could not find {dirname} under {root}')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _require_file(root: Path, name: str) -> Path:
    direct = Path(root) / name
    if direct.exists():
        return direct
    hits = list(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _load_registration(reg_root: Path, vessel: str):
    p = reg_root / vessel / 'registration_model.json'
    if not p.exists():
        raise FileNotFoundError(f'Missing canonical registration model: {p}')
    model = json.loads(p.read_text())
    if model.get('vessel') != vessel:
        raise RuntimeError(f'{p} identifies vessel {model.get("vessel")!r}, expected {vessel!r}.')
    status = model.get('status')
    if status not in ALLOWED_REGISTRATION_STATUSES:
        raise RuntimeError(f'{vessel} registration status {status!r} does not support longitudinal fusion.')
    longfit = model.get('longitudinal_fit', {})
    if float(longfit.get('score', -np.inf)) < LONGITUDINAL_SCORE_GATE:
        raise RuntimeError(f'{vessel} longitudinal score failed gate.')
    if float(longfit.get('gradient_corr', -np.inf)) < LONGITUDINAL_GRADIENT_GATE:
        raise RuntimeError(f'{vessel} longitudinal gradient correlation failed gate.')
    return model


def _canonical_length(canonical_root: Path, vessel: str) -> tuple[float, Path]:
    name = f'{vessel}_canonical_source_centerline.csv'
    p = canonical_root / name
    if not p.exists():
        raise FileNotFoundError(f'Missing canonical centerline {p}')
    df = pd.read_csv(p)
    if 'arc_mm' not in df.columns or len(df) < 3:
        raise RuntimeError(f'Invalid canonical centerline {p}')
    arc = pd.to_numeric(df['arc_mm'], errors='coerce')
    if not np.isfinite(arc).all() or float(arc.max()) <= 5:
        raise RuntimeError(f'Invalid arc_mm in {p}')
    return float(arc.max()), p


def map_native_profile_to_source(profile: pd.DataFrame, registration_model: dict, source_length_mm: float) -> pd.DataFrame:
    required = {
        'native_axis', 'slice_index', 'vote_sum', 'voxels_vote_ge3',
        'voxels_vote_ge4', 'voxels_vote_5', 'mean_vote_among_any_plaque'
    }
    missing = required.difference(profile.columns)
    if missing:
        raise RuntimeError(f'Native plaque profile missing columns: {sorted(missing)}')
    axes = pd.to_numeric(profile['native_axis'], errors='coerce').dropna().astype(int).unique()
    if len(axes) != 1:
        raise RuntimeError(f'Expected one native longitudinal axis, found {axes.tolist()}')
    lf = registration_model['longitudinal_fit']
    long_axis = int(lf['long_axis'])
    if int(axes[0]) != long_axis:
        raise RuntimeError(
            f'Plaque profile native_axis={int(axes[0])} does not match validated registration long_axis={long_axis}.'
        )
    start = float(lf['start_px'])
    npx = int(lf['n_pixels'])
    if npx < 2:
        raise RuntimeError('Registration n_pixels must be >=2.')
    pix = pd.to_numeric(profile['slice_index'], errors='coerce').to_numpy(float)
    frac = (pix - start) / float(npx - 1)
    if bool(lf.get('long_flip', False)):
        frac = 1.0 - frac
    valid = np.isfinite(frac) & (frac >= 0.0) & (frac <= 1.0)
    out = profile.loc[valid].copy()
    out['registration_pixel_fraction'] = frac[valid]
    out['source_arc_mm'] = frac[valid] * float(source_length_mm)
    out['registration_longitudinal_score'] = float(lf['score'])
    out['registration_gradient_corr'] = float(lf['gradient_corr'])
    return out.sort_values('source_arc_mm').reset_index(drop=True)


def aggregate_source_bins(mapped: pd.DataFrame, source_length_mm: float) -> pd.DataFrame:
    if not len(mapped):
        raise RuntimeError('No native plaque-profile slices fall inside the validated longitudinal registration window.')
    q = mapped.copy()
    q['source_bin'] = np.floor(q['source_arc_mm'].to_numpy(float) + 1e-9).astype(int)
    max_bin = max(0, int(math.ceil(float(source_length_mm))) - 1)
    q['source_bin'] = q['source_bin'].clip(0, max_bin)
    for c in ['vote_sum', 'voxels_vote_ge3', 'voxels_vote_ge4', 'voxels_vote_5']:
        q[c] = pd.to_numeric(q[c], errors='coerce').fillna(0.0)
    q['mean_vote_among_any_plaque'] = pd.to_numeric(q['mean_vote_among_any_plaque'], errors='coerce')
    grouped = q.groupby('source_bin', as_index=False).agg(
        native_slices_mapped=('slice_index', 'count'),
        mapped_native_vote_sum=('vote_sum', 'sum'),
        mapped_native_voxels_vote_ge3=('voxels_vote_ge3', 'sum'),
        mapped_native_voxels_vote_ge4=('voxels_vote_ge4', 'sum'),
        mapped_native_voxels_vote_5=('voxels_vote_5', 'sum'),
        max_native_mean_vote=('mean_vote_among_any_plaque', 'max'),
        mean_native_mean_vote=('mean_vote_among_any_plaque', 'mean'),
        source_arc_sample_min_mm=('source_arc_mm', 'min'),
        source_arc_sample_max_mm=('source_arc_mm', 'max'),
    )
    bins = pd.DataFrame({'source_bin': np.arange(max_bin + 1, dtype=int)})
    out = bins.merge(grouped, how='left', on='source_bin')
    count_cols = [
        'native_slices_mapped', 'mapped_native_vote_sum', 'mapped_native_voxels_vote_ge3',
        'mapped_native_voxels_vote_ge4', 'mapped_native_voxels_vote_5'
    ]
    out[count_cols] = out[count_cols].fillna(0)
    out['arc_start_mm'] = out['source_bin'].astype(float)
    out['arc_end_mm'] = np.minimum(out['arc_start_mm'] + 1.0, float(source_length_mm))
    out['any_plaque_signal'] = out['mapped_native_vote_sum'] > 0
    out['majority_3plus_signal'] = out['mapped_native_voxels_vote_ge3'] > 0
    out['high_4plus_signal'] = out['mapped_native_voxels_vote_ge4'] > 0
    out['strict_5of5_signal'] = out['mapped_native_voxels_vote_5'] > 0
    ordered = [
        'source_bin', 'arc_start_mm', 'arc_end_mm', 'native_slices_mapped',
        'mapped_native_vote_sum', 'mapped_native_voxels_vote_ge3',
        'mapped_native_voxels_vote_ge4', 'mapped_native_voxels_vote_5',
        'max_native_mean_vote', 'mean_native_mean_vote',
        'any_plaque_signal', 'majority_3plus_signal', 'high_4plus_signal', 'strict_5of5_signal',
        'source_arc_sample_min_mm', 'source_arc_sample_max_mm'
    ]
    return out[ordered]


def fuse_rca_pcat(plaque_bins: pd.DataFrame, pcat: pd.DataFrame) -> pd.DataFrame:
    required = {'wall_margin_mm', 'arc_start_mm', 'arc_end_mm', 'fat_voxels', 'mean_hu'}
    missing = required.difference(pcat.columns)
    if missing:
        raise RuntimeError(f'Locked RCA PCAT profile missing columns: {sorted(missing)}')
    p = pcat.copy()
    p['arc_start_mm'] = pd.to_numeric(p['arc_start_mm'], errors='coerce')
    p['arc_end_mm'] = pd.to_numeric(p['arc_end_mm'], errors='coerce')
    widths = p['arc_end_mm'] - p['arc_start_mm']
    if not np.allclose(widths, 1.0, atol=1e-6):
        raise RuntimeError('Locked PCAT longitudinal profile is not in 1-mm bins.')
    if float(p['arc_start_mm'].min()) < 9.99 or float(p['arc_end_mm'].max()) > 50.01:
        raise RuntimeError('Expected locked RCA PCAT window to be 10-50 mm.')
    p['source_bin'] = np.rint(p['arc_start_mm']).astype(int)
    p = p.rename(columns={'mean_hu': 'pcat_mean_hu', 'fat_voxels': 'pcat_fat_voxels', 'wall_margin_mm': 'pcat_wall_margin_mm'})
    keep = ['source_bin', 'pcat_wall_margin_mm', 'pcat_fat_voxels', 'pcat_mean_hu']
    out = plaque_bins.merge(p[keep], how='inner', on='source_bin')
    if len(out) != len(p):
        raise RuntimeError(f'RCA fusion produced {len(out)} bins for {len(p)} PCAT bins.')
    return out


def _weighted_mean(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float('nan')
    return float(np.average(v[m], weights=w[m]))


def _contiguous_intervals(profile: pd.DataFrame, signal_col: str, level: str, pcat_fusion: pd.DataFrame | None = None):
    sig = profile[signal_col].astype(bool).to_numpy()
    bins = profile['source_bin'].to_numpy(int)
    rows = []
    start_i = None
    for i, on in enumerate(np.r_[sig, False]):
        if on and start_i is None:
            start_i = i
        elif (not on) and start_i is not None:
            block = profile.iloc[start_i:i].copy()
            start_bin = int(block['source_bin'].iloc[0])
            end_bin = int(block['source_bin'].iloc[-1])
            rec = {
                'confidence_level': level,
                'arc_start_mm': float(block['arc_start_mm'].iloc[0]),
                'arc_end_mm': float(block['arc_end_mm'].iloc[-1]),
                'duration_mm': float(block['arc_end_mm'].iloc[-1] - block['arc_start_mm'].iloc[0]),
                'mapped_native_vote_sum': float(block['mapped_native_vote_sum'].sum()),
                'mapped_native_voxels_vote_ge3': float(block['mapped_native_voxels_vote_ge3'].sum()),
                'mapped_native_voxels_vote_ge4': float(block['mapped_native_voxels_vote_ge4'].sum()),
                'mapped_native_voxels_vote_5': float(block['mapped_native_voxels_vote_5'].sum()),
                'peak_native_mean_vote': float(block['max_native_mean_vote'].max()) if block['max_native_mean_vote'].notna().any() else np.nan,
                'pcat_overlap_bins': 0,
                'pcat_mean_hu_unweighted': np.nan,
                'pcat_mean_hu_fat_weighted': np.nan,
            }
            if pcat_fusion is not None:
                f = pcat_fusion[pcat_fusion['source_bin'].between(start_bin, end_bin)].copy()
                rec['pcat_overlap_bins'] = int(len(f))
                if len(f):
                    rec['pcat_mean_hu_unweighted'] = float(f['pcat_mean_hu'].mean())
                    rec['pcat_mean_hu_fat_weighted'] = _weighted_mean(f['pcat_mean_hu'], f['pcat_fat_voxels'])
            rows.append(rec)
            start_i = None
    return rows


def _plot_rca(out: Path, fusion: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = fusion['arc_start_mm'].to_numpy(float) + 0.5
    axes[0].plot(x, fusion['mapped_native_voxels_vote_ge3'], label='>=3/5 support')
    axes[0].plot(x, fusion['mapped_native_voxels_vote_ge4'], label='>=4/5 support')
    axes[0].plot(x, fusion['mapped_native_voxels_vote_5'], label='5/5 support')
    axes[0].set_ylabel('Mapped native support count')
    axes[0].set_title('RCA longitudinal plaque confidence on canonical source arc')
    axes[0].legend()
    axes[1].plot(x, fusion['pcat_mean_hu'])
    axes[1].set_ylabel('OpenPlaque PCAT attenuation (HU)')
    axes[1].set_xlabel('Canonical RCA arc length (mm)')
    axes[1].set_title('Locked RCA PCAT profile (10-50 mm)')
    fig.tight_layout()
    p = out / '01_RCA_longitudinal_plaque_PCAT_fusion.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def _plot_vessel_profile(out: Path, vessel: str, profile: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = profile['arc_start_mm'].to_numpy(float) + 0.5
    ax.plot(x, profile['mapped_native_voxels_vote_ge3'], label='>=3/5 support')
    ax.plot(x, profile['mapped_native_voxels_vote_ge4'], label='>=4/5 support')
    ax.plot(x, profile['mapped_native_voxels_vote_5'], label='5/5 support')
    ax.set_xlabel(f'Canonical {vessel} arc length (mm)')
    ax.set_ylabel('Mapped native support count')
    ax.set_title(f'{vessel} plaque confidence — validated longitudinal mapping only')
    ax.legend()
    fig.tight_layout()
    p = out / f'02_{vessel}_source_arc_plaque_confidence.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'Longitudinal_Plaque_PCAT_Fusion_v1')
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'run_state.json', {'status': 'STARTED', 'baseline': BASELINE})

    canonical_root = _find_dir_recursive(drive_root, 'Canonical_Source_Coronary_Centerlines_v1')
    canonical_state = json.loads((canonical_root / 'run_state.json').read_text())
    if canonical_state.get('status') != 'COMPLETE':
        raise RuntimeError('Canonical source-centerline bundle is not COMPLETE.')

    reg_root = _find_dir_recursive(drive_root, 'Curved_Plaque_to_Source_Registration_Canonical_v1')
    reg_state = json.loads((reg_root / 'run_state.json').read_text())
    if reg_state.get('state') != 'COMPLETE':
        raise RuntimeError('Canonical curved-plaque registration run is not COMPLETE.')

    atlas_root = _find_dir_recursive(drive_root, 'Plaque_Ensemble_Confidence_Atlas_v1')
    pcat_root = _find_dir_recursive(drive_root, 'PCAT_RCA_10_50_Reproducibility_Lock')
    pcat_path = _require_file(pcat_root, 'pcat_canonical_primary_longitudinal.csv')
    pcat = pd.read_csv(pcat_path)

    vessel_profiles = {}
    mapped_raw = {}
    registration_summary = {}
    provenance = {
        'baseline_commit': BASELINE,
        'canonical_centerline_root': str(canonical_root),
        'canonical_registration_root': str(reg_root),
        'plaque_confidence_atlas_root': str(atlas_root),
        'rca_pcat_reproducibility_lock_root': str(pcat_root),
        'rca_pcat_longitudinal_file': str(pcat_path),
        'coordinate_scope': 'one-dimensional canonical source-centerline arc only',
        'explicit_non_claims': [
            'No circumferential/radial plaque localization is asserted.',
            'No 3-D source-space plaque mask is asserted.',
            'Mapped native plaque-support counts are not anatomical mm3 and are not validated TPV.',
            'No LAD plaque-PCAT coupling is asserted because no locked LAD PCAT profile is used.',
        ],
    }

    for vessel in ('RCA', 'LAD'):
        model = _load_registration(reg_root, vessel)
        length, centerline_path = _canonical_length(canonical_root, vessel)
        native_path = _require_file(atlas_root, f'{vessel}_native_longitudinal_profile.csv')
        native = pd.read_csv(native_path)
        mapped = map_native_profile_to_source(native, model, length)
        bins = aggregate_source_bins(mapped, length)
        mapped.to_csv(out / f'{vessel}_native_profile_mapped_to_source_arc.csv', index=False)
        bins.to_csv(out / f'{vessel}_source_longitudinal_plaque_profile_1mm.csv', index=False)
        mapped_raw[vessel] = mapped
        vessel_profiles[vessel] = bins
        lf = model['longitudinal_fit']
        registration_summary[vessel] = {
            'registration_status': model['status'],
            'source_length_mm': length,
            'longitudinal_score': float(lf['score']),
            'longitudinal_gradient_corr': float(lf['gradient_corr']),
            'long_axis': int(lf['long_axis']),
            'start_px': int(lf['start_px']),
            'n_pixels': int(lf['n_pixels']),
            'long_flip': bool(lf.get('long_flip', False)),
            'canonical_centerline': str(centerline_path),
            'native_plaque_profile': str(native_path),
            'mapped_native_rows': int(len(mapped)),
        }

    rca_fusion = fuse_rca_pcat(vessel_profiles['RCA'], pcat)
    rca_fusion.to_csv(out / 'RCA_pcat_plaque_longitudinal_fusion_10_50.csv', index=False)

    interval_rows = []
    for col, level in [
        ('majority_3plus_signal', 'majority_3plus'),
        ('high_4plus_signal', 'high_4plus'),
        ('strict_5of5_signal', 'strict_5of5'),
    ]:
        interval_rows.extend(_contiguous_intervals(vessel_profiles['RCA'], col, level, rca_fusion))
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(out / 'RCA_longitudinal_plaque_intervals.csv', index=False)

    majority = rca_fusion['majority_3plus_signal'].astype(bool)
    overlap_summary = {
        'rca_pcat_bins': int(len(rca_fusion)),
        'rca_bins_any_plaque_signal': int(rca_fusion['any_plaque_signal'].sum()),
        'rca_bins_majority_3plus_signal': int(majority.sum()),
        'rca_bins_high_4plus_signal': int(rca_fusion['high_4plus_signal'].sum()),
        'rca_bins_strict_5of5_signal': int(rca_fusion['strict_5of5_signal'].sum()),
        'pcat_mean_hu_all_10_50_unweighted': float(rca_fusion['pcat_mean_hu'].mean()),
        'pcat_mean_hu_all_10_50_fat_weighted': _weighted_mean(rca_fusion['pcat_mean_hu'], rca_fusion['pcat_fat_voxels']),
        'pcat_mean_hu_majority_bins_unweighted': float(rca_fusion.loc[majority, 'pcat_mean_hu'].mean()) if majority.any() else np.nan,
        'pcat_mean_hu_majority_bins_fat_weighted': _weighted_mean(
            rca_fusion.loc[majority, 'pcat_mean_hu'], rca_fusion.loc[majority, 'pcat_fat_voxels']
        ) if majority.any() else np.nan,
        'pcat_mean_hu_nonmajority_bins_unweighted': float(rca_fusion.loc[~majority, 'pcat_mean_hu'].mean()) if (~majority).any() else np.nan,
    }

    figs = [
        _plot_rca(out, rca_fusion),
        _plot_vessel_profile(out, 'LAD', vessel_profiles['LAD']),
    ]

    summary = {
        'status': 'COMPLETE_LONGITUDINAL_RESEARCH_FUSION',
        'registration': registration_summary,
        'rca_overlap_summary': overlap_summary,
        'interpretation': (
            'Plaque confidence has been mapped onto canonical source-centerline arc using validated longitudinal registration only. '
            'RCA plaque support is aligned descriptively with the locked 10-50 mm OpenPlaque PCAT attenuation profile. '
            'This is a 1-D research fusion and does not validate circumferential plaque localization, source-space plaque volume, or TPV.'
        ),
        'lad_note': 'LAD source-arc plaque confidence is reported separately; no locked LAD PCAT profile is fused.',
    }
    _write_json(out / 'fusion_summary.json', summary)
    _write_json(out / 'input_provenance.json', provenance)

    html = out / 'OPENPLAQUE_LONGITUDINAL_PLAQUE_PCAT_FUSION_REPORT.html'
    interval_html = intervals.to_html(index=False) if len(intervals) else '<p>No RCA >=3/5, >=4/5, or 5/5 intervals were mapped into the canonical arc.</p>'
    html.write_text(
        '<html><head><meta charset="utf-8"><title>OpenPlaque Longitudinal Plaque + PCAT Fusion</title></head>'
        '<body style="font-family:Arial;max-width:1200px;margin:auto">'
        '<h1>OpenPlaque — Longitudinal Plaque-Confidence + PCAT Fusion</h1>'
        '<p><b>Research use only.</b></p>'
        '<p><b>Coordinate scope:</b> 1-D canonical source-centerline arc only. The curved plaque series are nonspatial/rotation-stack data. '
        'The plaque values below are mapped native support counts, not anatomical mm³, not validated TPV, and not a 3-D plaque mask.</p>'
        '<p><b>RCA PCAT:</b> locked OpenPlaque PCAT attenuation, 10-50 mm, 0.75-mm outer-wall margin, fat window -190 to -30 HU. '
        '<b>Not Caristo FAI-Score.</b></p>'
        '<h2>Registration provenance</h2>' + pd.DataFrame(registration_summary).T.to_html() +
        '<h2>RCA descriptive overlap</h2><pre>' + json.dumps(overlap_summary, indent=2, default=str) + '</pre>' +
        f'<img src="{figs[0].name}" style="max-width:100%">' +
        '<h2>RCA longitudinal plaque intervals</h2>' + interval_html +
        '<h2>LAD longitudinal plaque confidence</h2>'
        '<p>No LAD PCAT coupling is attempted because no locked LAD PCAT profile is supplied to this workflow.</p>' +
        f'<img src="{figs[1].name}" style="max-width:100%">' +
        '<h2>Interpretation boundary</h2><p>' + summary['interpretation'] + '</p>' +
        '</body></html>'
    )

    zf = out / 'OPENPLAQUE_LONGITUDINAL_PLAQUE_PCAT_FUSION_REPORT_BACK.zip'
    with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p != zf:
                z.write(p, p.relative_to(out))

    state = {
        'status': 'COMPLETE',
        'scientific_status': 'LONGITUDINAL_ONLY',
        'output_dir': str(out),
        'report': str(html),
        'zip': str(zf),
    }
    _write_json(out / 'run_state.json', state)
    return {'summary': summary, **state}
