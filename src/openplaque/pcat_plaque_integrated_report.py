from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def _find_optional(root: Path, name: str):
    hits = list(root.rglob(name))
    if not hits:
        return None
    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def _metric_cards(canonical: pd.Series, plaque_conf=None):
    cards = [
        ('RCA PCAT mean', f"{canonical['pcat_mean_hu']:.1f} HU"),
        ('RCA PCAT median', f"{canonical['pcat_median_hu']:.1f} HU"),
        ('PCAT fat volume', f"{canonical['fat_volume_ml']:.2f} mL"),
        ('Sampling wall margin', f"{canonical['wall_margin_mm']:.2f} mm"),
    ]
    if plaque_conf is not None and len(plaque_conf):
        r = plaque_conf.loc[plaque_conf['vessel'].eq('RCA')]
        if len(r):
            rr = r.iloc[0]
            cards += [('RCA plaque >=4/5', f"{rr['high_4plus_mm3']:.0f} mm³"), ('RCA plaque 5/5', f"{rr['strict_5of5_mm3']:.0f} mm³")]
    return cards


def _save_pcat_longitudinal(out, longitudinal):
    x = (longitudinal['arc_start_mm'] + longitudinal['arc_end_mm']) / 2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, longitudinal['mean_hu'], marker='o', markersize=3)
    ax.axhline(longitudinal['mean_hu'].mean(), linestyle='--', linewidth=1, label='mean of 1-mm bins')
    ax.set_xlabel('RCA source-centerline arc length (mm)')
    ax.set_ylabel('PCAT attenuation (HU)')
    ax.set_title('Canonical RCA PCAT longitudinal profile')
    ax.legend(); fig.tight_layout()
    p = out / '01_rca_pcat_longitudinal.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def _save_pcat_sensitivity(out, sensitivity):
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'wall_margin_mm' in sensitivity and 'pcat_mean_hu' in sensitivity:
        q = sensitivity.dropna(subset=['wall_margin_mm'])
        ax.plot(q['wall_margin_mm'], q['pcat_mean_hu'], marker='o')
        ax.set_xlabel('Outer-wall margin (mm)')
        ax.set_ylabel('PCAT mean (HU)')
    else:
        ax.text(.5, .5, 'Sensitivity columns unavailable', ha='center', va='center')
    ax.set_title('RCA PCAT geometry sensitivity')
    fig.tight_layout()
    p = out / '02_rca_pcat_geometry_sensitivity.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def _save_plaque_fold_variation(out, fold_summary):
    q = fold_summary[fold_summary['fold'].astype(str) != 'consensus'].copy()
    q['fold_num'] = pd.to_numeric(q['fold'], errors='coerce')
    fig, ax = plt.subplots(figsize=(9, 5))
    for vessel in ['RCA', 'LAD']:
        v = q[q['vessel'].eq(vessel)]
        if len(v): ax.plot(v['fold_num'], v['plaque_volume_mm3'], marker='o', label=vessel)
    ax.set_xlabel('nnU-Net fold')
    ax.set_ylabel('Plaque volume (mm³)')
    ax.set_title('Plaque model fold-to-fold variation')
    ax.legend(); fig.tight_layout()
    p = out / '03_plaque_fold_variation.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def _save_separate_tracks(out, pcat_longitudinal, plaque_profile):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x = (pcat_longitudinal['arc_start_mm'] + pcat_longitudinal['arc_end_mm']) / 2
    axes[0].plot(x, pcat_longitudinal['mean_hu'])
    axes[0].set_xlabel('RCA source-centerline arc (mm)')
    axes[0].set_ylabel('PCAT HU')
    axes[0].set_title('Source-coordinate RCA PCAT')
    axes[1].plot(plaque_profile['native_longitudinal_mm'], plaque_profile['voxels_vote_ge3'], label='>=3/5')
    axes[1].plot(plaque_profile['native_longitudinal_mm'], plaque_profile['voxels_vote_ge4'], label='>=4/5')
    axes[1].plot(plaque_profile['native_longitudinal_mm'], plaque_profile['voxels_vote_5'], label='5/5')
    axes[1].set_xlabel('Plaque-model native longitudinal coordinate (mm)')
    axes[1].set_ylabel('Plaque voxels per slice')
    axes[1].set_title('RCA plaque confidence — separate native coordinate system')
    axes[1].legend()
    fig.suptitle('PCAT and plaque shown together without false coordinate registration')
    fig.tight_layout()
    p = out / '04_rca_pcat_and_plaque_separate_tracks.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'PCAT_Plaque_Integrated_Report_v1')
    out.mkdir(parents=True, exist_ok=True)

    canonical_path = _find_one(drive_root, 'pcat_canonical_primary.csv')
    longitudinal_path = _find_one(drive_root, 'pcat_canonical_primary_longitudinal.csv')
    sensitivity_path = _find_optional(drive_root, 'pcat_circular_sensitivity_locked.csv')
    canonical = pd.read_csv(canonical_path).iloc[0]
    longitudinal = pd.read_csv(longitudinal_path)
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path else pd.DataFrame()

    fold_summary_path = _find_one(drive_root, 'plaque_5fold_summary.csv')
    fold_summary = pd.read_csv(fold_summary_path)
    plaque_conf_path = _find_optional(drive_root, 'plaque_vote_confidence_summary.csv')
    plaque_conf = pd.read_csv(plaque_conf_path) if plaque_conf_path else None
    rca_profile_path = _find_optional(drive_root, 'RCA_native_longitudinal_profile.csv')
    rca_profile = pd.read_csv(rca_profile_path) if rca_profile_path else None

    fold_stats = []
    for vessel in ['RCA', 'LAD']:
        vals = fold_summary[(fold_summary['vessel'] == vessel) & (fold_summary['fold'].astype(str) != 'consensus')]['plaque_volume_mm3'].astype(float)
        consensus = fold_summary[(fold_summary['vessel'] == vessel) & (fold_summary['fold'].astype(str) == 'consensus')]['plaque_volume_mm3'].astype(float)
        fold_stats.append({
            'vessel': vessel,
            'fold_mean_mm3': float(vals.mean()),
            'fold_sd_mm3': float(vals.std(ddof=1)),
            'fold_cv_pct': float(100 * vals.std(ddof=1) / vals.mean()) if vals.mean() else np.nan,
            'fold_min_mm3': float(vals.min()),
            'fold_max_mm3': float(vals.max()),
            'majority_consensus_mm3': float(consensus.iloc[0]) if len(consensus) else np.nan,
        })
    fold_stats_df = pd.DataFrame(fold_stats)
    fold_stats_df.to_csv(out / 'plaque_fold_uncertainty_summary.csv', index=False)

    warm = longitudinal.loc[longitudinal['mean_hu'].idxmax()]
    cool = longitudinal.loc[longitudinal['mean_hu'].idxmin()]
    inflammation_summary = {
        'metric_name': 'OpenPlaque PCAT Attenuation',
        'not_fai_score': True,
        'rca_segment_mm': [10.0, 50.0],
        'pcat_mean_hu': float(canonical['pcat_mean_hu']),
        'pcat_median_hu': float(canonical['pcat_median_hu']),
        'pcat_sd_hu': float(canonical['pcat_sd_hu']),
        'pcat_fat_volume_ml': float(canonical['fat_volume_ml']),
        'pcat_shell_volume_ml': float(canonical['shell_volume_ml']),
        'pcat_fat_fraction': float(canonical['fat_fraction']),
        'warmest_1mm_bin': {'arc_start_mm': float(warm['arc_start_mm']), 'arc_end_mm': float(warm['arc_end_mm']), 'mean_hu': float(warm['mean_hu'])},
        'coolest_1mm_bin': {'arc_start_mm': float(cool['arc_start_mm']), 'arc_end_mm': float(cool['arc_end_mm']), 'mean_hu': float(cool['mean_hu'])},
        'coordinate_note': 'PCAT uses source-centerline arc. Plaque ensemble uses curved-series native coordinates; the tracks are intentionally not overlaid on a shared x-axis.'
    }
    (out / 'inflammation_plaque_summary.json').write_text(json.dumps(inflammation_summary, indent=2))

    figs = [_save_pcat_longitudinal(out, longitudinal), _save_plaque_fold_variation(out, fold_summary)]
    if len(sensitivity): figs.insert(1, _save_pcat_sensitivity(out, sensitivity))
    if rca_profile is not None and len(rca_profile): figs.append(_save_separate_tracks(out, longitudinal, rca_profile))

    cards = _metric_cards(canonical, plaque_conf)
    cards_html = '<div style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(
        f'<div style="border:1px solid #bbb;border-radius:8px;padding:12px;min-width:180px"><b>{k}</b><br><span style="font-size:24px">{v}</span></div>' for k, v in cards
    ) + '</div>'
    conf_html = plaque_conf.to_html(index=False) if plaque_conf is not None else '<p>Plaque confidence atlas not yet run; using the 5-fold batch summary only.</p>'
    figs_html = ''.join(f'<h2>{p.stem}</h2><img src="{p.name}" style="max-width:100%">' for p in figs)

    html = out / 'OPENPLAQUE_PCAT_PLAQUE_INTEGRATED_REPORT.html'
    html.write_text(
        '<html><head><meta charset="utf-8"><title>OpenPlaque Coronary Inflammation + Plaque</title></head><body style="font-family:Arial;max-width:1100px;margin:auto">'
        '<h1>OpenPlaque Coronary Inflammation + Plaque Report</h1>'
        '<p><b>Research use only.</b> OpenPlaque PCAT Attenuation is not Caristo FAI-Score, and this report does not provide a validated risk percentile or CaRi-Heart Risk.</p>'
        + cards_html + '<h2>Plaque fold uncertainty</h2>' + fold_stats_df.to_html(index=False)
        + '<h2>Plaque confidence volumes</h2>' + conf_html
        + '<p><b>Coordinate-system note:</b> source PCAT and curved-series plaque masks are shown together but are not falsely co-registered. '
          'A future validated transform is required before lesion-level spatial PCAT/plaque coupling can be claimed.</p>'
        + figs_html + '</body></html>'
    )
    zf = out / 'OPENPLAQUE_PCAT_PLAQUE_INTEGRATED_REPORT_BACK.zip'
    with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zf: z.write(p, p.name)
    return {'output_dir': str(out), 'report': str(html), 'zip': str(zf), 'inflammation_summary': inflammation_summary, 'plaque_fold_stats': fold_stats}
