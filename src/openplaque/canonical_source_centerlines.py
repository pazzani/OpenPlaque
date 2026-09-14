from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = '0593b453959f5a353d644267fbeef24b514ef4d7'
SECONDARY_QUANTITATIVE_CUTOFF_MM = 12.2


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _find_source_cache(drive_root: Path) -> Path:
    preferred = drive_root / 'Cache' / 'Secondary_3D_Vesselness_Topology_v1'
    if (preferred / 'series7_int16.npy').exists() and (preferred / 'series7_int16.json').exists():
        return preferred
    hits = []
    for p in drive_root.rglob('series7_int16.npy'):
        if (p.parent / 'series7_int16.json').exists():
            hits.append(p.parent)
    if not hits:
        raise FileNotFoundError('Could not locate source-CCTA cache (series7_int16.npy/json).')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _source_from_cache(cache_dir: Path):
    arr = np.load(cache_dir / 'series7_int16.npy', mmap_mode='r')
    meta = json.loads((cache_dir / 'series7_int16.json').read_text())
    img = sitk.GetImageFromArray(arr)
    sp_zyx = np.asarray(meta['spacing_zyx'], float)
    img.SetSpacing(tuple(sp_zyx[::-1]))
    positions = np.asarray(meta['positions_lps_mm'], float)
    img.SetOrigin(tuple(positions[0]))
    iop = np.asarray(meta['image_orientation_patient'], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([
        [row[0], col[0], slc[0]],
        [row[1], col[1], slc[1]],
        [row[2], col[2], slc[2]],
    ], float)
    img.SetDirection(tuple(direction.ravel()))
    return img, arr, meta


def _phys_to_zyx(img: sitk.Image, pts_lps):
    pts = np.asarray(pts_lps, float)
    origin = np.asarray(img.GetOrigin(), float)
    spacing = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - origin) @ np.linalg.inv(direction).T) / spacing
    return idx_xyz[:, ::-1]


def _zyx_to_phys(img: sitk.Image, zyx):
    zyx = np.asarray(zyx, float)
    idx_xyz = zyx[:, ::-1]
    origin = np.asarray(img.GetOrigin(), float)
    spacing = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return origin + (idx_xyz * spacing) @ direction.T


def _arc(pts):
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return np.zeros(0)
    if len(pts) == 1:
        return np.zeros(1)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _find_mask(drive_root: Path, legacy: bool) -> Path:
    roots = [drive_root / 'TotalSegmentator_Cardiovascular_Cache_v1', drive_root / 'GPU_Batch_Pipeline_v1', drive_root]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.nii*'):
            s = str(p).lower()
            if 'coronary' not in s or 'arter' not in s:
                continue
            if ('legacy' in s) != legacy:
                continue
            if any(x in s for x in ('intersection', 'union', 'disagreement', 'consensus')):
                continue
            score = 0
            if 'totalsegmentator_cardiovascular_cache_v1' in s:
                score += 10
            if 'coronary_arteries' in s:
                score += 5
            hits.append((score, len(str(p)), p))
    if not hits:
        raise FileNotFoundError(f"Could not locate {'legacy' if legacy else 'current'} coronary mask")
    hits.sort(key=lambda x: (-x[0], x[1], str(x[2])))
    return hits[0][2]


def _mask_tree(mask_img: sitk.Image):
    arr = sitk.GetArrayFromImage(mask_img) > 0
    zyx = np.argwhere(arr)
    if len(zyx) == 0:
        raise RuntimeError('Coronary mask is empty.')
    xyz = _zyx_to_phys(mask_img, zyx)
    return arr, cKDTree(xyz)


def _inside(mask_img, mask_arr, pts):
    zyx = np.rint(_phys_to_zyx(mask_img, pts)).astype(int)
    valid = np.ones(len(zyx), bool)
    for k, s in enumerate(mask_arr.shape):
        valid &= (zyx[:, k] >= 0) & (zyx[:, k] < s)
    inside = np.zeros(len(zyx), bool)
    q = zyx[valid]
    inside[valid] = mask_arr[q[:, 0], q[:, 1], q[:, 2]]
    return inside, valid


def _sample_hu(source_img, source_arr, pts):
    zyx = _phys_to_zyx(source_img, pts)
    shp = np.asarray(source_arr.shape)
    valid = np.ones(len(zyx), bool)
    for k in range(3):
        valid &= (zyx[:, k] >= 0) & (zyx[:, k] <= shp[k] - 1)
    hu = map_coordinates(source_arr, zyx.T, order=1, mode='constant', cval=-1024.0).astype(float)
    return hu, valid


def _resolve_saved_path(path_text: str, drive_root: Path) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    marker = '/OpenPlaque/'
    s = str(path_text).replace('\\', '/')
    if marker in s:
        rel = s.split(marker, 1)[1]
        q = drive_root / rel
        if q.exists():
            return q
    raise FileNotFoundError(f'Candidate source file no longer exists: {path_text}')


def _find_reconciliation_summary(drive_root: Path) -> Path:
    preferred = drive_root / 'Left_Coronary_Coordinate_Reconciliation_v1' / 'coordinate_reconciliation_summary.csv'
    if preferred.exists():
        return preferred
    hits = list(drive_root.rglob('coordinate_reconciliation_summary.csv'))
    if not hits:
        raise FileNotFoundError('Run the left-coronary reconciliation notebook first.')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _find_original_rca(drive_root: Path) -> Path:
    hits = [p for p in drive_root.rglob('RCA_source_centerline.csv') if 'source_volume_coronary_centerlines' in str(p).lower()]
    if not hits:
        hits = list(drive_root.rglob('RCA_source_centerline.csv'))
    if not hits:
        raise FileNotFoundError('Could not locate validated RCA_source_centerline.csv')
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _candidate_from_summary(summary: pd.DataFrame, label: str, lo: float, hi: float) -> pd.Series:
    q = summary[
        summary['label'].eq(label)
        & summary['coord_variant'].eq('recomputed_from_zyx')
        & summary['length_mm'].between(lo, hi, inclusive='both')
        & (summary['source_hu_gt200_fraction'] >= 0.90)
        & (summary['current_median_distance_mm'] <= 1.0)
        & (summary['legacy_median_distance_mm'] <= 1.0)
    ].copy()
    if not len(q):
        raise RuntimeError(f'No {label} candidate passed canonical selection gates: length {lo}-{hi} mm, HU>200 >=90%, both mask medians <=1 mm.')
    q = q.sort_values(
        ['combined_reconciliation_score', 'current_median_distance_mm', 'legacy_median_distance_mm', 'source_hu_median'],
        ascending=[False, True, True, False],
    )
    return q.iloc[0]


def _canonicalize_csv(path: Path, source_img: sitk.Image, force_recompute_lps=True):
    df = pd.read_csv(path)
    zyx_cols = None
    for cols in [('z', 'y', 'x'), ('voxel_z', 'voxel_y', 'voxel_x'), ('z_vox', 'y_vox', 'x_vox')]:
        if set(cols).issubset(df.columns):
            zyx_cols = cols
            break
    lps_cols = None
    for cols in [('lps_x_mm', 'lps_y_mm', 'lps_z_mm'), ('x_lps_mm', 'y_lps_mm', 'z_lps_mm'), ('x_lps', 'y_lps', 'z_lps')]:
        if set(cols).issubset(df.columns):
            lps_cols = cols
            break
    if zyx_cols is not None and force_recompute_lps:
        zyx = df[list(zyx_cols)].to_numpy(float)
        pts = _zyx_to_phys(source_img, zyx)
    elif lps_cols is not None:
        pts = df[list(lps_cols)].to_numpy(float)
        zyx = _phys_to_zyx(source_img, pts)
    elif zyx_cols is not None:
        zyx = df[list(zyx_cols)].to_numpy(float)
        pts = _zyx_to_phys(source_img, zyx)
    else:
        raise ValueError(f'No usable source coordinates in {path}')
    arc = _arc(pts)
    out = pd.DataFrame({
        'arc_mm': arc,
        'z': zyx[:, 0], 'y': zyx[:, 1], 'x': zyx[:, 2],
        'lps_x_mm': pts[:, 0], 'lps_y_mm': pts[:, 1], 'lps_z_mm': pts[:, 2],
    })
    return out


def _truncate_at_arc(df: pd.DataFrame, cutoff_mm: float) -> pd.DataFrame:
    if float(df['arc_mm'].iloc[-1]) < cutoff_mm:
        raise ValueError('Trajectory is shorter than requested cutoff.')
    left = df[df['arc_mm'] <= cutoff_mm].copy()
    if np.isclose(left['arc_mm'].iloc[-1], cutoff_mm, atol=1e-6):
        return left
    j = int(np.searchsorted(df['arc_mm'].to_numpy(float), cutoff_mm))
    a = df.iloc[j - 1]
    b = df.iloc[j]
    t = (cutoff_mm - float(a.arc_mm)) / (float(b.arc_mm) - float(a.arc_mm))
    row = {'arc_mm': cutoff_mm}
    for c in ['z', 'y', 'x', 'lps_x_mm', 'lps_y_mm', 'lps_z_mm']:
        row[c] = float(a[c] + t * (b[c] - a[c]))
    return pd.concat([left, pd.DataFrame([row])], ignore_index=True)


def _validate(name, df, source_img, source_arr, masks):
    pts = df[['lps_x_mm', 'lps_y_mm', 'lps_z_mm']].to_numpy(float)
    hu, src_valid = _sample_hu(source_img, source_arr, pts)
    profile = df.copy()
    profile['source_hu'] = hu
    profile['in_source_image'] = src_valid
    metrics = {
        'vessel': name,
        'points': int(len(df)),
        'length_mm': float(df['arc_mm'].iloc[-1]),
        'source_hu_median': float(np.median(hu[src_valid])) if src_valid.any() else np.nan,
        'source_hu_gt200_fraction': float(np.mean(hu[src_valid] > 200)) if src_valid.any() else 0.0,
        'source_in_bounds_fraction': float(np.mean(src_valid)),
    }
    for mname, m in masks.items():
        inside, valid = _inside(m['img'], m['arr'], pts)
        dist = m['tree'].query(pts, k=1)[0]
        profile[f'{mname}_inside'] = inside
        profile[f'{mname}_distance_mm'] = dist
        metrics[f'{mname}_inside_fraction'] = float(np.mean(inside))
        metrics[f'{mname}_median_distance_mm'] = float(np.median(dist))
        metrics[f'{mname}_p90_distance_mm'] = float(np.quantile(dist, 0.90))
        metrics[f'{mname}_max_distance_mm'] = float(np.max(dist))
        metrics[f'{mname}_in_bounds_fraction'] = float(np.mean(valid))
    metrics['passes_canonical_gate'] = bool(
        metrics['source_hu_gt200_fraction'] >= 0.90
        and metrics['source_in_bounds_fraction'] >= 0.99
        and metrics['current_median_distance_mm'] <= 1.0
        and metrics['legacy_median_distance_mm'] <= 1.0
        and metrics['current_inside_fraction'] >= 0.80
        and metrics['legacy_inside_fraction'] >= 0.80
    )
    return metrics, profile


def _plot_validation(out: Path, profiles: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for name, df in profiles.items():
        axes[0].plot(df['arc_mm'], df['current_distance_mm'], label=f'{name} current')
        axes[0].plot(df['arc_mm'], df['legacy_distance_mm'], linestyle='--', label=f'{name} legacy')
    axes[0].axhline(1.0, linewidth=1, linestyle=':')
    axes[0].set_ylabel('Distance to coronary mask (mm)')
    axes[0].set_title('Canonical source-centerline validation')
    axes[0].legend(ncol=2, fontsize=8)
    for name, df in profiles.items():
        axes[1].plot(df['arc_mm'], df['source_hu'], label=name)
    axes[1].axhline(200, linewidth=1, linestyle=':')
    axes[1].set_xlabel('Arc length (mm)')
    axes[1].set_ylabel('Source CCTA HU')
    axes[1].legend()
    fig.tight_layout()
    p = out / 'canonical_centerline_validation.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'Canonical_Source_Coronary_Centerlines_v1')
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'run_state.json', {'status': 'STARTED', 'baseline': BASELINE})

    source_cache = _find_source_cache(drive_root)
    source_img, source_arr, source_meta = _source_from_cache(source_cache)
    current_path = _find_mask(drive_root, legacy=False)
    legacy_path = _find_mask(drive_root, legacy=True)
    current_img = sitk.ReadImage(str(current_path))
    legacy_img = sitk.ReadImage(str(legacy_path))
    current_arr, current_tree = _mask_tree(current_img)
    legacy_arr, legacy_tree = _mask_tree(legacy_img)
    masks = {
        'current': {'img': current_img, 'arr': current_arr, 'tree': current_tree},
        'legacy': {'img': legacy_img, 'arr': legacy_arr, 'tree': legacy_tree},
    }

    recon_path = _find_reconciliation_summary(drive_root)
    recon = pd.read_csv(recon_path)

    rca_src = _find_original_rca(drive_root)
    lad_row = _candidate_from_summary(recon, 'LAD', 56.5, 58.5)
    sec_row = _candidate_from_summary(recon, 'SECONDARY', 13.0, 15.0)
    lad_src = _resolve_saved_path(str(lad_row['path']), drive_root)
    sec_src = _resolve_saved_path(str(sec_row['path']), drive_root)

    rca = _canonicalize_csv(rca_src, source_img, force_recompute_lps=True)
    lad = _canonicalize_csv(lad_src, source_img, force_recompute_lps=True)
    secondary = _canonicalize_csv(sec_src, source_img, force_recompute_lps=True)
    secondary_quant = _truncate_at_arc(secondary, SECONDARY_QUANTITATIVE_CUTOFF_MM)

    files = {
        'RCA': (rca, out / 'RCA_canonical_source_centerline.csv'),
        'LAD': (lad, out / 'LAD_canonical_source_centerline.csv'),
        'SECONDARY_TRAJECTORY': (secondary, out / 'SECONDARY_canonical_source_trajectory.csv'),
        'SECONDARY_COMPACT_0_12p2': (secondary_quant, out / 'SECONDARY_quantitative_compact_lumen_0_12p2mm.csv'),
    }
    for _, (df, p) in files.items():
        df.to_csv(p, index=False)

    validation_rows = []
    profiles = {}
    for name in ['RCA', 'LAD', 'SECONDARY_TRAJECTORY']:
        df, _ = files[name]
        metrics, prof = _validate(name, df, source_img, source_arr, masks)
        validation_rows.append(metrics)
        profiles[name] = prof
        prof.to_csv(out / f'{name}_pointwise_validation.csv', index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(out / 'canonical_validation_summary.csv', index=False)
    if not validation['passes_canonical_gate'].all():
        _write_json(out / 'run_state.json', {'status': 'REJECTED', 'reason': 'One or more canonical candidates failed validation gates.'})
        raise RuntimeError('Canonical bundle rejected: one or more candidates failed validation gates.')

    fig = _plot_validation(out, profiles)

    manifest = {
        'schema': 'openplaque-canonical-source-coronary-centerlines-v1',
        'baseline_commit': BASELINE,
        'source_cache': str(source_cache),
        'reconciliation_summary': str(recon_path),
        'totalsegmentator_current_mask': str(current_path),
        'totalsegmentator_legacy_mask': str(legacy_path),
        'canonical': {
            'RCA': {
                'file': 'RCA_canonical_source_centerline.csv',
                'source_file': str(rca_src),
                'source_sha256': _sha256(rca_src),
                'status': 'validated coronary centerline; canonical RCA source path',
                'length_mm': float(rca['arc_mm'].iloc[-1]),
            },
            'LAD': {
                'file': 'LAD_canonical_source_centerline.csv',
                'source_file': str(lad_src),
                'source_sha256': _sha256(lad_src),
                'selection_rule': 'recomputed_from_zyx; 56.5-58.5 mm; HU>200 >=90%; both TotalSegmentator median distances <=1 mm; highest reconciliation score',
                'status': 'validated ~57 mm LAD; replaces stale Source_Volume_Coronary_Centerlines/LAD_source_centerline.csv',
                'length_mm': float(lad['arc_mm'].iloc[-1]),
            },
            'SECONDARY': {
                'trajectory_file': 'SECONDARY_canonical_source_trajectory.csv',
                'quantitative_compact_file': 'SECONDARY_quantitative_compact_lumen_0_12p2mm.csv',
                'source_file': str(sec_src),
                'source_sha256': _sha256(sec_src),
                'selection_rule': 'recomputed_from_zyx; 13-15 mm; HU>200 >=90%; both TotalSegmentator median distances <=1 mm; highest reconciliation score',
                'status': 'coronary-like geometric trajectory; vessel identity intentionally unresolved',
                'trajectory_length_mm': float(secondary['arc_mm'].iloc[-1]),
                'quantitative_compact_lumen_cutoff_mm': SECONDARY_QUANTITATIVE_CUTOFF_MM,
                'quantitative_rule': 'Use only 0-12.2 mm for compact-lumen quantitative analyses unless newer source-resolution evidence supersedes this cutoff.',
            },
        },
        'deprecated_inputs': {
            'LAD_source_centerline.csv': 'Do not use as canonical LAD; stale/wrong ~117 mm left-coronary file selected by the first geometry integration.',
            'LCX_source_centerline.csv': 'Do not use as canonical secondary branch; stale/wrong ~113 mm file. Secondary vessel identity remains unresolved.',
        },
    }
    _write_json(out / 'canonical_manifest.json', manifest)

    html = out / 'OPENPLAQUE_CANONICAL_SOURCE_CENTERLINE_BUNDLE_REPORT.html'
    html.write_text(
        '<html><head><meta charset="utf-8"><title>OpenPlaque Canonical Source Centerlines</title></head>'
        '<body style="font-family:Arial;max-width:1150px;margin:auto">'
        '<h1>OpenPlaque — Canonical Source-CCTA Coronary Centerline Bundle</h1>'
        '<p><b>Research use only.</b> This bundle supersedes stale left-coronary source-centerline files for downstream OpenPlaque experiments.</p>'
        '<h2>Validation</h2>' + validation.to_html(index=False) +
        '<h2>Canonical manifest</h2><pre>' + json.dumps(manifest, indent=2) + '</pre>' +
        f'<h2>Validation plot</h2><img src="{fig.name}" style="max-width:100%">' +
        '</body></html>'
    )
    zf = out / 'OPENPLAQUE_CANONICAL_SOURCE_CENTERLINE_BUNDLE_REPORT_BACK.zip'
    with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p != zf:
                z.write(p, p.relative_to(out))
    state = {
        'status': 'COMPLETE',
        'output_dir': str(out),
        'manifest': str(out / 'canonical_manifest.json'),
        'report': str(html),
        'zip': str(zf),
    }
    _write_json(out / 'run_state.json', state)
    return state
