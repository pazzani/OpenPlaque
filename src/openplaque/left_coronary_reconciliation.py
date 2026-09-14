from __future__ import annotations

import hashlib
import json
import traceback
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


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _find_mask(drive_root: Path, legacy: bool) -> Path:
    roots = [
        drive_root / 'TotalSegmentator_Cardiovascular_Cache_v1',
        drive_root / 'GPU_Batch_Pipeline_v1',
        drive_root,
    ]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.nii*'):
            s = str(p).lower()
            if 'coronary' not in s or 'arter' not in s:
                continue
            is_legacy = 'legacy' in s
            if is_legacy != legacy:
                continue
            if any(x in s for x in ('intersection', 'union', 'disagreement', 'consensus')):
                continue
            score = 0
            if 'totalsegmentator_cardiovascular_cache_v1' in s:
                score += 10
            if 'coronary_arteries' in s:
                score += 5
            if p.name.lower() in ('coronary_arteries.nii.gz', 'coronary_arteries.nii'):
                score += 5
            hits.append((score, len(str(p)), p))
    if not hits:
        which = 'legacy' if legacy else 'current'
        raise FileNotFoundError(f'Could not locate {which} TotalSegmentator coronary mask')
    hits.sort(key=lambda x: (-x[0], x[1], str(x[2])))
    return hits[0][2]


def _source_from_cache(cache_dir: Path):
    arr_path = cache_dir / 'series7_int16.npy'
    meta_path = cache_dir / 'series7_int16.json'
    if not arr_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing source cache under {cache_dir}')
    arr = np.load(arr_path, mmap_mode='r')
    meta = json.loads(meta_path.read_text())
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


def _sample_hu(source_img, source_arr, pts):
    zyx = _phys_to_zyx(source_img, pts)
    shape = np.asarray(source_arr.shape, int)
    inside = np.ones(len(zyx), bool)
    for k in range(3):
        inside &= (zyx[:, k] >= 0) & (zyx[:, k] <= shape[k] - 1)
    hu = map_coordinates(source_arr, zyx.T, order=1, mode='constant', cval=-1024.0)
    return hu.astype(float), inside


def _mask_points_and_tree(mask_img, max_tree_points=250000):
    arr = sitk.GetArrayFromImage(mask_img) > 0
    zyx = np.argwhere(arr)
    if len(zyx) == 0:
        raise RuntimeError('Coronary mask is empty.')
    if len(zyx) > max_tree_points:
        sel = np.linspace(0, len(zyx) - 1, max_tree_points).astype(int)
        zyx_tree = zyx[sel]
    else:
        zyx_tree = zyx
    xyz = _zyx_to_phys(mask_img, zyx_tree)
    return arr, cKDTree(xyz), len(zyx)


def _same_geometry(a: sitk.Image, b: sitk.Image, atol=1e-5):
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def _resample_mask_like(moving: sitk.Image, reference: sitk.Image):
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID(),
    )


def _inside_mask(mask_img, mask_arr, pts):
    zyx = np.rint(_phys_to_zyx(mask_img, pts)).astype(int)
    valid = np.ones(len(zyx), bool)
    for k, s in enumerate(mask_arr.shape):
        valid &= (zyx[:, k] >= 0) & (zyx[:, k] < s)
    inside = np.zeros(len(zyx), bool)
    q = zyx[valid]
    inside[valid] = mask_arr[q[:, 0], q[:, 1], q[:, 2]]
    return inside, valid


def _curve_length(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _arc_from_pts(pts):
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return np.zeros(0)
    if len(pts) == 1:
        return np.zeros(1)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def _label_path(path: Path):
    s = str(path).lower().replace('\\', '/')
    name = path.name.lower()
    if 'rca' in name or '/rca' in s:
        return 'RCA'
    if 'lad' in name or '/lad' in s:
        return 'LAD'
    if 'secondary' in s or 'lcx' in name or '/lcx' in s:
        return 'SECONDARY'
    if 'left_main' in s or 'left-main' in s or name.startswith('lm_') or '_lm_' in name:
        return 'LM'
    if 'left' in s and 'centerline' in s:
        return 'LEFT_OTHER'
    return None


def _extract_coords(df: pd.DataFrame, source_img):
    cols = set(df.columns)
    stored = None
    stored_cols = None
    for trip in [
        ('lps_x_mm', 'lps_y_mm', 'lps_z_mm'),
        ('x_lps_mm', 'y_lps_mm', 'z_lps_mm'),
        ('x_lps', 'y_lps', 'z_lps'),
    ]:
        if set(trip).issubset(cols):
            stored = df[list(trip)].to_numpy(float)
            stored_cols = trip
            break
    voxel = None
    voxel_cols = None
    for trip in [
        ('z', 'y', 'x'),
        ('voxel_z', 'voxel_y', 'voxel_x'),
        ('z_vox', 'y_vox', 'x_vox'),
    ]:
        if set(trip).issubset(cols):
            voxel = df[list(trip)].to_numpy(float)
            voxel_cols = trip
            break
    recomputed = _zyx_to_phys(source_img, voxel) if voxel is not None else None
    return stored, stored_cols, voxel, voxel_cols, recomputed


def _discover_candidates(drive_root: Path, source_img, max_files=500, exclude_root=None):
    rows = []
    seen_geom = {}
    csvs = []
    exclude_root = Path(exclude_root).resolve() if exclude_root is not None else None
    for p in drive_root.rglob('*.csv'):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if exclude_root is not None:
            try:
                if rp == exclude_root or exclude_root in rp.parents:
                    continue
            except Exception:
                pass
        s = str(p).lower()
        if 'left_coronary_coordinate_reconciliation_v1' in s:
            continue
        label = _label_path(p)
        if label is None:
            continue
        if not any(k in s for k in ('centerline', 'track', 'backbone', 'path', 'lad', 'lcx', 'secondary', 'rca')):
            continue
        csvs.append(p)
    csvs.sort(key=lambda p: (len(str(p)), str(p)))
    for p in csvs[:max_files]:
        try:
            df = pd.read_csv(p)
            if len(df) < 3:
                continue
            stored, stored_cols, voxel, voxel_cols, recomputed = _extract_coords(df, source_img)
            if stored is None and recomputed is None:
                continue
            label = _label_path(p)
            variants = []
            if stored is not None and np.isfinite(stored).all():
                variants.append(('stored_lps', stored))
                ras = stored.copy()
                ras[:, 0] *= -1
                ras[:, 1] *= -1
                variants.append(('ras_to_lps_hypothesis', ras))
            if recomputed is not None and np.isfinite(recomputed).all():
                variants.append(('recomputed_from_zyx', recomputed))
            coord_delta_med = np.nan
            coord_delta_p95 = np.nan
            if stored is not None and recomputed is not None and len(stored) == len(recomputed):
                d = np.linalg.norm(stored - recomputed, axis=1)
                coord_delta_med = float(np.median(d))
                coord_delta_p95 = float(np.quantile(d, .95))
            is_source_volume = 'source_volume_coronary_centerlines' in str(p).lower()
            for variant_name, pts in variants:
                digest = hashlib.sha1(np.round(pts, 2).tobytes()).hexdigest()[:16]
                duplicate_of = seen_geom.get((label, variant_name, digest))
                if duplicate_of is None:
                    seen_geom[(label, variant_name, digest)] = str(p)
                rows.append({
                    'candidate_id': f'{label}:{variant_name}:{digest}',
                    'label': label,
                    'path': str(p),
                    'file_name': p.name,
                    'coord_variant': variant_name,
                    'points': pts,
                    'n_points': int(len(pts)),
                    'length_mm': _curve_length(pts),
                    'stored_cols': stored_cols,
                    'voxel_cols': voxel_cols,
                    'coord_delta_stored_vs_recomputed_median_mm': coord_delta_med,
                    'coord_delta_stored_vs_recomputed_p95_mm': coord_delta_p95,
                    'is_source_volume_file': bool(is_source_volume),
                    'duplicate_of': duplicate_of,
                })
        except Exception:
            continue
    return rows


def _evaluate_candidate(rec, source_img, source_arr, masks):
    pts = rec['points']
    hu, in_image = _sample_hu(source_img, source_arr, pts)
    out = {k: v for k, v in rec.items() if k != 'points'}
    out.update({
        'in_image_fraction': float(np.mean(in_image)),
        'source_hu_median': float(np.median(hu[in_image])) if in_image.any() else np.nan,
        'source_hu_p10': float(np.quantile(hu[in_image], .10)) if in_image.any() else np.nan,
        'source_hu_p90': float(np.quantile(hu[in_image], .90)) if in_image.any() else np.nan,
        'source_hu_gt150_fraction': float(np.mean(hu[in_image] > 150)) if in_image.any() else 0.0,
        'source_hu_gt200_fraction': float(np.mean(hu[in_image] > 200)) if in_image.any() else 0.0,
        'source_hu_gt300_fraction': float(np.mean(hu[in_image] > 300)) if in_image.any() else 0.0,
    })
    for name, m in masks.items():
        inside, valid = _inside_mask(m['img'], m['arr'], pts)
        dist = m['tree'].query(pts, k=1)[0]
        out[f'{name}_inside_fraction'] = float(np.mean(inside))
        out[f'{name}_in_bounds_fraction'] = float(np.mean(valid))
        out[f'{name}_median_distance_mm'] = float(np.median(dist))
        out[f'{name}_p90_distance_mm'] = float(np.quantile(dist, .90))
        out[f'{name}_max_distance_mm'] = float(np.max(dist))
    out['lumen_like_source_support'] = bool(
        out['in_image_fraction'] >= .90 and
        out['source_hu_median'] >= 200 and
        out['source_hu_gt200_fraction'] >= .60
    )
    md = out.get('union_median_distance_mm', np.inf)
    out['mask_alignment_score'] = float(max(0.0, 1.0 - md / 5.0))
    out['source_lumen_score'] = float(
        0.65 * out['source_hu_gt200_fraction'] +
        0.20 * out['source_hu_gt150_fraction'] +
        0.15 * out['in_image_fraction']
    )
    out['combined_reconciliation_score'] = float(
        0.55 * out['mask_alignment_score'] + 0.45 * out['source_lumen_score']
    )
    return out, hu, in_image


def _prefix_metrics(rec, source_img, source_arr, masks, cutoffs=(12.2, 20.0, 40.0, 60.0)):
    pts = rec['points']
    arc = _arc_from_pts(pts)
    rows = []
    for cutoff in cutoffs:
        use = arc <= min(float(cutoff), float(arc[-1]))
        if use.sum() < 3:
            continue
        sub = dict(rec)
        sub['points'] = pts[use]
        sub['candidate_id'] = rec['candidate_id'] + f':prefix_{cutoff:g}'
        ev, _, _ = _evaluate_candidate(sub, source_img, source_arr, masks)
        ev['prefix_cutoff_mm'] = float(cutoff)
        ev['actual_prefix_length_mm'] = float(arc[use][-1])
        rows.append(ev)
    return rows


def _per_point_profile(rec, masks, source_img, source_arr):
    pts = rec['points']
    arc = _arc_from_pts(pts)
    hu, in_image = _sample_hu(source_img, source_arr, pts)
    df = pd.DataFrame({
        'arc_mm': arc,
        'lps_x_mm': pts[:, 0],
        'lps_y_mm': pts[:, 1],
        'lps_z_mm': pts[:, 2],
        'source_hu': hu,
        'in_source_image': in_image,
    })
    for name, m in masks.items():
        inside, valid = _inside_mask(m['img'], m['arr'], pts)
        dist = m['tree'].query(pts, k=1)[0]
        df[f'{name}_inside'] = inside
        df[f'{name}_distance_mm'] = dist
        df[f'{name}_in_bounds'] = valid
    return df


def _select_best(summary: pd.DataFrame, label: str, source_only=False):
    q = summary[summary['label'].eq(label)].copy()
    q = q[q['coord_variant'].isin(['stored_lps', 'recomputed_from_zyx'])]
    if source_only:
        q = q[q['is_source_volume_file']]
    if not len(q):
        return None
    q = q.sort_values(
        ['combined_reconciliation_score', 'source_lumen_score', 'union_median_distance_mm'],
        ascending=[False, False, True],
    )
    return q.iloc[0]


def _classification(summary: pd.DataFrame):
    result = {
        'rca_positive_control': 'NOT_EVALUATED',
        'lad_conclusion': 'UNRESOLVED',
        'secondary_conclusion': 'UNRESOLVED',
        'primary_conclusion': 'UNRESOLVED',
        'notes': [],
    }
    rca = _select_best(summary, 'RCA', source_only=True)
    if rca is None:
        rca = _select_best(summary, 'RCA')
    if rca is not None:
        rca_ok = bool(rca['intersection_median_distance_mm'] < 1.0 and rca['intersection_inside_fraction'] > .75)
        result['rca_positive_control'] = 'PASS' if rca_ok else 'FAIL'
    else:
        rca_ok = False
        result['notes'].append('No RCA positive-control centerline discovered.')

    for label, key in [('LAD', 'lad_conclusion'), ('SECONDARY', 'secondary_conclusion')]:
        q = summary[(summary['label'] == label) & summary['coord_variant'].isin(['stored_lps', 'recomputed_from_zyx'])].copy()
        if not len(q):
            result[key] = 'NO_CANDIDATE_DISCOVERED'
            continue
        stored = q[q['coord_variant'].eq('stored_lps')]
        recomp = q[q['coord_variant'].eq('recomputed_from_zyx')]
        coord_fix = False
        if len(stored) and len(recomp):
            s = stored.sort_values('union_median_distance_mm').iloc[0]
            r = recomp.sort_values('union_median_distance_mm').iloc[0]
            coord_fix = bool(
                s['union_median_distance_mm'] > 8.0
                and r['union_median_distance_mm'] < 2.0
                and r['union_median_distance_mm'] < .25 * s['union_median_distance_mm']
            )
        if coord_fix:
            result[key] = 'COORDINATE_RECONSTRUCTION_ERROR_IDENTIFIED'
            continue
        aligned = q[(q['union_median_distance_mm'] < 2.0) & (q['source_hu_gt200_fraction'] > .40)]
        if len(aligned):
            source_q = q[q['is_source_volume_file'] & q['coord_variant'].eq('stored_lps')]
            if len(source_q) and source_q['union_median_distance_mm'].min() > 8.0:
                result[key] = 'WRONG_OR_OUTDATED_CENTERLINE_FILE_SELECTED'
            else:
                result[key] = 'RECONCILED_WITH_TOTALSEG'
            continue
        lumen = q.sort_values('source_lumen_score', ascending=False).iloc[0]
        if rca_ok and bool(lumen['lumen_like_source_support']) and lumen['union_median_distance_mm'] > 5.0:
            result[key] = 'SOURCE_LUMEN_SUPPORTED_BUT_TOTALSEG_UNSUPPORTED'
        elif lumen['source_lumen_score'] >= .55 and lumen['union_median_distance_mm'] > 5.0:
            result[key] = 'POSSIBLE_TOTALSEG_LEFT_CORONARY_UNDERSEGMENTATION'
        else:
            result[key] = 'CENTERLINE_IDENTITY_OR_GEOMETRY_NOT_RECONCILED'

    vals = {result['lad_conclusion'], result['secondary_conclusion']}
    if any('COORDINATE_RECONSTRUCTION_ERROR' in x for x in vals):
        result['primary_conclusion'] = 'LEFT_CORONARY_COORDINATE_FRAME_ERROR'
    elif any('WRONG_OR_OUTDATED' in x for x in vals):
        result['primary_conclusion'] = 'WRONG_OR_OUTDATED_LEFT_CENTERLINE_SELECTED'
    elif any('SOURCE_LUMEN_SUPPORTED_BUT_TOTALSEG_UNSUPPORTED' in x for x in vals):
        result['primary_conclusion'] = 'TOTALSEG_LEFT_CORONARY_UNDERSEGMENTATION_LIKELY'
    elif all(x == 'RECONCILED_WITH_TOTALSEG' for x in vals):
        result['primary_conclusion'] = 'LEFT_CORONARY_RECONCILED'
    else:
        result['primary_conclusion'] = 'LEFT_CORONARY_RECONCILIATION_UNRESOLVED'
    return result


def _plot_rankings(out: Path, summary: pd.DataFrame):
    figs = []
    q = summary[(summary['label'].isin(['LAD', 'SECONDARY'])) & summary['coord_variant'].isin(['stored_lps', 'recomputed_from_zyx'])].copy()
    q = q.sort_values('combined_reconciliation_score', ascending=False).head(18)
    if len(q):
        labels = [f"{r.label}:{r.coord_variant}\n{Path(r.path).name}" for _, r in q.iterrows()]
        fig, ax = plt.subplots(figsize=(12, max(5, .48 * len(q))))
        y = np.arange(len(q))
        ax.barh(y, q['union_median_distance_mm'].clip(upper=70))
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel('Median distance to TotalSegmentator union (mm; clipped at 70)')
        ax.set_title('Left-coronary candidate alignment')
        fig.tight_layout()
        p = out / '01_left_candidate_union_distance.png'
        fig.savefig(p, dpi=170); plt.close(fig); figs.append(p)

        fig, ax = plt.subplots(figsize=(12, max(5, .48 * len(q))))
        ax.barh(y, q['source_hu_median'])
        ax.axvline(200, linestyle='--', linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel('Median source-CCTA HU along candidate')
        ax.set_title('Source-lumen support for left-coronary candidates')
        fig.tight_layout()
        p = out / '02_left_candidate_source_hu.png'
        fig.savefig(p, dpi=170); plt.close(fig); figs.append(p)
    return figs


def _plot_profiles(out: Path, profiles):
    figs = []
    for label, df in profiles.items():
        if df is None or not len(df):
            continue
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df['arc_mm'], df['current_distance_mm'], label='current')
        ax.plot(df['arc_mm'], df['legacy_distance_mm'], label='legacy')
        ax.plot(df['arc_mm'], df['union_distance_mm'], label='union', linewidth=2)
        ax.axhline(2.0, linestyle='--', linewidth=1)
        ax.set_xlabel('Centerline arc length (mm)')
        ax.set_ylabel('Distance to mask (mm)')
        ax.set_title(f'{label}: distance to independent coronary masks')
        ax.legend(); fig.tight_layout()
        p = out / f'03_{label.lower()}_mask_distance_along_arc.png'
        fig.savefig(p, dpi=170); plt.close(fig); figs.append(p)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df['arc_mm'], df['source_hu'])
        ax.axhline(200, linestyle='--', linewidth=1)
        ax.set_xlabel('Centerline arc length (mm)')
        ax.set_ylabel('Source-CCTA HU')
        ax.set_title(f'{label}: source-CCTA intensity along candidate')
        fig.tight_layout()
        p = out / f'04_{label.lower()}_source_hu_along_arc.png'
        fig.savefig(p, dpi=170); plt.close(fig); figs.append(p)
    return figs


def _plot_projection(out: Path, masks, chosen):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    projections = [(0, 1, 'LPS X', 'LPS Y'), (0, 2, 'LPS X', 'LPS Z'), (1, 2, 'LPS Y', 'LPS Z')]
    mask_xyz = {}
    for name in ('current', 'legacy'):
        zyx = np.argwhere(masks[name]['arr'])
        if len(zyx) > 16000:
            sel = np.linspace(0, len(zyx)-1, 16000).astype(int)
            zyx = zyx[sel]
        mask_xyz[name] = _zyx_to_phys(masks[name]['img'], zyx)
    for ax, (a, b, la, lb) in zip(axes, projections):
        for name in ('current', 'legacy'):
            xyz = mask_xyz[name]
            ax.scatter(xyz[:, a], xyz[:, b], s=.3, alpha=.18, label=name)
        for label, pts in chosen.items():
            if pts is not None and len(pts):
                ax.plot(pts[:, a], pts[:, b], linewidth=2, label=label)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect('equal', adjustable='datalim')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(5, len(labels)))
    fig.suptitle('Source physical LPS: TotalSegmentator masks and selected centerlines')
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = out / '05_source_lps_projection_reconciliation.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None,
        source_cache=None, max_candidate_files=500):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'Left_Coronary_Coordinate_Reconciliation_v1')
    out.mkdir(parents=True, exist_ok=True)
    state = {'status': 'STARTED', 'baseline': BASELINE}
    _write_json(out / 'run_state.json', state)
    try:
        source_cache = Path(source_cache or drive_root / 'Cache/Secondary_3D_Vesselness_Topology_v1')
        source_img, source_arr, source_meta = _source_from_cache(source_cache)
        current_path = _find_mask(drive_root, legacy=False)
        legacy_path = _find_mask(drive_root, legacy=True)
        current_img = sitk.ReadImage(str(current_path))
        legacy_img = sitk.ReadImage(str(legacy_path))
        geometry_note = 'native_geometry_match'
        if not _same_geometry(current_img, legacy_img):
            legacy_img = _resample_mask_like(legacy_img, current_img)
            geometry_note = 'legacy_resampled_to_current_geometry_nearest_neighbor'
        current_arr, current_tree, current_n = _mask_points_and_tree(current_img)
        legacy_arr, legacy_tree, legacy_n = _mask_points_and_tree(legacy_img)
        union_arr = current_arr | legacy_arr
        inter_arr = current_arr & legacy_arr
        union_img = sitk.GetImageFromArray(union_arr.astype(np.uint8)); union_img.CopyInformation(current_img)
        inter_img = sitk.GetImageFromArray(inter_arr.astype(np.uint8)); inter_img.CopyInformation(current_img)
        _, union_tree, union_n = _mask_points_and_tree(union_img)
        _, inter_tree, inter_n = _mask_points_and_tree(inter_img)
        masks = {
            'current': {'img': current_img, 'arr': current_arr, 'tree': current_tree},
            'legacy': {'img': legacy_img, 'arr': legacy_arr, 'tree': legacy_tree},
            'union': {'img': union_img, 'arr': union_arr, 'tree': union_tree},
            'intersection': {'img': inter_img, 'arr': inter_arr, 'tree': inter_tree},
        }
        mask_meta = {
            'current_path': str(current_path), 'legacy_path': str(legacy_path),
            'current_positive_voxels': int(current_n), 'legacy_positive_voxels': int(legacy_n),
            'union_positive_voxels': int(union_n), 'intersection_positive_voxels': int(inter_n),
            'current_legacy_geometry_handling': geometry_note,
            'source_size_xyz': list(source_img.GetSize()), 'source_spacing_xyz': list(source_img.GetSpacing()),
            'source_origin_lps': list(source_img.GetOrigin()), 'source_direction': list(source_img.GetDirection()),
        }
        _write_json(out / 'mask_and_source_geometry.json', mask_meta)

        candidates = _discover_candidates(drive_root, source_img, max_files=max_candidate_files, exclude_root=out)
        if not candidates:
            raise RuntimeError('No coordinate-bearing coronary centerline candidates were discovered.')
        eval_rows = []
        candidate_lookup = {}
        prefix_rows = []
        for rec in candidates:
            ev, _, _ = _evaluate_candidate(rec, source_img, source_arr, masks)
            eval_rows.append(ev)
            candidate_lookup[rec['candidate_id']] = rec
            if rec['label'] in ('LAD', 'SECONDARY') and rec['coord_variant'] in ('stored_lps', 'recomputed_from_zyx'):
                prefix_rows.extend(_prefix_metrics(rec, source_img, source_arr, masks))
        summary = pd.DataFrame(eval_rows)
        summary.to_csv(out / 'left_coronary_candidate_inventory_and_metrics.csv', index=False)
        pd.DataFrame(prefix_rows).to_csv(out / 'left_coronary_prefix_metrics.csv', index=False)

        coord_cols = [
            'label', 'path', 'file_name', 'coord_variant', 'n_points', 'length_mm',
            'coord_delta_stored_vs_recomputed_median_mm', 'coord_delta_stored_vs_recomputed_p95_mm',
            'in_image_fraction', 'source_hu_median', 'source_hu_gt200_fraction',
            'current_median_distance_mm', 'legacy_median_distance_mm',
            'union_median_distance_mm', 'intersection_median_distance_mm',
            'current_inside_fraction', 'legacy_inside_fraction', 'union_inside_fraction',
            'is_source_volume_file', 'combined_reconciliation_score'
        ]
        summary[coord_cols].sort_values(['label', 'combined_reconciliation_score'], ascending=[True, False]).to_csv(
            out / 'coordinate_reconciliation_summary.csv', index=False)

        classification = _classification(summary)
        chosen = {}
        profiles = {}
        for label in ('RCA', 'LAD', 'SECONDARY'):
            row = _select_best(summary, label)
            if row is None:
                chosen[label] = None
                profiles[label] = None
                continue
            rec = candidate_lookup.get(row['candidate_id'])
            if rec is None:
                chosen[label] = None
                profiles[label] = None
                continue
            chosen[label] = rec['points']
            prof = _per_point_profile(rec, masks, source_img, source_arr)
            prof.to_csv(out / f'{label}_best_candidate_pointwise_diagnostic.csv', index=False)
            profiles[label] = prof
            classification[f'{label.lower()}_selected_candidate'] = {
                'path': row['path'], 'coord_variant': row['coord_variant'],
                'length_mm': float(row['length_mm']),
                'source_hu_median': float(row['source_hu_median']),
                'source_hu_gt200_fraction': float(row['source_hu_gt200_fraction']),
                'current_median_distance_mm': float(row['current_median_distance_mm']),
                'legacy_median_distance_mm': float(row['legacy_median_distance_mm']),
                'union_median_distance_mm': float(row['union_median_distance_mm']),
                'intersection_median_distance_mm': float(row['intersection_median_distance_mm']),
            }

        figs = _plot_rankings(out, summary)
        figs += _plot_profiles(out, {k: v for k, v in profiles.items() if k in ('LAD', 'SECONDARY')})
        figs.append(_plot_projection(out, masks, chosen))

        classification['interpretation_rules'] = {
            'coordinate_error': 'Accepted only when source voxel coordinates recomputed with source metadata improve a >8 mm mismatch to <2 mm.',
            'totalseg_undersegmentation': 'Requires RCA positive-control pass plus strong source-CCTA lumen HU support while both coronary masks remain >5 mm away.',
            'wrong_centerline': 'Requires an alternate left-coronary candidate that aligns <2 mm while the previously used source-volume file remains >8 mm away.',
            'no_free_rigid_registration': 'No arbitrary rigid/translation fit is used, because that could manufacture agreement.'
        }
        _write_json(out / 'reconciliation_conclusion.json', classification)

        top = summary[(summary['label'].isin(['RCA', 'LAD', 'SECONDARY'])) & summary['coord_variant'].isin(['stored_lps', 'recomputed_from_zyx'])].copy()
        top = top.sort_values(['label', 'combined_reconciliation_score'], ascending=[True, False]).groupby('label').head(8)
        table_cols = ['label', 'file_name', 'coord_variant', 'length_mm', 'source_hu_median', 'source_hu_gt200_fraction',
                      'current_median_distance_mm', 'legacy_median_distance_mm', 'union_median_distance_mm',
                      'intersection_median_distance_mm', 'combined_reconciliation_score']
        figs_html = ''.join(f'<h2>{p.stem}</h2><img src="{p.name}" style="max-width:100%">' for p in figs)
        html = out / 'OPENPLAQUE_LEFT_CORONARY_COORDINATE_RECONCILIATION_REPORT.html'
        html.write_text(
            '<html><head><meta charset="utf-8"><title>OpenPlaque Left Coronary Reconciliation</title></head>'
            '<body style="font-family:Arial;max-width:1200px;margin:auto">'
            '<h1>OpenPlaque — Left-Coronary Coordinate / Segmentation Reconciliation</h1>'
            '<p><b>Research use only.</b> RCA is used as the positive-control source-coordinate anchor. '
            'No arbitrary free rigid registration is allowed.</p>'
            f'<h2>Conclusion</h2><pre>{json.dumps(classification, indent=2)}</pre>'
            '<h2>Best candidate metrics</h2>' + top[table_cols].to_html(index=False) + figs_html + '</body></html>'
        )
        zf = out / 'OPENPLAQUE_LEFT_CORONARY_COORDINATE_RECONCILIATION_REPORT_BACK.zip'
        with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file() and p != zf:
                    z.write(p, p.relative_to(out))
        state = {'status': 'COMPLETE', 'conclusion': classification['primary_conclusion'],
                 'report': str(html), 'zip': str(zf)}
        _write_json(out / 'run_state.json', state)
        return {'output_dir': str(out), 'report': str(html), 'zip': str(zf), 'conclusion': classification}
    except Exception as e:
        state = {'status': 'ERROR', 'error': repr(e), 'traceback': traceback.format_exc()}
        _write_json(out / 'run_state.json', state)
        (out / 'ERROR.txt').write_text(state['traceback'])
        raise
