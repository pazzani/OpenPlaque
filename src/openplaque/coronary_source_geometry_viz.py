from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def _centerline_points(df: pd.DataFrame) -> np.ndarray:
    required = ['lps_x_mm', 'lps_y_mm', 'lps_z_mm']
    if not all(c in df.columns for c in required):
        raise ValueError(f'Centerline missing physical-coordinate columns {required}')
    return df[required].to_numpy(float)


def _physical_to_zyx(img: sitk.Image, pts_lps: np.ndarray) -> np.ndarray:
    out = []
    for p in pts_lps:
        x, y, z = img.TransformPhysicalPointToContinuousIndex(tuple(map(float, p)))
        out.append([z, y, x])
    return np.asarray(out, float)


def _mask_tree(img: sitk.Image, mask: np.ndarray) -> cKDTree:
    """Build a physical-LPS KD tree from positive mask voxel centers.

    This is deliberately used instead of a full-volume Euclidean distance
    transform. The latter allocates a large float64 volume and previously
    caused the source-geometry notebook to exhaust Colab RAM before writing
    its first result.
    """
    zyx = np.argwhere(mask.astype(bool))
    if len(zyx) == 0:
        raise ValueError('Coronary mask is empty')
    xyz = zyx[:, ::-1].astype(np.float64)
    spacing = np.asarray(img.GetSpacing(), float)
    origin = np.asarray(img.GetOrigin(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    pts_lps = origin + (xyz * spacing) @ direction.T
    return cKDTree(pts_lps)


def centerline_mask_qc(img: sitk.Image, mask: np.ndarray, df: pd.DataFrame, tree: cKDTree | None = None) -> pd.DataFrame:
    pts_lps = _centerline_points(df)
    pts_zyx = _physical_to_zyx(img, pts_lps)
    idx = np.rint(pts_zyx).astype(int)
    valid = np.ones(len(idx), bool)
    for a in range(3):
        valid &= (idx[:, a] >= 0) & (idx[:, a] < mask.shape[a])
    inside = np.zeros(len(idx), bool)
    good = idx[valid]
    inside[valid] = mask[good[:, 0], good[:, 1], good[:, 2]]

    if tree is None:
        tree = _mask_tree(img, mask)
    d, _ = tree.query(pts_lps, k=1, workers=-1)
    d = np.asarray(d, float)
    d[~valid] = np.nan

    arc = df['arc_mm'].to_numpy(float) if 'arc_mm' in df.columns else np.arange(len(df), dtype=float)
    return pd.DataFrame({
        'arc_mm': arc,
        'inside_coronary_mask': inside,
        'distance_to_coronary_mask_mm': d,
        'in_image': valid,
    })


def _mesh_from_mask(mask: np.ndarray, img: sitk.Image, stride: int = 2, max_faces: int = 50000):
    pos = np.argwhere(mask)
    if len(pos) == 0:
        return None, None
    lo = np.maximum(pos.min(axis=0) - 2, 0)
    hi = np.minimum(pos.max(axis=0) + 3, np.asarray(mask.shape))
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    m = crop[::stride, ::stride, ::stride].astype(np.uint8)
    if m.max() == 0 or min(m.shape) < 2:
        return None, None
    verts_local_zyx, faces, _, _ = marching_cubes(m, 0.5)
    verts_global_zyx = verts_local_zyx * stride + lo
    xyz_index = verts_global_zyx[:, ::-1]
    spacing = np.asarray(img.GetSpacing(), float)
    origin = np.asarray(img.GetOrigin(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    verts = origin + (xyz_index * spacing) @ direction.T
    if len(faces) > max_faces:
        sel = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = faces[sel]
    return verts, faces


def _add_surface(ax, mask, img, alpha=0.15):
    v, f = _mesh_from_mask(mask, img)
    if v is None:
        return
    poly = Poly3DCollection(v[f], alpha=alpha, linewidths=0)
    ax.add_collection3d(poly)


def _set_axes_from_centerlines(ax, centerlines: dict[str, pd.DataFrame], pad=12.0):
    pts = np.concatenate([_centerline_points(df) for df in centerlines.values()], axis=0)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])


def _save_3d_context(out: Path, img: sitk.Image, coronary: np.ndarray, centerlines: dict[str, pd.DataFrame], myocardium_path=None, aorta_path=None):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    if myocardium_path:
        mi = sitk.ReadImage(str(myocardium_path)); m = sitk.GetArrayFromImage(mi) > 0
        _add_surface(ax, m, mi, alpha=0.06)
    if aorta_path:
        ai = sitk.ReadImage(str(aorta_path)); a = sitk.GetArrayFromImage(ai) > 0
        _add_surface(ax, a, ai, alpha=0.10)
    _add_surface(ax, coronary, img, alpha=0.35)
    for name, df in centerlines.items():
        p = _centerline_points(df)
        label = 'Secondary (source file LCX)' if name == 'LCX' else name
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=2.0, label=label)
    _set_axes_from_centerlines(ax, centerlines)
    ax.set_xlabel('LPS X (mm)'); ax.set_ylabel('LPS Y (mm)'); ax.set_zlabel('LPS Z (mm)')
    ax.set_title('Source-CCTA coronary geometry with independent segmentation')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    p = out / '01_source_coronary_heart_context_3d.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def _save_distance_plot(out: Path, qc: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, df in qc.items():
        label = 'Secondary (source file LCX)' if name == 'LCX' else name
        ax.plot(df['arc_mm'], df['distance_to_coronary_mask_mm'], label=label)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel('Centerline arc length (mm)')
    ax.set_ylabel('Distance to TotalSegmentator coronary mask (mm)')
    ax.set_title('Independent source-mask support along frozen centerlines')
    ax.legend()
    fig.tight_layout()
    p = out / '02_centerline_mask_distance.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def _save_agreement_mips(out: Path, cur: np.ndarray, leg: np.ndarray):
    dis = cur ^ leg
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    names = [('Current', cur), ('Legacy', leg), ('Disagreement', dis)]
    for r, (name, a) in enumerate(names):
        for c, axis in enumerate((0, 1, 2)):
            axes[r, c].imshow(a.max(axis=axis), cmap='gray')
            axes[r, c].set_title(f'{name} MIP axis {axis}')
            axes[r, c].axis('off')
    fig.tight_layout()
    p = out / '03_coronary_model_agreement_mips.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / 'Coronary_Source_Geometry_Visualization_v1')
    out.mkdir(parents=True, exist_ok=True)

    batch = drive_root / 'GPU_Batch_Pipeline_v1'
    if not batch.exists():
        candidates = [p for p in drive_root.rglob('GPU_Batch_Pipeline_v1') if p.is_dir()]
        if not candidates:
            raise FileNotFoundError('GPU_Batch_Pipeline_v1 folder not found')
        batch = candidates[0]

    cur_hits = list((batch / '02_coronary_ensemble').rglob('current/coronary_arteries.nii.gz'))
    leg_hits = list((batch / '02_coronary_ensemble').rglob('legacy/coronary_arteries.nii.gz'))
    if not cur_hits:
        raise FileNotFoundError('current coronary_arteries.nii.gz not found')
    if not leg_hits:
        raise FileNotFoundError('legacy coronary_arteries.nii.gz not found')
    cur_path, leg_path = cur_hits[0], leg_hits[0]
    cur_img = sitk.ReadImage(str(cur_path)); leg_img = sitk.ReadImage(str(leg_path))
    cur = sitk.GetArrayFromImage(cur_img) > 0
    leg = sitk.GetArrayFromImage(leg_img) > 0
    intersection = cur & leg
    union = cur | leg

    centerline_dir = drive_root / 'Source_Volume_Coronary_Centerlines'
    if not centerline_dir.exists():
        candidates = [p for p in drive_root.rglob('Source_Volume_Coronary_Centerlines') if p.is_dir()]
        if not candidates:
            raise FileNotFoundError('Source_Volume_Coronary_Centerlines folder not found')
        centerline_dir = candidates[0]
    centerlines = {}
    for vessel in ['RCA', 'LAD', 'LCX']:
        p = _find_optional(centerline_dir, f'{vessel}_source_centerline.csv')
        if p:
            centerlines[vessel] = pd.read_csv(p)
    if 'RCA' not in centerlines or 'LAD' not in centerlines:
        raise RuntimeError('RCA and LAD source centerlines are required')

    context_root = batch / '03_whole_heart'
    myocardium = _find_optional(context_root, 'heart_myocardium.nii.gz')
    aorta_hits = [p for p in context_root.rglob('*aorta.nii.gz')]
    aorta = None
    for p in aorta_hits:
        if 'heartchambers_highres' in str(p):
            aorta = p
            break
    if aorta is None and aorta_hits:
        aorta = aorta_hits[0]

    # One compact KD tree replaces three multi-gigabyte full-volume EDTs.
    tree = _mask_tree(cur_img, intersection)
    qc = {name: centerline_mask_qc(cur_img, intersection, df, tree=tree) for name, df in centerlines.items()}
    rows = []
    for name, df in qc.items():
        finite = df['distance_to_coronary_mask_mm'].dropna()
        rows.append({
            'vessel': name if name != 'LCX' else 'secondary_source_file_LCX',
            'points': len(df),
            'inside_intersection_fraction': float(df['inside_coronary_mask'].mean()),
            'median_distance_mm': float(finite.median()) if len(finite) else np.nan,
            'p90_distance_mm': float(finite.quantile(.9)) if len(finite) else np.nan,
            'max_distance_mm': float(finite.max()) if len(finite) else np.nan,
        })
        df.to_csv(out / f'{name}_centerline_vs_coronary_mask.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'centerline_coronary_mask_summary.csv', index=False)

    vv = float(np.prod(cur_img.GetSpacing()))
    denom = max(cur.sum() + leg.sum(), 1)
    metrics = {
        'current_volume_mm3': float(cur.sum() * vv),
        'legacy_volume_mm3': float(leg.sum() * vv),
        'intersection_volume_mm3': float(intersection.sum() * vv),
        'union_volume_mm3': float(union.sum() * vv),
        'disagreement_volume_mm3': float((cur ^ leg).sum() * vv),
        'dice_current_vs_legacy': float(2 * intersection.sum() / denom),
        'coordinate_system': 'source CCTA physical LPS',
        'distance_method': 'physical-LPS KD tree to intersection-mask voxel centers',
        'secondary_identity_note': 'LCX_source_centerline.csv is displayed as secondary; anatomical identity remains unresolved.',
        'figure_errors': [],
    }
    (out / 'coronary_geometry_metrics.json').write_text(json.dumps(metrics, indent=2))

    figs = []
    figure_jobs = [
        ('3d_context', lambda: _save_3d_context(out, cur_img, intersection, centerlines, myocardium, aorta)),
        ('distance_plot', lambda: _save_distance_plot(out, qc)),
        ('agreement_mips', lambda: _save_agreement_mips(out, cur, leg)),
    ]
    for label, fn in figure_jobs:
        try:
            figs.append(fn())
        except Exception as e:
            metrics['figure_errors'].append({'figure': label, 'error': str(e)})
            print(f'Figure {label} failed but tabular outputs are preserved: {e}')
    (out / 'coronary_geometry_metrics.json').write_text(json.dumps(metrics, indent=2))

    html = out / 'OPENPLAQUE_CORONARY_SOURCE_GEOMETRY_VISUALIZATION_REPORT.html'
    blocks = ''.join(f'<h2>{p.stem}</h2><img src="{p.name}" style="max-width:100%">' for p in figs)
    html.write_text(
        '<html><head><meta charset="utf-8"><title>OpenPlaque Source Coronary Geometry</title></head><body>'
        '<h1>OpenPlaque Source-CCTA Coronary Geometry Integration</h1>'
        '<p>Research use only. Source-coordinate TotalSegmentator coronary masks are compared with frozen source-volume centerlines. '
        'The secondary branch is not relabeled as LCX merely because an older source filename contains LCX.</p>'
        + summary.to_html(index=False) + blocks + '</body></html>'
    )
    zf = out / 'OPENPLAQUE_CORONARY_SOURCE_GEOMETRY_VISUALIZATION_REPORT_BACK.zip'
    with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zf:
                z.write(p, p.name)
    return {'output_dir': str(out), 'report': str(html), 'zip': str(zf), 'metrics': metrics, 'summary': rows}
