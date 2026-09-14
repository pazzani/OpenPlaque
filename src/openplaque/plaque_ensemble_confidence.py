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
from scipy.ndimage import label as cc_label


def _fold_masks(vessel_root: Path, vessel: str):
    hits = []
    for fold in range(5):
        p = vessel_root / f'fold_{fold}' / f'{vessel}.nii.gz'
        if p.exists(): hits.append((fold, p))
    if len(hits) < 3:
        raise RuntimeError(f'{vessel}: need >=3 fold masks, found {len(hits)}')
    return hits


def _physical_centroid(img: sitk.Image, centroid_zyx):
    z, y, x = centroid_zyx
    return img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))


def _component_table(mask, vote, img, vessel):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = cc_label(mask, structure=structure)
    spacing_zyx = np.asarray(img.GetSpacing(), float)[::-1]
    vv = float(np.prod(img.GetSpacing()))
    rows = []
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if len(pts) == 0: continue
        centroid = pts.mean(0)
        lo = pts.min(0); hi = pts.max(0)
        px, py, pz = _physical_centroid(img, centroid)
        rows.append({
            'vessel': vessel,
            'lesion_id': k,
            'voxels': int(len(pts)),
            'volume_mm3': float(len(pts) * vv),
            'mean_vote': float(vote[lab == k].mean()),
            'max_vote': int(vote[lab == k].max()),
            'centroid_z': float(centroid[0]), 'centroid_y': float(centroid[1]), 'centroid_x': float(centroid[2]),
            'native_physical_x': float(px), 'native_physical_y': float(py), 'native_physical_z': float(pz),
            'bbox_z0': int(lo[0]), 'bbox_y0': int(lo[1]), 'bbox_x0': int(lo[2]),
            'bbox_z1': int(hi[0]), 'bbox_y1': int(hi[1]), 'bbox_x1': int(hi[2]),
        })
    return pd.DataFrame(rows).sort_values('volume_mm3', ascending=False) if rows else pd.DataFrame()


def _longitudinal_axis(arr, img):
    spacing_zyx = np.asarray(img.GetSpacing(), float)[::-1]
    physical_extent = np.asarray(arr.shape, float) * spacing_zyx
    return int(np.argmax(physical_extent)), spacing_zyx


def _profile(vote, img, vessel):
    axis, spacing_zyx = _longitudinal_axis(vote, img)
    moved = np.moveaxis(vote, axis, 0)
    rows = []
    for i, sl in enumerate(moved):
        active = sl > 0
        rows.append({
            'vessel': vessel,
            'native_axis': axis,
            'slice_index': i,
            'native_longitudinal_mm': float(i * spacing_zyx[axis]),
            'vote_sum': int(sl.sum()),
            'voxels_vote_ge3': int((sl >= 3).sum()),
            'voxels_vote_ge4': int((sl >= 4).sum()),
            'voxels_vote_5': int((sl == 5).sum()),
            'mean_vote_among_any_plaque': float(sl[active].mean()) if active.any() else np.nan,
        })
    return pd.DataFrame(rows), axis


def _save_vote_mips(out, vessel, vote):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, axis in zip(axes, (0, 1, 2)):
        im = ax.imshow(vote.max(axis=axis), vmin=0, vmax=5, cmap='viridis')
        ax.set_title(f'{vessel}: max plaque votes, axis {axis}')
        ax.axis('off')
    fig.colorbar(im, ax=axes.tolist(), shrink=.75, label='fold votes (0–5)')
    fig.suptitle(f'{vessel} 5-fold plaque confidence atlas — native curved-series coordinates')
    p = out / f'{vessel}_01_vote_mips.png'
    fig.savefig(p, dpi=180, bbox_inches='tight'); plt.close(fig)
    return p


def _save_profile(out, vessel, profile):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(profile['native_longitudinal_mm'], profile['voxels_vote_ge3'], label='>=3/5')
    ax.plot(profile['native_longitudinal_mm'], profile['voxels_vote_ge4'], label='>=4/5')
    ax.plot(profile['native_longitudinal_mm'], profile['voxels_vote_5'], label='5/5')
    ax.set_xlabel('Native vessel-volume longitudinal coordinate (mm)')
    ax.set_ylabel('Plaque voxels per slice')
    ax.set_title(f'{vessel}: plaque confidence along native vessel volume')
    ax.legend(); fig.tight_layout()
    p = out / f'{vessel}_02_native_longitudinal_profile.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def _save_informative_slices(out, vessel, vote, axis, img):
    moved = np.moveaxis(vote, axis, 0)
    scores = np.array([(sl >= 3).sum() for sl in moved])
    uncertain = np.array([((sl == 2) | (sl == 3)).sum() for sl in moved])
    candidates = []
    for arr in (scores, uncertain):
        for idx in np.argsort(arr)[::-1]:
            if arr[idx] <= 0: break
            if all(abs(int(idx) - j) >= 3 for j in candidates):
                candidates.append(int(idx))
            if len(candidates) >= 6: break
        if len(candidates) >= 6: break
    if not candidates: candidates = [len(moved)//2]
    n = len(candidates)
    fig, axes = plt.subplots(2, int(np.ceil(n/2)), figsize=(4*int(np.ceil(n/2)), 7))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, candidates):
        ax.imshow(moved[idx], vmin=0, vmax=5, cmap='viridis')
        ax.set_title(f'slice {idx}: >=3/5={int((moved[idx]>=3).sum())}')
        ax.axis('off')
    for ax in axes[n:]: ax.axis('off')
    fig.suptitle(f'{vessel}: automatically selected informative plaque-vote slices')
    fig.tight_layout()
    p = out / f'{vessel}_03_informative_vote_slices.png'
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_root=None):
    drive_root = Path(drive_root)
    batch = drive_root / 'GPU_Batch_Pipeline_v1' / '01_plaque_5fold'
    if not batch.exists():
        hits = [p for p in drive_root.rglob('01_plaque_5fold') if p.is_dir()]
        if not hits: raise FileNotFoundError('01_plaque_5fold not found')
        batch = hits[0]
    out = Path(output_root or drive_root / 'Plaque_Ensemble_Confidence_Atlas_v1')
    out.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_lesions = []
    all_figs = []
    coordinate_notes = {}
    for vessel in ['RCA', 'LAD']:
        hits = _fold_masks(batch / vessel, vessel)
        imgs = [sitk.ReadImage(str(p)) for _, p in hits]
        arrays = [sitk.GetArrayFromImage(im).astype(np.uint8) for im in imgs]
        if len({a.shape for a in arrays}) != 1:
            raise RuntimeError(f'{vessel}: fold masks differ in shape')
        plaque = np.stack([a == 2 for a in arrays], axis=0)
        vote = plaque.sum(0).astype(np.uint8)
        ref = imgs[0]
        vv = float(np.prod(ref.GetSpacing()))

        vi = sitk.GetImageFromArray(vote); vi.CopyInformation(ref)
        sitk.WriteImage(vi, str(out / f'{vessel}_plaque_vote_0to5.nii.gz'))
        for threshold, name in [(5, 'strict_5of5'), (4, 'high_4plus'), (3, 'majority_3plus')]:
            m = (vote >= threshold).astype(np.uint8)
            mi = sitk.GetImageFromArray(m); mi.CopyInformation(ref)
            sitk.WriteImage(mi, str(out / f'{vessel}_{name}.nii.gz'))

        exact = {f'vote_{k}_volume_mm3': float((vote == k).sum() * vv) for k in range(1, 6)}
        row = {
            'vessel': vessel,
            'n_folds': len(hits),
            'native_coordinate_system': 'curved vessel-series model coordinates',
            'strict_5of5_mm3': float((vote == 5).sum() * vv),
            'high_4plus_mm3': float((vote >= 4).sum() * vv),
            'majority_3plus_mm3': float((vote >= 3).sum() * vv),
            'any_fold_mm3': float((vote >= 1).sum() * vv),
            **exact,
        }
        all_summary.append(row)

        lesions = _component_table(vote >= 3, vote, ref, vessel)
        if not lesions.empty:
            lesions.to_csv(out / f'{vessel}_majority_lesions.csv', index=False)
            all_lesions.append(lesions)
        profile, axis = _profile(vote, ref, vessel)
        profile.to_csv(out / f'{vessel}_native_longitudinal_profile.csv', index=False)
        all_figs += [_save_vote_mips(out, vessel, vote), _save_profile(out, vessel, profile), _save_informative_slices(out, vessel, vote, axis, ref)]
        coordinate_notes[vessel] = {'array_shape_zyx': list(vote.shape), 'spacing_xyz_mm': list(ref.GetSpacing()), 'selected_native_longitudinal_axis_zyx': axis}

    summary = pd.DataFrame(all_summary)
    summary.to_csv(out / 'plaque_vote_confidence_summary.csv', index=False)
    lesions_all = pd.concat(all_lesions, ignore_index=True) if all_lesions else pd.DataFrame()
    lesions_all.to_csv(out / 'plaque_majority_lesions_all.csv', index=False)
    (out / 'coordinate_system_notes.json').write_text(json.dumps({
        'important': 'Plaque masks are in curved vessel-series model coordinates and are not voxel-co-registered with source CCTA or TotalSegmentator masks.',
        'vessels': coordinate_notes
    }, indent=2))

    html = out / 'OPENPLAQUE_PLAQUE_ENSEMBLE_CONFIDENCE_ATLAS_REPORT.html'
    figs_html = ''.join(f'<h3>{p.stem}</h3><img src="{p.name}" style="max-width:100%">' for p in all_figs)
    html.write_text(
        '<html><head><meta charset="utf-8"><title>OpenPlaque Plaque Ensemble Confidence</title></head><body>'
        '<h1>OpenPlaque 5-Fold Plaque Confidence Atlas</h1>'
        '<p><b>Coordinate-system warning:</b> these masks are in native curved vessel-series coordinates. They are not directly overlaid on source-CCTA geometry. '
        'Confidence is reported as the number of folds (0–5) voting plaque at each voxel.</p>'
        + summary.to_html(index=False) + '<h2>Majority-vote lesion components</h2>' + lesions_all.head(100).to_html(index=False) + figs_html + '</body></html>'
    )
    zf = out / 'OPENPLAQUE_PLAQUE_ENSEMBLE_CONFIDENCE_ATLAS_REPORT_BACK.zip'
    with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zf: z.write(p, p.name)
    return {'output_dir': str(out), 'report': str(html), 'zip': str(zf), 'summary': all_summary, 'n_lesions': int(len(lesions_all))}
