from __future__ import annotations

import gc
import json
import math
import zipfile
from dataclasses import dataclass
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
ALGORITHM = 'lcx-parent-continuation-topology-v1.0-lowmem'
OUTPUT_DIRNAME = 'LCX_Parent_Continuation_Topology_v1'
SOURCE_CACHE = Path('Cache/Secondary_3D_Vesselness_Topology_v1')
MASTER = Path('Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
TS = Path('TotalSegmentator_Cardiovascular_Cache_v1')
COR_CURRENT = TS / 'coronary_arteries/coronary_arteries.nii.gz'
COR_LEGACY = TS / 'coronary_arteries_LEGACY/coronary_arteries.nii.gz'
PRIOR = Path('Joint_Three_Vessel_Template_Classifier_v1')
PRIOR_RANKING = PRIOR / 'LCX_joint_candidate_ranking.csv'
LEAF_FILES = [PRIOR / f'candidate_{i:02d}_source_path.csv' for i in range(1, 6)]
RECURSIVE = Path('LCX_Family2_Recursive_AV_Groove_v1')
C6_ID, C7_ID, C9_ID = 6, 7, 9

STATUS_NOT_TRUNCATION = 'C6_C9_NOT_TRUNCATION_EQUIVALENT'
STATUS_AMBIG = 'F2_PARENT_DAUGHTER_TOPOLOGY_AMBIGUOUS'
STATUS_PARENT = 'C6_PARENT_CONTINUATION_C7_SIDE_BRANCH_REQUIRES_VISUAL_QC'


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=True, default=str), encoding='utf-8')


@dataclass(frozen=True)
class Geometry:
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        idx = ((pts - self.origin) @ np.linalg.inv(self.direction).T) / self.spacing
        return idx[:, ::-1]


def _source(cache):
    arr = np.load(_req(cache / 'series7_int16.npy'), mmap_mode='r')
    m = json.loads(_req(cache / 'series7_int16.json').read_text())
    sp = np.asarray(m['spacing_zyx'], float)[::-1]
    iop = np.asarray(m['image_orientation_patient'], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    d = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    return Geometry(np.asarray(m['positions_lps_mm'][0], float), sp, d), arr


def _sample(arr, g, pts, order=1, cval=-1024.0):
    return map_coordinates(arr, g.xyz_to_zyx(pts).T, order=order, mode='constant', cval=cval)


def _load_path(path):
    d = pd.read_csv(_req(path))
    cols = ('lps_x_mm', 'lps_y_mm', 'lps_z_mm')
    if not all(c in d.columns for c in cols):
        raise ValueError(f'Expected LPS path columns in {path}: {list(d.columns)}')
    return d[list(cols)].to_numpy(float)


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))


def _interp(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=.25):
    a = _arc(p)
    if len(a) < 2 or a[-1] <= 0:
        return np.asarray(p, float), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _orient(paths):
    ref = np.asarray(paths[0], float)[0]
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref):
            p = p[::-1].copy()
        out.append(p)
    return out


def _angle(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a /= max(np.linalg.norm(a), 1e-9); b /= max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def _vector_over(p, start_arc, length=2.0):
    total = _arc(p)[-1]
    a0 = min(max(0.0, start_arc), total)
    a1 = min(total, a0 + length)
    x = _interp(p, [a0, a1])
    return x[-1] - x[0]


def _consensus_prefix(paths, step=.25, tol=.60, sustain=4):
    paths = _orient(paths)
    minlen = min(_arc(p)[-1] for p in paths)
    q = np.arange(0, minlen + 1e-9, step)
    s = np.stack([_interp(p, q) for p in paths], axis=0)
    centroid = np.median(s, axis=0)
    dev = np.max(np.linalg.norm(s - centroid[None, :, :], axis=2), axis=0)
    first_bad = None
    start = max(1, int(round(5.0 / step)))
    for i in range(start, max(start, len(q) - sustain + 1)):
        if np.all(dev[i:i+sustain] > tol):
            first_bad = i
            break
    end_i = first_bad - 1 if first_bad is not None else len(q) - 1
    return centroid[:end_i+1], q[:end_i+1], dev[:end_i+1], None if first_bad is None else float(q[first_bad])


def _img_zyx_to_xyz(im, pts):
    idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
    o = np.asarray(im.GetOrigin()); sp = np.asarray(im.GetSpacing()); d = np.asarray(im.GetDirection()).reshape(3, 3)
    return o + (idx * sp) @ d.T


def _tree(path, max_points=350000):
    im = sitk.ReadImage(str(_req(path)))
    a = sitk.GetArrayViewFromImage(im)
    z = np.argwhere(a > 0)
    if not len(z):
        raise ValueError(f'No foreground in {path}')
    if len(z) > max_points:
        z = z[::int(math.ceil(len(z) / max_points))]
    xyz = _img_zyx_to_xyz(im, z).astype(np.float32, copy=False)
    del z, a, im
    gc.collect()
    return cKDTree(xyz)


def _support(tree, path):
    p, _ = _resample(path, .25)
    return float(np.mean(tree.query(p)[0] <= 1.0))


def _hu(src, g, path):
    p, _ = _resample(path, .25)
    h = _sample(src, g, p)
    return float(np.mean((h >= 120) & (h <= 1200))), float(np.median(h))


def _truncation_metrics(long_path, short_path, step=.25):
    long_path, short_path = _orient([long_path, short_path])
    L = _arc(long_path)[-1]; S = _arc(short_path)[-1]
    q = np.arange(0, S + 1e-9, step)
    if q[-1] < S - 1e-6:
        q = np.r_[q, S]
    a = _interp(long_path, q); b = _interp(short_path, q)
    d = np.linalg.norm(a - b, axis=1)
    terminal_window = min(2.0, max(.5, S))
    va = _vector_over(long_path, max(0, S-terminal_window), terminal_window)
    vb = _vector_over(short_path, max(0, S-terminal_window), terminal_window)
    out = {
        'C6_length_mm': float(L), 'C9_length_mm': float(S), 'C6_extension_beyond_C9_mm': float(L-S),
        'median_prefix_separation_mm': float(np.median(d)), 'p95_prefix_separation_mm': float(np.percentile(d, 95)),
        'max_prefix_separation_mm': float(np.max(d)), 'C9_endpoint_to_C6_same_arc_mm': float(d[-1]),
        'terminal_direction_angle_deg': _angle(va, vb),
    }
    out['truncation_gate_pass'] = bool(out['C6_extension_beyond_C9_mm'] >= 2.0 and out['p95_prefix_separation_mm'] <= .35 and out['max_prefix_separation_mm'] <= .60 and out['C9_endpoint_to_C6_same_arc_mm'] <= .50 and out['terminal_direction_angle_deg'] <= 20.0)
    return out, q, d


def _plane(g, src, center, tangent, half=7., step=.2):
    t = tangent / max(np.linalg.norm(tangent), 1e-9)
    axes = np.eye(3); seed = axes[np.argmin(np.abs(axes @ t))]
    n = np.cross(t, seed); n /= max(np.linalg.norm(n), 1e-9)
    b = np.cross(t, n); b /= max(np.linalg.norm(b), 1e-9)
    q = np.arange(-half, half + 1e-9, step); yy, xx = np.meshgrid(q, q, indexing='ij')
    pts = center[None, None, :] + xx[..., None] * n + yy[..., None] * b
    return _sample(src, g, pts.reshape(-1, 3)).reshape(len(q), len(q)), q


def synthetic_parent_continuation_self_test():
    t = np.linspace(0, 30, 121)
    c6 = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    c9 = c6[t <= 24].copy()
    c7 = c6.copy(); c7[t > 20, 1] = (t[t > 20] - 20) * 1.2
    m, _, _ = _truncation_metrics(c6, c9)
    assert m['truncation_gate_pass']
    con, q, _, split = _consensus_prefix([c6, c7, c9], .25, .4, 3)
    assert 19.5 <= q[-1] <= 20.5 and split is not None
    return {'ok': True, 'truncation': m, 'common_mm': float(q[-1])}


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_dir=None):
    root = Path(drive_root); out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'run_state.json', {'status': 'STARTED', 'algorithm': ALGORITHM, 'baseline_commit': BASELINE})
    required = [root/SOURCE_CACHE/'series7_int16.npy', root/SOURCE_CACHE/'series7_int16.json', root/MASTER, root/COR_CURRENT, root/COR_LEGACY, root/PRIOR_RANKING] + [root/p for p in LEAF_FILES]
    for p in required: _req(p)

    master = json.loads((root/MASTER).read_text())
    ranking = pd.read_csv(root/PRIOR_RANKING)
    paths = _orient([_load_path(root/p) for p in LEAF_FILES])
    mapping = {}
    priors = {}
    for i in range(1, 6):
        row = ranking.iloc[i-1].to_dict(); cid = int(row['candidate_id'])
        mapping[cid] = paths[i-1]; priors[cid] = row
    for cid in (C6_ID, C7_ID, C9_ID):
        if cid not in mapping: raise RuntimeError(f'Missing source candidate C{cid}')
    c6, c7, c9 = mapping[C6_ID], mapping[C7_ID], mapping[C9_ID]

    trunc, tq, td = _truncation_metrics(c6, c9)
    _write_json(out/'C6_C9_truncation_summary.json', trunc)
    pd.DataFrame({'arc_mm': tq, 'C6_C9_separation_mm': td}).to_csv(out/'C6_C9_prefix_separation.csv', index=False)

    f2_con, f2_q, f2_dev, first_split = _consensus_prefix([c6, c7, c9], .25, .60, 4)
    split_arc = float(f2_q[-1])
    pd.DataFrame(f2_con, columns=['lps_x_mm','lps_y_mm','lps_z_mm']).assign(arc_mm=f2_q, max_between_path_deviation_mm=f2_dev).to_csv(out/'family2_parent_trunk.csv', index=False)
    incoming = _vector_over(f2_con, max(0.0, split_arc-2.0), 2.0)
    c6_vec = _vector_over(c6, split_arc, 2.0); c7_vec = _vector_over(c7, split_arc, 2.0); c9_vec = _vector_over(c9, split_arc, min(2.0, max(.5, _arc(c9)[-1]-split_arc)))
    c6_ang = _angle(incoming, c6_vec); c7_ang = _angle(incoming, c7_vec); c9_ang = _angle(incoming, c9_vec)
    daughter_angle = _angle(c6_vec, c7_vec)

    g, src = _source(root/SOURCE_CACHE)
    cur = _tree(root/COR_CURRENT); leg = _tree(root/COR_LEGACY)
    def post(path):
        total = _arc(path)[-1]; q = np.arange(split_arc, total + 1e-9, .25)
        if q[-1] < total - 1e-6: q = np.r_[q, total]
        p = _interp(path, q)
        return p
    c6_post, c7_post = post(c6), post(c7)
    c6_cur, c6_leg = _support(cur, c6_post), _support(leg, c6_post)
    c7_cur, c7_leg = _support(cur, c7_post), _support(leg, c7_post)
    c6_hf, c6_hu = _hu(src, g, c6_post); c7_hf, c7_hu = _hu(src, g, c7_post)

    topo = {
        'family2_split_arc_from_candidate_origin_mm': split_arc,
        'first_sustained_split_arc_mm': first_split,
        'C6_parent_angle_from_incoming_deg': c6_ang,
        'C9_angle_from_incoming_deg': c9_ang,
        'C7_side_branch_angle_from_incoming_deg': c7_ang,
        'C6_vs_C7_daughter_angle_deg': daughter_angle,
        'parent_side_angle_separation_deg': c7_ang - c6_ang,
        'C6_current_support_fraction': c6_cur, 'C6_legacy_support_fraction': c6_leg, 'C6_robust_hu_fraction': c6_hf, 'C6_median_hu': c6_hu,
        'C7_current_support_fraction': c7_cur, 'C7_legacy_support_fraction': c7_leg, 'C7_robust_hu_fraction': c7_hf, 'C7_median_hu': c7_hu,
    }
    topo['topology_gate_pass'] = bool(trunc['truncation_gate_pass'] and c6_ang <= 25.0 and c7_ang >= 35.0 and daughter_angle >= 25.0 and (c7_ang-c6_ang) >= 20.0 and min(c6_cur,c6_leg,c7_cur,c7_leg,c6_hf,c7_hf) >= .90)
    _write_json(out/'parent_daughter_topology_summary.json', topo)

    role_rows = [
        {'source_candidate_id': 6, 'role_hypothesis': 'parent_continuation', 'angle_from_incoming_deg': c6_ang, 'current_support_fraction': c6_cur, 'legacy_support_fraction': c6_leg, 'robust_hu_fraction': c6_hf, 'median_hu': c6_hu, 'prior_LCX_score': float(priors[6]['LCX_score'])},
        {'source_candidate_id': 9, 'role_hypothesis': 'truncated_duplicate_of_C6' if trunc['truncation_gate_pass'] else 'independent_path', 'angle_from_incoming_deg': c9_ang, 'current_support_fraction': np.nan, 'legacy_support_fraction': np.nan, 'robust_hu_fraction': np.nan, 'median_hu': np.nan, 'prior_LCX_score': float(priors[9]['LCX_score'])},
        {'source_candidate_id': 7, 'role_hypothesis': 'daughter_side_branch', 'angle_from_incoming_deg': c7_ang, 'current_support_fraction': c7_cur, 'legacy_support_fraction': c7_leg, 'robust_hu_fraction': c7_hf, 'median_hu': c7_hu, 'prior_LCX_score': float(priors[7]['LCX_score'])},
    ]
    pd.DataFrame(role_rows).to_csv(out/'candidate_topology_roles.csv', index=False)

    # Prior recursive AV-groove metrics are context only, never decision inputs.
    prior_context = None
    if (root/RECURSIVE/'family2_leaf_scores.csv').exists():
        pctx = pd.read_csv(root/RECURSIVE/'family2_leaf_scores.csv')
        prior_context = pctx[pctx.source_candidate_id.isin([6,7,9])].to_dict(orient='records')
        _write_json(out/'prior_recursive_av_groove_context.json', {'used_in_decision': False, 'rows': prior_context})

    plt.figure(figsize=(7,4))
    plt.plot(tq, td, lw=1.5); plt.axhline(.35, ls='--', label='p95 truncation gate'); plt.axhline(.60, ls=':', label='max gate')
    plt.xlabel('C9 arc (mm)'); plt.ylabel('C6–C9 separation (mm)'); plt.title('C9 prefix overlap with C6'); plt.legend(); plt.tight_layout(); plt.savefig(out/'01_C6_C9_prefix_overlap.png', dpi=180); plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(['C6 parent','C9 short','C7 daughter'], [c6_ang,c9_ang,c7_ang]); plt.axhline(25, ls='--', label='parent gate'); plt.axhline(35, ls=':', label='daughter gate')
    plt.ylabel('angle from incoming F2 trunk (deg)'); plt.title(f'Parent/daughter topology | daughter separation={daughter_angle:.1f}°'); plt.legend(); plt.tight_layout(); plt.savefig(out/'02_parent_daughter_angles.png', dpi=180); plt.close()

    fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(f2_con[:,0], f2_con[:,1], f2_con[:,2], c='k', lw=3, label='F2 shared trunk')
    for cid,p in [(6,c6),(9,c9),(7,c7)]:
        total=_arc(p)[-1]; q=np.linspace(split_arc,min(total,split_arc+7),40); s=_interp(p,q); ax.plot(s[:,0],s[:,1],s[:,2],lw=2,label=f'C{cid}')
    ax.scatter(f2_con[-1,0],f2_con[-1,1],f2_con[-1,2],c='k',s=45); ax.set_title('F2 parent continuation versus daughter branch'); ax.legend(); plt.tight_layout(); plt.savefig(out/'03_parent_daughter_geometry.png',dpi=180); plt.close()

    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for r,(cid,p) in enumerate([(6,c6),(7,c7)]):
        total=_arc(p)[-1]; qs=np.linspace(split_arc,min(total,split_arc+5),3)
        for j,qa in enumerate(qs):
            center=_interp(p,[qa])[0]; tangent=_vector_over(p,max(0,qa-.5),1.0); im,qv=_plane(g,src,center,tangent); ax=axes[r,j]; ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=25); ax.set_title(f'C{cid} +{qa-split_arc:.1f} mm'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    plt.tight_layout(); plt.savefig(out/'04_parent_side_branch_orthogonal_qc.png',dpi=180); plt.close()

    if not trunc['truncation_gate_pass']:
        status = STATUS_NOT_TRUNCATION
    elif topo['topology_gate_pass']:
        status = STATUS_PARENT
    else:
        status = STATUS_AMBIG

    summary = {
        'status': status, 'algorithm': ALGORITHM, 'baseline_commit': BASELINE,
        'master_status': master.get('status'), 'LCX_master_status': 'UNRESOLVED',
        'C6_C9_truncation': trunc, 'parent_daughter_topology': topo,
        'parent_representative_source_candidate_id': 6 if trunc['truncation_gate_pass'] else None,
        'side_branch_source_candidate_id': 7 if topo['topology_gate_pass'] else None,
        'template_similarity_used_in_decision': False,
        'prior_AV_groove_metrics_used_in_decision': False,
        'decision_rule': 'C9 must be a quantitative prefix/truncation of C6; C6 must continue the incoming F2 trunk <=25 deg while C7 branches >=35 deg, branch separation >=25 deg, parent/side angle difference >=20 deg, with >=0.90 dual-mask and HU support.',
        'scientific_boundary': 'This establishes source-space parent-versus-daughter topology if gates pass. It does not by itself establish clinical LCX or obtuse-marginal identity. LM remains unresolved.'
    }
    _write_json(out/'summary.json', summary)
    report = out/'OPENPLAQUE_LCX_PARENT_CONTINUATION_TOPOLOGY_REPORT.html'
    report.write_text(f"<html><body><h1>OpenPlaque LCX Parent Continuation Topology</h1><p><b>Status:</b> {status}</p><p>C9 truncation of C6: {trunc['truncation_gate_pass']}; p95 separation {trunc['p95_prefix_separation_mm']:.3f} mm; C6 extension {trunc['C6_extension_beyond_C9_mm']:.3f} mm.</p><p>C6 incoming angle {c6_ang:.2f}°, C7 incoming angle {c7_ang:.2f}°, daughter separation {daughter_angle:.2f}°.</p><p>Template and prior AV-groove scores have zero decision weight.</p></body></html>", encoding='utf-8')
    _write_json(out/'run_state.json', {'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE})
    zpath = out/'OPENPLAQUE_LCX_PARENT_CONTINUATION_TOPOLOGY_REPORT_BACK.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file(): z.write(p,p.name)
    return {'summary': summary, 'report': str(report), 'zip': str(zpath)}
