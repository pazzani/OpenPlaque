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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

BASELINE = '0593b453959f5a353d644267fbeef24b514ef4d7'
ALGORITHM = 'lcx-av-groove-locus-identity-v1.0-lowmem'
OUTPUT_DIRNAME = 'LCX_AV_Groove_Locus_Identity_v1'
SOURCE_CACHE = Path('Cache/Secondary_3D_Vesselness_Topology_v1')
MASTER = Path('Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json')
LAD_PATH = Path('Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv')
RCA_PATH = Path('PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv')
TS = Path('TotalSegmentator_Cardiovascular_Cache_v1')
HEART = TS / 'heartchambers_highres'
LA = HEART / 'heart_atrium_left.nii.gz'
LV = HEART / 'heart_ventricle_left.nii.gz'
RA = HEART / 'heart_atrium_right.nii.gz'
RV = HEART / 'heart_ventricle_right.nii.gz'
MYO = HEART / 'heart_myocardium.nii.gz'
COR_CURRENT = TS / 'coronary_arteries/coronary_arteries.nii.gz'
COR_LEGACY = TS / 'coronary_arteries_LEGACY/coronary_arteries.nii.gz'
PRIOR = Path('Joint_Three_Vessel_Template_Classifier_v1')
PRIOR_RANKING = PRIOR / 'LCX_joint_candidate_ranking.csv'
LEAF_FILES = [PRIOR / f'candidate_{i:02d}_source_path.csv' for i in range(1, 6)]
PARENT_RUN = Path('LCX_Parent_Continuation_Topology_v1/summary.json')

STATUS_PREREQ_FAIL = 'LCX_AV_GROOVE_LOCUS_PARENT_TOPOLOGY_PREREQUISITE_FAILED'
STATUS_CONTROL_FAIL = 'LCX_AV_GROOVE_LOCUS_CONTROL_FAILED'
STATUS_AMBIG = 'LCX_VS_OM_GROOVE_IDENTITY_AMBIGUOUS'
STATUS_C6 = 'C6_LCX_C7_OM_CANDIDATES_REQUIRE_VISUAL_QC'
STATUS_C7 = 'C7_LCX_C6_OM_CANDIDATES_REQUIRE_VISUAL_QC'


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

    def zyx_to_xyz(self, pts):
        idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
        return self.origin + (idx * self.spacing) @ self.direction.T


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


def _load_path(path, g):
    d = pd.read_csv(_req(path))
    for cols in [('lps_x_mm', 'lps_y_mm', 'lps_z_mm'), ('x_mm', 'y_mm', 'z_mm')]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [('zyx_z', 'zyx_y', 'zyx_x'), ('source_z', 'source_y', 'source_x'), ('z', 'y', 'x')]:
        if all(c in d.columns for c in cols):
            return g.zyx_to_xyz(d[list(cols)].to_numpy(float))
    raise ValueError(f'Unrecognized path columns in {path}: {list(d.columns)}')


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


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


def _img_zyx_to_xyz(im, pts):
    idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
    o = np.asarray(im.GetOrigin()); sp = np.asarray(im.GetSpacing()); d = np.asarray(im.GetDirection()).reshape(3, 3)
    return o + (idx * sp) @ d.T


def _surface_points(path, max_points=180000):
    im = sitk.ReadImage(str(_req(path)))
    work = sitk.LabelContour(sitk.Cast(im > 0, sitk.sitkUInt8), False)
    a = sitk.GetArrayViewFromImage(work)
    z = np.argwhere(a > 0)
    if not len(z):
        raise ValueError(f'No surface foreground in {path}')
    if len(z) > max_points:
        stride = int(math.ceil(len(z) / max_points))
        z = z[::stride]
    xyz = _img_zyx_to_xyz(work, z).astype(np.float32, copy=False)
    del z, a, work, im
    gc.collect()
    return xyz


def _mask_tree(path, max_points=300000):
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


def _dedup_points(points, voxel_mm=1.0):
    p = np.asarray(points, float)
    key = np.floor(p / voxel_mm).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return p[np.sort(idx)]


def _build_groove_cloud(myo_pts, atrium_pts, vent_pts, focus_pts, label):
    focus_tree = cKDTree(np.asarray(focus_pts, float))
    near = focus_tree.query(myo_pts)[0] <= 38.0
    local = np.asarray(myo_pts[near], float)
    if len(local) < 500:
        raise RuntimeError(f'{label}: too few local myocardium surface points ({len(local)})')
    ta, tv = cKDTree(atrium_pts), cKDTree(vent_pts)
    da = ta.query(local)[0]
    dv = tv.query(local)[0]
    balance = np.abs(da - dv)
    prox = np.maximum(da, dv)
    balance_cut = min(5.0, float(np.percentile(balance, 30)) + .75)
    balanced = balance <= balance_cut
    if balanced.sum() < 250:
        balance_cut = min(7.0, float(np.percentile(balance, 45)) + 1.0)
        balanced = balance <= balance_cut
    prox_pool = prox[balanced]
    prox_cut = min(28.0, float(np.percentile(prox_pool, 55)) + 1.5) if len(prox_pool) else 28.0
    keep = balanced & (prox <= prox_cut)
    cloud = local[keep]
    if len(cloud) < 180:
        rank = np.argsort(balance + .12 * prox)
        cloud = local[rank[:min(1200, max(180, len(local)//20))]]
    cloud = _dedup_points(cloud, 1.0)
    meta = {
        'label': label,
        'n_local_myocardium_surface_points': int(len(local)),
        'balance_cut_mm': float(balance_cut),
        'proximity_cut_mm': float(prox_cut),
        'n_groove_cloud_points': int(len(cloud)),
        'median_abs_atrium_minus_ventricle_distance_mm': float(np.median(balance[keep])) if keep.any() else float('nan'),
        'median_max_chamber_surface_distance_mm': float(np.median(prox[keep])) if keep.any() else float('nan'),
    }
    return cloud, meta


def _centerline_from_cloud(cloud, max_nodes=3200):
    p = _dedup_points(np.asarray(cloud, float), .8)
    if len(p) > max_nodes:
        center = np.median(p, axis=0)
        _, _, vh = np.linalg.svd(p - center, full_matrices=False)
        s = (p - center) @ vh[0]
        order = np.argsort(s)
        take = np.linspace(0, len(order)-1, max_nodes).round().astype(int)
        p = p[order[take]]
    center = np.median(p, axis=0)
    _, _, vh = np.linalg.svd(p - center, full_matrices=False)
    s = (p - center) @ vh[0]
    start = int(np.argmin(s)); end = int(np.argmax(s))
    tree = cKDTree(p)
    for max_edge in (3.5, 5.0, 7.0):
        dist, idx = tree.query(p, k=min(10, len(p)))
        rows = []; cols = []; vals = []
        for i in range(len(p)):
            for d, j in zip(dist[i, 1:], idx[i, 1:]):
                if np.isfinite(d) and d <= max_edge:
                    rows.extend([i, int(j)]); cols.extend([int(j), i]); vals.extend([float(d), float(d)])
        graph = csr_matrix((vals, (rows, cols)), shape=(len(p), len(p)))
        dd, pred = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
        if np.isfinite(dd[end]):
            chain = [end]
            cur = end
            while cur != start:
                cur = int(pred[cur])
                if cur < 0:
                    break
                chain.append(cur)
            if chain[-1] == start:
                line = p[np.array(chain[::-1])]
                line, q = _resample(line, .5)
                return line, {'n_cloud_nodes_used': int(len(p)), 'graph_max_edge_mm': float(max_edge), 'centerline_length_mm': float(q[-1])}
    raise RuntimeError('Could not construct connected groove centerline from cloud')


def _tangents(p):
    p = np.asarray(p, float)
    d = np.gradient(p, axis=0)
    n = np.linalg.norm(d, axis=1); n[n < 1e-9] = 1.0
    return d / n[:, None]


def _slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def _score_path(path, groove_line, atrium_tree, vent_tree, max_mm=None):
    p, q = _resample(path, .25)
    if max_mm is not None:
        keep = q <= min(q[-1], max_mm) + 1e-9
        p, q = p[keep], q[keep]
    gt = cKDTree(groove_line)
    dg, gi = gt.query(p)
    gv = _tangents(groove_line)
    pt = _tangents(p)
    align = np.abs(np.sum(pt * gv[gi], axis=1))
    da = atrium_tree.query(p)[0]
    dv = vent_tree.query(p)[0]
    balance = np.abs(da - dv)
    ds = _slope(q, dg)
    endpoint = float(dg[-1])
    med_d = float(np.median(dg))
    med_a = float(np.median(align))
    med_b = float(np.median(balance))
    distance_score = float(np.exp(-med_d / 8.0))
    retention = float(np.exp(-max(0.0, ds) / .45))
    balance_score = float(np.exp(-med_b / 8.0))
    score = float(.40 * distance_score + .30 * med_a + .20 * retention + .10 * balance_score)
    return {
        'groove_identity_score': score,
        'median_groove_locus_distance_mm': med_d,
        'p90_groove_locus_distance_mm': float(np.percentile(dg, 90)),
        'endpoint_groove_locus_distance_mm': endpoint,
        'groove_distance_slope_mm_per_mm': ds,
        'groove_retention_score': retention,
        'median_groove_tangent_alignment': med_a,
        'median_abs_atrium_minus_ventricle_distance_mm': med_b,
        'arc_mm': q,
        'groove_distance_profile_mm': dg,
        'alignment_profile': align,
        'points': p,
    }


def _support(tree, path):
    p, _ = _resample(path, .25)
    return float(np.mean(tree.query(p)[0] <= 1.0))


def _hu(src, g, path):
    p, _ = _resample(path, .25)
    h = _sample(src, g, p)
    return float(np.mean((h >= 120) & (h <= 1200))), float(np.median(h))


def _post_split(path, split_arc, max_mm=7.0):
    total = _arc(path)[-1]
    end = min(total, split_arc + max_mm)
    q = np.arange(split_arc, end + 1e-9, .25)
    if q[-1] < end - 1e-6:
        q = np.r_[q, end]
    return _interp(path, q)


def _plane(g, src, center, tangent, half=7., step=.2):
    t = np.asarray(tangent, float); t /= max(np.linalg.norm(t), 1e-9)
    axes = np.eye(3); seed = axes[np.argmin(np.abs(axes @ t))]
    n = np.cross(t, seed); n /= max(np.linalg.norm(n), 1e-9)
    b = np.cross(t, n); b /= max(np.linalg.norm(b), 1e-9)
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing='ij')
    pts = center[None, None, :] + xx[..., None] * n + yy[..., None] * b
    return _sample(src, g, pts.reshape(-1, 3)).reshape(len(q), len(q)), q


def synthetic_groove_locus_self_test():
    x = np.linspace(0, 20, 81)
    groove = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    atr = cKDTree(np.column_stack([x, np.full_like(x, 5.0), np.zeros_like(x)]))
    ven = cKDTree(np.column_stack([x, np.full_like(x, -5.0), np.zeros_like(x)]))
    follow = groove.copy()
    leave = groove.copy(); leave[:, 1] = np.linspace(0, 8, len(x))
    a = _score_path(follow, groove, atr, ven)
    b = _score_path(leave, groove, atr, ven)
    assert a['groove_identity_score'] > b['groove_identity_score'] + .10
    assert a['median_groove_locus_distance_mm'] < b['median_groove_locus_distance_mm']
    return {'ok': True, 'follow_score': a['groove_identity_score'], 'leave_score': b['groove_identity_score']}


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/'run_state.json', {'status':'STARTED','algorithm':ALGORITHM,'baseline_commit':BASELINE})

    required = [root/SOURCE_CACHE/'series7_int16.npy', root/SOURCE_CACHE/'series7_int16.json', root/MASTER,
                root/LAD_PATH, root/RCA_PATH, root/LA, root/LV, root/RA, root/RV, root/MYO,
                root/COR_CURRENT, root/COR_LEGACY, root/PRIOR_RANKING, root/PARENT_RUN] + [root/p for p in LEAF_FILES]
    for p in required:
        _req(p)

    master = json.loads((root/MASTER).read_text())
    parent = json.loads((root/PARENT_RUN).read_text())
    prereq = bool(parent.get('status') == 'C6_PARENT_CONTINUATION_C7_SIDE_BRANCH_REQUIRES_VISUAL_QC' and
                  parent.get('C6_C9_truncation', {}).get('truncation_gate_pass') and
                  parent.get('parent_daughter_topology', {}).get('topology_gate_pass'))
    if not prereq:
        summary = {'status':STATUS_PREREQ_FAIL,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'parent_summary':parent}
        _write_json(out/'summary.json', summary); _write_json(out/'run_state.json', {'status':'COMPLETE','scientific_status':STATUS_PREREQ_FAIL,'algorithm':ALGORITHM,'baseline_commit':BASELINE})
        return {'summary':summary,'report':None,'zip':None}

    g, src = _source(root/SOURCE_CACHE)
    ranking = pd.read_csv(root/PRIOR_RANKING)
    leaves = _orient([_load_path(root/p, g) for p in LEAF_FILES])
    mapping = {}
    for i in range(1, 6):
        cid = int(ranking.iloc[i-1]['candidate_id'])
        mapping[cid] = leaves[i-1]
    c6, c7 = mapping[6], mapping[7]
    split_arc = float(parent['parent_daughter_topology']['family2_split_arc_from_candidate_origin_mm'])
    c6_post = _post_split(c6, split_arc, 7.0)
    c7_post = _post_split(c7, split_arc, 7.0)
    lad = _load_path(root/LAD_PATH, g)
    rca = _load_path(root/RCA_PATH, g)

    print('Loading sparse chamber/myocardium surfaces...')
    la_pts = _surface_points(root/LA); lv_pts = _surface_points(root/LV)
    ra_pts = _surface_points(root/RA); rv_pts = _surface_points(root/RV)
    myo_pts = _surface_points(root/MYO, 240000)
    tla, tlv, tra, trv = cKDTree(la_pts), cKDTree(lv_pts), cKDTree(ra_pts), cKDTree(rv_pts)

    left_focus = np.vstack([_resample(lad,.75)[0], _resample(c6,.50)[0], _resample(c7,.50)[0]])
    right_focus = _resample(rca,.50)[0]
    left_cloud, left_meta = _build_groove_cloud(myo_pts, la_pts, lv_pts, left_focus, 'left_AV')
    right_cloud, right_meta = _build_groove_cloud(myo_pts, ra_pts, rv_pts, right_focus, 'right_AV')
    left_line, left_line_meta = _centerline_from_cloud(left_cloud)
    right_line, right_line_meta = _centerline_from_cloud(right_cloud)
    pd.DataFrame(left_cloud, columns=['lps_x_mm','lps_y_mm','lps_z_mm']).to_csv(out/'left_av_groove_locus_cloud.csv', index=False)
    pd.DataFrame(left_line, columns=['lps_x_mm','lps_y_mm','lps_z_mm']).assign(arc_mm=_arc(left_line)).to_csv(out/'left_av_groove_centerline.csv', index=False)
    pd.DataFrame(right_line, columns=['lps_x_mm','lps_y_mm','lps_z_mm']).assign(arc_mm=_arc(right_line)).to_csv(out/'right_av_groove_centerline_control.csv', index=False)
    _write_json(out/'groove_locus_build_summary.json', {'left_cloud':left_meta,'left_centerline':left_line_meta,'right_cloud':right_meta,'right_centerline':right_line_meta})

    rca_m = _score_path(rca, right_line, tra, trv, max_mm=45.0)
    lad_m = _score_path(lad, left_line, tla, tlv, max_mm=22.0)
    control_margin = float(rca_m['groove_identity_score'] - lad_m['groove_identity_score'])
    control_pass = bool(rca_m['groove_identity_score'] >= .50 and rca_m['median_groove_locus_distance_mm'] <= 12.0 and control_margin >= .10)
    controls = {
        'RCA_right_AV_groove_score':rca_m['groove_identity_score'],
        'RCA_median_groove_locus_distance_mm':rca_m['median_groove_locus_distance_mm'],
        'RCA_median_tangent_alignment':rca_m['median_groove_tangent_alignment'],
        'LAD_left_AV_negative_score':lad_m['groove_identity_score'],
        'LAD_median_groove_locus_distance_mm':lad_m['median_groove_locus_distance_mm'],
        'LAD_median_tangent_alignment':lad_m['median_groove_tangent_alignment'],
        'control_margin':control_margin,
        'control_pass':control_pass,
    }
    _write_json(out/'groove_locus_controls.json', controls)

    c6_m = _score_path(c6_post, left_line, tla, tlv)
    c7_m = _score_path(c7_post, left_line, tla, tlv)
    cur = _mask_tree(root/COR_CURRENT); leg = _mask_tree(root/COR_LEGACY)
    rows = []
    for cid, path, m in [(6,c6_post,c6_m),(7,c7_post,c7_m)]:
        cs = _support(cur,path); ls = _support(leg,path); hf, hm = _hu(src,g,path)
        row = {k:v for k,v in m.items() if not isinstance(v,np.ndarray)}
        row.update({'source_candidate_id':cid,'current_support_fraction':cs,'legacy_support_fraction':ls,'robust_hu_fraction':hf,'median_hu':hm})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('groove_identity_score',ascending=False).reset_index(drop=True)
    df.insert(0,'rank',np.arange(1,len(df)+1)); df.to_csv(out/'C6_C7_groove_identity_scores.csv',index=False)

    win = df.iloc[0]; lose = df.iloc[1]
    margin = float(win.groove_identity_score - lose.groove_identity_score)
    distance_adv = float(lose.median_groove_locus_distance_mm - win.median_groove_locus_distance_mm)
    align_adv = float(win.median_groove_tangent_alignment - lose.median_groove_tangent_alignment)
    departure_adv = float(lose.groove_distance_slope_mm_per_mm - win.groove_distance_slope_mm_per_mm)
    endpoint_adv = float(lose.endpoint_groove_locus_distance_mm - win.endpoint_groove_locus_distance_mm)
    support_pass = bool(min(win.current_support_fraction,win.legacy_support_fraction,win.robust_hu_fraction,lose.current_support_fraction,lose.legacy_support_fraction,lose.robust_hu_fraction) >= .90)
    winner_gate = bool(control_pass and margin >= .08 and win.groove_identity_score >= lad_m['groove_identity_score'] + .10 and
                       win.groove_retention_score >= .55 and support_pass and
                       (distance_adv >= 1.5 or align_adv >= .15) and
                       (departure_adv >= .20 or endpoint_adv >= 2.0))
    decision = {
        'winner_source_candidate_id':int(win.source_candidate_id),
        'loser_source_candidate_id':int(lose.source_candidate_id),
        'score_margin':margin,
        'median_groove_distance_advantage_mm':distance_adv,
        'tangent_alignment_advantage':align_adv,
        'departure_slope_advantage_mm_per_mm':departure_adv,
        'endpoint_groove_distance_advantage_mm':endpoint_adv,
        'dual_mask_and_HU_support_pass':support_pass,
        'identity_gate_pass':winner_gate,
    }
    _write_json(out/'LCX_vs_OM_identity_decision.json', decision)

    # Visual QC
    plt.figure(figsize=(7,4))
    plt.bar(['RCA +control','LAD -control','C6','C7'], [rca_m['groove_identity_score'],lad_m['groove_identity_score'],c6_m['groove_identity_score'],c7_m['groove_identity_score']])
    plt.ylabel('explicit groove-locus identity score'); plt.title(f'AV-groove locus identity | candidate margin={margin:.3f}'); plt.tight_layout(); plt.savefig(out/'01_groove_identity_scores.png',dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    for cid,m in [(6,c6_m),(7,c7_m)]: plt.plot(m['arc_mm'],m['groove_distance_profile_mm'],marker='o',ms=2,label=f'C{cid}')
    plt.xlabel('arc after F2 split (mm)'); plt.ylabel('distance to explicit left AV-groove locus (mm)'); plt.legend(); plt.tight_layout(); plt.savefig(out/'02_groove_distance_profiles.png',dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    for cid,m in [(6,c6_m),(7,c7_m)]: plt.plot(m['arc_mm'],m['alignment_profile'],marker='o',ms=2,label=f'C{cid}')
    plt.xlabel('arc after F2 split (mm)'); plt.ylabel('|tangent dot groove-locus tangent|'); plt.ylim(0,1.02); plt.legend(); plt.tight_layout(); plt.savefig(out/'03_groove_tangent_alignment_profiles.png',dpi=180); plt.close()

    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
    show = left_cloud[::max(1,len(left_cloud)//1200)]
    ax.scatter(show[:,0],show[:,1],show[:,2],s=2,alpha=.18,label='LA-LV surface-derived groove locus')
    ax.plot(left_line[:,0],left_line[:,1],left_line[:,2],c='k',lw=3,label='explicit groove centerline')
    for cid,p in [(6,c6_post),(7,c7_post)]: ax.plot(p[:,0],p[:,1],p[:,2],lw=3,label=f'C{cid}')
    ax.set_title('Explicit left AV-groove locus vs C6/C7'); ax.legend(); plt.tight_layout(); plt.savefig(out/'04_left_av_groove_locus_geometry.png',dpi=180); plt.close()

    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for r,(cid,p) in enumerate([(6,c6_post),(7,c7_post)]):
        a=_arc(p); qs=np.linspace(0,a[-1],3)
        for j,qa in enumerate(qs):
            pts=_interp(p,[max(0,qa-.5),qa,min(a[-1],qa+.5)]); t=pts[-1]-pts[0]
            im,qv=_plane(g,src,pts[1],t); ax=axes[r,j]; ax.imshow(im,cmap='gray',vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=25); ax.set_title(f'C{cid} +{qa:.1f} mm'); ax.set_xlabel('mm'); ax.set_ylabel('mm')
    plt.tight_layout(); plt.savefig(out/'05_C6_C7_orthogonal_source_qc.png',dpi=180); plt.close()

    if not control_pass:
        status = STATUS_CONTROL_FAIL
    elif not winner_gate:
        status = STATUS_AMBIG
    else:
        status = STATUS_C6 if int(win.source_candidate_id) == 6 else STATUS_C7

    summary = {
        'status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE,'master_status':master.get('status'),'LCX_master_status':'UNRESOLVED',
        'parent_topology_prerequisite_pass':prereq,'split_arc_mm':split_arc,'groove_locus_controls':controls,
        'C6_metrics':{k:v for k,v in c6_m.items() if not isinstance(v,np.ndarray)},
        'C7_metrics':{k:v for k,v in c7_m.items() if not isinstance(v,np.ndarray)},
        'decision':decision,
        'template_similarity_used_in_decision':False,
        'prior_local_AV_groove_score_used_in_decision':False,
        'decision_rule':'Explicit LA-LV surface-derived groove locus. Positive RCA/right-AV and negative LAD/left-AV controls must pass. Winning daughter needs >=0.08 score margin, >=0.10 over LAD negative, groove retention >=0.55, >=0.90 dual-mask/HU support, plus either >=1.5 mm median locus-distance or >=0.15 tangent-alignment advantage and explicit downstream departure evidence (>=0.20 mm/mm slope or >=2 mm endpoint distance advantage).',
        'scientific_boundary':'This can nominate an LCX-like groove-following daughter and an OM-like groove-departing daughter after source-space parent/daughter topology is established. It does not establish clinical vessel identity; LM remains unresolved.'
    }
    _write_json(out/'summary.json',summary)
    report=out/'OPENPLAQUE_LCX_AV_GROOVE_LOCUS_IDENTITY_REPORT.html'
    report.write_text(f"<html><body><h1>OpenPlaque LCX vs OM Explicit AV-Groove Locus Identity</h1><p><b>Status:</b> {status}</p><p>Control: RCA {rca_m['groove_identity_score']:.3f}, LAD {lad_m['groove_identity_score']:.3f}, margin {control_margin:.3f}, pass {control_pass}.</p><p>C6 score {c6_m['groove_identity_score']:.3f}; C7 score {c7_m['groove_identity_score']:.3f}; margin {margin:.3f}; identity gate {winner_gate}.</p><p>Template similarity and prior local AV-groove scores have zero decision weight.</p></body></html>",encoding='utf-8')
    _write_json(out/'run_state.json',{'status':'COMPLETE','scientific_status':status,'algorithm':ALGORITHM,'baseline_commit':BASELINE})
    zpath=out/'OPENPLAQUE_LCX_AV_GROOVE_LOCUS_IDENTITY_REPORT_BACK.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file(): z.write(p,p.name)
    return {'summary':summary,'report':str(report),'zip':str(zpath)}
