from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import lcx_local_av_junction_identity as base

ALGORITHM = 'lcx-local-av-junction-ribbon-v1.1-robust'


def _robust_local_ribbon(anchor, atrium_tree, ventricle_tree, myocardium_tree, radius=20.0):
    anchor = np.asarray(anchor, float)
    pts = None
    used_radius = None
    for r in (radius, 25.0, 30.0, 35.0):
        ids = myocardium_tree.query_ball_point(anchor, r)
        if len(ids) >= 100:
            pts = np.asarray(myocardium_tree.data[np.asarray(ids, dtype=int)], float)
            used_radius = float(r)
            break
    if pts is None:
        k = min(1200, len(myocardium_tree.data))
        _, ids = myocardium_tree.query(anchor, k=k)
        ids = np.atleast_1d(ids).astype(int)
        pts = np.asarray(myocardium_tree.data[ids], float)
        used_radius = float(np.max(np.linalg.norm(pts - anchor[None, :], axis=1))) if len(pts) else float('nan')
    if len(pts) < 20:
        raise RuntimeError(f'Only {len(pts)} myocardial surface points available near anchor after fallback')

    da = atrium_tree.query(pts)[0]
    dv = ventricle_tree.query(pts)[0]
    chosen = None
    used = None
    for balance, prox in [(2.5,7.0),(3.5,8.5),(5.0,10.0),(6.5,12.0),(8.0,15.0)]:
        m = (np.abs(da-dv) <= balance) & (np.maximum(da,dv) <= prox)
        if int(m.sum()) >= 40:
            chosen = pts[m]
            used = (balance, prox, 'threshold')
            break
    if chosen is None:
        score = np.abs(da-dv) + 0.35*np.maximum(da,dv)
        ix = np.argsort(score)[:min(300, len(pts))]
        chosen = pts[ix]
        used = (float(np.max(np.abs(da[ix]-dv[ix]))), float(np.max(np.maximum(da[ix],dv[ix]))), 'ranked_fallback')

    key = np.round(chosen / .7).astype(int)
    _, ix = np.unique(key, axis=0, return_index=True)
    chosen = chosen[np.sort(ix)]
    if len(chosen) < 12:
        raise RuntimeError(f'Only {len(chosen)} unique ribbon points after fallback')
    tree = cKDTree(chosen.astype(np.float32, copy=False))
    return tree, {
        'anchor_lps_mm': [float(x) for x in anchor],
        'requested_radius_mm': float(radius),
        'used_radius_mm': used_radius,
        'n_local_myocardium_points': int(len(pts)),
        'n_ribbon_points': int(len(chosen)),
        'balance_cut_mm': float(used[0]),
        'proximity_cut_mm': float(used[1]),
        'selection_mode': used[2],
    }


def run(drive_root='/content/drive/MyDrive/OpenPlaque', output_dir=None):
    out = Path(output_dir) if output_dir else Path(drive_root) / 'LCX_Local_AV_Junction_Ribbon_v1_robust'
    out.mkdir(parents=True, exist_ok=True)
    base._local_ribbon = _robust_local_ribbon
    try:
        result = base.run(drive_root, str(out))
        result['summary']['robust_runner_algorithm'] = ALGORITHM
        (out / 'summary.json').write_text(json.dumps(result['summary'], indent=2, default=str), encoding='utf-8')
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        (out / 'failure_traceback.txt').write_text(tb, encoding='utf-8')
        (out / 'run_state.json').write_text(json.dumps({
            'status': 'FAILED',
            'algorithm': ALGORITHM,
            'exception_type': type(exc).__name__,
            'exception_message': str(exc),
        }, indent=2), encoding='utf-8')
        print(tb, flush=True)
        raise
