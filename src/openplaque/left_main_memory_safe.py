from __future__ import annotations

"""Memory-safe replacements for the staged left-main experiment.

This module keeps the v1 anatomy/QC logic but avoids full-volume coordinate grids
and full-volume MCP allocations. Candidate discovery uses sparse argwhere lists,
and each graph search is restricted to a local physical bounding box around the
start point and candidate endpoints.
"""

import numpy as np
import pandas as pd
from skimage.graph import MCP_Geometric

from . import left_main_coronary as base

ALGORITHM_VERSION = "left-main-bifurcation-v1.1-memory-safe"


def detect_left_ostia(evd, rca_seed_ds, topn=18, preselect=1200):
    support, ct, aorta = evd["support"], evd["ct"], evd["aorta"]
    da, lr, sp = evd["dist_aorta_mm"], evd["lumen_radius_mm"], np.asarray(evd["spacing_zyx"], float)
    rca_seed_ds = np.asarray(rca_seed_ds, float)
    ctr = base._aorta_center_at_z(aorta, rca_seed_ds[0])
    rca_vec = (rca_seed_ds - ctr) * sp
    rca_vec[0] *= 0.35
    rca_vec /= max(np.linalg.norm(rca_vec), 1e-6)

    # No np.indices(): z restriction is broadcast from a 1-D vector.
    z_ok = np.abs((np.arange(aorta.shape[0], dtype=np.float32) - rca_seed_ds[0]) * sp[0]) <= 15.0
    shell = ((~aorta) & (da >= 1.2) & (da <= 5.8) & (ct >= 150) &
             (ct <= 950) & (lr <= 4.8) & z_ok[:, None, None])
    if not np.any(shell):
        raise RuntimeError("No aortic-wall voxels available for left ostium search")
    sth = float(np.percentile(support[shell], 65))
    shell &= support >= sth
    coords = np.argwhere(shell)
    del shell
    if len(coords) == 0:
        raise RuntimeError("No viable left ostium candidates")

    # Vectorize the cheap radial/opposition screen and ray-sample only a compact shortlist.
    centers = {}
    for z in np.unique(coords[:, 0]):
        centers[int(z)] = base._aorta_center_at_z(aorta, int(z))
    C = np.vstack([centers[int(z)] for z in coords[:, 0]])
    rv = (coords.astype(np.float32) - C.astype(np.float32)) * sp.astype(np.float32)
    rv[:, 0] *= 0.35
    norms = np.linalg.norm(rv, axis=1)
    valid = norms >= 2.0
    U = np.zeros_like(rv, dtype=np.float32)
    U[valid] = rv[valid] / norms[valid, None]
    opposite = -(U @ rca_vec.astype(np.float32))
    valid &= opposite >= -0.05
    if not np.any(valid):
        raise RuntimeError("No left ostium candidates survive opposite-side screening")
    coords = coords[valid]
    U = U[valid]
    opposite = opposite[valid]
    quick = (0.62 * support[tuple(coords.T)].astype(np.float32) +
             0.28 * opposite.astype(np.float32) +
             0.10 * np.clip((ct[tuple(coords.T)] - 150.0) / 500.0, 0, 1).astype(np.float32))
    if len(coords) > preselect:
        take = np.argpartition(quick, -preselect)[-preselect:]
        coords, U, opposite, quick = coords[take], U[take], opposite[take], quick[take]

    rows = []
    shape = np.asarray(ct.shape)
    for p, u, opp in zip(coords, U, opposite):
        rs, rh = [], []
        pmm = p.astype(float) * sp
        for mm in np.linspace(2, 14, 9):
            q = (pmm + u.astype(float) * mm) / sp
            if np.any(q < 1) or np.any(q >= shape - 2):
                continue
            rs.append(float(base.map_coordinates(support, q[:, None], order=1, mode="nearest")[0]))
            rh.append(float(base.map_coordinates(ct, q[:, None], order=1, mode="nearest")[0]))
        if len(rs) < 6:
            continue
        rs, rh = np.asarray(rs), np.asarray(rh)
        plausible = float(np.mean((rh >= 120) & (rh <= 950)))
        score = (0.44 * float(np.mean(rs)) + 0.22 * float(np.percentile(rs, 25)) +
                 0.18 * float(opp) + 0.11 * plausible + 0.05 * float(support[tuple(p)]))
        rows.append({"score": score, "z": float(p[0]), "y": float(p[1]), "x": float(p[2]),
                     "dir_z": float(u[0]), "dir_y": float(u[1]), "dir_x": float(u[2]),
                     "opposite_rca": float(opp), "outward_mean_support": float(np.mean(rs)),
                     "outward_plausible_hu": plausible})
    if not rows:
        raise RuntimeError("No viable left ostium candidates after outward-ray test")
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(topn).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _endpoint_candidates_sparse(evd, start, heading, min_r, max_r, min_forward,
                                max_endpoints=40, support_percentile=78):
    support, da, lr, ct = evd["support"], evd["dist_aorta_mm"], evd["lumen_radius_mm"], evd["ct"]
    sp = np.asarray(evd["spacing_zyx"], float)
    start = np.asarray(start, float)
    half = np.ceil((float(max_r) + 6.0) / sp).astype(int)
    lo = np.maximum(0, np.floor(start).astype(int) - half)
    hi = np.minimum(np.asarray(support.shape), np.ceil(start).astype(int) + half + 1)
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))

    s = support[sl]; dd = da[sl]; rr = lr[sl]; hv = ct[sl]
    pos = support[support > 0]
    sth = float(np.percentile(pos, support_percentile)) if pos.size else 0.2
    basic = (s >= sth) & (dd >= 1.5) & (rr <= 5.0) & (hv >= 120) & (hv <= 1000)
    loc = np.argwhere(basic)
    del basic
    if len(loc) == 0:
        return []
    cand = loc + lo[None, :]
    D = (cand.astype(np.float32) - start.astype(np.float32)[None, :]) * sp.astype(np.float32)[None, :]
    rad = np.linalg.norm(D, axis=1)
    h = np.asarray(heading, float); h /= max(np.linalg.norm(h), 1e-6)
    forward = D @ h.astype(np.float32)
    keep = (rad >= min_r) & (rad <= max_r) & (forward >= min_forward)
    if not np.any(keep):
        return []
    cand, forward = cand[keep], forward[keep]
    su = support[tuple(cand.T)]; dval = da[tuple(cand.T)]; rval = lr[tuple(cand.T)]
    val = su + 0.025 * np.minimum(dval, 16) - 0.08 * np.maximum(rval - 3.5, 0) + 0.002 * np.minimum(forward, 80)
    order = np.argsort(val)[::-1]
    eps = []
    for j in order:
        p = cand[j]
        if all(np.linalg.norm((p - q) * sp) >= 4.0 for q in eps):
            eps.append(p)
        if len(eps) >= max_endpoints:
            break
    return eps


def _mcp_paths_local(cost, start, endpoints, spacing_zyx, pad_mm=10.0):
    if not endpoints:
        return []
    sp = np.asarray(spacing_zyx, float)
    pts = np.vstack([np.rint(start).astype(int), np.asarray(endpoints, int)])
    pad = np.ceil(float(pad_mm) / sp).astype(int)
    lo = np.maximum(0, pts.min(axis=0) - pad)
    hi = np.minimum(np.asarray(cost.shape), pts.max(axis=0) + pad + 1)
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    # Keep float32 and restrict Dijkstra state to the physically relevant box.
    local_cost = np.ascontiguousarray(cost[sl], dtype=np.float32)
    local_start = np.rint(start).astype(int) - lo
    local_eps = [np.asarray(ep, int) - lo for ep in endpoints]
    mcp = MCP_Geometric(local_cost, fully_connected=True)
    mcp.find_costs([tuple(local_start)])
    out = []
    for ep in local_eps:
        try:
            p = np.asarray(mcp.traceback(tuple(ep)), int) + lo[None, :]
        except Exception:
            continue
        if len(p) >= 2:
            out.append(p)
    return out


def trace_routes(evd, start_ds, heading_mm, min_len, max_len, max_routes=30,
                 min_forward=3.0, support_percentile=78):
    eps = _endpoint_candidates_sparse(evd, start_ds, heading_mm, min_len, max_len,
                                      min_forward, max_endpoints=max(40, max_routes * 2),
                                      support_percentile=support_percentile)
    if not eps:
        return []
    pad = 9.0 if max_len <= 35 else 16.0
    routes = []
    for p in _mcp_paths_local(evd["cost"], start_ds, eps, evd["spacing_zyx"], pad_mm=pad):
        m = base._path_metrics(p, evd, heading_mm)
        if not (min_len <= m["length_mm"] <= max_len):
            continue
        if m["aorta_reentry_fraction"] > 0.04 or m["large_lumen_fraction"] > 0.18:
            continue
        score = (1.9 * m["mean_support"] + 0.65 * m["p10_support"] +
                 0.010 * min(m["length_mm"], 90) + 0.020 * min(m["end_dist_aorta_mm"], 18) -
                 0.34 * m["mean_turn_rad"] - 0.11 * m["p95_turn_rad"] -
                 0.005 * m["heading_angle_deg"] - 1.1 * m["large_lumen_fraction"])
        routes.append({"path": p, "graph_score": float(score), **m})
    routes.sort(key=lambda r: r["graph_score"], reverse=True)
    return routes[:max_routes]


def score_left_main_candidates(evd, ostia_df, ct_source, spacing_source, rca_calibration,
                               lo_source_zyx, zoom_zyx, max_ostia=10):
    candidates, rows = [], []
    for _, o in ostia_df.head(max_ostia).iterrows():
        seed = np.array([o.z, o.y, o.x], float)
        heading = np.array([o.dir_z, o.dir_y, o.dir_x], float)
        routes = trace_routes(evd, seed, heading, min_len=6, max_len=30, max_routes=14,
                              min_forward=2.5, support_percentile=72)
        for r in routes:
            src = base.ds_to_source(r["path"], lo_source_zyx, zoom_zyx)
            qcdf, qcs = base.serial_lumen_qc(src, ct_source, spacing_source, rca_calibration,
                                             n_samples=9, label_name="left_main")
            length_quality = float(np.exp(-0.5 * ((r["length_mm"] - 14) / 8.0) ** 2))
            score = (0.42 * r["graph_score"] + 0.34 * qcs["median_plane_score"] +
                     0.19 * qcs["plane_pass_fraction"] + 0.05 * length_quality)
            rec = {"ostium_rank": int(o["rank"]), "combined_score": float(score),
                   **{k: v for k, v in r.items() if k != "path"},
                   **{f"serial_{k}": v for k, v in qcs.items() if k != "label"}}
            rows.append(rec)
            candidates.append({"path_ds": r["path"], "path_source": src, "qc_df": qcdf, "summary": rec})
    if not candidates:
        raise RuntimeError("No left-main route candidates")
    order = np.argsort([c["summary"]["combined_score"] for c in candidates])[::-1]
    candidates = [candidates[i] for i in order]
    table = pd.DataFrame(rows).sort_values("combined_score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return candidates, table


def select_branch_pair_from_trunk(evd, trunk_source, ct_source, spacing_source,
                                  rca_calibration, image, lo_source_zyx, zoom_zyx,
                                  max_routes=22):
    p = base.resample_path(trunk_source, spacing_source, 0.7)
    i0 = max(0, len(p) - 8)
    heading = (p[-1] - p[i0]) * np.asarray(spacing_source)
    heading /= max(np.linalg.norm(heading), 1e-6)
    seed_ds = base.source_to_ds(p[-1], lo_source_zyx, zoom_zyx)
    routes = trace_routes(evd, seed_ds, heading, min_len=28, max_len=135,
                          max_routes=max_routes, min_forward=6, support_percentile=78)
    enriched = []
    for r in routes:
        src = base.ds_to_source(r["path"], lo_source_zyx, zoom_zyx)
        qcdf, qcs = base.serial_lumen_qc(src, ct_source, spacing_source, rca_calibration,
                                         n_samples=11, label_name="branch")
        r2 = dict(r); r2["path_source"] = src; r2["qc_df"] = qcdf; r2["qc_summary"] = qcs
        r2["branch_score"] = float(0.55 * r["graph_score"] + 0.27 * qcs["median_plane_score"] + 0.18 * qcs["plane_pass_fraction"])
        enriched.append(r2)
    enriched.sort(key=lambda x: x["branch_score"], reverse=True)
    pairs = []
    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            a, b = enriched[i], enriched[j]
            common = base._common_length_mm(a["path"], b["path"], evd["spacing_zyx"], tol_mm=3.0)
            sep = float(np.linalg.norm((a["path"][-1] - b["path"][-1]) * evd["spacing_zyx"]))
            if common > 18 or sep < 24:
                continue
            if min(a["qc_summary"]["plane_pass_fraction"], b["qc_summary"]["plane_pass_fraction"]) < 0.40:
                continue
            divergence_quality = float(np.exp(-0.5 * ((common - 5) / 6.5) ** 2))
            score = a["branch_score"] + b["branch_score"] + 0.012 * min(sep, 80) + 0.20 * divergence_quality
            pairs.append((score, common, sep, a, b))
    if not pairs:
        return None, enriched
    pairs.sort(key=lambda x: x[0], reverse=True)
    score, common, sep, a, b = pairs[0]
    start_ph = base.physical_lps_from_zyx(image, [trunk_source[-1]])[0]
    pha = base.physical_lps_from_zyx(image, [a["path_source"][-1]])[0]
    phb = base.physical_lps_from_zyx(image, [b["path_source"][-1]])[0]
    da, db = pha - start_ph, phb - start_ph
    lad_a = -1.00 * da[2] - 0.45 * da[1] + 0.10 * da[0]
    lad_b = -1.00 * db[2] - 0.45 * db[1] + 0.10 * db[0]
    lad, lcx = (a, b) if lad_a >= lad_b else (b, a)
    return {"pair_score": float(score), "common_after_trunk_mm": float(common),
            "endpoint_separation_mm": float(sep), "LAD": lad, "LCX": lcx,
            "lad_metric_a": float(lad_a), "lad_metric_b": float(lad_b)}, enriched


def patch_base_module():
    """Patch the symbols imported by the workflow before that workflow is imported."""
    base.ALGORITHM_VERSION = ALGORITHM_VERSION
    base.detect_left_ostia = detect_left_ostia
    base.trace_routes = trace_routes
    base.score_left_main_candidates = score_left_main_candidates
    base.select_branch_pair_from_trunk = select_branch_pair_from_trunk
