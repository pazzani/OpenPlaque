from __future__ import annotations
import gc, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-hierarchical-distal-tree-v1.0-lowmem"
OUTPUT_DIRNAME = "LCX_Hierarchical_Distal_Tree_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
HEART = TS / "heartchambers_highres"
COR_CURRENT = TS / "coronary_arteries/coronary_arteries.nii.gz"
COR_LEGACY = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
LA = HEART / "heart_atrium_left.nii.gz"
LV = HEART / "heart_ventricle_left.nii.gz"
RA = HEART / "heart_atrium_right.nii.gz"
RV = HEART / "heart_ventricle_right.nii.gz"
MYO = HEART / "heart_myocardium.nii.gz"

PRIOR = Path("Joint_Three_Vessel_Template_Classifier_v1")
PRIOR_RANKING = PRIOR / "LCX_joint_candidate_ranking.csv"
LEAF_FILES = [PRIOR / f"candidate_{i:02d}_source_path.csv" for i in range(1, 6)]

STATUS_NO_TRUNK = "NO_CONSENSUS_LEFT_CORONARY_BRANCH_TRUNK"
STATUS_CONTROL_FAIL = "HIERARCHICAL_AV_GROOVE_CONTROL_FAILED"
STATUS_AMBIG = "CONSENSUS_TRUNK_ESTABLISHED_HIERARCHICAL_LCX_AMBIGUOUS"
STATUS_CANDIDATE = "HIERARCHICAL_LCX_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC"

def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")

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
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    m = json.loads(_req(cache / "series7_int16.json").read_text())
    sp = np.asarray(m["spacing_zyx"], float)[::-1]
    iop = np.asarray(m["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    d = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    return Geometry(np.asarray(m["positions_lps_mm"][0], float), sp, d), arr

def _sample(arr, g, pts, order=1, cval=0.0):
    return map_coordinates(arr, g.xyz_to_zyx(pts).T, order=order, mode="constant", cval=cval)

def _load_path(path, g):
    d = pd.read_csv(_req(path))
    for cols in [("lps_x_mm","lps_y_mm","lps_z_mm"), ("x_mm","y_mm","z_mm")]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [("zyx_z","zyx_y","zyx_x"), ("source_z","source_y","source_x"), ("z","y","x")]:
        if all(c in d.columns for c in cols):
            return g.zyx_to_xyz(d[list(cols)].to_numpy(float))
    raise ValueError(f"Unrecognized path columns: {list(d.columns)}")

def _img_zyx_to_xyz(im, pts):
    idx = np.atleast_2d(np.asarray(pts, float))[:, ::-1]
    o = np.asarray(im.GetOrigin(), float)
    sp = np.asarray(im.GetSpacing(), float)
    d = np.asarray(im.GetDirection(), float).reshape(3, 3)
    return o + (idx * sp) @ d.T

def _tree(path, surface=False, max_points=250000):
    im = sitk.ReadImage(str(_req(path)))
    work = sitk.LabelContour(sitk.Cast(im > 0, sitk.sitkUInt8), False) if surface else sitk.Cast(im > 0, sitk.sitkUInt8)
    a = sitk.GetArrayViewFromImage(work)
    z = np.argwhere(a > 0)
    if not len(z):
        raise ValueError(f"No foreground in {path}")
    if len(z) > max_points:
        z = z[::int(math.ceil(len(z) / max_points))]
    xyz = _img_zyx_to_xyz(work, z).astype(np.float32, copy=False)
    del z, a, work, im
    gc.collect()
    return cKDTree(xyz)

def _arc(points):
    p = np.asarray(points, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

def _interp_path(points, q):
    p = np.asarray(points, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])

def _resample_path(points, step=0.25):
    p = np.asarray(points, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p, a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp_path(p, q), q

def _orient_common(paths):
    ref = np.asarray(paths[0], float)[0]
    out = []
    for p in paths:
        p = np.asarray(p, float)
        if np.linalg.norm(p[-1] - ref) < np.linalg.norm(p[0] - ref):
            p = p[::-1].copy()
        out.append(p)
    return out

def _consensus_prefix(paths, step=0.25, tol_mm=0.60, sustain=4):
    paths = _orient_common(paths)
    minlen = min(_arc(p)[-1] for p in paths)
    q = np.arange(0, minlen + 1e-9, step)
    samples = np.stack([_interp_path(p, q) for p in paths], axis=0)
    centroid = np.median(samples, axis=0)
    dev = np.max(np.linalg.norm(samples - centroid[None, :, :], axis=2), axis=0)
    first_bad = None
    start = max(1, int(round(5.0 / step)))
    for i in range(start, max(start, len(q) - sustain + 1)):
        if np.all(dev[i:i+sustain] > tol_mm):
            first_bad = i
            break
    end_i = first_bad - 1 if first_bad is not None else len(q) - 1
    return centroid[:end_i+1], q[:end_i+1], dev[:end_i+1], {
        "common_prefix_total_mm": float(q[end_i]),
        "prefix_tolerance_mm": float(tol_mm),
        "prefix_max_deviation_mm": float(np.max(dev[:end_i+1])),
        "first_sustained_split_arc_mm": None if first_bad is None else float(q[first_bad]),
    }

def _nearest_dist(points, ref):
    return cKDTree(np.asarray(ref, float)).query(np.asarray(points, float))[0]

def _first_sustained(arr, threshold, step, duration_mm=1.0):
    arr = np.asarray(arr, float)
    n = max(1, int(round(duration_mm / step)))
    for i in range(0, len(arr) - n + 1):
        if np.all(arr[i:i+n] >= threshold):
            return i
    return None

def _union_find_groups(points, threshold=1.5):
    points = np.asarray(points, float)
    n = len(points)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= threshold:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

def _dist_normal(tree, pts):
    p = np.asarray(pts, float)
    d, idx = tree.query(p)
    near = np.asarray(tree.data[idx], float)
    v = p - near
    n = np.linalg.norm(v, axis=1)
    good = n > 0.15
    out = np.zeros_like(v)
    out[good] = v[good] / n[good, None]
    return d, out, good

def _tangents(p):
    p = np.asarray(p, float)
    d = np.gradient(p, axis=0)
    n = np.linalg.norm(d, axis=1)
    n[n < 1e-9] = 1.0
    return d / n[:, None]

def _groove(na, nv):
    x = np.cross(na, nv)
    n = np.linalg.norm(x, axis=1)
    good = n > 1e-6
    out = np.zeros_like(x)
    out[good] = x[good] / n[good, None]
    return out, good

def _slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return 0.0 if m.sum() < 3 or np.ptp(x[m]) < 1e-6 else float(np.polyfit(x[m], y[m], 1)[0])

def _continuity(incoming, p, lookahead_mm=2.0):
    q, arc = _resample_path(p, 0.25)
    if len(q) < 2:
        return 180.0, 0.0
    j = min(len(q) - 1, max(1, int(round(lookahead_mm / 0.25))))
    v = q[j] - q[0]
    v /= max(np.linalg.norm(v), 1e-9)
    c = float(np.clip(np.dot(incoming, v), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))
    return ang, float(np.exp(-ang / 55.0))

def _metrics(path, ta, tv, incoming=None, max_mm=4.0):
    p, q = _resample_path(path, 0.25)
    keep = q <= min(float(q[-1]), float(max_mm)) + 1e-9
    p, q = p[keep], q[keep]
    t = _tangents(p)
    da, na, ga = _dist_normal(ta, p)
    dv, nv, gv = _dist_normal(tv, p)
    g, gg = _groove(na, nv)
    good = ga & gv & gg
    prof = np.full(len(p), np.nan)
    if good.sum() < 3:
        align = at = vt = 0.0
        med = p90 = 90.0
    else:
        dot = np.abs(np.sum(t[good] * g[good], axis=1))
        ang = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
        align = float(np.median(dot))
        med = float(np.median(ang))
        p90 = float(np.percentile(ang, 90))
        at = float(np.median(1.0 - np.abs(np.sum(t[good] * na[good], axis=1))))
        vt = float(np.median(1.0 - np.abs(np.sum(t[good] * nv[good], axis=1))))
        prof[np.flatnonzero(good)] = ang
    sa = _slope(q, da)
    retention = float(np.exp(-max(0.0, sa) / 0.35))
    ca, cs = (0.0, 1.0) if incoming is None else _continuity(incoming, p)
    score = float(0.55 * align + 0.15 * at + 0.10 * vt + 0.10 * retention + 0.10 * cs)
    return {
        "local_geometry_score": score,
        "local_groove_alignment_cos": align,
        "median_groove_tangent_angle_deg": med,
        "p90_groove_tangent_angle_deg": p90,
        "left_atrium_surface_tangency": at,
        "left_ventricle_surface_tangency": vt,
        "left_atrium_distance_slope_mm_per_mm": sa,
        "atrium_retention_score": retention,
        "incoming_continuation_angle_deg": ca,
        "incoming_continuity_score": cs,
        "profile_arc_mm": q,
        "profile_groove_angle_deg": prof,
        "points": p,
        "da": da,
        "dv": dv,
    }

def _control(path, ta, tv):
    p, q = _resample_path(path, 0.50)
    if len(p) > 12:
        p, q = p[4:-4], q[4:-4] - q[4]
    t = _tangents(p)
    da, na, ga = _dist_normal(ta, p)
    dv, nv, gv = _dist_normal(tv, p)
    g, gg = _groove(na, nv)
    good = ga & gv & gg
    if good.sum() < 5:
        return {"score": 0.0, "median_angle_deg": 90.0, "n_valid": int(good.sum())}
    dot = np.abs(np.sum(t[good] * g[good], axis=1))
    ang = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
    at = 1.0 - np.abs(np.sum(t[good] * na[good], axis=1))
    vt = 1.0 - np.abs(np.sum(t[good] * nv[good], axis=1))
    ret = float(np.exp(-max(0.0, _slope(q, da)) / 0.35))
    score = float(0.65 * np.median(dot) + 0.15 * np.median(at) + 0.10 * np.median(vt) + 0.10 * ret)
    return {"score": score, "median_angle_deg": float(np.median(ang)), "n_valid": int(good.sum())}

def _support(tree, path):
    p, _ = _resample_path(path, 0.25)
    return float(np.mean(tree.query(p)[0] <= 1.0))

def _hu(src, g, path):
    h = _sample(src, g, _resample_path(path, 0.25)[0], 1, -1024.0)
    return float(np.mean((h >= 120) & (h <= 1200))), float(np.median(h))

def _frame(points, i):
    p = np.asarray(points, float)
    a, b = max(0, i - 2), min(len(p) - 1, i + 2)
    t = p[b] - p[a]
    t = t / max(np.linalg.norm(t), 1e-9)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    n = np.cross(t, seed)
    n /= max(np.linalg.norm(n), 1e-9)
    bv = np.cross(t, n)
    bv /= max(np.linalg.norm(bv), 1e-9)
    return n, bv

def _plane(g, src, c, n, b, half=7.0, step=0.20):
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    pts = c[None,None,:] + xx[...,None]*n + yy[...,None]*b
    return _sample(src, g, pts.reshape(-1,3), 1, -1024.0).reshape(len(q), len(q)), q

def _consensus_segment(paths, start_arc, end_arc, step=0.25):
    minlen = min(_arc(p)[-1] for p in paths)
    end_arc = min(float(end_arc), float(minlen))
    if end_arc <= start_arc:
        end_arc = min(minlen, start_arc + 0.5)
    q = np.arange(start_arc, end_arc + 1e-9, step)
    if len(q) < 3:
        q = np.linspace(start_arc, end_arc, 3)
    samples = np.stack([_interp_path(p, q) for p in paths], axis=0)
    return np.median(samples, axis=0), q

def _find_split_groups(paths, member_indices, prefix_end, threshold=1.5):
    if len(member_indices) <= 1:
        return None, [member_indices]
    minlen = min(_arc(paths[i])[-1] for i in member_indices)
    for off in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        probe = prefix_end + off
        if probe >= minlen - 0.25:
            continue
        pts = np.vstack([_interp_path(paths[i], [probe])[0] for i in member_indices])
        local_groups = _union_find_groups(pts, threshold)
        if len(local_groups) > 1:
            groups = [[member_indices[j] for j in gr] for gr in local_groups]
            groups = sorted(groups, key=lambda x: (min(x), len(x)))
            return float(probe), groups
    return None, [member_indices]

def _build_hierarchy(paths, member_indices, node_id="N0", depth=0, max_depth=4):
    sub = [paths[i] for i in member_indices]
    con, q, dev, meta = _consensus_prefix(sub, 0.25, 0.60, 4)
    prefix_end = float(meta["common_prefix_total_mm"])
    probe_arc, groups = _find_split_groups(paths, member_indices, prefix_end, 1.5)
    node = {
        "node_id": node_id,
        "depth": int(depth),
        "member_leaf_indices": [int(i+1) for i in member_indices],
        "common_prefix_end_arc_mm": prefix_end,
        "probe_arc_mm": probe_arc,
        "n_children": int(len(groups)) if probe_arc is not None else 0,
        "children": [],
    }
    if depth >= max_depth or probe_arc is None or len(groups) <= 1:
        return node
    for k, group in enumerate(groups, 1):
        child = _build_hierarchy(paths, group, f"{node_id}.{k}", depth + 1, max_depth)
        node["children"].append(child)
    return node

def _flatten_nodes(node):
    out = [node]
    for c in node.get("children", []):
        out.extend(_flatten_nodes(c))
    return out

def _incoming_tangent(paths, members, split_arc):
    q0 = max(0.0, split_arc - 2.0)
    p, _ = _consensus_segment([paths[i] for i in members], q0, split_arc)
    v = p[-1] - p[0]
    return v / max(np.linalg.norm(v), 1e-9)

def _child_segment(paths, child_members, parent_split_arc, child_prefix_end):
    minlen = min(_arc(paths[i])[-1] for i in child_members)
    end = min(minlen, parent_split_arc + 4.0)
    if child_prefix_end > parent_split_arc + 0.75:
        end = min(end, child_prefix_end)
    if end <= parent_split_arc + 0.25:
        end = min(minlen, parent_split_arc + 1.0)
    return _consensus_segment([paths[i] for i in child_members], parent_split_arc, end)[0]

def _branch_valid(m, lad_negative, csup, lsup, hfrac):
    return bool(
        csup >= 0.90 and lsup >= 0.90 and hfrac >= 0.90
        and m["local_geometry_score"] >= lad_negative + 0.05
        and m["local_groove_alignment_cos"] >= 0.35
        and m["atrium_retention_score"] >= 0.55
        and m["incoming_continuation_angle_deg"] <= 50.0
    )

def _sibling_decision(rows):
    rows = sorted(rows, key=lambda r: r["local_geometry_score"], reverse=True)
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[1]
    score_margin = float(a["local_geometry_score"] - b["local_geometry_score"])
    continuity_adv = float(b["incoming_continuation_angle_deg"] - a["incoming_continuation_angle_deg"])
    groove_adv = float(a["local_groove_alignment_cos"] - b["local_groove_alignment_cos"])
    decisive = bool(
        a["branch_gate_pass"] and (
            score_margin >= 0.05 or
            (continuity_adv >= 20.0 and groove_adv >= 0.10)
        )
    )
    return {
        "selected_child_node_id": a["child_node_id"],
        "selected_members": a["member_source_candidate_ids"],
        "runner_up_child_node_id": b["child_node_id"],
        "score_margin": score_margin,
        "continuity_angle_advantage_deg": continuity_adv,
        "groove_alignment_advantage": groove_adv,
        "decision_pass": decisive,
        "decision_rule": "score_margin>=0.05 OR (continuity_advantage>=20deg AND groove_alignment_advantage>=0.10), with branch gate",
    }

def synthetic_hierarchy_self_test():
    t = np.linspace(0, 30, 121)
    base = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    p1 = base.copy(); p2 = base.copy(); p3 = base.copy(); p4 = base.copy()
    m = t > 15
    p1[m,1] = (t[m]-15)*0.8
    p2[m,1] = (t[m]-15)*0.8
    p3[m,2] = (t[m]-15)*0.8
    p4[m,2] = (t[m]-15)*0.8
    m2 = t > 22
    p4[m2,0] += (t[m2]-22)*0.6
    tree = _build_hierarchy([p1,p2,p3,p4], [0,1,2,3])
    assert tree["n_children"] >= 2
    return {"ok": True, "root_children": tree["n_children"], "root_prefix_mm": tree["common_prefix_end_arc_mm"]}

def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})

    required = [
        root/SOURCE_CACHE/"series7_int16.npy", root/SOURCE_CACHE/"series7_int16.json",
        root/MASTER, root/LAD_PATH, root/RCA_PATH, root/COR_CURRENT, root/COR_LEGACY,
        root/LA, root/LV, root/RA, root/RV, root/MYO, root/PRIOR_RANKING
    ] + [root/p for p in LEAF_FILES]
    for p in required:
        _req(p)

    master = json.loads((root/MASTER).read_text())
    g, src = _source(root/SOURCE_CACHE)
    lad = _load_path(root/LAD_PATH, g)
    rca = _load_path(root/RCA_PATH, g)
    leaves = _orient_common([_load_path(root/p, g) for p in LEAF_FILES])
    ranking = pd.read_csv(root/PRIOR_RANKING)
    prior = {i: ranking.iloc[i-1].to_dict() for i in range(1,6)}

    print("Building sparse coronary/chamber trees...")
    cur = _tree(root/COR_CURRENT, False)
    leg = _tree(root/COR_LEGACY, False)
    la = _tree(root/LA, True)
    lv = _tree(root/LV, True)
    ra = _tree(root/RA, True)
    rv = _tree(root/RV, True)
    myo = _tree(root/MYO, True)

    con, qc, dev, cm = _consensus_prefix(leaves, 0.25, 0.60, 4)
    dl = _nearest_dist(con, lad)
    di = _first_sustained(dl, 1.5, 0.25, 1.0)
    di = len(con)-1 if di is None else di
    trunk = con[di:]
    ta = _arc(trunk)
    tc, tl = _support(cur, trunk), _support(leg, trunk)
    hf, hm = _hu(src, g, trunk)
    trunk_ok = bool(ta[-1] >= 10.0 and tc >= .90 and tl >= .90 and hf >= .90)
    pd.DataFrame(trunk, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).assign(
        arc_mm=ta, lad_distance_mm=dl[di:]
    ).to_csv(out/"consensus_left_coronary_branch_trunk.csv", index=False)
    trunk_summary = {
        **cm,
        "lad_divergence_arc_from_candidate_origin_mm": float(qc[di]),
        "post_lad_divergence_common_trunk_mm": float(ta[-1]),
        "current_support_fraction": tc,
        "legacy_support_fraction": tl,
        "robust_hu_fraction": hf,
        "median_hu": hm,
        "consensus_trunk_gate_pass": trunk_ok,
    }
    _write_json(out/"consensus_trunk_summary.json", trunk_summary)

    rc = _control(rca, ra, rv)
    lc = _control(lad, la, lv)
    ctrl_margin = float(rc["score"] - lc["score"])
    control_pass = bool(rc["score"] >= .45 and ctrl_margin >= .08 and rc["n_valid"] >= 5)
    controls = {
        "RCA_right_AV_local_tangent_score": rc["score"],
        "RCA_median_groove_tangent_angle_deg": rc["median_angle_deg"],
        "LAD_left_AV_local_tangent_negative_score": lc["score"],
        "LAD_median_groove_tangent_angle_deg": lc["median_angle_deg"],
        "control_margin": ctrl_margin,
        "control_pass": control_pass,
        "normal_method": "nearest sparse chamber-surface point",
    }
    _write_json(out/"hierarchical_controls.json", controls)

    tree = _build_hierarchy(leaves, list(range(5)))
    nodes = _flatten_nodes(tree)
    _write_json(out/"hierarchical_tree.json", tree)

    node_rows = []
    decisions = []
    source_ids_by_leaf = {i: int(prior[i]["candidate_id"]) for i in range(1,6)}
    for node in nodes:
        children = node.get("children", [])
        if len(children) < 2:
            continue
        parent_members0 = [i-1 for i in node["member_leaf_indices"]]
        split_arc = float(node["common_prefix_end_arc_mm"])
        incoming = _incoming_tangent(leaves, parent_members0, split_arc)
        child_rows = []
        for child in children:
            members0 = [i-1 for i in child["member_leaf_indices"]]
            seg = _child_segment(leaves, members0, split_arc, float(child["common_prefix_end_arc_mm"]))
            m = _metrics(seg, la, lv, incoming, 4.0)
            csup, lsup = _support(cur, seg), _support(leg, seg)
            hfrac, hmed = _hu(src, g, seg)
            myod = float(np.median(myo.query(m["points"])[0]))
            source_ids = [source_ids_by_leaf[i+1] for i in members0]
            valid = _branch_valid(m, lc["score"], csup, lsup, hfrac)
            row = {
                "parent_node_id": node["node_id"],
                "child_node_id": child["node_id"],
                "parent_split_arc_mm": split_arc,
                "child_common_prefix_end_arc_mm": float(child["common_prefix_end_arc_mm"]),
                "n_leaf_members": len(members0),
                "member_leaf_indices": ";".join(map(str, child["member_leaf_indices"])),
                "member_source_candidate_ids": ";".join(map(str, source_ids)),
                "local_geometry_score": m["local_geometry_score"],
                "local_groove_alignment_cos": m["local_groove_alignment_cos"],
                "median_groove_tangent_angle_deg": m["median_groove_tangent_angle_deg"],
                "p90_groove_tangent_angle_deg": m["p90_groove_tangent_angle_deg"],
                "incoming_continuation_angle_deg": m["incoming_continuation_angle_deg"],
                "incoming_continuity_score": m["incoming_continuity_score"],
                "left_atrium_surface_tangency": m["left_atrium_surface_tangency"],
                "left_ventricle_surface_tangency": m["left_ventricle_surface_tangency"],
                "left_atrium_distance_slope_mm_per_mm": m["left_atrium_distance_slope_mm_per_mm"],
                "atrium_retention_score": m["atrium_retention_score"],
                "median_myocardium_surface_mm": myod,
                "current_support_fraction": csup,
                "legacy_support_fraction": lsup,
                "robust_hu_fraction": hfrac,
                "median_hu": hmed,
                "branch_gate_pass": valid,
            }
            node_rows.append(row)
            child_rows.append(row)
        decision = _sibling_decision(child_rows)
        if decision:
            decision["parent_node_id"] = node["node_id"]
            decision["parent_split_arc_mm"] = split_arc
            decisions.append(decision)

    ndf = pd.DataFrame(node_rows)
    ndf.to_csv(out/"hierarchical_node_branch_scores.csv", index=False)
    ddf = pd.DataFrame(decisions)
    ddf.to_csv(out/"hierarchical_node_decisions.csv", index=False)

    selected_nodes = []
    current = tree
    while current.get("children"):
        dec = next((d for d in decisions if d["parent_node_id"] == current["node_id"]), None)
        if not dec or not dec["decision_pass"]:
            break
        cid = dec["selected_child_node_id"]
        selected_nodes.append(cid)
        current = next(c for c in current["children"] if c["node_id"] == cid)

    terminal_members = current["member_leaf_indices"]
    terminal_source_ids = [source_ids_by_leaf[i] for i in terminal_members]
    unique_terminal = len(terminal_members) == 1
    selected_source_candidate_id = terminal_source_ids[0] if unique_terminal else None

    root_decision = next((d for d in decisions if d["parent_node_id"] == "N0"), None)
    root_pass = bool(root_decision and root_decision["decision_pass"])
    candidate_pass = bool(root_pass and unique_terminal and len(selected_nodes) >= 2)

    if not trunk_ok:
        status = STATUS_NO_TRUNK
    elif not control_pass:
        status = STATUS_CONTROL_FAIL
    elif candidate_pass:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_AMBIG

    if selected_source_candidate_id is not None:
        leaf_idx = terminal_members[0]
        selected_path = leaves[leaf_idx-1]
        pd.DataFrame(selected_path, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).assign(
            arc_mm=_arc(selected_path),
            source_candidate_id=selected_source_candidate_id,
        ).to_csv(out/"hierarchical_lcx_candidate_path.csv", index=False)

    fig, ax = plt.subplots(figsize=(10,6))
    def draw_node(node, y):
        x = node["common_prefix_end_arc_mm"]
        label_ids = [source_ids_by_leaf[i] for i in node["member_leaf_indices"]]
        ax.scatter([x],[y],s=50)
        ax.text(x+0.15,y,f'{node["node_id"]}: C{",".join(map(str,label_ids))}',va="center",fontsize=8)
        kids = node.get("children",[])
        if kids:
            offsets = np.linspace(-0.8,0.8,len(kids))
            for off,c in zip(offsets,kids):
                cx = c["common_prefix_end_arc_mm"]
                cy = y + off
                ax.plot([x,cx],[y,cy],lw=2)
                draw_node(c,cy)
    draw_node(tree,0.0)
    ax.set_xlabel("Arc from candidate origin (mm)")
    ax.set_ylabel("Tree branch display coordinate")
    ax.set_title(status)
    fig.tight_layout()
    fig.savefig(out/"01_hierarchical_tree_topology.png",dpi=170)
    plt.close(fig)

    if len(ndf):
        parents = list(ndf.parent_node_id.unique())
        fig, axes = plt.subplots(len(parents), 1, figsize=(9, 4*len(parents)), squeeze=False)
        for rr,pid in enumerate(parents):
            s = ndf[ndf.parent_node_id==pid]
            x = np.arange(len(s))
            axes[rr,0].bar(x, s.local_geometry_score)
            axes[rr,0].set_xticks(x, s.member_source_candidate_ids)
            axes[rr,0].axhline(lc["score"],ls=":",label="LAD negative")
            axes[rr,0].axhline(rc["score"],ls="--",label="RCA control")
            for j,(_,r) in enumerate(s.iterrows()):
                axes[rr,0].text(j,r.local_geometry_score+0.01,f'{r.incoming_continuation_angle_deg:.1f}° / align {r.local_groove_alignment_cos:.2f}',ha="center",fontsize=8)
            axes[rr,0].set_ylabel("Local geometry score")
            axes[rr,0].set_title(f"{pid} split @ {s.parent_split_arc_mm.iloc[0]:.2f} mm")
            axes[rr,0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out/"02_hierarchical_node_scores.png",dpi=170)
        plt.close(fig)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111,projection="3d")
    for i,p in enumerate(leaves,1):
        sid = source_ids_by_leaf[i]
        lw = 4 if sid == selected_source_candidate_id else 1.5
        ax.plot(p[:,0],p[:,1],p[:,2],lw=lw,label=f"C{sid}")
    ax.plot(trunk[:,0],trunk[:,1],trunk[:,2],lw=5,label="Consensus trunk")
    ax.legend(fontsize=8)
    ax.set_title("Hierarchical distal tree; selected route emphasized")
    fig.tight_layout()
    fig.savefig(out/"03_hierarchical_tree_geometry.png",dpi=170)
    plt.close(fig)

    qc_items = []
    for d in decisions:
        if d["decision_pass"]:
            pid = d["parent_node_id"]
            parent = next(n for n in nodes if n["node_id"] == pid)
            split_arc = parent["common_prefix_end_arc_mm"]
            members0 = [i-1 for i in parent["member_leaf_indices"]]
            conp,_ = _consensus_segment([leaves[i] for i in members0], max(0,split_arc-1), split_arc)
            qc_items.append((pid, conp[-1]))
    if qc_items:
        fig, axes = plt.subplots(len(qc_items), 3, figsize=(12,4*len(qc_items)), squeeze=False)
        for rr,(pid,center) in enumerate(qc_items):
            parent = next(n for n in nodes if n["node_id"] == pid)
            members0 = [i-1 for i in parent["member_leaf_indices"]]
            split_arc = parent["common_prefix_end_arc_mm"]
            incoming = _incoming_tangent(leaves,members0,split_arc)
            axes3 = np.eye(3)
            seed = axes3[np.argmin(np.abs(axes3@incoming))]
            n = np.cross(incoming,seed); n/=max(np.linalg.norm(n),1e-9)
            b = np.cross(incoming,n); b/=max(np.linalg.norm(b),1e-9)
            for cc,delta in enumerate([-0.5,0.0,0.75]):
                pcenter = _consensus_segment([leaves[i] for i in members0], max(0,split_arc+delta), max(0,split_arc+delta)+0.01)[0][0]
                ct,qg = _plane(g,src,pcenter,n,b)
                axes[rr,cc].imshow(ct,cmap="gray",vmin=-100,vmax=900,extent=[qg[0],qg[-1],qg[-1],qg[0]])
                axes[rr,cc].scatter([0],[0],marker="+")
                axes[rr,cc].set_title(f"{pid} {delta:+.2f} mm")
        fig.tight_layout()
        fig.savefig(out/"04_decisive_split_orthogonal_qc.png",dpi=170)
        plt.close(fig)

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "LCX_master_status": "UNRESOLVED",
        "consensus_trunk": trunk_summary,
        "controls": controls,
        "tree_root": tree,
        "node_decisions": decisions,
        "selected_node_path": selected_nodes,
        "terminal_leaf_indices": terminal_members,
        "terminal_source_candidate_ids": terminal_source_ids,
        "selected_source_candidate_id": selected_source_candidate_id if candidate_pass else None,
        "candidate_pass": candidate_pass,
        "template_similarity_used_in_decision": False,
        "memory_strategy": "source memmap + sparse point-cloud KD-trees; no full-volume distance transforms",
        "decision_policy": {
            "branch_gate": "support/HU >=0.90, score >= LAD+0.05, groove alignment >=0.35, atrium retention >=0.55, continuity angle <=50deg",
            "sibling_decision": "score margin >=0.05 OR continuity advantage >=20deg with groove-alignment advantage >=0.10",
            "candidate_requires": "decisive root plus decisive downstream split terminating in a singleton leaf",
        },
        "scientific_boundary": "Research-only LCX-like path nomination. No Master Anatomy update; LM remains unresolved.",
    }
    _write_json(out/"summary.json", summary)

    report = out/"OPENPLAQUE_LCX_HIERARCHICAL_DISTAL_TREE_REPORT.html"
    report.write_text(
        f"<html><body><h1>OpenPlaque LCX hierarchical distal tree</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p><b>Post-LAD consensus trunk:</b> {trunk_summary['post_lad_divergence_common_trunk_mm']:.2f} mm</p>"
        f"<p><b>RCA control:</b> {rc['score']:.3f}; <b>LAD negative:</b> {lc['score']:.3f}; margin {ctrl_margin:.3f}</p>"
        f"<p><b>Selected source candidate:</b> {summary['selected_source_candidate_id']}</p>"
        f"<p>Hierarchical node-by-node adjudication; curved-template similarity has zero decision weight.</p>"
        f"<img src='01_hierarchical_tree_topology.png' style='max-width:95%'><br>"
        f"<img src='02_hierarchical_node_scores.png' style='max-width:95%'><br>"
        f"<img src='03_hierarchical_tree_geometry.png' style='max-width:95%'><br>"
        f"<img src='04_decisive_split_orthogonal_qc.png' style='max-width:95%'>"
        f"</body></html>",
        encoding="utf-8"
    )
    _write_json(out/"run_state.json", {"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    zp = out/"OPENPLAQUE_LCX_HIERARCHICAL_DISTAL_TREE_REPORT_BACK.zip"
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.name != zp.name:
                z.write(p,arcname=p.name)
    return {"summary":summary,"report":str(report),"zip":str(zp)}
