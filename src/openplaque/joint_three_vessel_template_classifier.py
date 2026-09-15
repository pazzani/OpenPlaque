from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "joint-three-vessel-template-classifier-v1.0"
OUTPUT_DIRNAME = "Joint_Three_Vessel_Template_Classifier_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "total/aorta.nii.gz"
CURVED = Path("UCLA_Plaque_Context_Verification")
IMAGES = {
    "RCA": CURVED / "RCA_input/RCA_0000.nii.gz",
    "LAD": CURVED / "LAD_input/LAD_0000.nii.gz",
    "LCX": CURVED / "LCX_input/LCX_0000.nii.gz",
}
MASKS = {k: CURVED / f"nnunet_masks/{k}.nii.gz" for k in ("RCA", "LAD", "LCX")}

STATUS_CALIBRATION_FAILED = "JOINT_RCA_LAD_CALIBRATION_FAILED"
STATUS_NO_MATCH = "NO_DISTINCT_LCX_SOURCE_PATH"
STATUS_CANDIDATE = "LCX_SOURCE_PATH_CANDIDATE_REQUIRES_ANATOMICAL_QC"

RECIPES = {
    "hybrid": {"median": 0.34, "p90": 0.34, "gradient": 0.18, "plaque": 0.14},
    "intensity_shape": {"median": 0.42, "p90": 0.42, "gradient": 0.16, "plaque": 0.0},
    "median_landmark": {"median": 0.52, "p90": 0.18, "gradient": 0.15, "plaque": 0.15},
    "p90_landmark": {"median": 0.18, "p90": 0.52, "gradient": 0.15, "plaque": 0.15},
}


def _req(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source(cache):
    arr = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = json.loads(_req(cache / "series7_int16.json").read_text())
    img = sitk.GetImageFromArray(np.asarray(arr))
    spacing_zyx = np.asarray(meta["spacing_zyx"], float)
    img.SetSpacing(tuple(spacing_zyx[::-1]))
    img.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([
        [row[0], col[0], slc[0]],
        [row[1], col[1], slc[1]],
        [row[2], col[2], slc[2]],
    ], float)
    img.SetDirection(tuple(direction.ravel()))
    return img, np.asarray(arr)


def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (
        im.GetSize() == ref.GetSize()
        and np.allclose(im.GetSpacing(), ref.GetSpacing())
        and np.allclose(im.GetOrigin(), ref.GetOrigin())
        and np.allclose(im.GetDirection(), ref.GetDirection())
    )
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0


def _xyz_to_zyx(img, pts_lps):
    pts = np.atleast_2d(np.asarray(pts_lps, float))
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(direction).T) / sp
    return idx_xyz[:, ::-1]


def _zyx_to_xyz(img, pts_zyx):
    pts = np.atleast_2d(np.asarray(pts_zyx, float))
    idx_xyz = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ direction.T


def _sample(img, arr, pts_lps, order=1, cval=-1024.0):
    zyx = _xyz_to_zyx(img, pts_lps)
    return map_coordinates(np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval)


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in [("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")]:
        if all(c in d.columns for c in cols):
            return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    if all(c in d.columns for c in ("x", "y", "z")):
        return d[["x", "y", "z"]].to_numpy(float)
    raise ValueError(f"No recognized path coordinates in {path}; columns={list(d.columns)}")


def _arc(points):
    p = np.asarray(points, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _resample_path(points, step=0.25):
    p = np.asarray(points, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p, a
    q = np.arange(0.0, a[-1] + 1e-9, float(step))
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    out = np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])
    return out, q


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return -1.0
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 1e-10 else -1.0


def _fill(x):
    x = np.asarray(x, float).copy()
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    if good.sum() == 1:
        x[~good] = x[good][0]
        return x
    idx = np.arange(len(x))
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _interp(x, n=128):
    x = _fill(x)
    if len(x) == 0:
        return np.zeros(n, float)
    if len(x) == 1:
        return np.repeat(x, n)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def _align_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape == tuple(shape):
        return mask
    for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        q = np.transpose(mask, p)
        if q.shape == tuple(shape):
            return q
    raise RuntimeError(f"Mask shape {mask.shape} cannot align to image shape {tuple(shape)}")


def _curved_pair(image_path, mask_path):
    image = sitk.ReadImage(str(_req(image_path)))
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(_req(mask_path))))
    mask = _align_mask(mask, volume.shape)
    return volume, mask


def _template_fingerprint(volume, mask):
    volume = np.asarray(volume, float)
    mask = np.asarray(mask)
    candidates = []
    for long_axis in (1, 2):
        support_full = (mask > 0).sum(axis=tuple(ax for ax in range(3) if ax != long_axis))
        active = support_full > 0
        idx = np.where(active)[0]
        if len(idx) < 8:
            continue
        start, end = int(idx.min()), int(idx.max()) + 1
        vv = np.moveaxis(volume, long_axis, -1)[..., start:end]
        mm = np.moveaxis(mask, long_axis, -1)[..., start:end]
        support = (mm > 0).sum(axis=(0,1)).astype(float)
        plaque = (mm == 2).sum(axis=(0,1)).astype(float)
        hu_med = np.full(end-start, np.nan, float)
        hu_p90 = np.full(end-start, np.nan, float)
        for j in range(end-start):
            vals = vv[..., j][mm[..., j] > 0]
            if len(vals):
                hu_med[j] = float(np.median(vals))
                hu_p90[j] = float(np.quantile(vals, 0.90))
        candidates.append({
            "long_axis": int(long_axis), "start_px": start, "end_px": end,
            "span_px": int(end-start), "support": support,
            "plaque_fraction": plaque / np.maximum(support, 1.0),
            "hu_median": hu_med, "hu_p90": hu_p90,
        })
    if not candidates:
        raise RuntimeError("Could not identify longitudinal axis in curved template")
    return max(candidates, key=lambda r: r["span_px"])


def _frame(points, i):
    p = np.asarray(points, float)
    a = max(0, i-2)
    b = min(len(p)-1, i+2)
    tangent = p[b] - p[a]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    seed = np.eye(3)[np.argmin(np.abs(np.eye(3) @ tangent))]
    n = np.cross(tangent, seed)
    n = n / max(np.linalg.norm(n), 1e-9)
    bvec = np.cross(tangent, n)
    bvec = bvec / max(np.linalg.norm(bvec), 1e-9)
    return n, bvec


def _source_profile(img, source, points):
    pts, arc = _resample_path(points, 0.25)
    if len(pts) == 0:
        raise ValueError("empty source path")
    offsets_2d = [(0.0,0.0)]
    for y in np.arange(-1.2, 1.21, 0.4):
        for x in np.arange(-1.2, 1.21, 0.4):
            if x*x + y*y <= 1.2*1.2 + 1e-9 and not (abs(x) < 1e-9 and abs(y) < 1e-9):
                offsets_2d.append((float(x), float(y)))
    med = np.zeros(len(pts), float)
    p90 = np.zeros(len(pts), float)
    center = _sample(img, source, pts)
    for i, c in enumerate(pts):
        n, b = _frame(pts, i)
        samples = np.vstack([c + x*n + y*b for x,y in offsets_2d])
        vals = _sample(img, source, samples)
        med[i] = float(np.median(vals))
        p90[i] = float(np.quantile(vals, 0.90))
    robust = np.isfinite(center) & (center >= 120) & (center <= 1200)
    landmark = np.maximum(p90 - np.nanquantile(p90, 0.75), 0.0)
    if np.nanmax(landmark) > 0:
        landmark = landmark / np.nanmax(landmark)
    return {
        "points": pts, "arc": arc, "center_hu": center,
        "tube_median": med, "tube_p90": p90, "landmark": landmark,
        "robust_fraction": float(robust.mean()),
        "median_hu": float(np.nanmedian(center)),
        "length_mm": float(arc[-1]) if len(arc) else 0.0,
    }


def _window_lengths(span, expected):
    vals = sorted(set(max(8, min(span, int(round(expected*f)))) for f in (0.85,0.925,1.0,1.075,1.15)))
    return vals


def _score_window(template, profile, recipe, mm_per_px):
    weights = RECIPES[recipe]
    span = int(template["span_px"])
    expected = max(8.0, float(profile["length_mm"]) / max(mm_per_px, 1e-6))
    src_med = _interp(profile["tube_median"])
    src_p90 = _interp(profile["tube_p90"])
    src_land = _interp(profile["landmark"])
    src_grad = np.gradient(src_med)
    best = None
    for w in _window_lengths(span, expected):
        starts = range(0, span-w+1, max(1, w//16))
        if span-w not in starts:
            starts = list(starts) + [span-w]
        for start in starts:
            sl = slice(start, start+w)
            t_med0 = _interp(template["hu_median"][sl])
            t_p900 = _interp(template["hu_p90"][sl])
            t_plq0 = _interp(template["plaque_fraction"][sl])
            for orientation in ("forward", "reverse"):
                t_med = t_med0 if orientation == "forward" else t_med0[::-1]
                t_p90 = t_p900 if orientation == "forward" else t_p900[::-1]
                t_plq = t_plq0 if orientation == "forward" else t_plq0[::-1]
                c_med = _corr(t_med, src_med)
                c_p90 = _corr(t_p90, src_p90)
                c_grad = _corr(np.gradient(t_med), src_grad)
                c_plq = _corr(t_plq, src_land) if np.nanstd(t_plq) > 1e-8 else 0.0
                comps = {"median": c_med, "p90": c_p90, "gradient": c_grad, "plaque": c_plq}
                corr_score = sum(weights[k] * ((comps[k]+1.0)/2.0) for k in weights)
                length_score = math.exp(-abs(math.log(max(w,1.0)/max(expected,1.0))))
                score = 0.88*corr_score + 0.07*length_score + 0.05*float(profile["robust_fraction"])
                rec = {
                    "score": float(score), "window_start_px": int(template["start_px"]+start),
                    "window_end_px": int(template["start_px"]+start+w), "window_span_px": int(w),
                    "orientation": orientation, "median_corr": float(c_med), "p90_corr": float(c_p90),
                    "gradient_corr": float(c_grad), "plaque_landmark_corr": float(c_plq),
                    "length_score": float(length_score), "robust_fraction": float(profile["robust_fraction"]),
                    "median_hu": float(profile["median_hu"]), "length_mm": float(profile["length_mm"]),
                }
                if best is None or rec["score"] > best["score"]:
                    best = rec
    return best


def _calibrate(templates, source_profiles):
    rca_scale = source_profiles["RCA"]["length_mm"] / max(templates["RCA"]["span_px"], 1)
    rows = []
    recipe_summaries = []
    for recipe in RECIPES:
        scores = {}
        for source_name in ("RCA", "LAD"):
            for template_name in ("RCA", "LAD", "LCX"):
                rec = _score_window(templates[template_name], source_profiles[source_name], recipe, rca_scale)
                scores[(source_name, template_name)] = rec
                rows.append({"recipe": recipe, "source_path": source_name, "template": template_name, **rec})
        rca_pos = scores[("RCA","RCA")]["score"]
        lad_pos = scores[("LAD","LAD")]["score"]
        rca_wrong = max(scores[("RCA","LAD")]["score"], scores[("RCA","LCX")]["score"])
        lad_wrong = max(scores[("LAD","RCA")]["score"], scores[("LAD","LCX")]["score"])
        rca_margin = rca_pos - rca_wrong
        lad_margin = lad_pos - lad_wrong
        objective = min(rca_margin, lad_margin) + 0.15*((rca_pos+lad_pos)/2.0)
        recipe_summaries.append({
            "recipe": recipe, "rca_positive": rca_pos, "lad_positive": lad_pos,
            "rca_margin": rca_margin, "lad_margin": lad_margin,
            "min_margin": min(rca_margin, lad_margin), "objective": objective,
        })
    summary_df = pd.DataFrame(recipe_summaries).sort_values(["objective","min_margin"], ascending=False).reset_index(drop=True)
    chosen = str(summary_df.iloc[0]["recipe"])
    chosen_row = summary_df.iloc[0]
    passed = bool(chosen_row["rca_positive"] >= 0.52 and chosen_row["lad_positive"] >= 0.52 and chosen_row["rca_margin"] >= 0.03 and chosen_row["lad_margin"] >= 0.03)
    return {
        "recipe": chosen, "mm_per_px": float(rca_scale), "passed": passed,
        "summary": summary_df, "confusion": pd.DataFrame(rows),
        "rca_margin": float(chosen_row["rca_margin"]), "lad_margin": float(chosen_row["lad_margin"]),
        "rca_positive": float(chosen_row["rca_positive"]), "lad_positive": float(chosen_row["lad_positive"]),
    }


def _distance_to_mask(img, mask, points):
    z = np.argwhere(mask)
    if len(z) == 0:
        return np.full(len(points), np.inf)
    if len(z) > 150000:
        z = z[::int(math.ceil(len(z)/150000))]
    return cKDTree(_zyx_to_xyz(img, z)).query(np.asarray(points,float))[0]


def _orient_lad_from_aorta(img, aorta, lad):
    d = _distance_to_mask(img, aorta, np.vstack([lad[0], lad[-1]]))
    return lad if d[0] <= d[1] else lad[::-1].copy()


def _candidate_paths(img, source, coronary_union, aorta, lad, rca, max_paths=32):
    lad = _orient_lad_from_aorta(img, aorta, lad)
    lad_r, lad_arc = _resample_path(lad, 0.25)
    rca_r, _ = _resample_path(rca, 0.25)
    anchor = lad_r[0]
    skel = skeletonize(coronary_union)
    zyx = np.argwhere(skel)
    if len(zyx) < 10:
        return [], {"reason":"too_few_nodes","n_skeleton_nodes":int(len(zyx))}
    phys = _zyx_to_xyz(img, zyx)
    d_lad, j_lad = cKDTree(lad_r).query(phys)
    d_rca = cKDTree(rca_r).query(phys)[0]
    d_anchor = np.linalg.norm(phys-anchor[None,:], axis=1)
    near_prox_lad = (d_lad <= 1.25) & (lad_arc[j_lad] <= 4.0)
    novel = d_lad >= 1.50
    allowed = (near_prox_lad | novel) & (d_rca >= 1.50) & (d_anchor <= 40.0)
    keep = np.where(allowed)[0]
    if len(keep) < 10:
        return [], {"reason":"too_few_allowed_nodes","n_skeleton_nodes":int(len(zyx)),"n_allowed_nodes":int(len(keep))}
    P = phys[keep]
    tree = cKDTree(P)
    pairs = list(tree.query_pairs(1.50))
    if not pairs:
        return [], {"reason":"no_graph_edges","n_allowed_nodes":int(len(P))}
    rows=[]; cols=[]; vals=[]
    for i,j in pairs:
        dist = float(np.linalg.norm(P[i]-P[j]))
        hu = float(_sample(img, source, (0.5*(P[i]+P[j]))[None,:])[0])
        penalty = 1.0 if hu >= 120 else 2.5
        rows += [i,j]; cols += [j,i]; vals += [dist*penalty, dist*penalty]
    n=len(P); super_node=n
    start = tree.query_ball_point(anchor, 3.0)
    if not start:
        _, near = tree.query(anchor, k=min(5,n))
        start = np.atleast_1d(near).astype(int).tolist()
    for i in start:
        dd=float(np.linalg.norm(P[i]-anchor))
        rows += [super_node,int(i)]; cols += [int(i),super_node]; vals += [dd,dd]
    graph=csr_matrix((vals,(rows,cols)),shape=(n+1,n+1))
    dist,pred=dijkstra(graph,directed=False,indices=super_node,return_predecessors=True)
    euclid=np.linalg.norm(P-anchor[None,:],axis=1)
    order=np.argsort(dist[:n])[::-1]
    selected=[]; paths=[]
    for target in order:
        if len(paths)>=max_paths: break
        if not np.isfinite(dist[target]) or euclid[target] < 6.0 or dist[target] < 8.0 or dist[target] > 48.0: continue
        if d_lad[keep[target]] < 2.0: continue
        if any(np.linalg.norm(P[target]-P[q]) < 3.0 for q in selected): continue
        chain=[]; cur=int(target); guard=0
        while cur != super_node and cur >= 0 and guard < n+5:
            chain.append(cur); cur=int(pred[cur]); guard+=1
        if cur != super_node or len(chain)<3: continue
        path=np.vstack([anchor,P[np.asarray(chain[::-1],int)]])
        path,arc=_resample_path(path,0.25)
        if len(path)<10 or arc[-1]<6.0: continue
        paths.append(path); selected.append(int(target))
    return paths, {"n_skeleton_nodes":int(len(zyx)),"n_allowed_nodes":int(len(P)),"n_graph_edges":int(len(pairs)),"n_start_nodes":int(len(start)),"n_candidate_paths":int(len(paths))}


def _candidate_scores(paths, templates, img, source, calibration):
    rows=[]; path_map={}
    for i,path in enumerate(paths):
        prof=_source_profile(img,source,path)
        recs={t:_score_window(templates[t],prof,calibration["recipe"],calibration["mm_per_px"]) for t in ("RCA","LAD","LCX")}
        other=max(recs["RCA"]["score"],recs["LAD"]["score"])
        rows.append({
            "candidate_id":int(i), "length_mm":prof["length_mm"], "median_hu":prof["median_hu"], "robust_fraction":prof["robust_fraction"],
            "RCA_score":recs["RCA"]["score"], "LAD_score":recs["LAD"]["score"], "LCX_score":recs["LCX"]["score"],
            "LCX_margin":recs["LCX"]["score"]-other,
            "LCX_window_start_px":recs["LCX"]["window_start_px"], "LCX_window_end_px":recs["LCX"]["window_end_px"],
            "LCX_orientation":recs["LCX"]["orientation"], "LCX_median_corr":recs["LCX"]["median_corr"],
            "LCX_p90_corr":recs["LCX"]["p90_corr"], "LCX_gradient_corr":recs["LCX"]["gradient_corr"],
            "LCX_plaque_landmark_corr":recs["LCX"]["plaque_landmark_corr"],
        })
        path_map[int(i)] = path
    df=pd.DataFrame(rows)
    if len(df):
        df=df.sort_values(["LCX_margin","LCX_score"],ascending=False).reset_index(drop=True)
    return df,path_map


def _decision(calibration, ranking):
    if not calibration["passed"]:
        return STATUS_CALIBRATION_FAILED
    if ranking is None or len(ranking)==0:
        return STATUS_NO_MATCH
    top=ranking.iloc[0]
    runner_margin=float(top["LCX_margin"] - ranking.iloc[1]["LCX_margin"]) if len(ranking)>1 else 1.0
    required_margin=max(0.05,0.5*min(calibration["rca_margin"],calibration["lad_margin"]))
    if (float(top["LCX_score"]) >= 0.55 and float(top["LCX_margin"]) >= required_margin and runner_margin >= 0.02 and float(top["robust_fraction"]) >= 0.75 and 8.0 <= float(top["length_mm"]) <= 45.0):
        return STATUS_CANDIDATE
    return STATUS_NO_MATCH


def _plane(img, source, center, normal, bvec, half=4.5, pix=0.15):
    q=np.arange(-half,half+pix/2,pix)
    yy,xx=np.meshgrid(q,q,indexing="ij")
    pts=center[None,None,:]+xx[...,None]*normal[None,None,:]+yy[...,None]*bvec[None,None,:]
    vals=_sample(img,source,pts.reshape(-1,3)).reshape(len(q),len(q))
    return vals,q


def _write_qc(out,img,source,ranking,path_map):
    if ranking is None or len(ranking)==0: return
    fig,axes=plt.subplots(2,4,figsize=(14,7)); axes=np.asarray(axes).ravel(); slot=0
    for rank in range(min(2,len(ranking))):
        path=path_map[int(ranking.iloc[rank]["candidate_id"])]
        for frac in (0.15,0.40,0.65,0.85):
            i=int(round(frac*(len(path)-1)))
            n,b=_frame(path,i); plane,q=_plane(img,source,path[i],n,b)
            ax=axes[slot]; ax.imshow(plane,cmap="gray",vmin=-200,vmax=1000,extent=[q[0],q[-1],q[-1],q[0]])
            ax.scatter([0],[0],s=28,marker="+"); ax.set_title(f"Rank {rank+1} — {frac:.0%} arc"); ax.set_xlabel("mm"); ax.set_ylabel("mm"); slot+=1
    for ax in axes[slot:]: ax.axis("off")
    fig.suptitle("Joint three-vessel classifier: top LCX candidates — source CCTA orthogonal QC")
    fig.tight_layout(); fig.savefig(out/"03_top_candidate_source_orthogonal_qc.png",dpi=180,bbox_inches="tight"); plt.close(fig)


def synthetic_joint_classifier_self_test():
    t={}
    x=np.linspace(0,1,100)
    for name,phase in (("RCA",0.0),("LAD",0.7),("LCX",1.4)):
        med=500+80*np.sin(2*np.pi*x+phase)+20*np.sin(6*np.pi*x+phase)
        p90=med+120+30*np.cos(4*np.pi*x+phase)
        plq=np.maximum(np.sin(3*np.pi*x+phase),0)
        t[name]={"start_px":0,"end_px":100,"span_px":100,"hu_median":med,"hu_p90":p90,"plaque_fraction":plq,"support":np.ones(100)}
    def prof(name,L=25.0):
        z=t[name]
        return {"tube_median":z["hu_median"],"tube_p90":z["hu_p90"],"landmark":z["plaque_fraction"],"robust_fraction":1.0,"median_hu":550.0,"length_mm":L,"points":np.zeros((100,3)),"arc":np.linspace(0,L,100),"center_hu":z["hu_median"]}
    sp={"RCA":prof("RCA",25.0),"LAD":prof("LAD",25.0)}
    c=_calibrate(t,sp)
    ranking=pd.DataFrame([{"LCX_score":0.70,"LCX_margin":0.12,"robust_fraction":1.0,"length_mm":20.0},{"LCX_score":0.60,"LCX_margin":0.05,"robust_fraction":1.0,"length_mm":18.0}])
    status=_decision({**c,"passed":True,"rca_margin":0.1,"lad_margin":0.1},ranking)
    return {"passed":bool(c["recipe"] in RECIPES and status==STATUS_CANDIDATE),"recipe":c["recipe"],"status":status,"algorithm":ALGORITHM}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    root=Path(drive_root); out=Path(output_root or root/OUTPUT_DIRNAME); out.mkdir(parents=True,exist_ok=True)
    _json(out/"run_state.json",{"status":"STARTED","baseline_commit":BASELINE,"algorithm":ALGORITHM})
    master=json.loads(_req(root/MASTER).read_text())
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN": raise RuntimeError("Master Anatomy v2 is not frozen")
    if "LCX" not in set(master.get("unresolved",[])): raise RuntimeError("LCX is no longer unresolved; experiment stale")
    img,source=_source(root/SOURCE_CACHE)
    lad=_load_path(root/LAD_PATH,img); rca=_load_path(root/RCA_PATH,img)
    current=_resample_mask(root/CUR,img); legacy=_resample_mask(root/LEG,img); aorta=_resample_mask(root/AORTA,img)
    templates={}
    for name in ("RCA","LAD","LCX"):
        vol,mask=_curved_pair(root/IMAGES[name],root/MASKS[name]); templates[name]=_template_fingerprint(vol,mask)
    source_profiles={"RCA":_source_profile(img,source,rca),"LAD":_source_profile(img,source,lad)}
    calibration=_calibrate(templates,source_profiles)
    calibration["summary"].to_csv(out/"calibration_recipe_summary.csv",index=False)
    calibration["confusion"].to_csv(out/"known_vessel_template_scores.csv",index=False)
    _json(out/"selected_calibration.json",{k:v for k,v in calibration.items() if k not in ("summary","confusion")})
    paths,graph_meta=_candidate_paths(img,source,current|legacy,aorta,lad,rca,max_paths=32)
    ranking,path_map=_candidate_scores(paths,templates,img,source,calibration)
    ranking.to_csv(out/"LCX_joint_candidate_ranking.csv",index=False)
    if len(ranking):
        for rank,row in ranking.iterrows():
            p=path_map[int(row["candidate_id"])]
            pd.DataFrame(p,columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/f"candidate_{rank+1:02d}_source_path.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4.5)); q=calibration["summary"]
    ax.bar(np.arange(len(q)),q["min_margin"].to_numpy(float)); ax.axhline(0.03,linestyle="--"); ax.set_xticks(np.arange(len(q))); ax.set_xticklabels(q["recipe"],rotation=20,ha="right"); ax.set_ylabel("minimum correct-vessel margin"); ax.set_title("RCA/LAD patient-specific calibration — recipe selection"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/"01_calibration_recipe_margins.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));
    if len(ranking):
        q=ranking.head(15); x=np.arange(len(q)); ax.bar(x-0.25,q["RCA_score"],width=.25,label="RCA"); ax.bar(x,q["LAD_score"],width=.25,label="LAD"); ax.bar(x+0.25,q["LCX_score"],width=.25,label="LCX"); ax.set_xticks(x); ax.set_xticklabels([f"C{int(i)}" for i in q["candidate_id"]]); ax.legend()
    ax.set_ylabel("calibrated template score"); ax.set_title("Three-template scores for candidate source paths"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/"02_candidate_three_template_scores.png",dpi=180); plt.close(fig)
    _write_qc(out,img,source,ranking,path_map)
    status=_decision(calibration,ranking)
    top=ranking.iloc[0].to_dict() if len(ranking) else None
    summary={
        "status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"LCX_master_status":"UNRESOLVED",
        "calibration_passed":bool(calibration["passed"]),"selected_recipe":calibration["recipe"],"mm_per_curved_pixel_from_RCA":calibration["mm_per_px"],
        "RCA_positive_score":calibration["rca_positive"],"LAD_positive_score":calibration["lad_positive"],"RCA_correct_margin":calibration["rca_margin"],"LAD_correct_margin":calibration["lad_margin"],
        "graph":graph_meta,"top_candidate":top,"stale_LCX_source_centerline_used":False,
        "scientific_boundary":"RCA and LAD calibrate the classifier before LCX ranking. A passing LCX result is only a source-path candidate requiring independent anatomical/topological QC; it does not establish circumferential registration or plaque volume."
    }
    _json(out/"summary.json",summary)
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>OpenPlaque joint three-vessel classifier</title><style>body{{font-family:Arial;max-width:1200px;margin:30px auto;line-height:1.4}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px}}.warn{{background:#fff3cd;padding:12px}}</style></head><body><h1>OpenPlaque — joint RCA/LAD/LCX curved-template classifier</h1><p><b>Status:</b> {status}</p><div class="warn">RCA and LAD are the only calibration vessels. LCX is not used to select the scoring recipe. No candidate is accepted as LCX without later anatomical/topological validation.</div><h2>Calibration recipes</h2>{calibration['summary'].to_html(index=False,float_format=lambda x:f'{x:.4f}')}<h2>Known-vessel template scores</h2>{calibration['confusion'].to_html(index=False,float_format=lambda x:f'{x:.4f}')}<h2>LCX candidate ranking</h2>{ranking.head(15).to_html(index=False,float_format=lambda x:f'{x:.4f}')}<h2>Summary</h2><pre>{json.dumps(summary,indent=2,default=str)}</pre><img src="03_top_candidate_source_orthogonal_qc.png" style="max-width:100%"></body></html>'''
    report=out/"OPENPLAQUE_JOINT_THREE_VESSEL_TEMPLATE_CLASSIFIER_REPORT.html"; report.write_text(html,encoding="utf-8")
    _json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"baseline_commit":BASELINE,"algorithm":ALGORITHM})
    zip_path=out/"OPENPLAQUE_JOINT_THREE_VESSEL_TEMPLATE_CLASSIFIER_REPORT_BACK.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out.iterdir()):
            if fp.is_file() and fp != zip_path: zf.write(fp,arcname=fp.name)
    return {"summary":summary,"report":str(report),"zip":str(zip_path),"output_dir":str(out)}
