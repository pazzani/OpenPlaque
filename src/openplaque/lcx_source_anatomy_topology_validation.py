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

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-source-anatomy-topology-validation-v1.0"
OUTPUT_DIRNAME = "LCX_Source_Anatomy_Topology_Validation_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "total/aorta.nii.gz"
PRIOR = Path("Joint_Three_Vessel_Template_Classifier_v1")
PRIOR_RANKING = PRIOR / "LCX_joint_candidate_ranking.csv"

STATUS_NO_CANDIDATE = "NO_SOURCE_ANATOMY_LCX_CANDIDATE"
STATUS_AMBIGUOUS = "SOURCE_ANATOMY_LCX_CANDIDATES_AMBIGUOUS"
STATUS_CANDIDATE = "SOURCE_ANATOMY_LCX_CANDIDATE_REQUIRES_VISUAL_QC"


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
    raise ValueError(f"No recognized coordinates in {path}; columns={list(d.columns)}")


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


def _mask_tree(img, mask, limit=200000):
    z = np.argwhere(mask)
    if len(z) == 0:
        return None
    if len(z) > limit:
        z = z[::int(math.ceil(len(z) / limit))]
    return cKDTree(_zyx_to_xyz(img, z))


def _dist_tree(tree, points):
    if tree is None:
        return np.full(len(points), np.inf)
    return tree.query(np.asarray(points, float))[0]


def _orient_lad_from_aorta(lad, aorta_tree):
    d0, d1 = _dist_tree(aorta_tree, np.vstack([lad[0], lad[-1]]))
    return lad if d0 <= d1 else lad[::-1].copy()


def _orient_candidate(path, proximal_lad_point):
    p = np.asarray(path, float)
    if np.linalg.norm(p[-1] - proximal_lad_point) < np.linalg.norm(p[0] - proximal_lad_point):
        p = p[::-1].copy()
    return p


def _first_sustained(mask, n=5):
    x = np.asarray(mask, bool)
    if len(x) < n:
        return None
    run = np.convolve(x.astype(int), np.ones(n, int), mode="valid")
    idx = np.where(run == n)[0]
    return int(idx[0]) if len(idx) else None


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle_deg(a, b):
    aa, bb = _unit(a), _unit(b)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _local_tangent(points, i, span=6):
    p = np.asarray(points, float)
    a = max(0, int(i) - span)
    b = min(len(p) - 1, int(i) + span)
    return _unit(p[b] - p[a])


def _branch_angle(candidate, div_i, lad, lad_tree):
    div_pt = candidate[int(div_i)]
    _, j = lad_tree.query(div_pt)
    cand_t = _local_tangent(candidate, min(len(candidate)-1, int(div_i)+4), span=4)
    lad_t = _local_tangent(lad, int(j), span=6)
    return _angle_deg(cand_t, lad_t), int(j)


def _smoothness_score(points):
    p, _ = _resample_path(points, 0.5)
    if len(p) < 5:
        return 0.0, 180.0
    v = np.diff(p, axis=0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    dots = np.clip(np.sum(v[:-1] * v[1:], axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    p90 = float(np.quantile(ang, 0.90)) if len(ang) else 180.0
    return float(np.clip(1.0 - p90 / 75.0, 0.0, 1.0)), p90


def _score_prox_divergence(mm):
    if not np.isfinite(mm):
        return 0.0
    if 0.5 <= mm <= 4.0:
        return 1.0
    if mm < 0.5:
        return float(np.clip(mm / 0.5, 0.0, 1.0))
    return float(np.clip(1.0 - (mm - 4.0) / 3.0, 0.0, 1.0))


def _score_angle(deg):
    if not np.isfinite(deg):
        return 0.0
    rise = np.clip((deg - 15.0) / 35.0, 0.0, 1.0)
    fall = np.clip((165.0 - deg) / 35.0, 0.0, 1.0)
    return float(min(rise, fall))


def _anatomy_metrics(img, source, path, lad, rca, trees):
    lad_tree = cKDTree(lad)
    rca_tree = cKDTree(rca)
    p, arc = _resample_path(path, 0.25)
    p = _orient_candidate(p, lad[0])
    arc = _arc(p)

    d_lad = lad_tree.query(p)[0]
    d_rca = rca_tree.query(p)[0]
    d_cur = _dist_tree(trees["current"], p)
    d_leg = _dist_tree(trees["legacy"], p)
    d_aorta = _dist_tree(trees["aorta"], p)
    hu = _sample(img, source, p)

    div_i = _first_sustained(d_lad > 1.5, n=5)
    divergence_found = div_i is not None
    if div_i is None:
        div_i = max(0, min(len(p)-1, int(round(0.2 * len(p)))))
        divergence_arc = float("nan")
        angle = float("nan")
        lad_j = 0
    else:
        divergence_arc = float(arc[div_i])
        angle, lad_j = _branch_angle(p, div_i, lad, lad_tree)

    post = np.arange(len(p)) >= int(div_i)
    distal = np.arange(len(p)) >= max(int(div_i), int(round(0.55 * len(p))))
    if not np.any(post):
        post[:] = True
    if not np.any(distal):
        distal = post.copy()

    cur_support = d_cur <= 1.0
    leg_support = d_leg <= 1.0
    union_support = cur_support | leg_support
    intersection_support = cur_support & leg_support
    robust_hu = np.isfinite(hu) & (hu >= 120) & (hu <= 1200)

    smooth_score, turn_p90 = _smoothness_score(p)
    straight = float(np.linalg.norm(p[-1] - p[0]))
    length = float(arc[-1]) if len(arc) else 0.0
    tortuosity = float(length / max(straight, 1e-6))

    row = {
        "length_mm": length,
        "divergence_found": bool(divergence_found),
        "divergence_arc_mm": divergence_arc,
        "branch_angle_deg": float(angle),
        "branch_nearest_lad_index": int(lad_j),
        "endpoint_lad_separation_mm": float(d_lad[-1]),
        "distal_median_lad_separation_mm": float(np.median(d_lad[distal])),
        "postdiv_median_lad_separation_mm": float(np.median(d_lad[post])),
        "postdiv_rejoin_fraction": float(np.mean(d_lad[post] < 1.5)),
        "min_rca_separation_mm": float(np.min(d_rca[post])),
        "median_rca_separation_mm": float(np.median(d_rca[post])),
        "aorta_exit_fraction": float(np.mean(d_aorta[post] > 1.0)),
        "endpoint_aorta_separation_mm": float(d_aorta[-1]),
        "current_support_fraction": float(np.mean(cur_support[post])),
        "legacy_support_fraction": float(np.mean(leg_support[post])),
        "union_support_fraction": float(np.mean(union_support[post])),
        "intersection_support_fraction": float(np.mean(intersection_support[post])),
        "robust_hu_fraction": float(np.mean(robust_hu[post])),
        "median_hu": float(np.median(hu[post])),
        "tortuosity": tortuosity,
        "turn_angle_p90_deg": float(turn_p90),
        "smoothness_score": smooth_score,
    }

    prox = _score_prox_divergence(divergence_arc)
    ang_score = _score_angle(angle)
    distal_score = float(np.clip((row["distal_median_lad_separation_mm"] - 2.0) / 6.0, 0.0, 1.0))
    rca_score = float(np.clip((row["min_rca_separation_mm"] - 3.0) / 5.0, 0.0, 1.0))
    anatomy_score = (
        0.24 * row["intersection_support_fraction"]
        + 0.18 * ang_score
        + 0.18 * distal_score
        + 0.14 * prox
        + 0.10 * row["aorta_exit_fraction"]
        + 0.07 * rca_score
        + 0.05 * smooth_score
        + 0.04 * row["robust_hu_fraction"]
    )
    row.update({
        "proximal_divergence_score": prox,
        "branch_angle_score": ang_score,
        "distal_separation_score": distal_score,
        "rca_separation_score": rca_score,
        "anatomy_score": float(anatomy_score),
    })

    gates = {
        "gate_divergence_found": bool(divergence_found),
        "gate_divergence_proximal": bool(divergence_found and divergence_arc <= 5.0),
        "gate_branch_angle": bool(divergence_found and 25.0 <= angle <= 155.0),
        "gate_union_support": bool(row["union_support_fraction"] >= 0.90),
        "gate_dual_mask_support": bool(row["intersection_support_fraction"] >= 0.60),
        "gate_distal_lad_separation": bool(row["distal_median_lad_separation_mm"] >= 3.0 and row["endpoint_lad_separation_mm"] >= 4.0),
        "gate_no_lad_rejoin": bool(row["postdiv_rejoin_fraction"] <= 0.20),
        "gate_rca_separation": bool(row["min_rca_separation_mm"] >= 3.0),
        "gate_aorta_exit": bool(row["aorta_exit_fraction"] >= 0.65),
        "gate_source_contrast": bool(row["robust_hu_fraction"] >= 0.85),
        "gate_tortuosity": bool(row["tortuosity"] <= 2.25),
    }
    gates["anatomy_gate_pass"] = bool(all(gates.values()))
    return row, gates, {"points": p, "arc": arc, "d_lad": d_lad, "d_rca": d_rca, "d_aorta": d_aorta, "hu": hu, "div_i": int(div_i)}


def _load_hypotheses(root, img, top_n=5):
    ranking = pd.read_csv(_req(root / PRIOR_RANKING))
    if len(ranking) < top_n:
        raise RuntimeError(f"Need at least {top_n} prior candidate rows; found {len(ranking)}")
    rows = []
    for rank in range(top_n):
        meta = ranking.iloc[rank].to_dict()
        fp = root / PRIOR / f"candidate_{rank+1:02d}_source_path.csv"
        path = _load_path(fp, img)
        rows.append((rank + 1, int(meta["candidate_id"]), path, meta, fp))
    return rows, ranking


def _decision(metrics):
    q = metrics.sort_values("anatomy_score", ascending=False).reset_index(drop=True)
    passing = q[q["anatomy_gate_pass"] == True].copy()
    if len(passing) == 0:
        return STATUS_NO_CANDIDATE
    top = passing.iloc[0]
    if float(top["anatomy_score"]) < 0.65:
        return STATUS_NO_CANDIDATE
    if len(passing) > 1:
        margin = float(passing.iloc[0]["anatomy_score"] - passing.iloc[1]["anatomy_score"])
        if margin < 0.05:
            return STATUS_AMBIGUOUS
    return STATUS_CANDIDATE


def _frame(points, i):
    p = np.asarray(points, float)
    a = max(0, i - 3)
    b = min(len(p) - 1, i + 3)
    tangent = _unit(p[b] - p[a])
    seed = np.eye(3)[np.argmin(np.abs(np.eye(3) @ tangent))]
    n = _unit(np.cross(tangent, seed))
    bvec = _unit(np.cross(tangent, n))
    return n, bvec


def _plane(img, source, center, n, b, half=5.0, step=0.25):
    q = np.arange(-half, half + 1e-9, step)
    xx, yy = np.meshgrid(q, q)
    pts = center[None,None,:] + xx[...,None] * n[None,None,:] + yy[...,None] * b[None,None,:]
    vals = _sample(img, source, pts.reshape(-1,3)).reshape(len(q),len(q))
    return vals, q


def _write_figures(out, metrics, details, lad, rca, img, source):
    q = metrics.sort_values("anatomy_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(q))
    ax.bar(x, q["anatomy_score"].to_numpy(float), label="anatomy score")
    ax.scatter(x, q["prior_LCX_margin"].to_numpy(float), marker="o", label="prior LCX-template margin (support only)")
    ax.axhline(0.65, linestyle="--", linewidth=1, label="anatomy candidate threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(v)}" for v in q["source_candidate_id"]])
    ax.set_ylabel("score / margin")
    ax.set_title("LCX source-space anatomy ranking — template score excluded from anatomy score")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_candidate_anatomy_scores.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(q), 1, figsize=(11, 2.2*len(q)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, q.iterrows()):
        cid = int(row["source_candidate_id"])
        d = details[cid]
        ax.plot(d["arc"], d["d_lad"], label="distance to accepted LAD")
        ax.plot(d["arc"], d["d_rca"], label="distance to accepted RCA")
        ax.plot(d["arc"], d["d_aorta"], label="distance to aorta")
        ax.axvline(d["arc"][d["div_i"]], linestyle="--", linewidth=1)
        ax.set_ylabel(f"C{cid}\nmm")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("candidate arc length (mm)")
    fig.suptitle("Candidate topology profiles; dashed line = sustained LAD divergence")
    fig.tight_layout()
    fig.savefig(out / "02_candidate_distance_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    dims = [(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]
    lad_r,_ = _resample_path(lad,0.25); rca_r,_ = _resample_path(rca,0.25)
    for ax,(a,b,xl,yl) in zip(axes,dims):
        ax.plot(lad_r[:,a],lad_r[:,b],linewidth=3,label="accepted LAD")
        ax.plot(rca_r[:,a],rca_r[:,b],linewidth=3,label="accepted RCA")
        for _,row in q.head(5).iterrows():
            cid=int(row["source_candidate_id"]); p=details[cid]["points"]
            ax.plot(p[:,a],p[:,b],linewidth=1.5,label=f"C{cid}")
            di=details[cid]["div_i"]; ax.scatter([p[di,a]],[p[di,b]],s=20)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.axis("equal"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Source-space geometry projections; dots mark sustained divergence from accepted LAD")
    fig.tight_layout()
    fig.savefig(out / "03_source_geometry_projections.png", dpi=180)
    plt.close(fig)

    top = q.head(min(3,len(q)))
    fig, axes = plt.subplots(len(top), 4, figsize=(14, 3.4*len(top)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(top.iterrows()):
        cid=int(row["source_candidate_id"]); d=details[cid]; p=d["points"]; div=d["div_i"]
        post_n=max(1,len(p)-div-1)
        inds=[div, min(len(p)-1,div+int(.25*post_n)), min(len(p)-1,div+int(.60*post_n)), min(len(p)-1,div+int(.90*post_n))]
        labels=["divergence","25% post-div","60% post-div","90% post-div"]
        for c,(i,label) in enumerate(zip(inds,labels)):
            n,b=_frame(p,i); plane,qq=_plane(img,source,p[i],n,b)
            ax=axes[r,c]; ax.imshow(plane,cmap="gray",vmin=-200,vmax=1000,extent=[qq[0],qq[-1],qq[-1],qq[0]])
            ax.scatter([0],[0],marker="+",s=30)
            ax.set_title(f"C{cid} {label}")
            ax.set_xlabel("mm"); ax.set_ylabel("mm")
    fig.suptitle("Top anatomy-ranked candidates — source CCTA orthogonal QC")
    fig.tight_layout()
    fig.savefig(out / "04_top_candidates_orthogonal_qc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def synthetic_anatomy_topology_self_test():
    lad = np.column_stack([np.linspace(0,20,81), np.zeros(81), np.zeros(81)])
    cand = np.column_stack([np.linspace(0,14,57), np.r_[np.zeros(9), np.linspace(0,10,48)], np.zeros(57)])
    c,_ = _resample_path(cand,0.25)
    d = cKDTree(lad).query(c)[0]
    div = _first_sustained(d > 1.5, 5)
    assert div is not None
    angle,_ = _branch_angle(c,div,lad,cKDTree(lad))
    assert 20 < angle < 90
    assert _score_angle(angle) > 0
    return {"passed": True, "divergence_index": int(div), "angle_deg": float(angle), "algorithm": ALGORITHM}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    root = Path(drive_root)
    out = Path(output_root or root / OUTPUT_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "run_state.json", {"status":"STARTED","baseline_commit":BASELINE,"algorithm":ALGORITHM})

    master = json.loads(_req(root / MASTER).read_text())
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Master coronary anatomy v2 is not frozen")
    if "LCX" not in set(master.get("unresolved", [])):
        raise RuntimeError("LCX is no longer unresolved; experiment is stale")

    img, source = _source(root / SOURCE_CACHE)
    current = _resample_mask(root / CUR, img)
    legacy = _resample_mask(root / LEG, img)
    aorta = _resample_mask(root / AORTA, img)
    trees = {"current":_mask_tree(img,current),"legacy":_mask_tree(img,legacy),"aorta":_mask_tree(img,aorta)}

    lad = _load_path(root / LAD_PATH, img)
    lad = _orient_lad_from_aorta(lad, trees["aorta"])
    lad,_ = _resample_path(lad,0.25)
    rca = _load_path(root / RCA_PATH, img)
    rca,_ = _resample_path(rca,0.25)

    hypotheses, prior_ranking = _load_hypotheses(root, img, top_n=5)
    rows=[]; details={}
    for prior_rank,cid,path,meta,fp in hypotheses:
        metrics,gates,det = _anatomy_metrics(img,source,path,lad,rca,trees)
        row={
            "prior_rank":int(prior_rank),
            "source_candidate_id":int(cid),
            "prior_LCX_score":float(meta.get("LCX_score",np.nan)),
            "prior_LAD_score":float(meta.get("LAD_score",np.nan)),
            "prior_RCA_score":float(meta.get("RCA_score",np.nan)),
            "prior_LCX_margin":float(meta.get("LCX_margin",np.nan)),
            "candidate_file":str(fp.relative_to(root)),
            **metrics, **gates,
        }
        rows.append(row); details[int(cid)] = det

    metrics_df=pd.DataFrame(rows).sort_values("anatomy_score",ascending=False).reset_index(drop=True)
    metrics_df.insert(0,"anatomy_rank",np.arange(1,len(metrics_df)+1))
    metrics_df.to_csv(out / "candidate_anatomy_topology_metrics.csv",index=False)
    _write_figures(out,metrics_df,details,lad,rca,img,source)

    status=_decision(metrics_df)
    passing=metrics_df[metrics_df["anatomy_gate_pass"]==True].copy()
    top=metrics_df.iloc[0].to_dict() if len(metrics_df) else None
    top_passing=passing.iloc[0].to_dict() if len(passing) else None
    summary={
        "status":status,
        "algorithm":ALGORITHM,
        "baseline_commit":BASELINE,
        "master_status":master.get("status"),
        "LCX_master_status":"UNRESOLVED",
        "n_hypotheses":int(len(metrics_df)),
        "n_anatomy_gate_pass":int(len(passing)),
        "top_anatomy_ranked":top,
        "top_passing_candidate":top_passing,
        "template_similarity_used_in_anatomy_score":False,
        "prior_template_evidence_role":"supporting metadata only",
        "stale_LCX_source_centerline_used":False,
        "scientific_boundary":"This experiment validates source-space anatomy/topology of previously nominated paths. Even a passing result remains an LCX candidate requiring visual and independent anatomical confirmation because LM/LCX identity is unresolved in Master v2. It does not establish plaque localization or plaque volume."
    }
    _json(out / "summary.json",summary)

    cols=["anatomy_rank","source_candidate_id","anatomy_score","anatomy_gate_pass","divergence_arc_mm","branch_angle_deg","intersection_support_fraction","distal_median_lad_separation_mm","endpoint_lad_separation_mm","min_rca_separation_mm","aorta_exit_fraction","prior_LCX_margin"]
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>OpenPlaque LCX source anatomy/topology validation</title><style>body{{font-family:Arial;max-width:1250px;margin:30px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px}}.warn{{background:#fff3cd;padding:12px}}</style></head><body><h1>OpenPlaque — LCX source-space anatomy/topology validation</h1><p><b>Status:</b> {status}</p><div class="warn">Primary ranking is source-space anatomy only. Prior curved-template LCX scores are shown as supporting metadata and are excluded from the anatomy score and anatomy gates.</div><h2>Candidate summary</h2>{metrics_df[cols].to_html(index=False,float_format=lambda x:f'{x:.4f}')}<h2>All metrics and gates</h2>{metrics_df.to_html(index=False,float_format=lambda x:f'{x:.4f}')}<h2>Summary</h2><pre>{json.dumps(summary,indent=2,default=str)}</pre><img src="01_candidate_anatomy_scores.png" style="max-width:100%"><img src="02_candidate_distance_profiles.png" style="max-width:100%"><img src="03_source_geometry_projections.png" style="max-width:100%"><img src="04_top_candidates_orthogonal_qc.png" style="max-width:100%"></body></html>'''
    report=out / "OPENPLAQUE_LCX_SOURCE_ANATOMY_TOPOLOGY_VALIDATION_REPORT.html"
    report.write_text(html,encoding="utf-8")

    _json(out / "run_state.json", {"status":"COMPLETE","scientific_status":status,"baseline_commit":BASELINE,"algorithm":ALGORITHM})
    zip_path=out / "OPENPLAQUE_LCX_SOURCE_ANATOMY_TOPOLOGY_VALIDATION_REPORT_BACK.zip"
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out.iterdir()):
            if fp.is_file() and fp != zip_path:
                zf.write(fp,arcname=fp.name)
    return {"summary":summary,"metrics":metrics_df,"report":str(report),"zip":str(zip_path),"output_dir":str(out)}
