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

from openplaque.study import OpenPlaqueStudy

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-curved-template-reacquisition-v1.1-clean"
OUTPUT_DIRNAME = "LCX_Curved_Template_Reacquisition_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
AORTA = TS / "total/aorta.nii.gz"
OLD_MASK_DIR = Path("UCLA_Plaque_Context_Verification/nnunet_masks")
STUDY_ZIP = Path("Full_DICOM.zip")
SERIES = {"RCA": 1035, "LCX": 1039}

STATUS_MATCH = "LCX_CURVED_TEMPLATE_MATCHES_SOURCE_PATH_REQUIRES_ANATOMICAL_QC"
STATUS_NO_MATCH = "NO_DISTINCT_LCX_TEMPLATE_SOURCE_PATH"
STATUS_CALIBRATION_FAILED = "RCA_TEMPLATE_POSITIVE_CONTROL_FAILED"


def _json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )


def _req(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


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
    direction = np.array(
        [
            [row[0], col[0], slc[0]],
            [row[1], col[1], slc[1]],
            [row[2], col[2], slc[2]],
        ],
        float,
    )
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
        im = sitk.Resample(
            im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
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
    return map_coordinates(
        np.asarray(arr), zyx.T, order=order, mode="constant", cval=cval
    )


def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for cols in [
        ("lps_x_mm", "lps_y_mm", "lps_z_mm"),
        ("x_mm", "y_mm", "z_mm"),
    ]:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in [
        ("zyx_z", "zyx_y", "zyx_x"),
        ("source_z", "source_y", "source_x"),
        ("z", "y", "x"),
    ]:
        if all(c in d.columns for c in cols):
            return _zyx_to_xyz(ref, d[list(cols)].to_numpy(float))
    if all(c in d.columns for c in ("x", "y", "z")):
        return d[["x", "y", "z"]].to_numpy(float)
    raise ValueError(
        f"No recognized path coordinates in {path}; columns={list(d.columns)}"
    )


def _arc(points):
    p = np.asarray(points, float)
    if len(p) == 0:
        return np.array([], float)
    if len(p) == 1:
        return np.array([0.0])
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return -1.0
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / den) if den > 1e-10 else -1.0


def _fill_interp(x):
    x = np.asarray(x, float).copy()
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    if good.sum() == 1:
        x[~good] = x[good][0]
        return x
    xx = np.arange(len(x))
    x[~good] = np.interp(xx[~good], xx[good], x[good])
    return x


def _interp_unit(x, n=128):
    x = _fill_interp(x)
    if len(x) == 0:
        return np.zeros(n, float)
    if len(x) == 1:
        return np.repeat(x, n)
    q = np.linspace(0, len(x) - 1, n)
    return np.interp(q, np.arange(len(x)), x)


def _series_and_mask(study, series_number, mask_path):
    image, vol, files = study.load_series(int(series_number))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(_req(mask_path))))
    vol = np.asarray(vol)
    if mask.shape != vol.shape:
        perms = [
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        ]
        matched = next(
            (np.transpose(mask, p) for p in perms
             if np.transpose(mask, p).shape == vol.shape),
            None,
        )
        if matched is None:
            raise RuntimeError(
                f"Mask shape {mask.shape} does not match series {vol.shape}"
            )
        mask = matched
    return image, vol, mask, files


def _template_fingerprint(vol, mask):
    """Extract a 1-D longitudinal signature from a rotation-stack mask.

    Axis 0 is the 24-view rotation dimension. The longitudinal coordinate is
    chosen between in-plane axes 1 and 2 by the largest active span.
    """
    vol = np.asarray(vol, float)
    mask = np.asarray(mask)
    if vol.shape != mask.shape or vol.ndim != 3:
        raise ValueError("vol and mask must be same-shape 3-D arrays")

    candidates = []
    for long_axis in (1, 2):
        support_full = (mask > 0).sum(
            axis=tuple(ax for ax in range(3) if ax != long_axis)
        )
        active = support_full > 0
        idx = np.where(active)[0]
        if len(idx) < 8:
            continue
        start, end = int(idx.min()), int(idx.max()) + 1

        vv = np.moveaxis(vol, long_axis, -1)[..., start:end]
        mm = np.moveaxis(mask, long_axis, -1)[..., start:end]
        support = (mm > 0).sum(axis=(0, 1)).astype(float)
        plaque = (mm == 2).sum(axis=(0, 1)).astype(float)
        plaque_fraction = plaque / np.maximum(support, 1.0)

        hu_med = np.full(end - start, np.nan, float)
        hu_p90 = np.full(end - start, np.nan, float)
        for j in range(end - start):
            vals = vv[..., j][mm[..., j] > 0]
            if len(vals):
                hu_med[j] = float(np.median(vals))
                hu_p90[j] = float(np.quantile(vals, 0.90))

        span = int(end - start)
        occupancy = float((support > 0).mean())
        candidates.append(
            {
                "long_axis": int(long_axis),
                "start_px": start,
                "end_px": end,
                "span_px": span,
                "occupancy": occupancy,
                "support": support,
                "plaque": plaque,
                "plaque_fraction": plaque_fraction,
                "hu_median": hu_med,
                "hu_p90": hu_p90,
            }
        )

    if not candidates:
        raise RuntimeError("Could not determine curved-series longitudinal axis")
    return max(candidates, key=lambda r: (r["span_px"], r["occupancy"]))


def _path_profile(img, source, points):
    pts, arc = _resample_path(points, 0.25)
    hu = _sample(img, source, pts)
    offsets = np.array(
        [
            [0, 0, 0],
            [0.8, 0, 0], [-0.8, 0, 0],
            [0, 0.8, 0], [0, -0.8, 0],
            [0, 0, 0.8], [0, 0, -0.8],
            [1.2, 0, 0], [-1.2, 0, 0],
            [0, 1.2, 0], [0, -1.2, 0],
            [0, 0, 1.2], [0, 0, -1.2],
        ],
        float,
    )
    stacks = [_sample(img, source, pts + off[None, :]) for off in offsets]
    local_max = np.max(np.vstack(stacks), axis=0)
    robust = np.isfinite(hu) & (hu >= 120) & (hu <= 1200)
    return {
        "points": pts,
        "arc": arc,
        "hu": hu,
        "local_max_hu": local_max,
        "robust_fraction": float(robust.mean()) if len(robust) else 0.0,
        "median_hu": float(np.nanmedian(hu)) if len(hu) else np.nan,
        "length_mm": float(arc[-1]) if len(arc) else 0.0,
    }


def _match_score(template, profile):
    th = _interp_unit(template["hu_median"])
    tp = _interp_unit(template.get("plaque_fraction", np.zeros_like(th)))
    sh = _interp_unit(profile["hu"])
    sl = _interp_unit(profile.get("local_max_hu", profile["hu"]))

    if np.nanstd(sl) < 1e-6:
        src_landmark = np.zeros_like(sl)
    else:
        q = np.nanquantile(sl, 0.75)
        src_landmark = np.maximum(sl - q, 0.0)
        den = np.nanmax(src_landmark)
        if den > 0:
            src_landmark = src_landmark / den

    best = None
    for orientation, h, lmk in [
        ("forward", sh, src_landmark),
        ("reverse", sh[::-1], src_landmark[::-1]),
    ]:
        intensity_corr = _corr(th, h)
        plaque_corr = _corr(tp, lmk) if np.nanstd(tp) > 1e-8 else 0.0
        t_span = float(template.get("span_px", len(template["hu_median"])))
        p_len = float(profile.get("length_mm", 0.0))
        length_score = math.exp(
            -abs(math.log(max(p_len, 1.0) / max(t_span * 0.25, 1.0)))
        )
        robust = float(profile.get("robust_fraction", 0.0))
        score = (
            0.50 * ((intensity_corr + 1.0) / 2.0)
            + 0.25 * ((plaque_corr + 1.0) / 2.0)
            + 0.10 * length_score
            + 0.15 * robust
        )
        rec = {
            "score": float(score),
            "intensity_corr": float(intensity_corr),
            "plaque_landmark_corr": float(plaque_corr),
            "length_score": float(length_score),
            "orientation": orientation,
            "robust_fraction": robust,
            "median_hu": float(profile.get("median_hu", np.nan)),
            "length_mm": p_len,
        }
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best


def _decision(rca_control, wrong_controls, ranking):
    if (
        float(rca_control.get("score", -1)) < 0.45
        or float(rca_control.get("intensity_corr", -1)) < 0.25
    ):
        return STATUS_CALIBRATION_FAILED
    if ranking is None or len(ranking) == 0:
        return STATUS_NO_MATCH
    top = ranking.iloc[0]
    wrong_max = max(
        [float(x.get("score", -1)) for x in wrong_controls] + [-1.0]
    )
    if (
        float(top["score"]) >= 0.45
        and float(top["score"]) - wrong_max >= 0.10
        and float(top.get("robust_fraction", 0.0)) >= 0.65
        and 150.0 <= float(top.get("median_hu", np.nan)) <= 1000.0
    ):
        return STATUS_MATCH
    return STATUS_NO_MATCH


def synthetic_lcx_template_self_test():
    vol = np.zeros((24, 40, 90), dtype=np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[:, 18:22, 10:80] = 1
    mask[:, 16:24, 45:52] = 2
    vol[:] = 100
    vol[mask > 0] = 520
    vol[mask == 2] = 700
    fp = _template_fingerprint(vol, mask)

    t = {
        "hu_median": np.linspace(350, 700, 80),
        "plaque_fraction": np.r_[np.zeros(30), np.ones(10), np.zeros(40)],
        "span_px": 80,
    }
    good = {
        "hu": np.linspace(355, 695, 100),
        "local_max_hu": np.r_[
            np.ones(35) * 300, np.ones(15) * 700, np.ones(50) * 300
        ],
        "robust_fraction": 0.9,
        "median_hu": 525.0,
        "length_mm": 16.0,
    }
    bad = {
        "hu": np.random.default_rng(4).normal(400, 180, 100),
        "local_max_hu": np.ones(100) * 280,
        "robust_fraction": 0.5,
        "median_hu": 400.0,
        "length_mm": 16.0,
    }
    good_score = _match_score(t, good)["score"]
    bad_score = _match_score(t, bad)["score"]
    ranking = pd.DataFrame(
        [{"score": 0.50, "robust_fraction": 0.90, "median_hu": 500.0}]
    )
    status = _decision(
        {"score": 0.60, "intensity_corr": 0.50},
        [{"score": 0.20}, {"score": 0.15}],
        ranking,
    )
    return {
        "passed": bool(
            fp["long_axis"] == 2
            and fp["plaque"].max() > 0
            and good_score > bad_score
            and status == STATUS_MATCH
        ),
        "algorithm": ALGORITHM,
        "good_score": float(good_score),
        "bad_score": float(bad_score),
    }


def _distance_to_mask(img, mask, points):
    z = np.argwhere(mask)
    if len(z) == 0:
        return np.full(len(points), np.inf)
    if len(z) > 150000:
        z = z[:: int(math.ceil(len(z) / 150000))]
    phys = _zyx_to_xyz(img, z)
    return cKDTree(phys).query(np.asarray(points, float))[0]


def _orient_lad_from_aorta(img, aorta, lad):
    lad = np.asarray(lad, float)
    d = _distance_to_mask(img, aorta, np.vstack([lad[0], lad[-1]]))
    return lad if d[0] <= d[1] else lad[::-1].copy()


def _candidate_paths(img, source, coronary_union, aorta, lad, rca, max_paths=24):
    lad = _orient_lad_from_aorta(img, aorta, lad)
    lad_r, lad_arc = _resample_path(lad, 0.25)
    rca_r, _ = _resample_path(rca, 0.25)
    anchor = lad_r[0]

    skel = skeletonize(coronary_union)
    zyx = np.argwhere(skel)
    if len(zyx) < 10:
        return [], {"n_skeleton_nodes": int(len(zyx)), "reason": "too_few_nodes"}
    phys = _zyx_to_xyz(img, zyx)

    lad_tree = cKDTree(lad_r)
    rca_tree = cKDTree(rca_r)
    d_lad, j_lad = lad_tree.query(phys)
    d_rca = rca_tree.query(phys)[0]
    d_anchor = np.linalg.norm(phys - anchor[None, :], axis=1)

    near_prox_lad = (d_lad <= 1.25) & (lad_arc[j_lad] <= 4.0)
    novel = d_lad >= 1.50
    allowed = (
        (near_prox_lad | novel)
        & (d_rca >= 1.50)
        & (d_anchor <= 35.0)
    )
    keep_idx = np.where(allowed)[0]
    if len(keep_idx) < 10:
        return [], {
            "n_skeleton_nodes": int(len(zyx)),
            "n_allowed_nodes": int(len(keep_idx)),
            "reason": "too_few_allowed_nodes",
        }

    P = phys[keep_idx]
    local_tree = cKDTree(P)
    pairs = list(local_tree.query_pairs(1.50))
    if not pairs:
        return [], {
            "n_skeleton_nodes": int(len(zyx)),
            "n_allowed_nodes": int(len(P)),
            "reason": "no_graph_edges",
        }

    rows, cols, vals = [], [], []
    for i, j in pairs:
        dist = float(np.linalg.norm(P[i] - P[j]))
        mid = 0.5 * (P[i] + P[j])
        hu_mid = float(_sample(img, source, mid[None, :])[0])
        support_penalty = 1.0 if hu_mid >= 120 else 2.5
        cost = dist * support_penalty
        rows += [i, j]
        cols += [j, i]
        vals += [cost, cost]

    n = len(P)
    super_node = n
    start = local_tree.query_ball_point(anchor, 3.0)
    if not start:
        _, near = local_tree.query(anchor, k=min(5, n))
        start = np.atleast_1d(near).astype(int).tolist()
    for i in start:
        d = float(np.linalg.norm(P[i] - anchor))
        rows += [super_node, int(i)]
        cols += [int(i), super_node]
        vals += [d, d]

    graph = csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    dist, pred = dijkstra(
        graph, directed=False, indices=super_node, return_predecessors=True
    )

    euclid_anchor = np.linalg.norm(P - anchor[None, :], axis=1)
    candidate_order = np.argsort(dist[:n])[::-1]
    selected_endpoints = []
    paths = []

    for target in candidate_order:
        if len(paths) >= max_paths:
            break
        if not np.isfinite(dist[target]):
            continue
        if euclid_anchor[target] < 6.0 or dist[target] < 8.0 or dist[target] > 45.0:
            continue
        if d_lad[keep_idx[target]] < 2.0:
            continue
        if any(np.linalg.norm(P[target] - P[q]) < 3.0 for q in selected_endpoints):
            continue

        chain = []
        cur = int(target)
        guard = 0
        while cur != super_node and cur >= 0 and guard < n + 5:
            chain.append(cur)
            cur = int(pred[cur])
            guard += 1
        if cur != super_node or len(chain) < 3:
            continue
        chain = chain[::-1]
        path = np.vstack([anchor, P[np.asarray(chain, int)]])
        path, arc = _resample_path(path, 0.25)
        if len(path) < 10 or arc[-1] < 6.0:
            continue
        paths.append(path)
        selected_endpoints.append(int(target))

    meta = {
        "n_skeleton_nodes": int(len(zyx)),
        "n_allowed_nodes": int(len(P)),
        "n_graph_edges": int(len(pairs)),
        "n_start_nodes": int(len(start)),
        "n_candidate_paths": int(len(paths)),
    }
    return paths, meta


def _frame(points, i):
    p = np.asarray(points, float)
    a = max(0, i - 2)
    b = min(len(p) - 1, i + 2)
    tangent = p[b] - p[a]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ tangent))]
    normal = np.cross(tangent, seed)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    bvec = np.cross(tangent, normal)
    bvec = bvec / max(np.linalg.norm(bvec), 1e-9)
    return normal, bvec


def _plane(img, source, center, normal, bvec, half=4.5, pix=0.15):
    q = np.arange(-half, half + pix / 2.0, pix)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    pts = (
        center[None, None, :]
        + xx[..., None] * normal[None, None, :]
        + yy[..., None] * bvec[None, None, :]
    )
    vals = _sample(img, source, pts.reshape(-1, 3)).reshape(len(q), len(q))
    return vals, q


def _write_qc(out, img, source, ranked_paths):
    if not ranked_paths:
        return []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = np.asarray(axes).ravel()
    slot = 0
    for rank, path in enumerate(ranked_paths[:2], start=1):
        for frac in (0.15, 0.40, 0.65, 0.85):
            if slot >= len(axes):
                break
            i = int(round(frac * (len(path) - 1)))
            normal, bvec = _frame(path, i)
            plane, q = _plane(img, source, path[i], normal, bvec)
            ax = axes[slot]
            ax.imshow(
                plane, cmap="gray", vmin=-200, vmax=1000,
                extent=[q[0], q[-1], q[-1], q[0]],
            )
            ax.scatter([0], [0], s=28, marker="+")
            ax.set_title(f"Candidate {rank} — {frac:.0%} arc")
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
            slot += 1
    for ax in axes[slot:]:
        ax.axis("off")
    fig.suptitle(
        "Top LCX-template candidate paths: source-CCTA orthogonal QC\n"
        "Center marker = candidate path; template match does not establish LCX identity"
    )
    fig.tight_layout()
    path = out / "03_top_candidate_source_orthogonal_qc.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def _report_html(out, summary, ranking, controls):
    top_html = ranking.head(10).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    ctrl_html = controls.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OpenPlaque LCX curved-template reacquisition</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;line-height:1.4}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px}}
.warn{{background:#fff3cd;padding:12px}}</style></head><body>
<h1>OpenPlaque — LCX curved-template source reacquisition</h1>
<p><b>Status:</b> {summary["status"]}</p>
<div class="warn"><b>Scientific boundary:</b> the historical Series-1039 LCX-labeled mask is used only
as an unlabeled curved-series coronary fingerprint. No candidate is accepted as LCX by this experiment.
No stale LCX source centerline is read. No source-space plaque localization or plaque volume is claimed.</div>
<h2>Calibration and wrong-vessel controls</h2>{ctrl_html}
<h2>Candidate ranking</h2>{top_html}
<h2>Automatic source-CCTA QC</h2>
<img src="03_top_candidate_source_orthogonal_qc.png" style="max-width:100%">
<h2>Summary</h2><pre>{json.dumps(summary, indent=2, default=str)}</pre>
</body></html>"""


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    root = Path(drive_root)
    out = Path(output_root or root / OUTPUT_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)
    _json(
        out / "run_state.json",
        {"status": "STARTED", "baseline_commit": BASELINE, "algorithm": ALGORITHM},
    )

    master = json.loads(_req(root / MASTER).read_text())
    unresolved = set(master.get("unresolved", []))
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Master Anatomy v2 is not frozen")
    if "LCX" not in unresolved:
        raise RuntimeError("LCX is no longer unresolved; this experiment is stale")

    img, source = _source(root / SOURCE_CACHE)
    lad = _load_path(root / LAD, img)
    rca = _load_path(root / RCA, img)

    current = _resample_mask(root / CUR, img)
    legacy = _resample_mask(root / LEG, img)
    aorta = _resample_mask(root / AORTA, img)
    coronary_union = current | legacy

    study = OpenPlaqueStudy(
        str(_req(root / STUDY_ZIP)),
        extract_root="/content/openplaque_lcx_template_dicom",
    )
    _, rca_vol, rca_mask, _ = _series_and_mask(
        study, SERIES["RCA"], root / OLD_MASK_DIR / "RCA.nii.gz"
    )
    _, lcx_vol, lcx_mask, _ = _series_and_mask(
        study, SERIES["LCX"], root / OLD_MASK_DIR / "LCX.nii.gz"
    )
    rca_template = _template_fingerprint(rca_vol, rca_mask)
    lcx_template = _template_fingerprint(lcx_vol, lcx_mask)

    rca_profile = _path_profile(img, source, rca)
    lad_profile = _path_profile(img, source, lad)
    rca_control = _match_score(rca_template, rca_profile)
    wrong_rca = _match_score(lcx_template, rca_profile)
    wrong_lad = _match_score(lcx_template, lad_profile)
    wrong_controls = [wrong_rca, wrong_lad]

    controls = pd.DataFrame(
        [
            {"comparison": "RCA template -> accepted RCA", "role": "positive_control", **rca_control},
            {"comparison": "LCX template -> accepted RCA", "role": "wrong_vessel_control", **wrong_rca},
            {"comparison": "LCX template -> accepted LAD", "role": "wrong_vessel_control", **wrong_lad},
        ]
    )
    controls.to_csv(out / "template_control_scores.csv", index=False)

    paths, graph_meta = _candidate_paths(
        img, source, coronary_union, aorta, lad, rca, max_paths=24
    )

    rows = []
    scored_paths = []
    for i, path in enumerate(paths):
        prof = _path_profile(img, source, path)
        score = _match_score(lcx_template, prof)
        rca_score = _match_score(rca_template, prof)
        rec = {
            "candidate_id": int(i),
            **score,
            "RCA_template_score": float(rca_score["score"]),
            "LCX_minus_RCA_template_score": float(score["score"] - rca_score["score"]),
        }
        rows.append(rec)
        scored_paths.append((score["score"], i, path))

    ranking = pd.DataFrame(rows)
    if len(ranking):
        ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        ranking = pd.DataFrame(
            columns=[
                "candidate_id", "score", "intensity_corr", "plaque_landmark_corr",
                "length_score", "orientation", "robust_fraction", "median_hu",
                "length_mm", "RCA_template_score", "LCX_minus_RCA_template_score",
            ]
        )
    ranking.to_csv(out / "LCX_template_candidate_ranking.csv", index=False)

    ordered_paths = []
    if len(ranking):
        by_id = {i: p for _, i, p in scored_paths}
        for rank, row in ranking.iterrows():
            path = by_id[int(row.candidate_id)]
            ordered_paths.append(path)
            pd.DataFrame(
                path, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]
            ).to_csv(
                out / f"candidate_{rank+1:02d}_source_path.csv", index=False
            )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(_interp_unit(lcx_template["hu_median"]), label="LCX curved template median HU")
    axes[0].plot(_interp_unit(rca_template["hu_median"]), label="RCA curved template median HU")
    axes[0].set_ylabel("HU")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(_interp_unit(lcx_template["plaque_fraction"]), label="LCX template plaque fraction")
    axes[1].plot(_interp_unit(rca_template["plaque_fraction"]), label="RCA template plaque fraction")
    axes[1].set_xlabel("Normalized longitudinal position")
    axes[1].set_ylabel("Plaque-support fraction")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Historical curved-series templates used only as longitudinal fingerprints")
    fig.tight_layout()
    fig.savefig(out / "01_curved_template_fingerprints.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(ranking):
        q = ranking.head(12)
        ax.bar(np.arange(len(q)), q["score"].to_numpy(float))
        ax.axhline(float(wrong_rca["score"]), linestyle="--", label="LCX template -> accepted RCA")
        ax.axhline(float(wrong_lad["score"]), linestyle=":", label="LCX template -> accepted LAD")
        ax.set_xticks(np.arange(len(q)))
        ax.set_xticklabels([f"C{int(x)+1}" for x in q["candidate_id"]])
        ax.legend()
    ax.set_ylabel("Template match score")
    ax.set_title("LCX-template candidate ranking vs wrong-vessel controls")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "02_candidate_match_scores.png", dpi=180)
    plt.close(fig)

    _write_qc(out, img, source, ordered_paths)

    status = _decision(rca_control, wrong_controls, ranking)
    top = ranking.iloc[0].to_dict() if len(ranking) else None
    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "LCX_master_status": "UNRESOLVED",
        "stale_LCX_source_centerline_used": False,
        "RCA_positive_control": rca_control,
        "LCX_wrong_vessel_controls": {
            "accepted_RCA": wrong_rca,
            "accepted_LAD": wrong_lad,
        },
        "graph": graph_meta,
        "top_candidate": top,
        "scientific_boundary": (
            "A passing template match is only a source-path reacquisition candidate. "
            "It does not establish LCX identity, circumferential registration, "
            "source-space plaque localization, or plaque volume."
        ),
    }
    _json(out / "summary.json", summary)

    report = out / "OPENPLAQUE_LCX_CURVED_TEMPLATE_REACQUISITION_REPORT.html"
    report.write_text(
        _report_html(out, summary, ranking, controls), encoding="utf-8"
    )

    _json(
        out / "run_state.json",
        {
            "status": "COMPLETE",
            "scientific_status": status,
            "baseline_commit": BASELINE,
            "algorithm": ALGORITHM,
        },
    )

    zip_path = out / "OPENPLAQUE_LCX_CURVED_TEMPLATE_REACQUISITION_REPORT_BACK.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out.iterdir()):
            if fp.is_file() and fp != zip_path:
                zf.write(fp, arcname=fp.name)

    return {
        "summary": summary,
        "report": str(report),
        "zip": str(zip_path),
        "output_dir": str(out),
    }
