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
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-coronary-aorta-constrained-geodesic-bridge-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Aorta_Constrained_Geodesic_Bridge_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
LCX_COMMON_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")

CHAMBER_DIR = Path("TotalSegmentator_Cardiovascular_Cache_v1/heartchambers_highres")
AORTA_MASK = CHAMBER_DIR / "aorta.nii.gz"
PULMONARY_MASK = CHAMBER_DIR / "pulmonary_artery.nii.gz"
CHAMBER_MASKS = (
    CHAMBER_DIR / "heart_atrium_left.nii.gz",
    CHAMBER_DIR / "heart_atrium_right.nii.gz",
    CHAMBER_DIR / "heart_ventricle_left.nii.gz",
    CHAMBER_DIR / "heart_ventricle_right.nii.gz",
)

EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"

CONTROL_START_ARC_MM = 10.0
ROOT_Z_HALF_MM = 14.0
ROOT_RADIAL_LIMIT_MM = 38.0
ROI_PAD_MM = 10.0
CHAMBER_DILATION_MM = 0.8

COST_VARIANTS = (
    {"name": "hu140", "bright_threshold_hu": 140.0, "hu_weight": 4.0, "centrality_weight": 2.0},
    {"name": "hu180", "bright_threshold_hu": 180.0, "hu_weight": 4.5, "centrality_weight": 2.5},
)

RESAMPLE_PATH_STEP_MM = 0.50
QC_N_ANGLES = 48
QC_RADIAL_STEP_MM = 0.12
QC_MAX_RADIUS_MM = 4.0
QC_MIN_RADIUS_MM = 0.50

MIN_CONTROL_QC = 0.75
MAX_CONTROL_MEDIAN_DIST_MM = 1.5
MAX_CONTROL_P90_DIST_MM = 3.0
MAX_CONTROL_TARGET_DIST_MM = 3.5

MIN_LAD_QC = 0.65
MIN_LAD_HIGH_HU_FRACTION = 0.70
MAX_LAD_TORTUOSITY = 1.8
MAX_VARIANT_MEDIAN_SEP_MM = 2.0
MAX_VARIANT_P90_SEP_MM = 4.0
MAX_VARIANT_TARGET_SEP_MM = 4.0

STATUS_CONTROL_FAIL = "LEFT_CORONARY_AORTA_GEODESIC_RCA_CONTROL_FAILED"
STATUS_NO_STABLE_BRIDGE = "LEFT_CORONARY_AORTA_GEODESIC_NO_STABLE_LAD_ROOT_BRIDGE"
STATUS_CANDIDATE = "LEFT_CORONARY_AORTA_CONSTRAINED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED"


def _req(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(path):
    return json.loads(_req(path).read_text(encoding="utf-8"))


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_lps_csv(path):
    d = pd.read_csv(_req(path))
    candidates = (
        ("lps_x_mm", "lps_y_mm", "lps_z_mm"),
        ("x_mm", "y_mm", "z_mm"),
        ("x", "y", "z"),
    )
    for cols in candidates:
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {path}: {list(d.columns)}")


def _arc(points):
    p = np.asarray(points, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp_path(points, q):
    p = np.asarray(points, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample_path(points, step=RESAMPLE_PATH_STEP_MM):
    p = np.asarray(points, float)
    a = _arc(p)
    if len(a) == 0 or a[-1] == 0:
        return p.copy(), a
    q = np.arange(0.0, float(a[-1]) + 1e-9, float(step))
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp_path(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[int(np.argmin(np.abs(axes @ t)))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row = iop[:3]
        col = iop[3:]
        slc = np.cross(row, col)
        self.D = np.array(
            [
                [row[0], col[0], slc[0]],
                [row[1], col[1], slc[1]],
                [row[2], col[2], slc[2]],
            ],
            float,
        )
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]

    def zyx_to_xyz(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz_idx = pts[:, ::-1]
        return self.origin + (xyz_idx * self.spacing_xyz) @ self.D.T


def _load_source(root):
    cache = root / SOURCE_CACHE
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = SourceGeometry(meta)
    return geom, src, meta


def _sitk_reference(shape_zyx, geom):
    ref = sitk.Image([int(v) for v in shape_zyx[::-1]], sitk.sitkUInt8)
    ref.SetSpacing(tuple(float(v) for v in geom.spacing_xyz))
    ref.SetOrigin(tuple(float(v) for v in geom.origin))
    ref.SetDirection(tuple(float(v) for v in geom.D.ravel()))
    return ref


def _load_mask(path, source_shape, geom):
    im = sitk.ReadImage(str(_req(path)))
    ref = _sitk_reference(source_shape, geom)
    same = (
        im.GetSize() == ref.GetSize()
        and np.allclose(im.GetSpacing(), ref.GetSpacing())
        and np.allclose(im.GetOrigin(), ref.GetOrigin())
        and np.allclose(im.GetDirection(), ref.GetDirection())
    )
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(im) > 0
    if tuple(arr.shape) != tuple(source_shape):
        raise RuntimeError(f"Mask shape mismatch after resampling: {path}: {arr.shape} vs {source_shape}")
    return arr


def _sample_source(geom, src, pts, cval=-1024.0):
    zyx = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), zyx.T, order=1, mode="constant", cval=float(cval))


def _surface_outside(mask):
    return ndi.binary_dilation(mask, iterations=1) & (~mask)


def _surface_points(mask, geom):
    idx = np.argwhere(_surface_outside(mask))
    return idx, geom.zyx_to_xyz(idx)


def _nearest_surface_info(point_xyz, surf_idx, surf_xyz):
    tree = cKDTree(surf_xyz)
    d, i = tree.query(np.asarray(point_xyz, float), k=1)
    return float(d), surf_idx[int(i)], surf_xyz[int(i)]


def _orient_path_proximal_first(path, surface_xyz):
    p = np.asarray(path, float)
    tree = cKDTree(surface_xyz)
    d0 = float(tree.query(p[0])[0])
    d1 = float(tree.query(p[-1])[0])
    if d0 <= d1:
        return p.copy(), d0, d1
    return p[::-1].copy(), d1, d0


def _root_target_mask(aorta, geom, rca_root_xyz):
    surf = _surface_outside(aorta)
    idx = np.argwhere(surf)
    xyz = geom.zyx_to_xyz(idx)
    dz = np.abs(xyz[:, 2] - float(rca_root_xyz[2]))
    radial = np.linalg.norm(xyz - np.asarray(rca_root_xyz, float)[None, :], axis=1)
    keep = (dz <= ROOT_Z_HALF_MM) & (radial <= ROOT_RADIAL_LIMIT_MM)
    out = np.zeros_like(aorta, dtype=bool)
    kidx = idx[keep]
    out[kidx[:, 0], kidx[:, 1], kidx[:, 2]] = True
    if not np.any(out):
        raise RuntimeError("No aortic-root target surface voxels after root-band restriction")
    return out


def _crop_bounds(source_shape, geom, start_xyz, target_mask, pad_mm=ROI_PAD_MM):
    target_idx = np.argwhere(target_mask)
    start_zyx = geom.xyz_to_zyx([start_xyz])[0]
    lo = np.minimum(np.min(target_idx, axis=0), np.floor(start_zyx).astype(int))
    hi = np.maximum(np.max(target_idx, axis=0), np.ceil(start_zyx).astype(int))
    pad = np.ceil(float(pad_mm) / geom.spacing_zyx).astype(int)
    lo = np.maximum(lo - pad, 0).astype(int)
    hi = np.minimum(hi + pad + 1, np.asarray(source_shape)).astype(int)
    return lo, hi


def _dilate_mm(mask, spacing_zyx, mm):
    if mm <= 0:
        return mask.copy()
    dist = ndi.distance_transform_edt(~mask, sampling=spacing_zyx)
    return dist <= float(mm)


def _prepare_cost(src_crop, spacing_zyx, forbidden, aorta_crop, target_crop, variant):
    hu = np.asarray(src_crop, np.float32)
    threshold = float(variant["bright_threshold_hu"])
    bright = hu >= threshold
    centrality = ndi.distance_transform_edt(bright, sampling=spacing_zyx)
    centrality_score = np.clip(centrality / 1.5, 0.0, 1.0)

    hu_score = np.clip((hu - 80.0) / 520.0, 0.0, 1.0)
    cost = (
        1.0
        + float(variant["hu_weight"]) * (1.0 - hu_score)
        + float(variant["centrality_weight"]) * (1.0 - centrality_score)
    )
    cost = cost.astype(np.float32)
    cost[hu < 0] += 10.0
    cost[hu < -200] += 20.0

    inner_aorta = aorta_crop & (~target_crop)
    blocked = forbidden | inner_aorta
    cost[blocked] = np.inf
    cost[target_crop] = np.minimum(cost[target_crop], 1.0)
    return cost


def _mcp_path(cost, spacing_zyx, source_idx, target_mask):
    from skimage.graph import MCP_Geometric

    mcp = MCP_Geometric(cost, sampling=tuple(float(v) for v in spacing_zyx))
    targets = np.argwhere(target_mask)
    if len(targets) == 0:
        raise RuntimeError("Empty target mask")
    costs, _ = mcp.find_costs(
        starts=[tuple(int(v) for v in source_idx)],
        ends=[tuple(int(v) for v in row) for row in targets],
        find_all_ends=False,
    )
    vals = costs[target_mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise RuntimeError("No finite geodesic path to aortic-root target")
    target_candidates = targets[finite]
    target_costs = vals[finite]
    target = target_candidates[int(np.argmin(target_costs))]
    trace = np.asarray(mcp.traceback(tuple(int(v) for v in target)), int)
    return trace, target, float(np.min(target_costs))


def _smooth_xyz(path_xyz):
    p = np.asarray(path_xyz, float)
    if len(p) < 5:
        return p.copy()
    out = p.copy()
    for k in range(3):
        x = p[:, k]
        kernel = np.ones(5, float) / 5.0
        sm = np.convolve(x, kernel, mode="same")
        sm[:2] = x[:2]
        sm[-2:] = x[-2:]
        out[:, k] = sm
    out[0] = p[0]
    out[-1] = p[-1]
    return out


def _path_tangents(path_xyz):
    p = np.asarray(path_xyz, float)
    if len(p) < 2:
        return np.zeros_like(p)
    g = np.gradient(p, axis=0)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-9)


def _plane_qc_station(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2 * np.pi, QC_N_ANGLES, endpoint=False)
    dirs = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
    center_pts = np.vstack([c, c + 0.15*u, c - 0.15*u, c + 0.15*v, c - 0.15*v])
    center_hu = float(np.median(_sample_source(geom, src, center_pts)))
    threshold = float(np.clip(0.55 * center_hu, 200.0, 500.0))
    rr = np.arange(0.2, QC_MAX_RADIUS_MM + 1e-9, QC_RADIAL_STEP_MM)
    pts = c[None, None, :] + dirs[:, None, :] * rr[None, :, None]
    hu = _sample_source(geom, src, pts.reshape(-1, 3)).reshape(QC_N_ANGLES, len(rr))

    radii = np.full(QC_N_ANGLES, np.nan)
    for i in range(QC_N_ANGLES):
        below = (hu[i] < threshold) & (rr >= QC_MIN_RADIUS_MM)
        ix = np.flatnonzero(below[:-1] & below[1:])
        if len(ix):
            radii[i] = rr[int(ix[0])]

    valid = np.isfinite(radii)
    vf = float(valid.mean())
    if valid.any():
        med = float(np.median(radii[valid]))
        radii[~valid] = med
        p10, p50, p90 = np.percentile(radii, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
    else:
        p10 = p50 = p90 = axis = np.nan

    passed = bool(
        center_hu >= 160.0
        and vf >= 0.50
        and np.isfinite(axis)
        and axis <= 3.0
    )
    return {
        "center_hu": center_hu,
        "valid_radial_fraction": vf,
        "lumen_radius_median_mm": float(p50),
        "axis_proxy": float(axis),
        "plane_qc_pass": passed,
    }


def _path_metrics(geom, src, path_xyz, target_xyz, evaluate_tail_exclusion_mm=1.5):
    p, arc = _resample_path(path_xyz, RESAMPLE_PATH_STEP_MM)
    tangents = _path_tangents(p)
    center_hu = _sample_source(geom, src, p)
    total_len = float(arc[-1]) if len(arc) else 0.0
    straight = float(np.linalg.norm(p[-1] - p[0])) if len(p) >= 2 else 0.0
    tort = total_len / max(straight, 1e-6)

    rows = []
    for i, (c, t, a) in enumerate(zip(p, tangents, arc)):
        q = _plane_qc_station(geom, src, c, t)
        rows.append({"station_index": i, "arc_mm": float(a), **q})
    qdf = pd.DataFrame(rows)
    if total_len > 2 * evaluate_tail_exclusion_mm:
        use = qdf[
            (qdf.arc_mm >= evaluate_tail_exclusion_mm)
            & (qdf.arc_mm <= total_len - evaluate_tail_exclusion_mm)
        ].copy()
    else:
        use = qdf.copy()

    return {
        "path_length_mm": total_len,
        "straight_distance_mm": straight,
        "tortuosity": float(tort),
        "center_hu_median": float(np.median(center_hu)),
        "center_hu_p10": float(np.percentile(center_hu, 10)),
        "high_hu_fraction_ge180": float(np.mean(center_hu >= 180.0)),
        "plane_qc_fraction": float(use.plane_qc_pass.mean()) if len(use) else 0.0,
        "plane_qc_count": int(len(use)),
        "target_lps_mm": [float(v) for v in target_xyz],
        "qdf": qdf,
        "resampled_path": p,
    }


def _symmetric_path_separation(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ta = cKDTree(a)
    tb = cKDTree(b)
    da = tb.query(a)[0]
    db = ta.query(b)[0]
    both = np.r_[da, db]
    return {
        "median_mm": float(np.median(both)),
        "p90_mm": float(np.percentile(both, 90)),
        "max_mm": float(np.max(both)),
        "endpoint_distance_mm": float(np.linalg.norm(a[-1] - b[-1])),
    }


def _known_path_distance(candidate, known):
    candidate = np.asarray(candidate, float)
    known = np.asarray(known, float)
    tree = cKDTree(known)
    d = tree.query(candidate)[0]
    return {
        "median_mm": float(np.median(d)),
        "p90_mm": float(np.percentile(d, 90)),
        "max_mm": float(np.max(d)),
    }


def _common_path_posthoc(candidate, root, common_path):
    if common_path is None:
        return {"available": False}
    p = np.asarray(candidate, float)
    c = np.asarray(common_path, float)
    tree = cKDTree(c)
    d = tree.query(p)[0]
    return {
        "available": True,
        "minimum_distance_mm": float(np.min(d)),
        "median_distance_mm": float(np.median(d)),
        "fraction_within_2mm": float(np.mean(d <= 2.0)),
        "root_to_common_min_distance_mm": float(np.min(np.linalg.norm(c - np.asarray(root)[None, :], axis=1))),
    }


def _plot_mpr_summary(src, geom, rca_known, control_paths, lad_known, lad_paths, root_xyz, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title, known, paths in (
        (axes[0], "RCA positive control", rca_known, control_paths),
        (axes[1], "LAD-to-aortic-root candidate", lad_known, lad_paths),
    ):
        pts = [known]
        pts.extend(paths.values())
        allp = np.vstack(pts)
        xy = allp[:, :2]
        lo = xy.min(axis=0) - 5
        hi = xy.max(axis=0) + 5

        z = float(np.median(allp[:, 2]))
        nx = ny = 220
        xs = np.linspace(lo[0], hi[0], nx)
        ys = np.linspace(lo[1], hi[1], ny)
        xx, yy = np.meshgrid(xs, ys)
        sample = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
        im = _sample_source(geom, src, sample).reshape(ny, nx)
        ax.imshow(im, cmap="gray", vmin=-100, vmax=850, extent=[xs[0], xs[-1], ys[-1], ys[0]])
        ax.plot(known[:, 0], known[:, 1], linewidth=2.5, label="known path")
        for name, p in paths.items():
            ax.plot(p[:, 0], p[:, 1], linewidth=2, label=name)
        ax.scatter([root_xyz[0]], [root_xyz[1]], s=35, marker="x", label="RCA root reference")
        ax.set_title(title + f" | LPS z~{z:.1f}")
        ax.set_xlabel("LPS x (mm)")
        ax.set_ylabel("LPS y (mm)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_3d(rca_known, control_paths, lad_known, lad_paths, target_xyzs, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rca_known[:, 0], rca_known[:, 1], rca_known[:, 2], linewidth=2.5, label="known RCA")
    ax.plot(lad_known[:, 0], lad_known[:, 1], lad_known[:, 2], linewidth=2.5, label="frozen LAD")
    for name, p in control_paths.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=2, label=f"RCA {name}")
    for name, p in lad_paths.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=2, label=f"LAD {name}")
    for name, pt in target_xyzs.items():
        ax.scatter([pt[0]], [pt[1]], [pt[2]], s=35, label=f"target {name}")
    ax.set_title("Aorta-constrained geodesic bridge experiment")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_qc(qc_by, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, qdf in qc_by.items():
        ax.plot(qdf.arc_mm, qdf.center_hu, label=name)
    ax.axhline(180.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Path arc (mm)")
    ax.set_ylabel("Source center HU")
    ax.set_title("Geodesic-path source-CCTA center intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _run_one(
    label,
    start_xyz,
    root_target_mask,
    source_shape,
    geom,
    src,
    aorta,
    forbidden,
    variant,
):
    lo, hi = _crop_bounds(source_shape, geom, start_xyz, root_target_mask)
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    src_crop = np.asarray(src[sl])
    aorta_crop = aorta[sl]
    forbidden_crop = forbidden[sl]
    target_crop = root_target_mask[sl]

    start_global = np.rint(geom.xyz_to_zyx([start_xyz])[0]).astype(int)
    start_local = start_global - lo
    start_local = np.clip(start_local, 0, np.asarray(src_crop.shape) - 1)

    # Guarantee that the exact source voxel and a tiny neighborhood are not blocked.
    rr = np.indices(src_crop.shape)
    dvox = np.sqrt(
        ((rr[0] - start_local[0]) * geom.spacing_zyx[0]) ** 2
        + ((rr[1] - start_local[1]) * geom.spacing_zyx[1]) ** 2
        + ((rr[2] - start_local[2]) * geom.spacing_zyx[2]) ** 2
    )
    forbidden_crop = forbidden_crop.copy()
    forbidden_crop[dvox <= 1.0] = False

    cost = _prepare_cost(
        src_crop,
        geom.spacing_zyx,
        forbidden_crop,
        aorta_crop,
        target_crop,
        variant,
    )
    cost[tuple(start_local)] = min(float(cost[tuple(start_local)]), 1.0)
    trace_local, target_local, total_cost = _mcp_path(cost, geom.spacing_zyx, start_local, target_crop)
    trace_global = trace_local + lo[None, :]
    xyz = geom.zyx_to_xyz(trace_global)
    xyz = _smooth_xyz(xyz)
    target_global = target_local + lo
    target_xyz = geom.zyx_to_xyz([target_global])[0]
    metrics = _path_metrics(geom, src, xyz, target_xyz)
    metrics["geodesic_total_cost"] = total_cost
    metrics["crop_lo_zyx"] = lo.tolist()
    metrics["crop_hi_zyx"] = hi.tolist()
    return metrics


def synthetic_self_test():
    p = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    q, a = _resample_path(p, 0.5)
    assert len(q) == 5
    assert np.isclose(a[-1], 2.0)
    sep = _symmetric_path_separation(q, q + np.array([0.2, 0, 0]))
    assert sep["median_mm"] <= 0.3
    return {"ok": True, "resampled_points": len(q)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    if master.get("status") != EXPECTED_MASTER_STATUS:
        raise RuntimeError(f"Unexpected frozen master status: {master.get('status')}")

    geom, src, _ = _load_source(root)
    aorta = _load_mask(root / AORTA_MASK, src.shape, geom)
    pulmonary = _load_mask(root / PULMONARY_MASK, src.shape, geom)
    chambers = np.zeros(src.shape, dtype=bool)
    for p in CHAMBER_MASKS:
        chambers |= _load_mask(root / p, src.shape, geom)

    forbidden = chambers | pulmonary
    forbidden = _dilate_mm(forbidden, geom.spacing_zyx, CHAMBER_DILATION_MM)

    surf_idx, surf_xyz = _surface_points(aorta, geom)

    rca_raw = _load_lps_csv(root / RCA_PATH)
    lad_raw = _load_lps_csv(root / LAD_PATH)
    rca, rca_prox_dist, rca_other_dist = _orient_path_proximal_first(rca_raw, surf_xyz)
    lad, lad_prox_dist, lad_other_dist = _orient_path_proximal_first(lad_raw, surf_xyz)

    rca_arc = _arc(rca)
    control_start_arc = min(CONTROL_START_ARC_MM, max(2.0, float(rca_arc[-1]) * 0.25))
    rca_control_start = _interp_path(rca, [control_start_arc])[0]
    _, _, rca_root_xyz = _nearest_surface_info(rca[0], surf_idx, surf_xyz)

    root_target = _root_target_mask(aorta, geom, rca_root_xyz)

    lad_start = lad[0]
    root_target_xyz = geom.zyx_to_xyz(np.argwhere(root_target))
    lad_start_to_root = float(cKDTree(root_target_xyz).query(lad_start)[0])

    control_results = {}
    control_paths = {}
    control_qc = {}
    for v in COST_VARIANTS:
        m = _run_one(
            "RCA_control",
            rca_control_start,
            root_target,
            src.shape,
            geom,
            src,
            aorta,
            forbidden,
            v,
        )
        p = m.pop("resampled_path")
        qdf = m.pop("qdf")
        control_paths[v["name"]] = p
        control_qc[v["name"]] = qdf
        known_prox = rca[rca_arc <= control_start_arc + 2.0]
        kd = _known_path_distance(p, known_prox)
        target_dist = float(np.linalg.norm(p[-1] - rca_root_xyz))
        m["distance_to_known_RCA"] = kd
        m["target_distance_to_known_RCA_root_mm"] = target_dist
        control_results[v["name"]] = m
        pd.DataFrame(p, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
            out / f"RCA_control_path_{v['name']}.csv", index=False
        )
        qdf.to_csv(out / f"RCA_control_QC_{v['name']}.csv", index=False)

    control_sep = _symmetric_path_separation(
        control_paths[COST_VARIANTS[0]["name"]],
        control_paths[COST_VARIANTS[1]["name"]],
    )
    control_pass_each = []
    for name, m in control_results.items():
        control_pass_each.append(
            m["plane_qc_fraction"] >= MIN_CONTROL_QC
            and m["distance_to_known_RCA"]["median_mm"] <= MAX_CONTROL_MEDIAN_DIST_MM
            and m["distance_to_known_RCA"]["p90_mm"] <= MAX_CONTROL_P90_DIST_MM
            and m["target_distance_to_known_RCA_root_mm"] <= MAX_CONTROL_TARGET_DIST_MM
        )
    control_pass = bool(
        all(control_pass_each)
        and control_sep["median_mm"] <= MAX_VARIANT_MEDIAN_SEP_MM
        and control_sep["p90_mm"] <= MAX_VARIANT_P90_SEP_MM
        and control_sep["endpoint_distance_mm"] <= MAX_VARIANT_TARGET_SEP_MM
    )

    lad_results = {}
    lad_paths = {}
    lad_qc = {}
    if control_pass:
        for v in COST_VARIANTS:
            m = _run_one(
                "LAD_target",
                lad_start,
                root_target,
                src.shape,
                geom,
                src,
                aorta,
                forbidden,
                v,
            )
            p = m.pop("resampled_path")
            qdf = m.pop("qdf")
            lad_paths[v["name"]] = p
            lad_qc[v["name"]] = qdf
            lad_results[v["name"]] = m
            pd.DataFrame(p, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
                out / f"LAD_aortic_bridge_{v['name']}.csv", index=False
            )
            qdf.to_csv(out / f"LAD_aortic_bridge_QC_{v['name']}.csv", index=False)

        lad_sep = _symmetric_path_separation(
            lad_paths[COST_VARIANTS[0]["name"]],
            lad_paths[COST_VARIANTS[1]["name"]],
        )

        lad_pass_each = [
            m["plane_qc_fraction"] >= MIN_LAD_QC
            and m["high_hu_fraction_ge180"] >= MIN_LAD_HIGH_HU_FRACTION
            and m["tortuosity"] <= MAX_LAD_TORTUOSITY
            for m in lad_results.values()
        ]
        stable_lad = bool(
            all(lad_pass_each)
            and lad_sep["median_mm"] <= MAX_VARIANT_MEDIAN_SEP_MM
            and lad_sep["p90_mm"] <= MAX_VARIANT_P90_SEP_MM
            and lad_sep["endpoint_distance_mm"] <= MAX_VARIANT_TARGET_SEP_MM
        )
    else:
        lad_sep = None
        stable_lad = False

    common_path = None
    if (root / LCX_COMMON_PATH).exists():
        common_path = _load_lps_csv(root / LCX_COMMON_PATH)

    posthoc = {}
    if stable_lad:
        for name, p in lad_paths.items():
            posthoc[name] = _common_path_posthoc(p, p[-1], common_path)

    if not control_pass:
        status = STATUS_CONTROL_FAIL
    elif stable_lad:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_NO_STABLE_BRIDGE

    gate_rows = [
        {"gate": "RCA_control_pass", "pass": control_pass},
        {"gate": "LAD_variant_stability_pass", "pass": stable_lad if control_pass else False},
    ]
    pd.DataFrame(gate_rows).to_csv(out / "geodesic_bridge_gates.csv", index=False)

    if control_paths and lad_paths:
        _plot_mpr_summary(
            src, geom, rca, control_paths, lad, lad_paths, rca_root_xyz,
            out / "01_geodesic_bridge_source_MPR.png",
        )
        _plot_3d(
            rca, control_paths, lad, lad_paths,
            {name: p[-1] for name, p in lad_paths.items()},
            out / "02_geodesic_bridge_3D.png",
        )
        qc_all = {}
        for name, qdf in control_qc.items():
            qc_all[f"RCA_{name}"] = qdf
        for name, qdf in lad_qc.items():
            qc_all[f"LAD_{name}"] = qdf
        _plot_qc(qc_all, out / "03_geodesic_bridge_center_HU_QC.png")
    elif control_paths:
        _plot_3d(
            rca, control_paths, lad, {},
            {name: p[-1] for name, p in control_paths.items()},
            out / "02_geodesic_bridge_3D.png",
        )
        _plot_qc(
            {f"RCA_{k}": v for k, v in control_qc.items()},
            out / "03_geodesic_bridge_center_HU_QC.png",
        )

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "inputs": {
            "RCA_centerline": str(root / RCA_PATH),
            "LAD_centerline": str(root / LAD_PATH),
            "aorta_mask": str(root / AORTA_MASK),
            "chamber_masks": [str(root / p) for p in CHAMBER_MASKS],
            "pulmonary_mask": str(root / PULMONARY_MASK),
        },
        "anatomy": {
            "RCA_proximal_endpoint_distance_to_aorta_surface_mm": rca_prox_dist,
            "RCA_other_endpoint_distance_to_aorta_surface_mm": rca_other_dist,
            "LAD_closest_endpoint_distance_to_aorta_surface_mm": lad_prox_dist,
            "LAD_other_endpoint_distance_to_aorta_surface_mm": lad_other_dist,
            "LAD_start_distance_to_root_target_mm": lad_start_to_root,
            "RCA_control_start_arc_mm": control_start_arc,
        },
        "RCA_control": {
            "pass": control_pass,
            "variant_results": control_results,
            "cross_variant_agreement": control_sep,
        },
        "LAD_bridge": {
            "attempted": control_pass,
            "stable_candidate_pass": stable_lad,
            "variant_results": lad_results,
            "cross_variant_agreement": lad_sep,
            "posthoc_distance_to_C6_common_path": posthoc,
        },
        "clinical_LM_established": False,
        "research_bridge_candidate_established": bool(status == STATUS_CANDIDATE),
        "scientific_boundary": (
            "This experiment tests whether a chamber-excluded, aorta-constrained high-HU geodesic can connect the accepted frozen LAD endpoint to the aortic-root surface after the same method first recovers a known RCA positive control. "
            "A positive result establishes only a source-supported left-coronary proximal bridge candidate. It does not establish clinical left-main identity because common bifurcation continuity to independently established LCX/OM anatomy is not proven, and it does not modify the frozen master anatomy."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(
        out / "input_provenance.json",
        {
            "baseline": BASELINE,
            "source_cache": str(root / SOURCE_CACHE),
            "master": str(root / MASTER),
            "LAD": str(root / LAD_PATH),
            "RCA": str(root / RCA_PATH),
            "LCX_common_optional": str(root / LCX_COMMON_PATH),
            "aorta": str(root / AORTA_MASK),
        },
    )

    report = out / "OPENPLAQUE_LEFT_CORONARY_AORTA_CONSTRAINED_GEODESIC_BRIDGE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Left-Coronary Aorta-Constrained Geodesic Bridge v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>RCA positive control pass: {control_pass}</p>"
        f"<p>LAD closest frozen endpoint to aortic surface: {lad_prox_dist:.2f} mm.</p>"
        f"<p>LAD distance to restricted aortic-root target surface: {lad_start_to_root:.2f} mm.</p>"
        f"<p>Stable LAD-to-root bridge candidate: {stable_lad}</p>"
        "<p><b>Boundary:</b> Even a positive bridge is not clinical LM. It is only a source-supported proximal left-coronary bridge candidate; frozen anatomy remains unchanged.</p>"
        + ('<img src="01_geodesic_bridge_source_MPR.png" style="max-width:100%">' if (out / "01_geodesic_bridge_source_MPR.png").exists() else "")
        + ('<img src="02_geodesic_bridge_3D.png" style="max-width:100%">' if (out / "02_geodesic_bridge_3D.png").exists() else "")
        + ('<img src="03_geodesic_bridge_center_HU_QC.png" style="max-width:100%">' if (out / "03_geodesic_bridge_center_HU_QC.png").exists() else "")
        + "</body></html>",
        encoding="utf-8",
    )

    _write_json(
        out / "run_state.json",
        {"status": "COMPLETE", "result_status": status, "algorithm": ALGORITHM, "baseline": BASELINE},
    )

    zpath = out / "OPENPLAQUE_LEFT_CORONARY_AORTA_CONSTRAINED_GEODESIC_BRIDGE_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)

    return {"summary": summary, "report": str(report), "zip": str(zpath)}
