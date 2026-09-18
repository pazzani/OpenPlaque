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
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-coronary-local-root-directed-bridge-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Local_Root_Directed_Bridge_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA_PATH = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")

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
LOCAL_TARGET_RADIUS_MM = 8.0
CORRIDOR_RADIUS_MM = 12.0
ROI_PAD_MM = 15.0
KNOWN_BACKTRACK_EXCLUSION_MM = 1.5
KNOWN_PATH_EXCLUSION_RADIUS_MM = 1.5
DEEP_ANATOMY_EROSION_MM = 0.8

COST_VARIANTS = (
    {"name": "hu140", "bright_threshold_hu": 140.0, "hu_weight": 4.0, "centrality_weight": 2.0},
    {"name": "hu180", "bright_threshold_hu": 180.0, "hu_weight": 4.5, "centrality_weight": 2.5},
)

RESAMPLE_PATH_STEP_MM = 0.50
QC_N_ANGLES = 48
QC_RADIAL_STEP_MM = 0.12
QC_MAX_RADIUS_MM = 4.0
QC_MIN_RADIUS_MM = 0.50

PROGRESS_STEP_TOL_MM = 0.35
MIN_PROGRESS_STEP_FRACTION = 0.80
MAX_PROGRESS_BACKTRACK_MM = 2.0

MIN_CONTROL_QC = 0.75
MAX_CONTROL_MEDIAN_DIST_MM = 1.5
MAX_CONTROL_P90_DIST_MM = 3.0
MAX_CONTROL_TARGET_DIST_MM = 3.5
MAX_CONTROL_TORTUOSITY = 1.35

MIN_LAD_QC = 0.65
MIN_LAD_HIGH_HU_FRACTION = 0.70
MAX_LAD_TORTUOSITY = 1.60
MAX_VARIANT_MEDIAN_SEP_MM = 2.0
MAX_VARIANT_P90_SEP_MM = 4.0
MAX_VARIANT_ENDPOINT_SEP_MM = 4.0
MAX_TARGET_FROM_LOCAL_NEAREST_MM = LOCAL_TARGET_RADIUS_MM + 1.5

STATUS_CONTROL_FAIL = "LEFT_CORONARY_LOCAL_ROOT_DIRECTED_RCA_CONTROL_FAILED"
STATUS_NO_BRIDGE = "LEFT_CORONARY_LOCAL_ROOT_DIRECTED_NO_STABLE_BRIDGE"
STATUS_CANDIDATE = "LEFT_CORONARY_LOCAL_ROOT_DIRECTED_BRIDGE_CANDIDATE_SOURCE_SUPPORTED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_lps_csv(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm"), ("x", "y", "z")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp_path(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample_path(p, step=RESAMPLE_PATH_STEP_MM):
    p = np.asarray(p, float)
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
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array(
            [[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]],
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
    return SourceGeometry(meta), src


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


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


def _surface_outside(mask):
    return ndi.binary_dilation(mask, iterations=1) & (~mask)


def _surface_cloud(aorta, geom):
    idx = np.argwhere(_surface_outside(aorta))
    xyz = geom.zyx_to_xyz(idx)
    return idx, xyz, cKDTree(xyz)


def _orient_proximal_first(path, surface_tree):
    p = np.asarray(path, float)
    d0 = float(surface_tree.query(p[0])[0])
    d1 = float(surface_tree.query(p[-1])[0])
    return (p.copy(), d0, d1) if d0 <= d1 else (p[::-1].copy(), d1, d0)


def _nearest_surface_point(point_xyz, surf_idx, surf_xyz, surf_tree):
    d, i = surf_tree.query(np.asarray(point_xyz, float), k=1)
    return float(d), surf_idx[int(i)], surf_xyz[int(i)]


def _local_target_mask(aorta, geom, nearest_surface_xyz, radius_mm=LOCAL_TARGET_RADIUS_MM):
    surf = _surface_outside(aorta)
    idx = np.argwhere(surf)
    xyz = geom.zyx_to_xyz(idx)
    d = np.linalg.norm(xyz - np.asarray(nearest_surface_xyz, float)[None, :], axis=1)
    keep = d <= float(radius_mm)
    out = np.zeros_like(aorta, dtype=bool)
    kidx = idx[keep]
    if len(kidx) == 0:
        raise RuntimeError("No local aortic-surface target voxels")
    out[kidx[:, 0], kidx[:, 1], kidx[:, 2]] = True
    return out


def _line_distance(points_xyz, a_xyz, b_xyz):
    p = np.asarray(points_xyz, float)
    a = np.asarray(a_xyz, float)
    b = np.asarray(b_xyz, float)
    ab = b - a
    den = max(float(ab @ ab), 1e-9)
    t = np.clip(((p - a) @ ab) / den, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(p - proj, axis=1)


def _crop_bounds(shape, geom, start_xyz, target_mask, pad_mm=ROI_PAD_MM):
    target_idx = np.argwhere(target_mask)
    start_zyx = geom.xyz_to_zyx([start_xyz])[0]
    lo = np.minimum(np.min(target_idx, axis=0), np.floor(start_zyx).astype(int))
    hi = np.maximum(np.max(target_idx, axis=0), np.ceil(start_zyx).astype(int))
    pad = np.ceil(float(pad_mm) / geom.spacing_zyx).astype(int)
    lo = np.maximum(lo - pad, 0).astype(int)
    hi = np.minimum(hi + pad + 1, np.asarray(shape)).astype(int)
    return lo, hi


def _erode_mm(mask, spacing_zyx, mm):
    """Keep only deep interior of anatomy masks for hard exclusion.

    Coronary arteries lie immediately adjacent to cardiac structures, so dilating
    blood-pool/arterial masks can create an artificial impenetrable barrier.
    Deep-interior erosion preserves the scientific exclusion (the path may not
    traverse a chamber or pulmonary-artery lumen) while not blocking the true
    epicardial boundary neighborhood.
    """
    if mm <= 0:
        return mask.copy()
    inside = ndi.distance_transform_edt(mask, sampling=spacing_zyx)
    return mask & (inside >= float(mm))


def _known_path_exclusion(crop_xyz, known_downstream_xyz, radius_mm=KNOWN_PATH_EXCLUSION_RADIUS_MM):
    if known_downstream_xyz is None or len(known_downstream_xyz) == 0:
        return np.zeros(len(crop_xyz), dtype=bool)
    tree = cKDTree(np.asarray(known_downstream_xyz, float))
    d = tree.query(crop_xyz)[0]
    return d <= float(radius_mm)


def _prepare_cost(src_crop, spacing_zyx, corridor_mask, forbidden, aorta_crop, target_crop, variant):
    hu = np.asarray(src_crop, np.float32)
    bright = hu >= float(variant["bright_threshold_hu"])
    centrality = ndi.distance_transform_edt(bright, sampling=spacing_zyx)
    centrality_score = np.clip(centrality / 1.5, 0.0, 1.0)
    hu_score = np.clip((hu - 80.0) / 520.0, 0.0, 1.0)

    cost = (
        1.0
        + float(variant["hu_weight"]) * (1.0 - hu_score)
        + float(variant["centrality_weight"]) * (1.0 - centrality_score)
    ).astype(np.float32)

    cost[hu < 0] += 10.0
    cost[hu < -200] += 20.0

    blocked = forbidden | (~corridor_mask) | (aorta_crop & (~target_crop))
    cost[blocked] = np.inf
    cost[target_crop] = np.minimum(cost[target_crop], 1.0)
    return cost


def _mcp(cost, spacing_zyx, start_idx, target_mask):
    from skimage.graph import MCP_Geometric

    targets = np.argwhere(target_mask)
    if len(targets) == 0:
        raise RuntimeError("Empty local target mask")

    mcp = MCP_Geometric(cost, sampling=tuple(float(v) for v in spacing_zyx))
    costs, _ = mcp.find_costs(
        starts=[tuple(int(v) for v in start_idx)],
        ends=[tuple(int(v) for v in row) for row in targets],
        find_all_ends=False,
    )
    vals = costs[target_mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise RuntimeError("No finite root-directed path to local aortic target")

    finite_targets = targets[finite]
    finite_costs = vals[finite]
    target = finite_targets[int(np.argmin(finite_costs))]
    trace = np.asarray(mcp.traceback(tuple(int(v) for v in target)), int)
    return trace, target, float(np.min(finite_costs))


def _smooth_xyz(path):
    p = np.asarray(path, float)
    if len(p) < 5:
        return p.copy()
    out = p.copy()
    kernel = np.ones(5, float) / 5.0
    for k in range(3):
        sm = np.convolve(p[:, k], kernel, mode="same")
        sm[:2] = p[:2, k]
        sm[-2:] = p[-2:, k]
        out[:, k] = sm
    out[0] = p[0]
    out[-1] = p[-1]
    return out


def _tangents(path):
    p = np.asarray(path, float)
    g = np.gradient(p, axis=0)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def _plane_qc(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2*np.pi, QC_N_ANGLES, endpoint=False)
    dirs = np.cos(th)[:, None]*u + np.sin(th)[:, None]*v

    center_pts = np.vstack([c, c+.15*u, c-.15*u, c+.15*v, c-.15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55*center_hu, 200.0, 500.0))

    rr = np.arange(.2, QC_MAX_RADIUS_MM + 1e-9, QC_RADIAL_STEP_MM)
    pts = c[None, None, :] + dirs[:, None, :]*rr[None, :, None]
    hu = _sample(geom, src, pts.reshape(-1, 3)).reshape(QC_N_ANGLES, len(rr))

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
        p50 = axis = np.nan

    passed = bool(center_hu >= 160 and vf >= .50 and np.isfinite(axis) and axis <= 3.0)
    return {
        "center_hu": center_hu,
        "valid_radial_fraction": vf,
        "lumen_radius_median_mm": float(p50),
        "axis_proxy": float(axis),
        "plane_qc_pass": passed,
    }


def _surface_progress(path_xyz, surf_tree):
    p = np.asarray(path_xyz, float)
    d = surf_tree.query(p)[0].astype(float)
    diff = np.diff(d)
    frac = float(np.mean(diff <= PROGRESS_STEP_TOL_MM)) if len(diff) else 1.0
    runmin = np.minimum.accumulate(d)
    backtrack = float(np.max(d - runmin)) if len(d) else 0.0
    return {
        "start_surface_distance_mm": float(d[0]),
        "end_surface_distance_mm": float(d[-1]),
        "net_surface_progress_mm": float(d[0] - d[-1]),
        "progress_step_fraction": frac,
        "max_backtrack_from_running_min_mm": backtrack,
        "distance_series_mm": d,
    }


def _path_metrics(geom, src, path_xyz, surf_tree):
    p, arc = _resample_path(path_xyz, RESAMPLE_PATH_STEP_MM)
    t = _tangents(p)
    center_hu = _sample(geom, src, p)
    total = float(arc[-1]) if len(arc) else 0.0
    straight = float(np.linalg.norm(p[-1] - p[0])) if len(p) >= 2 else 0.0

    rows = []
    for i, (c, tt, a) in enumerate(zip(p, t, arc)):
        rows.append({"station_index": i, "arc_mm": float(a), **_plane_qc(geom, src, c, tt)})
    qdf = pd.DataFrame(rows)

    use = qdf.copy()
    if total > 3.0:
        use = qdf[(qdf.arc_mm >= 1.5) & (qdf.arc_mm <= total - 1.5)].copy()

    prog = _surface_progress(p, surf_tree)
    return {
        "path_length_mm": total,
        "straight_distance_mm": straight,
        "tortuosity": total / max(straight, 1e-6),
        "center_hu_median": float(np.median(center_hu)),
        "center_hu_p10": float(np.percentile(center_hu, 10)),
        "high_hu_fraction_ge180": float(np.mean(center_hu >= 180.0)),
        "plane_qc_fraction": float(use.plane_qc_pass.mean()) if len(use) else 0.0,
        "plane_qc_count": int(len(use)),
        "progress": {k: v for k, v in prog.items() if k != "distance_series_mm"},
        "surface_distance_series_mm": prog["distance_series_mm"],
        "qdf": qdf,
        "resampled_path": p,
    }


def _symmetric_separation(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    d = np.r_[da, db]
    return {
        "median_mm": float(np.median(d)),
        "p90_mm": float(np.percentile(d, 90)),
        "max_mm": float(np.max(d)),
        "endpoint_distance_mm": float(np.linalg.norm(a[-1] - b[-1])),
    }


def _known_path_distance(candidate, known):
    d = cKDTree(np.asarray(known, float)).query(np.asarray(candidate, float))[0]
    return {
        "median_mm": float(np.median(d)),
        "p90_mm": float(np.percentile(d, 90)),
        "max_mm": float(np.max(d)),
    }


def _posthoc_to_c6(candidate, c6):
    d = cKDTree(np.asarray(c6, float)).query(np.asarray(candidate, float))[0]
    return {
        "minimum_distance_mm": float(np.min(d)),
        "median_distance_mm": float(np.median(d)),
        "fraction_within_2mm": float(np.mean(d <= 2.0)),
        "fraction_within_3mm": float(np.mean(d <= 3.0)),
    }


def _run_one(
    geom,
    src,
    aorta,
    forbidden_global,
    surf_tree,
    start_xyz,
    nearest_surface_xyz,
    target_mask,
    known_downstream_xyz,
    variant,
):
    lo, hi = _crop_bounds(src.shape, geom, start_xyz, target_mask)
    sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    src_crop = np.asarray(src[sl])
    aorta_crop = aorta[sl]
    target_crop = target_mask[sl]
    forbidden = forbidden_global[sl].copy()

    zz, yy, xx = np.indices(src_crop.shape)
    global_zyx = np.stack([zz + lo[0], yy + lo[1], xx + lo[2]], axis=-1).reshape(-1, 3).astype(float)
    xyz = geom.zyx_to_xyz(global_zyx)

    corridor_flat = _line_distance(xyz, start_xyz, nearest_surface_xyz) <= CORRIDOR_RADIUS_MM
    corridor = corridor_flat.reshape(src_crop.shape)

    known_flat = _known_path_exclusion(xyz, known_downstream_xyz)
    known_block = known_flat.reshape(src_crop.shape)
    forbidden |= known_block

    start_global = np.rint(geom.xyz_to_zyx([start_xyz])[0]).astype(int)
    start_local = np.clip(start_global - lo, 0, np.asarray(src_crop.shape) - 1)

    # Unmask only the immediate 1-mm launch ball around the exact start.
    dist_start = np.linalg.norm(xyz - np.asarray(start_xyz)[None, :], axis=1).reshape(src_crop.shape)
    forbidden[dist_start <= 1.0] = False
    corridor[dist_start <= 1.0] = True

    cost = _prepare_cost(
        src_crop,
        geom.spacing_zyx,
        corridor,
        forbidden,
        aorta_crop,
        target_crop,
        variant,
    )
    cost[tuple(start_local)] = min(float(cost[tuple(start_local)]), 1.0)

    trace_local, target_local, total_cost = _mcp(cost, geom.spacing_zyx, start_local, target_crop)
    trace_global = trace_local + lo[None, :]
    raw_xyz = geom.zyx_to_xyz(trace_global)
    sm_xyz = _smooth_xyz(raw_xyz)
    metrics = _path_metrics(geom, src, sm_xyz, surf_tree)
    metrics["geodesic_total_cost"] = float(total_cost)
    metrics["target_lps_mm"] = [float(v) for v in geom.zyx_to_xyz([target_local + lo])[0]]
    metrics["target_distance_from_local_nearest_surface_mm"] = float(
        np.linalg.norm(np.asarray(metrics["target_lps_mm"]) - np.asarray(nearest_surface_xyz))
    )
    metrics["crop_lo_zyx"] = lo.tolist()
    metrics["crop_hi_zyx"] = hi.tolist()
    metrics["corridor_radius_mm"] = CORRIDOR_RADIUS_MM
    metrics["known_backtrack_exclusion_radius_mm"] = KNOWN_PATH_EXCLUSION_RADIUS_MM
    return metrics


def _control_gate(m, known_dist, target_to_root):
    p = m["progress"]
    return bool(
        m["plane_qc_fraction"] >= MIN_CONTROL_QC
        and m["tortuosity"] <= MAX_CONTROL_TORTUOSITY
        and known_dist["median_mm"] <= MAX_CONTROL_MEDIAN_DIST_MM
        and known_dist["p90_mm"] <= MAX_CONTROL_P90_DIST_MM
        and target_to_root <= MAX_CONTROL_TARGET_DIST_MM
        and p["progress_step_fraction"] >= MIN_PROGRESS_STEP_FRACTION
        and p["max_backtrack_from_running_min_mm"] <= MAX_PROGRESS_BACKTRACK_MM
    )


def _lad_gate(m):
    p = m["progress"]
    return bool(
        m["plane_qc_fraction"] >= MIN_LAD_QC
        and m["high_hu_fraction_ge180"] >= MIN_LAD_HIGH_HU_FRACTION
        and m["tortuosity"] <= MAX_LAD_TORTUOSITY
        and p["progress_step_fraction"] >= MIN_PROGRESS_STEP_FRACTION
        and p["max_backtrack_from_running_min_mm"] <= MAX_PROGRESS_BACKTRACK_MM
        and m["target_distance_from_local_nearest_surface_mm"] <= MAX_TARGET_FROM_LOCAL_NEAREST_MM
    )


def _plane_image(geom, src, c, t, half=5.0, step=.15):
    u, v = _orth_basis(t)
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    pts = c + xx[..., None]*u + yy[..., None]*v
    im = _sample(geom, src, pts.reshape(-1, 3)).reshape(len(q), len(q))
    return im, q


def _plot_planes(geom, src, path, out):
    p, arc = _resample_path(path, .5)
    t = _tangents(p)
    frac = (.10, .25, .40, .55, .70, .85)
    ids = [int(np.argmin(np.abs(arc - f*arc[-1]))) for f in frac]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, i in zip(axes.ravel(), ids):
        im, q = _plane_image(geom, src, p[i], t[i])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=850, extent=[q[0], q[-1], q[-1], q[0]])
        ax.scatter([0], [0], s=10)
        ax.set_title(f"arc {arc[i]:.1f} mm")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("LAD local root-directed bridge: automatically selected source planes")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_3d(rca, lad, rca_paths, lad_paths, nearest, out):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rca[:, 0], rca[:, 1], rca[:, 2], lw=2.5, label="known RCA")
    ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], lw=2.5, label="frozen LAD")
    for n, p in rca_paths.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=2, label=f"RCA {n}")
    for n, p in lad_paths.items():
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=2, label=f"LAD {n}")
    ax.scatter([nearest["RCA"][0]], [nearest["RCA"][1]], [nearest["RCA"][2]], s=45, label="RCA local aorta target")
    ax.scatter([nearest["LAD"][0]], [nearest["LAD"][1]], [nearest["LAD"][2]], s=45, label="LAD local aorta target")
    ax.set_title("Local root-directed coronary bridge experiment")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_progress(progress_by, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, pair in progress_by.items():
        path, dist = pair
        arc = _arc(path)
        # Distance series corresponds to 0.5-mm resampled path.
        q = np.linspace(0.0, float(arc[-1]), len(dist))
        ax.plot(q, dist, label=name)
    ax.set_xlabel("Path arc (mm)")
    ax.set_ylabel("Distance to aortic surface (mm)")
    ax.set_title("Root-directed progress along candidate paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_hu(qc_by, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, d in qc_by.items():
        ax.plot(d.arc_mm, d.center_hu, label=name)
    ax.axhline(180.0, ls="--", lw=1)
    ax.set_xlabel("Path arc (mm)")
    ax.set_ylabel("Source center HU")
    ax.set_title("Local root-directed path source-CCTA intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    p = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    q, a = _resample_path(p, .5)
    assert len(q) == 5 and np.isclose(a[-1], 2.0)
    pts = np.array([[1., 1., 0.], [1., 0., 0.], [3., 0., 0.]])
    d = _line_distance(pts, np.array([0.,0.,0.]), np.array([2.,0.,0.]))
    assert np.allclose(d, [1., 0., 1.])
    return {"ok": True}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED", "algorithm":ALGORITHM, "baseline":BASELINE})

    master = _read_json(root / MASTER)
    if master.get("status") != EXPECTED_MASTER_STATUS:
        raise RuntimeError(f"Unexpected master status {master.get('status')}")

    geom, src = _load_source(root)
    aorta = _load_mask(root/AORTA_MASK, src.shape, geom)
    pulmonary = _load_mask(root/PULMONARY_MASK, src.shape, geom)
    chambers = np.zeros(src.shape, bool)
    for p in CHAMBER_MASKS:
        chambers |= _load_mask(root/p, src.shape, geom)
    forbidden_global = _erode_mm(chambers | pulmonary, geom.spacing_zyx, DEEP_ANATOMY_EROSION_MM)

    surf_idx, surf_xyz, surf_tree = _surface_cloud(aorta, geom)

    rca_raw = _load_lps_csv(root/RCA_PATH)
    lad_raw = _load_lps_csv(root/LAD_PATH)
    rca, rca_prox_d, rca_other_d = _orient_proximal_first(rca_raw, surf_tree)
    lad, lad_prox_d, lad_other_d = _orient_proximal_first(lad_raw, surf_tree)

    rca_arc = _arc(rca)
    rca_start_arc = min(CONTROL_START_ARC_MM, max(2.0, float(rca_arc[-1])*0.25))
    rca_start = _interp_path(rca, [rca_start_arc])[0]
    _, _, rca_nearest = _nearest_surface_point(rca_start, surf_idx, surf_xyz, surf_tree)
    _, _, rca_root = _nearest_surface_point(rca[0], surf_idx, surf_xyz, surf_tree)
    rca_target = _local_target_mask(aorta, geom, rca_nearest)

    rca_downstream = _interp_path(
        rca,
        np.arange(min(rca_start_arc + KNOWN_BACKTRACK_EXCLUSION_MM, rca_arc[-1]), rca_arc[-1] + 1e-9, .5)
    ) if rca_arc[-1] > rca_start_arc + KNOWN_BACKTRACK_EXCLUSION_MM else np.empty((0,3))
    rca_known_prox = _interp_path(rca, np.arange(0.0, rca_start_arc + .001, .5))

    control_results = {}
    control_paths = {}
    control_qc = {}
    control_gates = {}
    for v in COST_VARIANTS:
        m = _run_one(
            geom, src, aorta, forbidden_global, surf_tree,
            rca_start, rca_nearest, rca_target, rca_downstream, v
        )
        p = m.pop("resampled_path")
        qdf = m.pop("qdf")
        dist_series = m.pop("surface_distance_series_mm")
        control_paths[v["name"]] = p
        control_qc[v["name"]] = qdf
        known_dist = _known_path_distance(p, rca_known_prox)
        target_to_root = float(np.linalg.norm(p[-1] - rca_root))
        m["distance_to_known_RCA"] = known_dist
        m["target_distance_to_known_RCA_root_mm"] = target_to_root
        m["surface_distance_series_mm"] = [float(x) for x in dist_series]
        gate = _control_gate(m, known_dist, target_to_root)
        control_results[v["name"]] = m
        control_gates[v["name"]] = gate
        pd.DataFrame(p, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/f"RCA_local_control_{v['name']}.csv", index=False)
        qdf.to_csv(out/f"RCA_local_control_QC_{v['name']}.csv", index=False)

    control_sep = _symmetric_separation(control_paths["hu140"], control_paths["hu180"])
    control_pass = bool(
        all(control_gates.values())
        and control_sep["median_mm"] <= MAX_VARIANT_MEDIAN_SEP_MM
        and control_sep["p90_mm"] <= MAX_VARIANT_P90_SEP_MM
        and control_sep["endpoint_distance_mm"] <= MAX_VARIANT_ENDPOINT_SEP_MM
    )

    lad_results = {}
    lad_paths = {}
    lad_qc = {}
    lad_gates = {}
    if control_pass:
        lad_start = lad[0]
        _, _, lad_nearest = _nearest_surface_point(lad_start, surf_idx, surf_xyz, surf_tree)
        lad_target = _local_target_mask(aorta, geom, lad_nearest)
        lad_arc = _arc(lad)
        lad_downstream = _interp_path(
            lad,
            np.arange(KNOWN_BACKTRACK_EXCLUSION_MM, lad_arc[-1] + 1e-9, .5)
        ) if lad_arc[-1] > KNOWN_BACKTRACK_EXCLUSION_MM else np.empty((0,3))

        for v in COST_VARIANTS:
            m = _run_one(
                geom, src, aorta, forbidden_global, surf_tree,
                lad_start, lad_nearest, lad_target, lad_downstream, v
            )
            p = m.pop("resampled_path")
            qdf = m.pop("qdf")
            dist_series = m.pop("surface_distance_series_mm")
            m["surface_distance_series_mm"] = [float(x) for x in dist_series]
            lad_paths[v["name"]] = p
            lad_qc[v["name"]] = qdf
            lad_gates[v["name"]] = _lad_gate(m)
            lad_results[v["name"]] = m
            pd.DataFrame(p, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/f"LAD_local_root_bridge_{v['name']}.csv", index=False)
            qdf.to_csv(out/f"LAD_local_root_bridge_QC_{v['name']}.csv", index=False)

        lad_sep = _symmetric_separation(lad_paths["hu140"], lad_paths["hu180"])
        lad_pass = bool(
            all(lad_gates.values())
            and lad_sep["median_mm"] <= MAX_VARIANT_MEDIAN_SEP_MM
            and lad_sep["p90_mm"] <= MAX_VARIANT_P90_SEP_MM
            and lad_sep["endpoint_distance_mm"] <= MAX_VARIANT_ENDPOINT_SEP_MM
        )
    else:
        lad_nearest = np.array([np.nan, np.nan, np.nan])
        lad_sep = None
        lad_pass = False

    c6_posthoc = {}
    if lad_pass and (root/C6_PATH).exists():
        c6 = _load_lps_csv(root/C6_PATH)
        c6_posthoc = {name: _posthoc_to_c6(p, c6) for name, p in lad_paths.items()}

    if not control_pass:
        status = STATUS_CONTROL_FAIL
    elif lad_pass:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_NO_BRIDGE

    gates = [
        {"gate":"RCA_local_control_pass", "pass":control_pass},
        {"gate":"LAD_local_variant_stability_pass", "pass":lad_pass if control_pass else False},
    ]
    for name, val in control_gates.items():
        gates.append({"gate":f"RCA_{name}_individual_gate", "pass":bool(val)})
    for name, val in lad_gates.items():
        gates.append({"gate":f"LAD_{name}_individual_gate", "pass":bool(val)})
    pd.DataFrame(gates).to_csv(out/"local_root_bridge_gates.csv", index=False)

    nearest = {"RCA":rca_nearest, "LAD":lad_nearest}
    _plot_3d(rca, lad, control_paths, lad_paths, nearest, out/"01_local_root_bridge_3D.png")

    progress_by = {}
    for name, p in control_paths.items():
        progress_by[f"RCA_{name}"] = (p, np.asarray(control_results[name]["surface_distance_series_mm"], float))
    for name, p in lad_paths.items():
        progress_by[f"LAD_{name}"] = (p, np.asarray(lad_results[name]["surface_distance_series_mm"], float))
    _plot_progress(progress_by, out/"02_local_root_bridge_aorta_progress.png")

    qc_by = {f"RCA_{k}":v for k,v in control_qc.items()}
    qc_by.update({f"LAD_{k}":v for k,v in lad_qc.items()})
    _plot_hu(qc_by, out/"03_local_root_bridge_center_HU.png")
    if lad_paths:
        _plot_planes(geom, src, lad_paths["hu140"], out/"04_LAD_local_root_bridge_source_planes.png")

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "design": {
            "local_target_radius_mm": LOCAL_TARGET_RADIUS_MM,
            "corridor_radius_mm": CORRIDOR_RADIUS_MM,
            "known_path_exclusion_radius_mm": KNOWN_PATH_EXCLUSION_RADIUS_MM,
            "known_backtrack_exclusion_mm": KNOWN_BACKTRACK_EXCLUSION_MM,
            "progress_step_tolerance_mm": PROGRESS_STEP_TOL_MM,
            "minimum_progress_step_fraction": MIN_PROGRESS_STEP_FRACTION,
            "maximum_progress_backtrack_mm": MAX_PROGRESS_BACKTRACK_MM,
            "deep_anatomy_hard_exclusion_erosion_mm": DEEP_ANATOMY_EROSION_MM,
        },
        "anatomy": {
            "RCA_proximal_endpoint_distance_to_aorta_surface_mm": rca_prox_d,
            "RCA_other_endpoint_distance_to_aorta_surface_mm": rca_other_d,
            "LAD_proximal_endpoint_distance_to_aorta_surface_mm": lad_prox_d,
            "LAD_other_endpoint_distance_to_aorta_surface_mm": lad_other_d,
            "RCA_control_start_arc_mm": rca_start_arc,
            "RCA_control_start_to_local_aorta_mm": float(surf_tree.query(rca_start)[0]),
            "LAD_start_to_local_aorta_mm": float(surf_tree.query(lad[0])[0]),
        },
        "RCA_control": {
            "pass": control_pass,
            "individual_variant_gates": control_gates,
            "variant_results": control_results,
            "cross_variant_agreement": control_sep,
        },
        "LAD_local_bridge": {
            "attempted": control_pass,
            "stable_candidate_pass": lad_pass,
            "individual_variant_gates": lad_gates,
            "variant_results": lad_results,
            "cross_variant_agreement": lad_sep,
            "posthoc_distance_to_C6": c6_posthoc,
        },
        "clinical_LM_established": False,
        "research_local_bridge_candidate_established": bool(status == STATUS_CANDIDATE),
        "scientific_boundary": (
            "This experiment directly diagnoses the failure mode of the prior global geodesic: a proximal bridge must move locally toward the nearest aortic surface rather than retrace the known distal artery or take a long route to the RCA ostium. "
            "The search is therefore constrained to a prespecified straight-line corridor, excludes already-known downstream coronary centerline, hard-excludes only the deep interior of chamber/pulmonary masks (to avoid an artificial barrier at epicardial boundaries), and requires monotonic aortic-surface progress after the same rules recover the RCA positive control. "
            "A positive result would still establish only a source-supported local proximal left-coronary bridge candidate, not clinical left-main identity. The frozen master remains unchanged."
        ),
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"input_provenance.json", {
        "source_cache":str(root/SOURCE_CACHE),
        "master":str(root/MASTER),
        "RCA":str(root/RCA_PATH),
        "LAD":str(root/LAD_PATH),
        "C6_optional":str(root/C6_PATH),
        "aorta":str(root/AORTA_MASK),
        "chambers":[str(root/p) for p in CHAMBER_MASKS],
        "pulmonary":str(root/PULMONARY_MASK),
    })

    report = out/"OPENPLAQUE_LEFT_CORONARY_LOCAL_ROOT_DIRECTED_BRIDGE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Left-Coronary Local Root-Directed Bridge v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>RCA local positive control pass: {control_pass}</p>"
        f"<p>Frozen LAD proximal endpoint to nearest aortic surface: {lad_prox_d:.2f} mm.</p>"
        f"<p>Stable local LAD-to-aorta bridge candidate: {lad_pass}</p>"
        "<p><b>Boundary:</b> This corrects the prior global-search backtracking failure. Even a positive local bridge is research-only and is not clinical LM. Frozen anatomy remains unchanged.</p>"
        '<img src="01_local_root_bridge_3D.png" style="max-width:100%">'
        '<img src="02_local_root_bridge_aorta_progress.png" style="max-width:100%">'
        '<img src="03_local_root_bridge_center_HU.png" style="max-width:100%">'
        + ('<img src="04_LAD_local_root_bridge_source_planes.png" style="max-width:100%">' if (out/"04_LAD_local_root_bridge_source_planes.png").exists() else "")
        + "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json", {"status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE})
    zpath = out/"OPENPLAQUE_LEFT_CORONARY_LOCAL_ROOT_DIRECTED_BRIDGE_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)

    return {"summary":summary, "report":str(report), "zip":str(zpath)}
