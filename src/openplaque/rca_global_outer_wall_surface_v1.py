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
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata, spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-global-outer-wall-surface-v1.0"
OUTPUT_DIRNAME = "RCA_Global_Outer_Wall_Surface_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
RCA_CENTERLINE = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
PCAT = Path("PCAT_RCA_10_50_Reproducibility_Lock/pcat_canonical_primary_longitudinal.csv")
FIXED_SHELL_SUMMARY = Path("RCA_Source_Space_Plaque_Excess_Specificity_v1/summary.json")
ADAPTIVE_SUMMARY = Path("RCA_Adaptive_Outer_Wall_Plaque_v1/summary.json")

ARC_STEP_MM = 0.50
N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.55

WALL_MIN_MM = 0.35
WALL_MAX_MM = 2.20
OUTER_SAMPLE_MAX_MM = 2.50
PRE_BAND = 2
POST_BAND = 2
FAT_HU = -30.0

DIRECT_MIN_DROP_HU = 35.0
DIRECT_MIN_PRE_HU = -5.0
WEAK_MIN_DROP_HU = 85.0
WEAK_MAX_POST_HU = 140.0
ANCHOR_DIRECT_WEIGHT_MIN = 2.0
ANCHOR_DIRECT_WEIGHT_MAX = 6.0
ANCHOR_WEAK_WEIGHT = 0.35
GLOBAL_PRIOR_WEIGHT = 0.01

SURFACE_VARIANTS = {
    "flexible": {"lambda_angle": 0.40, "lambda_long": 0.15},
    "nominal": {"lambda_angle": 0.80, "lambda_long": 0.35},
    "smooth": {"lambda_angle": 1.60, "lambda_long": 0.70},
}
NOMINAL_VARIANT = "nominal"

REFERENCE_ARC = (20.0, 50.0)
HOLDOUT_ARC = (0.0, 20.0)
REFERENCE_RESIDUAL_QUANTILE = 0.90
HUBER_K = 1.35
COMPONENTS = (
    "low_attenuation_mm3",
    "noncalcified_mm3",
    "mixed_intermediate_mm3",
    "calcified_mm3",
)

MIN_STATION_SURFACE_QC = 0.85
MIN_MEDIAN_DIRECT_ANCHOR_FRACTION = 0.25
MAX_MEDIAN_WEIGHTED_ANCHOR_ERROR_MM = 0.45
MIN_POSITIVE_BINS = 4
MIN_NEGATIVE_BINS = 5
MIN_STRICT_BINS = 2
MIN_MAJORITY_AUC = 0.80
MIN_STRICT_AUC = 0.90
MIN_POS_NEG_RATIO = 2.0
MIN_CROSS_VARIANT_SPEARMAN = 0.80

STATUS_GEOMETRY_FAIL = "RCA_GLOBAL_OUTER_WALL_SURFACE_GEOMETRY_FAILED"
STATUS_SPEC_FAIL = "RCA_GLOBAL_OUTER_WALL_SURFACE_SPECIFICITY_FAILED"
STATUS_PASS_NO_GAIN = "RCA_GLOBAL_OUTER_WALL_SURFACE_PASS_NO_CLEAR_GAIN"
STATUS_GAIN = "RCA_GLOBAL_OUTER_WALL_SURFACE_SPECIFICITY_GAIN"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_path(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinate columns in {p}: {list(d.columns)}")


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


def _resample(p, step=ARC_STEP_MM):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.arange(0.0, float(a[-1]) + 1e-9, step)
    if len(q) == 0 or q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _arc_weights(q):
    q = np.asarray(q, float)
    if len(q) == 1:
        return np.ones(1)
    w = np.empty(len(q), float)
    w[0] = 0.5 * (q[1] - q[0])
    w[-1] = 0.5 * (q[-1] - q[-2])
    if len(q) > 2:
        w[1:-1] = 0.5 * (q[2:] - q[:-2])
    return w


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array(
            [[row[0], col[0], slc[0]],
             [row[1], col[1], slc[1]],
             [row[2], col[2], slc[2]]], float
        )
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]


def _load_source(cache):
    cache = Path(cache)
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = SourceGeometry(meta)
    vv = float(np.prod(geom.spacing_zyx))
    if not (0.001 < vv < 0.5):
        raise RuntimeError(f"Unexpected Series-7 voxel volume {vv}")
    return geom, src, vv


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


def _cmedian(x):
    return np.median(np.stack([np.roll(x, k) for k in range(-2, 3)]), axis=0)


def _lumen(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    dirs = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
    center_pts = np.vstack([c, c + .15*u, c - .15*u, c + .15*v, c - .15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55 * center_hu, 220.0, 500.0))
    rr = np.arange(.2, LUMEN_MAX_RADIUS_MM + 1e-9, RADIAL_STEP_MM)
    P = c[None, None, :] + dirs[:, None, :] * rr[None, :, None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(N_ANGLES, len(rr))
    r = np.full(N_ANGLES, np.nan)
    for i in range(N_ANGLES):
        b = (hu[i] < threshold) & (rr >= LUMEN_MIN_RADIUS_MM)
        ix = np.flatnonzero(b[:-1] & b[1:])
        if len(ix):
            r[i] = rr[int(ix[0])]
    valid = np.isfinite(r)
    vf = float(valid.mean())
    if valid.any():
        med = float(np.median(r[valid]))
        r[~valid] = med
        r = _cmedian(r)
        r = np.clip(r, med - .75, med + .75)
        r = np.clip(r, .55, 4.0)
        p10, p50, p90 = np.percentile(r, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
        area = float(.5*np.sum(r*r)*(2*np.pi/N_ANGLES))
    else:
        p10 = p50 = p90 = axis = area = np.nan
    qc = bool(center_hu >= 200 and vf >= .60 and np.isfinite(axis) and axis <= 2.5)
    return dict(
        center_hu=center_hu,
        lumen_threshold_hu=threshold,
        valid_radial_fraction=vf,
        lumen_radius_p10_mm=float(p10),
        lumen_radius_median_mm=float(p50),
        lumen_radius_p90_mm=float(p90),
        lumen_axis_proxy=float(axis),
        lumen_area_mm2=float(area),
        station_qc_pass=qc,
        radii=r, theta=th, u=u, v=v,
    )


def _radial_tensor(geom, src, centers, tangents, lumens):
    off = np.arange(RADIAL_STEP_MM/2.0, OUTER_SAMPLE_MAX_MM + .30, RADIAL_STEP_MM)
    tensor = np.full((len(centers), N_ANGLES, len(off)), np.nan, float)
    smooth = np.full_like(tensor, np.nan)
    for i, (c, L) in enumerate(zip(centers, lumens)):
        if not L["station_qc_pass"]:
            continue
        th = L["theta"]
        dirs = np.cos(th)[:, None]*L["u"] + np.sin(th)[:, None]*L["v"]
        rr = L["radii"][:, None] + off[None, :]
        P = c[None, None, :] + dirs[:, None, :]*rr[..., None]
        hu = _sample(geom, src, P.reshape(-1, 3)).reshape(rr.shape)
        tensor[i] = hu
        xp = np.pad(hu, ((0, 0), (1, 1)), mode="edge")
        smooth[i] = .25*xp[:, :-2] + .50*xp[:, 1:-1] + .25*xp[:, 2:]
    return off, tensor, smooth


def _anchors_from_radial(off, smooth):
    ns, na, nr = smooth.shape
    anchor = np.full((ns, na), np.nan, float)
    weight = np.zeros((ns, na), float)
    mode = np.zeros((ns, na), np.int8)  # 0 none, 1 direct fat, 2 weak gradient
    drop_arr = np.full((ns, na), np.nan, float)
    post_arr = np.full((ns, na), np.nan, float)
    for s in range(ns):
        for a in range(na):
            x = smooth[s, a]
            if not np.isfinite(x).any():
                continue
            direct = []
            weak = []
            for j, t in enumerate(off):
                if t < WALL_MIN_MM or t > WALL_MAX_MM:
                    continue
                pre0 = max(0, j - PRE_BAND)
                pre1 = j + 1
                post0 = min(nr, j + 1)
                post1 = min(nr, j + 1 + POST_BAND + 1)
                if pre1 <= pre0 or post1 <= post0:
                    continue
                pre = float(np.mean(x[pre0:pre1]))
                post = float(np.mean(x[post0:post1]))
                drop = pre - post
                if post < FAT_HU and pre > DIRECT_MIN_PRE_HU and drop >= DIRECT_MIN_DROP_HU:
                    direct.append((float(t), drop, post))
                if drop >= WEAK_MIN_DROP_HU and post <= WEAK_MAX_POST_HU:
                    score = drop - 4.0*float(t)
                    weak.append((score, float(t), drop, post))
            if direct:
                # Earliest convincing transition to fat is a physical outer-wall anchor.
                t, drop, post = sorted(direct, key=lambda z: z[0])[0]
                w = float(np.clip(drop / 35.0, ANCHOR_DIRECT_WEIGHT_MIN, ANCHOR_DIRECT_WEIGHT_MAX))
                anchor[s, a] = t
                weight[s, a] = w
                mode[s, a] = 1
                drop_arr[s, a] = drop
                post_arr[s, a] = post
            elif weak:
                _, t, drop, post = max(weak, key=lambda z: z[0])
                anchor[s, a] = t
                weight[s, a] = ANCHOR_WEAK_WEIGHT
                mode[s, a] = 2
                drop_arr[s, a] = drop
                post_arr[s, a] = post
    return anchor, weight, mode, drop_arr, post_arr


def _solve_surface(anchor, weight, lambda_angle, lambda_long):
    anchor = np.asarray(anchor, float)
    weight = np.asarray(weight, float)
    ns, na = anchor.shape
    direct = np.isfinite(anchor) & (weight >= ANCHOR_DIRECT_WEIGHT_MIN)
    any_anchor = np.isfinite(anchor) & (weight > 0)
    if int(direct.sum()) < max(50, ns):
        raise RuntimeError(f"Too few direct outer-wall anchors: {int(direct.sum())}")
    prior = float(np.median(anchor[direct]))
    rows, cols, vals = [], [], []
    b = np.zeros(ns*na, float)

    def add(i, j, val):
        rows.append(i); cols.append(j); vals.append(float(val))

    for s in range(ns):
        for a in range(na):
            k = s*na + a
            w = float(weight[s, a]) if any_anchor[s, a] else 0.0
            diag = w + GLOBAL_PRIOR_WEIGHT
            if w:
                b[k] += w*float(anchor[s, a])
            b[k] += GLOBAL_PRIOR_WEIGHT*prior

            # Circular angular edge, counted only forward.
            a2 = (a + 1) % na
            k2 = s*na + a2
            lam = float(lambda_angle)
            diag += lam
            add(k, k2, -lam)

            # Backward angular contribution to maintain symmetry.
            a0 = (a - 1) % na
            k0 = s*na + a0
            diag += lam
            add(k, k0, -lam)

            if s > 0:
                k0s = (s-1)*na + a
                diag += float(lambda_long)
                add(k, k0s, -float(lambda_long))
            if s + 1 < ns:
                k1s = (s+1)*na + a
                diag += float(lambda_long)
                add(k, k1s, -float(lambda_long))
            add(k, k, diag)

    A = coo_matrix((vals, (rows, cols)), shape=(ns*na, ns*na)).tocsr()
    x = spsolve(A, b).reshape(ns, na)
    x = np.clip(x, WALL_MIN_MM, WALL_MAX_MM)

    err = np.full_like(x, np.nan)
    err[any_anchor] = np.abs(x[any_anchor] - anchor[any_anchor])
    return x, prior, err


def _surface_station_metrics(surface, anchor, weight, mode, err, lumens):
    rows = []
    for s in range(surface.shape[0]):
        direct = mode[s] == 1
        weak = mode[s] == 2
        anya = direct | weak
        weighted = weight[s] > 0
        mae = float(np.average(err[s, weighted], weights=weight[s, weighted])) if weighted.any() else np.nan
        thick = surface[s]
        iqr = float(np.percentile(thick, 75) - np.percentile(thick, 25))
        qc = bool(
            lumens[s]["station_qc_pass"]
            and float(direct.mean()) >= .10
            and float(anya.mean()) >= .25
            and np.isfinite(mae) and mae <= .60
            and iqr <= .90
            and float(np.median(thick)) >= WALL_MIN_MM
            and float(np.median(thick)) <= 1.80
        )
        rows.append({
            "station_index": s,
            "lumen_qc_pass": bool(lumens[s]["station_qc_pass"]),
            "direct_anchor_fraction": float(direct.mean()),
            "weak_anchor_fraction": float(weak.mean()),
            "any_anchor_fraction": float(anya.mean()),
            "weighted_anchor_mae_mm": mae,
            "median_surface_thickness_mm": float(np.median(thick)),
            "surface_thickness_iqr_mm": iqr,
            "surface_qc_pass": qc,
        })
    return pd.DataFrame(rows)


def _integrate_surface(geom, src, c, L, thick, ds):
    th = L["theta"]
    dirs = np.cos(th)[:, None]*L["u"] + np.sin(th)[:, None]*L["v"]
    max_t = float(np.max(thick))
    off = np.arange(RADIAL_STEP_MM/2.0, max_t + 1e-9, RADIAL_STEP_MM)
    rr = L["radii"][:, None] + off[None, :]
    P = c[None, None, :] + dirs[:, None, :]*rr[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(rr.shape)
    inside = off[None, :] <= (thick[:, None] + 1e-9)
    area = rr*RADIAL_STEP_MM*(2*np.pi/N_ANGLES)
    vol = area*float(ds)
    finite = np.isfinite(hu) & inside
    masks = {
        "fatlike_excluded_mm3": finite & (hu < -30),
        "low_attenuation_mm3": finite & (hu >= -30) & (hu < 30),
        "noncalcified_mm3": finite & (hu >= 30) & (hu < 130),
        "mixed_intermediate_mm3": finite & (hu >= 130) & (hu < 350),
        "calcified_mm3": finite & (hu >= 350),
    }
    vals = {k: float(vol[m].sum()) for k, m in masks.items()}
    vals["wall_volume_mm3"] = float(vol[finite].sum())
    vals["total_plaque_proxy_mm3"] = sum(vals[c] for c in COMPONENTS)
    vals["wall_mean_hu"] = float(np.average(hu[finite], weights=area[finite])) if finite.any() else np.nan
    return vals


def _build_station_table(geom, src, centers, arcs, weights, lumens, surface, metrics, variant):
    rows = []
    for i, (c, a, ds, L) in enumerate(zip(centers, arcs, weights, lumens)):
        m = metrics.iloc[i]
        base = {k: v for k, v in L.items() if k not in ("radii", "theta", "u", "v")}
        if bool(m.surface_qc_pass):
            wall = _integrate_surface(geom, src, c, L, surface[i], ds)
        else:
            wall = {k: np.nan for k in (
                "fatlike_excluded_mm3", "low_attenuation_mm3", "noncalcified_mm3",
                "mixed_intermediate_mm3", "calcified_mm3", "wall_volume_mm3",
                "total_plaque_proxy_mm3", "wall_mean_hu",
            )}
        rows.append({
            "variant": variant,
            "station_index": i,
            "arc_mm": float(a),
            "integration_ds_mm": float(ds),
            **base,
            **m.to_dict(),
            **wall,
        })
    return pd.DataFrame(rows)


def _design(df):
    r = df["lumen_radius_median_mm"].to_numpy(float)
    hu = df["center_hu"].to_numpy(float) / 1000.0
    wt = df["median_surface_thickness_mm"].to_numpy(float)
    return np.column_stack([np.ones(len(df)), r, r*r, hu, wt])


def _huber_fit(X, y, k=HUBER_K, n_iter=40):
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 4:
        raise RuntimeError("Insufficient finite reference stations")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        resid = y - X @ beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = max(1.4826*mad, 1e-6)
        a = np.abs(resid - med)/scale
        w = np.ones_like(a)
        hi = a > k
        w[hi] = k/np.maximum(a[hi], 1e-12)
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X*sw[:, None], y*sw, rcond=None)[0]
        if np.linalg.norm(new-beta) < 1e-10:
            beta = new
            break
        beta = new
    return beta, y-X@beta


def _fit_reference(stations, variant):
    d = stations[(stations.variant == variant) & stations.surface_qc_pass.astype(bool)].copy()
    ref = (d.arc_mm >= REFERENCE_ARC[0]) & (d.arc_mm < REFERENCE_ARC[1])
    if int(ref.sum()) < 25:
        raise RuntimeError(f"Too few global-surface reference stations for {variant}: {int(ref.sum())}")
    X = _design(d)
    models = []
    for c in COMPONENTS:
        frac = d[c].to_numpy(float)/np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        beta, resid = _huber_fit(X[ref], frac[ref])
        q = float(np.quantile(resid[np.isfinite(resid)], REFERENCE_RESIDUAL_QUANTILE))
        models.append({
            "variant": variant,
            "component": c,
            "beta_intercept": float(beta[0]),
            "beta_lumen_radius": float(beta[1]),
            "beta_lumen_radius_sq": float(beta[2]),
            "beta_center_hu_per_1000": float(beta[3]),
            "beta_median_surface_thickness": float(beta[4]),
            "reference_residual_p90": q,
            "reference_station_count": int(ref.sum()),
        })
    return pd.DataFrame(models)


def _apply_reference(stations, models, variant):
    d = stations[stations.variant == variant].copy()
    X = _design(d)
    for c in COMPONENTS:
        m = models[(models.variant == variant) & (models.component == c)]
        if len(m) != 1:
            raise RuntimeError(f"Missing model for {variant} {c}")
        m = m.iloc[0]
        beta = np.array([
            m.beta_intercept, m.beta_lumen_radius, m.beta_lumen_radius_sq,
            m.beta_center_hu_per_1000, m.beta_median_surface_thickness,
        ], float)
        frac = d[c].to_numpy(float)/np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        threshold = X@beta + float(m.reference_residual_p90)
        excess = np.maximum(frac-threshold, 0.0)*d.wall_volume_mm3.to_numpy(float)
        excess[~d.surface_qc_pass.to_numpy(bool)] = np.nan
        stem = c.replace("_mm3", "")
        d[f"expected_p90_{stem}_fraction"] = threshold
        d[f"excess_{stem}_mm3"] = excess
    ex = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    d["excess_total_plaque_proxy_mm3"] = d[ex].sum(axis=1, min_count=1)
    return d


def _profile_1mm(scored):
    d = scored[scored.surface_qc_pass.astype(bool)].copy()
    d["arc_start_mm"] = np.floor(d.arc_mm.to_numpy(float)).astype(float)
    rows = []
    excols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    for a, g in d.groupby("arc_start_mm", sort=True):
        r = {
            "variant": str(g.variant.iloc[0]),
            "arc_start_mm": float(a),
            "arc_end_mm": float(a+1.0),
            "station_count": int(len(g)),
            "surface_qc_fraction": float(g.surface_qc_pass.mean()),
            "mean_surface_thickness_mm": float(g.median_surface_thickness_mm.mean()),
            "mean_direct_anchor_fraction": float(g.direct_anchor_fraction.mean()),
            "mean_weighted_anchor_mae_mm": float(g.weighted_anchor_mae_mm.mean()),
            "raw_wall_volume_mm3": float(g.wall_volume_mm3.sum()),
            "raw_total_plaque_proxy_mm3": float(g.total_plaque_proxy_mm3.sum()),
        }
        for c in excols:
            r[c] = float(g[c].sum())
        r["excess_total_plaque_proxy_mm3"] = float(g.excess_total_plaque_proxy_mm3.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def _auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ok = np.isfinite(score) & np.isin(y, [0, 1])
    y, score = y[ok], score[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return np.nan
    r = rankdata(score, method="average")
    return float((r[y == 1].sum() - n1*(n1+1)/2.0)/(n1*n0))


def _check_reference(prior):
    ref = prior[(prior.arc_start_mm >= REFERENCE_ARC[0]) & (prior.arc_start_mm < REFERENCE_ARC[1])]
    bad = int((ref.mapped_native_voxels_vote_ge3.fillna(0).to_numpy(float) > 0).sum())
    maj = int(ref.majority_3plus_signal.fillna(False).astype(bool).sum())
    return bool(len(ref) and bad == 0 and maj == 0), {
        "reference_bins": int(len(ref)),
        "bins_with_vote_ge3": bad,
        "majority_positive_bins": maj,
    }


def _holdout_metrics(profile, prior):
    z = profile.merge(prior, on=["arc_start_mm", "arc_end_mm"], how="inner")
    z = z[(z.arc_start_mm >= HOLDOUT_ARC[0]) & (z.arc_start_mm < HOLDOUT_ARC[1])].copy()
    pos = z.majority_3plus_signal.fillna(False).astype(bool)
    neg = z.mapped_native_vote_sum.fillna(0).to_numpy(float) == 0
    strict = z.strict_5of5_signal.fillna(False).astype(bool)
    score = z.excess_total_plaque_proxy_mm3.to_numpy(float)
    out = {
        "usable": bool(int(pos.sum()) >= MIN_POSITIVE_BINS and int(neg.sum()) >= MIN_NEGATIVE_BINS),
        "holdout_bins": int(len(z)),
        "positive_bins": int(pos.sum()),
        "strict_5of5_bins": int(strict.sum()),
        "vote_free_negative_bins": int(neg.sum()),
    }
    if not out["usable"]:
        return z, out
    out["majority_vs_vote_free_auc"] = _auc(
        np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))],
        np.r_[score[pos], score[neg]],
    )
    out["strict5_vs_vote_free_auc"] = _auc(
        np.r_[np.ones(int(strict.sum())), np.zeros(int(neg.sum()))],
        np.r_[score[strict], score[neg]],
    ) if int(strict.sum()) >= MIN_STRICT_BINS else np.nan
    mp = float(np.median(score[pos]))
    mn = float(np.median(score[neg]))
    out["positive_median_excess_mm3"] = mp
    out["negative_median_excess_mm3"] = mn
    out["positive_negative_median_ratio"] = mp/max(mn, 1e-6)
    out["spearman_excess_vs_vote_ge3"] = float(
        spearmanr(z.excess_total_plaque_proxy_mm3, z.mapped_native_voxels_vote_ge3).statistic
    )
    return z, out


def _variant_consistency(profiles):
    nom = profiles[NOMINAL_VARIANT].set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"]
    rows = []
    for name, p in profiles.items():
        y = p.set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"].reindex(nom.index)
        ok = nom.notna() & y.notna()
        rho = float(spearmanr(nom[ok], y[ok]).statistic) if int(ok.sum()) >= 5 else np.nan
        rows.append({"variant": name, "profile_spearman_vs_nominal": rho})
    d = pd.DataFrame(rows)
    f = d[d.variant != NOMINAL_VARIANT].profile_spearman_vs_nominal
    f = f[np.isfinite(f)]
    return d, float(f.min()) if len(f) else np.nan


def _specificity_pass(m, min_rho):
    return bool(
        m.get("usable", False)
        and m.get("majority_vs_vote_free_auc", -np.inf) >= MIN_MAJORITY_AUC
        and (
            m.get("strict_5of5_bins", 0) < MIN_STRICT_BINS
            or m.get("strict5_vs_vote_free_auc", -np.inf) >= MIN_STRICT_AUC
        )
        and m.get("positive_negative_median_ratio", -np.inf) >= MIN_POS_NEG_RATIO
        and np.isfinite(min_rho) and min_rho >= MIN_CROSS_VARIANT_SPEARMAN
    )


def _plane_image(geom, src, c, t, half=5.0, step=.15):
    u, v = _orth_basis(t)
    q = np.arange(-half, half+1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[..., None]*u + yy[..., None]*v
    im = _sample(geom, src, P.reshape(-1, 3)).reshape(len(q), len(q))
    return im, q


def _plot_surface_map(surface, arcs, out):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(
        surface.T, aspect="auto", origin="lower",
        extent=[float(arcs[0]), float(arcs[-1]), 0, 360],
        vmin=WALL_MIN_MM, vmax=WALL_MAX_MM,
    )
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Circumferential angle (deg)")
    ax.set_title("Nominal global outer-wall thickness surface")
    fig.colorbar(im, ax=ax, label="wall thickness (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_longitudinal(profile, matched, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(profile.arc_start_mm+.5, profile.excess_total_plaque_proxy_mm3, linewidth=2, label="global-surface excess")
    ax.plot(profile.arc_start_mm+.5, profile.raw_wall_volume_mm3, alpha=.30, label="raw segmented wall volume")
    ax.axvspan(*REFERENCE_ARC, alpha=.05)
    ax.axvspan(*HOLDOUT_ARC, alpha=.04)
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Volume per 1-mm bin (mm³)")
    ax2 = ax.twinx()
    if len(matched):
        ax2.step(matched.arc_start_mm+.5, matched.mapped_native_voxels_vote_ge3.fillna(0),
                 where="mid", alpha=.45, label="prior 3+/5 vote voxels")
    ax2.set_ylabel("Prior ensemble 3+/5 vote voxels")
    lines = ax.get_lines()+ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("RCA global outer-wall surface plaque excess")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_variants(profiles, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, p in profiles.items():
        ax.plot(p.arc_start_mm+.5, p.excess_total_plaque_proxy_mm3, label=name)
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Excess plaque proxy per 1-mm bin (mm³)")
    ax.set_title("Global-surface regularization sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_qc(geom, src, centers, tangents, lumens, surface, scored, out):
    good = scored[scored.surface_qc_pass.astype(bool)].copy()
    if good.empty:
        return
    ids = list(good.nlargest(4, "excess_total_plaque_proxy_mm3").station_index.astype(int))
    ids += list(good.nsmallest(2, "excess_total_plaque_proxy_mm3").station_index.astype(int))
    ids = list(dict.fromkeys(ids))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = np.asarray(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, ids):
        row = scored[scored.station_index == i].iloc[0]
        im, q = _plane_image(geom, src, centers[i], tangents[i])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=800, extent=[q[0], q[-1], q[-1], q[0]])
        th = lumens[i]["theta"]
        rin = lumens[i]["radii"]
        rout = rin + surface[i]
        ax.plot(rin*np.cos(th), rin*np.sin(th), linewidth=1.1)
        ax.plot(rout*np.cos(th), rout*np.sin(th), linewidth=1.1, linestyle="--")
        ax.scatter([0], [0], s=8)
        ax.set_title(
            f"{row.arc_mm:.1f} mm | wall {row.median_surface_thickness_mm:.2f} | "
            f"anchor MAE {row.weighted_anchor_mae_mm:.2f}"
        )
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle("RCA global outer-wall source-CCTA QC")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def synthetic_self_test():
    # Small 3x8 surface with partial anchors should solve smoothly and remain bounded.
    anchor = np.full((30, 8), np.nan)
    weight = np.zeros((30, 8))
    anchor[:, 0] = np.linspace(0.7, 0.9, 30)
    anchor[:, 4] = np.linspace(1.0, 1.2, 30)
    weight[:, [0, 4]] = 3.0
    surf, prior, err = _solve_surface(anchor, weight, .8, .35)
    assert surf.shape == anchor.shape
    assert np.isfinite(surf).all()
    assert WALL_MIN_MM <= surf.min() <= surf.max() <= WALL_MAX_MM
    assert .7 <= prior <= 1.2
    assert _auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.0
    return {
        "ok": True,
        "surface_min_mm": float(surf.min()),
        "surface_max_mm": float(surf.max()),
        "prior_mm": float(prior),
    }


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root/MASTER)
    fixed = _read_json(root/FIXED_SHELL_SUMMARY)
    adaptive = _read_json(root/ADAPTIVE_SUMMARY) if (root/ADAPTIVE_SUMMARY).exists() else {}
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if fixed.get("status") != "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS":
        raise RuntimeError("Successful fixed-shell RCA prerequisite failed")

    prior = pd.read_csv(_req(root/PRIOR))
    ref_ok, ref_check = _check_reference(prior)
    if not ref_ok:
        raise RuntimeError(f"RCA reference region is no longer vote-free: {ref_check}")

    geom, src, voxel_volume = _load_source(root/SOURCE_CACHE)
    rca = _load_path(root/RCA_CENTERLINE)
    centers, arcs = _resample(rca)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    weights = _arc_weights(arcs)

    center_hu = _sample(geom, src, centers)
    if float(np.mean(center_hu >= 200)) < .95:
        raise RuntimeError("Canonical RCA source-support control failed")

    lumens = [_lumen(geom, src, c, t) for c, t in zip(centers, tangents)]
    lumen_qc = float(np.mean([x["station_qc_pass"] for x in lumens]))
    if lumen_qc < .90:
        raise RuntimeError(f"Original RCA lumen geometry control failed: {lumen_qc:.3f}")

    off, radial, smooth = _radial_tensor(geom, src, centers, tangents, lumens)
    anchor, anchor_weight, anchor_mode, anchor_drop, anchor_post = _anchors_from_radial(off, smooth)

    direct_mask = anchor_mode == 1
    weak_mask = anchor_mode == 2
    anchor_summary = {
        "global_direct_anchor_fraction": float(direct_mask.mean()),
        "global_weak_anchor_fraction": float(weak_mask.mean()),
        "global_any_anchor_fraction": float((direct_mask | weak_mask).mean()),
        "direct_anchor_thickness_median_mm": float(np.nanmedian(anchor[direct_mask])) if direct_mask.any() else np.nan,
        "direct_anchor_thickness_iqr_mm": float(np.nanpercentile(anchor[direct_mask], 75) - np.nanpercentile(anchor[direct_mask], 25)) if direct_mask.any() else np.nan,
    }

    surfaces = {}
    station_sets = {}
    scored_sets = {}
    models_all = []
    profiles = {}
    metrics = {}
    surface_summaries = []
    nominal_anchor_err = None

    for name, cfg in SURFACE_VARIANTS.items():
        surf, prior_thick, err = _solve_surface(
            anchor, anchor_weight, cfg["lambda_angle"], cfg["lambda_long"]
        )
        surfaces[name] = surf
        met = _surface_station_metrics(surf, anchor, anchor_weight, anchor_mode, err, lumens)
        st = _build_station_table(geom, src, centers, arcs, weights, lumens, surf, met, name)
        station_sets[name] = st
        models = _fit_reference(st, name)
        models_all.append(models)
        sc = _apply_reference(st, models, name)
        scored_sets[name] = sc
        p = _profile_1mm(sc)
        profiles[name] = p
        matched, hm = _holdout_metrics(p, prior)
        metrics[name] = hm
        matched.to_csv(out/f"RCA_{name}_holdout_bins.csv", index=False)
        direct_station = met.direct_anchor_fraction.to_numpy(float)
        surface_summaries.append({
            "variant": name,
            "lambda_angle": cfg["lambda_angle"],
            "lambda_long": cfg["lambda_long"],
            "global_prior_thickness_mm": prior_thick,
            "station_qc_fraction": float(met.surface_qc_pass.mean()),
            "median_direct_anchor_fraction": float(np.median(direct_station)),
            "median_any_anchor_fraction": float(np.median(met.any_anchor_fraction)),
            "median_weighted_anchor_mae_mm": float(np.nanmedian(met.weighted_anchor_mae_mm)),
            "median_surface_thickness_mm": float(np.median(met.median_surface_thickness_mm)),
            "median_surface_iqr_mm": float(np.median(met.surface_thickness_iqr_mm)),
        })
        if name == NOMINAL_VARIANT:
            nominal_anchor_err = err

    pd.concat(station_sets.values(), ignore_index=True).to_csv(
        out/"RCA_global_outer_wall_stations_all_variants.csv", index=False
    )
    pd.concat(scored_sets.values(), ignore_index=True).to_csv(
        out/"RCA_global_outer_wall_excess_stations.csv", index=False
    )
    pd.concat(models_all, ignore_index=True).to_csv(
        out/"RCA_global_outer_wall_reference_models.csv", index=False
    )
    pd.concat(profiles.values(), ignore_index=True).to_csv(
        out/"RCA_global_outer_wall_profiles_1mm_all_variants.csv", index=False
    )
    profiles[NOMINAL_VARIANT].to_csv(
        out/"RCA_global_outer_wall_profile_1mm_nominal.csv", index=False
    )
    surface_summary = pd.DataFrame(surface_summaries)
    surface_summary.to_csv(out/"RCA_global_outer_wall_surface_summary.csv", index=False)

    # Ray-level diagnostics for the nominal surface.
    ray_rows = []
    nom_surf = surfaces[NOMINAL_VARIANT]
    for s in range(len(centers)):
        for a in range(N_ANGLES):
            ray_rows.append({
                "station_index": s,
                "arc_mm": float(arcs[s]),
                "angle_index": a,
                "theta_rad": float(lumens[s]["theta"][a]),
                "lumen_radius_mm": float(lumens[s]["radii"][a]),
                "anchor_thickness_mm": float(anchor[s, a]) if np.isfinite(anchor[s, a]) else np.nan,
                "anchor_weight": float(anchor_weight[s, a]),
                "anchor_mode": int(anchor_mode[s, a]),
                "anchor_drop_hu": float(anchor_drop[s, a]) if np.isfinite(anchor_drop[s, a]) else np.nan,
                "anchor_post_hu": float(anchor_post[s, a]) if np.isfinite(anchor_post[s, a]) else np.nan,
                "surface_thickness_mm": float(nom_surf[s, a]),
                "anchor_abs_error_mm": float(nominal_anchor_err[s, a]) if np.isfinite(nominal_anchor_err[s, a]) else np.nan,
            })
    pd.DataFrame(ray_rows).to_csv(out/"RCA_nominal_global_surface_ray_diagnostics.csv", index=False)

    consistency, min_rho = _variant_consistency(profiles)
    consistency.to_csv(out/"RCA_global_outer_wall_variant_consistency.csv", index=False)

    nom_st = station_sets[NOMINAL_VARIANT]
    nom_sc = scored_sets[NOMINAL_VARIANT]
    nom_profile = profiles[NOMINAL_VARIANT]
    nom_metric = metrics[NOMINAL_VARIANT]
    nom_summary_row = surface_summary[surface_summary.variant == NOMINAL_VARIANT].iloc[0]
    surface_qc = float(nom_summary_row.station_qc_fraction)
    median_direct = float(nom_summary_row.median_direct_anchor_fraction)
    median_anchor_err = float(nom_summary_row.median_weighted_anchor_mae_mm)
    geometry_pass = bool(
        surface_qc >= MIN_STATION_SURFACE_QC
        and median_direct >= MIN_MEDIAN_DIRECT_ANCHOR_FRACTION
        and median_anchor_err <= MAX_MEDIAN_WEIGHTED_ANCHOR_ERROR_MM
    )
    specificity_pass = _specificity_pass(nom_metric, min_rho)

    old_auc = float(fixed["nominal_holdout_specificity"]["majority_vs_vote_free_auc"])
    old_strict = float(fixed["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"])
    new_auc = float(nom_metric.get("majority_vs_vote_free_auc", np.nan))
    new_strict = float(nom_metric.get("strict5_vs_vote_free_auc", np.nan))
    auc_delta = new_auc-old_auc if np.isfinite(new_auc) else np.nan
    strict_delta = new_strict-old_strict if np.isfinite(new_strict) else np.nan

    if not geometry_pass:
        status = STATUS_GEOMETRY_FAIL
    elif not specificity_pass:
        status = STATUS_SPEC_FAIL
    elif np.isfinite(auc_delta) and auc_delta >= .02:
        status = STATUS_GAIN
    else:
        status = STATUS_PASS_NO_GAIN

    matched_nom = pd.read_csv(out/f"RCA_{NOMINAL_VARIANT}_holdout_bins.csv")
    _plot_surface_map(nom_surf, arcs, out/"01_RCA_global_outer_wall_surface_map.png")
    _plot_longitudinal(nom_profile, matched_nom, out/"02_RCA_global_surface_excess_vs_prior.png")
    _plot_variants(profiles, out/"03_RCA_global_surface_regularization_sensitivity.png")
    _plot_qc(geom, src, centers, tangents, lumens, nom_surf, nom_sc,
             out/"04_RCA_global_outer_wall_source_QC.png")

    pcat_fusion = pd.DataFrame()
    if (root/PCAT).exists():
        pcat = pd.read_csv(root/PCAT)
        pcat_fusion = nom_profile.merge(pcat, on=["arc_start_mm", "arc_end_mm"], how="inner")
        if not pcat_fusion.empty:
            pcat_fusion.to_csv(out/"RCA_global_outer_wall_PCAT_fusion_10_50.csv", index=False)

    excols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    totals = {c: float(nom_profile[c].sum()) for c in excols+["excess_total_plaque_proxy_mm3"]}
    totals["raw_wall_volume_mm3"] = float(nom_profile.raw_wall_volume_mm3.sum())

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "source_voxel_volume_mm3": voxel_volume,
        "original_lumen_station_qc_fraction": lumen_qc,
        "reference_zone_mm": list(REFERENCE_ARC),
        "holdout_zone_mm": list(HOLDOUT_ARC),
        "reference_vote_free_check": ref_check,
        "anchor_evidence": anchor_summary,
        "surface_method": {
            "representation": "single global wall-thickness surface over longitudinal arc x circumferential angle",
            "direct_anchor": "earliest source-HU transition to < -30 HU perivascular fat",
            "weak_anchor": "high-confidence negative radial gradient when direct fat transition is absent",
            "surface_variants": SURFACE_VARIANTS,
            "global_prior_weight": GLOBAL_PRIOR_WEIGHT,
            "selection_uses_plaque_labels": False,
        },
        "nominal_surface_geometry": {
            "station_qc_fraction": surface_qc,
            "median_direct_anchor_fraction": median_direct,
            "median_weighted_anchor_mae_mm": median_anchor_err,
            "median_surface_thickness_mm": float(nom_summary_row.median_surface_thickness_mm),
            "median_surface_iqr_mm": float(nom_summary_row.median_surface_iqr_mm),
            "geometry_gate_pass": geometry_pass,
        },
        "nominal_holdout_specificity": nom_metric,
        "min_cross_variant_profile_spearman": min_rho,
        "specificity_gate_pass": specificity_pass,
        "comparison_to_fixed_1mm_shell_excess": {
            "prior_majority_auc": old_auc,
            "global_surface_majority_auc": new_auc,
            "majority_auc_delta": auc_delta,
            "prior_strict_auc": old_strict,
            "global_surface_strict_auc": new_strict,
            "strict_auc_delta": strict_delta,
            "prior_fixed_shell_total_excess_mm3": float(
                fixed["nominal_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"]
            ),
        },
        "prior_adaptive_outer_wall_status": adaptive.get("status"),
        "nominal_global_surface_totals_mm3": totals,
        "pcat_fusion_available": bool(not pcat_fusion.empty),
        "is_validated_clinical_tpv": False,
        "scientific_boundary": (
            "This is a developmental single-scan explicit global outer-wall surface model. Boundary anchors and smoothness are source-CCTA-derived "
            "without plaque labels, but the same RCA has already been used during method development. Normal-wall composition is fit only in the "
            "prespecified vote-free RCA 20-50 mm reference region and evaluated in 0-20 mm. This is not independent validation and not clinical TPV."
        ),
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"input_provenance.json", {
        "source_cache": str(root/SOURCE_CACHE),
        "rca_centerline": str(root/RCA_CENTERLINE),
        "master": str(root/MASTER),
        "prior_profile": str(root/PRIOR),
        "fixed_shell_summary": str(root/FIXED_SHELL_SUMMARY),
        "adaptive_outer_wall_summary": str(root/ADAPTIVE_SUMMARY),
        "pcat_profile": str(root/PCAT),
    })

    report = out/"OPENPLAQUE_RCA_GLOBAL_OUTER_WALL_SURFACE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Global Outer-Wall Surface v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Original lumen QC: {lumen_qc:.3f}; global-surface QC: {surface_qc:.3f}; "
        f"median direct-anchor fraction: {median_direct:.3f}; median anchor error: {median_anchor_err:.3f} mm.</p>"
        f"<p>Holdout majority AUC: {new_auc:.3f}; strict AUC: {new_strict:.3f}; "
        f"positive/negative ratio: {nom_metric.get('positive_negative_median_ratio', float('nan')):.3f}; "
        f"cross-variant Spearman: {min_rho:.3f}.</p>"
        f"<p>Compared with fixed 1-mm shell: majority AUC Δ {auc_delta:+.3f}; strict AUC Δ {strict_delta:+.3f}.</p>"
        "<p><b>Boundary:</b> developmental explicit global source-CCTA wall surface; not independent validation and not clinical TPV.</p>"
        "<h2>Anchor evidence</h2><pre>"+json.dumps(anchor_summary, indent=2, default=str)+"</pre>"
        "<h2>Nominal surface geometry</h2><pre>"+json.dumps(summary["nominal_surface_geometry"], indent=2, default=str)+"</pre>"
        "<h2>Nominal specificity</h2><pre>"+json.dumps(nom_metric, indent=2, default=str)+"</pre>"
        "<h2>Comparison with fixed shell</h2><pre>"+json.dumps(summary["comparison_to_fixed_1mm_shell_excess"], indent=2, default=str)+"</pre>"
        "<h2>Nominal totals</h2><pre>"+json.dumps(totals, indent=2, default=str)+"</pre>"
        '<img src="01_RCA_global_outer_wall_surface_map.png" style="max-width:100%">'
        '<img src="02_RCA_global_surface_excess_vs_prior.png" style="max-width:100%">'
        '<img src="03_RCA_global_surface_regularization_sensitivity.png" style="max-width:100%">'
        '<img src="04_RCA_global_outer_wall_source_QC.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json", {
        "status": "COMPLETE", "result_status": status, "algorithm": ALGORITHM, "baseline": BASELINE
    })
    zpath = out/"OPENPLAQUE_RCA_GLOBAL_OUTER_WALL_SURFACE_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
