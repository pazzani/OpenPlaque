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
from scipy.stats import rankdata, spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-adaptive-outer-wall-plaque-v1.0"
OUTPUT_DIRNAME = "RCA_Adaptive_Outer_Wall_Plaque_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
RCA_CENTERLINE = Path("Cache/Source_Volume_Coronary_Centerlines/RCA_source_centerline.csv")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
PCAT = Path("PCAT_RCA_10_50_Reproducibility_Lock/pcat_canonical_primary_longitudinal.csv")
FIXED_SHELL_SUMMARY = Path("RCA_Source_Space_Plaque_Excess_Specificity_v1/summary.json")

ARC_STEP_MM = 0.50
N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.55

OUTER_MIN_THICKNESS_MM = 0.35
OUTER_MAX_SAMPLE_MM = 2.40
OUTER_POST_BAND = 2
OUTER_PRE_BAND = 2
FAT_HU = -30.0

VARIANTS = {
    "strict": {
        "min_drop_hu": 100.0,
        "max_post_hu": 80.0,
        "max_thickness_mm": 1.80,
    },
    "nominal": {
        "min_drop_hu": 70.0,
        "max_post_hu": 130.0,
        "max_thickness_mm": 2.00,
    },
    "liberal": {
        "min_drop_hu": 45.0,
        "max_post_hu": 180.0,
        "max_thickness_mm": 2.20,
    },
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

MIN_OUTER_WALL_STATION_QC = 0.80
MIN_MEDIAN_DIRECT_EDGE_FRACTION = 0.25
MIN_POSITIVE_BINS = 4
MIN_NEGATIVE_BINS = 5
MIN_STRICT_BINS = 2
MIN_MAJORITY_AUC = 0.80
MIN_STRICT_AUC = 0.90
MIN_POS_NEG_RATIO = 2.0
MIN_CROSS_VARIANT_SPEARMAN = 0.80

STATUS_PREREQ = "RCA_ADAPTIVE_OUTER_WALL_PREREQUISITE_FAILED"
STATUS_GEOMETRY_FAIL = "RCA_ADAPTIVE_OUTER_WALL_GEOMETRY_FAILED"
STATUS_SPEC_FAIL = "RCA_ADAPTIVE_OUTER_WALL_SPECIFICITY_FAILED"
STATUS_PASS_NO_GAIN = "RCA_ADAPTIVE_OUTER_WALL_PASS_NO_CLEAR_GAIN"
STATUS_GAIN = "RCA_ADAPTIVE_OUTER_WALL_SPECIFICITY_GAIN"


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


def _cmedian(x, width=5):
    half = width // 2
    return np.median(np.stack([np.roll(x, k) for k in range(-half, half + 1)]), axis=0)


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
        r = _cmedian(r, 5)
        r = np.clip(r, med - .75, med + .75)
        r = np.clip(r, .55, 4.0)
        p10, p50, p90 = np.percentile(r, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
        area = float(.5 * np.sum(r*r) * (2*np.pi/N_ANGLES))
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


def _radial_outer_profiles(geom, src, c, L):
    th = L["theta"]
    dirs = np.cos(th)[:, None] * L["u"] + np.sin(th)[:, None] * L["v"]
    off = np.arange(RADIAL_STEP_MM / 2.0, OUTER_MAX_SAMPLE_MM + .30, RADIAL_STEP_MM)
    rr = L["radii"][:, None] + off[None, :]
    P = c[None, None, :] + dirs[:, None, :] * rr[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(rr.shape)
    # Mild radial smoothing only for edge detection; raw values are retained for composition.
    sm = np.empty_like(hu)
    for i in range(len(hu)):
        x = hu[i]
        xp = np.pad(x, (1, 1), mode="edge")
        sm[i] = .25 * xp[:-2] + .50 * xp[1:-1] + .25 * xp[2:]
    return off, hu, sm


def _fill_circular(raw):
    raw = np.asarray(raw, float)
    n = len(raw)
    ok = np.isfinite(raw)
    if not ok.any():
        return np.full(n, np.nan)
    idx = np.flatnonzero(ok)
    vals = raw[ok]
    ext_idx = np.r_[idx - n, idx, idx + n]
    ext_val = np.r_[vals, vals, vals]
    filled = np.interp(np.arange(n), ext_idx, ext_val)
    med = _cmedian(filled, 5)
    return .5 * filled + .5 * med


def _detect_outer_wall(off, sm, variant):
    cfg = VARIANTS[variant]
    n = sm.shape[0]
    thick = np.full(n, np.nan)
    mode = np.zeros(n, dtype=int)  # 0 missing, 1 direct-fat, 2 gradient
    edge_drop = np.full(n, np.nan)
    post_hu = np.full(n, np.nan)
    for i in range(n):
        x = sm[i]
        direct = []
        grad = []
        for j, t in enumerate(off):
            if t < OUTER_MIN_THICKNESS_MM or t > cfg["max_thickness_mm"]:
                continue
            lo0 = max(0, j - OUTER_PRE_BAND)
            lo1 = j + 1
            hi0 = min(len(x), j + 1)
            hi1 = min(len(x), j + 1 + OUTER_POST_BAND + 1)
            if hi1 <= hi0 or lo1 <= lo0:
                continue
            pre = float(np.mean(x[lo0:lo1]))
            post = float(np.mean(x[hi0:hi1]))
            drop = pre - post
            # A robust transition into perivascular fat is the strongest direct evidence.
            if post < FAT_HU and pre > FAT_HU + 25.0 and drop >= 35.0:
                direct.append((float(t), drop, post))
            if drop >= cfg["min_drop_hu"] and post <= cfg["max_post_hu"]:
                score = drop + .15 * max(cfg["max_post_hu"] - post, 0.0) - 8.0 * float(t)
                grad.append((score, float(t), drop, post))
        if direct:
            # Prefer the earliest convincing fat interface so plaque/wall is not expanded into PVAT.
            t, drop, post = sorted(direct, key=lambda z: z[0])[0]
            thick[i], mode[i], edge_drop[i], post_hu[i] = t, 1, drop, post
        elif grad:
            _, t, drop, post = max(grad, key=lambda z: z[0])
            thick[i], mode[i], edge_drop[i], post_hu[i] = t, 2, drop, post
    detected = np.isfinite(thick)
    detected_fraction = float(detected.mean())
    direct_fraction = float((mode == 1).mean())
    if detected_fraction >= .15:
        smooth = _fill_circular(thick)
        smooth = np.clip(smooth, OUTER_MIN_THICKNESS_MM, cfg["max_thickness_mm"])
    else:
        smooth = np.full(n, np.nan)
    return {
        "raw_thickness_mm": thick,
        "smoothed_thickness_mm": smooth,
        "edge_mode": mode,
        "edge_drop_hu": edge_drop,
        "post_edge_hu": post_hu,
        "detected_fraction": detected_fraction,
        "direct_fat_fraction": direct_fraction,
    }


def _integrate_adaptive_wall(geom, src, c, L, outer, ds):
    thick = np.asarray(outer["smoothed_thickness_mm"], float)
    if not np.isfinite(thick).all():
        return None
    th = L["theta"]
    dirs = np.cos(th)[:, None] * L["u"] + np.sin(th)[:, None] * L["v"]
    max_t = float(np.nanmax(thick))
    off = np.arange(RADIAL_STEP_MM/2.0, max_t + 1e-9, RADIAL_STEP_MM)
    rr = L["radii"][:, None] + off[None, :]
    P = c[None, None, :] + dirs[:, None, :] * rr[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(rr.shape)
    inside = off[None, :] <= (thick[:, None] + 1e-9)
    area = rr * RADIAL_STEP_MM * (2*np.pi/N_ANGLES)
    vol = area * float(ds)
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
    vals["median_outer_wall_thickness_mm"] = float(np.median(thick))
    vals["outer_wall_thickness_iqr_mm"] = float(np.percentile(thick, 75) - np.percentile(thick, 25))
    return vals


def _quantify_variant(geom, src, centers, tangents, arcs, weights, variant):
    rows = []
    ray_rows = []
    for i, (c, t, a, ds) in enumerate(zip(centers, tangents, arcs, weights)):
        L = _lumen(geom, src, c, t)
        off, raw_hu, sm_hu = _radial_outer_profiles(geom, src, c, L)
        O = _detect_outer_wall(off, sm_hu, variant) if L["station_qc_pass"] else {
            "smoothed_thickness_mm": np.full(N_ANGLES, np.nan),
            "raw_thickness_mm": np.full(N_ANGLES, np.nan),
            "edge_mode": np.zeros(N_ANGLES, int),
            "edge_drop_hu": np.full(N_ANGLES, np.nan),
            "post_edge_hu": np.full(N_ANGLES, np.nan),
            "detected_fraction": 0.0,
            "direct_fat_fraction": 0.0,
        }
        wall = _integrate_adaptive_wall(geom, src, c, L, O, ds) if L["station_qc_pass"] else None
        outer_qc = bool(
            L["station_qc_pass"]
            and wall is not None
            and O["detected_fraction"] >= .25
            and O["direct_fat_fraction"] >= .10
            and wall["median_outer_wall_thickness_mm"] >= OUTER_MIN_THICKNESS_MM
            and wall["outer_wall_thickness_iqr_mm"] <= 1.10
        )
        base = {k: v for k, v in L.items() if k not in ("radii", "theta", "u", "v")}
        rec = {
            "variant": variant,
            "station_index": i,
            "arc_mm": float(a),
            "integration_ds_mm": float(ds),
            **base,
            "outer_wall_detected_fraction": float(O["detected_fraction"]),
            "direct_fat_edge_fraction": float(O["direct_fat_fraction"]),
            "adaptive_outer_wall_qc_pass": outer_qc,
        }
        if wall is None:
            rec.update({k: np.nan for k in (
                "wall_volume_mm3", "fatlike_excluded_mm3", "low_attenuation_mm3",
                "noncalcified_mm3", "mixed_intermediate_mm3", "calcified_mm3",
                "total_plaque_proxy_mm3", "wall_mean_hu", "median_outer_wall_thickness_mm",
                "outer_wall_thickness_iqr_mm",
            )})
        else:
            rec.update(wall)
        rows.append(rec)
        if variant == NOMINAL_VARIANT:
            for j in range(N_ANGLES):
                ray_rows.append({
                    "station_index": i, "arc_mm": float(a), "angle_index": j,
                    "theta_rad": float(L["theta"][j]),
                    "lumen_radius_mm": float(L["radii"][j]),
                    "raw_outer_wall_thickness_mm": float(O["raw_thickness_mm"][j]) if np.isfinite(O["raw_thickness_mm"][j]) else np.nan,
                    "smoothed_outer_wall_thickness_mm": float(O["smoothed_thickness_mm"][j]) if np.isfinite(O["smoothed_thickness_mm"][j]) else np.nan,
                    "edge_mode": int(O["edge_mode"][j]),
                    "edge_drop_hu": float(O["edge_drop_hu"][j]) if np.isfinite(O["edge_drop_hu"][j]) else np.nan,
                    "post_edge_hu": float(O["post_edge_hu"][j]) if np.isfinite(O["post_edge_hu"][j]) else np.nan,
                })
    return pd.DataFrame(rows), pd.DataFrame(ray_rows)


def _design(df):
    r = df["lumen_radius_median_mm"].to_numpy(float)
    hu = df["center_hu"].to_numpy(float) / 1000.0
    wt = df["median_outer_wall_thickness_mm"].to_numpy(float)
    return np.column_stack([np.ones(len(df)), r, r*r, hu, wt])


def _huber_fit(X, y, k=HUBER_K, n_iter=40):
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 4:
        raise RuntimeError("Insufficient finite RCA reference stations for adaptive-wall model")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        resid = y - X @ beta
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = max(1.4826 * mad, 1e-6)
        a = np.abs(resid - med) / scale
        w = np.ones_like(a)
        hi = a > k
        w[hi] = k / np.maximum(a[hi], 1e-12)
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(new - beta) < 1e-10:
            beta = new
            break
        beta = new
    return beta, y - X @ beta


def _fit_excess_model(stations, variant):
    d = stations[(stations.variant == variant) & stations.adaptive_outer_wall_qc_pass.astype(bool)].copy()
    ref = (d.arc_mm >= REFERENCE_ARC[0]) & (d.arc_mm < REFERENCE_ARC[1])
    if int(ref.sum()) < 25:
        raise RuntimeError(f"Too few adaptive-wall reference stations for {variant}: {int(ref.sum())}")
    X = _design(d)
    models = []
    for c in COMPONENTS:
        frac = d[c].to_numpy(float) / np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        beta, resid = _huber_fit(X[ref], frac[ref])
        q = float(np.quantile(resid[np.isfinite(resid)], REFERENCE_RESIDUAL_QUANTILE))
        models.append({
            "variant": variant,
            "component": c,
            "beta_intercept": float(beta[0]),
            "beta_lumen_radius": float(beta[1]),
            "beta_lumen_radius_sq": float(beta[2]),
            "beta_center_hu_per_1000": float(beta[3]),
            "beta_median_wall_thickness": float(beta[4]),
            "reference_residual_p90": q,
            "reference_station_count": int(ref.sum()),
        })
    return pd.DataFrame(models)


def _apply_excess_model(stations, models, variant):
    d = stations[stations.variant == variant].copy()
    X = _design(d)
    for c in COMPONENTS:
        m = models[(models.variant == variant) & (models.component == c)]
        if len(m) != 1:
            raise RuntimeError(f"Missing adaptive-wall model for {variant} {c}")
        m = m.iloc[0]
        beta = np.array([
            m.beta_intercept, m.beta_lumen_radius, m.beta_lumen_radius_sq,
            m.beta_center_hu_per_1000, m.beta_median_wall_thickness,
        ], float)
        frac = d[c].to_numpy(float) / np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        threshold = X @ beta + float(m.reference_residual_p90)
        excess = np.maximum(frac - threshold, 0.0) * d.wall_volume_mm3.to_numpy(float)
        excess[~d.adaptive_outer_wall_qc_pass.to_numpy(bool)] = np.nan
        stem = c.replace("_mm3", "")
        d[f"expected_p90_{stem}_fraction"] = threshold
        d[f"excess_{stem}_mm3"] = excess
    ex = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    d["excess_total_plaque_proxy_mm3"] = d[ex].sum(axis=1, min_count=1)
    return d


def _profile_1mm(scored):
    d = scored[scored.adaptive_outer_wall_qc_pass.astype(bool)].copy()
    d["arc_start_mm"] = np.floor(d.arc_mm.to_numpy(float)).astype(float)
    rows = []
    excols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    for a, g in d.groupby("arc_start_mm", sort=True):
        r = {
            "variant": str(g.variant.iloc[0]),
            "arc_start_mm": float(a),
            "arc_end_mm": float(a + 1.0),
            "station_count": int(len(g)),
            "outer_wall_qc_fraction": float(g.adaptive_outer_wall_qc_pass.mean()),
            "mean_wall_thickness_mm": float(g.median_outer_wall_thickness_mm.mean()),
            "mean_direct_fat_edge_fraction": float(g.direct_fat_edge_fraction.mean()),
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
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))


def _validate_reference(prior):
    ref = prior[(prior.arc_start_mm >= REFERENCE_ARC[0]) & (prior.arc_start_mm < REFERENCE_ARC[1])]
    bad = int((ref.mapped_native_voxels_vote_ge3.fillna(0).to_numpy(float) > 0).sum())
    majority = int(ref.majority_3plus_signal.fillna(False).astype(bool).sum())
    return bool(len(ref) and bad == 0 and majority == 0), {
        "reference_bins": int(len(ref)),
        "bins_with_vote_ge3": bad,
        "majority_positive_bins": majority,
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
    out["positive_negative_median_ratio"] = mp / max(mn, 1e-6)
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
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[..., None]*u + yy[..., None]*v
    im = _sample(geom, src, P.reshape(-1, 3)).reshape(len(q), len(q))
    return im, q


def _plot_longitudinal(profile, matched, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(profile.arc_start_mm + .5, profile.excess_total_plaque_proxy_mm3, linewidth=2, label="adaptive-wall excess")
    ax.plot(profile.arc_start_mm + .5, profile.raw_wall_volume_mm3, alpha=.35, label="raw wall volume")
    ax.axvspan(*REFERENCE_ARC, alpha=.05)
    ax.axvspan(*HOLDOUT_ARC, alpha=.04)
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Volume per 1-mm bin (mm³)")
    ax2 = ax.twinx()
    if len(matched):
        ax2.step(
            matched.arc_start_mm + .5,
            matched.mapped_native_voxels_vote_ge3.fillna(0),
            where="mid", alpha=.45, label="prior 3+/5 vote voxels"
        )
    ax2.set_ylabel("Prior ensemble 3+/5 vote voxels")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("RCA adaptive outer-wall plaque excess")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_geometry(stations, out):
    d = stations[stations.variant == NOMINAL_VARIANT].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(d.arc_mm, d.median_outer_wall_thickness_mm, label="median adaptive wall thickness")
    ax2 = ax.twinx()
    ax2.plot(d.arc_mm, d.outer_wall_detected_fraction, alpha=.55, label="detected edge fraction")
    ax2.plot(d.arc_mm, d.direct_fat_edge_fraction, alpha=.45, label="direct fat-edge fraction")
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Wall thickness (mm)")
    ax2.set_ylabel("Ray fraction")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("Adaptive outer-wall detection diagnostics")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_variants(profiles, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, p in profiles.items():
        ax.plot(p.arc_start_mm + .5, p.excess_total_plaque_proxy_mm3, label=name)
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Excess plaque proxy per 1-mm bin (mm³)")
    ax.set_title("Adaptive outer-wall detection sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_qc(geom, src, centers, tangents, stations, rays, scored, out):
    d = scored.copy()
    good = d[d.adaptive_outer_wall_qc_pass.astype(bool)]
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
        row = d[d.station_index == i].iloc[0]
        rr = rays[rays.station_index == i].sort_values("angle_index")
        if len(rr) != N_ANGLES:
            continue
        im, q = _plane_image(geom, src, centers[i], tangents[i])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=800, extent=[q[0], q[-1], q[-1], q[0]])
        th = rr.theta_rad.to_numpy(float)
        rin = rr.lumen_radius_mm.to_numpy(float)
        thick = rr.smoothed_outer_wall_thickness_mm.to_numpy(float)
        rout = rin + thick
        ax.plot(rin*np.cos(th), rin*np.sin(th), linewidth=1.1)
        ax.plot(rout*np.cos(th), rout*np.sin(th), linewidth=1.1, linestyle="--")
        ax.scatter([0], [0], s=8)
        ax.set_title(
            f"{row.arc_mm:.1f} mm | wall {row.median_outer_wall_thickness_mm:.2f} | "
            f"direct {row.direct_fat_edge_fraction:.2f}"
        )
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle("RCA adaptive outer-wall source-CCTA QC")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def synthetic_self_test():
    # Circular interpolation should recover a bounded smooth wall thickness.
    raw = np.array([.8, np.nan, np.nan, 1.0, np.nan, .9, np.nan, np.nan], float)
    filled = _fill_circular(raw)
    assert np.isfinite(filled).all()
    assert float(filled.min()) >= .75 and float(filled.max()) <= 1.05
    assert _auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.0
    th = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
    r = np.full(N_ANGLES, 1.5)
    area = .5*np.sum(r*r)*(2*np.pi/N_ANGLES)
    assert abs(area - np.pi*1.5**2) < 1e-10
    return {"ok": True, "filled_min": float(filled.min()), "filled_max": float(filled.max())}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    fixed = _read_json(root / FIXED_SHELL_SUMMARY)
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if fixed.get("status") != "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS":
        raise RuntimeError("Fixed-shell RCA specificity prerequisite failed")

    prior = pd.read_csv(_req(root / PRIOR))
    ref_ok, ref_check = _validate_reference(prior)
    if not ref_ok:
        raise RuntimeError(f"Prespecified RCA 20-50 mm reference zone is no longer vote-free: {ref_check}")

    geom, src, voxel_volume = _load_source(root / SOURCE_CACHE)
    rca = _load_path(root / RCA_CENTERLINE)
    centers, arcs = _resample(rca)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    weights = _arc_weights(arcs)

    center_hu = _sample(geom, src, centers)
    if float(np.mean(center_hu >= 200)) < .95:
        raise RuntimeError("Canonical RCA source-support control failed")

    station_sets = {}
    ray_nominal = None
    models_all = []
    scored_sets = {}
    profiles = {}
    metrics = {}
    geometry_rows = []

    for variant in VARIANTS:
        st, rays = _quantify_variant(geom, src, centers, tangents, arcs, weights, variant)
        station_sets[variant] = st
        if variant == NOMINAL_VARIANT:
            ray_nominal = rays
        models = _fit_excess_model(st, variant)
        models_all.append(models)
        sc = _apply_excess_model(st, models, variant)
        scored_sets[variant] = sc
        profiles[variant] = _profile_1mm(sc)
        matched, met = _holdout_metrics(profiles[variant], prior)
        metrics[variant] = met
        qc = float(st.adaptive_outer_wall_qc_pass.mean())
        med_direct = float(st.direct_fat_edge_fraction.median())
        geometry_rows.append({
            "variant": variant,
            "station_qc_fraction": qc,
            "median_direct_fat_edge_fraction": med_direct,
            "median_detected_edge_fraction": float(st.outer_wall_detected_fraction.median()),
            "median_wall_thickness_mm": float(st.median_outer_wall_thickness_mm.median()),
        })
        matched.to_csv(out / f"RCA_{variant}_holdout_bins.csv", index=False)

    stations = pd.concat(station_sets.values(), ignore_index=True)
    stations.to_csv(out / "RCA_adaptive_outer_wall_stations_all_variants.csv", index=False)
    ray_nominal.to_csv(out / "RCA_nominal_outer_wall_ray_diagnostics.csv", index=False)
    models_df = pd.concat(models_all, ignore_index=True)
    models_df.to_csv(out / "RCA_adaptive_outer_wall_reference_models.csv", index=False)
    scored = pd.concat(scored_sets.values(), ignore_index=True)
    scored.to_csv(out / "RCA_adaptive_outer_wall_excess_stations.csv", index=False)
    all_profiles = pd.concat(profiles.values(), ignore_index=True)
    all_profiles.to_csv(out / "RCA_adaptive_outer_wall_profiles_1mm_all_variants.csv", index=False)
    profiles[NOMINAL_VARIANT].to_csv(out / "RCA_adaptive_outer_wall_profile_1mm_nominal.csv", index=False)

    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(out / "RCA_adaptive_outer_wall_geometry_summary.csv", index=False)
    consistency, min_variant_rho = _variant_consistency(profiles)
    consistency.to_csv(out / "RCA_adaptive_outer_wall_variant_consistency.csv", index=False)

    nominal_st = station_sets[NOMINAL_VARIANT]
    nominal_sc = scored_sets[NOMINAL_VARIANT]
    nominal_profile = profiles[NOMINAL_VARIANT]
    nominal_metric = metrics[NOMINAL_VARIANT]
    nominal_qc = float(nominal_st.adaptive_outer_wall_qc_pass.mean())
    median_direct = float(nominal_st.direct_fat_edge_fraction.median())
    geometry_pass = bool(
        nominal_qc >= MIN_OUTER_WALL_STATION_QC
        and median_direct >= MIN_MEDIAN_DIRECT_EDGE_FRACTION
    )
    specificity_pass = _specificity_pass(nominal_metric, min_variant_rho)

    old_auc = float(fixed["nominal_holdout_specificity"]["majority_vs_vote_free_auc"])
    old_strict = float(fixed["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"])
    new_auc = float(nominal_metric.get("majority_vs_vote_free_auc", np.nan))
    new_strict = float(nominal_metric.get("strict5_vs_vote_free_auc", np.nan))
    auc_gain = new_auc - old_auc if np.isfinite(new_auc) else np.nan
    strict_gain = new_strict - old_strict if np.isfinite(new_strict) else np.nan

    if not geometry_pass:
        status = STATUS_GEOMETRY_FAIL
    elif not specificity_pass:
        status = STATUS_SPEC_FAIL
    elif np.isfinite(auc_gain) and auc_gain >= .02:
        status = STATUS_GAIN
    else:
        status = STATUS_PASS_NO_GAIN

    matched_nominal = pd.read_csv(out / "RCA_nominal_holdout_bins.csv")
    _plot_longitudinal(nominal_profile, matched_nominal, out / "01_RCA_adaptive_outer_wall_excess_vs_prior.png")
    _plot_geometry(nominal_st, out / "02_RCA_adaptive_outer_wall_geometry.png")
    _plot_variants(profiles, out / "03_RCA_adaptive_outer_wall_variant_sensitivity.png")
    _plot_qc(
        geom, src, centers, tangents, nominal_st, ray_nominal, nominal_sc,
        out / "04_RCA_adaptive_outer_wall_source_QC.png"
    )

    pcat_fusion = pd.DataFrame()
    if (root / PCAT).exists():
        pcat = pd.read_csv(root / PCAT)
        pcat_fusion = nominal_profile.merge(pcat, on=["arc_start_mm", "arc_end_mm"], how="inner")
        if not pcat_fusion.empty:
            pcat_fusion.to_csv(out / "RCA_adaptive_outer_wall_PCAT_fusion_10_50.csv", index=False)

    excols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    nominal_totals = {c: float(nominal_profile[c].sum()) for c in excols + ["excess_total_plaque_proxy_mm3"]}
    nominal_totals["raw_wall_volume_mm3"] = float(nominal_profile.raw_wall_volume_mm3.sum())

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "source_voxel_volume_mm3": voxel_volume,
        "reference_zone_mm": list(REFERENCE_ARC),
        "holdout_zone_mm": list(HOLDOUT_ARC),
        "reference_vote_free_check": ref_check,
        "outer_wall_method": {
            "min_thickness_mm": OUTER_MIN_THICKNESS_MM,
            "variants": VARIANTS,
            "nominal_variant": NOMINAL_VARIANT,
            "edge_modes": {"1": "direct transition to < -30 HU perivascular fat", "2": "negative radial gradient fallback"},
            "angular_fill": "circular interpolation plus 5-ray median regularization",
        },
        "nominal_geometry": {
            "station_qc_fraction": nominal_qc,
            "median_direct_fat_edge_fraction": median_direct,
            "median_detected_edge_fraction": float(nominal_st.outer_wall_detected_fraction.median()),
            "median_wall_thickness_mm": float(nominal_st.median_outer_wall_thickness_mm.median()),
            "geometry_gate_pass": geometry_pass,
        },
        "nominal_holdout_specificity": nominal_metric,
        "min_cross_variant_profile_spearman": min_variant_rho,
        "specificity_gate_pass": specificity_pass,
        "comparison_to_fixed_1mm_shell_excess": {
            "prior_majority_auc": old_auc,
            "adaptive_majority_auc": new_auc,
            "majority_auc_delta": auc_gain,
            "prior_strict_auc": old_strict,
            "adaptive_strict_auc": new_strict,
            "strict_auc_delta": strict_gain,
            "prior_fixed_shell_total_excess_mm3": float(fixed["nominal_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"]),
        },
        "nominal_adaptive_wall_totals_mm3": nominal_totals,
        "pcat_fusion_available": bool(not pcat_fusion.empty),
        "is_validated_clinical_tpv": False,
        "scientific_boundary": (
            "This is a developmental adaptive outer-wall source-CCTA experiment on the same RCA previously used to develop the fixed-shell proxy. "
            "Outer-wall candidates are selected from radial source HU transitions without plaque labels, then normal-wall composition is modeled only "
            "in the prespecified RCA 20-50 mm reference zone and evaluated against prior plaque votes in 0-20 mm. "
            "This is not independent external validation and not validated clinical TPV."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "input_provenance.json", {
        "source_cache": str(root / SOURCE_CACHE),
        "rca_centerline": str(root / RCA_CENTERLINE),
        "master": str(root / MASTER),
        "prior_profile": str(root / PRIOR),
        "fixed_shell_summary": str(root / FIXED_SHELL_SUMMARY),
        "pcat_profile": str(root / PCAT),
    })

    report = out / "OPENPLAQUE_RCA_ADAPTIVE_OUTER_WALL_PLAQUE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Adaptive Outer-Wall Plaque v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Nominal outer-wall QC: {nominal_qc:.3f}; median direct-fat edge fraction: {median_direct:.3f}; "
        f"cross-variant Spearman: {min_variant_rho:.3f}.</p>"
        f"<p>Holdout majority AUC: {new_auc:.3f}; strict AUC: {new_strict:.3f}; "
        f"positive/negative ratio: {nominal_metric.get('positive_negative_median_ratio', float('nan')):.3f}.</p>"
        f"<p>Compared with fixed 1-mm shell: majority AUC Δ {auc_gain:+.3f}; strict AUC Δ {strict_gain:+.3f}.</p>"
        "<p><b>Boundary:</b> developmental adaptive outer-wall research proxy on the same RCA; not clinical TPV and not independent validation.</p>"
        "<h2>Nominal geometry</h2><pre>" + json.dumps(summary["nominal_geometry"], indent=2, default=str) + "</pre>"
        "<h2>Nominal specificity</h2><pre>" + json.dumps(nominal_metric, indent=2, default=str) + "</pre>"
        "<h2>Comparison with fixed shell</h2><pre>" + json.dumps(summary["comparison_to_fixed_1mm_shell_excess"], indent=2, default=str) + "</pre>"
        "<h2>Nominal totals</h2><pre>" + json.dumps(nominal_totals, indent=2, default=str) + "</pre>"
        '<img src="01_RCA_adaptive_outer_wall_excess_vs_prior.png" style="max-width:100%">'
        '<img src="02_RCA_adaptive_outer_wall_geometry.png" style="max-width:100%">'
        '<img src="03_RCA_adaptive_outer_wall_variant_sensitivity.png" style="max-width:100%">'
        '<img src="04_RCA_adaptive_outer_wall_source_QC.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out / "run_state.json", {
        "status": "COMPLETE", "result_status": status, "algorithm": ALGORITHM, "baseline": BASELINE
    })
    zpath = out / "OPENPLAQUE_RCA_ADAPTIVE_OUTER_WALL_PLAQUE_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
