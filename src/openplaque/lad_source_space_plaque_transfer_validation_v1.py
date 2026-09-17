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
ALGORITHM = "lad-source-space-plaque-transfer-validation-v1.0"
OUTPUT_DIRNAME = "LAD_Source_Space_Plaque_Transfer_Validation_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
FROZEN_LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
DISTAL_EXTENSION = Path("LAD_Distal_Endpoint_Continuation_v1/best_independent_distal_extension.csv")
DISTAL_BIDIR_SUMMARY = Path("LAD_Distal_Bidirectional_Validation_v1/summary.json")
RCA_STATIONS = Path("RCA_Source_Space_Plaque_Quantification_v1/RCA_source_space_station_quantification.csv")
RCA_EXCESS_SUMMARY = Path("RCA_Source_Space_Plaque_Excess_Specificity_v1/summary.json")
RCA_PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
LAD_PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/LAD_source_longitudinal_plaque_profile_1mm.csv")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

ARC_STEP_MM = 0.50
N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.55
SHELLS = (0.75, 1.0, 1.25, 1.5)
NOMINAL_SHELL = 1.0
RCA_REFERENCE_ARC = (20.0, 50.0)
REFERENCE_RESIDUAL_QUANTILE = 0.90
HUBER_K = 1.35

COMPONENTS = (
    "low_attenuation_mm3",
    "noncalcified_mm3",
    "mixed_intermediate_mm3",
    "calcified_mm3",
)

MIN_FROZEN_STATION_QC = 0.90
MIN_EXTENSION_STATION_QC = 0.85
MIN_POSITIVE_BINS = 4
MIN_NEGATIVE_BINS = 5
MIN_STRICT_BINS = 2
MIN_MAJORITY_AUC = 0.75
MIN_STRICT_AUC = 0.85
MIN_POS_NEG_RATIO = 1.50
MIN_CROSS_SHELL_SPEARMAN = 0.75

STATUS_PREREQ = "LAD_SOURCE_SPACE_PLAQUE_TRANSFER_PREREQUISITE_FAILED"
STATUS_FAIL = "LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATION_FAILED"
STATUS_PASS = "LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATED"


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
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0.0, float(a[-1]) + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
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


def _integrate_shell(geom, src, c, L, shell, ds):
    th = L["theta"]
    dirs = np.cos(th)[:, None] * L["u"] + np.sin(th)[:, None] * L["v"]
    off = np.arange(RADIAL_STEP_MM/2.0, shell, RADIAL_STEP_MM)
    if len(off) == 0:
        off = np.array([shell/2.0])
    r = L["radii"][:, None] + off[None, :]
    P = c[None, None, :] + dirs[:, None, :] * r[..., None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(r.shape)
    area = r * RADIAL_STEP_MM * (2*np.pi/N_ANGLES)
    vol = area * float(ds)
    finite = np.isfinite(hu)
    masks = {
        "fatlike_excluded_mm3": finite & (hu < -30),
        "low_attenuation_mm3": finite & (hu >= -30) & (hu < 30),
        "noncalcified_mm3": finite & (hu >= 30) & (hu < 130),
        "mixed_intermediate_mm3": finite & (hu >= 130) & (hu < 350),
        "calcified_mm3": finite & (hu >= 350),
    }
    vals = {k: float(vol[m].sum()) for k, m in masks.items()}
    vals["shell_volume_mm3"] = float(vol[finite].sum())
    vals["total_plaque_proxy_mm3"] = sum(vals[c] for c in COMPONENTS)
    return vals


def _plane_image(geom, src, c, t, half=5.0, step=.15):
    u, v = _orth_basis(t)
    q = np.arange(-half, half + 1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[..., None]*u + yy[..., None]*v
    im = _sample(geom, src, P.reshape(-1, 3)).reshape(len(q), len(q))
    return im, q


def _join_frozen_and_extension(frozen, extension, max_join_mm=1.5):
    frozen = np.asarray(frozen, float)
    extension = np.asarray(extension, float)
    d_first = min(
        float(np.linalg.norm(extension[0] - frozen[0])),
        float(np.linalg.norm(extension[-1] - frozen[0])),
    )
    d_last = min(
        float(np.linalg.norm(extension[0] - frozen[-1])),
        float(np.linalg.norm(extension[-1] - frozen[-1])),
    )
    if d_first > max_join_mm:
        raise RuntimeError(
            f"Validated distal extension does not join frozen LAD arc-0 endpoint: "
            f"first={d_first:.3f} mm, last={d_last:.3f} mm"
        )
    ext = extension if np.linalg.norm(extension[-1] - frozen[0]) <= np.linalg.norm(extension[0] - frozen[0]) else extension[::-1].copy()
    join_gap = float(np.linalg.norm(ext[-1] - frozen[0]))
    if join_gap > max_join_mm:
        raise RuntimeError(f"Extension/frozen LAD join gap {join_gap:.3f} mm exceeds {max_join_mm}")
    ext_len = float(_arc(ext)[-1])
    combined = np.vstack([ext, frozen[1:]])
    return combined, ext_len, join_gap


def _design(df):
    r = df["lumen_radius_median_mm"].to_numpy(float)
    hu = df["center_hu"].to_numpy(float) / 1000.0
    return np.column_stack([np.ones(len(df)), r, r*r, hu])


def _huber_fit(X, y, k=HUBER_K, n_iter=40):
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 4:
        raise RuntimeError("Insufficient finite RCA reference stations")
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


def _fit_rca_models(stations):
    models = []
    for shell in SHELLS:
        d = stations[np.isclose(stations.shell_thickness_mm, shell) & stations.station_qc_pass.astype(bool)].copy()
        ref = (d.arc_mm >= RCA_REFERENCE_ARC[0]) & (d.arc_mm < RCA_REFERENCE_ARC[1])
        if int(ref.sum()) < 25:
            raise RuntimeError(f"Too few RCA reference stations for shell {shell}: {int(ref.sum())}")
        X = _design(d)
        for c in COMPONENTS:
            frac = d[c].to_numpy(float) / np.maximum(d.shell_volume_mm3.to_numpy(float), 1e-9)
            beta, resid_ref = _huber_fit(X[ref], frac[ref])
            q = float(np.quantile(resid_ref[np.isfinite(resid_ref)], REFERENCE_RESIDUAL_QUANTILE))
            models.append({
                "shell_thickness_mm": float(shell),
                "component": c,
                "beta_intercept": float(beta[0]),
                "beta_lumen_radius": float(beta[1]),
                "beta_lumen_radius_sq": float(beta[2]),
                "beta_center_hu_per_1000": float(beta[3]),
                "reference_residual_p90": q,
                "reference_station_count": int(ref.sum()),
            })
    return pd.DataFrame(models)


def _apply_models(stations, models):
    out = []
    for shell in SHELLS:
        d = stations[np.isclose(stations.shell_thickness_mm, shell)].copy()
        X = _design(d)
        for c in COMPONENTS:
            m = models[(np.isclose(models.shell_thickness_mm, shell)) & (models.component == c)]
            if len(m) != 1:
                raise RuntimeError(f"Missing unique model for shell {shell}, component {c}")
            m = m.iloc[0]
            beta = np.array([
                m.beta_intercept, m.beta_lumen_radius, m.beta_lumen_radius_sq, m.beta_center_hu_per_1000
            ], float)
            frac = d[c].to_numpy(float) / np.maximum(d.shell_volume_mm3.to_numpy(float), 1e-9)
            threshold = X @ beta + float(m.reference_residual_p90)
            excess = np.maximum(frac - threshold, 0.0) * d.shell_volume_mm3.to_numpy(float)
            stem = c.replace("_mm3", "")
            d[f"rca_expected_p90_{stem}_fraction"] = threshold
            d[f"excess_{stem}_mm3"] = excess
        ex = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
        d["excess_total_plaque_proxy_mm3"] = d[ex].sum(axis=1)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _profile_1mm(d, shell):
    x = d[np.isclose(d.shell_thickness_mm, shell) & d.station_qc_pass.astype(bool)].copy()
    x["arc_start_mm"] = np.floor(x.frozen_arc_mm.to_numpy(float)).astype(float)
    rows = []
    ex_cols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    for a, g in x.groupby("arc_start_mm", sort=True):
        r = {
            "shell_thickness_mm": float(shell),
            "arc_start_mm": float(a),
            "arc_end_mm": float(a + 1.0),
            "region": "distal_extension" if a < 0 else "frozen_lad",
            "station_count": int(len(g)),
            "station_qc_fraction": float(g.station_qc_pass.mean()),
            "raw_total_plaque_proxy_mm3": float(g.total_plaque_proxy_mm3.sum()),
        }
        for c in ex_cols:
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
    u = float(r[y == 1].sum() - n1*(n1+1)/2.0)
    return u / float(n1*n0)


def _transfer_metrics(profile, prior, frozen_length):
    p = profile[(profile.arc_start_mm >= 0) & (profile.arc_start_mm < math.floor(frozen_length))].copy()
    z = p.merge(prior, on=["arc_start_mm", "arc_end_mm"], how="inner")
    pos = z.majority_3plus_signal.fillna(False).astype(bool)
    neg = z.mapped_native_vote_sum.fillna(0).to_numpy(float) == 0
    strict = z.strict_5of5_signal.fillna(False).astype(bool)
    score = z.excess_total_plaque_proxy_mm3.to_numpy(float)
    out = {
        "usable": bool(int(pos.sum()) >= MIN_POSITIVE_BINS and int(neg.sum()) >= MIN_NEGATIVE_BINS),
        "matched_frozen_bins": int(len(z)),
        "positive_bins": int(pos.sum()),
        "strict_5of5_bins": int(strict.sum()),
        "vote_free_negative_bins": int(neg.sum()),
    }
    if not out["usable"]:
        return z, out
    y = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
    s = np.r_[score[pos], score[neg]]
    out["majority_vs_vote_free_auc"] = float(_auc(y, s))
    if int(strict.sum()) >= MIN_STRICT_BINS:
        ys = np.r_[np.ones(int(strict.sum())), np.zeros(int(neg.sum()))]
        ss = np.r_[score[strict], score[neg]]
        out["strict5_vs_vote_free_auc"] = float(_auc(ys, ss))
    else:
        out["strict5_vs_vote_free_auc"] = np.nan
    med_pos = float(np.median(score[pos]))
    med_neg = float(np.median(score[neg]))
    out["positive_median_excess_mm3"] = med_pos
    out["negative_median_excess_mm3"] = med_neg
    out["positive_negative_median_ratio"] = med_pos / max(med_neg, 1e-6)
    out["spearman_excess_vs_vote_ge3"] = float(
        spearmanr(z.excess_total_plaque_proxy_mm3, z.mapped_native_voxels_vote_ge3).statistic
    )
    return z, out


def _cross_shell(profiles, frozen_length):
    nom = profiles[NOMINAL_SHELL]
    nom = nom[(nom.arc_start_mm >= 0) & (nom.arc_start_mm < math.floor(frozen_length))]
    nom = nom.set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"]
    rows = []
    for shell, p in profiles.items():
        p = p[(p.arc_start_mm >= 0) & (p.arc_start_mm < math.floor(frozen_length))]
        y = p.set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"].reindex(nom.index)
        ok = nom.notna() & y.notna()
        rho = float(spearmanr(nom[ok], y[ok]).statistic) if int(ok.sum()) >= 5 else np.nan
        rows.append({"shell_thickness_mm": float(shell), "profile_spearman_vs_nominal": rho})
    d = pd.DataFrame(rows)
    f = d[~np.isclose(d.shell_thickness_mm, NOMINAL_SHELL)].profile_spearman_vs_nominal
    f = f[np.isfinite(f)]
    return d, float(f.min()) if len(f) else np.nan


def _plot_transfer(profile, matched, frozen_length, extension_length, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(profile.arc_start_mm + .5, profile.excess_total_plaque_proxy_mm3, label="LAD RCA-trained excess")
    ax.axvline(0, linestyle="--", linewidth=1, label="frozen distal endpoint")
    ax.axvspan(-extension_length, 0, alpha=.06, label="bidirectionally confirmed distal extension")
    ax.set_xlabel("LAD frozen-axis arc (mm; extension is negative)")
    ax.set_ylabel("Excess plaque proxy per 1-mm bin (mm³)")
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
    ax.set_title("LAD cross-vessel plaque transfer: RCA-trained normal-wall model")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_shells(profiles, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for shell, p in profiles.items():
        ax.plot(p.arc_start_mm + .5, p.excess_total_plaque_proxy_mm3, label=f"shell {shell:.2f} mm")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("LAD frozen-axis arc (mm)")
    ax.set_ylabel("Excess plaque proxy per 1-mm bin (mm³)")
    ax.set_title("LAD excess profile across shell sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_components(profile, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = profile.arc_start_mm + .5
    for c in COMPONENTS:
        stem = c.replace("_mm3", "")
        ax.plot(x, profile[f"excess_{stem}_mm3"], label=stem.replace("_", " "))
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("LAD frozen-axis arc (mm)")
    ax.set_ylabel("Excess component volume per 1-mm bin (mm³)")
    ax.set_title("LAD transferred excess composition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_qc(geom, src, centers, tangents, nominal, out):
    good = nominal[nominal.station_qc_pass.astype(bool)].copy()
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
        row = nominal[nominal.station_index == i].iloc[0]
        im, q = _plane_image(geom, src, centers[i], tangents[i])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=800, extent=[q[0], q[-1], q[-1], q[0]])
        r = float(row.lumen_radius_median_mm)
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linewidth=1.2))
        ax.add_patch(plt.Circle((0, 0), r + NOMINAL_SHELL, fill=False, linewidth=1.2, linestyle="--"))
        ax.scatter([0], [0], s=8)
        ax.set_title(f"arc {row.frozen_arc_mm:.1f} | excess {row.excess_total_plaque_proxy_mm3:.2f} mm³")
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle("Automatically selected LAD source-CCTA QC")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def synthetic_self_test():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(_auc(y, s) - 1.0) < 1e-12
    frozen = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    ext = np.array([[-2., 0., 0.], [-1., 0., 0.], [0., 0., 0.]])
    combined, elen, gap = _join_frozen_and_extension(frozen, ext)
    assert len(combined) == 5 and abs(elen - 2.0) < 1e-12 and gap < 1e-12
    return {"ok": True, "auc": 1.0, "joined_length_mm": float(_arc(combined)[-1])}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    rca_summary = _read_json(root / RCA_EXCESS_SUMMARY)
    lad_bidir = _read_json(root / DISTAL_BIDIR_SUMMARY)
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if rca_summary.get("status") != "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS":
        raise RuntimeError("RCA excess-specificity prerequisite failed")
    if lad_bidir.get("status") != "LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED":
        raise RuntimeError("Bidirectional LAD distal-extension prerequisite failed")

    frozen = _load_path(root / FROZEN_LAD)
    extension = _load_path(root / DISTAL_EXTENSION)
    combined, extension_length, join_gap = _join_frozen_and_extension(frozen, extension)
    frozen_length = float(_arc(frozen)[-1])

    geom, src, voxel_volume = _load_source(root / SOURCE_CACHE)
    centers, combined_arc = _resample(combined, ARC_STEP_MM)
    frozen_arc = combined_arc - extension_length
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    weights = _arc_weights(combined_arc)

    rows = []
    for i, (c, t, ca, fa, ds) in enumerate(zip(centers, tangents, combined_arc, frozen_arc, weights)):
        L = _lumen(geom, src, c, t)
        base = {k: v for k, v in L.items() if k not in ("radii", "theta", "u", "v")}
        region = "distal_extension" if fa < -0.25 else "frozen_lad"
        for shell in SHELLS:
            if L["station_qc_pass"]:
                comp = _integrate_shell(geom, src, c, L, shell, ds)
            else:
                comp = {k: np.nan for k in (
                    "shell_volume_mm3", "fatlike_excluded_mm3", "low_attenuation_mm3",
                    "noncalcified_mm3", "mixed_intermediate_mm3", "calcified_mm3",
                    "total_plaque_proxy_mm3"
                )}
            rows.append({
                "station_index": i,
                "combined_arc_mm": float(ca),
                "frozen_arc_mm": float(fa),
                "integration_ds_mm": float(ds),
                "shell_thickness_mm": float(shell),
                "region": region,
                **base, **comp,
            })
    stations = pd.DataFrame(rows)
    stations.to_csv(out / "LAD_source_space_station_quantification.csv", index=False)

    rca_stations = pd.read_csv(_req(root / RCA_STATIONS))
    models = _fit_rca_models(rca_stations)
    models.to_csv(out / "RCA_reference_normal_wall_models_frozen.csv", index=False)

    scored = _apply_models(stations, models)
    scored.to_csv(out / "LAD_source_space_transferred_excess_stations.csv", index=False)

    profiles = {shell: _profile_1mm(scored, shell) for shell in SHELLS}
    all_profiles = pd.concat(profiles.values(), ignore_index=True)
    all_profiles.to_csv(out / "LAD_transferred_excess_profile_1mm_all_shells.csv", index=False)
    nominal = profiles[NOMINAL_SHELL].copy()
    nominal.to_csv(out / "LAD_transferred_excess_profile_1mm_nominal.csv", index=False)

    prior = pd.read_csv(_req(root / LAD_PRIOR))
    matched, transfer = _transfer_metrics(nominal, prior, frozen_length)
    matched.to_csv(out / "LAD_transfer_validation_bins.csv", index=False)

    shell_consistency, min_shell_rho = _cross_shell(profiles, frozen_length)
    shell_consistency.to_csv(out / "LAD_transfer_shell_consistency.csv", index=False)

    nominal_st = scored[np.isclose(scored.shell_thickness_mm, NOMINAL_SHELL)].copy()
    frozen_qc = float(nominal_st[nominal_st.frozen_arc_mm >= -0.25].station_qc_pass.mean())
    ext_qc = float(nominal_st[nominal_st.frozen_arc_mm < -0.25].station_qc_pass.mean())

    frozen_nom = nominal[(nominal.arc_start_mm >= 0) & (nominal.arc_start_mm < math.ceil(frozen_length))]
    ext_nom = nominal[nominal.arc_start_mm < 0]
    ex_cols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    frozen_totals = {c: float(frozen_nom[c].sum()) for c in ex_cols + ["excess_total_plaque_proxy_mm3"]}
    extension_totals = {c: float(ext_nom[c].sum()) for c in ex_cols + ["excess_total_plaque_proxy_mm3"]}

    transfer_pass = bool(
        transfer.get("usable", False)
        and transfer.get("majority_vs_vote_free_auc", -np.inf) >= MIN_MAJORITY_AUC
        and (
            transfer.get("strict_5of5_bins", 0) < MIN_STRICT_BINS
            or transfer.get("strict5_vs_vote_free_auc", -np.inf) >= MIN_STRICT_AUC
        )
        and transfer.get("positive_negative_median_ratio", -np.inf) >= MIN_POS_NEG_RATIO
        and np.isfinite(min_shell_rho) and min_shell_rho >= MIN_CROSS_SHELL_SPEARMAN
        and frozen_qc >= MIN_FROZEN_STATION_QC
        and ext_qc >= MIN_EXTENSION_STATION_QC
    )
    status = STATUS_PASS if transfer_pass else STATUS_FAIL

    _plot_transfer(nominal, matched, frozen_length, extension_length, out / "01_LAD_transferred_excess_vs_prior.png")
    _plot_shells(profiles, out / "02_LAD_transfer_shell_consistency.png")
    _plot_components(nominal, out / "03_LAD_transferred_excess_composition.png")
    _plot_qc(geom, src, centers, tangents, nominal_st, out / "04_LAD_source_orthogonal_QC.png")

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "rca_model_prerequisite_status": rca_summary.get("status"),
        "lad_bidirectional_prerequisite_status": lad_bidir.get("status"),
        "frozen_lad_length_mm": frozen_length,
        "bidirectionally_confirmed_distal_extension_length_mm": extension_length,
        "combined_research_lad_length_mm": float(_arc(combined)[-1]),
        "extension_frozen_join_gap_mm": join_gap,
        "source_voxel_volume_mm3": voxel_volume,
        "nominal_frozen_station_qc_fraction": frozen_qc,
        "nominal_extension_station_qc_fraction": ext_qc,
        "transfer_validation": transfer,
        "min_cross_shell_frozen_profile_spearman": min_shell_rho,
        "nominal_frozen_excess_totals_mm3": frozen_totals,
        "nominal_distal_extension_excess_totals_mm3": extension_totals,
        "is_validated_clinical_tpv": False,
        "scientific_boundary": (
            "The RCA normal-wall/excess model is frozen from the prespecified RCA 20-50 mm reference zone "
            "and transferred to the LAD without LAD refitting. Validation uses prior LAD ensemble plaque votes "
            "only on the frozen LAD arc. Distal-extension volumes are exploratory research measurements. "
            "This is not independently segmented outer-wall TPV and not validated clinical TPV."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "input_provenance.json", {
        "source_cache": str(root / SOURCE_CACHE),
        "frozen_lad": str(root / FROZEN_LAD),
        "distal_extension": str(root / DISTAL_EXTENSION),
        "distal_bidirectional_summary": str(root / DISTAL_BIDIR_SUMMARY),
        "rca_stations": str(root / RCA_STATIONS),
        "rca_excess_summary": str(root / RCA_EXCESS_SUMMARY),
        "lad_prior_profile": str(root / LAD_PRIOR),
    })

    report = out / "OPENPLAQUE_LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATION_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque LAD Source-Space Plaque Transfer Validation v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Frozen LAD: {frozen_length:.2f} mm; confirmed distal extension: {extension_length:.2f} mm; "
        f"combined research path: {_arc(combined)[-1]:.2f} mm.</p>"
        f"<p>Frozen station QC: {frozen_qc:.3f}; extension station QC: {ext_qc:.3f}; "
        f"minimum cross-shell Spearman: {min_shell_rho:.3f}.</p>"
        "<h2>Transfer validation</h2><pre>" + json.dumps(transfer, indent=2, default=str) + "</pre>"
        "<h2>Frozen LAD nominal excess totals</h2><pre>" + json.dumps(frozen_totals, indent=2) + "</pre>"
        "<h2>Distal extension nominal excess totals (exploratory)</h2><pre>" + json.dumps(extension_totals, indent=2) + "</pre>"
        "<p><b>Research boundary:</b> RCA-trained reference-normalized excess proxy transferred without LAD refitting; "
        "not validated clinical TPV.</p>"
        '<img src="01_LAD_transferred_excess_vs_prior.png" style="max-width:100%">'
        '<img src="02_LAD_transfer_shell_consistency.png" style="max-width:100%">'
        '<img src="03_LAD_transferred_excess_composition.png" style="max-width:100%">'
        '<img src="04_LAD_source_orthogonal_QC.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )
    _write_json(out / "run_state.json", {
        "status": "COMPLETE", "result_status": status, "algorithm": ALGORITHM, "baseline": BASELINE
    })
    zpath = out / "OPENPLAQUE_LAD_SOURCE_SPACE_PLAQUE_TRANSFER_VALIDATION_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
