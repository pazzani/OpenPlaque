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
from scipy.stats import spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-expert-outer-wall-validation-v1.0"
OUTPUT_DIRNAME = "RCA_Expert_Outer_Wall_Validation_v1"

PACK_DIR = Path("RCA_Expert_Outer_Wall_Annotation_Pack_v1/expert_pack")
LOCK_DIR = Path("RCA_Plaque_PCAT_Research_Lock_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

PACK_NPZ = PACK_DIR / "RCA_expert_outer_wall_planes.npz"
MANIFEST = PACK_DIR / "RCA_expert_outer_wall_manifest.csv"
LOCK_SUMMARY = LOCK_DIR / "summary.json"
LOCK_PROFILE = LOCK_DIR / "RCA_locked_research_plaque_profile_1mm.csv"

MASK_CANDIDATES = (
    PACK_DIR / "RCA_outer_wall_expert_mask.npy",
    Path("RCA_Expert_Outer_Wall_Annotation_Pack_v1/RCA_outer_wall_expert_mask.npy"),
    Path("RCA_outer_wall_expert_mask.npy"),
)

EXPECTED_LOCK_STATUS = "RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED"
STATUS_MASK_MISSING = "RCA_EXPERT_OUTER_WALL_MASK_MISSING"
STATUS_MASK_QC_FAILED = "RCA_EXPERT_OUTER_WALL_MASK_QC_FAILED"
STATUS_COMPLETE = "RCA_EXPERT_OUTER_WALL_VALIDATION_COMPLETE"

REFERENCE_ARC = (20.0, 50.0)
HOLDOUT_ARC = (0.0, 20.0)
REFERENCE_RESIDUAL_QUANTILE = 0.90
HUBER_K = 1.35
N_ANGLES = 72

MIN_ANNOTATED_SLICES = 41
MIN_REFERENCE_SLICES = 20
MIN_HOLDOUT_SLICES = 15
MIN_CONTAINMENT_FRACTION = 0.98

COMPONENTS = (
    "low_attenuation_mm3",
    "noncalcified_mm3",
    "mixed_intermediate_mm3",
    "calcified_mm3",
)


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _find_mask(root):
    for rel in MASK_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    return None


def _integration_weights(arcs):
    arcs = np.asarray(arcs, float)
    if len(arcs) == 1:
        return np.ones(1)
    w = np.empty(len(arcs), float)
    w[0] = 0.5 * (arcs[1] - arcs[0])
    w[-1] = 0.5 * (arcs[-1] - arcs[-2])
    if len(arcs) > 2:
        w[1:-1] = 0.5 * (arcs[2:] - arcs[:-2])
    return w


def _component_volumes(hu, wall, pixel_mm, ds_mm):
    vals = hu[wall]
    pixvol = float(pixel_mm * pixel_mm * ds_mm)
    rec = {
        "fatlike_excluded_mm3": float(np.sum(vals < -30) * pixvol),
        "low_attenuation_mm3": float(np.sum((vals >= -30) & (vals < 30)) * pixvol),
        "noncalcified_mm3": float(np.sum((vals >= 30) & (vals < 130)) * pixvol),
        "mixed_intermediate_mm3": float(np.sum((vals >= 130) & (vals < 350)) * pixvol),
        "calcified_mm3": float(np.sum(vals >= 350) * pixvol),
    }
    rec["total_plaque_proxy_mm3"] = sum(rec[c] for c in COMPONENTS)
    rec["wall_volume_mm3"] = float(np.sum(wall) * pixvol)
    rec["wall_mean_hu"] = float(np.mean(vals)) if len(vals) else np.nan
    return rec


def _radius_map(shape, pixel_mm):
    ny, nx = shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices(shape)
    x = (xx - cx) * pixel_mm
    y = (yy - cy) * pixel_mm
    r = np.sqrt(x*x + y*y)
    ang = np.mod(np.arctan2(y, x), 2*np.pi)
    return r, ang


def _mask_outer_radii(mask, pixel_mm, n_angles=N_ANGLES):
    mask = np.asarray(mask, bool)
    r, ang = _radius_map(mask.shape, pixel_mm)
    bins = np.floor(ang / (2*np.pi) * n_angles).astype(int) % n_angles
    out = np.full(n_angles, np.nan)
    for a in range(n_angles):
        sel = mask & (bins == a)
        if np.any(sel):
            out[a] = float(np.max(r[sel]))
    if np.isfinite(out).any():
        idx = np.flatnonzero(np.isfinite(out))
        vals = out[idx]
        ext_idx = np.r_[idx - n_angles, idx, idx + n_angles]
        ext_val = np.r_[vals, vals, vals]
        out = np.interp(np.arange(n_angles), ext_idx, ext_val)
    return out


def _fixed_outer_mask(lumen_radii, pixel_mm, shape, shell_mm=1.0):
    r, ang = _radius_map(shape, pixel_mm)
    n = len(lumen_radii)
    bins = np.floor(ang / (2*np.pi) * n).astype(int) % n
    outer = np.asarray(lumen_radii, float) + float(shell_mm)
    return r <= outer[bins]


def _dice(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    den = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / den) if den else np.nan


def _jaccard(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    den = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / den) if den else np.nan


def _design(df):
    r = df["lumen_radius_median_mm"].to_numpy(float)
    hu = df["center_hu"].to_numpy(float) / 1000.0
    return np.column_stack([np.ones(len(df)), r, r*r, hu])


def _huber_fit(X, y, k=HUBER_K, n_iter=40):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[ok]
    y = y[ok]
    if len(y) < X.shape[1] + 4:
        raise RuntimeError("Insufficient reference slices for expert-wall robust fit")
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


def _expert_reference_normalize(df):
    d = df[df["annotated"] & df["outer_contains_lumen"]].copy()
    ref = (d.arc_mm >= REFERENCE_ARC[0]) & (d.arc_mm <= REFERENCE_ARC[1])
    if int(ref.sum()) < MIN_REFERENCE_SLICES:
        raise RuntimeError(f"Need at least {MIN_REFERENCE_SLICES} annotated reference slices, found {int(ref.sum())}")

    X = _design(d)
    models = []
    for c in COMPONENTS:
        frac = d[c].to_numpy(float) / np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        beta, resid = _huber_fit(X[ref], frac[ref])
        q = float(np.quantile(resid[np.isfinite(resid)], REFERENCE_RESIDUAL_QUANTILE))
        models.append({
            "component": c,
            "beta_intercept": float(beta[0]),
            "beta_lumen_radius": float(beta[1]),
            "beta_lumen_radius_sq": float(beta[2]),
            "beta_center_hu_per_1000": float(beta[3]),
            "reference_residual_p90": q,
            "reference_slice_count": int(ref.sum()),
        })

    models_df = pd.DataFrame(models)
    Xall = _design(d)
    for c in COMPONENTS:
        m = models_df[models_df.component == c].iloc[0]
        beta = np.array([
            m.beta_intercept,
            m.beta_lumen_radius,
            m.beta_lumen_radius_sq,
            m.beta_center_hu_per_1000,
        ], float)
        frac = d[c].to_numpy(float) / np.maximum(d.wall_volume_mm3.to_numpy(float), 1e-9)
        threshold = Xall @ beta + float(m.reference_residual_p90)
        excess = np.maximum(frac - threshold, 0.0) * d.wall_volume_mm3.to_numpy(float)
        stem = c.replace("_mm3", "")
        d[f"expert_excess_{stem}_mm3"] = excess

    excols = [f"expert_excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    d["expert_excess_total_mm3"] = d[excols].sum(axis=1)
    return d, models_df


def _compare_locked(expert_df, locked):
    e = expert_df.copy()
    e["arc_start_mm"] = e.arc_mm.astype(float)
    e["arc_end_mm"] = e.arc_start_mm + 1.0
    z = e.merge(
        locked[[
            "arc_start_mm", "arc_end_mm",
            "excess_total_plaque_proxy_mm3",
            "excess_low_attenuation_mm3",
            "excess_noncalcified_mm3",
            "excess_mixed_intermediate_mm3",
            "excess_calcified_mm3",
        ]],
        on=["arc_start_mm", "arc_end_mm"],
        how="inner",
    )
    hold = z[(z.arc_start_mm >= HOLDOUT_ARC[0]) & (z.arc_start_mm < HOLDOUT_ARC[1])].copy()
    if len(hold) < MIN_HOLDOUT_SLICES:
        return z, {
            "usable": False,
            "matched_holdout_slices": int(len(hold)),
        }

    rho = float(spearmanr(
        hold.expert_excess_total_mm3,
        hold.excess_total_plaque_proxy_mm3,
    ).statistic)
    diff = hold.excess_total_plaque_proxy_mm3 - hold.expert_excess_total_mm3
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    locked_total = float(hold.excess_total_plaque_proxy_mm3.sum())
    expert_total = float(hold.expert_excess_total_mm3.sum())
    return z, {
        "usable": True,
        "matched_holdout_slices": int(len(hold)),
        "spearman_locked_vs_expert_excess": rho,
        "mean_absolute_difference_mm3_per_slice": mae,
        "mean_locked_minus_expert_bias_mm3_per_slice": bias,
        "locked_holdout_total_excess_mm3": locked_total,
        "expert_holdout_total_excess_mm3": expert_total,
        "locked_to_expert_total_ratio": locked_total / max(expert_total, 1e-9),
    }


def _plot_boundary_agreement(df, out):
    g = df[df.annotated].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(g.arc_mm, g.expert_median_wall_thickness_mm, label="expert median wall thickness")
    ax.axhline(1.0, linestyle="--", linewidth=1, label="locked fixed-shell thickness")
    ax2 = ax.twinx()
    ax2.plot(g.arc_mm, g.fixed_outer_vs_expert_dice, alpha=.55, label="fixed-vs-expert Dice")
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Wall thickness (mm)")
    ax2.set_ylabel("Outer-vessel Dice")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("Expert outer wall vs locked 1.0-mm fixed shell")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_profile_comparison(z, out):
    d = z[(z.arc_start_mm >= HOLDOUT_ARC[0]) & (z.arc_start_mm < HOLDOUT_ARC[1])].copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(d.arc_start_mm + .5, d.excess_total_plaque_proxy_mm3, linewidth=2, label="locked fixed-shell excess")
    ax.plot(d.arc_start_mm + .5, d.expert_excess_total_mm3, linewidth=2, label="expert-wall excess")
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Reference-normalized excess per 1-mm slice (mm³)")
    ax.set_title("Locked plaque proxy vs blinded expert outer-wall proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_area_bias(df, out):
    d = df[df.annotated].copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(d.arc_mm, d.expert_wall_area_mm2, label="expert wall area")
    ax.plot(d.arc_mm, d.fixed_shell_wall_area_mm2, label="fixed-shell wall area")
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Wall area (mm²)")
    ax.set_title("Expert vs fixed-shell wall area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    shape = (67, 67)
    pixel = 0.15
    lumen = np.full(N_ANGLES, 1.5)
    fixed = _fixed_outer_mask(lumen, pixel, shape, 1.0)
    r, _ = _radius_map(shape, pixel)
    expert = r <= 2.3
    assert 0 < _dice(fixed, expert) <= 1
    assert 0 < _jaccard(fixed, expert) <= 1
    radii = _mask_outer_radii(expert, pixel)
    assert np.isfinite(radii).all()
    assert 2.1 < float(np.median(radii)) < 2.5
    return {
        "ok": True,
        "dice": _dice(fixed, expert),
        "median_expert_radius_mm": float(np.median(radii)),
    }


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    lock_summary = _read_json(root / LOCK_SUMMARY)
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if lock_summary.get("status") != EXPECTED_LOCK_STATUS:
        raise RuntimeError(f"Locked RCA plaque prerequisite failed: {lock_summary.get('status')}")

    mask_path = _find_mask(root)
    if mask_path is None:
        _write_json(out / "run_state.json", {
            "status": "COMPLETE",
            "result_status": STATUS_MASK_MISSING,
            "algorithm": ALGORITHM,
            "baseline": BASELINE,
        })
        _write_json(out / "summary.json", {
            "status": STATUS_MASK_MISSING,
            "algorithm": ALGORITHM,
            "expected_mask_candidates": [str(root / p) for p in MASK_CANDIDATES],
            "next_action": "Place blinded expert RCA_outer_wall_expert_mask.npy in one of the expected locations and rerun.",
        })
        return {"summary": _read_json(out / "summary.json")}

    pack = np.load(_req(root / PACK_NPZ))
    manifest = pd.read_csv(_req(root / MANIFEST))
    locked = pd.read_csv(_req(root / LOCK_PROFILE))
    hu = np.asarray(pack["hu"], np.float32)
    lumen = np.asarray(pack["lumen_mask"], bool)
    lumen_radii = np.asarray(pack["lumen_radii_mm"], float)
    arcs = np.asarray(pack["arc_mm"], float)
    pixel_mm = float(np.asarray(pack["plane_pixel_mm"]).ravel()[0])

    expert = np.load(mask_path)
    if expert.shape != hu.shape:
        raise RuntimeError(f"Expert mask shape {expert.shape} != HU stack shape {hu.shape}")
    unique = np.unique(expert)
    if not np.all(np.isin(unique, [0, 1, False, True])):
        raise RuntimeError(f"Expert mask must be binary; values include {unique[:20]}")
    expert = expert.astype(bool)

    weights = _integration_weights(arcs)
    rows = []
    ray_rows = []

    for i, arc_mm in enumerate(arcs):
        annotated = bool(expert[i].any())
        containment = bool(np.all(expert[i][lumen[i]])) if annotated else False
        fixed_outer = _fixed_outer_mask(lumen_radii[i], pixel_mm, expert[i].shape, 1.0)
        fixed_wall = fixed_outer & ~lumen[i]

        if annotated:
            expert_wall = expert[i] & ~lumen[i]
            er = _mask_outer_radii(expert[i], pixel_mm)
            wall_thick = er - lumen_radii[i]
            finite = np.isfinite(wall_thick)
            expert_median = float(np.median(wall_thick[finite])) if finite.any() else np.nan
            expert_iqr = (
                float(np.percentile(wall_thick[finite], 75) - np.percentile(wall_thick[finite], 25))
                if finite.any() else np.nan
            )
            ev = _component_volumes(hu[i], expert_wall, pixel_mm, weights[i])
            for a in range(N_ANGLES):
                ray_rows.append({
                    "slice_index": i,
                    "arc_mm": float(arc_mm),
                    "angle_index": a,
                    "lumen_radius_mm": float(lumen_radii[i, a]),
                    "expert_outer_radius_mm": float(er[a]) if np.isfinite(er[a]) else np.nan,
                    "expert_wall_thickness_mm": float(wall_thick[a]) if np.isfinite(wall_thick[a]) else np.nan,
                    "fixed_shell_thickness_mm": 1.0,
                    "fixed_minus_expert_wall_thickness_mm": (
                        float(1.0 - wall_thick[a]) if np.isfinite(wall_thick[a]) else np.nan
                    ),
                })
        else:
            expert_wall = np.zeros_like(lumen[i], bool)
            expert_median = expert_iqr = np.nan
            ev = {k: np.nan for k in (
                "fatlike_excluded_mm3", "low_attenuation_mm3", "noncalcified_mm3",
                "mixed_intermediate_mm3", "calcified_mm3", "total_plaque_proxy_mm3",
                "wall_volume_mm3", "wall_mean_hu",
            )}

        fv = _component_volumes(hu[i], fixed_wall, pixel_mm, weights[i])
        rec = {
            "slice_index": i,
            "arc_mm": float(arc_mm),
            "annotated": annotated,
            "outer_contains_lumen": containment,
            "lumen_radius_median_mm": float(manifest.loc[i, "lumen_radius_median_mm"]),
            "center_hu": float(manifest.loc[i, "center_hu"]),
            "expert_median_wall_thickness_mm": expert_median,
            "expert_wall_thickness_iqr_mm": expert_iqr,
            "expert_wall_area_mm2": (
                float(expert_wall.sum() * pixel_mm * pixel_mm) if annotated else np.nan
            ),
            "fixed_shell_wall_area_mm2": float(fixed_wall.sum() * pixel_mm * pixel_mm),
            "fixed_outer_vs_expert_dice": _dice(fixed_outer, expert[i]) if annotated else np.nan,
            "fixed_outer_vs_expert_jaccard": _jaccard(fixed_outer, expert[i]) if annotated else np.nan,
        }
        rec.update(ev)
        for k, v in fv.items():
            rec[f"fixed_{k}"] = v
        rows.append(rec)

    df = pd.DataFrame(rows)
    rays = pd.DataFrame(ray_rows)
    df.to_csv(out / "RCA_expert_outer_wall_slice_metrics.csv", index=False)
    rays.to_csv(out / "RCA_expert_outer_wall_ray_metrics.csv", index=False)

    annotated = df.annotated.astype(bool)
    n_annotated = int(annotated.sum())
    containment_fraction = float(df.loc[annotated, "outer_contains_lumen"].mean()) if n_annotated else 0.0
    ref_annot = int(((df.arc_mm >= REFERENCE_ARC[0]) & (df.arc_mm <= REFERENCE_ARC[1]) & annotated).sum())
    hold_annot = int(((df.arc_mm >= HOLDOUT_ARC[0]) & (df.arc_mm < HOLDOUT_ARC[1]) & annotated).sum())

    mask_qc_pass = bool(
        n_annotated >= MIN_ANNOTATED_SLICES
        and containment_fraction >= MIN_CONTAINMENT_FRACTION
        and ref_annot >= MIN_REFERENCE_SLICES
        and hold_annot >= MIN_HOLDOUT_SLICES
    )

    if not mask_qc_pass:
        summary = {
            "status": STATUS_MASK_QC_FAILED,
            "algorithm": ALGORITHM,
            "baseline_commit": BASELINE,
            "mask_path": str(mask_path),
            "annotated_slices": n_annotated,
            "containment_fraction": containment_fraction,
            "annotated_reference_slices": ref_annot,
            "annotated_holdout_slices": hold_annot,
            "requirements": {
                "min_annotated_slices": MIN_ANNOTATED_SLICES,
                "min_containment_fraction": MIN_CONTAINMENT_FRACTION,
                "min_reference_slices": MIN_REFERENCE_SLICES,
                "min_holdout_slices": MIN_HOLDOUT_SLICES,
            },
            "is_validated_clinical_tpv": False,
        }
        _write_json(out / "summary.json", summary)
        _write_json(out / "run_state.json", {
            "status": "COMPLETE", "result_status": STATUS_MASK_QC_FAILED,
            "algorithm": ALGORITHM, "baseline": BASELINE,
        })
        return {"summary": summary}

    expert_scored, models = _expert_reference_normalize(df)
    expert_scored.to_csv(out / "RCA_expert_outer_wall_reference_normalized_excess.csv", index=False)
    models.to_csv(out / "RCA_expert_outer_wall_reference_models.csv", index=False)

    joined, agreement = _compare_locked(expert_scored, locked)
    joined.to_csv(out / "RCA_locked_vs_expert_outer_wall_profile_comparison.csv", index=False)

    g = df[annotated].copy()
    geometry = {
        "annotated_slices": n_annotated,
        "containment_fraction": containment_fraction,
        "annotated_reference_slices": ref_annot,
        "annotated_holdout_slices": hold_annot,
        "median_expert_wall_thickness_mm": float(np.nanmedian(g.expert_median_wall_thickness_mm)),
        "median_expert_wall_thickness_iqr_mm": float(np.nanmedian(g.expert_wall_thickness_iqr_mm)),
        "median_fixed_outer_vs_expert_dice": float(np.nanmedian(g.fixed_outer_vs_expert_dice)),
        "median_fixed_outer_vs_expert_jaccard": float(np.nanmedian(g.fixed_outer_vs_expert_jaccard)),
        "median_fixed_minus_expert_wall_area_mm2": float(np.nanmedian(
            g.fixed_shell_wall_area_mm2 - g.expert_wall_area_mm2
        )),
    }

    expert_totals = {
        c: float(expert_scored[c].sum()) for c in (
            "wall_volume_mm3", "fatlike_excluded_mm3",
            "low_attenuation_mm3", "noncalcified_mm3",
            "mixed_intermediate_mm3", "calcified_mm3",
            "total_plaque_proxy_mm3", "expert_excess_total_mm3",
        )
    }
    locked_total = float(lock_summary["nominal_plaque_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"])

    _plot_boundary_agreement(df, out / "01_RCA_expert_vs_fixed_boundary_agreement.png")
    _plot_profile_comparison(joined, out / "02_RCA_locked_vs_expert_excess_profile.png")
    _plot_area_bias(df, out / "03_RCA_expert_vs_fixed_wall_area.png")

    summary = {
        "status": STATUS_COMPLETE,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "mask_path": str(mask_path),
        "mask_qc_pass": True,
        "geometry_comparison": geometry,
        "expert_wall_totals_mm3": expert_totals,
        "locked_nominal_total_excess_mm3_full_rca": locked_total,
        "locked_vs_expert_holdout_agreement": agreement,
        "reference_zone_mm": list(REFERENCE_ARC),
        "holdout_zone_mm": list(HOLDOUT_ARC),
        "is_validated_clinical_tpv": False,
        "scientific_boundary": (
            "The expert mask supplies an independent blinded outer-vessel contour. Raw expert wall volume is anatomical wall volume on resampled "
            "orthogonal source planes, not plaque volume by itself. The expert-wall excess metric uses the same prespecified 20-50 mm normal-wall "
            "reference-normalization strategy to permit a like-for-like comparison with the locked fixed-shell research proxy. The locked method is "
            "not retuned in this experiment, and no clinical TPV claim is made."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "input_provenance.json", {
        "expert_pack": str(root / PACK_NPZ),
        "expert_mask": str(mask_path),
        "manifest": str(root / MANIFEST),
        "locked_profile": str(root / LOCK_PROFILE),
        "locked_summary": str(root / LOCK_SUMMARY),
        "master": str(root / MASTER),
    })

    report = out / "OPENPLAQUE_RCA_EXPERT_OUTER_WALL_VALIDATION_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Expert Outer-Wall Validation v1</h1>"
        f"<p><b>Status:</b> {STATUS_COMPLETE}</p>"
        f"<p>Annotated slices: {n_annotated}/51; lumen containment: {containment_fraction:.3f}.</p>"
        f"<p>Median expert wall thickness: {geometry['median_expert_wall_thickness_mm']:.3f} mm; "
        f"median fixed-shell outer-vessel Dice: {geometry['median_fixed_outer_vs_expert_dice']:.3f}.</p>"
        f"<p>Locked-vs-expert holdout Spearman: {agreement.get('spearman_locked_vs_expert_excess', float('nan')):.3f}; "
        f"locked/expert holdout total ratio: {agreement.get('locked_to_expert_total_ratio', float('nan')):.3f}.</p>"
        "<p><b>Boundary:</b> blinded outer-wall validation; no retuning of the locked RCA plaque method; not clinical TPV.</p>"
        "<h2>Geometry comparison</h2><pre>" + json.dumps(geometry, indent=2, default=str) + "</pre>"
        "<h2>Expert wall totals</h2><pre>" + json.dumps(expert_totals, indent=2, default=str) + "</pre>"
        "<h2>Locked vs expert holdout agreement</h2><pre>" + json.dumps(agreement, indent=2, default=str) + "</pre>"
        '<img src="01_RCA_expert_vs_fixed_boundary_agreement.png" style="max-width:100%">'
        '<img src="02_RCA_locked_vs_expert_excess_profile.png" style="max-width:100%">'
        '<img src="03_RCA_expert_vs_fixed_wall_area.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out / "run_state.json", {
        "status": "COMPLETE",
        "result_status": STATUS_COMPLETE,
        "algorithm": ALGORITHM,
        "baseline": BASELINE,
    })
    zpath = out / "OPENPLAQUE_RCA_EXPERT_OUTER_WALL_VALIDATION_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)

    return {"summary": summary, "report": str(report), "zip": str(zpath)}
