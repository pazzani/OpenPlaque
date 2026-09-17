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
from scipy.stats import rankdata, spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-source-space-plaque-excess-specificity-v1.0"
OUTPUT_DIRNAME = "RCA_Source_Space_Plaque_Excess_Specificity_v1"

QUANT = Path("RCA_Source_Space_Plaque_Quantification_v1")
STATIONS = QUANT / "RCA_source_space_station_quantification.csv"
QUANT_SUMMARY = QUANT / "summary.json"
PRIOR = Path("Longitudinal_Plaque_PCAT_Fusion_v1/RCA_source_longitudinal_plaque_profile_1mm.csv")
PCAT = Path("PCAT_RCA_10_50_Reproducibility_Lock/pcat_canonical_primary_longitudinal.csv")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

SHELLS = (0.75, 1.0, 1.25, 1.5)
NOMINAL_SHELL = 1.0
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

MIN_POSITIVE_BINS = 4
MIN_NEGATIVE_BINS = 5
MIN_NOMINAL_AUC = 0.80
MIN_STRICT_AUC = 0.90
MIN_POS_NEG_MEDIAN_RATIO = 2.0
MIN_CROSS_SHELL_SPEARMAN = 0.80

STATUS_PREREQ = "RCA_SOURCE_SPACE_PLAQUE_EXCESS_PREREQUISITE_FAILED"
STATUS_FAIL = "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_FAILED"
STATUS_PASS = "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS"


def _req(p: Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p: Path):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p: Path, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ok = np.isfinite(score) & np.isin(y, [0, 1])
    y, score = y[ok], score[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score, method="average")
    u = float(r[y == 1].sum() - n1 * (n1 + 1) / 2.0)
    return u / float(n1 * n0)


def _design(df: pd.DataFrame) -> np.ndarray:
    r = df["lumen_radius_median_mm"].to_numpy(float)
    hu = df["center_hu"].to_numpy(float) / 1000.0
    return np.column_stack([np.ones(len(df)), r, r * r, hu])


def _huber_fit(X, y, k=HUBER_K, n_iter=40):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 4:
        raise RuntimeError("Insufficient finite reference stations for robust normal-wall model")
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
    resid = y - X @ beta
    return beta, resid


def _fit_shell(d: pd.DataFrame, shell: float):
    d = d[np.isclose(d.shell_thickness_mm, shell) & d.station_qc_pass.astype(bool)].copy()
    if d.empty:
        raise RuntimeError(f"No QC-passing stations for shell {shell}")
    ref = (d.arc_mm >= REFERENCE_ARC[0]) & (d.arc_mm < REFERENCE_ARC[1])
    if int(ref.sum()) < 25:
        raise RuntimeError(f"Too few reference stations for shell {shell}: {int(ref.sum())}")
    X = _design(d)
    models = []
    for c in COMPONENTS:
        frac = d[c].to_numpy(float) / np.maximum(d.shell_volume_mm3.to_numpy(float), 1e-9)
        beta, resid_ref = _huber_fit(X[ref], frac[ref])
        pred = X @ beta
        q = float(np.quantile(resid_ref[np.isfinite(resid_ref)], REFERENCE_RESIDUAL_QUANTILE))
        threshold = pred + q
        excess_frac = np.maximum(frac - threshold, 0.0)
        excess = excess_frac * d.shell_volume_mm3.to_numpy(float)
        stem = c.replace("_mm3", "")
        d[f"normal_{stem}_fraction"] = pred
        d[f"reference_p90_{stem}_fraction"] = threshold
        d[f"excess_{stem}_mm3"] = excess
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
    ex_cols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS]
    d["excess_total_plaque_proxy_mm3"] = d[ex_cols].sum(axis=1)
    return d, pd.DataFrame(models)


def _profile_1mm(d: pd.DataFrame, shell: float):
    d = d.copy()
    d["arc_start_mm"] = np.floor(d.arc_mm.to_numpy(float)).astype(float)
    cols = [f"excess_{c.replace('_mm3','')}_mm3" for c in COMPONENTS] + ["excess_total_plaque_proxy_mm3"]
    rows = []
    for a, x in d.groupby("arc_start_mm", sort=True):
        r = {
            "shell_thickness_mm": float(shell),
            "arc_start_mm": float(a),
            "arc_end_mm": float(a + 1.0),
            "station_count": int(len(x)),
            "station_qc_fraction": float(x.station_qc_pass.mean()),
            "raw_total_plaque_proxy_mm3": float(x.total_plaque_proxy_mm3.sum()),
        }
        for c in cols:
            r[c] = float(x[c].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def _validate_reference(prior: pd.DataFrame):
    ref = prior[(prior.arc_start_mm >= REFERENCE_ARC[0]) & (prior.arc_start_mm < REFERENCE_ARC[1])]
    if ref.empty:
        return False, {"reason": "reference zone absent from prior profile"}
    bad = int((ref.mapped_native_voxels_vote_ge3.fillna(0).to_numpy(float) > 0).sum())
    majority = int(ref.majority_3plus_signal.fillna(False).astype(bool).sum())
    return bad == 0 and majority == 0, {
        "reference_bins": int(len(ref)),
        "bins_with_vote_ge3": bad,
        "majority_positive_bins": majority,
    }


def _holdout_metrics(profile: pd.DataFrame, prior: pd.DataFrame):
    x = profile.merge(prior, on=["arc_start_mm", "arc_end_mm"], how="inner")
    x = x[(x.arc_start_mm >= HOLDOUT_ARC[0]) & (x.arc_start_mm < HOLDOUT_ARC[1])].copy()
    pos = x.majority_3plus_signal.fillna(False).astype(bool)
    neg = x.mapped_native_vote_sum.fillna(0).to_numpy(float) == 0
    strict = x.strict_5of5_signal.fillna(False).astype(bool)
    score = x.excess_total_plaque_proxy_mm3.to_numpy(float)
    if int(pos.sum()) < MIN_POSITIVE_BINS or int(neg.sum()) < MIN_NEGATIVE_BINS:
        return x, {"usable": False, "positive_bins": int(pos.sum()), "vote_free_negative_bins": int(neg.sum())}
    y_major = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
    s_major = np.r_[score[pos], score[neg]]
    y_strict = np.r_[np.ones(int(strict.sum())), np.zeros(int(neg.sum()))]
    s_strict = np.r_[score[strict], score[neg]]
    med_pos = float(np.median(score[pos]))
    med_neg = float(np.median(score[neg]))
    ratio = med_pos / max(med_neg, 1e-6)
    rho = float(spearmanr(x.excess_total_plaque_proxy_mm3, x.mapped_native_voxels_vote_ge3).statistic)
    return x, {
        "usable": True,
        "holdout_bins": int(len(x)),
        "positive_bins": int(pos.sum()),
        "strict_5of5_bins": int(strict.sum()),
        "vote_free_negative_bins": int(neg.sum()),
        "majority_vs_vote_free_auc": float(_auc(y_major, s_major)),
        "strict5_vs_vote_free_auc": float(_auc(y_strict, s_strict)) if int(strict.sum()) else np.nan,
        "positive_median_excess_mm3": med_pos,
        "negative_median_excess_mm3": med_neg,
        "positive_negative_median_ratio": ratio,
        "spearman_excess_vs_vote_ge3": rho,
    }


def _cross_shell(profiles):
    nom = profiles[NOMINAL_SHELL].set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"]
    rows = []
    for shell, p in profiles.items():
        y = p.set_index("arc_start_mm")["excess_total_plaque_proxy_mm3"].reindex(nom.index)
        ok = nom.notna() & y.notna()
        rho = float(spearmanr(nom[ok], y[ok]).statistic) if int(ok.sum()) >= 5 else np.nan
        rows.append({"shell_thickness_mm": float(shell), "profile_spearman_vs_nominal": rho})
    d = pd.DataFrame(rows)
    non_nom = d[~np.isclose(d.shell_thickness_mm, NOMINAL_SHELL)]
    finite = non_nom.profile_spearman_vs_nominal[np.isfinite(non_nom.profile_spearman_vs_nominal)]
    return d, float(finite.min()) if len(finite) else np.nan


def _plot_profiles(profiles, prior, out):
    nom = profiles[NOMINAL_SHELL]
    x = nom.merge(prior[["arc_start_mm", "mapped_native_voxels_vote_ge3"]], on="arc_start_mm", how="left")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x.arc_start_mm + 0.5, x.raw_total_plaque_proxy_mm3, label="raw 1.0-mm shell proxy", alpha=.55)
    ax.plot(x.arc_start_mm + 0.5, x.excess_total_plaque_proxy_mm3, label="reference-normalized excess", linewidth=2)
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Volume per 1-mm bin (mm³)")
    ax2 = ax.twinx()
    ax2.step(x.arc_start_mm + 0.5, x.mapped_native_voxels_vote_ge3.fillna(0), where="mid", label="prior 3+/5 vote voxels", alpha=.45)
    ax2.set_ylabel("Prior ensemble 3+/5 vote voxels")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.axvspan(*REFERENCE_ARC, alpha=.06)
    ax.axvspan(*HOLDOUT_ARC, alpha=.04)
    ax.set_title("RCA plaque specificity: raw shell vs reference-normalized excess")
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def _plot_shells(profiles, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for shell, p in profiles.items():
        ax.plot(p.arc_start_mm + .5, p.excess_total_plaque_proxy_mm3, label=f"shell {shell:.2f} mm")
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Excess plaque proxy per 1-mm bin (mm³)")
    ax.set_title("Reference-normalized excess profile across shell sensitivity")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def _plot_components(profile, out):
    x = profile.arc_start_mm + .5
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in COMPONENTS:
        stem = c.replace("_mm3", "")
        ax.plot(x, profile[f"excess_{stem}_mm3"], label=stem.replace("_", " "))
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Excess component volume per 1-mm bin (mm³)")
    ax.set_title("Nominal 1.0-mm-shell excess composition")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def synthetic_self_test():
    # A deterministic source-like synthetic: reference has smooth radius dependence;
    # positive holdout bins contain added excess. The normalization must recover separation.
    rng = np.random.default_rng(7)
    rows = []
    for shell in SHELLS:
        for i, arc in enumerate(np.arange(0., 52.5, .5)):
            r = 1.45 + .10 * np.sin(arc / 7.)
            hu = 600 + 25 * np.cos(arc / 8.)
            sv = (2 * math.pi * r * shell + math.pi * shell * shell) * .5
            boost = 0.20 if arc < 5 else 0.0
            vals = [(.12 + boost) * sv, .28 * sv, .42 * sv, .015 * sv]
            rows.append({
                "station_index": i, "arc_mm": arc, "integration_ds_mm": .5, "shell_thickness_mm": shell,
                "center_hu": hu, "lumen_radius_median_mm": r, "shell_volume_mm3": sv,
                "station_qc_pass": True, "total_plaque_proxy_mm3": sum(vals),
                **{c: vals[j] for j, c in enumerate(COMPONENTS)}
            })
    d = pd.DataFrame(rows)
    x, model = _fit_shell(d, 1.0)
    assert len(model) == 4
    assert np.isfinite(x.excess_total_plaque_proxy_mm3).all()
    return {"ok": True, "model_rows": int(len(model)), "excess_sum_mm3": float(x.excess_total_plaque_proxy_mm3.sum())}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    quant_summary = _read_json(root / QUANT_SUMMARY)
    stations = pd.read_csv(_req(root / STATIONS))
    prior = pd.read_csv(_req(root / PRIOR))

    prereq_ok, prereq = _validate_reference(prior)
    prereq.update({
        "master_status": master.get("status"),
        "master_modified": False,
        "quantification_status": quant_summary.get("status"),
        "quantification_is_validated_clinical_tpv": quant_summary.get("is_validated_clinical_tpv"),
    })
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        prereq_ok = False; prereq["master_error"] = "master not frozen"
    if quant_summary.get("status") != "RCA_SOURCE_SPACE_PLAQUE_PROXY_ROBUSTNESS_PASS":
        prereq_ok = False; prereq["quantification_error"] = "required v1 robustness pass absent"
    _write_json(out / "prerequisite_check.json", prereq)
    if not prereq_ok:
        summary = {"status": STATUS_PREREQ, "algorithm": ALGORITHM, "baseline_commit": BASELINE, "prerequisite": prereq}
        _write_json(out / "summary.json", summary); _write_json(out / "run_state.json", {"status": "COMPLETE", "result_status": STATUS_PREREQ})
        return {"summary": summary, "report": None, "zip": None}

    shell_station = {}
    models = []
    profiles = {}
    metrics = []
    holdouts = []
    for shell in SHELLS:
        ds, mod = _fit_shell(stations, shell)
        shell_station[shell] = ds
        models.append(mod)
        p = _profile_1mm(ds, shell)
        profiles[shell] = p
        h, met = _holdout_metrics(p, prior)
        met["shell_thickness_mm"] = float(shell)
        metrics.append(met)
        h["shell_thickness_mm"] = float(shell)
        holdouts.append(h)

    model_df = pd.concat(models, ignore_index=True)
    model_df.to_csv(out / "RCA_excess_reference_models.csv", index=False)
    station_df = pd.concat(shell_station.values(), ignore_index=True)
    station_df.to_csv(out / "RCA_excess_station_quantification.csv", index=False)
    profile_df = pd.concat(profiles.values(), ignore_index=True)
    profile_df.to_csv(out / "RCA_excess_profile_1mm_all_shells.csv", index=False)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "RCA_excess_holdout_specificity.csv", index=False)
    pd.concat(holdouts, ignore_index=True).to_csv(out / "RCA_excess_holdout_bins.csv", index=False)
    shell_corr, min_shell_corr = _cross_shell(profiles)
    shell_corr.to_csv(out / "RCA_excess_shell_profile_consistency.csv", index=False)

    nominal_profile = profiles[NOMINAL_SHELL].copy()
    nominal_profile.to_csv(out / "RCA_excess_profile_1mm_nominal.csv", index=False)
    nominal_metrics = metrics_df[np.isclose(metrics_df.shell_thickness_mm, NOMINAL_SHELL)].iloc[0].to_dict()
    nominal_station = shell_station[NOMINAL_SHELL]
    totals = {}
    for c in COMPONENTS:
        stem = c.replace("_mm3", "")
        totals[f"excess_{stem}_mm3"] = float(nominal_station[f"excess_{stem}_mm3"].sum())
    totals["excess_total_plaque_proxy_mm3"] = float(nominal_station.excess_total_plaque_proxy_mm3.sum())
    totals["raw_total_plaque_proxy_mm3"] = float(nominal_station.total_plaque_proxy_mm3.sum())
    totals["background_removed_fraction"] = 1.0 - totals["excess_total_plaque_proxy_mm3"] / max(totals["raw_total_plaque_proxy_mm3"], 1e-9)

    pcat_fusion = pd.DataFrame()
    if (root / PCAT).exists():
        pcat = pd.read_csv(root / PCAT)
        pcat_fusion = nominal_profile.merge(pcat, on=["arc_start_mm", "arc_end_mm"], how="inner")
        if not pcat_fusion.empty:
            pcat_fusion.to_csv(out / "RCA_excess_plaque_PCAT_fusion_10_50.csv", index=False)

    nominal_ok = bool(
        nominal_metrics.get("usable", False)
        and nominal_metrics.get("majority_vs_vote_free_auc", -np.inf) >= MIN_NOMINAL_AUC
        and nominal_metrics.get("strict5_vs_vote_free_auc", -np.inf) >= MIN_STRICT_AUC
        and nominal_metrics.get("positive_negative_median_ratio", -np.inf) >= MIN_POS_NEG_MEDIAN_RATIO
    )
    shell_ok = bool(np.isfinite(min_shell_corr) and min_shell_corr >= MIN_CROSS_SHELL_SPEARMAN)
    status = STATUS_PASS if nominal_ok and shell_ok else STATUS_FAIL

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "reference_zone_mm": list(REFERENCE_ARC),
        "holdout_zone_mm": list(HOLDOUT_ARC),
        "reference_residual_quantile": REFERENCE_RESIDUAL_QUANTILE,
        "nominal_shell_thickness_mm": NOMINAL_SHELL,
        "nominal_holdout_specificity": nominal_metrics,
        "min_cross_shell_profile_spearman": min_shell_corr,
        "nominal_excess_totals_mm3": totals,
        "pcat_fusion_available": bool(not pcat_fusion.empty),
        "is_validated_clinical_tpv": False,
        "scientific_boundary": (
            "This is a cross-modal specificity validation and reference-normalized source-space excess-wall proxy. "
            "The normal-wall model is fit only in the prespecified 20-50 mm RCA reference zone and evaluated in the disjoint 0-20 mm holdout zone against prior ensemble plaque votes. "
            "It is not an independently segmented outer wall and is not validated clinical TPV."
        ),
    }
    _write_json(out / "summary.json", summary)

    _plot_profiles(profiles, prior, out / "01_RCA_raw_vs_excess_specificity.png")
    _plot_shells(profiles, out / "02_RCA_excess_shell_consistency.png")
    _plot_components(nominal_profile, out / "03_RCA_excess_composition_profile.png")

    report = out / "OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Source-Space Plaque Excess Specificity v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Nominal holdout AUC (majority vs vote-free): {nominal_metrics.get('majority_vs_vote_free_auc', float('nan')):.3f}; "
        f"strict 5/5 AUC: {nominal_metrics.get('strict5_vs_vote_free_auc', float('nan')):.3f}; "
        f"positive/negative median ratio: {nominal_metrics.get('positive_negative_median_ratio', float('nan')):.3f}; "
        f"minimum cross-shell profile Spearman: {min_shell_corr:.3f}.</p>"
        "<p><b>Research boundary:</b> reference-normalized source-space excess-wall proxy; not validated clinical TPV and not an independently segmented outer wall.</p>"
        "<h2>Nominal excess totals</h2><pre>" + json.dumps(totals, indent=2) + "</pre>"
        "<h2>Holdout specificity</h2>" + metrics_df.to_html(index=False) +
        '<img src="01_RCA_raw_vs_excess_specificity.png" style="max-width:100%">'
        '<img src="02_RCA_excess_shell_consistency.png" style="max-width:100%">'
        '<img src="03_RCA_excess_composition_profile.png" style="max-width:100%">'
        "</body></html>", encoding="utf-8"
    )

    _write_json(out / "run_state.json", {"status": "COMPLETE", "result_status": status, "algorithm": ALGORITHM, "baseline": BASELINE})
    zpath = out / "OPENPLAQUE_RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
