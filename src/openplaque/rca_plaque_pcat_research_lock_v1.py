from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "rca-plaque-pcat-research-lock-v1.0"
OUTPUT_DIRNAME = "RCA_Plaque_PCAT_Research_Lock_v1"

FIXED = Path("RCA_Source_Space_Plaque_Excess_Specificity_v1")
ADAPTIVE = Path("RCA_Adaptive_Outer_Wall_Plaque_v1")
GLOBAL = Path("RCA_Global_Outer_Wall_Surface_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

FIXED_SUMMARY = FIXED / "summary.json"
FIXED_PROFILE = FIXED / "RCA_excess_profile_1mm_nominal.csv"
FIXED_ALL_SHELLS = FIXED / "RCA_excess_profile_1mm_all_shells.csv"
FIXED_SHELL_CONSISTENCY = FIXED / "RCA_excess_shell_profile_consistency.csv"
FIXED_PCAT_FUSION = FIXED / "RCA_excess_plaque_PCAT_fusion_10_50.csv"
ADAPTIVE_SUMMARY = ADAPTIVE / "summary.json"
GLOBAL_SUMMARY = GLOBAL / "summary.json"

EXPECTED_FIXED_STATUS = "RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS"
EXPECTED_ADAPTIVE_STATUS = "RCA_ADAPTIVE_OUTER_WALL_SPECIFICITY_FAILED"
EXPECTED_GLOBAL_STATUS = "RCA_GLOBAL_OUTER_WALL_SURFACE_SPECIFICITY_FAILED"
STATUS = "RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _shell_summary(all_shells):
    comp = [
        "excess_low_attenuation_mm3",
        "excess_noncalcified_mm3",
        "excess_mixed_intermediate_mm3",
        "excess_calcified_mm3",
        "excess_total_plaque_proxy_mm3",
    ]
    rows = []
    for shell, g in all_shells.groupby("shell_thickness_mm", sort=True):
        row = {"shell_thickness_mm": float(shell)}
        for c in comp:
            row[c] = float(g[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _component_percentages(totals):
    total = float(totals["excess_total_plaque_proxy_mm3"])
    out = {}
    for k in (
        "excess_low_attenuation_mm3",
        "excess_noncalcified_mm3",
        "excess_mixed_intermediate_mm3",
        "excess_calcified_mm3",
    ):
        out[k.replace("_mm3", "_percent")] = 100.0 * float(totals[k]) / max(total, 1e-12)
    return out


def _pcat_summary(fusion):
    d = fusion[np.isfinite(fusion.mean_hu) & (fusion.fat_voxels > 0)].copy()
    if d.empty:
        return {"available": False}
    w = d.fat_voxels.to_numpy(float)
    hu = d.mean_hu.to_numpy(float)
    weighted = float(np.average(hu, weights=w))
    return {
        "available": True,
        "arc_start_mm": float(d.arc_start_mm.min()),
        "arc_end_mm": float(d.arc_end_mm.max()),
        "bin_count": int(len(d)),
        "fat_voxels_total": int(d.fat_voxels.sum()),
        "fat_voxel_weighted_mean_hu": weighted,
        "unweighted_mean_bin_hu": float(np.mean(hu)),
        "median_bin_hu": float(np.median(hu)),
        "minimum_bin_hu": float(np.min(hu)),
        "maximum_bin_hu": float(np.max(hu)),
        "wall_margin_mm": float(d.wall_margin_mm.iloc[0]) if "wall_margin_mm" in d else np.nan,
    }


def _hotspots(profile, fusion, n=5):
    p = profile.copy().sort_values("excess_total_plaque_proxy_mm3", ascending=False).head(n)
    plaque = p[[
        "arc_start_mm", "arc_end_mm", "excess_total_plaque_proxy_mm3",
        "excess_low_attenuation_mm3", "excess_noncalcified_mm3",
        "excess_mixed_intermediate_mm3", "excess_calcified_mm3",
    ]].to_dict("records")
    q = fusion[np.isfinite(fusion.mean_hu)].copy().sort_values("mean_hu", ascending=False).head(n)
    pcat = q[["arc_start_mm", "arc_end_mm", "mean_hu", "fat_voxels"]].to_dict("records")
    return plaque, pcat


def _fusion_relationship(fusion):
    d = fusion[
        np.isfinite(fusion.excess_total_plaque_proxy_mm3)
        & np.isfinite(fusion.mean_hu)
    ].copy()
    if len(d) < 5:
        return {"usable": False, "bin_count": int(len(d))}
    rho = float(spearmanr(d.excess_total_plaque_proxy_mm3, d.mean_hu).statistic)
    return {
        "usable": True,
        "bin_count": int(len(d)),
        "spearman_plaque_excess_vs_pcat_mean_hu": rho,
        "interpretation_boundary": (
            "Descriptive within-scan association only. More positive PCAT HU means less negative attenuation, "
            "but no causal or clinical inflammation inference is made from this correlation."
        ),
    }


def _method_comparison(fixed, adaptive, global_):
    rows = [
        {
            "method": "fixed_1mm_reference_normalized",
            "status": fixed.get("status"),
            "majority_auc": fixed["nominal_holdout_specificity"]["majority_vs_vote_free_auc"],
            "strict_auc": fixed["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"],
            "positive_negative_ratio": fixed["nominal_holdout_specificity"]["positive_negative_median_ratio"],
            "stability": fixed["min_cross_shell_profile_spearman"],
            "total_excess_mm3": fixed["nominal_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"],
        },
        {
            "method": "independent_ray_adaptive_outer_wall",
            "status": adaptive.get("status"),
            "majority_auc": adaptive["nominal_holdout_specificity"]["majority_vs_vote_free_auc"],
            "strict_auc": adaptive["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"],
            "positive_negative_ratio": adaptive["nominal_holdout_specificity"]["positive_negative_median_ratio"],
            "stability": adaptive["min_cross_variant_profile_spearman"],
            "total_excess_mm3": adaptive["nominal_adaptive_wall_totals_mm3"]["excess_total_plaque_proxy_mm3"],
        },
        {
            "method": "global_outer_wall_surface",
            "status": global_.get("status"),
            "majority_auc": global_["nominal_holdout_specificity"]["majority_vs_vote_free_auc"],
            "strict_auc": global_["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"],
            "positive_negative_ratio": global_["nominal_holdout_specificity"]["positive_negative_median_ratio"],
            "stability": global_["min_cross_variant_profile_spearman"],
            "total_excess_mm3": global_["nominal_global_surface_totals_mm3"]["excess_total_plaque_proxy_mm3"],
        },
    ]
    return pd.DataFrame(rows)


def _plot_plaque_profile(profile, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = profile.arc_start_mm + .5
    ax.plot(x, profile.excess_total_plaque_proxy_mm3, linewidth=2, label="total excess plaque proxy")
    ax.plot(x, profile.excess_low_attenuation_mm3, label="low attenuation")
    ax.plot(x, profile.excess_noncalcified_mm3, label="noncalcified")
    ax.plot(x, profile.excess_mixed_intermediate_mm3, label="mixed/intermediate")
    ax.plot(x, profile.excess_calcified_mm3, label="calcified")
    ax.set_xlabel("RCA source-centerline arc (mm)")
    ax.set_ylabel("Reference-normalized excess per 1-mm bin (mm³)")
    ax.set_title("Locked RCA research plaque-excess profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_shell_uncertainty(shells, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(shells.shell_thickness_mm, shells.excess_total_plaque_proxy_mm3, marker="o")
    ax.set_xlabel("Fixed outer-shell thickness (mm)")
    ax.set_ylabel("Total excess plaque proxy (mm³)")
    ax.set_title("Locked RCA plaque proxy: shell-sensitivity envelope")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_pcat_fusion(fusion, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = fusion.arc_start_mm + .5
    ax.plot(x, fusion.excess_total_plaque_proxy_mm3, linewidth=2, label="plaque excess")
    ax.set_xlabel("RCA arc (mm)")
    ax.set_ylabel("Plaque excess per 1-mm bin (mm³)")
    ax2 = ax.twinx()
    ax2.plot(x, fusion.mean_hu, linewidth=1.6, label="PCAT mean HU")
    ax2.set_ylabel("PCAT attenuation (HU)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax.set_title("Locked RCA plaque + PCAT research profile")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_method_comparison(methods, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(methods))
    w = .32
    ax.bar(x - w/2, methods.majority_auc, width=w, label="majority AUROC")
    ax.bar(x + w/2, methods.strict_auc, width=w, label="strict 5/5 AUROC")
    ax.axhline(.80, linestyle="--", linewidth=1)
    ax.axhline(.90, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["fixed shell", "adaptive rays", "global surface"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("RCA plaque-method development comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    d = pd.DataFrame({
        "shell_thickness_mm": [1.0, 1.0, 1.25, 1.25],
        "excess_low_attenuation_mm3": [1., 2., 1.5, 2.5],
        "excess_noncalcified_mm3": [2., 3., 2.5, 3.5],
        "excess_mixed_intermediate_mm3": [0., 1., 0., 1.],
        "excess_calcified_mm3": [1., 1., 1., 1.],
        "excess_total_plaque_proxy_mm3": [4., 7., 5., 8.],
    })
    s = _shell_summary(d)
    assert len(s) == 2
    assert float(s.loc[np.isclose(s.shell_thickness_mm, 1.0), "excess_total_plaque_proxy_mm3"].iloc[0]) == 11.0
    pct = _component_percentages({
        "excess_low_attenuation_mm3": 1.,
        "excess_noncalcified_mm3": 2.,
        "excess_mixed_intermediate_mm3": 1.,
        "excess_calcified_mm3": 0.,
        "excess_total_plaque_proxy_mm3": 4.,
    })
    assert abs(sum(pct.values()) - 100.0) < 1e-9
    return {"ok": True, "shell_rows": len(s)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline": BASELINE})

    master = _read_json(root / MASTER)
    fixed = _read_json(root / FIXED_SUMMARY)
    adaptive = _read_json(root / ADAPTIVE_SUMMARY)
    global_ = _read_json(root / GLOBAL_SUMMARY)

    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Frozen master prerequisite failed")
    if fixed.get("status") != EXPECTED_FIXED_STATUS:
        raise RuntimeError(f"Fixed-shell benchmark prerequisite failed: {fixed.get('status')}")
    if adaptive.get("status") != EXPECTED_ADAPTIVE_STATUS:
        raise RuntimeError(f"Adaptive-wall expected failed status not found: {adaptive.get('status')}")
    if global_.get("status") != EXPECTED_GLOBAL_STATUS:
        raise RuntimeError(f"Global-surface expected failed status not found: {global_.get('status')}")

    profile = pd.read_csv(_req(root / FIXED_PROFILE))
    all_shells = pd.read_csv(_req(root / FIXED_ALL_SHELLS))
    shell_consistency = pd.read_csv(_req(root / FIXED_SHELL_CONSISTENCY))
    fusion = pd.read_csv(_req(root / FIXED_PCAT_FUSION))

    shell_totals = _shell_summary(all_shells)
    shell_totals.to_csv(out / "RCA_locked_plaque_shell_uncertainty.csv", index=False)

    methods = _method_comparison(fixed, adaptive, global_)
    methods.to_csv(out / "RCA_plaque_method_development_comparison.csv", index=False)

    locked = profile.copy()
    locked["research_status"] = STATUS
    locked.to_csv(out / "RCA_locked_research_plaque_profile_1mm.csv", index=False)

    locked_fusion = fusion.copy()
    locked_fusion["research_status"] = STATUS
    locked_fusion.to_csv(out / "RCA_locked_research_plaque_PCAT_profile_10_50.csv", index=False)

    totals = dict(fixed["nominal_excess_totals_mm3"])
    percentages = _component_percentages(totals)
    pcat = _pcat_summary(fusion)
    plaque_hotspots, pcat_hotspots = _hotspots(profile, fusion)
    assoc = _fusion_relationship(fusion)

    nominal_shell = float(fixed["nominal_shell_thickness_mm"])
    nominal_row = shell_totals[np.isclose(shell_totals.shell_thickness_mm, nominal_shell)]
    if len(nominal_row) != 1:
        raise RuntimeError("Nominal 1.0-mm shell total not uniquely recoverable")
    sensitivity = {
        "shells_mm": [float(x) for x in shell_totals.shell_thickness_mm],
        "total_excess_mm3_by_shell": [float(x) for x in shell_totals.excess_total_plaque_proxy_mm3],
        "minimum_total_excess_mm3": float(shell_totals.excess_total_plaque_proxy_mm3.min()),
        "maximum_total_excess_mm3": float(shell_totals.excess_total_plaque_proxy_mm3.max()),
        "nominal_total_excess_mm3": float(nominal_row.excess_total_plaque_proxy_mm3.iloc[0]),
        "minimum_cross_shell_profile_spearman": float(
            shell_consistency[~np.isclose(shell_consistency.shell_thickness_mm, nominal_shell)]
            .profile_spearman_vs_nominal.min()
        ),
    }

    benchmark = {
        "majority_vs_vote_free_auc": float(fixed["nominal_holdout_specificity"]["majority_vs_vote_free_auc"]),
        "strict5_vs_vote_free_auc": float(fixed["nominal_holdout_specificity"]["strict5_vs_vote_free_auc"]),
        "positive_negative_median_ratio": float(fixed["nominal_holdout_specificity"]["positive_negative_median_ratio"]),
        "holdout_bins": int(fixed["nominal_holdout_specificity"]["holdout_bins"]),
        "reference_zone_mm": list(fixed["reference_zone_mm"]),
        "holdout_zone_mm": list(fixed["holdout_zone_mm"]),
    }

    _plot_plaque_profile(profile, out / "01_RCA_locked_research_plaque_profile.png")
    _plot_shell_uncertainty(shell_totals, out / "02_RCA_locked_shell_uncertainty.png")
    _plot_pcat_fusion(fusion, out / "03_RCA_locked_plaque_PCAT_profile.png")
    _plot_method_comparison(methods, out / "04_RCA_plaque_method_comparison.png")

    summary = {
        "status": STATUS,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "locked_plaque_method": "RCA source-space 1.0-mm fixed shell with RCA 20-50 mm reference-normalized excess model",
        "locked_plaque_source_status": fixed.get("status"),
        "retired_developmental_outer_wall_methods": {
            "independent_ray_adaptive_wall": adaptive.get("status"),
            "global_outer_wall_surface": global_.get("status"),
        },
        "plaque_validation_benchmark": benchmark,
        "nominal_plaque_excess_totals_mm3": totals,
        "nominal_plaque_composition_percent": percentages,
        "shell_sensitivity": sensitivity,
        "pcat": pcat,
        "plaque_pcat_descriptive_relationship": assoc,
        "top_plaque_excess_bins": plaque_hotspots,
        "least_negative_pcat_bins": pcat_hotspots,
        "pcat_fusion_available": True,
        "is_validated_clinical_tpv": False,
        "is_proprietary_fai": False,
        "research_boundary": (
            "This lock selects the best-performing developmental RCA plaque proxy after two more anatomical outer-wall models failed to improve it. "
            "The locked plaque quantity is a reference-normalized source-space excess-wall proxy, not independently segmented clinical TPV. "
            "PCAT is direct attenuation in the locked RCA 10-50 mm region using the established -190 to -30 HU fat window and is not proprietary FAI. "
            "Future method improvement should use independent expert wall labels or an independently trained wall-segmentation model rather than additional tuning on this scan."
        ),
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "input_provenance.json", {
        "fixed_shell_summary": str(root / FIXED_SUMMARY),
        "fixed_shell_profile": str(root / FIXED_PROFILE),
        "fixed_shell_all_shells": str(root / FIXED_ALL_SHELLS),
        "fixed_shell_pcat_fusion": str(root / FIXED_PCAT_FUSION),
        "adaptive_outer_wall_summary": str(root / ADAPTIVE_SUMMARY),
        "global_outer_wall_summary": str(root / GLOBAL_SUMMARY),
        "master": str(root / MASTER),
    })

    report = out / "OPENPLAQUE_RCA_PLAQUE_PCAT_RESEARCH_LOCK_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque RCA Plaque + PCAT Research Lock v1</h1>"
        f"<p><b>Status:</b> {STATUS}</p>"
        f"<p>Locked plaque proxy: {totals['excess_total_plaque_proxy_mm3']:.3f} mm³. "
        f"Majority AUROC: {benchmark['majority_vs_vote_free_auc']:.3f}; strict AUROC: {benchmark['strict5_vs_vote_free_auc']:.3f}; "
        f"positive/negative ratio: {benchmark['positive_negative_median_ratio']:.3f}.</p>"
        f"<p>PCAT 10-50 mm fat-voxel-weighted mean: {pcat.get('fat_voxel_weighted_mean_hu', float('nan')):.2f} HU.</p>"
        "<p><b>Research boundary:</b> plaque is a reference-normalized excess-wall proxy, not clinical TPV. "
        "PCAT is direct attenuation, not proprietary FAI. Further outer-wall development on this same scan is retired.</p>"
        "<h2>Plaque totals</h2><pre>" + json.dumps(totals, indent=2, default=str) + "</pre>"
        "<h2>Composition</h2><pre>" + json.dumps(percentages, indent=2, default=str) + "</pre>"
        "<h2>Shell sensitivity</h2><pre>" + json.dumps(sensitivity, indent=2, default=str) + "</pre>"
        "<h2>PCAT</h2><pre>" + json.dumps(pcat, indent=2, default=str) + "</pre>"
        "<h2>Method comparison</h2>" + methods.to_html(index=False) +
        '<img src="01_RCA_locked_research_plaque_profile.png" style="max-width:100%">'
        '<img src="02_RCA_locked_shell_uncertainty.png" style="max-width:100%">'
        '<img src="03_RCA_locked_plaque_PCAT_profile.png" style="max-width:100%">'
        '<img src="04_RCA_plaque_method_comparison.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out / "run_state.json", {
        "status": "COMPLETE",
        "result_status": STATUS,
        "algorithm": ALGORITHM,
        "baseline": BASELINE,
    })

    zpath = out / "OPENPLAQUE_RCA_PLAQUE_PCAT_RESEARCH_LOCK_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)

    return {"summary": summary, "report": str(report), "zip": str(zpath)}
