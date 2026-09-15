from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
OUTPUT_DIRNAME = "Final_Integrated_Research_Report_v1"


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _find_dir(root: Path, dirname: str) -> Path:
    root = Path(root)
    direct = root / dirname
    if direct.is_dir():
        return direct
    hits = [p for p in root.rglob(dirname) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"Could not find {dirname} under {root}")
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _require_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    hits = [p for p in root.rglob(name) if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    hits.sort(key=lambda p: (len(str(p)), str(p)))
    return hits[0]


def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def _require_complete(path: Path, keys=("status", "state")):
    obj = _read_json(path)
    values = [str(obj.get(k, "")).upper() for k in keys if k in obj]
    if not values or not any(v == "COMPLETE" or v.startswith("COMPLETE_") for v in values):
        raise RuntimeError(f"{path} is not complete: {obj}")
    return obj


def _contiguous_intervals(df: pd.DataFrame, signal_col: str):
    sig = df[signal_col].astype(bool).to_numpy()
    rows = []
    start = None
    for i, on in enumerate(np.r_[sig, False]):
        if on and start is None:
            start = i
        elif (not on) and start is not None:
            block = df.iloc[start:i]
            rows.append({
                "arc_start_mm": float(block["arc_start_mm"].iloc[0]),
                "arc_end_mm": float(block["arc_end_mm"].iloc[-1]),
                "duration_mm": float(block["arc_end_mm"].iloc[-1] - block["arc_start_mm"].iloc[0]),
            })
            start = None
    return rows


def _best_interval(intervals: pd.DataFrame, level: str):
    q = intervals[intervals["confidence_level"].eq(level)].copy()
    if q.empty:
        return None
    q = q.sort_values(["mapped_native_voxels_vote_ge3", "duration_mm"], ascending=False)
    r = q.iloc[0]
    return {
        "arc_start_mm": float(r["arc_start_mm"]),
        "arc_end_mm": float(r["arc_end_mm"]),
        "duration_mm": float(r["duration_mm"]),
        "mapped_native_voxels_vote_ge3": float(r["mapped_native_voxels_vote_ge3"]),
        "mapped_native_voxels_vote_ge4": float(r["mapped_native_voxels_vote_ge4"]),
        "mapped_native_voxels_vote_5": float(r["mapped_native_voxels_vote_5"]),
    }


def _format_interval_list(rows):
    if not rows:
        return "none"
    return ", ".join(f"{r['arc_start_mm']:.0f}–{r['arc_end_mm']:.0f} mm" for r in rows)


def _plot_anatomy(out: Path, anatomy: pd.DataFrame):
    q = anatomy.copy()
    q["label"] = q["vessel"].replace({"SECONDARY_TRAJECTORY": "Secondary"})
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    x = np.arange(len(q))
    ax.scatter(x, q["current_median_distance_mm"], s=70, label="current")
    ax.scatter(x, q["legacy_median_distance_mm"], s=35, marker="x", label="legacy")
    ax.set_xticks(x, q["label"])
    ax.set_ylabel("Median centerline-to-mask distance (mm)")
    ax.set_title("Canonical coronary anatomy — independent segmentation agreement")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    p = out / "01_canonical_anatomy_validation.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def _plot_plaque_confidence(out: Path, plaque: pd.DataFrame):
    q = plaque.set_index("vessel").loc[["RCA", "LAD"]]
    labels = ["Majority ≥3/5", "High ≥4/5", "Strict 5/5"]
    cols = ["majority_3plus_mm3", "high_4plus_mm3", "strict_5of5_mm3"]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width/2, [q.loc["RCA", c] for c in cols], width, label="RCA")
    ax.bar(x + width/2, [q.loc["LAD", c] for c in cols], width, label="LAD")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Native curved-series vote volume (mm³)")
    ax.set_title("5-fold plaque confidence atlas")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    p = out / "02_plaque_confidence_summary.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def _plot_rca_fusion(out: Path, fusion: pd.DataFrame):
    x = fusion["arc_start_mm"].to_numpy(float) + 0.5
    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax1.plot(x, fusion["mapped_native_voxels_vote_ge3"], label="Plaque ≥3/5 support")
    ax1.plot(x, fusion["mapped_native_voxels_vote_ge4"], label="Plaque ≥4/5 support")
    ax1.set_xlabel("Canonical RCA arc length (mm)")
    ax1.set_ylabel("Mapped native plaque-support count")
    ax1.set_xlim(10, 50)
    ax2 = ax1.twinx()
    ax2.plot(x, fusion["pcat_mean_hu"], linestyle="--", alpha=.8, label="PCAT mean HU")
    ax2.set_ylabel("OpenPlaque PCAT attenuation (HU)")
    ax1.set_title("RCA: validated longitudinal plaque support vs locked PCAT, 10–50 mm")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    ax1.grid(axis="x", alpha=.15)
    fig.tight_layout()
    p = out / "03_RCA_longitudinal_plaque_PCAT.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def _plot_lad(out: Path, lad: pd.DataFrame):
    x = lad["arc_start_mm"].to_numpy(float) + 0.5
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.plot(x, lad["mapped_native_voxels_vote_ge3"], label="≥3/5 support")
    ax.plot(x, lad["mapped_native_voxels_vote_ge4"], label="≥4/5 support")
    ax.plot(x, lad["mapped_native_voxels_vote_5"], label="5/5 support")
    ax.set_xlabel("Canonical LAD arc length (mm)")
    ax.set_ylabel("Mapped native plaque-support count")
    ax.set_title("LAD plaque confidence — validated longitudinal mapping only")
    ax.legend()
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    p = out / "04_LAD_longitudinal_plaque.png"
    fig.savefig(p, dpi=190)
    plt.close(fig)
    return p


def _pct(v):
    return f"{100*float(v):.1f}%"


def _html_table(df: pd.DataFrame, columns=None, formats=None):
    q = df.copy()
    if columns is not None:
        q = q[columns]
    if formats:
        for c, fn in formats.items():
            if c in q.columns:
                q[c] = q[c].map(fn)
    return q.to_html(index=False, border=0, classes="data")


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    drive_root = Path(drive_root)
    out = Path(output_root or drive_root / OUTPUT_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "baseline_commit": BASELINE})

    canonical_root = _find_dir(drive_root, "Canonical_Source_Coronary_Centerlines_v1")
    atlas_root = _find_dir(drive_root, "Plaque_Ensemble_Confidence_Atlas_v1")
    pcat_root = _find_dir(drive_root, "PCAT_RCA_10_50_Reproducibility_Lock")
    fusion_root = _find_dir(drive_root, "Longitudinal_Plaque_PCAT_Fusion_v1")
    reg_root = _find_dir(drive_root, "Curved_Plaque_to_Source_Registration_Canonical_v1")

    canonical_state = _require_complete(_require_file(canonical_root, "run_state.json"))
    fusion_state = _require_complete(_require_file(fusion_root, "run_state.json"))
    reg_state = _require_complete(_require_file(reg_root, "run_state.json"))

    anatomy = pd.read_csv(_require_file(canonical_root, "canonical_validation_summary.csv"))
    manifest = _read_json(_require_file(canonical_root, "canonical_manifest.json"))
    plaque = pd.read_csv(_require_file(atlas_root, "plaque_vote_confidence_summary.csv"))
    pcat = pd.read_csv(_require_file(pcat_root, "pcat_canonical_primary.csv"))
    pcat_long = pd.read_csv(_require_file(pcat_root, "pcat_canonical_primary_longitudinal.csv"))
    fusion_summary = _read_json(_require_file(fusion_root, "fusion_summary.json"))
    rca_intervals = pd.read_csv(_require_file(fusion_root, "RCA_longitudinal_plaque_intervals.csv"))
    rca_fusion = pd.read_csv(_require_file(fusion_root, "RCA_pcat_plaque_longitudinal_fusion_10_50.csv"))
    lad_long = pd.read_csv(_require_file(fusion_root, "LAD_source_longitudinal_plaque_profile_1mm.csv"))
    axis_recon = _read_json(_require_file(fusion_root, "axis_reconciliation.json"))

    if not anatomy["passes_canonical_gate"].astype(bool).all():
        raise RuntimeError("Canonical anatomy contains a failed validation row.")
    if int(axis_recon["vessels"]["RCA"]["registration_long_axis"]) != 1:
        raise RuntimeError("Expected corrected RCA longitudinal axis 1.")
    if int(axis_recon["vessels"]["LAD"]["registration_long_axis"]) != 2:
        raise RuntimeError("Expected corrected LAD longitudinal axis 2.")
    if fusion_summary.get("status") != "COMPLETE_LONGITUDINAL_RESEARCH_FUSION":
        raise RuntimeError("Longitudinal plaque/PCAT fusion is not the final completed result.")

    anatomy_by = anatomy.set_index("vessel")
    plaque_by = plaque.set_index("vessel")
    pcat_row = pcat.iloc[0]
    rca_overlap = fusion_summary["rca_overlap_summary"]

    lad_majority = _contiguous_intervals(lad_long, "majority_3plus_signal")
    lad_high = _contiguous_intervals(lad_long, "high_4plus_signal")
    lad_strict = _contiguous_intervals(lad_long, "strict_5of5_signal")
    rca_best_majority = _best_interval(rca_intervals, "majority_3plus")
    rca_best_high = _best_interval(rca_intervals, "high_4plus")
    rca_best_strict = _best_interval(rca_intervals, "strict_5of5")

    secondary_cutoff = 12.2
    for key in ("quantitative_compact_lumen_cutoff_mm", "secondary_quantitative_compact_lumen_cutoff_mm"):
        if key in manifest:
            secondary_cutoff = float(manifest[key])
    if isinstance(manifest.get("secondary"), dict):
        secondary_cutoff = float(manifest["secondary"].get("quantitative_compact_lumen_cutoff_mm", secondary_cutoff))

    summary = {
        "status": "COMPLETE_FINAL_INTEGRATED_RESEARCH_REPORT",
        "baseline_commit": BASELINE,
        "scientific_scope": "canonical anatomy + native plaque confidence + validated 1-D plaque localization + locked RCA PCAT",
        "anatomy": {
            "RCA": {
                "length_mm": float(anatomy_by.loc["RCA", "length_mm"]),
                "median_distance_mm": float(anatomy_by.loc["RCA", "current_median_distance_mm"]),
                "inside_current": float(anatomy_by.loc["RCA", "current_inside_fraction"]),
                "inside_legacy": float(anatomy_by.loc["RCA", "legacy_inside_fraction"]),
            },
            "LAD": {
                "length_mm": float(anatomy_by.loc["LAD", "length_mm"]),
                "median_distance_mm": float(anatomy_by.loc["LAD", "current_median_distance_mm"]),
                "inside_current": float(anatomy_by.loc["LAD", "current_inside_fraction"]),
                "inside_legacy": float(anatomy_by.loc["LAD", "legacy_inside_fraction"]),
            },
            "Secondary": {
                "trajectory_length_mm": float(anatomy_by.loc["SECONDARY_TRAJECTORY", "length_mm"]),
                "quantitative_compact_lumen_cutoff_mm": secondary_cutoff,
                "median_distance_mm": float(anatomy_by.loc["SECONDARY_TRAJECTORY", "current_median_distance_mm"]),
            },
        },
        "plaque_confidence_native_curved_space": {
            v: {
                "majority_3plus_mm3": float(plaque_by.loc[v, "majority_3plus_mm3"]),
                "high_4plus_mm3": float(plaque_by.loc[v, "high_4plus_mm3"]),
                "strict_5of5_mm3": float(plaque_by.loc[v, "strict_5of5_mm3"]),
                "any_fold_mm3": float(plaque_by.loc[v, "any_fold_mm3"]),
            } for v in ("RCA", "LAD")
        },
        "rca_pcat_lock": {
            "window_mm": [10.0, 50.0],
            "wall_margin_mm": float(pcat_row["wall_margin_mm"]),
            "mean_hu": float(pcat_row["pcat_mean_hu"]),
            "median_hu": float(pcat_row["pcat_median_hu"]),
            "sd_hu": float(pcat_row["pcat_sd_hu"]),
            "fat_volume_ml": float(pcat_row["fat_volume_ml"]),
            "shell_volume_ml": float(pcat_row["shell_volume_ml"]),
            "fat_fraction": float(pcat_row["fat_fraction"]),
        },
        "longitudinal_fusion": {
            "RCA": {
                "dominant_majority_interval": rca_best_majority,
                "dominant_high_interval": rca_best_high,
                "dominant_strict_interval": rca_best_strict,
                "pcat_window_bins": int(rca_overlap["rca_pcat_bins"]),
                "pcat_window_any_plaque_bins": int(rca_overlap["rca_bins_any_plaque_signal"]),
                "pcat_window_majority_bins": int(rca_overlap["rca_bins_majority_3plus_signal"]),
                "pcat_window_high_bins": int(rca_overlap["rca_bins_high_4plus_signal"]),
                "pcat_window_strict_bins": int(rca_overlap["rca_bins_strict_5of5_signal"]),
                "pcat_mean_all_10_50_fat_weighted_hu": float(rca_overlap["pcat_mean_hu_all_10_50_fat_weighted"]),
                "pcat_mean_majority_bins_fat_weighted_hu": float(rca_overlap["pcat_mean_hu_majority_bins_fat_weighted"]),
            },
            "LAD": {
                "majority_intervals": lad_majority,
                "high_intervals": lad_high,
                "strict_intervals": lad_strict,
            },
        },
        "interpretation": [
            "Canonical RCA, LAD and secondary geometry pass source-CCTA and two-model coronary validation.",
            "Plaque ensemble volumes remain native curved-series vote volumes, not validated source-space TPV.",
            "The dominant reproducible RCA plaque signal is proximal (0–5 mm; strict core 1–4 mm), largely before the standard 10–50 mm RCA PCAT window.",
            "Within the 10–50 mm PCAT window, only the 11–12 mm bin reaches majority/high plaque confidence; no 5/5 bin overlaps that window.",
            "The 11–12 mm overlap bin has more negative PCAT attenuation than the overall 10–50 mm average; this single-bin observation is descriptive only.",
            "Full 3-D plaque localization remains unvalidated because circumferential/angular registration did not pass.",
        ],
        "explicit_non_claims": [
            "No source-space plaque volume or TPV is reported from the curved-series plaque model.",
            "No circumferential/radial plaque localization is asserted.",
            "No LAD plaque-PCAT relationship is asserted because no locked LAD PCAT profile is used.",
            "OpenPlaque PCAT attenuation is not Caristo FAI-Score and no proprietary percentile/risk score is inferred.",
            "The secondary branch is not automatically labeled LCX beyond the validated compact-lumen evidence.",
        ],
    }

    _write_json(out / "final_integrated_summary.json", summary)
    provenance = {
        "baseline_commit": BASELINE,
        "canonical_root": str(canonical_root),
        "plaque_confidence_atlas_root": str(atlas_root),
        "rca_pcat_lock_root": str(pcat_root),
        "longitudinal_fusion_root": str(fusion_root),
        "canonical_registration_root": str(reg_root),
        "canonical_state": canonical_state,
        "fusion_state": fusion_state,
        "registration_state": reg_state,
        "axis_reconciliation": axis_recon,
    }
    _write_json(out / "input_provenance.json", provenance)

    metrics_rows = [
        ["RCA canonical length", summary["anatomy"]["RCA"]["length_mm"], "mm"],
        ["LAD canonical length", summary["anatomy"]["LAD"]["length_mm"], "mm"],
        ["Secondary trajectory length", summary["anatomy"]["Secondary"]["trajectory_length_mm"], "mm"],
        ["Secondary compact-lumen quantitative cutoff", secondary_cutoff, "mm"],
        ["RCA plaque majority >=3/5", plaque_by.loc["RCA", "majority_3plus_mm3"], "native curved-series mm3"],
        ["RCA plaque high >=4/5", plaque_by.loc["RCA", "high_4plus_mm3"], "native curved-series mm3"],
        ["RCA plaque strict 5/5", plaque_by.loc["RCA", "strict_5of5_mm3"], "native curved-series mm3"],
        ["LAD plaque majority >=3/5", plaque_by.loc["LAD", "majority_3plus_mm3"], "native curved-series mm3"],
        ["RCA PCAT mean", pcat_row["pcat_mean_hu"], "HU"],
        ["RCA PCAT median", pcat_row["pcat_median_hu"], "HU"],
        ["RCA PCAT fat volume", pcat_row["fat_volume_ml"], "mL"],
        ["RCA PCAT fat fraction", pcat_row["fat_fraction"], "fraction"],
        ["RCA PCAT-window majority plaque bins", rca_overlap["rca_bins_majority_3plus_signal"], "1-mm bins"],
    ]
    pd.DataFrame(metrics_rows, columns=["metric", "value", "unit"]).to_csv(out / "final_integrated_metrics.csv", index=False)

    figs = [
        _plot_anatomy(out, anatomy),
        _plot_plaque_confidence(out, plaque),
        _plot_rca_fusion(out, rca_fusion),
        _plot_lad(out, lad_long),
    ]

    anatomy_table = anatomy.copy()
    anatomy_table["vessel"] = anatomy_table["vessel"].replace({"SECONDARY_TRAJECTORY": "Secondary trajectory"})
    anatomy_html = _html_table(
        anatomy_table,
        ["vessel", "length_mm", "source_hu_median", "current_inside_fraction", "current_median_distance_mm", "legacy_inside_fraction", "legacy_median_distance_mm"],
        {
            "length_mm": lambda x: f"{x:.2f}",
            "source_hu_median": lambda x: f"{x:.0f}",
            "current_inside_fraction": _pct,
            "current_median_distance_mm": lambda x: f"{x:.3f}",
            "legacy_inside_fraction": _pct,
            "legacy_median_distance_mm": lambda x: f"{x:.3f}",
        },
    )
    plaque_html = _html_table(
        plaque,
        ["vessel", "majority_3plus_mm3", "high_4plus_mm3", "strict_5of5_mm3", "any_fold_mm3"],
        {c: lambda x: f"{x:.0f}" for c in ["majority_3plus_mm3", "high_4plus_mm3", "strict_5of5_mm3", "any_fold_mm3"]},
    )
    rca_interval_html = _html_table(
        rca_intervals,
        ["confidence_level", "arc_start_mm", "arc_end_mm", "duration_mm", "mapped_native_voxels_vote_ge3", "mapped_native_voxels_vote_ge4", "mapped_native_voxels_vote_5", "pcat_overlap_bins", "pcat_mean_hu_fat_weighted"],
        {
            "arc_start_mm": lambda x: f"{x:.1f}",
            "arc_end_mm": lambda x: f"{x:.1f}",
            "duration_mm": lambda x: f"{x:.1f}",
            "pcat_mean_hu_fat_weighted": lambda x: "" if pd.isna(x) else f"{x:.2f}",
        },
    )

    css = """
    body{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;color:#17202a;line-height:1.45}
    h1{margin-bottom:4px} h2{border-bottom:1px solid #d5d8dc;padding-bottom:6px;margin-top:30px}
    h3{margin-bottom:6px}.muted{color:#566573}.warning{background:#fff4e5;border-left:5px solid #d68910;padding:12px 14px}
    .good{background:#eafaf1;border-left:5px solid #239b56;padding:12px 14px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
    .card{border:1px solid #d5d8dc;border-radius:8px;padding:12px;background:#fafafa}.big{font-size:1.55em;font-weight:700}
    table.data{border-collapse:collapse;width:100%;font-size:.92em} table.data th,table.data td{border-bottom:1px solid #e5e7e9;padding:7px;text-align:right}
    table.data th:first-child,table.data td:first-child{text-align:left} img{max-width:100%;height:auto}.small{font-size:.88em}
    @media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}
    """
    rca_major = rca_best_majority or {"arc_start_mm": math.nan, "arc_end_mm": math.nan}
    rca_strict = rca_best_strict or {"arc_start_mm": math.nan, "arc_end_mm": math.nan}
    overlap_hu = float(rca_overlap["pcat_mean_hu_majority_bins_fat_weighted"])
    overall_hu = float(rca_overlap["pcat_mean_hu_all_10_50_fat_weighted"])
    lad_major_text = _format_interval_list(lad_majority)
    lad_strict_text = _format_interval_list(lad_strict)

    html = out / "OPENPLAQUE_FINAL_INTEGRATED_RESEARCH_REPORT.html"
    html.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>OpenPlaque Final Integrated Research Report</title><style>{css}</style></head><body>
    <h1>OpenPlaque — Integrated Coronary Research Report</h1>
    <p class="muted">Canonical source-CCTA anatomy · 5-fold plaque confidence · validated 1-D plaque localization · locked RCA PCAT</p>
    <div class="warning"><b>Research use only.</b> This report deliberately separates what is validated from what is not. Native plaque vote volumes are not source-space TPV, and full 3-D plaque localization is not claimed.</div>

    <h2>Executive summary</h2>
    <div class="grid">
      <div class="card"><div class="big">{summary['anatomy']['RCA']['length_mm']:.1f} mm</div><b>Canonical RCA</b><br><span class="small">median model distance {summary['anatomy']['RCA']['median_distance_mm']:.3f} mm</span></div>
      <div class="card"><div class="big">{summary['anatomy']['LAD']['length_mm']:.1f} mm</div><b>Canonical LAD</b><br><span class="small">median model distance {summary['anatomy']['LAD']['median_distance_mm']:.3f} mm</span></div>
      <div class="card"><div class="big">{float(pcat_row['pcat_mean_hu']):.1f} HU</div><b>RCA PCAT mean</b><br><span class="small">10–50 mm; median {float(pcat_row['pcat_median_hu']):.0f} HU</span></div>
      <div class="card"><div class="big">{int(rca_overlap['rca_bins_majority_3plus_signal'])}</div><b>RCA majority-plaque bin</b><br><span class="small">inside the 40-bin PCAT window</span></div>
    </div>

    <div class="good" style="margin-top:16px"><b>Integrated finding.</b> The strongest reproducible RCA plaque signal is proximal, about {rca_major['arc_start_mm']:.0f}–{rca_major['arc_end_mm']:.0f} mm, with a strict 5/5 core about {rca_strict['arc_start_mm']:.0f}–{rca_strict['arc_end_mm']:.0f} mm. Most of this lies before the locked 10–50 mm RCA PCAT segment. Within that PCAT segment, only 11–12 mm reaches majority/high plaque confidence and its PCAT attenuation is {overlap_hu:.2f} HU versus {overall_hu:.2f} HU for the overall fat-weighted 10–50 mm profile. This is descriptive, not evidence of a biological plaque–PCAT association.</div>

    <h2>1. Canonical coronary anatomy</h2>
    <p>The stale left-coronary centerline problem has been removed from downstream analysis. RCA, LAD, and the secondary trajectory all pass source-CCTA contrast and independent current/legacy coronary-mask validation. The secondary geometry extends to {summary['anatomy']['Secondary']['trajectory_length_mm']:.2f} mm, while quantitative compact-lumen use remains capped at {secondary_cutoff:.1f} mm.</p>
    {anatomy_html}
    <img src="{figs[0].name}" alt="Canonical anatomy validation">

    <h2>2. Plaque ensemble confidence</h2>
    <p>These plaque volumes are measured in the <b>native curved vessel-series model coordinates</b>. They quantify 5-fold model agreement and should not be read as source-space TPV.</p>
    {plaque_html}
    <p>RCA has a substantially more stable ensemble core than LAD. LAD remains markedly model-uncertain despite the now-validated anatomical centerline.</p>
    <img src="{figs[1].name}" alt="Plaque confidence summary">

    <h2>3. RCA PCAT reproducibility lock</h2>
    <p><b>OpenPlaque PCAT Attenuation — not Caristo FAI-Score.</b> Locked RCA analysis uses 10–50 mm, wall margin {float(pcat_row['wall_margin_mm']):.2f} mm, fat window −190 to −30 HU. Mean attenuation is {float(pcat_row['pcat_mean_hu']):.2f} HU; median {float(pcat_row['pcat_median_hu']):.0f} HU; SD {float(pcat_row['pcat_sd_hu']):.2f} HU. Fat volume is {float(pcat_row['fat_volume_ml']):.3f} mL within a {float(pcat_row['shell_volume_ml']):.3f} mL shell (fat fraction {float(pcat_row['fat_fraction']):.3f}).</p>

    <h2>4. Validated longitudinal plaque localization</h2>
    <h3>RCA</h3>
    <p>Full circumferential mapping failed its angular-validation gate, but longitudinal mapping passed strongly. The main confident RCA plaque is proximal and largely outside the standard PCAT segment. Only three PCAT bins contain any mapped plaque signal; only one bin (11–12 mm) reaches ≥3/5 and ≥4/5 confidence, and none reaches 5/5 inside 10–50 mm.</p>
    {rca_interval_html}
    <img src="{figs[2].name}" alt="RCA longitudinal plaque and PCAT">

    <h3>LAD</h3>
    <p>The corrected LAD profile uses validated registration axis 2. Majority-confidence plaque intervals occur at {lad_major_text}. Strict 5/5 support occurs at {lad_strict_text}. No LAD PCAT relationship is reported because no locked LAD PCAT profile is part of this workflow.</p>
    <img src="{figs[3].name}" alt="LAD longitudinal plaque confidence">

    <h2>5. Interpretation and limitations</h2>
    <ul>
      <li><b>Anatomy:</b> high confidence for RCA and LAD canonical centerlines; secondary coronary-like trajectory validated geometrically, with compact-lumen quantitative cutoff retained at {secondary_cutoff:.1f} mm.</li>
      <li><b>Plaque:</b> ensemble confidence is valid in native curved-series coordinates; it is not a validated source-space plaque volume.</li>
      <li><b>Registration:</b> longitudinal position is supported; circumferential/angular mapping is not. No 3-D source-space plaque mask is claimed.</li>
      <li><b>PCAT:</b> RCA result is locked and reproducible over 10–50 mm. The dominant RCA plaque is mostly proximal to that window, limiting lesion-level overlap.</li>
      <li><b>Overlap:</b> the sole majority/high-confidence RCA plaque bin in the PCAT window has more negative PCAT attenuation ({overlap_hu:.2f} HU) than the overall profile ({overall_hu:.2f} HU); the observation is based on one bin and should not be generalized.</li>
      <li><b>Terminology:</b> OpenPlaque PCAT attenuation is not Caristo FAI-Score. No proprietary percentile, normalized score, or clinical risk estimate is inferred.</li>
    </ul>

    <h2>Research/QC provenance</h2>
    <p class="small">Frozen baseline: <code>{BASELINE}</code>. This report requires the completed canonical centerline bundle, plaque confidence atlas, canonical curved-series registration, locked RCA PCAT dataset, and corrected longitudinal plaque/PCAT fusion. See <code>input_provenance.json</code> and <code>final_integrated_summary.json</code> in the package.</p>
    </body></html>""")

    zf = out / "OPENPLAQUE_FINAL_INTEGRATED_RESEARCH_REPORT_BACK.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != zf:
                z.write(p, p.name)

    state = {
        "status": "COMPLETE",
        "scientific_status": "INTEGRATED_WITH_LONGITUDINAL_PLAQUE_ONLY",
        "output_dir": str(out),
        "report": str(html),
        "zip": str(zf),
    }
    _write_json(out / "run_state.json", state)
    return {"summary": summary, **state}
