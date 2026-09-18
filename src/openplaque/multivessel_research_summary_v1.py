from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "openplaque-multivessel-research-summary-v1.0"
OUTPUT_DIRNAME = "OpenPlaque_Multivessel_Research_Summary_v1"

MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
RCA_LOCK = Path("RCA_Plaque_PCAT_Research_Lock_v1/summary.json")
LAD_PCAT = Path("LAD_Source_Space_PCAT_Feasibility_v1/summary.json")
LAD_PLAQUE = Path("LAD_Distal_Reference_Plaque_Self_Calibration_v1/summary.json")
LCX_FREEZE = Path("LCX_Structural_Source_QC_Freeze_v1/summary.json")
LCX_PCAT = Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/summary.json")

EXPECTED = {
    "master": "CORONARY_ANATOMY_BASELINE_V2_FROZEN",
    "rca": "RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED",
    "lad_pcat": "LAD_SOURCE_SPACE_PCAT_FEASIBILITY_COMPLETE",
    "lcx_freeze": "LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN",
    "lcx_pcat": "LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_COMPLETE",
}

STATUS = "OPENPLAQUE_MULTIVESSEL_RESEARCH_SUMMARY_COMPLETE"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _measurement_rows(rca, lad_pcat, lad_plaque, lcx_pcat):
    rows = []

    rows.append({
        "artery_or_segment": "RCA",
        "measurement_type": "plaque_excess_proxy",
        "value": float(rca["nominal_plaque_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"]),
        "unit": "mm3",
        "status": "LOCKED_RESEARCH_BENCHMARK",
        "segment_definition": "accepted RCA; reference-normalized fixed-shell plaque proxy",
        "clinical_tpv": False,
        "notes": "Only currently locked plaque burden proxy.",
    })
    rows.append({
        "artery_or_segment": "RCA",
        "measurement_type": "direct_pcat_mean",
        "value": float(rca["pcat"]["fat_voxel_weighted_mean_hu"]),
        "unit": "HU",
        "status": "LOCKED_RESEARCH_BENCHMARK",
        "segment_definition": f"{rca['pcat']['arc_start_mm']:.0f}-{rca['pcat']['arc_end_mm']:.0f} mm",
        "clinical_tpv": False,
        "notes": "Direct attenuation; not proprietary FAI.",
    })

    frozen_lad = lad_pcat["segments"]["frozen_LAD"]
    rows.append({
        "artery_or_segment": "LAD frozen",
        "measurement_type": "direct_pcat_mean",
        "value": float(frozen_lad["pcat_mean_hu"]),
        "unit": "HU",
        "status": "TECHNICALLY_FEASIBLE_RESEARCH_MEASUREMENT",
        "segment_definition": f"{frozen_lad['frozen_arc_start_mm']:.1f}-{frozen_lad['frozen_arc_end_mm']:.1f} mm frozen-LAD coordinate",
        "clinical_tpv": False,
        "notes": "100% source-plane QC; direct attenuation only.",
    })
    distal = lad_pcat["segments"]["validated_distal_continuation"]
    rows.append({
        "artery_or_segment": "LAD validated distal continuation",
        "measurement_type": "direct_pcat_mean",
        "value": float(distal["pcat_mean_hu"]),
        "unit": "HU",
        "status": "RESEARCH_ONLY_ANATOMY",
        "segment_definition": f"{distal['frozen_arc_start_mm']:.1f} to {distal['frozen_arc_end_mm']:.1f} mm",
        "clinical_tpv": False,
        "notes": "Distal continuation is independently anatomy-validated but not part of frozen master.",
    })

    rows.append({
        "artery_or_segment": "LAD frozen",
        "measurement_type": "plaque_excess_proxy_developmental",
        "value": float(lad_plaque["nominal_frozen_lad_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"]),
        "unit": "mm3",
        "status": str(lad_plaque["status"]),
        "segment_definition": "frozen LAD; LAD-specific distal-reference self-calibration",
        "clinical_tpv": False,
        "notes": (
            f"Developmental only; majority AUROC {lad_plaque['developmental_specificity']['majority_vs_vote_free_auc']:.3f}, "
            f"strict AUROC {lad_plaque['developmental_specificity']['strict5_vs_vote_free_auc']:.3f}, "
            f"min shell rho {lad_plaque['min_cross_shell_profile_spearman']:.3f}."
        ),
    })

    for key, label in (("C6", "C6 LCX-like parent"), ("C7", "C7 OM-like daughter")):
        p = lcx_pcat["pcat_primary"][key]
        rows.append({
            "artery_or_segment": label,
            "measurement_type": "direct_pcat_mean",
            "value": float(p["pcat_mean_hu"]),
            "unit": "HU",
            "status": "TECHNICALLY_FEASIBLE_RESEARCH_MEASUREMENT",
            "segment_definition": f"{p['segment_arc_start_mm']:.1f}-{p['segment_arc_end_mm']:.1f} mm post-split",
            "clinical_tpv": False,
            "notes": "Research structural label only; not proprietary FAI.",
        })
        s = lcx_pcat["nominal_1mm_raw_shell_totals"][key]
        rows.append({
            "artery_or_segment": label,
            "measurement_type": "raw_1mm_nonfatlike_shell",
            "value": float(s["raw_nonfatlike_shell_mm3"]),
            "unit": "mm3",
            "status": "RAW_COMPOSITION_ONLY",
            "segment_definition": f"{s['evaluable_length_mm']:.2f} mm evaluable path",
            "clinical_tpv": False,
            "notes": "Not plaque burden and must not be interpreted as TPV.",
        })
    return pd.DataFrame(rows)


def _anatomy_rows(master, lad_pcat, lcx_freeze):
    return pd.DataFrame([
        {
            "structure": "RCA",
            "anatomy_status": "frozen accepted",
            "length_mm": 52.1971,
            "research_quantification": "locked plaque proxy + locked PCAT",
            "clinical_identity_status": "accepted RCA",
        },
        {
            "structure": "LAD",
            "anatomy_status": "frozen accepted",
            "length_mm": float(lad_pcat["frozen_lad_length_mm"]),
            "research_quantification": "PCAT feasible; plaque developmental only",
            "clinical_identity_status": "accepted LAD",
        },
        {
            "structure": "LAD distal continuation",
            "anatomy_status": "bidirectionally source-confirmed; not in master",
            "length_mm": float(lad_pcat["distal_extension_length_mm"]),
            "research_quantification": "PCAT feasible",
            "clinical_identity_status": "research continuation only",
        },
        {
            "structure": "C6",
            "anatomy_status": lcx_freeze["decision"]["research_structural_label_C6"],
            "length_mm": float(lcx_freeze["C6_post_split_length_mm"]),
            "research_quantification": "raw shell composition + PCAT feasible",
            "clinical_identity_status": "clinical LCX identity not established",
        },
        {
            "structure": "C7",
            "anatomy_status": lcx_freeze["decision"]["research_structural_label_C7"],
            "length_mm": float(lcx_freeze["C7_post_split_length_mm"]),
            "research_quantification": "raw shell composition + PCAT feasible",
            "clinical_identity_status": "clinical OM identity not established",
        },
        {
            "structure": "Left main",
            "anatomy_status": "UNRESOLVED",
            "length_mm": np.nan,
            "research_quantification": "disabled",
            "clinical_identity_status": "unresolved",
        },
    ])


def _plot_pcat(measurements, out):
    d = measurements[measurements.measurement_type == "direct_pcat_mean"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(d))
    ax.bar(x, d.value.to_numpy(float))
    ax.set_xticks(x, d.artery_or_segment, rotation=25, ha="right")
    ax.set_ylabel("Direct PCAT mean (HU)")
    ax.set_title("OpenPlaque direct PCAT research measurements")
    for i, v in enumerate(d.value):
        ax.text(i, float(v)+1.0, f"{float(v):.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.01, 0.02,
        "Different segment definitions and lengths; descriptive comparison only. Not proprietary FAI.",
        transform=ax.transAxes, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_readiness(anatomy, out):
    labels = anatomy.structure.tolist()
    readiness = []
    for _, r in anatomy.iterrows():
        if r.structure == "RCA":
            readiness.append(3)
        elif r.structure == "LAD":
            readiness.append(2)
        elif r.structure == "LAD distal continuation":
            readiness.append(1)
        elif r.structure in ("C6", "C7"):
            readiness.append(1)
        else:
            readiness.append(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, readiness)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_yticks(
        [0,1,2,3],
        ["unresolved","research anatomy only","quantification feasible","locked benchmark"],
    )
    ax.set_ylim(-0.2, 3.4)
    ax.set_title("OpenPlaque research readiness by coronary structure")
    ax.text(
        0.01, 0.02,
        "Readiness categories are workflow states, not clinical grades or disease severity.",
        transform=ax.transAxes, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    dummy = pd.DataFrame({
        "measurement_type":["direct_pcat_mean","plaque_excess_proxy"],
        "value":[-90.0,20.0],
    })
    assert int((dummy.measurement_type == "direct_pcat_mean").sum()) == 1
    return {"ok": True}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master = _read_json(root/MASTER)
    rca = _read_json(root/RCA_LOCK)
    lad_pcat = _read_json(root/LAD_PCAT)
    lad_plaque = _read_json(root/LAD_PLAQUE)
    lcx_freeze = _read_json(root/LCX_FREEZE)
    lcx_pcat = _read_json(root/LCX_PCAT)

    if master.get("status") != EXPECTED["master"]:
        raise RuntimeError(f"Unexpected master status {master.get('status')}")
    if rca.get("status") != EXPECTED["rca"]:
        raise RuntimeError(f"Unexpected RCA lock status {rca.get('status')}")
    if lad_pcat.get("status") != EXPECTED["lad_pcat"]:
        raise RuntimeError(f"Unexpected LAD PCAT status {lad_pcat.get('status')}")
    if lcx_freeze.get("status") != EXPECTED["lcx_freeze"]:
        raise RuntimeError(f"Unexpected LCX structural freeze status {lcx_freeze.get('status')}")
    if lcx_pcat.get("status") != EXPECTED["lcx_pcat"]:
        raise RuntimeError(f"Unexpected LCX/OM PCAT status {lcx_pcat.get('status')}")

    measurements = _measurement_rows(rca, lad_pcat, lad_plaque, lcx_pcat)
    anatomy = _anatomy_rows(master, lad_pcat, lcx_freeze)

    measurements.to_csv(out/"coronary_research_measurements.csv", index=False)
    anatomy.to_csv(out/"coronary_anatomy_research_status.csv", index=False)

    pcat_rows = measurements[measurements.measurement_type == "direct_pcat_mean"].copy()
    pcat_rows.to_csv(out/"coronary_direct_pcat_summary.csv", index=False)

    _plot_pcat(measurements, out/"01_multivessel_direct_PCAT.png")
    _plot_readiness(anatomy, out/"02_coronary_research_readiness.png")

    rca_total = float(rca["nominal_plaque_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"])
    lad_dev = float(lad_plaque["nominal_frozen_lad_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"])

    summary = {
        "status": STATUS,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "current_quantitative_state": {
            "RCA": {
                "anatomy": "frozen accepted",
                "plaque": {
                    "status": "LOCKED_RESEARCH_BENCHMARK",
                    "excess_proxy_mm3": rca_total,
                    "majority_auc": rca["plaque_validation_benchmark"]["majority_vs_vote_free_auc"],
                    "strict_auc": rca["plaque_validation_benchmark"]["strict5_vs_vote_free_auc"],
                },
                "pcat": {
                    "status": "LOCKED_RESEARCH_BENCHMARK",
                    "mean_hu": rca["pcat"]["fat_voxel_weighted_mean_hu"],
                    "segment_mm": [rca["pcat"]["arc_start_mm"], rca["pcat"]["arc_end_mm"]],
                },
            },
            "LAD": {
                "anatomy": "frozen accepted; distal continuation separately source-confirmed",
                "plaque": {
                    "status": lad_plaque["status"],
                    "developmental_excess_proxy_mm3": lad_dev,
                    "majority_auc": lad_plaque["developmental_specificity"]["majority_vs_vote_free_auc"],
                    "strict_auc": lad_plaque["developmental_specificity"]["strict5_vs_vote_free_auc"],
                    "min_cross_shell_spearman": lad_plaque["min_cross_shell_profile_spearman"],
                    "locked": False,
                },
                "pcat": {
                    "status": lad_pcat["status"],
                    "frozen_lad_mean_hu": lad_pcat["segments"]["frozen_LAD"]["pcat_mean_hu"],
                    "distal_continuation_mean_hu": lad_pcat["segments"]["validated_distal_continuation"]["pcat_mean_hu"],
                    "locked": False,
                },
            },
            "C6_C7": {
                "anatomy": {
                    "C6": "LCX-like parent continuation",
                    "C7": "OM-like daughter",
                    "clinical_identity_established": False,
                },
                "plaque": "No validated plaque-excess model; raw shell composition only.",
                "pcat": {
                    "C6_mean_hu": lcx_pcat["pcat_primary"]["C6"]["pcat_mean_hu"],
                    "C7_mean_hu": lcx_pcat["pcat_primary"]["C7"]["pcat_mean_hu"],
                    "status": lcx_pcat["status"],
                },
            },
            "left_main": {
                "status": "UNRESOLVED",
                "quantification_allowed": False,
            },
        },
        "reportable_research_measurements": measurements.to_dict(orient="records"),
        "scientific_boundary": (
            "This is an integrated research summary, not a clinical coronary report. RCA plaque is the only locked plaque-burden proxy and remains a "
            "reference-normalized research excess-wall quantity rather than clinical TPV. LAD plaque remains developmental because strict-plaque discrimination "
            "and shell stability missed prespecified gates. C6/C7 raw shell volumes are not plaque burden. Direct PCAT values are not proprietary FAI. "
            "C6/C7 clinical identities and left main remain unresolved, and the frozen master anatomy is unchanged."
        ),
    }
    _write_json(out/"research_measurements.json", summary)
    _write_json(out/"input_provenance.json", {
        "master": str(root/MASTER),
        "rca_lock": str(root/RCA_LOCK),
        "lad_pcat": str(root/LAD_PCAT),
        "lad_plaque": str(root/LAD_PLAQUE),
        "lcx_freeze": str(root/LCX_FREEZE),
        "lcx_pcat": str(root/LCX_PCAT),
    })

    report = out/"OPENPLAQUE_MULTIVESSEL_RESEARCH_SUMMARY_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Multivessel Research Summary v1</h1>"
        f"<p><b>Status:</b> {STATUS}</p>"
        "<h2>Current state</h2>"
        f"<p><b>RCA:</b> frozen anatomy; locked plaque excess proxy {rca_total:.2f} mm³; "
        f"locked direct PCAT {rca['pcat']['fat_voxel_weighted_mean_hu']:.2f} HU.</p>"
        f"<p><b>LAD:</b> frozen anatomy {lad_pcat['frozen_lad_length_mm']:.2f} mm; "
        f"direct PCAT {lad_pcat['segments']['frozen_LAD']['pcat_mean_hu']:.2f} HU. "
        f"Developmental plaque estimate {lad_dev:.2f} mm³ did not pass all specificity/stability gates and is not locked.</p>"
        f"<p><b>LAD distal continuation:</b> {lad_pcat['distal_extension_length_mm']:.2f} mm independently source-confirmed; "
        f"direct PCAT {lad_pcat['segments']['validated_distal_continuation']['pcat_mean_hu']:.2f} HU; not part of frozen master.</p>"
        f"<p><b>C6 LCX-like parent:</b> direct PCAT {lcx_pcat['pcat_primary']['C6']['pcat_mean_hu']:.2f} HU; "
        f"raw 1-mm non-fatlike shell {lcx_pcat['nominal_1mm_raw_shell_totals']['C6']['raw_nonfatlike_shell_mm3']:.2f} mm³.</p>"
        f"<p><b>C7 OM-like daughter:</b> direct PCAT {lcx_pcat['pcat_primary']['C7']['pcat_mean_hu']:.2f} HU; "
        f"raw 1-mm non-fatlike shell {lcx_pcat['nominal_1mm_raw_shell_totals']['C7']['raw_nonfatlike_shell_mm3']:.2f} mm³.</p>"
        "<p><b>Left main:</b> unresolved; downstream quantification disabled.</p>"
        "<p><b>Boundary:</b> research measurements only. RCA plaque is not clinical TPV; direct PCAT is not proprietary FAI; "
        "LAD plaque is developmental; C6/C7 raw shell composition is not plaque burden.</p>"
        '<img src="01_multivessel_direct_PCAT.png" style="max-width:100%">'
        '<img src="02_coronary_research_readiness.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json", {
        "status":"COMPLETE",
        "result_status":STATUS,
        "algorithm":ALGORITHM,
        "baseline":BASELINE,
    })

    zpath = out/"OPENPLAQUE_MULTIVESSEL_RESEARCH_SUMMARY_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p, p.name)

    return {"summary": summary, "report": str(report), "zip": str(zpath)}
