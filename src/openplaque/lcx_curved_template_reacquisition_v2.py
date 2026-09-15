from __future__ import annotations

"""Cache-only LCX curved-template reacquisition runner.

This is a same-experiment runtime correction.  It deliberately avoids the
historical Full_DICOM.zip dependency.  Curved RCA/LCX image volumes and their
cached masks are loaded directly from UCLA_Plaque_Context_Verification.
"""

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from . import lcx_curved_template_reacquisition as base

BASELINE = base.BASELINE
ALGORITHM = "lcx-curved-template-reacquisition-v1.2-cache-only"
OUTPUT_DIRNAME = base.OUTPUT_DIRNAME

CURVED_ROOT = Path("UCLA_Plaque_Context_Verification")
RCA_INPUT = CURVED_ROOT / "RCA_input/RCA_0000.nii.gz"
LCX_INPUT = CURVED_ROOT / "LCX_input/LCX_0000.nii.gz"
OLD_MASK_DIR = CURVED_ROOT / "nnunet_masks"


def _align_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape == tuple(shape):
        return mask
    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    for p in perms:
        q = np.transpose(mask, p)
        if q.shape == tuple(shape):
            return q
    raise RuntimeError(f"Mask shape {mask.shape} cannot be aligned to image shape {tuple(shape)}")


def _cached_curved_pair(image_path, mask_path):
    image = sitk.ReadImage(str(base._req(image_path)))
    mask_image = sitk.ReadImage(str(base._req(mask_path)))
    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask = _align_mask(sitk.GetArrayFromImage(mask_image), volume.shape)
    return image, volume, mask


def _write_template_figures(out, lcx_template, rca_template, ranking, wrong_rca, wrong_lad):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(base._interp_unit(lcx_template["hu_median"]), label="LCX curved template median HU")
    axes[0].plot(base._interp_unit(rca_template["hu_median"]), label="RCA curved template median HU")
    axes[0].set_ylabel("HU")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(base._interp_unit(lcx_template["plaque_fraction"]), label="LCX template plaque fraction")
    axes[1].plot(base._interp_unit(rca_template["plaque_fraction"]), label="RCA template plaque fraction")
    axes[1].set_xlabel("Normalized longitudinal position")
    axes[1].set_ylabel("Plaque-support fraction")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Historical curved-series templates used only as longitudinal fingerprints")
    fig.tight_layout()
    fig.savefig(out / "01_curved_template_fingerprints.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(ranking):
        q = ranking.head(12)
        ax.bar(np.arange(len(q)), q["score"].to_numpy(float))
        ax.axhline(float(wrong_rca["score"]), linestyle="--", label="LCX template -> accepted RCA")
        ax.axhline(float(wrong_lad["score"]), linestyle=":", label="LCX template -> accepted LAD")
        ax.set_xticks(np.arange(len(q)))
        ax.set_xticklabels([f"C{int(x)+1}" for x in q["candidate_id"]])
        ax.legend()
    ax.set_ylabel("Template match score")
    ax.set_title("LCX-template candidate ranking vs wrong-vessel controls")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "02_candidate_match_scores.png", dpi=180)
    plt.close(fig)


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    root = Path(drive_root)
    out = Path(output_root or root / OUTPUT_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)
    base._json(out / "run_state.json", {
        "status": "STARTED", "baseline_commit": BASELINE, "algorithm": ALGORITHM,
        "input_mode": "cached_curved_nifti_only",
    })

    master = json.loads(base._req(root / base.MASTER).read_text())
    unresolved = set(master.get("unresolved", []))
    if master.get("status") != "CORONARY_ANATOMY_BASELINE_V2_FROZEN":
        raise RuntimeError("Master Anatomy v2 is not frozen")
    if "LCX" not in unresolved:
        raise RuntimeError("LCX is no longer unresolved; this experiment is stale")

    img, source = base._source(root / base.SOURCE_CACHE)
    lad = base._load_path(root / base.LAD, img)
    rca = base._load_path(root / base.RCA, img)

    current = base._resample_mask(root / base.CUR, img)
    legacy = base._resample_mask(root / base.LEG, img)
    aorta = base._resample_mask(root / base.AORTA, img)
    coronary_union = current | legacy

    _, rca_vol, rca_mask = _cached_curved_pair(
        root / RCA_INPUT, root / OLD_MASK_DIR / "RCA.nii.gz"
    )
    _, lcx_vol, lcx_mask = _cached_curved_pair(
        root / LCX_INPUT, root / OLD_MASK_DIR / "LCX.nii.gz"
    )
    rca_template = base._template_fingerprint(rca_vol, rca_mask)
    lcx_template = base._template_fingerprint(lcx_vol, lcx_mask)

    rca_profile = base._path_profile(img, source, rca)
    lad_profile = base._path_profile(img, source, lad)
    rca_control = base._match_score(rca_template, rca_profile)
    wrong_rca = base._match_score(lcx_template, rca_profile)
    wrong_lad = base._match_score(lcx_template, lad_profile)
    wrong_controls = [wrong_rca, wrong_lad]

    controls = pd.DataFrame([
        {"comparison": "RCA template -> accepted RCA", "role": "positive_control", **rca_control},
        {"comparison": "LCX template -> accepted RCA", "role": "wrong_vessel_control", **wrong_rca},
        {"comparison": "LCX template -> accepted LAD", "role": "wrong_vessel_control", **wrong_lad},
    ])
    controls.to_csv(out / "template_control_scores.csv", index=False)

    paths, graph_meta = base._candidate_paths(
        img, source, coronary_union, aorta, lad, rca, max_paths=24
    )

    rows, scored_paths = [], []
    for i, path in enumerate(paths):
        prof = base._path_profile(img, source, path)
        score = base._match_score(lcx_template, prof)
        rca_score = base._match_score(rca_template, prof)
        rows.append({
            "candidate_id": int(i), **score,
            "RCA_template_score": float(rca_score["score"]),
            "LCX_minus_RCA_template_score": float(score["score"] - rca_score["score"]),
        })
        scored_paths.append((score["score"], i, path))

    columns = [
        "candidate_id", "score", "intensity_corr", "plaque_landmark_corr",
        "length_score", "orientation", "robust_fraction", "median_hu",
        "length_mm", "RCA_template_score", "LCX_minus_RCA_template_score",
    ]
    ranking = pd.DataFrame(rows)
    if len(ranking):
        ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        ranking = pd.DataFrame(columns=columns)
    ranking.to_csv(out / "LCX_template_candidate_ranking.csv", index=False)

    ordered_paths = []
    if len(ranking):
        by_id = {i: p for _, i, p in scored_paths}
        for rank, row in ranking.iterrows():
            path = by_id[int(row.candidate_id)]
            ordered_paths.append(path)
            pd.DataFrame(path, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
                out / f"candidate_{rank+1:02d}_source_path.csv", index=False
            )

    _write_template_figures(out, lcx_template, rca_template, ranking, wrong_rca, wrong_lad)
    base._write_qc(out, img, source, ordered_paths)

    status = base._decision(rca_control, wrong_controls, ranking)
    top = ranking.iloc[0].to_dict() if len(ranking) else None
    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "input_mode": "cached_curved_nifti_only",
        "historical_Full_DICOM_zip_used": False,
        "master_status": master.get("status"),
        "LCX_master_status": "UNRESOLVED",
        "stale_LCX_source_centerline_used": False,
        "RCA_positive_control": rca_control,
        "LCX_wrong_vessel_controls": {"accepted_RCA": wrong_rca, "accepted_LAD": wrong_lad},
        "graph": graph_meta,
        "top_candidate": top,
        "scientific_boundary": (
            "A passing template match is only a source-path reacquisition candidate. "
            "It does not establish LCX identity, circumferential registration, "
            "source-space plaque localization, or plaque volume."
        ),
    }
    base._json(out / "summary.json", summary)

    report = out / "OPENPLAQUE_LCX_CURVED_TEMPLATE_REACQUISITION_REPORT.html"
    report.write_text(base._report_html(out, summary, ranking, controls), encoding="utf-8")

    base._json(out / "run_state.json", {
        "status": "COMPLETE", "scientific_status": status,
        "baseline_commit": BASELINE, "algorithm": ALGORITHM,
        "input_mode": "cached_curved_nifti_only",
    })

    zip_path = out / "OPENPLAQUE_LCX_CURVED_TEMPLATE_REACQUISITION_REPORT_BACK.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out.iterdir()):
            if fp.is_file() and fp != zip_path:
                zf.write(fp, arcname=fp.name)

    return {"summary": summary, "report": str(report), "zip": str(zip_path), "output_dir": str(out)}


def synthetic_lcx_template_v2_self_test():
    a = np.arange(24 * 5 * 7).reshape(24, 5, 7)
    assert _align_mask(a, a.shape).shape == a.shape
    b = np.transpose(a, (0, 2, 1))
    assert _align_mask(b, a.shape).shape == a.shape
    return {
        "passed": True,
        "algorithm": ALGORITHM,
        "historical_Full_DICOM_zip_used": False,
    }
