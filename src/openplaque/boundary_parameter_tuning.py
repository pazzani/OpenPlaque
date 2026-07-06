"""Supervised boundary-parameter evaluation using cached nnU-Net predictions.

This module is intentionally *not* cross-validation. Boundary refinement is a
fixed deterministic post-processing step applied after nnU-Net prediction. The
recommended workflow is therefore:

1. collect paired sample image/label cases,
2. run nnU-Net once per case and cache predictions,
3. evaluate every boundary-parameter combination against all labeled cases,
4. rank parameter sets by average supervised metrics.

Supported layouts:

    Sample_Dataset/P02_LAD_axial_0000.nii.gz  # CT input image
    Sample_Dataset/P02_LAD_axial.nii.gz       # expert label mask 0/1/2

or nnU-Net raw layout:

    Dataset001_CCTA_DHM/imagesTr/*_0000.nii.gz
    Dataset001_CCTA_DHM/labelsTr/*.nii.gz

Research use only. Not clinically validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .boundary import refine_plaque_mask

PLAQUE_LABEL = 2
PARAM_COLUMNS = [
    "min_component_voxels",
    "lumen_distance_voxels",
    "high_hu_threshold",
    "low_hu_threshold",
    "closing_radius_voxels",
    "fill_holes",
    "connectivity",
    "erode_core",
    "erosion_iterations",
]

DEFAULT_GRID: dict[str, Sequence[Any]] = {
    # Existing/core parameters.
    "min_component_voxels": [1, 5, 10, 25, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None, 700, 850, 1000],
    "low_hu_threshold": [None, -100, -50],
    # Added parameters.
    "closing_radius_voxels": [0, 1],
    "fill_holes": [False, True],
    "connectivity": [6, 18, 26],
    # Fixed for main estimate; core masks can be generated separately.
    "erode_core": [False],
    "erosion_iterations": [1],
}

SMALL_GRID: dict[str, Sequence[Any]] = {
    "min_component_voxels": [1, 10, 25],
    "lumen_distance_voxels": [0, 1],
    "high_hu_threshold": [None, 850],
    "low_hu_threshold": [None],
    "closing_radius_voxels": [0, 1],
    "fill_holes": [False],
    "connectivity": [26],
    "erode_core": [False],
    "erosion_iterations": [1],
}


@dataclass(frozen=True)
class SampleCase:
    case_id: str
    image_path: Path
    label_path: Path


def strip_nii_suffix(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def case_id_from_image(path: str | Path) -> str:
    stem = strip_nii_suffix(Path(path).name)
    return stem[:-5] if stem.endswith("_0000") else stem


def _candidate_label_paths(image_path: Path, dataset_dir: Path) -> list[Path]:
    cid = case_id_from_image(image_path)
    candidates = [
        image_path.with_name(f"{cid}.nii.gz"),
        image_path.with_name(f"{cid}.nii"),
        dataset_dir / "labelsTr" / f"{cid}.nii.gz",
        dataset_dir / "labelsTr" / f"{cid}.nii",
        dataset_dir / "labels" / f"{cid}.nii.gz",
        dataset_dir / "labels" / f"{cid}.nii",
    ]
    try:
        candidates.extend(dataset_dir.rglob(f"{cid}.nii.gz"))
        candidates.extend(dataset_dir.rglob(f"{cid}.nii"))
    except Exception:
        pass
    seen, out = set(), []
    for p in candidates:
        p = Path(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def collect_sample_cases(dataset_dir: str | Path, limit: Optional[int] = None) -> list[SampleCase]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    image_paths: list[Path] = []
    if (dataset_dir / "imagesTr").exists():
        image_paths.extend((dataset_dir / "imagesTr").glob("*_0000.nii.gz"))
        image_paths.extend((dataset_dir / "imagesTr").glob("*_0000.nii"))
    image_paths.extend(dataset_dir.rglob("*_0000.nii.gz"))
    image_paths.extend(dataset_dir.rglob("*_0000.nii"))
    image_paths = sorted(set(image_paths))

    cases: list[SampleCase] = []
    for img in image_paths:
        label_path = None
        for cand in _candidate_label_paths(img, dataset_dir):
            if cand.exists() and cand.resolve() != img.resolve():
                label_path = cand
                break
        if label_path is not None:
            cases.append(SampleCase(case_id_from_image(img), img, label_path))

    dedup: dict[str, SampleCase] = {}
    for c in cases:
        dedup.setdefault(c.case_id, c)
    cases = sorted(dedup.values(), key=lambda c: c.case_id)
    if limit is not None:
        cases = cases[:int(limit)]
    if not cases:
        raise RuntimeError(
            f"No paired *_0000 image + label cases found under {dataset_dir}. "
            "Expected image P02_LAD_axial_0000.nii.gz and label P02_LAD_axial.nii.gz, "
            "or nnU-Net imagesTr/labelsTr layout."
        )
    return cases


def read_nifti(path: str | Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def detect_available_model_folds(results_root: str | Path, dataset_name: str = "Dataset001_CCTA_DHM") -> list[str]:
    root = Path(results_root) / dataset_name
    folds = sorted({p.name.replace("fold_", "") for p in root.rglob("fold_*") if p.is_dir()})
    return folds or ["0"]


def prepare_prediction_input(cases: Sequence[SampleCase], input_dir: str | Path, overwrite: bool = True) -> Path:
    input_dir = Path(input_dir)
    if input_dir.exists() and overwrite:
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        shutil.copy2(c.image_path, input_dir / f"{c.case_id}_0000.nii.gz")
    return input_dir


def prediction_path(pred_dir: str | Path, case_id: str) -> Path:
    pred_dir = Path(pred_dir)
    for suffix in (".nii.gz", ".nii"):
        p = pred_dir / f"{case_id}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Prediction not found for {case_id} in {pred_dir}")


def prediction_cache_status(cases: Sequence[SampleCase], pred_dir: str | Path) -> pd.DataFrame:
    pred_dir = Path(pred_dir)
    rows = []
    for c in cases:
        found = None
        for suffix in (".nii.gz", ".nii"):
            p = pred_dir / f"{c.case_id}{suffix}"
            if p.exists():
                found = p
                break
        rows.append({"case_id": c.case_id, "prediction_exists": found is not None, "prediction_path": None if found is None else str(found)})
    return pd.DataFrame(rows)


def ensure_prediction_cache(
    cases: Sequence[SampleCase],
    pred_dir: str | Path,
    input_dir: str | Path,
    dataset_name_or_id: str = "Dataset001_CCTA_DHM",
    configuration: str = "3d_fullres",
    folds: Optional[Sequence[str | int]] = None,
    trainer: Optional[str] = None,
    plans: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Run nnUNetv2_predict once for all cases, or reuse cached predictions."""
    pred_dir = Path(pred_dir)
    input_dir = Path(input_dir)
    expected = [pred_dir / f"{c.case_id}.nii.gz" for c in cases]
    if pred_dir.exists() and all(p.exists() for p in expected) and not overwrite:
        print(f"Using cached predictions in {pred_dir}")
        return pred_dir

    prepare_prediction_input(cases, input_dir, overwrite=True)
    if pred_dir.exists() and overwrite:
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    folds = [str(f) for f in (folds if folds is not None else [0])]
    cmd = [
        "nnUNetv2_predict", "-i", str(input_dir), "-o", str(pred_dir),
        "-d", str(dataset_name_or_id), "-c", configuration, "-f", *folds,
    ]
    if trainer:
        cmd += ["-tr", trainer]
    if plans:
        cmd += ["-p", plans]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return pred_dir


def iter_parameter_grid(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None):
    grid = dict(DEFAULT_GRID if parameter_grid is None else parameter_grid)
    for col in PARAM_COLUMNS:
        grid.setdefault(col, DEFAULT_GRID[col])
    for candidate_id, values in enumerate(product(*[grid[c] for c in PARAM_COLUMNS])):
        params = dict(zip(PARAM_COLUMNS, values))
        params["min_component_voxels"] = int(params["min_component_voxels"])
        params["lumen_distance_voxels"] = int(params["lumen_distance_voxels"])
        params["closing_radius_voxels"] = int(params["closing_radius_voxels"])
        params["fill_holes"] = bool(params["fill_holes"])
        params["connectivity"] = int(params["connectivity"])
        params["erode_core"] = bool(params["erode_core"])
        params["erosion_iterations"] = int(params["erosion_iterations"])
        yield candidate_id, params


def parameter_grid_size(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> int:
    return sum(1 for _ in iter_parameter_grid(parameter_grid))


def parameter_grid_dataframe(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> pd.DataFrame:
    return pd.DataFrame([{"candidate_id": cid, **params} for cid, params in iter_parameter_grid(parameter_grid)])


def mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, spacing=(1.0, 1.0, 1.0), label: int = PLAQUE_LABEL) -> dict[str, float]:
    pred = np.asarray(pred_mask) == label
    true = np.asarray(true_mask) == label
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    pred_n = int(pred.sum())
    true_n = int(true.sum())
    dice = 1.0 if (pred_n + true_n) == 0 else 2 * tp / (pred_n + true_n)
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    voxel_vol = float(np.prod(spacing))
    pred_tpv = pred_n * voxel_vol
    true_tpv = true_n * voxel_vol
    abs_tpv_err_frac = 0.0 if true_tpv == 0 and pred_tpv == 0 else (1.0 if true_tpv == 0 else abs(pred_tpv - true_tpv) / true_tpv)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pred_tpv_mm3": float(pred_tpv),
        "true_tpv_mm3": float(true_tpv),
        "tpv_error_mm3": float(pred_tpv - true_tpv),
        "abs_tpv_error_fraction": float(abs_tpv_err_frac),
        "pred_voxels": int(pred_n),
        "true_voxels": int(true_n),
    }


def score_metrics(m: Mapping[str, float]) -> float:
    tpv_term = max(0.0, 1.0 - min(float(m["abs_tpv_error_fraction"]), 1.0))
    return float(0.35*m["dice"] + 0.20*m["iou"] + 0.20*tpv_term + 0.15*m["precision"] + 0.10*m["recall"])


def evaluate_case_grid(case: SampleCase, pred_dir: str | Path, parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> pd.DataFrame:
    img, vol = read_nifti(case.image_path)
    _, label = read_nifti(case.label_path)
    _, pred = read_nifti(prediction_path(pred_dir, case.case_id))
    spacing = tuple(float(x) for x in img.GetSpacing())
    rows = []
    for candidate_id, params in iter_parameter_grid(parameter_grid):
        ref = refine_plaque_mask(
            volume=vol,
            mask=pred,
            spacing=spacing,
            remove_small=params["min_component_voxels"] > 0,
            min_component_voxels=max(1, params["min_component_voxels"]),
            trim_lumen_adjacent=params["lumen_distance_voxels"] > 0,
            lumen_distance_voxels=params["lumen_distance_voxels"],
            erode_core=params["erode_core"],
            erosion_iterations=params["erosion_iterations"],
            high_hu_threshold=params["high_hu_threshold"],
            low_hu_threshold=params["low_hu_threshold"],
            closing_radius_voxels=params["closing_radius_voxels"],
            fill_holes=params["fill_holes"],
            connectivity=params["connectivity"],
        )
        m = mask_metrics(ref.refined_mask, label, spacing=spacing)
        rows.append({
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            **params,
            **m,
            "score": score_metrics(m),
            "raw_pred_tpv_mm3": ref.original_tpv_mm3,
            "refined_tpv_mm3": ref.refined_tpv_mm3,
            "removed_tpv_mm3": ref.removed_volume_mm3,
        })
    return pd.DataFrame(rows)


def evaluate_all_cases(cases: Sequence[SampleCase], pred_dir: str | Path, parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> pd.DataFrame:
    frames = []
    for i, case in enumerate(cases, start=1):
        print(f"Evaluating {case.case_id} ({i}/{len(cases)})")
        frames.append(evaluate_case_grid(case, pred_dir, parameter_grid=parameter_grid))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_by_params(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("No rows to aggregate")
    summary = rows.groupby(PARAM_COLUMNS, dropna=False).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_abs_tpv_error_fraction=("abs_tpv_error_fraction", "mean"),
        mean_tpv_error_mm3=("tpv_error_mm3", "mean"),
        n_cases=("case_id", "nunique"),
        n_rows=("case_id", "count"),
    ).reset_index()
    summary["std_score"] = summary["std_score"].fillna(0.0)
    summary = summary.sort_values(["mean_score", "mean_dice", "mean_abs_tpv_error_fraction"], ascending=[False, False, True]).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary)+1))
    return summary


def params_from_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for col in PARAM_COLUMNS:
        val = row[col]
        if pd.isna(val):
            val = None
        elif col in ("min_component_voxels", "lumen_distance_voxels", "closing_radius_voxels", "connectivity", "erosion_iterations"):
            val = int(val)
        elif col in ("erode_core", "fill_holes"):
            val = bool(val)
        params[col] = val
    params["remove_small"] = int(params["min_component_voxels"]) > 0
    params["trim_lumen_adjacent"] = int(params["lumen_distance_voxels"]) > 0
    return params


def select_best_params(rows: pd.DataFrame) -> dict[str, Any]:
    return params_from_summary_row(aggregate_by_params(rows).iloc[0])


def save_parameter_evaluation_outputs(
    all_case_results: pd.DataFrame,
    final_params: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_by_params(all_case_results)
    paths = {
        "case_results": output_dir / "all_case_parameter_results.csv",
        "summary": output_dir / "parameter_summary.csv",
        "best_json": output_dir / "best_boundary_parameters.json",
        "html": output_dir / "boundary_parameter_tuning_report.html",
    }
    all_case_results.to_csv(paths["case_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    payload = {
        "final_parameters_selected_on_all_cases": dict(final_params),
        "best_mean_score": float(summary.iloc[0]["mean_score"]),
        "best_mean_dice": float(summary.iloc[0]["mean_dice"]),
        "best_mean_abs_tpv_error_fraction": float(summary.iloc[0]["mean_abs_tpv_error_fraction"]),
        "scoring": "0.35*Dice + 0.20*IoU + 0.20*(1-min(abs TPV error fraction,1)) + 0.15*Precision + 0.10*Recall",
        "prediction_cache_note": "nnU-Net prediction is run once per case and cached; boundary parameters are evaluated downstream on cached masks.",
        "cross_validation": False,
        "research_use_only": True,
    }
    paths["best_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_parameter_evaluation_html(paths["html"], summary, final_params)
    return paths


def write_parameter_evaluation_html(output_html: str | Path, summary: pd.DataFrame, final_params: Mapping[str, Any]) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    top = summary.head(50).copy()
    for col in top.columns:
        if str(col).startswith("mean_") or str(col).startswith("std_"):
            top[col] = top[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenPlaque Boundary Parameter Tuning</title>
<style>body {{font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color:#222;}}
table {{border-collapse: collapse; width:100%; margin: 16px 0 28px; font-size: 13px;}}
th,td {{border:1px solid #ddd; padding:6px 8px; text-align:right;}} th:first-child,td:first-child {{text-align:left;}} th {{background:#f4f4f4;}}
pre {{background:#f7f7f7; padding:12px; border-radius:8px; overflow-x:auto;}}
.notice {{background:#fff3cd; border:1px solid #ffe08a; padding:12px 14px; border-radius:8px;}}</style></head><body>
<h1>OpenPlaque Boundary Parameter Tuning</h1>
<p class='notice'><b>Research use only.</b> Uses labeled sample data for parameter selection. nnU-Net predictions are cached once per case. No cross-validation or bootstrap is used in this version.</p>
<h2>Final parameters selected on all cases</h2><pre>{json.dumps(dict(final_params), indent=2)}</pre>
<h2>Top parameter sets</h2>{top.to_html(index=False, escape=False)}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def archive_prediction_cache(pred_dir: str | Path, archive_path: str | Path, cases: Optional[Sequence[SampleCase]] = None) -> Path:
    """Compress cached nnU-Net predictions into a reusable ZIP archive.

    Parameters
    ----------
    pred_dir:
        Folder containing prediction files named <case_id>.nii.gz or <case_id>.nii.
    archive_path:
        Destination .zip file, usually on Google Drive.
    cases:
        Optional case list. When supplied, only expected case predictions are archived.

    Returns
    -------
    Path to the archive.
    """
    import zipfile

    pred_dir = Path(pred_dir)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    if cases is None:
        files = sorted(list(pred_dir.glob("*.nii.gz")) + list(pred_dir.glob("*.nii")))
    else:
        files = []
        for case in cases:
            for suffix in (".nii.gz", ".nii"):
                p = pred_dir / f"{case.case_id}{suffix}"
                if p.exists():
                    files.append(p)
                    break
            else:
                raise FileNotFoundError(f"Missing prediction for {case.case_id} in {pred_dir}")

    if not files:
        raise RuntimeError(f"No prediction NIfTI files found in {pred_dir}")

    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        manifest = {
            "format": "OpenPlaque prediction cache",
            "n_predictions": len(files),
            "files": [p.name for p in files],
        }
        zf.writestr("prediction_cache_manifest.json", json.dumps(manifest, indent=2))
        for p in files:
            zf.write(p, arcname=p.name)
    tmp_path.replace(archive_path)
    return archive_path


def restore_prediction_cache_from_archive(archive_path: str | Path, pred_dir: str | Path, overwrite: bool = False) -> Path:
    """Restore cached nnU-Net predictions from a ZIP archive into pred_dir."""
    import zipfile

    archive_path = Path(archive_path)
    pred_dir = Path(pred_dir)
    if not archive_path.exists():
        raise FileNotFoundError(f"Prediction cache archive not found: {archive_path}")
    if pred_dir.exists() and overwrite:
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            if member.endswith("/") or member == "prediction_cache_manifest.json":
                continue
            name = Path(member).name
            if not (name.endswith(".nii.gz") or name.endswith(".nii")):
                continue
            target = pred_dir / name
            if target.exists() and not overwrite:
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return pred_dir


def ensure_prediction_cache_with_archive(
    cases: Sequence[SampleCase],
    pred_dir: str | Path,
    input_dir: str | Path,
    archive_path: str | Path,
    dataset_name_or_id: str = "Dataset001_CCTA_DHM",
    configuration: str = "3d_fullres",
    folds: Optional[Sequence[str | int]] = None,
    trainer: Optional[str] = None,
    plans: Optional[str] = None,
    overwrite_predictions: bool = False,
    overwrite_archive: bool = False,
) -> Path:
    """Ensure predictions exist, preferring an on-Drive compressed archive.

    Order of operations:
    1. If local pred_dir contains all expected predictions and overwrite_predictions=False, reuse it.
    2. Else if archive_path exists and overwrite_predictions=False, restore predictions from archive.
    3. Else run nnUNetv2_predict once for all cases.
    4. Save/refresh the compressed archive on Google Drive.
    """
    pred_dir = Path(pred_dir)
    archive_path = Path(archive_path)
    expected = [pred_dir / f"{c.case_id}.nii.gz" for c in cases]

    if pred_dir.exists() and all(p.exists() for p in expected) and not overwrite_predictions:
        print(f"Using local cached predictions in {pred_dir}")
        if overwrite_archive or not archive_path.exists():
            print(f"Writing prediction archive: {archive_path}")
            archive_prediction_cache(pred_dir, archive_path, cases=cases)
        return pred_dir

    if archive_path.exists() and not overwrite_predictions:
        print(f"Restoring predictions from archive: {archive_path}")
        restore_prediction_cache_from_archive(archive_path, pred_dir, overwrite=True)
        expected = [pred_dir / f"{c.case_id}.nii.gz" for c in cases]
        if all(p.exists() for p in expected):
            return pred_dir
        print("Archive restored, but some expected predictions are missing; regenerating predictions.")

    ensure_prediction_cache(
        cases=cases,
        pred_dir=pred_dir,
        input_dir=input_dir,
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        folds=folds,
        trainer=trainer,
        plans=plans,
        overwrite=True,
    )
    print(f"Writing prediction archive: {archive_path}")
    archive_prediction_cache(pred_dir, archive_path, cases=cases)
    return pred_dir
