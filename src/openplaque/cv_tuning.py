"""Cross-validation tuning for OpenPlaque boundary refinement.

This module selects boundary-refinement parameters using labeled DHM-style
nnU-Net data, e.g.::

    Dataset001_CCTA_DHM/
        imagesTr/case001_0000.nii.gz
        labelsTr/case001.nii.gz

It compares refined nnU-Net predictions against expert labels and returns a
DataFrame of metrics for every parameter candidate across cross-validation
folds.

Research use only. Not clinically validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .boundary import refine_plaque_mask

PLAQUE_LABEL = 2
VESSEL_LABEL = 1

DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "min_component_voxels": [1, 5, 10, 25, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None, 850, 1000],
    "low_hu_threshold": [None, -100],
    "erode_core": [False],
    "erosion_iterations": [1],
}

PARAM_COLUMNS = [
    "min_component_voxels",
    "lumen_distance_voxels",
    "high_hu_threshold",
    "low_hu_threshold",
    "erode_core",
    "erosion_iterations",
]


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    image_path: Path
    label_path: Path


def read_image(path: str | Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def case_id_from_image_name(path: str | Path) -> str:
    name = Path(path).name
    for suffix in ("_0000.nii.gz", "_0000.nii", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def collect_dhm_cases(dataset_dir: str | Path, limit: Optional[int] = None) -> list[DatasetCase]:
    """Collect paired imagesTr/labelsTr cases from a DHM/nnU-Net raw dataset."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected imagesTr and labelsTr under {dataset_dir}. "
            "Use a nnU-Net raw dataset folder such as Dataset001_CCTA_DHM."
        )
    image_paths = sorted(list(images_dir.glob("*_0000.nii.gz")) + list(images_dir.glob("*_0000.nii")))
    cases: list[DatasetCase] = []
    for image_path in image_paths:
        cid = case_id_from_image_name(image_path)
        label_path = labels_dir / f"{cid}.nii.gz"
        if not label_path.exists():
            alt = labels_dir / f"{cid}.nii"
            if alt.exists():
                label_path = alt
            else:
                continue
        cases.append(DatasetCase(cid, image_path, label_path))
    if limit is not None:
        cases = cases[: int(limit)]
    if not cases:
        raise RuntimeError(f"No paired images/labels found in {dataset_dir}")
    return cases


def make_kfold_splits(cases: Sequence[DatasetCase], n_splits: int = 5, seed: int = 17) -> list[list[DatasetCase]]:
    """Return validation-case lists for K folds."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cases))
    rng.shuffle(idx)
    folds_idx = np.array_split(idx, n_splits)
    return [[cases[int(i)] for i in fold] for fold in folds_idx]


def _iter_grid(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None):
    grid = dict(DEFAULT_GRID if parameter_grid is None else parameter_grid)
    for col in PARAM_COLUMNS:
        grid.setdefault(col, DEFAULT_GRID[col])
    for candidate_id, values in enumerate(product(*[grid[c] for c in PARAM_COLUMNS])):
        params = dict(zip(PARAM_COLUMNS, values))
        params["min_component_voxels"] = int(params["min_component_voxels"])
        params["lumen_distance_voxels"] = int(params["lumen_distance_voxels"])
        params["erode_core"] = bool(params["erode_core"])
        params["erosion_iterations"] = int(params["erosion_iterations"])
        yield candidate_id, params


def prediction_path_for_case(pred_dir: str | Path, case_id: str) -> Path:
    pred_dir = Path(pred_dir)
    for name in (f"{case_id}.nii.gz", f"{case_id}.nii"):
        p = pred_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing prediction for {case_id} in {pred_dir}")


def prepare_prediction_input(cases: Sequence[DatasetCase], input_dir: str | Path, overwrite: bool = True) -> Path:
    """Copy selected validation images into an nnUNetv2_predict input folder."""
    input_dir = Path(input_dir)
    if input_dir.exists() and overwrite:
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        target = input_dir / f"{case.case_id}_0000.nii.gz"
        shutil.copy2(case.image_path, target)
    return input_dir


def run_nnunet_predictions_for_fold(
    cases: Sequence[DatasetCase],
    fold: int,
    output_root: str | Path,
    dataset_name_or_id: str = "Dataset001_CCTA_DHM",
    configuration: str = "3d_fullres",
    trainer: Optional[str] = None,
    plans: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Run nnUNetv2_predict for one validation fold and return prediction dir."""
    output_root = Path(output_root)
    input_dir = output_root / f"fold_{fold}_input"
    pred_dir = output_root / f"fold_{fold}_pred"
    if pred_dir.exists() and list(pred_dir.glob("*.nii*")) and not overwrite:
        return pred_dir
    prepare_prediction_input(cases, input_dir, overwrite=True)
    if pred_dir.exists() and overwrite:
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(pred_dir),
        "-d", str(dataset_name_or_id),
        "-c", configuration,
        "-f", str(fold),
    ]
    if trainer:
        cmd += ["-tr", trainer]
    if plans:
        cmd += ["-p", plans]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return pred_dir


def mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, spacing=(1.0, 1.0, 1.0), label: int = PLAQUE_LABEL) -> dict[str, float]:
    pred = np.asarray(pred_mask) == label
    true = np.asarray(true_mask) == label
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    pred_n = int(pred.sum())
    true_n = int(true.sum())
    denom_dice = pred_n + true_n
    denom_iou = tp + fp + fn
    dice = 1.0 if denom_dice == 0 else (2.0 * tp / denom_dice)
    iou = 1.0 if denom_iou == 0 else (tp / denom_iou)
    precision = 1.0 if (tp + fp) == 0 else (tp / (tp + fp))
    recall = 1.0 if (tp + fn) == 0 else (tp / (tp + fn))
    voxel_vol = float(np.prod(spacing))
    pred_tpv = pred_n * voxel_vol
    true_tpv = true_n * voxel_vol
    if true_tpv == 0:
        abs_tpv_error_fraction = 0.0 if pred_tpv == 0 else 1.0
    else:
        abs_tpv_error_fraction = abs(pred_tpv - true_tpv) / true_tpv
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pred_tpv_mm3": float(pred_tpv),
        "true_tpv_mm3": float(true_tpv),
        "tpv_error_mm3": float(pred_tpv - true_tpv),
        "abs_tpv_error_fraction": float(abs_tpv_error_fraction),
        "pred_voxels": pred_n,
        "true_voxels": true_n,
    }


def score_metrics(metrics: Mapping[str, float]) -> float:
    """Composite score used to choose parameters; higher is better."""
    tpv_term = max(0.0, 1.0 - min(float(metrics["abs_tpv_error_fraction"]), 1.0))
    return float(
        0.35 * metrics["dice"]
        + 0.20 * metrics["iou"]
        + 0.20 * tpv_term
        + 0.15 * metrics["precision"]
        + 0.10 * metrics["recall"]
    )


def evaluate_case_parameter_grid(
    case: DatasetCase,
    prediction_path: str | Path,
    parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None,
) -> pd.DataFrame:
    """Evaluate every parameter set for one case with a prediction and label."""
    image, volume = read_image(case.image_path)
    label_img, label_arr = read_image(case.label_path)
    pred_img, pred_arr = read_image(prediction_path)
    spacing = tuple(float(x) for x in image.GetSpacing())
    rows = []
    for candidate_id, params in _iter_grid(parameter_grid):
        ref = refine_plaque_mask(
            volume=volume,
            mask=pred_arr,
            spacing=spacing,
            remove_small=True,
            min_component_voxels=params["min_component_voxels"],
            trim_lumen_adjacent=params["lumen_distance_voxels"] > 0,
            lumen_distance_voxels=params["lumen_distance_voxels"],
            erode_core=params["erode_core"],
            erosion_iterations=params["erosion_iterations"],
            high_hu_threshold=params["high_hu_threshold"],
            low_hu_threshold=params["low_hu_threshold"],
        )
        m = mask_metrics(ref.refined_mask, label_arr, spacing=spacing)
        row = {
            "case_id": case.case_id,
            "candidate_id": candidate_id,
            **params,
            **m,
            "score": score_metrics(m),
            "original_pred_tpv_mm3": ref.original_tpv_mm3,
            "refined_tpv_mm3": ref.refined_tpv_mm3,
            "removed_tpv_mm3": ref.removed_volume_mm3,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_fold_grid(
    fold: int,
    cases: Sequence[DatasetCase],
    pred_dir: str | Path,
    parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None,
) -> pd.DataFrame:
    frames = []
    for i, case in enumerate(cases, start=1):
        print(f"Fold {fold}: evaluating {case.case_id} ({i}/{len(cases)})")
        p = prediction_path_for_case(pred_dir, case.case_id)
        df = evaluate_case_parameter_grid(case, p, parameter_grid=parameter_grid)
        df.insert(0, "fold", fold)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate case-level CV results by parameter combination."""
    if cv_results.empty:
        raise ValueError("cv_results is empty")
    grouped = cv_results.groupby(PARAM_COLUMNS, dropna=False).agg(
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
    grouped["std_score"] = grouped["std_score"].fillna(0.0)
    grouped = grouped.sort_values(["mean_score", "mean_dice"], ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped


def select_best_parameters(cv_results: pd.DataFrame) -> dict[str, Any]:
    summary = aggregate_cv_results(cv_results)
    best = summary.iloc[0]
    params: dict[str, Any] = {}
    for col in PARAM_COLUMNS:
        val = best[col]
        if pd.isna(val):
            val = None
        elif col in ("min_component_voxels", "lumen_distance_voxels", "erosion_iterations"):
            val = int(val)
        elif col == "erode_core":
            val = bool(val)
        params[col] = val
    params["remove_small"] = True
    params["trim_lumen_adjacent"] = int(params["lumen_distance_voxels"]) > 0
    return params


def save_cv_outputs(cv_results: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_cv_results(cv_results)
    best_params = select_best_parameters(cv_results)
    paths = {
        "case_results_csv": output_dir / "cv_boundary_tuning_case_results.csv",
        "summary_csv": output_dir / "cv_boundary_tuning_summary.csv",
        "best_json": output_dir / "cv_best_boundary_parameters.json",
        "html": output_dir / "cv_boundary_tuning_report.html",
    }
    cv_results.to_csv(paths["case_results_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    payload = {
        "best_parameters": best_params,
        "best_summary_row": summary.iloc[0].to_dict(),
        "scoring": "0.35*Dice + 0.20*IoU + 0.20*(1-min(abs TPV error fraction,1)) + 0.15*Precision + 0.10*Recall",
        "research_use_only": True,
    }
    paths["best_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_cv_html_report(cv_results, summary, best_params, paths["html"])
    return paths


def write_cv_html_report(cv_results: pd.DataFrame, summary: pd.DataFrame, best_params: Mapping[str, Any], output_html: str | Path) -> Path:
    output_html = Path(output_html)
    top = summary.head(25).copy()
    for col in ["mean_score", "std_score", "mean_dice", "mean_iou", "mean_precision", "mean_recall", "mean_abs_tpv_error_fraction", "mean_tpv_error_mm3"]:
        top[col] = top[col].map(lambda x: f"{float(x):.4f}")
    best_json = json.dumps(dict(best_params), indent=2)
    fold_summary = cv_results.groupby("fold").agg(
        n_cases=("case_id", "nunique"),
        mean_score=("score", "mean"),
        mean_dice=("dice", "mean"),
        mean_abs_tpv_error_fraction=("abs_tpv_error_fraction", "mean"),
    ).reset_index()
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenPlaque CV Boundary Tuning</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ background: #f4f4f4; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
.notice {{ background: #fff3cd; border: 1px solid #ffe08a; padding: 12px 14px; border-radius: 8px; }}
</style></head><body>
<h1>OpenPlaque Cross-Validation Boundary Tuning</h1>
<p class='notice'><b>Research use only.</b> Parameters are selected on labeled sample data and should still be visually checked before use on new studies.</p>
<h2>Best parameters</h2><pre>{best_json}</pre>
<h2>Top parameter sets</h2>{top.to_html(index=False, escape=False)}
<h2>Fold summary</h2>{fold_summary.to_html(index=False, escape=False)}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
