"""5-fold supervised CV boundary-parameter selection for OpenPlaque.

Supports the old OpenPlaque Sample_Dataset layout:

    P02_LAD_axial_0000.nii.gz   # CT image / nnU-Net channel 0
    P02_LAD_axial.nii.gz        # expert label mask: 0 background, 1 vessel, 2 plaque

and nnU-Net raw dataset layout:

    Dataset001_CCTA_DHM/imagesTr/*_0000.nii.gz
    Dataset001_CCTA_DHM/labelsTr/*.nii.gz

The workflow:
1. Collect paired image/label cases.
2. Generate or reuse nnU-Net predictions for all cases.
3. Split cases into K folds.
4. For each fold, choose parameters using the other K-1 folds.
5. Evaluate those chosen parameters on the held-out fold.
6. Also select a final global parameter set using all cases.

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
PARAM_COLUMNS = [
    "min_component_voxels",
    "lumen_distance_voxels",
    "high_hu_threshold",
    "low_hu_threshold",
    "erode_core",
    "erosion_iterations",
]

DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "min_component_voxels": [0, 5, 10, 20, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None],
    "low_hu_threshold": [None],
    "erode_core": [False],
    "erosion_iterations": [0, 1],
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
    # Also search recursively for exact label filename; useful when Sample_Dataset has nested patient dirs.
    try:
        candidates.extend(dataset_dir.rglob(f"{cid}.nii.gz"))
        candidates.extend(dataset_dir.rglob(f"{cid}.nii"))
    except Exception:
        pass
    # De-duplicate preserving order.
    seen, out = set(), []
    for p in candidates:
        p = Path(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def collect_sample_cases(dataset_dir: str | Path, limit: Optional[int] = None) -> list[SampleCase]:
    """Find paired *_0000 image files and matching label masks."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    image_paths = []
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

    # De-duplicate by case_id.
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


def make_kfold_assignments(cases: Sequence[SampleCase], n_splits: int = 5, seed: int = 17) -> pd.DataFrame:
    if len(cases) < 2:
        raise ValueError("Need at least 2 cases for cross-validation")
    n_splits = min(int(n_splits), len(cases))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(cases))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    rows = []
    for fold_id, idxs in enumerate(folds):
        for idx in idxs:
            rows.append({"case_id": cases[int(idx)].case_id, "fold": fold_id})
    return pd.DataFrame(rows).sort_values(["fold", "case_id"]).reset_index(drop=True)


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
        target = input_dir / f"{c.case_id}_0000.nii.gz"
        shutil.copy2(c.image_path, target)
    return input_dir


def run_nnunet_predictions(
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


def prediction_path(pred_dir: str | Path, case_id: str) -> Path:
    pred_dir = Path(pred_dir)
    for suffix in (".nii.gz", ".nii"):
        p = pred_dir / f"{case_id}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Prediction not found for {case_id} in {pred_dir}")


def iter_parameter_grid(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None):
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
        "dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall),
        "pred_tpv_mm3": float(pred_tpv), "true_tpv_mm3": float(true_tpv),
        "tpv_error_mm3": float(pred_tpv - true_tpv), "abs_tpv_error_fraction": float(abs_tpv_err_frac),
        "pred_voxels": pred_n, "true_voxels": true_n,
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
            volume=vol, mask=pred, spacing=spacing,
            remove_small=params["min_component_voxels"] > 0,
            min_component_voxels=max(1, params["min_component_voxels"]),
            trim_lumen_adjacent=params["lumen_distance_voxels"] > 0,
            lumen_distance_voxels=params["lumen_distance_voxels"],
            erode_core=params["erode_core"] or params["erosion_iterations"] > 0,
            erosion_iterations=params["erosion_iterations"],
            high_hu_threshold=params["high_hu_threshold"],
            low_hu_threshold=params["low_hu_threshold"],
        )
        m = mask_metrics(ref.refined_mask, label, spacing=spacing)
        rows.append({
            "case_id": case.case_id, "candidate_id": candidate_id, **params, **m,
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
        mean_score=("score", "mean"), std_score=("score", "std"),
        mean_dice=("dice", "mean"), mean_iou=("iou", "mean"),
        mean_precision=("precision", "mean"), mean_recall=("recall", "mean"),
        mean_abs_tpv_error_fraction=("abs_tpv_error_fraction", "mean"),
        mean_tpv_error_mm3=("tpv_error_mm3", "mean"),
        n_cases=("case_id", "nunique"), n_rows=("case_id", "count"),
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
        elif col in ("min_component_voxels", "lumen_distance_voxels", "erosion_iterations"):
            val = int(val)
        elif col == "erode_core":
            val = bool(val)
        params[col] = val
    params["remove_small"] = int(params["min_component_voxels"]) > 0
    params["trim_lumen_adjacent"] = int(params["lumen_distance_voxels"]) > 0
    return params


def select_best_params(rows: pd.DataFrame) -> dict[str, Any]:
    return params_from_summary_row(aggregate_by_params(rows).iloc[0])


def add_fold_assignments(rows: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(assignments, on="case_id", how="left")


def run_5fold_parameter_cv(all_case_results: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """True parameter-selection CV: tune on K-1 folds, test selected params on held-out fold."""
    df = add_fold_assignments(all_case_results, assignments)
    fold_rows = []
    selected_rows = []
    n_folds = int(assignments["fold"].nunique())
    for fold in sorted(assignments["fold"].unique()):
        train = df[df["fold"] != fold]
        val = df[df["fold"] == fold]
        train_summary = aggregate_by_params(train)
        params = params_from_summary_row(train_summary.iloc[0])
        selected_rows.append({"fold": int(fold), **{k: params.get(k) for k in PARAM_COLUMNS}, "train_mean_score": float(train_summary.iloc[0]["mean_score"])})
        # Match val rows with selected parameter values. Need null-safe comparisons.
        mask = pd.Series(True, index=val.index)
        for col in PARAM_COLUMNS:
            target = params[col]
            if target is None:
                mask &= val[col].isna()
            else:
                mask &= (val[col] == target)
        held = val[mask].copy()
        held["selected_by_training_fold"] = int(fold)
        fold_rows.append(held)
    heldout = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    final_params = select_best_params(all_case_results)
    return heldout, selected, final_params


def save_cv_outputs(
    all_case_results: pd.DataFrame,
    assignments: pd.DataFrame,
    heldout_results: pd.DataFrame,
    selected_by_fold: pd.DataFrame,
    final_params: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_summary = aggregate_by_params(all_case_results)
    heldout_summary = heldout_results.groupby("selected_by_training_fold").agg(
        n_cases=("case_id", "nunique"), mean_score=("score", "mean"), mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"), mean_abs_tpv_error_fraction=("abs_tpv_error_fraction", "mean"),
    ).reset_index() if not heldout_results.empty else pd.DataFrame()
    paths = {
        "case_results": output_dir / "cv_all_case_parameter_results.csv",
        "full_summary": output_dir / "cv_full_dataset_parameter_summary.csv",
        "fold_assignments": output_dir / "cv_fold_assignments.csv",
        "heldout_results": output_dir / "cv_heldout_selected_parameter_results.csv",
        "selected_by_fold": output_dir / "cv_selected_parameters_by_fold.csv",
        "best_json": output_dir / "cv_best_boundary_parameters.json",
        "html": output_dir / "cv_boundary_tuning_report.html",
    }
    all_case_results.to_csv(paths["case_results"], index=False)
    full_summary.to_csv(paths["full_summary"], index=False)
    assignments.to_csv(paths["fold_assignments"], index=False)
    heldout_results.to_csv(paths["heldout_results"], index=False)
    selected_by_fold.to_csv(paths["selected_by_fold"], index=False)
    payload = {
        "final_parameters_selected_on_all_cases": dict(final_params),
        "heldout_cv_mean_score": None if heldout_results.empty else float(heldout_results["score"].mean()),
        "heldout_cv_mean_dice": None if heldout_results.empty else float(heldout_results["dice"].mean()),
        "heldout_cv_mean_abs_tpv_error_fraction": None if heldout_results.empty else float(heldout_results["abs_tpv_error_fraction"].mean()),
        "scoring": "0.35*Dice + 0.20*IoU + 0.20*(1-min(abs TPV error fraction,1)) + 0.15*Precision + 0.10*Recall",
        "research_use_only": True,
    }
    paths["best_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_cv_html(paths["html"], full_summary, heldout_summary, selected_by_fold, final_params)
    return paths


def write_cv_html(output_html: str | Path, full_summary: pd.DataFrame, heldout_summary: pd.DataFrame, selected_by_fold: pd.DataFrame, final_params: Mapping[str, Any]) -> Path:
    output_html = Path(output_html)
    top = full_summary.head(25).copy()
    for df in (top, heldout_summary):
        if df is not None and not df.empty:
            for col in df.columns:
                if str(col).startswith("mean_") or str(col).startswith("std_"):
                    df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>OpenPlaque 5-Fold CV Boundary Tuning</title>
<style>body {{font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color:#222;}}
table {{border-collapse: collapse; width:100%; margin: 16px 0 28px; font-size: 13px;}}
th,td {{border:1px solid #ddd; padding:6px 8px; text-align:right;}} th:first-child,td:first-child {{text-align:left;}} th {{background:#f4f4f4;}}
pre {{background:#f7f7f7; padding:12px; border-radius:8px; overflow-x:auto;}}
.notice {{background:#fff3cd; border:1px solid #ffe08a; padding:12px 14px; border-radius:8px;}}</style></head><body>
<h1>OpenPlaque 5-Fold CV Boundary Tuning</h1>
<p class='notice'><b>Research use only.</b> Uses labeled sample data for parameter selection. Still requires visual QC and expert review.</p>
<h2>Final parameters selected on all cases</h2><pre>{json.dumps(dict(final_params), indent=2)}</pre>
<h2>Held-out CV summary</h2>{heldout_summary.to_html(index=False, escape=False) if heldout_summary is not None and not heldout_summary.empty else '<p>No held-out results.</p>'}
<h2>Parameters selected inside each fold</h2>{selected_by_fold.to_html(index=False, escape=False)}
<h2>Top full-dataset parameter sets</h2>{top.to_html(index=False, escape=False)}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
