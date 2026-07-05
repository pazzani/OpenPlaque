"""
OpenPlaque boundary refinement using physical minimum component volume.

This replaces absolute voxel-count filtering with a resolution-aware
minimum plaque-component volume in mm^3.

Research software only. Not for clinical use.
"""

from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi


def voxel_volume_mm3(spacing):
    return float(np.prod(spacing))


def min_volume_to_voxels(min_component_volume_mm3, spacing):
    if min_component_volume_mm3 is None or min_component_volume_mm3 <= 0:
        return 0
    return int(np.ceil(float(min_component_volume_mm3) / voxel_volume_mm3(spacing)))


def remove_small_components_by_volume(mask, spacing, min_component_volume_mm3=0.0, plaque_label=2):
    """
    Remove disconnected plaque components smaller than a physical volume threshold.

    This is preferable to a fixed voxel threshold because it adapts to voxel spacing.
    """
    min_voxels = min_volume_to_voxels(min_component_volume_mm3, spacing)
    if min_voxels <= 0:
        return mask.copy()

    plaque = mask == plaque_label
    labels, n = ndi.label(plaque)

    if n == 0:
        return mask.copy()

    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(plaque, dtype=bool)

    for label_id, size in enumerate(sizes):
        if label_id == 0:
            continue
        if size >= min_voxels:
            keep |= labels == label_id

    refined = mask.copy()
    refined[plaque & ~keep] = 0
    return refined


def lumen_adjacent_trim(mask, vessel_label=1, plaque_label=2, distance_voxels=0):
    if distance_voxels <= 0:
        return mask.copy()

    plaque = mask == plaque_label
    vessel = mask == vessel_label

    vessel_dilated = ndi.binary_dilation(vessel, iterations=distance_voxels)
    remove = plaque & vessel_dilated

    refined = mask.copy()
    refined[remove] = 0
    return refined


def refine_mask_volume_based(
    volume,
    mask,
    spacing,
    min_component_volume_mm3=0.0,
    lumen_distance_voxels=0,
    low_hu_threshold=None,
    high_hu_threshold=None,
):
    """
    Focused refinement:
      1. Remove very small plaque components by physical volume.
      2. Optional lumen-adjacent trim.
      3. Optional HU trimming.

    No erosion by default; the goal is to avoid over-aggressive removal.
    """
    refined = mask.copy()

    refined = remove_small_components_by_volume(
        refined,
        spacing=spacing,
        min_component_volume_mm3=min_component_volume_mm3,
        plaque_label=2,
    )

    refined = lumen_adjacent_trim(
        refined,
        vessel_label=1,
        plaque_label=2,
        distance_voxels=lumen_distance_voxels,
    )

    plaque = refined == 2
    remove = np.zeros_like(plaque, dtype=bool)

    if low_hu_threshold is not None:
        remove |= plaque & (volume < low_hu_threshold)
    if high_hu_threshold is not None:
        remove |= plaque & (volume > high_hu_threshold)

    refined[remove] = 0
    return refined


def plaque_volume(mask, spacing, plaque_label=2):
    return int(np.sum(mask == plaque_label)) * voxel_volume_mm3(spacing)


def binary_metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    return dict(tp=tp, fp=fp, fn=fn, tn=tn, dice=dice, iou=iou, precision=precision, recall=recall)


def score_row(row, volume_error_scale=1000.0):
    """
    Composite score with a guard against removing too much.

    The volume penalty discourages large TPV errors.
    The removal penalty discourages refinements that erase most predicted plaque.
    """
    metric_score = (
        0.40 * row["dice"]
        + 0.25 * row["iou"]
        + 0.20 * row["precision"]
        + 0.15 * row["recall"]
    )
    volume_penalty = 0.10 * (row["abs_volume_error_mm3"] / volume_error_scale)

    # Penalize erasing most predicted plaque, especially all-plaque removal.
    removal_fraction = row.get("removed_fraction", 0.0)
    removal_penalty = 0.10 * max(0.0, removal_fraction - 0.60)

    if row.get("pred_plaque_voxels", 0) > 0 and row.get("refined_plaque_voxels", 0) == 0:
        removal_penalty += 0.25

    return metric_score - volume_penalty - removal_penalty


def list_sample_cases(sample_root):
    sample_root = Path(sample_root)
    image_dir = sample_root / "Images"
    label_dir = sample_root / "Labels"

    cases = []
    for img in sorted(image_dir.glob("*_0000.nii.gz")):
        case = img.name.replace("_0000.nii.gz", "")
        label = label_dir / f"{case}.nii.gz"
        if label.exists():
            cases.append(dict(case=case, image=img, label=label))
    return cases


def read_image(path):
    img = sitk.ReadImage(str(path))
    return img, sitk.GetArrayFromImage(img)


def run_nnunet_case(
    image_path,
    pred_dir,
    case_name,
    dataset="Dataset001_CCTA_DHM",
    configuration="3d_fullres",
    folds=(0, 1, 2, 3, 4),
    overwrite=False,
):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    out_file = pred_dir / f"{case_name}.nii.gz"
    if out_file.exists() and not overwrite:
        return out_file

    input_dir = pred_dir / f"_input_{case_name}"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, input_dir / f"{case_name}_0000.nii.gz")

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(pred_dir),
        "-d", dataset,
        "-c", configuration,
        "-f",
    ] + [str(f) for f in folds]

    subprocess.run(cmd, check=True)
    return out_file


def focused_parameter_grid(
    min_component_volume_mm3=(0, 1, 2, 3, 5, 8, 10, 15, 20),
    lumen_distance_voxels=(0, 1),
    high_hu_threshold=(None,),
    low_hu_threshold=(None,),
):
    grid = []
    for min_vol in min_component_volume_mm3:
        for lumen_dist in lumen_distance_voxels:
            for high_hu in high_hu_threshold:
                for low_hu in low_hu_threshold:
                    grid.append(dict(
                        min_component_volume_mm3=min_vol,
                        lumen_distance_voxels=lumen_dist,
                        low_hu_threshold=low_hu,
                        high_hu_threshold=high_hu,
                    ))
    return grid


def evaluate_case(volume, pred, truth, spacing, params):
    refined = refine_mask_volume_based(volume, pred, spacing, **params)

    metrics = binary_metrics(refined == 2, truth == 2)

    raw_tpv = plaque_volume(pred, spacing)
    refined_tpv = plaque_volume(refined, spacing)
    truth_tpv = plaque_volume(truth, spacing)

    pred_voxels = int(np.sum(pred == 2))
    refined_voxels = int(np.sum(refined == 2))
    removed_voxels = max(0, pred_voxels - refined_voxels)
    removed_fraction = removed_voxels / pred_voxels if pred_voxels else 0.0

    row = dict(metrics)
    row.update(params)
    row.update(dict(
        raw_tpv_mm3=raw_tpv,
        refined_tpv_mm3=refined_tpv,
        truth_tpv_mm3=truth_tpv,
        abs_volume_error_mm3=abs(refined_tpv - truth_tpv),
        raw_abs_volume_error_mm3=abs(raw_tpv - truth_tpv),
        pred_plaque_voxels=pred_voxels,
        refined_plaque_voxels=refined_voxels,
        removed_voxels=removed_voxels,
        removed_fraction=removed_fraction,
        min_component_voxels=min_volume_to_voxels(params["min_component_volume_mm3"], spacing),
    ))
    row["score"] = score_row(row)
    return row


def tune_volume_based_refinement(
    sample_root,
    pred_dir="/content/openplaque_predictions",
    max_cases=None,
    grid=None,
    overwrite_predictions=False,
):
    cases = list_sample_cases(sample_root)
    if max_cases is not None:
        cases = cases[:max_cases]
    if grid is None:
        grid = focused_parameter_grid()

    rows = []

    for i, case in enumerate(cases, start=1):
        case_name = case["case"]
        print(f"[{i}/{len(cases)}] {case_name}")

        img, volume = read_image(case["image"])
        truth_img, truth = read_image(case["label"])
        spacing = img.GetSpacing()

        pred_path = run_nnunet_case(
            image_path=case["image"],
            pred_dir=pred_dir,
            case_name=case_name,
            overwrite=overwrite_predictions,
        )
        pred_img, pred = read_image(pred_path)

        for params in grid:
            row = evaluate_case(volume, pred, truth, spacing, params)
            row["case"] = case_name
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_volume_tuning(df):
    group_cols = [
        "min_component_volume_mm3",
        "lumen_distance_voxels",
        "low_hu_threshold",
        "high_hu_threshold",
    ]

    return (
        df.groupby(group_cols, dropna=False)
          .agg(
              mean_dice=("dice", "mean"),
              median_dice=("dice", "median"),
              mean_iou=("iou", "mean"),
              mean_precision=("precision", "mean"),
              mean_recall=("recall", "mean"),
              mean_abs_volume_error_mm3=("abs_volume_error_mm3", "mean"),
              median_abs_volume_error_mm3=("abs_volume_error_mm3", "median"),
              mean_raw_abs_volume_error_mm3=("raw_abs_volume_error_mm3", "mean"),
              mean_removed_fraction=("removed_fraction", "mean"),
              zeroed_cases=("refined_plaque_voxels", lambda x: int(np.sum(np.asarray(x) == 0))),
              mean_score=("score", "mean"),
              n_cases=("case", "nunique"),
          )
          .reset_index()
          .sort_values("mean_score", ascending=False)
    )
