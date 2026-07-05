
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi


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


def plaque_volume_mm3(mask, spacing, plaque_label=2):
    return int(np.sum(mask == plaque_label)) * float(np.prod(spacing))


def remove_small_components(mask, min_voxels=10, plaque_label=2):
    plaque = mask == plaque_label
    labels, n = ndi.label(plaque)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(plaque, dtype=bool)
    for label_id, size in enumerate(sizes):
        if label_id != 0 and size >= min_voxels:
            keep |= labels == label_id
    refined = mask.copy()
    refined[plaque & ~keep] = 0
    return refined


def lumen_adjacent_trim(mask, vessel_label=1, plaque_label=2, distance_voxels=1):
    if distance_voxels <= 0:
        return mask.copy()
    plaque = mask == plaque_label
    vessel = mask == vessel_label
    vessel_dilated = ndi.binary_dilation(vessel, iterations=distance_voxels)
    refined = mask.copy()
    refined[plaque & vessel_dilated] = 0
    return refined


def erode_plaque(mask, plaque_label=2, iterations=1):
    if iterations <= 0:
        return mask.copy()
    plaque = mask == plaque_label
    core = ndi.binary_erosion(plaque, iterations=iterations)
    refined = mask.copy()
    refined[plaque & ~core] = 0
    return refined


def intensity_trim(volume, mask, plaque_label=2, low_hu_threshold=None, high_hu_threshold=None):
    plaque = mask == plaque_label
    remove = np.zeros_like(plaque, dtype=bool)
    if low_hu_threshold is not None:
        remove |= plaque & (volume < low_hu_threshold)
    if high_hu_threshold is not None:
        remove |= plaque & (volume > high_hu_threshold)
    refined = mask.copy()
    refined[remove] = 0
    return refined


def refine_mask(volume, mask, min_component_voxels=0, lumen_distance_voxels=0,
                erosion_iterations=0, low_hu_threshold=None, high_hu_threshold=None):
    refined = mask.copy()
    if min_component_voxels and min_component_voxels > 0:
        refined = remove_small_components(refined, min_voxels=min_component_voxels)
    if lumen_distance_voxels and lumen_distance_voxels > 0:
        refined = lumen_adjacent_trim(refined, distance_voxels=lumen_distance_voxels)
    if low_hu_threshold is not None or high_hu_threshold is not None:
        refined = intensity_trim(volume, refined, low_hu_threshold=low_hu_threshold,
                                 high_hu_threshold=high_hu_threshold)
    if erosion_iterations and erosion_iterations > 0:
        refined = erode_plaque(refined, iterations=erosion_iterations)
    return refined


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


def run_nnunet_case(image_path, pred_dir, case_name,
                    dataset="Dataset001_CCTA_DHM", configuration="3d_fullres",
                    folds=(0, 1, 2, 3, 4), overwrite=False):
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
    cmd = ["nnUNetv2_predict", "-i", str(input_dir), "-o", str(pred_dir),
           "-d", dataset, "-c", configuration, "-f"] + [str(f) for f in folds]
    subprocess.run(cmd, check=True)
    return out_file


def parameter_grid(min_component_voxels=(0, 5, 10, 20, 50),
                   lumen_distance_voxels=(0, 1, 2),
                   erosion_iterations=(0, 1),
                   high_hu_threshold=(None,),
                   low_hu_threshold=(None,)):
    grid = []
    for min_comp in min_component_voxels:
        for lumen_dist in lumen_distance_voxels:
            for erosion in erosion_iterations:
                for high_hu in high_hu_threshold:
                    for low_hu in low_hu_threshold:
                        grid.append(dict(min_component_voxels=min_comp,
                                         lumen_distance_voxels=lumen_dist,
                                         erosion_iterations=erosion,
                                         low_hu_threshold=low_hu,
                                         high_hu_threshold=high_hu))
    return grid


def evaluate_prediction(volume, pred_mask, truth_mask, spacing, params):
    refined = refine_mask(volume, pred_mask, **params)
    row = binary_metrics(refined == 2, truth_mask == 2)
    row["pred_tpv_mm3"] = plaque_volume_mm3(refined, spacing)
    row["truth_tpv_mm3"] = plaque_volume_mm3(truth_mask, spacing)
    row["abs_volume_error_mm3"] = abs(row["pred_tpv_mm3"] - row["truth_tpv_mm3"])
    row.update(params)
    return row


def score_row(row, volume_error_scale=1000.0):
    metric_score = 0.40 * row["dice"] + 0.25 * row["iou"] + 0.20 * row["precision"] + 0.15 * row["recall"]
    penalty = 0.10 * (row["abs_volume_error_mm3"] / volume_error_scale)
    return metric_score - penalty


def tune_boundary_refinement(sample_root, pred_dir="/content/openplaque_predictions",
                             max_cases=None, grid=None, overwrite_predictions=False):
    cases = list_sample_cases(sample_root)
    if max_cases is not None:
        cases = cases[:max_cases]
    if grid is None:
        grid = parameter_grid()
    rows = []
    for i, case in enumerate(cases, start=1):
        case_name = case["case"]
        print(f"[{i}/{len(cases)}] {case_name}")
        image_img, volume = read_image(case["image"])
        label_img, truth = read_image(case["label"])
        spacing = image_img.GetSpacing()
        pred_path = run_nnunet_case(case["image"], pred_dir, case_name, overwrite=overwrite_predictions)
        pred_img, pred = read_image(pred_path)
        for params in grid:
            row = evaluate_prediction(volume, pred, truth, spacing, params)
            row["case"] = case_name
            rows.append(row)
    df = pd.DataFrame(rows)
    df["score"] = df.apply(score_row, axis=1)
    return df


def summarize_tuning(df):
    group_cols = ["min_component_voxels", "lumen_distance_voxels",
                  "erosion_iterations", "low_hu_threshold", "high_hu_threshold"]
    return (df.groupby(group_cols, dropna=False)
              .agg(mean_dice=("dice", "mean"),
                   median_dice=("dice", "median"),
                   mean_iou=("iou", "mean"),
                   mean_precision=("precision", "mean"),
                   mean_recall=("recall", "mean"),
                   mean_abs_volume_error_mm3=("abs_volume_error_mm3", "mean"),
                   median_abs_volume_error_mm3=("abs_volume_error_mm3", "median"),
                   mean_score=("score", "mean"),
                   n_cases=("case", "nunique"))
              .reset_index()
              .sort_values("mean_score", ascending=False))
