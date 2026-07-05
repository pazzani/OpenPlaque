"""Working boundary-refinement parameter tuning for OpenPlaque.

This module runs a real grid search over ``refine_plaque_mask`` settings and
returns a pandas DataFrame named by the notebook as ``tuning_results``.

It is deliberately unsupervised: it does not claim clinical optimality. The
score favors modest cleanup, preservation of plaque volume, fewer fragments,
and a stable high-confidence core. Visual QC is still required.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Any
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .boundary import refine_plaque_mask

DEFAULT_PARAMETER_GRID: dict[str, Sequence[Any]] = {
    "min_component_voxels": [1, 5, 10, 25, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None, 700, 850, 1000],
    "low_hu_threshold": [None, -100, -50],
    "erode_core": [False],
    "erosion_iterations": [1],
}

CORE_PARAMETER_GRID: dict[str, Sequence[Any]] = {
    "min_component_voxels": [10],
    "lumen_distance_voxels": [1],
    "high_hu_threshold": [None],
    "low_hu_threshold": [None],
    "erode_core": [True],
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


def _spacing(report) -> tuple[float, float, float]:
    if hasattr(report, "mask_image") and report.mask_image is not None:
        return tuple(float(x) for x in report.mask_image.GetSpacing())
    if hasattr(report, "image") and report.image is not None:
        return tuple(float(x) for x in report.image.GetSpacing())
    raise AttributeError("Report must expose mask_image.GetSpacing() or image.GetSpacing().")


def _raw_tpv(report) -> float:
    if hasattr(report, "tpv_mm3"):
        return float(report.tpv_mm3)
    vox = int(np.sum(np.asarray(report.mask) == 2))
    return vox * float(np.prod(_spacing(report)))


def _plaque_components(mask: np.ndarray, plaque_label: int = 2) -> dict[str, float | int]:
    plaque = np.asarray(mask) == plaque_label
    vox = int(plaque.sum())
    if vox == 0:
        return {"component_count": 0, "largest_component_fraction": 0.0, "small_component_count": 0}
    labels, n = ndi.label(plaque)
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    small = int(np.sum(sizes < 10)) if sizes.size else 0
    return {
        "component_count": int(n),
        "largest_component_fraction": float(largest / vox) if vox else 0.0,
        "small_component_count": small,
    }


def _surface_voxels(mask: np.ndarray, plaque_label: int = 2) -> int:
    plaque = np.asarray(mask) == plaque_label
    if not plaque.any():
        return 0
    eroded = ndi.binary_erosion(plaque, iterations=1, border_value=0)
    return int((plaque & ~eroded).sum())


def _hu_stats(volume: np.ndarray, mask: np.ndarray, plaque_label: int = 2) -> dict[str, Optional[float]]:
    vals = np.asarray(volume)[np.asarray(mask) == plaque_label]
    if vals.size == 0:
        return {"mean_hu": None, "median_hu": None, "p10_hu": None, "p90_hu": None}
    return {
        "mean_hu": float(np.mean(vals)),
        "median_hu": float(np.median(vals)),
        "p10_hu": float(np.percentile(vals, 10)),
        "p90_hu": float(np.percentile(vals, 90)),
    }


def _iter_grid(parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None):
    grid = dict(DEFAULT_PARAMETER_GRID if parameter_grid is None else parameter_grid)
    for col in PARAM_COLUMNS:
        grid.setdefault(col, DEFAULT_PARAMETER_GRID[col])
    keys = list(PARAM_COLUMNS)
    for candidate_id, vals in enumerate(product(*[grid[k] for k in keys])):
        params = dict(zip(keys, vals))
        params["min_component_voxels"] = int(params["min_component_voxels"])
        params["lumen_distance_voxels"] = int(params["lumen_distance_voxels"])
        params["erode_core"] = bool(params["erode_core"])
        params["erosion_iterations"] = int(params["erosion_iterations"])
        yield candidate_id, params


def score_refinement(row: Mapping[str, Any], target_removed_fraction: float = 0.15) -> tuple[float, str]:
    """Return a heuristic score and reject reason for one candidate."""
    retained = float(row["retained_fraction"])
    removed = float(row["removed_fraction"])
    score = 100.0
    reject: list[str] = []

    if retained <= 0:
        reject.append("deleted_all_plaque")
    if retained < 0.35:
        reject.append("too_aggressive_retained_lt_35pct")
    if retained > 0.995:
        reject.append("no_effect_retained_gt_99_5pct")

    score -= 180.0 * abs(removed - target_removed_fraction)
    score += 15.0 * float(row.get("largest_component_fraction_after", 0.0))

    before = max(int(row.get("component_count_before", 0)), 1)
    after = int(row.get("component_count_after", 0))
    component_reduction = max(0.0, (before - after) / before)
    score += 20.0 * component_reduction

    small_before = int(row.get("small_component_count_before", 0))
    small_after = int(row.get("small_component_count_after", 0))
    if small_before > 0:
        score += 10.0 * max(0.0, (small_before - small_after) / small_before)

    # Surface/volume sanity: very ragged masks tend to have high surface ratios.
    surf_ratio = float(row.get("surface_voxel_ratio_after", 0.0))
    if surf_ratio > 0:
        score -= 3.0 * min(surf_ratio, 10.0)

    mean_hu = row.get("mean_hu")
    if mean_hu is not None and not pd.isna(mean_hu):
        if float(mean_hu) < -150 or float(mean_hu) > 1200:
            score -= 20.0
            reject.append("implausible_mean_hu")

    if bool(row.get("erode_core", False)):
        score -= 15.0

    if reject:
        score -= 1000.0
    return float(score), ";".join(reject)


def tune_boundary_for_report(report, parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> pd.DataFrame:
    """Tune one vessel report and return one row per parameter combination."""
    name = getattr(report, "name", "vessel")
    volume = np.asarray(report.volume)
    mask = np.asarray(report.mask)
    spacing = _spacing(report)
    raw_tpv = _raw_tpv(report)
    voxel_volume_mm3 = float(np.prod(spacing))
    before = _plaque_components(mask)
    surface_before = _surface_voxels(mask)
    plaque_voxels_before = int(np.sum(mask == 2))

    rows: list[dict[str, Any]] = []
    for candidate_id, params in _iter_grid(parameter_grid):
        ref = refine_plaque_mask(
            volume=volume,
            mask=mask,
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
        refined_mask = np.asarray(ref.refined_mask)
        after = _plaque_components(refined_mask)
        refined_tpv = float(ref.refined_tpv_mm3)
        retained = refined_tpv / raw_tpv if raw_tpv > 0 else 0.0
        surface_after = _surface_voxels(refined_mask)
        plaque_voxels_after = int(np.sum(refined_mask == 2))
        row = {
            "vessel": name,
            "candidate_id": candidate_id,
            **params,
            "raw_tpv_mm3": raw_tpv,
            "refined_tpv_mm3": refined_tpv,
            "removed_mm3": float(ref.removed_volume_mm3),
            "retained_fraction": float(retained),
            "removed_fraction": float(1.0 - retained),
            "plaque_voxels_before": plaque_voxels_before,
            "plaque_voxels_after": plaque_voxels_after,
            "voxel_volume_mm3": voxel_volume_mm3,
            "component_count_before": before["component_count"],
            "component_count_after": after["component_count"],
            "largest_component_fraction_before": before["largest_component_fraction"],
            "largest_component_fraction_after": after["largest_component_fraction"],
            "small_component_count_before": before["small_component_count"],
            "small_component_count_after": after["small_component_count"],
            "surface_voxels_before": surface_before,
            "surface_voxels_after": surface_after,
            "surface_voxel_ratio_after": float(surface_after / max(plaque_voxels_after, 1)),
            **_hu_stats(volume, refined_mask),
        }
        row["score"], row["reject_reason"] = score_refinement(row)
        rows.append(row)
    return pd.DataFrame(rows)


def tune_boundary_parameters(reports: Iterable, parameter_grid: Optional[Mapping[str, Sequence[Any]]] = None) -> pd.DataFrame:
    """Tune all vessel reports and return a DataFrame called ``tuning_results`` in notebooks."""
    frames = [tune_boundary_for_report(r, parameter_grid=parameter_grid) for r in reports]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["score", "vessel"], ascending=[False, True]).reset_index(drop=True)


def best_parameters(tuning_results: pd.DataFrame, require_all_vessels: bool = True) -> dict[str, Any]:
    """Pick a single global parameter set by summing score across vessels."""
    if tuning_results is None or tuning_results.empty:
        raise ValueError("tuning_results is empty. Run tune_boundary_parameters(reports) first.")
    df = tuning_results.copy()
    # Prefer non-rejected candidates if any exist.
    valid = df[df["reject_reason"].fillna("") == ""]
    if not valid.empty:
        df = valid
    group_cols = PARAM_COLUMNS
    grouped = df.groupby(group_cols, dropna=False).agg(
        total_score=("score", "sum"),
        mean_score=("score", "mean"),
        vessels=("vessel", "nunique"),
        mean_retained_fraction=("retained_fraction", "mean"),
        mean_removed_fraction=("removed_fraction", "mean"),
        mean_refined_tpv_mm3=("refined_tpv_mm3", "mean"),
    ).reset_index()
    if require_all_vessels:
        n_vessels = tuning_results["vessel"].nunique()
        complete = grouped[grouped["vessels"] == n_vessels]
        if not complete.empty:
            grouped = complete
    best = grouped.sort_values("total_score", ascending=False).iloc[0]
    out = {col: (None if pd.isna(best[col]) else best[col].item() if hasattr(best[col], "item") else best[col]) for col in group_cols}
    out["remove_small"] = True
    out["trim_lumen_adjacent"] = int(out["lumen_distance_voxels"]) > 0
    out["min_component_voxels"] = int(out["min_component_voxels"])
    out["lumen_distance_voxels"] = int(out["lumen_distance_voxels"])
    out["erode_core"] = bool(out["erode_core"])
    out["erosion_iterations"] = int(out["erosion_iterations"])
    return out


def best_rows_by_vessel(tuning_results: pd.DataFrame) -> pd.DataFrame:
    if tuning_results is None or tuning_results.empty:
        raise ValueError("tuning_results is empty. Run tune_boundary_parameters(reports) first.")
    df = tuning_results.copy()
    valid = df[df["reject_reason"].fillna("") == ""]
    if not valid.empty:
        df = valid
    idx = df.groupby("vessel")["score"].idxmax()
    return df.loc[idx].sort_values("vessel").reset_index(drop=True)


def apply_refinement_with_params(reports: Iterable, params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for report in reports:
        out[getattr(report, "name", "vessel")] = refine_plaque_mask(
            volume=np.asarray(report.volume),
            mask=np.asarray(report.mask),
            spacing=_spacing(report),
            **dict(params),
        )
    return out


def save_tuning_outputs(
    tuning_results: pd.DataFrame,
    output_dir: str | Path,
    selected_params: Optional[Mapping[str, Any]] = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected_params is None:
        selected_params = best_parameters(tuning_results)
    paths = {
        "csv": output_dir / "boundary_tuning_results.csv",
        "json": output_dir / "best_boundary_parameters.json",
        "html": output_dir / "boundary_tuning_report.html",
    }
    tuning_results.to_csv(paths["csv"], index=False)
    payload = {
        "selected_params": dict(selected_params),
        "best_by_vessel": best_rows_by_vessel(tuning_results).to_dict(orient="records"),
        "note": "Unsupervised heuristic boundary tuning; visual QC required; research use only.",
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tuning_html_report(tuning_results, paths["html"], selected_params=selected_params)
    return paths


def write_tuning_html_report(
    tuning_results: pd.DataFrame,
    output_html: str | Path,
    selected_params: Optional[Mapping[str, Any]] = None,
    title: str = "OpenPlaque Boundary Refinement Parameter Tuning",
) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if selected_params is None:
        selected_params = best_parameters(tuning_results)
    best_vessel = best_rows_by_vessel(tuning_results)

    display_cols = [
        "vessel", "candidate_id", "score", "min_component_voxels", "lumen_distance_voxels",
        "high_hu_threshold", "low_hu_threshold", "erode_core", "raw_tpv_mm3",
        "refined_tpv_mm3", "removed_fraction", "component_count_before",
        "component_count_after", "largest_component_fraction_after", "mean_hu", "reject_reason",
    ]
    all_rows = tuning_results.sort_values(["vessel", "score"], ascending=[True, False])[display_cols].copy()
    for col in ["raw_tpv_mm3", "refined_tpv_mm3", "score", "removed_fraction", "largest_component_fraction_after", "mean_hu"]:
        if col in all_rows:
            all_rows[col] = all_rows[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    best_html = best_vessel[display_cols].to_html(index=False, escape=False)
    all_html = all_rows.to_html(index=False, escape=False)
    params_html = json.dumps(dict(selected_params), indent=2)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #f4f4f4; position: sticky; top: 0; }}
th:first-child, td:first-child {{ text-align: left; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
.notice {{ background: #fff3cd; border: 1px solid #ffe08a; padding: 12px 14px; border-radius: 8px; }}
</style></head><body>
<h1>{title}</h1>
<p class='notice'><b>Research use only.</b> This is unsupervised heuristic tuning. It is not clinical validation; inspect overlays before using any TPV values.</p>
<h2>Selected global parameters</h2><pre>{params_html}</pre>
<h2>Best row by vessel</h2>{best_html}
<h2>All tested candidates</h2>{all_html}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def plot_tuning_summary(tuning_results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save basic score/retention plots. Returns PNG paths."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for vessel, sub in tuning_results.groupby("vessel"):
        p = output_dir / f"{vessel}_tuning_score_vs_removed.png"
        plt.figure(figsize=(7, 5))
        plt.scatter(sub["removed_fraction"], sub["score"])
        plt.xlabel("Removed fraction")
        plt.ylabel("Heuristic score")
        plt.title(f"{vessel}: tuning score vs removed fraction")
        plt.tight_layout()
        plt.savefig(p, dpi=140)
        plt.close()
        paths.append(p)
    return paths
