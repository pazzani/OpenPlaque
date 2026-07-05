"""Boundary-refinement parameter tuning for OpenPlaque.

This module performs unsupervised/sanity-check tuning over the AI plaque masks.
It does not require a manual ground-truth annotation. The default objective favors
stable, conservative refinements that remove likely boundary/noise voxels without
collapsing the predicted plaque volume.

Research use only. Not clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence
import csv
import json
import math

import numpy as np
from scipy import ndimage as ndi

from .boundary import refine_plaque_mask


DEFAULT_PARAMETER_GRID = {
    "min_component_voxels": [1, 5, 10, 25, 50],
    "lumen_distance_voxels": [0, 1, 2],
    "high_hu_threshold": [None, 700, 850, 1000],
    "low_hu_threshold": [None, -100, -50],
    "erode_core": [False, True],
    "erosion_iterations": [1],
}


@dataclass
class TuningRow:
    vessel: str
    candidate_id: int
    min_component_voxels: int
    lumen_distance_voxels: int
    high_hu_threshold: Optional[float]
    low_hu_threshold: Optional[float]
    erode_core: bool
    erosion_iterations: int
    raw_tpv_mm3: float
    refined_tpv_mm3: float
    removed_mm3: float
    retained_fraction: float
    removed_fraction: float
    components_before: int
    components_after: int
    largest_component_fraction: float
    mean_hu: Optional[float]
    median_hu: Optional[float]
    score: float
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TuningResult:
    rows: list[TuningRow]
    best_by_vessel: Dict[str, TuningRow]
    selected_params: dict
    notes: str

    def to_rows(self) -> list[dict]:
        return [r.to_dict() for r in self.rows]

    def save_csv(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def save_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selected_params": self.selected_params,
            "best_by_vessel": {k: v.to_dict() for k, v in self.best_by_vessel.items()},
            "rows": self.to_rows(),
            "notes": self.notes,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _parameter_candidates(grid: Optional[Mapping[str, Sequence]] = None):
    grid = dict(DEFAULT_PARAMETER_GRID if grid is None else grid)
    keys = list(grid.keys())
    for i, values in enumerate(product(*[grid[k] for k in keys])):
        p = dict(zip(keys, values))
        if p.get("lumen_distance_voxels", 1) == 0:
            trim_lumen_adjacent = False
        else:
            trim_lumen_adjacent = True
        yield i, {
            "remove_small": True,
            "min_component_voxels": int(p.get("min_component_voxels", 10)),
            "trim_lumen_adjacent": trim_lumen_adjacent,
            "lumen_distance_voxels": int(p.get("lumen_distance_voxels", 1)),
            "erode_core": bool(p.get("erode_core", False)),
            "erosion_iterations": int(p.get("erosion_iterations", 1)),
            "high_hu_threshold": p.get("high_hu_threshold"),
            "low_hu_threshold": p.get("low_hu_threshold"),
        }


def _component_stats(mask, plaque_label=2):
    plaque = np.asarray(mask) == plaque_label
    if not np.any(plaque):
        return 0, 0.0
    labels, n = ndi.label(plaque)
    sizes = np.bincount(labels.ravel())
    if len(sizes) <= 1:
        return int(n), 0.0
    largest = float(np.max(sizes[1:]))
    total = float(np.sum(sizes[1:]))
    return int(n), largest / total if total > 0 else 0.0


def _hu_stats(volume, mask, plaque_label=2):
    vals = np.asarray(volume)[np.asarray(mask) == plaque_label]
    if vals.size == 0:
        return None, None
    return float(np.mean(vals)), float(np.median(vals))


def _score_candidate(retained_fraction: float,
                     removed_fraction: float,
                     components_after: int,
                     components_before: int,
                     largest_component_fraction: float,
                     mean_hu: Optional[float],
                     erode_core: bool) -> tuple[float, str]:
    """Heuristic score. Higher is better.

    Rejects extreme settings that retain almost everything or delete too much.
    Preferred behavior is modest boundary cleanup, fewer tiny components, and
    stable plaque core preservation.
    """
    reasons = []
    if retained_fraction <= 0:
        reasons.append("deleted_all_plaque")
    if retained_fraction < 0.35:
        reasons.append("over_aggressive_retains_under_35pct")
    if retained_fraction > 0.995:
        reasons.append("no_effect")

    # Target a moderate cleanup. This is intentionally broad because data vary.
    target_removed = 0.15
    score = 100.0
    score -= 180.0 * abs(removed_fraction - target_removed)

    # Reward removing fragmented noise but do not demand one component.
    if components_before > 0:
        component_reduction = max(0, components_before - components_after) / components_before
        score += 18.0 * component_reduction

    # Reward masks with a dominant plaque component after refinement.
    score += 18.0 * largest_component_fraction

    # Penalize core erosion for the main refined estimate; core is reported separately.
    if erode_core:
        score -= 15.0

    # Soft HU plausibility: avoid extremely low/high mean plaque HU when thresholding is used.
    if mean_hu is not None:
        if mean_hu < -150 or mean_hu > 1100:
            score -= 20.0

    if reasons:
        score -= 1000.0
    return float(score), ";".join(reasons)


def tune_boundary_for_report(report,
                             parameter_grid: Optional[Mapping[str, Sequence]] = None,
                             min_retained_fraction: float = 0.35,
                             max_retained_fraction: float = 0.995) -> list[TuningRow]:
    rows: list[TuningRow] = []
    raw_tpv = float(report.tpv_mm3)
    components_before, _ = _component_stats(report.mask)

    for cid, params in _parameter_candidates(parameter_grid):
        ref = refine_plaque_mask(
            volume=report.volume,
            mask=report.mask,
            spacing=report.mask_image.GetSpacing(),
            **params,
        )
        refined_tpv = float(ref.refined_tpv_mm3)
        retained = refined_tpv / raw_tpv if raw_tpv > 0 else 0.0
        removed_fraction = 1.0 - retained
        components_after, largest_frac = _component_stats(ref.refined_mask)
        mean_hu, median_hu = _hu_stats(report.volume, ref.refined_mask)
        score, reject = _score_candidate(
            retained, removed_fraction, components_after, components_before,
            largest_frac, mean_hu, params["erode_core"]
        )
        if retained < min_retained_fraction and "over_aggressive" not in reject:
            reject = (reject + ";" if reject else "") + "below_min_retained_fraction"
            score -= 1000
        if retained > max_retained_fraction and "no_effect" not in reject:
            reject = (reject + ";" if reject else "") + "above_max_retained_fraction"
            score -= 1000
        rows.append(TuningRow(
            vessel=report.name,
            candidate_id=cid,
            min_component_voxels=params["min_component_voxels"],
            lumen_distance_voxels=params["lumen_distance_voxels"] if params["trim_lumen_adjacent"] else 0,
            high_hu_threshold=params["high_hu_threshold"],
            low_hu_threshold=params["low_hu_threshold"],
            erode_core=params["erode_core"],
            erosion_iterations=params["erosion_iterations"],
            raw_tpv_mm3=raw_tpv,
            refined_tpv_mm3=refined_tpv,
            removed_mm3=float(ref.removed_volume_mm3),
            retained_fraction=float(retained),
            removed_fraction=float(removed_fraction),
            components_before=components_before,
            components_after=components_after,
            largest_component_fraction=float(largest_frac),
            mean_hu=mean_hu,
            median_hu=median_hu,
            score=score,
            reject_reason=reject,
        ))
    return rows


def tune_boundary_parameters(reports: Iterable,
                             parameter_grid: Optional[Mapping[str, Sequence]] = None) -> TuningResult:
    all_rows: list[TuningRow] = []
    best_by_vessel: Dict[str, TuningRow] = {}
    for report in reports:
        rows = tune_boundary_for_report(report, parameter_grid=parameter_grid)
        all_rows.extend(rows)
        valid = [r for r in rows if not r.reject_reason]
        best = max(valid or rows, key=lambda r: r.score)
        best_by_vessel[report.name] = best

    # Choose a single global parameter set by summing scores across vessels.
    grouped = {}
    for r in all_rows:
        key = (
            r.min_component_voxels, r.lumen_distance_voxels,
            r.high_hu_threshold, r.low_hu_threshold,
            r.erode_core, r.erosion_iterations,
        )
        grouped.setdefault(key, 0.0)
        grouped[key] += r.score
    best_key = max(grouped, key=grouped.get) if grouped else (10, 1, None, None, False, 1)
    selected_params = {
        "remove_small": True,
        "min_component_voxels": best_key[0],
        "trim_lumen_adjacent": best_key[1] > 0,
        "lumen_distance_voxels": best_key[1],
        "erode_core": best_key[4],
        "erosion_iterations": best_key[5],
        "high_hu_threshold": best_key[2],
        "low_hu_threshold": best_key[3],
    }
    notes = (
        "Unsupervised tuning: selected by heuristic score using retained TPV, removed fraction, "
        "fragmentation reduction, largest-component fraction, and HU sanity checks. "
        "Use expert visual QC before interpreting results."
    )
    return TuningResult(rows=all_rows, best_by_vessel=best_by_vessel, selected_params=selected_params, notes=notes)


def apply_selected_refinement(reports: Iterable, selected_params: Mapping) -> Dict[str, object]:
    out = {}
    for report in reports:
        out[report.name] = refine_plaque_mask(
            volume=report.volume,
            mask=report.mask,
            spacing=report.mask_image.GetSpacing(),
            **dict(selected_params),
        )
    return out


def write_tuning_html_report(tuning_result: TuningResult, output_html, title="OpenPlaque Boundary Parameter Tuning Report") -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    rows = sorted(tuning_result.rows, key=lambda r: (r.vessel, -r.score))
    body_rows = []
    for r in rows:
        cls = " class='rejected'" if r.reject_reason else ""
        body_rows.append(
            f"<tr{cls}><td>{r.vessel}</td><td>{r.candidate_id}</td>"
            f"<td>{r.score:.2f}</td><td>{r.min_component_voxels}</td>"
            f"<td>{r.lumen_distance_voxels}</td><td>{fmt(r.high_hu_threshold)}</td>"
            f"<td>{fmt(r.low_hu_threshold)}</td><td>{fmt(r.erode_core)}</td>"
            f"<td>{r.raw_tpv_mm3:.2f}</td><td>{r.refined_tpv_mm3:.2f}</td>"
            f"<td>{100*r.retained_fraction:.1f}%</td><td>{r.components_before}</td>"
            f"<td>{r.components_after}</td><td>{r.largest_component_fraction:.3f}</td>"
            f"<td>{fmt(r.mean_hu)}</td><td>{r.reject_reason}</td></tr>"
        )

    best_rows = []
    for vessel, r in tuning_result.best_by_vessel.items():
        best_rows.append(
            f"<tr><td>{vessel}</td><td>{r.candidate_id}</td><td>{r.score:.2f}</td>"
            f"<td>{r.min_component_voxels}</td><td>{r.lumen_distance_voxels}</td>"
            f"<td>{fmt(r.high_hu_threshold)}</td><td>{fmt(r.low_hu_threshold)}</td>"
            f"<td>{fmt(r.erode_core)}</td><td>{r.refined_tpv_mm3:.2f}</td>"
            f"<td>{100*r.retained_fraction:.1f}%</td></tr>"
        )

    params_json = json.dumps(tuning_result.selected_params, indent=2)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ background: #f6f6f6; position: sticky; top: 0; }}
.rejected {{ color: #888; background: #fafafa; }}
.notice {{ padding: 12px 14px; background: #fff3cd; border: 1px solid #ffe08a; border-radius: 8px; }}
pre {{ background: #f7f7f7; padding: 12px; border-radius: 8px; overflow-x: auto; }}
</style></head><body>
<h1>{title}</h1>
<p class="notice"><strong>Research use only.</strong> This is unsupervised parameter tuning, not clinical validation. Use visual QC.</p>
<h2>Selected global parameters</h2><pre>{params_json}</pre>
<p>{tuning_result.notes}</p>
<h2>Best candidate by vessel</h2>
<table><thead><tr><th>Vessel</th><th>Candidate</th><th>Score</th><th>Min comp vox</th><th>Lumen dist</th><th>High HU</th><th>Low HU</th><th>Erode</th><th>Refined TPV</th><th>Retained</th></tr></thead><tbody>{''.join(best_rows)}</tbody></table>
<h2>All candidates</h2>
<table><thead><tr><th>Vessel</th><th>ID</th><th>Score</th><th>Min comp vox</th><th>Lumen dist</th><th>High HU</th><th>Low HU</th><th>Erode</th><th>Raw TPV</th><th>Refined TPV</th><th>Retained</th><th>Comp before</th><th>Comp after</th><th>Largest comp frac</th><th>Mean HU</th><th>Reject reason</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
