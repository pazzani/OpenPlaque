"""Run optimized OpenPlaque boundary refinement on a new CCTA study.

This module consumes the *current* tuning JSON format produced by the Bayesian
or all-data tuning notebooks. It does not require or create an extended metadata
format. The expected JSON has a top-level key:

    final_parameters_selected_on_all_cases

whose value is passed directly to ``refine_plaque_mask`` with minimal cleaning.

Research use only. Not clinically validated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import csv
import json

import numpy as np

from .boundary import refine_plaque_mask

DEFAULT_FALLBACK_SERIES = {"RCA": 1035, "LCX": 1039, "LAD": 1043}
PARAM_KEY = "final_parameters_selected_on_all_cases"


def load_best_boundary_parameters(path: str | Path) -> dict[str, Any]:
    """Load optimized parameters from the current best-parameters JSON file.

    The current tuning notebooks save parameters under the top-level key
    ``final_parameters_selected_on_all_cases``. This function preserves that
    format and returns only that dictionary.

    A direct parameter dictionary is accepted as a defensive fallback, but no
    new/extended JSON format is required.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Best-parameter JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and PARAM_KEY in payload:
        params = dict(payload[PARAM_KEY])
    elif isinstance(payload, Mapping):
        # Defensive fallback only: accept a plain params dictionary if the user
        # manually extracted it. The notebook still writes/reads the current format.
        params = dict(payload)
    else:
        raise ValueError(f"Could not parse parameter JSON: {path}")
    return normalize_boundary_parameters(params)


def normalize_boundary_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Clean JSON-loaded parameter types and add derived flags expected by boundary.py."""
    p = dict(params)

    int_keys = [
        "min_component_voxels",
        "lumen_distance_voxels",
        "closing_radius_voxels",
        "connectivity",
        "erosion_iterations",
    ]
    float_keys = ["min_plaque_length_mm"]
    bool_keys = ["fill_holes", "adaptive_hu_thresholds", "erode_core", "remove_small", "trim_lumen_adjacent"]

    for k in int_keys:
        if k in p and p[k] is not None:
            p[k] = int(p[k])
    for k in float_keys:
        if k in p and p[k] is not None:
            p[k] = float(p[k])
    for k in bool_keys:
        if k in p and p[k] is not None:
            p[k] = bool(p[k])

    # JSON may contain null for thresholds; leave as None.
    for k in ["high_hu_threshold", "low_hu_threshold"]:
        if k in p and p[k] is not None:
            p[k] = float(p[k])

    p.setdefault("min_component_voxels", 10)
    p.setdefault("lumen_distance_voxels", 1)
    p.setdefault("closing_radius_voxels", 0)
    p.setdefault("fill_holes", False)
    p.setdefault("min_plaque_length_mm", 0.0)
    p.setdefault("connectivity", 26)
    p.setdefault("adaptive_hu_thresholds", False)
    p.setdefault("erode_core", False)
    p.setdefault("erosion_iterations", 1)
    p.setdefault("high_hu_threshold", None)
    p.setdefault("low_hu_threshold", None)

    p["remove_small"] = bool(p.get("min_component_voxels", 0) and int(p["min_component_voxels"]) > 0)
    p["trim_lumen_adjacent"] = bool(p.get("lumen_distance_voxels", 0) and int(p["lumen_distance_voxels"]) > 0)
    return p


def make_core_parameters(params: Mapping[str, Any], erosion_iterations: int = 1) -> dict[str, Any]:
    """Create conservative core parameters from the optimized main parameters."""
    core = normalize_boundary_parameters(params)
    core["erode_core"] = True
    core["erosion_iterations"] = int(erosion_iterations)
    core["remove_small"] = True
    core["trim_lumen_adjacent"] = bool(core.get("lumen_distance_voxels", 0) > 0)
    return core


def refine_report_with_parameters(report, params: Mapping[str, Any]):
    """Apply optimized boundary refinement to one OpenPlaque SegmentationReport."""
    p = normalize_boundary_parameters(params)
    return refine_plaque_mask(
        volume=report.volume,
        mask=report.mask,
        spacing=report.mask_image.GetSpacing(),
        **p,
    )


def refine_reports_with_parameters(reports: Sequence[Any], params: Mapping[str, Any]) -> dict[str, Any]:
    return {report.name: refine_report_with_parameters(report, params) for report in reports}


def core_reports_with_parameters(reports: Sequence[Any], params: Mapping[str, Any], erosion_iterations: int = 1) -> dict[str, Any]:
    core_params = make_core_parameters(params, erosion_iterations=erosion_iterations)
    return {report.name: refine_report_with_parameters(report, core_params) for report in reports}


def tpv_summary_rows(reports: Sequence[Any], refinements: Mapping[str, Any], core_results: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        ref = refinements[report.name]
        core = None if core_results is None else core_results.get(report.name)
        rows.append({
            "vessel": report.name,
            "raw_tpv_mm3": float(report.tpv_mm3),
            "refined_tpv_mm3": float(ref.refined_tpv_mm3),
            "removed_tpv_mm3": float(ref.removed_volume_mm3),
            "core_tpv_mm3": None if core is None else float(core.refined_tpv_mm3),
            "raw_plaque_voxels": int(report.plaque_voxels),
            "refined_plaque_voxels": int(ref.refined_plaque_voxels),
            "core_plaque_voxels": None if core is None else int(core.refined_plaque_voxels),
        })
    total = {
        "vessel": "TOTAL",
        "raw_tpv_mm3": sum(r["raw_tpv_mm3"] for r in rows),
        "refined_tpv_mm3": sum(r["refined_tpv_mm3"] for r in rows),
        "removed_tpv_mm3": sum(r["removed_tpv_mm3"] for r in rows),
        "core_tpv_mm3": None if core_results is None else sum(float(r["core_tpv_mm3"] or 0.0) for r in rows),
        "raw_plaque_voxels": sum(r["raw_plaque_voxels"] for r in rows),
        "refined_plaque_voxels": sum(r["refined_plaque_voxels"] for r in rows),
        "core_plaque_voxels": None if core_results is None else sum(int(r["core_plaque_voxels"] or 0) for r in rows),
    }
    rows.append(total)
    return rows


def save_tpv_summary_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_refined_masks(reports: Sequence[Any], refinements: Mapping[str, Any], output_dir: str | Path, suffix: str = "optimized_refined") -> list[Path]:
    import SimpleITK as sitk

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for report in reports:
        ref = refinements[report.name]
        img = sitk.GetImageFromArray(ref.refined_mask.astype("uint8"))
        img.CopyInformation(report.mask_image)
        path = output_dir / f"{report.name}_{suffix}.nii.gz"
        sitk.WriteImage(img, str(path))
        paths.append(path)
    return paths


def save_overlay_png(volume: np.ndarray, mask: np.ndarray, path: str | Path, title: str, label: int = 2, z: Optional[int] = None, vmin=-200, vmax=800) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if z is None:
        counts = np.sum(np.asarray(mask) == label, axis=(1, 2))
        z = int(np.argmax(counts)) if counts.size else 0
    plt.figure(figsize=(7, 7))
    plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.imshow(np.asarray(mask)[z] == label, alpha=0.45)
    plt.title(f"{title} — slice {z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_overlays(reports: Sequence[Any], refinements: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    paths: list[Path] = []
    for report in reports:
        paths.append(save_overlay_png(report.volume, report.mask, output_dir / f"{report.name}_raw_overlay.png", f"{report.name} raw nnU-Net plaque"))
        paths.append(save_overlay_png(report.volume, refinements[report.name].refined_mask, output_dir / f"{report.name}_optimized_refined_overlay.png", f"{report.name} optimized refined plaque"))
    return paths


def write_new_data_html_report(
    output_html: str | Path,
    rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    overlay_paths: Optional[Sequence[str | Path]] = None,
    title: str = "OpenPlaque Optimized Boundary Refinement Report",
) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    headers = list(rows[0].keys()) if rows else []
    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    for r in rows:
        table += "<tr>" + "".join(f"<td>{fmt(r.get(h))}</td>" for h in headers) + "</tr>"
    table += "</tbody></table>"

    overlay_html = ""
    if overlay_paths:
        overlay_html += "<h2>Overlay files</h2><ul>"
        for p in overlay_paths:
            overlay_html += f"<li>{Path(p).name}</li>"
        overlay_html += "</ul>"

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;color:#222;}}
table {{border-collapse:collapse;width:100%;margin:16px 0 28px;}}
th,td {{border:1px solid #ddd;padding:7px 9px;text-align:right;}} th:first-child,td:first-child {{text-align:left;}} th {{background:#f4f4f4;}}
pre {{background:#f7f7f7;padding:12px;border-radius:8px;overflow-x:auto;}}
.notice {{background:#fff3cd;border:1px solid #ffe08a;padding:12px 14px;border-radius:8px;}}
</style></head><body>
<h1>{title}</h1>
<p class='notice'><b>Research use only.</b> Not clinically validated. Not for diagnosis or medical decision-making.</p>
<h2>Optimized parameters used</h2><pre>{json.dumps(dict(params), indent=2)}</pre>
<h2>TPV summary</h2>{table}
{overlay_html}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def auto_detect_or_fallback_series(study, fallback_series: Optional[Mapping[str, int]] = None) -> dict[str, int]:
    """Use OpenPlaque artery detection if available; otherwise fallback to UCLA series."""
    fallback = dict(fallback_series or DEFAULT_FALLBACK_SERIES)
    try:
        from .artery_detection import detect_artery_series
        detected = detect_artery_series(study, fallback_series=fallback)
        return {k: int(v) for k, v in detected.items()}
    except Exception as e:
        print("Artery auto-detection unavailable/failed; using fallback series.")
        print("Reason:", repr(e))
        return {k: int(v) for k, v in fallback.items()}


def process_new_study(
    study_zip: str | Path,
    best_parameters_json: str | Path,
    output_dir: str | Path,
    fallback_series: Optional[Mapping[str, int]] = None,
    save_masks: bool = True,
    save_png_overlays: bool = True,
) -> dict[str, Any]:
    """End-to-end processing for a new CCTA DICOM ZIP using optimized parameters."""
    from openplaque.study import OpenPlaqueStudy
    from openplaque.segmentation import segment_vessel

    params = load_best_boundary_parameters(best_parameters_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = OpenPlaqueStudy(str(study_zip))
    series_map = auto_detect_or_fallback_series(study, fallback_series=fallback_series)

    reports = []
    for artery in ["LAD", "RCA", "LCX"]:
        image, volume, _ = study.load_series(series_map[artery])
        report = segment_vessel(image, volume, artery)
        reports.append(report)

    refinements = refine_reports_with_parameters(reports, params)
    core_results = core_reports_with_parameters(reports, params)
    rows = tpv_summary_rows(reports, refinements, core_results=core_results)

    csv_path = save_tpv_summary_csv(rows, output_dir / "new_data_tpv_summary.csv")
    mask_paths = save_refined_masks(reports, refinements, output_dir) if save_masks else []
    overlay_paths = save_overlays(reports, refinements, output_dir / "overlays") if save_png_overlays else []
    html_path = write_new_data_html_report(output_dir / "new_data_optimized_tpv_report.html", rows, params, overlay_paths=overlay_paths)

    return {
        "params": params,
        "series_map": series_map,
        "reports": reports,
        "refinements": refinements,
        "core_results": core_results,
        "rows": rows,
        "csv_path": csv_path,
        "mask_paths": mask_paths,
        "overlay_paths": overlay_paths,
        "html_path": html_path,
    }
