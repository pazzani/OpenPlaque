"""Plaque type classification and quantification for OpenPlaque.

Consumes the refined plaque mask produced by optimized boundary refinement and
classifies plaque voxels by CT attenuation (HU). Default thresholds are simple,
transparent, and configurable.

Research use only. Not clinically validated. Not for diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import csv
import json

import numpy as np
from scipy import ndimage as ndi

DEFAULT_THRESHOLDS = {
    "low_attenuation": {"min_hu": None, "max_hu": 30.0},
    "noncalcified": {"min_hu": 30.0, "max_hu": 130.0},
    "mixed_intermediate": {"min_hu": 130.0, "max_hu": 350.0},
    "calcified": {"min_hu": 350.0, "max_hu": None},
}

PLAQUE_CLASS_LABELS = {
    0: "non_plaque",
    1: "low_attenuation",
    2: "noncalcified",
    3: "mixed_intermediate",
    4: "calcified",
}

CLASS_ORDER = ["low_attenuation", "noncalcified", "mixed_intermediate", "calcified"]


def _voxel_volume_mm3(spacing: Sequence[float]) -> float:
    return float(np.prod(tuple(float(x) for x in spacing)))


def _class_condition(volume: np.ndarray, spec: Mapping[str, Optional[float]]) -> np.ndarray:
    cond = np.ones(volume.shape, dtype=bool)
    lo = spec.get("min_hu")
    hi = spec.get("max_hu")
    if lo is not None:
        cond &= volume >= float(lo)
    if hi is not None:
        cond &= volume < float(hi)
    return cond


def classify_plaque_voxels(
    volume: np.ndarray,
    plaque_mask: np.ndarray,
    thresholds: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None,
    plaque_label: int = 2,
) -> np.ndarray:
    """Return integer class map for plaque voxels.

    Output labels:
        0 non-plaque
        1 low_attenuation
        2 noncalcified
        3 mixed_intermediate
        4 calcified
    """
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    volume = np.asarray(volume)
    plaque = np.asarray(plaque_mask) == plaque_label
    class_map = np.zeros(volume.shape, dtype=np.uint8)
    for label_value, name in [(1, "low_attenuation"), (2, "noncalcified"), (3, "mixed_intermediate"), (4, "calcified")]:
        if name not in thresholds:
            continue
        class_map[plaque & _class_condition(volume, thresholds[name])] = label_value
    # Any plaque voxel not caught by custom thresholds is conservatively assigned
    # to noncalcified so total classified volume equals TPV.
    class_map[plaque & (class_map == 0)] = 2
    return class_map


@dataclass
class PlaqueTypeSummary:
    vessel: str
    spacing: tuple[float, float, float]
    total_plaque_voxels: int
    total_tpv_mm3: float
    low_attenuation_voxels: int
    low_attenuation_volume_mm3: float
    low_attenuation_fraction: float
    noncalcified_voxels: int
    noncalcified_volume_mm3: float
    noncalcified_fraction: float
    mixed_intermediate_voxels: int
    mixed_intermediate_volume_mm3: float
    mixed_intermediate_fraction: float
    calcified_voxels: int
    calcified_volume_mm3: float
    calcified_fraction: float
    mean_plaque_hu: Optional[float]
    median_plaque_hu: Optional[float]
    p10_plaque_hu: Optional[float]
    p90_plaque_hu: Optional[float]
    lesion_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlaqueLesionSummary:
    vessel: str
    lesion_id: int
    voxel_count: int
    volume_mm3: float
    length_mm: float
    low_attenuation_volume_mm3: float
    noncalcified_volume_mm3: float
    mixed_intermediate_volume_mm3: float
    calcified_volume_mm3: float
    mean_hu: Optional[float]
    max_hu: Optional[float]
    min_hu: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_plaque_types(
    vessel: str,
    volume: np.ndarray,
    refined_mask: np.ndarray,
    spacing: Sequence[float],
    thresholds: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None,
    plaque_label: int = 2,
    connectivity: int = 26,
) -> tuple[PlaqueTypeSummary, list[PlaqueLesionSummary], np.ndarray]:
    """Classify and summarize plaque composition for one vessel."""
    volume = np.asarray(volume)
    mask = np.asarray(refined_mask)
    spacing = tuple(float(x) for x in spacing)
    voxel_volume = _voxel_volume_mm3(spacing)
    plaque = mask == plaque_label
    class_map = classify_plaque_voxels(volume, mask, thresholds=thresholds, plaque_label=plaque_label)
    total_voxels = int(plaque.sum())
    total_volume = total_voxels * voxel_volume

    vals = volume[plaque]
    if vals.size:
        mean_hu = float(np.mean(vals))
        median_hu = float(np.median(vals))
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
    else:
        mean_hu = median_hu = p10 = p90 = None

    counts = {name: int(np.sum(class_map == idx)) for idx, name in PLAQUE_CLASS_LABELS.items() if idx != 0}
    vols = {name: counts[name] * voxel_volume for name in counts}
    fracs = {name: (vols[name] / total_volume if total_volume > 0 else 0.0) for name in counts}

    structure = _structure_for_connectivity(connectivity)
    labels, n = ndi.label(plaque, structure=structure)
    lesions: list[PlaqueLesionSummary] = []
    objects = ndi.find_objects(labels)
    z_spacing = spacing[2] if len(spacing) >= 3 else 1.0
    for lesion_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        lesion = labels == lesion_id
        lesion_vox = int(lesion.sum())
        if lesion_vox == 0:
            continue
        lesion_vals = volume[lesion]
        z0, z1 = slc[0].start, slc[0].stop
        length_mm = float(max(0, z1 - z0) * z_spacing)
        lesions.append(PlaqueLesionSummary(
            vessel=vessel,
            lesion_id=lesion_id,
            voxel_count=lesion_vox,
            volume_mm3=lesion_vox * voxel_volume,
            length_mm=length_mm,
            low_attenuation_volume_mm3=float(np.sum(lesion & (class_map == 1)) * voxel_volume),
            noncalcified_volume_mm3=float(np.sum(lesion & (class_map == 2)) * voxel_volume),
            mixed_intermediate_volume_mm3=float(np.sum(lesion & (class_map == 3)) * voxel_volume),
            calcified_volume_mm3=float(np.sum(lesion & (class_map == 4)) * voxel_volume),
            mean_hu=float(np.mean(lesion_vals)) if lesion_vals.size else None,
            max_hu=float(np.max(lesion_vals)) if lesion_vals.size else None,
            min_hu=float(np.min(lesion_vals)) if lesion_vals.size else None,
        ))

    summary = PlaqueTypeSummary(
        vessel=vessel,
        spacing=spacing,
        total_plaque_voxels=total_voxels,
        total_tpv_mm3=float(total_volume),
        low_attenuation_voxels=counts["low_attenuation"],
        low_attenuation_volume_mm3=float(vols["low_attenuation"]),
        low_attenuation_fraction=float(fracs["low_attenuation"]),
        noncalcified_voxels=counts["noncalcified"],
        noncalcified_volume_mm3=float(vols["noncalcified"]),
        noncalcified_fraction=float(fracs["noncalcified"]),
        mixed_intermediate_voxels=counts["mixed_intermediate"],
        mixed_intermediate_volume_mm3=float(vols["mixed_intermediate"]),
        mixed_intermediate_fraction=float(fracs["mixed_intermediate"]),
        calcified_voxels=counts["calcified"],
        calcified_volume_mm3=float(vols["calcified"]),
        calcified_fraction=float(fracs["calcified"]),
        mean_plaque_hu=mean_hu,
        median_plaque_hu=median_hu,
        p10_plaque_hu=p10,
        p90_plaque_hu=p90,
        lesion_count=len(lesions),
    )
    return summary, lesions, class_map


def _structure_for_connectivity(connectivity: int = 26):
    if connectivity <= 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity <= 18:
        return ndi.generate_binary_structure(3, 2)
    return ndi.generate_binary_structure(3, 3)


def summarize_all_vessels(reports: Sequence[Any], refinements: Mapping[str, Any], thresholds=None, connectivity: int = 26):
    summaries: list[PlaqueTypeSummary] = []
    lesion_rows: list[PlaqueLesionSummary] = []
    class_maps: dict[str, np.ndarray] = {}
    for report in reports:
        refined = refinements[report.name].refined_mask
        summary, lesions, class_map = summarize_plaque_types(
            vessel=report.name,
            volume=report.volume,
            refined_mask=refined,
            spacing=report.mask_image.GetSpacing(),
            thresholds=thresholds,
            connectivity=connectivity,
        )
        summaries.append(summary)
        lesion_rows.extend(lesions)
        class_maps[report.name] = class_map
    return summaries, lesion_rows, class_maps


def total_summary_row(summaries: Sequence[PlaqueTypeSummary]) -> dict[str, Any]:
    rows = [s.to_dict() for s in summaries]
    total_tpv = sum(r["total_tpv_mm3"] for r in rows)
    out = {
        "vessel": "TOTAL",
        "total_plaque_voxels": sum(r["total_plaque_voxels"] for r in rows),
        "total_tpv_mm3": total_tpv,
        "low_attenuation_voxels": sum(r["low_attenuation_voxels"] for r in rows),
        "low_attenuation_volume_mm3": sum(r["low_attenuation_volume_mm3"] for r in rows),
        "noncalcified_voxels": sum(r["noncalcified_voxels"] for r in rows),
        "noncalcified_volume_mm3": sum(r["noncalcified_volume_mm3"] for r in rows),
        "mixed_intermediate_voxels": sum(r["mixed_intermediate_voxels"] for r in rows),
        "mixed_intermediate_volume_mm3": sum(r["mixed_intermediate_volume_mm3"] for r in rows),
        "calcified_voxels": sum(r["calcified_voxels"] for r in rows),
        "calcified_volume_mm3": sum(r["calcified_volume_mm3"] for r in rows),
        "lesion_count": sum(r["lesion_count"] for r in rows),
    }
    for cls in CLASS_ORDER:
        key = f"{cls}_volume_mm3"
        out[f"{cls}_fraction"] = (out[key] / total_tpv) if total_tpv > 0 else 0.0
    return out


def save_characterization_csv(summaries: Sequence[PlaqueTypeSummary], output_csv: str | Path, include_total: bool = True) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_dict() for s in summaries]
    if include_total:
        rows.append(total_summary_row(summaries))
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def save_lesion_csv(lesions: Sequence[PlaqueLesionSummary], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [l.to_dict() for l in lesions]
    fieldnames = list(rows[0].keys()) if rows else ["vessel", "lesion_id"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def save_class_map_nifti(class_map: np.ndarray, reference_image, path: str | Path) -> Path:
    import SimpleITK as sitk
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(class_map.astype("uint8"))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(path))
    return path


def save_class_maps_nifti(reports: Sequence[Any], class_maps: Mapping[str, np.ndarray], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    paths = []
    for report in reports:
        paths.append(save_class_map_nifti(class_maps[report.name], report.mask_image, output_dir / f"{report.name}_plaque_type_class_map.nii.gz"))
    return paths


def save_plaque_type_overlay_png(volume, class_map, path: str | Path, title: str, z: Optional[int] = None, vmin=-200, vmax=800) -> Path:
    """Save a color-coded plaque-type overlay.

    Colors:
        red: low attenuation
        orange: noncalcified
        cyan/blue: mixed/intermediate
        white: calcified
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    volume = np.asarray(volume)
    class_map = np.asarray(class_map)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if z is None:
        counts = np.sum(class_map > 0, axis=(1, 2))
        z = int(np.argmax(counts)) if counts.size else 0
    overlay = np.ma.masked_where(class_map[z] == 0, class_map[z])
    cmap = ListedColormap(["red", "orange", "deepskyblue", "white"])
    plt.figure(figsize=(7, 7))
    plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.imshow(overlay, cmap=cmap, vmin=1, vmax=4, alpha=0.65)
    plt.title(f"{title} — slice {z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_type_overlays(reports: Sequence[Any], class_maps: Mapping[str, np.ndarray], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    paths = []
    for report in reports:
        paths.append(save_plaque_type_overlay_png(report.volume, class_maps[report.name], output_dir / f"{report.name}_plaque_type_overlay.png", f"{report.name} plaque type"))
    return paths


def write_characterization_html_report(
    output_html: str | Path,
    summaries: Sequence[PlaqueTypeSummary],
    lesions: Sequence[PlaqueLesionSummary],
    thresholds: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None,
    overlay_paths: Optional[Sequence[str | Path]] = None,
    title: str = "OpenPlaque Plaque Characterization Report",
) -> Path:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    summary_rows = [s.to_dict() for s in summaries] + [total_summary_row(summaries)]
    lesion_rows = [l.to_dict() for l in lesions]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    def table(rows):
        if not rows:
            return "<p>No rows.</p>"
        headers = list(rows[0].keys())
        html = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
        for r in rows:
            html += "<tr>" + "".join(f"<td>{fmt(r.get(h))}</td>" for h in headers) + "</tr>"
        html += "</tbody></table>"
        return html

    overlay_html = ""
    if overlay_paths:
        overlay_html = "<h2>Overlay files</h2><ul>" + "".join(f"<li>{Path(p).name}</li>" for p in overlay_paths) + "</ul>"

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;color:#222;}}
table {{border-collapse:collapse;width:100%;margin:16px 0 28px;font-size:13px;}}
th,td {{border:1px solid #ddd;padding:6px 8px;text-align:right;}} th:first-child,td:first-child {{text-align:left;}} th {{background:#f4f4f4;}}
pre {{background:#f7f7f7;padding:12px;border-radius:8px;overflow-x:auto;}}
.notice {{background:#fff3cd;border:1px solid #ffe08a;padding:12px 14px;border-radius:8px;}}
.legend span {{display:inline-block;margin-right:16px;}}
.swatch {{width:14px;height:14px;border:1px solid #999;vertical-align:middle;margin-right:4px;}}
</style></head><body>
<h1>{title}</h1>
<p class='notice'><b>Research use only.</b> Plaque type classification is based on CT attenuation thresholds and is not clinically validated.</p>
<h2>Thresholds</h2><pre>{json.dumps(thresholds, indent=2)}</pre>
<div class='legend'><b>Overlay colors:</b>
<span><span class='swatch' style='background:red'></span>Low attenuation</span>
<span><span class='swatch' style='background:orange'></span>Noncalcified</span>
<span><span class='swatch' style='background:deepskyblue'></span>Mixed/intermediate</span>
<span><span class='swatch' style='background:white'></span>Calcified</span>
</div>
<h2>Per-vessel composition</h2>{table(summary_rows)}
<h2>Per-lesion composition</h2>{table(lesion_rows)}
{overlay_html}
</body></html>"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
