"""HU context analysis around nnU-Net plaque masks.

This module helps test whether nnU-Net plaque segmentations are restricted to
calcified cores by comparing HU distributions inside plaque masks with the
immediate surrounding dilated shells.

Research use only. Not clinically validated. Not for diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import csv
import html
import math

import numpy as np
from scipy import ndimage as ndi


HU_BIN_SPECS: tuple[tuple[str, str, Optional[float], Optional[float]], ...] = (
    ("lt_30", "<30", None, 30.0),
    ("hu_30_130", "30-130", 30.0, 130.0),
    ("hu_130_350", "130-350", 130.0, 350.0),
    ("hu_350_700", "350-700", 350.0, 700.0),
    ("hu_700_1000", "700-1000", 700.0, 1000.0),
    ("gt_1000", ">1000", 1000.0, None),
)

DEFAULT_HISTOGRAM_BINS = np.array(
    [-1000, -500, -200, 0, 30, 130, 350, 700, 1000, 1500, 2500],
    dtype=float,
)


@dataclass
class HUDistributionSummary:
    vessel: str
    region: str
    radius_voxels: int
    voxel_count: int
    mean_hu: Optional[float]
    median_hu: Optional[float]
    min_hu: Optional[float]
    max_hu: Optional[float]
    p10_hu: Optional[float]
    p90_hu: Optional[float]
    calcified_core_fraction: float
    noncalcified_context_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool_mask(mask: np.ndarray, label: int = 2) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr == label


def _structure_for_connectivity(mask_ndim: int, connectivity: int = 26) -> np.ndarray:
    if mask_ndim < 1:
        raise ValueError("Mask must have at least one dimension.")
    if mask_ndim == 2:
        rank = 1 if connectivity <= 4 else 2
    elif mask_ndim == 3:
        rank = 1 if connectivity <= 6 else 2 if connectivity <= 18 else 3
    else:
        rank = mask_ndim
    return ndi.generate_binary_structure(mask_ndim, rank)


def dilate_plaque_mask(
    plaque_mask: np.ndarray,
    radius_voxels: int = 1,
    label: int = 2,
    connectivity: int = 26,
) -> np.ndarray:
    """Return a boolean plaque mask dilated by ``radius_voxels``."""
    radius = int(radius_voxels)
    plaque = _as_bool_mask(plaque_mask, label=label)
    if radius < 0:
        raise ValueError("radius_voxels must be >= 0.")
    if radius == 0:
        return plaque.copy()
    structure = _structure_for_connectivity(plaque.ndim, connectivity=connectivity)
    return ndi.binary_dilation(plaque, structure=structure, iterations=radius)


def plaque_shell_mask(
    plaque_mask: np.ndarray,
    radius_voxels: int = 1,
    label: int = 2,
    connectivity: int = 26,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return voxels within a dilation radius but outside the plaque mask."""
    plaque = _as_bool_mask(plaque_mask, label=label)
    shell = dilate_plaque_mask(plaque, radius_voxels, label=label, connectivity=connectivity) & ~plaque
    if exclude_mask is not None:
        shell &= ~np.asarray(exclude_mask, dtype=bool)
    return shell


def compute_hu_histogram(
    volume: np.ndarray,
    mask: np.ndarray,
    bins: Sequence[float] = DEFAULT_HISTOGRAM_BINS,
    label: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute HU histogram counts for voxels selected by ``mask``."""
    volume = np.asarray(volume)
    selected = _as_bool_mask(mask, label=label)
    if volume.shape != selected.shape:
        raise ValueError("volume and mask must have the same shape.")
    counts, edges = np.histogram(volume[selected].astype(float), bins=np.asarray(bins, dtype=float))
    return counts.astype(int), edges


def shell_hu_values(
    volume: np.ndarray,
    plaque_mask: np.ndarray,
    radius_voxels: int,
    label: int = 2,
    connectivity: int = 26,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return HU values in the shell around the plaque mask."""
    volume = np.asarray(volume)
    shell = plaque_shell_mask(
        plaque_mask,
        radius_voxels=radius_voxels,
        label=label,
        connectivity=connectivity,
        exclude_mask=exclude_mask,
    )
    if volume.shape != shell.shape:
        raise ValueError("volume and plaque_mask must have the same shape.")
    return volume[shell].astype(float)


def summarize_hu_bins(values: np.ndarray) -> dict[str, Any]:
    """Summarize HU values into the OpenPlaque context bins."""
    values = np.asarray(values, dtype=float)
    total = int(values.size)
    out: dict[str, Any] = {"voxel_count": total}
    for key, label, lo, hi in HU_BIN_SPECS:
        cond = np.ones(values.shape, dtype=bool)
        if lo is not None:
            cond &= values > lo if key == "gt_1000" else values >= lo
        if hi is not None:
            cond &= values < hi if key != "hu_700_1000" else values <= hi
        count = int(np.sum(cond))
        out[f"{key}_label"] = label
        out[f"{key}_voxels"] = count
        out[f"{key}_fraction"] = (count / total) if total else 0.0
    return out


def summarize_hu_distribution(
    vessel: str,
    region: str,
    values: np.ndarray,
    radius_voxels: int = 0,
) -> dict[str, Any]:
    """Return descriptive statistics and requested HU-bin counts."""
    values = np.asarray(values, dtype=float)
    if values.size:
        summary = HUDistributionSummary(
            vessel=vessel,
            region=region,
            radius_voxels=int(radius_voxels),
            voxel_count=int(values.size),
            mean_hu=float(np.mean(values)),
            median_hu=float(np.median(values)),
            min_hu=float(np.min(values)),
            max_hu=float(np.max(values)),
            p10_hu=float(np.percentile(values, 10)),
            p90_hu=float(np.percentile(values, 90)),
            calcified_core_fraction=float(np.mean(values >= 350.0)),
            noncalcified_context_fraction=float(np.mean((values >= 30.0) & (values < 130.0))),
        )
    else:
        summary = HUDistributionSummary(
            vessel=vessel,
            region=region,
            radius_voxels=int(radius_voxels),
            voxel_count=0,
            mean_hu=None,
            median_hu=None,
            min_hu=None,
            max_hu=None,
            p10_hu=None,
            p90_hu=None,
            calcified_core_fraction=0.0,
            noncalcified_context_fraction=0.0,
        )
    row = summary.to_dict()
    row.update(summarize_hu_bins(values))
    return row


def compute_plaque_context(
    vessel: str,
    volume: np.ndarray,
    plaque_mask: np.ndarray,
    radii_voxels: Sequence[int] = (1, 2, 3),
    label: int = 2,
    connectivity: int = 26,
    exclude_mask: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Summarize HU inside plaque and in surrounding dilation shells."""
    volume = np.asarray(volume)
    plaque = _as_bool_mask(plaque_mask, label=label)
    if volume.shape != plaque.shape:
        raise ValueError("volume and plaque_mask must have the same shape.")

    rows = [summarize_hu_distribution(vessel, "plaque", volume[plaque], radius_voxels=0)]
    for radius in radii_voxels:
        shell = plaque_shell_mask(
            plaque,
            radius_voxels=int(radius),
            label=label,
            connectivity=connectivity,
            exclude_mask=exclude_mask,
        )
        rows.append(summarize_hu_distribution(vessel, f"shell_{int(radius)}vox", volume[shell], radius_voxels=int(radius)))
    return rows


def compute_reports_plaque_context(
    reports: Sequence[Any],
    masks: Optional[Mapping[str, np.ndarray]] = None,
    radii_voxels: Sequence[int] = (1, 2, 3),
    label: int = 2,
    connectivity: int = 26,
) -> list[dict[str, Any]]:
    """Compute plaque-context rows for OpenPlaque SegmentationReport objects."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        mask = report.mask if masks is None else masks[report.name]
        rows.extend(
            compute_plaque_context(
                vessel=report.name,
                volume=report.volume,
                plaque_mask=mask,
                radii_voxels=radii_voxels,
                label=label,
                connectivity=connectivity,
            )
        )
    return rows


def save_context_csv(rows: Sequence[Mapping[str, Any]], output_csv: str | Path) -> Path:
    """Write plaque-context summary rows to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["vessel", "region", "radius_voxels", "voxel_count"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def save_hu_histogram_png(
    volume: np.ndarray,
    plaque_mask: np.ndarray,
    output_png: str | Path,
    title: str,
    radii_voxels: Sequence[int] = (1, 2, 3),
    label: int = 2,
    connectivity: int = 26,
    bins: Sequence[float] = DEFAULT_HISTOGRAM_BINS,
) -> Path:
    """Save overlaid HU histograms for plaque and surrounding shells."""
    import matplotlib.pyplot as plt

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plaque = _as_bool_mask(plaque_mask, label=label)
    series = [("plaque", np.asarray(volume)[plaque].astype(float))]
    for radius in radii_voxels:
        series.append((f"shell {int(radius)} vox", shell_hu_values(volume, plaque, int(radius), label=label, connectivity=connectivity)))

    plt.figure(figsize=(8, 5))
    for name, values in series:
        if values.size:
            plt.hist(values, bins=bins, histtype="step", linewidth=2, label=f"{name} (n={values.size})")
    for boundary in [30, 130, 350, 700, 1000]:
        plt.axvline(boundary, color="0.7", linestyle="--", linewidth=1)
    plt.xlabel("HU")
    plt.ylabel("Voxel count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    return output_png


def save_context_overlay_png(
    volume: np.ndarray,
    plaque_mask: np.ndarray,
    output_png: str | Path,
    title: str,
    radius_voxels: int = 3,
    label: int = 2,
    connectivity: int = 26,
    z: Optional[int] = None,
    vmin: float = -200,
    vmax: float = 800,
) -> Path:
    """Save an overlay showing plaque core and surrounding context shell."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    volume = np.asarray(volume)
    plaque = _as_bool_mask(plaque_mask, label=label)
    shell = plaque_shell_mask(plaque, radius_voxels=radius_voxels, label=label, connectivity=connectivity)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if z is None:
        counts = np.sum(plaque | shell, axis=(1, 2))
        z = int(np.argmax(counts)) if counts.size else 0

    overlay = np.zeros(volume.shape, dtype=np.uint8)
    overlay[shell] = 1
    overlay[plaque] = 2
    masked = np.ma.masked_where(overlay[z] == 0, overlay[z])
    cmap = ListedColormap(["gold", "red"])
    plt.figure(figsize=(7, 7))
    plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.imshow(masked, cmap=cmap, vmin=1, vmax=2, alpha=0.55)
    plt.title(f"{title} - slice {z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    return output_png


def write_context_html_report(
    output_html: str | Path,
    rows: Sequence[Mapping[str, Any]],
    histogram_paths: Optional[Sequence[str | Path]] = None,
    overlay_paths: Optional[Sequence[str | Path]] = None,
    title: str = "OpenPlaque Plaque HU Context Report",
) -> Path:
    """Write an HTML report for plaque-core and shell HU distributions."""
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.3f}"
        return html.escape(str(value))

    def table(data: Sequence[Mapping[str, Any]]) -> str:
        if not data:
            return "<p>No context rows were generated.</p>"
        headers = list(data[0].keys())
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = ""
        for row in data:
            body += "<tr>" + "".join(f"<td>{fmt(row.get(h))}</td>" for h in headers) + "</tr>"
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def asset_list(paths: Optional[Sequence[str | Path]], heading: str) -> str:
        if not paths:
            return ""
        items = "".join(f"<li>{html.escape(Path(p).name)}</li>" for p in paths)
        return f"<h2>{heading}</h2><ul>{items}</ul>"

    bins = ", ".join(label for _, label, _, _ in HU_BIN_SPECS)
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body {{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;color:#222;}}
table {{border-collapse:collapse;width:100%;margin:16px 0 28px;font-size:12px;}}
th,td {{border:1px solid #ddd;padding:6px 8px;text-align:right;}} th:first-child,td:first-child {{text-align:left;}} th {{background:#f4f4f4;}}
.notice {{background:#fff3cd;border:1px solid #ffe08a;padding:12px 14px;border-radius:8px;}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class='notice'><b>Research use only.</b> Shell HU values test whether voxels around an nnU-Net plaque core contain lower-attenuation plaque-like tissue. This is not clinically validated.</p>
<p>HU bins: {html.escape(bins)}</p>
<h2>HU context summary</h2>{table(rows)}
{asset_list(histogram_paths, "Histogram files")}
{asset_list(overlay_paths, "Overlay files")}
</body></html>"""
    output_html.write_text(document, encoding="utf-8")
    return output_html
