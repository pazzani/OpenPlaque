"""HTML report generation for OpenPlaque TPV workflows."""

from pathlib import Path
from html import escape
from typing import Iterable, Mapping, Optional
import base64
import numpy as np


def _mask_positive(mask, label=2):
    arr = np.asarray(mask)
    if label is None or (arr.size and np.nanmax(arr) <= 1):
        return arr > 0
    return arr == label


def _best_slice(mask, label=2):
    pos = _mask_positive(mask, label=label)
    counts = np.sum(pos, axis=(1, 2))
    return int(np.argmax(counts)) if counts.size else 0


def save_overlay_png(volume, mask, path, title, z=None, label=2, vmin=-200, vmax=800):
    import matplotlib.pyplot as plt
    if z is None:
        z = _best_slice(mask, label=label)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.imshow(_mask_positive(mask, label=label)[z], alpha=0.45)
    plt.title(f"{title} — slice {z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _img_data_uri(path):
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _fmt(x):
    return f"{float(x):.2f}"


def write_html_report(
    output_html,
    reports: Iterable,
    refined_results: Mapping[str, object],
    core_results: Mapping[str, object],
    uncertainty_summary,
    overlay_dir: Optional[str] = None,
    title: str = "OpenPlaque TPV Boundary Refinement Report",
):
    """Write a self-contained HTML report with TPV tables and overlay PNGs."""
    output_html = Path(output_html)
    overlay_dir = Path(overlay_dir) if overlay_dir else output_html.with_suffix("").parent / "openplaque_report_assets"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    report_list = list(reports)
    overlay_paths = {}
    for report in report_list:
        name = report.name
        raw_path = save_overlay_png(report.volume, report.mask, overlay_dir / f"{name}_raw_overlay.png", f"{name} raw plaque")
        refined_path = save_overlay_png(report.volume, refined_results[name].refined_mask, overlay_dir / f"{name}_refined_overlay.png", f"{name} refined plaque")
        core_path = save_overlay_png(report.volume, core_results[name].refined_mask, overlay_dir / f"{name}_core_overlay.png", f"{name} high-confidence core")
        overlay_paths[name] = (raw_path, refined_path, core_path)

    rows_html = []
    for row in uncertainty_summary.rows():
        rows_html.append(
            "<tr>"
            f"<td>{escape(row['vessel'])}</td>"
            f"<td>{_fmt(row['core_tpv_mm3'])}</td>"
            f"<td>{_fmt(row['refined_tpv_mm3'])}</td>"
            f"<td>{_fmt(row['raw_tpv_mm3'])}</td>"
            f"<td>{_fmt(row['removed_boundary_mm3'])}</td>"
            f"<td>{_fmt(row['interval_low_mm3'])} – {_fmt(row['interval_high_mm3'])}</td>"
            f"<td>{_fmt(row['uncertainty_width_mm3'])}</td>"
            "</tr>"
        )

    overlay_html = []
    for report in report_list:
        name = report.name
        raw, refined, core = overlay_paths[name]
        overlay_html.append(f"<h2>{escape(name)} overlays</h2><div class='grid'>")
        for label, path in (("Raw", raw), ("Refined", refined), ("Core", core)):
            overlay_html.append(
                f"<figure><img src='{_img_data_uri(path)}' alt='{escape(name)} {label} overlay'>"
                f"<figcaption>{escape(name)} {label}</figcaption></figure>"
            )
        overlay_html.append("</div>")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #222; }}
h1 {{ margin-bottom: 0.25rem; }}
.notice {{ padding: 12px 14px; background: #fff3cd; border: 1px solid #ffe08a; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0 30px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f6f6f6; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; }}
figure {{ margin: 0; }}
img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
figcaption {{ text-align: center; font-size: 0.9rem; color: #555; margin-top: 6px; }}
.small {{ color: #666; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="notice"><strong>Research use only.</strong> This report is not clinically validated and is not for diagnosis or medical decision-making.</p>
<p class="small">Uncertainty interval is reported as high-confidence core TPV to raw AI TPV, with refined TPV as the midpoint-style estimate.</p>
<h2>TPV uncertainty summary</h2>
<table>
<thead><tr><th>Vessel</th><th>Core TPV mm³</th><th>Refined TPV mm³</th><th>Raw TPV mm³</th><th>Removed boundary mm³</th><th>Interval mm³</th><th>Width mm³</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
{''.join(overlay_html)}
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
