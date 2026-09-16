from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from openplaque import lcx_distal_source_extension as base

ALGORITHM = "lcx-distal-source-extension-v1.0-guarded"
OUTPUT_DIRNAME = base.OUTPUT_DIRNAME
STATUS_TRUNCATED = "DISTAL_SOURCE_EXTENSION_CROP_TRUNCATED"

def _boundary_margin_mm(point_lps, g, lo, shape):
    q = g.lps_to_zyx(np.asarray(point_lps, float))[0] - np.asarray(lo, float)
    shape = np.asarray(shape, float)
    low = q * g.spacing_zyx
    high = (shape - 1.0 - q) * g.spacing_zyx
    return float(np.min(np.r_[low, high]))

def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    r = base.run(drive_root, output_dir)
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    g, _ = base._load_source(root / base.SOURCE_CACHE)
    summary = json.loads((out / "summary.json").read_text())
    margins = {}
    truncated = False
    for label in ("c7", "c96"):
        meta = json.loads((out / f"{label}_search_meta.json").read_text())
        p = out / f"{label}_selected_extension.csv"
        if not p.exists():
            margins[label] = None
            continue
        d = pd.read_csv(p)
        pt = d[["lps_x_mm","lps_y_mm","lps_z_mm"]].to_numpy(float)[-1]
        m = _boundary_margin_mm(pt, g, meta["roi_lo_zyx"], meta["roi_shape"])
        margins[label] = m
        if m < 2.0:
            truncated = True
    summary["endpoint_roi_boundary_margin_mm"] = margins
    summary["crop_boundary_guard_mm"] = 2.0
    summary["algorithm"] = ALGORITHM
    if truncated:
        summary["pre_boundary_guard_status"] = summary["status"]
        summary["status"] = STATUS_TRUNCATED
        summary["scientific_boundary"] = (
            "At least one source-selected extension terminated within 2 mm of its search-crop boundary; "
            "the result is treated as truncated and cannot adjudicate LCX identity."
        )
    base._write_json(out / "summary.json", summary)
    base._write_json(out / "run_state.json", {
        "status": "COMPLETE", "scientific_status": summary["status"],
        "algorithm": ALGORITHM, "baseline_commit": base.BASELINE
    })
    report = out / "OPENPLAQUE_LCX_DISTAL_SOURCE_EXTENSION_REPORT.html"
    old = report.read_text(encoding="utf-8")
    old = old.replace("<h1>OpenPlaque LCX distal source extension</h1>",
                      "<h1>OpenPlaque LCX distal source extension</h1>"
                      f"<p><b>Boundary-guarded status:</b> {summary['status']}</p>"
                      f"<p><b>Endpoint ROI margins:</b> {margins}</p>")
    report.write_text(old, encoding="utf-8")
    zp = out / "OPENPLAQUE_LCX_DISTAL_SOURCE_EXTENSION_REPORT_BACK.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.name != zp.name and not p.name.endswith("_vesselness.npy") and not p.name.endswith("_best_scale.npy"):
                z.write(p, arcname=p.name)
    return {"summary": summary, "report": str(report), "zip": str(zp)}
