from __future__ import annotations

import json
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


DEFAULT_TASKS = [
    "total",
    "total_highres",              # present in some UI/task pickers; runtime registry decides support
    "heartchambers_highres",
    "coronary_arteries",
    "coronary_arteries_LEGACY",
    "aortic_sinuses",
    "aorta_annulus",
    "aortic_dissection",
    "pulmonary_artery_landmarks",
]

PRIOR_BATCH_RELATIVE = {
    "total": "03_whole_heart/total",
    "heartchambers_highres": "03_whole_heart/heartchambers_highres",
    "coronary_arteries": "02_coronary_ensemble/current",
    "coronary_arteries_LEGACY": "02_coronary_ensemble/legacy",
    "aortic_sinuses": "03_whole_heart/aortic_sinuses",
}


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def reconstruct_source(cache_dir, out="/content/source_ccta_totalseg_cache.nii.gz"):
    cache = Path(cache_dir)
    arr = np.load(cache / "series7_int16.npy", mmap_mode="r")
    meta = json.loads((cache / "series7_int16.json").read_text())
    img = sitk.GetImageFromArray(arr)
    sp = np.asarray(meta["spacing_zyx"], float)
    img.SetSpacing(tuple(sp[::-1]))
    img.SetOrigin(tuple(np.asarray(meta["positions_lps_mm"][0], float)))
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    direction = np.array([[row[0], col[0], slc[0]], [row[1], col[1], slc[1]], [row[2], col[2], slc[2]]], float)
    img.SetDirection(tuple(direction.ravel()))
    sitk.WriteImage(img, str(out))
    return Path(out)


def installed_registry():
    from totalsegmentator.registry import task_registry
    return task_registry()


def _count_masks(folder: Path):
    return len(list(folder.glob("*.nii.gz")))


def _folder_bytes(folder: Path):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _copy_prior_if_available(task, prior_batch_root, dst):
    rel = PRIOR_BATCH_RELATIVE.get(task)
    if not rel:
        return False
    src = Path(prior_batch_root) / rel
    if not (src.exists() and (src / "_SUCCESS").exists()):
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "_SUCCESS").write_text("copied-from-prior-gpu-batch\n")
    (dst / "_CACHE_SOURCE.json").write_text(json.dumps({"source": "GPU_Batch_Pipeline_v1", "task": task, "original_folder": str(src)}, indent=2))
    return True


def run_task(source, out_dir, task, *, reuse=True, save_probabilities=False):
    out_dir = Path(out_dir)
    done = out_dir / "_SUCCESS"
    if reuse and done.exists():
        return {"status": "REUSED", "seconds": 0.0}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    cmd = ["TotalSegmentator", "-i", str(source), "-o", str(out_dir), "-ta", task, "--device", "gpu"]
    if save_probabilities:
        cmd += ["--save_probabilities", str(out_dir / f"{task}_probabilities.npz")]
    start = time.time()
    with log.open("w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    if proc.returncode != 0:
        return {"status": "ERROR", "seconds": seconds, "error": f"exit code {proc.returncode}", "log": str(log)}
    done.write_text("done\n")
    return {"status": "COMPLETE", "seconds": seconds, "log": str(log)}


def build_html(root, df, registry_snapshot, requested_tasks):
    html = Path(root) / "OPENPLAQUE_TOTALSEG_CARDIOVASCULAR_CACHE_REPORT.html"
    rows = ["<html><head><meta charset='utf-8'><title>OpenPlaque TotalSegmentator cardiovascular cache</title></head><body>",
            "<h1>OpenPlaque TotalSegmentator cardiovascular cache</h1>",
            "<p>Durable cache of cardiovascular TotalSegmentator masks from the source CCTA. This report is an inventory; the NIfTI masks remain in per-task Drive folders.</p>",
            "<h2>Requested tasks</h2><pre>" + "\n".join(requested_tasks) + "</pre>",
            "<h2>Run/cache manifest</h2>", df.to_html(index=False),
            "<h2>Installed TotalSegmentator registry</h2>",
            f"<p>Version: {registry_snapshot.get('totalsegmentator_version')}</p>", "</body></html>"]
    html.write_text("\n".join(rows))
    return html


def run_cache(cfg):
    root = Path(cfg["CACHE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    source = reconstruct_source(cfg["SOURCE_CACHE"])
    registry = installed_registry()
    _write_json(root / "totalsegmentator_registry_snapshot.json", registry)
    available = registry["tasks"]
    requested = list(cfg.get("TASKS", DEFAULT_TASKS))
    rows = []
    for task in requested:
        dst = root / task
        info = available.get(task)
        if info is None:
            rows.append({"task": task, "status": "UNSUPPORTED_BY_INSTALLED_CLI", "license_required": None, "n_classes": None, "n_masks": 0, "size_mb": 0.0, "runtime_min": 0.0, "output_folder": str(dst), "note": "Task appears in some UI/task-picker contexts but is not in the installed CLI registry."})
            continue
        license_required = bool(info.get("license_required"))
        if license_required and not cfg.get("LICENSE_ACTIVE", False):
            rows.append({"task": task, "status": "SKIPPED_NO_LICENSE", "license_required": True, "n_classes": len(info.get("classes", {})), "n_masks": 0, "size_mb": 0.0, "runtime_min": 0.0, "output_folder": str(dst), "note": "Licensed task; activate TotalSegmentator license first."})
            continue
        result = None
        if cfg.get("IMPORT_PRIOR_BATCH", True) and not (dst / "_SUCCESS").exists():
            if _copy_prior_if_available(task, cfg["PRIOR_BATCH_ROOT"], dst):
                result = {"status": "IMPORTED_PRIOR_BATCH", "seconds": 0.0}
        if result is None:
            try:
                result = run_task(source, dst, task, reuse=cfg.get("REUSE", True), save_probabilities=task in set(cfg.get("PROBABILITY_TASKS", [])))
            except Exception as e:
                result = {"status": "ERROR", "seconds": 0.0, "error": str(e)}
                dst.mkdir(parents=True, exist_ok=True)
                (dst / "python_exception.txt").write_text(traceback.format_exc())
        rows.append({"task": task, "status": result.get("status"), "license_required": license_required, "n_classes": len(info.get("classes", {})), "n_masks": _count_masks(dst) if dst.exists() else 0, "size_mb": round(_folder_bytes(dst) / (1024**2), 2) if dst.exists() else 0.0, "runtime_min": round(float(result.get("seconds", 0.0)) / 60.0, 2), "output_folder": str(dst), "note": result.get("error", "")})
        pd.DataFrame(rows).to_csv(root / "cache_manifest.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(root / "cache_manifest.csv", index=False)
    summary = {"status": "COMPLETE_WITH_SKIPS" if any(df["status"].isin(["UNSUPPORTED_BY_INSTALLED_CLI", "SKIPPED_NO_LICENSE", "ERROR"])) else "COMPLETE", "totalsegmentator_version": registry.get("totalsegmentator_version"), "requested_tasks": requested, "completed_or_reused": df[df["status"].isin(["COMPLETE", "REUSED", "IMPORTED_PRIOR_BATCH"])]["task"].tolist(), "unsupported": df[df["status"] == "UNSUPPORTED_BY_INSTALLED_CLI"]["task"].tolist(), "errors": df[df["status"] == "ERROR"]["task"].tolist()}
    _write_json(root / "summary.json", summary)
    html = build_html(root, df, registry, requested)
    report_zip = root / "OPENPLAQUE_TOTALSEG_CARDIOVASCULAR_CACHE_REPORT_BACK.zip"
    with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [html, root / "summary.json", root / "cache_manifest.csv", root / "totalsegmentator_registry_snapshot.json"]:
            z.write(p, p.name)
    return summary, report_zip
