from __future__ import annotations

import base64
import gc
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from .study import OpenPlaqueStudy
from .left_main_coronary import arc_mm, resample_path, plane_lumen_metrics
from .left_main_memory_safe import detect_left_ostia, trace_routes

ALGORITHM_VERSION = "left-main-lowram-v1.2"
SOURCE_SERIES = 7


def _json_write(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        return float("nan")


def _ram(label):
    print(f"{label}: RSS {_rss_gb():.2f} GB")


def _stage_evidence_npz(npz_path, stage_dir):
    """Unpack one NPZ member at a time to local .npy files, then memory-map them."""
    npz_path = Path(npz_path)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    wanted = (
        "ct", "aorta", "support", "lumen_radius_mm", "dist_aorta_mm", "cost",
        "lo_source_zyx", "hi_source_zyx", "zoom_zyx", "spacing_zyx",
    )
    z = np.load(npz_path, allow_pickle=False)
    try:
        for key in wanted:
            out = stage_dir / f"{key}.npy"
            if out.exists():
                continue
            if key not in z.files:
                raise KeyError(f"{key} missing from {npz_path}")
            arr = z[key]
            np.save(out, arr, allow_pickle=False)
            del arr
            gc.collect()
    finally:
        z.close()
    evd = {}
    for key in wanted:
        evd[key] = np.load(stage_dir / f"{key}.npy", mmap_mode="r")
    evd["aorta"] = np.asarray(evd["aorta"], dtype=bool)
    return evd


def _source_to_ds(path_source_zyx, lo_zyx, zoom_zyx):
    return (np.asarray(path_source_zyx, float) - np.asarray(lo_zyx, float)) * np.asarray(zoom_zyx, float)


def _ds_to_source(path_ds_zyx, lo_zyx, zoom_zyx):
    return np.asarray(lo_zyx, float) + np.asarray(path_ds_zyx, float) / np.asarray(zoom_zyx, float)


def _orthogonal_basis(tangent_zyx_mm):
    t = np.asarray(tangent_zyx_mm, float)
    t /= max(np.linalg.norm(t), 1e-8)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, ref)
    u /= max(np.linalg.norm(u), 1e-8)
    v = np.cross(t, u)
    v /= max(np.linalg.norm(v), 1e-8)
    return u, v


def orthogonal_plane_lowram(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=6.0, pix_mm=0.18):
    """Sample a tiny plane without converting the full CT to float64."""
    u, v = _orthogonal_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    coords = [vox[..., 0], vox[..., 1], vox[..., 2]]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, coords, output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def serial_lumen_qc_lowram(path_source_zyx, ct, spacing_zyx, rca_calibration, n_samples=12, label_name="LEFT_MAIN"):
    p = resample_path(path_source_zyx, spacing_zyx, 0.45)
    s = arc_mm(p, spacing_zyx)
    if len(p) < 5 or s[-1] < 2:
        return pd.DataFrame(), {
            "label": label_name, "length_mm": float(s[-1]) if len(s) else 0.0,
            "median_plane_score": 0.0, "plane_pass_fraction": 0.0,
        }
    sample_s = np.linspace(min(1.2, s[-1] * 0.08), max(min(1.2, s[-1] * 0.08), s[-1] - 1.2), n_samples)
    rows = []
    rca_r = float(rca_calibration.get("median_radius_mm", 1.8))
    rca_h = float(rca_calibration.get("median_center_hu", 550))
    for ss in sample_s:
        i = int(np.argmin(np.abs(s - ss)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing_zyx, float)
        if np.linalg.norm(t) < 1e-6:
            continue
        im, c = orthogonal_plane_lowram(ct, p[i], t, spacing_zyx)
        m = plane_lumen_metrics(im, c)
        r = m["radius_mm"]
        radius_score = float(np.exp(-0.5 * ((r / (1.25 * rca_r + 1e-6) - 1) / 0.60) ** 2)) if np.isfinite(r) else 0.0
        offset_score = float(np.exp(-0.5 * (m["centroid_offset_mm"] / 0.85) ** 2)) if np.isfinite(m["centroid_offset_mm"]) else 0.0
        circ_score = float(np.clip(m["circularity"] / 0.55, 0, 1))
        contrast_score = float(1 / (1 + np.exp(-(m["core_minus_ring_hu"] - 20) / 90))) if np.isfinite(m["core_minus_ring_hu"]) else 0.0
        hu_score = float(np.exp(-0.5 * ((m["center_hu"] - rca_h) / 330) ** 2))
        score = 0.30 * radius_score + 0.27 * offset_score + 0.20 * circ_score + 0.14 * contrast_score + 0.09 * hu_score
        pass_plane = (
            np.isfinite(r)
            and 0.45 * rca_r <= r <= min(5.0, 2.3 * rca_r + 0.4)
            and m["centroid_offset_mm"] <= 1.45
            and m["circularity"] >= 0.23
            and 120 <= m["center_hu"] <= 1100
        )
        rows.append({"label": label_name, "arc_mm": float(s[i]), **m, "plane_score": float(score), "plane_pass": bool(pass_plane)})
    df = pd.DataFrame(rows)
    summary = {
        "label": label_name,
        "length_mm": float(s[-1]),
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else np.nan,
        "median_offset_mm": float(df.centroid_offset_mm.median()) if len(df) else np.nan,
        "median_circularity": float(df.circularity.median()) if len(df) else np.nan,
        "median_center_hu": float(df.center_hu.median()) if len(df) else np.nan,
        "median_core_minus_ring_hu": float(df.core_minus_ring_hu.median()) if len(df) else np.nan,
    }
    return df, summary


def _series7_files(root, extract_root="/content/full_dicom_left_main_lowram"):
    root = Path(root)
    dz = root / "Full_DICOM.zip"
    if not dz.exists():
        raise FileNotFoundError(dz)
    lz = Path("/content/Full_DICOM.zip")
    if not lz.exists() or lz.stat().st_size != dz.stat().st_size:
        shutil.copyfile(dz, lz)
    study = OpenPlaqueStudy(str(lz), extract_root=extract_root)
    match = [s for s in study.series if s["series_number"] == SOURCE_SERIES]
    if not match:
        raise RuntimeError("Series 7 not found")
    uid = match[0]["uid"]
    folder = match[0]["folder"]
    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(folder, uid))
    if not files:
        raise RuntimeError("No DICOM files for series 7")
    return files


def stream_source_ct_to_memmap(root, out_path="/content/openplaque_series7_int16.npy", persistent_cache=None, reuse=True):
    """Read one DICOM slice at a time into a local int16 HU memmap, with optional persistent reuse."""
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    persistent_cache = Path(persistent_cache) if persistent_cache is not None else None
    persistent_meta = persistent_cache.with_suffix(".json") if persistent_cache is not None else None
    if reuse and persistent_cache is not None and persistent_cache.exists() and persistent_meta.exists():
        if not out_path.exists() or out_path.stat().st_size != persistent_cache.stat().st_size:
            shutil.copyfile(persistent_cache, out_path)
        shutil.copyfile(persistent_meta, meta_path)
        meta = _json_read(meta_path)
        return np.load(out_path, mmap_mode="r"), meta
    if reuse and out_path.exists() and meta_path.exists():
        meta = _json_read(meta_path)
        return np.load(out_path, mmap_mode="r"), meta

    files = _series7_files(root)
    ds0 = pydicom.dcmread(files[0], force=True)
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    n = len(files)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int16, shape=(n, rows, cols))
    positions = []
    for i, fp in enumerate(files):
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        mm[i] = np.rint(np.clip(hu, -32768, 32767)).astype(np.int16)
        positions.append([float(x) for x in getattr(ds, "ImagePositionPatient", (0, 0, i))])
        if i % 100 == 0:
            mm.flush()
    mm.flush()
    ps = [float(x) for x in ds0.PixelSpacing]
    pos = np.asarray(positions, float)
    if len(pos) >= 2:
        dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0))
    meta = {
        "shape": [n, rows, cols],
        "spacing_zyx": [dz, ps[0], ps[1]],
        "image_orientation_patient": [float(x) for x in getattr(ds0, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0))],
        "positions_lps_mm": positions,
    }
    _json_write(meta, meta_path)
    del mm
    gc.collect()
    if persistent_cache is not None:
        persistent_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, persistent_cache)
        shutil.copyfile(meta_path, persistent_meta)
    return np.load(out_path, mmap_mode="r"), meta


def _save_path_csv(path_source, spacing_zyx, out_path):
    p = resample_path(path_source, spacing_zyx, 0.6)
    s = arc_mm(p, spacing_zyx)
    pd.DataFrame({"arc_mm": s, "z": p[:, 0], "y": p[:, 1], "x": p[:, 2]}).to_csv(out_path, index=False)
    return p


class LeftMainLowRamWorkflow:
    """Left-main-only acceptance gate. Distal LAD/LCX search is intentionally deferred."""

    COMPONENTS = ("ostia", "graph_candidates", "source_ct", "candidate_qc", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Left_Main_Bifurcation_v1"
        self.out = self.root / "Left_Main_LowRAM_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage = Path("/content/openplaque_evidence_shards")
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            unknown = set(reuse) - set(self.COMPONENTS)
            if unknown:
                raise ValueError(f"Unknown reuse controls: {sorted(unknown)}")
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.evd = None
        self.ostia = None
        self.graph_candidates = []
        self.ct = None
        self.ct_meta = None
        self.rca_cal = None
        self.best = None
        self.best_qc = None
        self.best_summary = None

    def stage_evidence(self):
        npz = self.cache / "source_evidence.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Need prior source evidence cache: {npz}")
        self.evd = _stage_evidence_npz(npz, self.stage)
        _ram("After staging evidence")
        return self.evd

    def load_rca_calibration(self):
        fp = self.cache / "rca_calibration.json"
        if not fp.exists():
            raise FileNotFoundError(f"Need prior RCA calibration: {fp}")
        self.rca_cal = _json_read(fp)
        return self.rca_cal

    def detect_ostia(self):
        fp = self.out / "left_ostium_candidates.csv"
        if self.reuse["ostia"] and fp.exists():
            self.ostia = pd.read_csv(fp)
            print("Reused:", fp)
            return self.ostia
        if self.evd is None:
            self.stage_evidence()
        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        rca = pd.read_csv(rca_fp)[["z", "y", "x"]].to_numpy(float)
        rca_ds = _source_to_ds(rca[0], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
        self.ostia = detect_left_ostia(self.evd, rca_ds, topn=12, preselect=500)
        self.ostia.to_csv(fp, index=False)
        _ram("After ostium detection")
        return self.ostia

    def build_graph_candidates(self):
        table_fp = self.out / "left_main_graph_candidates.csv"
        cdir = self.out / "candidate_paths"
        if self.reuse["graph_candidates"] and table_fp.exists() and cdir.exists():
            table = pd.read_csv(table_fp)
            paths = sorted(cdir.glob("candidate_*.npy"))
            if len(paths) >= len(table):
                self.graph_candidates = [
                    {"path_source": np.load(paths[i]), "rec": table.iloc[i].to_dict()}
                    for i in range(len(table))
                ]
                print("Reused:", table_fp)
                return table
        if self.ostia is None:
            self.detect_ostia()
        rows = []
        candidates = []
        for _, o in self.ostia.head(8).iterrows():
            seed = np.array([o.z, o.y, o.x], float)
            heading = np.array([o.dir_z, o.dir_y, o.dir_x], float)
            routes = trace_routes(
                self.evd, seed, heading,
                min_len=6, max_len=30, max_routes=8,
                min_forward=2.5, support_percentile=72,
            )
            for r in routes:
                src = _ds_to_source(r["path"], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
                rec = {
                    "ostium_rank": int(o["rank"]),
                    "graph_score": float(r["graph_score"]),
                    "length_mm_graph": float(r["length_mm"]),
                    "mean_support": float(r["mean_support"]),
                    "p10_support": float(r["p10_support"]),
                    "mean_hu_ds": float(r["mean_hu"]),
                    "p10_hu_ds": float(r["p10_hu"]),
                    "end_dist_aorta_mm": float(r["end_dist_aorta_mm"]),
                    "heading_angle_deg": float(r["heading_angle_deg"]),
                }
                rows.append(rec)
                candidates.append({"path_source": src, "rec": rec})
        if not candidates:
            raise RuntimeError("No short left-main graph candidates")
        order = np.argsort([c["rec"]["graph_score"] for c in candidates])[::-1][:24]
        self.graph_candidates = [candidates[i] for i in order]
        table = pd.DataFrame([c["rec"] for c in self.graph_candidates])
        table.insert(0, "graph_rank", np.arange(1, len(table) + 1))
        table.to_csv(table_fp, index=False)
        cdir.mkdir(exist_ok=True)
        for i, c in enumerate(self.graph_candidates, 1):
            np.save(cdir / f"candidate_{i:02d}.npy", c["path_source"].astype(np.float32))
        _ram("After graph candidates")
        return table

    def release_evidence(self):
        self.evd = None
        gc.collect()
        _ram("After releasing evidence")

    def load_source_ct(self):
        if self.evd is not None:
            self.release_evidence()
        persistent = self.cache / "series7_int16.npy"
        self.ct, self.ct_meta = stream_source_ct_to_memmap(
            self.root,
            out_path="/content/openplaque_series7_int16.npy",
            persistent_cache=persistent,
            reuse=self.reuse["source_ct"],
        )
        _ram("After source CT memmap")
        return self.ct

    def qc_candidates(self):
        summary_fp = self.out / "left_main_best_summary.json"
        best_fp = self.out / "left_main_best_centerline.csv"
        serial_fp = self.out / "left_main_best_serial_qc.csv"
        if self.reuse["candidate_qc"] and summary_fp.exists() and best_fp.exists() and serial_fp.exists():
            self.best_summary = _json_read(summary_fp)
            self.best = pd.read_csv(best_fp)[["z", "y", "x"]].to_numpy(float)
            self.best_qc = pd.read_csv(serial_fp)
            print("Reused:", summary_fp)
            return self.best_summary
        if not self.graph_candidates:
            cdir = self.out / "candidate_paths"
            table = pd.read_csv(self.out / "left_main_graph_candidates.csv")
            self.graph_candidates = []
            for i, row in table.iterrows():
                self.graph_candidates.append({
                    "path_source": np.load(cdir / f"candidate_{i+1:02d}.npy"),
                    "rec": row.to_dict(),
                })
        if self.ct is None:
            self.load_source_ct()
        if self.rca_cal is None:
            self.load_rca_calibration()
        spacing = np.asarray(self.ct_meta["spacing_zyx"], float)
        rows = []
        scored = []
        for i, c in enumerate(self.graph_candidates, 1):
            qdf, qsum = serial_lumen_qc_lowram(
                c["path_source"], self.ct, spacing, self.rca_cal,
                n_samples=12, label_name=f"LM_candidate_{i}",
            )
            length_quality = float(np.exp(-0.5 * ((qsum["length_mm"] - 14.0) / 8.0) ** 2))
            graph_score = float(c["rec"].get("graph_score", 0.0))
            combined = 0.36 * graph_score + 0.40 * qsum["median_plane_score"] + 0.20 * qsum["plane_pass_fraction"] + 0.04 * length_quality
            rec = {"candidate": i, "combined_score": combined, **c["rec"], **{f"serial_{k}": v for k, v in qsum.items() if k != "label"}}
            rows.append(rec)
            scored.append((combined, c, qdf, qsum))
            if i % 6 == 0:
                gc.collect()
        scored.sort(key=lambda x: x[0], reverse=True)
        combined, c, qdf, qsum = scored[0]
        self.best = resample_path(c["path_source"], spacing, 0.6)
        self.best_qc = qdf
        self.best_summary = dict(qsum)
        self.best_summary["combined_score"] = float(combined)
        L = self.best_summary["length_mm"]
        pf = self.best_summary["plane_pass_fraction"]
        ms = self.best_summary["median_plane_score"]
        self.best_summary["status"] = "OK" if (5 <= L <= 30 and pf >= 0.70 and ms >= 0.55) else ("REVIEW" if pf >= 0.50 and ms >= 0.45 else "FAIL")
        pd.DataFrame(rows).sort_values("combined_score", ascending=False).to_csv(self.out / "left_main_candidate_qc.csv", index=False)
        self.best_qc.to_csv(serial_fp, index=False)
        _save_path_csv(self.best, spacing, best_fp)
        _json_write(self.best_summary, summary_fp)
        _ram("After candidate QC")
        return self.best_summary

    def plot_qc(self):
        f1 = self.out / "01_left_main_lowram_mips.png"
        f2 = self.out / "02_left_main_lowram_cross_sections.png"
        if self.reuse["figures"] and f1.exists() and f2.exists():
            print("Reused figures")
            return [f1, f2]
        if self.best is None:
            self.qc_candidates()
        if self.ct is None:
            self.load_source_ct()
        spacing = np.asarray(self.ct_meta["spacing_zyx"], float)
        p = self.best
        pad = np.ceil(np.array([16.0, 22.0, 22.0]) / spacing).astype(int)
        lo = np.maximum(0, np.floor(p.min(axis=0)).astype(int) - pad)
        hi = np.minimum(np.asarray(self.ct.shape), np.ceil(p.max(axis=0)).astype(int) + pad + 1)
        roi = np.asarray(self.ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.int16)
        fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
        ims = [roi.max(axis=0), roi.max(axis=1), roi.max(axis=2)]
        titles = ["Axial MIP", "Coronal MIP", "Sagittal MIP"]
        q = p - lo
        coords = [(q[:, 2], q[:, 1]), (q[:, 2], q[:, 0]), (q[:, 1], q[:, 0])]
        for ax, im, title, xy in zip(axs, ims, titles, coords):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower")
            ax.plot(xy[0], xy[1], linewidth=2)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Best left-main candidate — {self.best_summary['status']}")
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(f1, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        s = arc_mm(p, spacing)
        sample_s = np.linspace(min(1.2, s[-1] * 0.08), max(min(1.2, s[-1] * 0.08), s[-1] - 1.2), 12)
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        for ax, ss in zip(axs.ravel(), sample_s):
            i = int(np.argmin(np.abs(s - ss)))
            i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
            t = (p[i1] - p[i0]) * spacing
            im, c = orthogonal_plane_lowram(self.ct, p[i], t, spacing)
            ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=[c[0], c[-1], c[0], c[-1]], origin="lower")
            ax.plot(0, 0, "+", markersize=9)
            ax.set_title(f"{s[i]:.1f} mm")
            ax.set_aspect("equal")
        fig.suptitle("Serial orthogonal cross-sections — best left-main candidate")
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(f2, dpi=180, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return [f1, f2]

    def package(self):
        zpath = self.out / "OPENPLAQUE_LEFT_MAIN_LOWRAM_REPORT_BACK.zip"
        if self.reuse["report"] and zpath.exists():
            print("Reused:", zpath)
            return zpath
        self.plot_qc()
        html = self.out / "OPENPLAQUE_LEFT_MAIN_LOWRAM_REPORT.html"

        def img(name):
            fp = self.out / name
            b64 = base64.b64encode(fp.read_bytes()).decode()
            return f"<h2>{name}</h2><img style='max-width:100%' src='data:image/png;base64,{b64}'>"

        cand = pd.read_csv(self.out / "left_main_candidate_qc.csv")
        serial = pd.read_csv(self.out / "left_main_best_serial_qc.csv")
        html.write_text(
            "<html><body><h1>OpenPlaque — low-RAM left-main acceptance gate</h1>"
            "<p><b>Research use only.</b> This run intentionally stops before LAD/LCX branching. "
            "The left main must pass serial source-volume lumen QC first.</p>"
            + img("01_left_main_lowram_mips.png")
            + img("02_left_main_lowram_cross_sections.png")
            + "<h2>Best summary</h2><pre>" + json.dumps(self.best_summary, indent=2) + "</pre>"
            + "<h2>Candidate ranking</h2>" + cand.head(24).to_html(index=False)
            + "<h2>Serial QC</h2>" + serial.to_html(index=False)
            + "</body></html>",
            encoding="utf-8",
        )
        names = [
            "01_left_main_lowram_mips.png", "02_left_main_lowram_cross_sections.png",
            "left_ostium_candidates.csv", "left_main_graph_candidates.csv",
            "left_main_candidate_qc.csv", "left_main_best_serial_qc.csv",
            "left_main_best_centerline.csv", "left_main_best_summary.json",
            html.name,
        ]
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                fp = self.out / n
                if fp.exists():
                    z.write(fp, arcname=n)
        return zpath
