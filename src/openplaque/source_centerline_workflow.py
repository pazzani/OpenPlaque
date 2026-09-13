from __future__ import annotations

import base64
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from .source_coronary import (
    ALGORITHM_VERSION,
    arc_mm,
    build_downsampled_evidence,
    centerline_qc,
    detect_left_ostium,
    ds_to_source,
    generate_rotating_cpr,
    parallel_transport_frame,
    physical_lps_from_zyx,
    resample_path,
    select_left_branch_pair,
    source_to_ds,
    trace_rca,
    trace_routes,
)
from .study import OpenPlaqueStudy

VESSELS = ("RCA", "LAD", "LCX")
SOURCE_SERIES = 7


def _json_dump(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_path_csv(path_zyx, spacing_zyx, image, path):
    p = resample_path(path_zyx, spacing_zyx, 0.6)
    s = arc_mm(p, spacing_zyx)
    phys = physical_lps_from_zyx(image, p)
    df = pd.DataFrame({
        "arc_mm": s,
        "z": p[:, 0], "y": p[:, 1], "x": p[:, 2],
        "lps_x_mm": phys[:, 0], "lps_y_mm": phys[:, 1], "lps_z_mm": phys[:, 2],
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _load_path_csv(path):
    df = pd.read_csv(path)
    return df[["z", "y", "x"]].to_numpy(float)


class SourceCenterlineWorkflow:
    COMPONENTS = (
        "source_evidence",
        "rca_centerline",
        "left_ostium",
        "left_branches",
        "cpr_stacks",
        "qc_figures",
        "report_package",
    )

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.out = self.root / "Source_Volume_Coronary_Centerlines_Report"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "Cache" / "Source_Volume_Coronary_Centerlines"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            unknown = set(reuse) - set(self.COMPONENTS)
            if unknown:
                raise ValueError(f"Unknown reuse flags: {sorted(unknown)}")
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.study = self.image = self.ct = self.aorta = None
        self.spacing_zyx = None
        self.evd = None
        self.paths = {}
        self.cpr = {}
        self.left_seed_ds = self.left_dir = None
        self.left_candidates = None
        self.centerline_qc_df = None
        self.provenance = []
        self._done = set()

    def _record(self, component, action, path, note=""):
        self.provenance.append({
            "component": component,
            "reuse_requested": self.reuse[component],
            "action": action,
            "path": str(path),
            "note": note,
        })
        pd.DataFrame(self.provenance).to_csv(self.out / "cache_provenance.csv", index=False)

    def _paths(self):
        return {
            "evidence": self.cache / "source_evidence.npz",
            "evidence_meta": self.cache / "source_evidence_meta.json",
            "rca": self.cache / "RCA_source_centerline.csv",
            "left_ostium": self.cache / "left_ostium.json",
            "left_ostium_candidates": self.cache / "left_ostium_candidates.csv",
            "lad": self.cache / "LAD_source_centerline.csv",
            "lcx": self.cache / "LCX_source_centerline.csv",
            "left_pair": self.cache / "left_branch_pair.json",
            "cpr_dir": self.cache / "cpr_stacks",
            "qc_meta": self.cache / "qc_figures.json",
            "report_meta": self.cache / "report.json",
        }

    def cache_status(self):
        p = self._paths()
        cpr_ok = all((p["cpr_dir"] / f"{v}_rotating_cpr.npz").exists() for v in VESSELS)
        qc_ok = p["qc_meta"].exists() and all((self.out / x).exists() for x in ("01_source_centerline_mips.png", "02_centerline_cross_sections.png", "03_generated_cpr_overview.png"))
        states = {
            "source_evidence": p["evidence"].exists() and p["evidence_meta"].exists(),
            "rca_centerline": p["rca"].exists() or (self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv").exists(),
            "left_ostium": p["left_ostium"].exists(),
            "left_branches": p["lad"].exists() and p["lcx"].exists(),
            "cpr_stacks": cpr_ok,
            "qc_figures": qc_ok,
            "report_package": (self.out / "OPENPLAQUE_SOURCE_CENTERLINES_REPORT_BACK.zip").exists() and p["report_meta"].exists(),
        }
        return pd.DataFrame([{
            "component": k,
            "reuse": self.reuse[k],
            "cache_available": states[k],
            "planned_action": "reuse" if self.reuse[k] and states[k] else "recompute_and_cache",
        } for k in self.COMPONENTS])

    def prepare_source(self):
        if self.image is not None:
            return self.image, self.ct
        dz = self.root / "Full_DICOM.zip"
        if not dz.exists():
            raise FileNotFoundError(dz)
        lz = Path("/content/Full_DICOM.zip")
        if not lz.exists() or lz.stat().st_size != dz.stat().st_size:
            shutil.copyfile(dz, lz)
        self.study = OpenPlaqueStudy(str(lz), extract_root="/content/full_dicom_source_centerlines")
        self.image, self.ct, _ = self.study.load_series(SOURCE_SERIES)
        self.ct = np.asarray(self.ct, np.float32)
        self.spacing_zyx = np.asarray(self.image.GetSpacing(), float)[::-1]
        ap = self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz"
        if not ap.exists():
            raise FileNotFoundError(f"Need validated aorta cache: {ap}")
        ai = sitk.ReadImage(str(ap))
        if ai.GetSize() != self.image.GetSize() or not np.allclose(ai.GetSpacing(), self.image.GetSpacing()):
            ai = sitk.Resample(ai, self.image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        self.aorta = sitk.GetArrayFromImage(ai) > 0
        return self.image, self.ct

    def _frozen_rca_path(self):
        fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Need frozen RCA source centerline seed/reference: {fp}")
        df = pd.read_csv(fp)
        return df[["z", "y", "x"]].to_numpy(float), fp

    def prepare_evidence(self):
        if "source_evidence" in self._done:
            return self.evd
        self.prepare_source()
        p = self._paths()
        rca_ref, rca_ref_path = self._frozen_rca_path()
        signature = {
            "algorithm": ALGORITHM_VERSION,
            "dicom_zip_size": int((self.root / "Full_DICOM.zip").stat().st_size),
            "aorta_size": int((self.root / "RCA_Ostium_TotalSegmentator" / "aorta_series7_totalseg.nii.gz").stat().st_size),
            "shape": list(map(int, self.ct.shape)),
            "spacing_zyx": list(map(float, self.spacing_zyx)),
        }
        loaded = False
        if self.reuse["source_evidence"] and p["evidence"].exists() and p["evidence_meta"].exists():
            try:
                meta = _json_load(p["evidence_meta"])
                if meta["signature"] != signature:
                    raise ValueError("source evidence signature changed")
                z = np.load(p["evidence"], allow_pickle=False)
                self.evd = {k: z[k] for k in z.files}
                for k in ("lo_source_zyx", "hi_source_zyx"):
                    self.evd[k] = self.evd[k].astype(int)
                loaded = True
                self._record("source_evidence", "reused", p["evidence"])
            except Exception as e:
                self._record("source_evidence", "cache_invalid", p["evidence"], repr(e))
        if not loaded:
            self.evd = build_downsampled_evidence(self.ct, self.aorta, self.spacing_zyx, rca_ref[0], half_mm=(82, 96, 96), target_mm=1.0)
            np.savez_compressed(
                p["evidence"],
                ct=self.evd["ct"].astype(np.float32),
                aorta=self.evd["aorta"].astype(np.uint8),
                support=self.evd["support"].astype(np.float32),
                lumen_radius_mm=self.evd["lumen_radius_mm"].astype(np.float32),
                dist_aorta_mm=self.evd["dist_aorta_mm"].astype(np.float32),
                cost=self.evd["cost"].astype(np.float32),
                lo_source_zyx=self.evd["lo_source_zyx"].astype(np.int32),
                hi_source_zyx=self.evd["hi_source_zyx"].astype(np.int32),
                zoom_zyx=self.evd["zoom_zyx"].astype(np.float32),
                spacing_zyx=self.evd["spacing_zyx"].astype(np.float32),
            )
            _json_dump({"signature": signature, "rca_reference": str(rca_ref_path)}, p["evidence_meta"])
            self._record("source_evidence", "recomputed_and_cached", p["evidence"])
        self._done.add("source_evidence")
        return self.evd

    def build_rca_centerline(self):
        if "rca_centerline" in self._done:
            return self.paths["RCA"]
        self.prepare_evidence()
        p = self._paths()
        frozen, frozen_path = self._frozen_rca_path()
        if self.reuse["rca_centerline"] and p["rca"].exists():
            path = _load_path_csv(p["rca"])
            action = "reused"
        elif self.reuse["rca_centerline"]:
            path = frozen
            _save_path_csv(path, self.spacing_zyx, self.image, p["rca"])
            action = "imported_frozen_validated_rca"
        else:
            seed_src = frozen[0]
            rr = resample_path(frozen, self.spacing_zyx, 0.8)
            j = min(len(rr)-1, max(3, int(np.argmin(np.abs(arc_mm(rr, self.spacing_zyx)-8.0)))))
            head_src_mm = (rr[j] - rr[0]) * self.spacing_zyx
            head_src_mm /= max(np.linalg.norm(head_src_mm), 1e-6)
            seed_ds = source_to_ds(seed_src, self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            route, _ = trace_rca(self.evd, seed_ds, head_src_mm)
            path = ds_to_source(route["path"], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            _save_path_csv(path, self.spacing_zyx, self.image, p["rca"])
            action = "recomputed_and_cached"
        self.paths["RCA"] = np.asarray(path, float)
        self._record("rca_centerline", action, p["rca"])
        self._done.add("rca_centerline")
        return self.paths["RCA"]

    def detect_left_ostium(self):
        if "left_ostium" in self._done:
            return self.left_seed_ds, self.left_dir
        self.prepare_evidence(); self.build_rca_centerline()
        p = self._paths()
        if self.reuse["left_ostium"] and p["left_ostium"].exists() and p["left_ostium_candidates"].exists():
            d = _json_load(p["left_ostium"])
            self.left_seed_ds = np.asarray(d["seed_ds_zyx"], float)
            self.left_dir = np.asarray(d["outward_direction_zyx_mm"], float)
            self.left_candidates = pd.read_csv(p["left_ostium_candidates"])
            action = "reused"
        else:
            rca_seed_ds = source_to_ds(self.paths["RCA"][0], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            seed, direction, cand = detect_left_ostium(self.evd, rca_seed_ds)
            self.left_seed_ds, self.left_dir, self.left_candidates = seed, direction, cand
            cand.to_csv(p["left_ostium_candidates"], index=False)
            seed_src = ds_to_source(seed, self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            phys = physical_lps_from_zyx(self.image, [seed_src])[0]
            _json_dump({
                "algorithm": ALGORITHM_VERSION,
                "seed_ds_zyx": seed.tolist(),
                "seed_source_zyx": seed_src.tolist(),
                "seed_lps_mm": phys.tolist(),
                "outward_direction_zyx_mm": direction.tolist(),
            }, p["left_ostium"])
            action = "recomputed_and_cached"
        self._record("left_ostium", action, p["left_ostium"])
        self._done.add("left_ostium")
        return self.left_seed_ds, self.left_dir

    def build_left_branches(self):
        if "left_branches" in self._done:
            return self.paths["LAD"], self.paths["LCX"]
        self.prepare_evidence(); self.detect_left_ostium()
        p = self._paths()
        if self.reuse["left_branches"] and p["lad"].exists() and p["lcx"].exists() and p["left_pair"].exists():
            lad, lcx = _load_path_csv(p["lad"]), _load_path_csv(p["lcx"])
            action = "reused"
        else:
            routes = trace_routes(self.evd, self.left_seed_ds, self.left_dir, min_len=32, max_len=145, max_routes=28)
            if len(routes) < 2:
                raise RuntimeError(f"Only {len(routes)} viable left-coronary routes; cannot establish LAD/LCX pair")
            lad_r, lcx_r, pair_meta = select_left_branch_pair(routes, self.evd, self.image, self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            lad = ds_to_source(lad_r["path"], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            lcx = ds_to_source(lcx_r["path"], self.evd["lo_source_zyx"], self.evd["zoom_zyx"])
            _save_path_csv(lad, self.spacing_zyx, self.image, p["lad"])
            _save_path_csv(lcx, self.spacing_zyx, self.image, p["lcx"])
            route_rows = []
            for i, r in enumerate(routes):
                route_rows.append({"rank": i+1, **{k:v for k,v in r.items() if k != "path"}, "endpoint_z": int(r["path"][-1,0]), "endpoint_y": int(r["path"][-1,1]), "endpoint_x": int(r["path"][-1,2])})
            pd.DataFrame(route_rows).to_csv(self.out / "left_route_candidates.csv", index=False)
            pair_meta.update({"algorithm": ALGORITHM_VERSION, "n_candidate_routes": len(routes)})
            _json_dump(pair_meta, p["left_pair"])
            action = "recomputed_and_cached"
        self.paths["LAD"], self.paths["LCX"] = np.asarray(lad, float), np.asarray(lcx, float)
        self._record("left_branches", action, p["left_pair"])
        self._done.add("left_branches")
        return self.paths["LAD"], self.paths["LCX"]

    def build_all_centerlines(self):
        self.build_rca_centerline(); self.build_left_branches()
        rows = [centerline_qc(self.paths[v], self.ct, self.spacing_zyx, v) for v in VESSELS]
        self.centerline_qc_df = pd.DataFrame(rows)
        self.centerline_qc_df.to_csv(self.out / "centerline_qc.csv", index=False)
        return self.centerline_qc_df

    def build_cpr_stacks(self):
        if "cpr_stacks" in self._done:
            return self.cpr
        self.build_all_centerlines()
        p = self._paths(); p["cpr_dir"].mkdir(parents=True, exist_ok=True)
        action = "reused"
        for v in VESSELS:
            fp = p["cpr_dir"] / f"{v}_rotating_cpr.npz"
            if self.reuse["cpr_stacks"] and fp.exists():
                z = np.load(fp, allow_pickle=False)
                self.cpr[v] = {k: z[k] for k in z.files}
            else:
                d = generate_rotating_cpr(self.ct, self.paths[v], self.spacing_zyx, n_rot=24, half_width_mm=18, cross_step_mm=0.45)
                np.savez_compressed(fp, **d)
                self.cpr[v] = d
                action = "recomputed_and_cached"
        self._record("cpr_stacks", action, p["cpr_dir"])
        self._done.add("cpr_stacks")
        return self.cpr

    def _crop_bounds_for_paths(self, pad_mm=22):
        P = np.vstack([self.paths[v] for v in VESSELS])
        pad = np.ceil(pad_mm / self.spacing_zyx).astype(int)
        lo = np.maximum(0, np.floor(P.min(axis=0)).astype(int) - pad)
        hi = np.minimum(np.asarray(self.ct.shape), np.ceil(P.max(axis=0)).astype(int) + pad + 1)
        return lo, hi

    def _plot_mips(self):
        lo, hi = self._crop_bounds_for_paths(25)
        roi = self.ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        projections = [np.max(roi, axis=0), np.max(roi, axis=1), np.max(roi, axis=2)]
        titles = ["Axial projection (y-x)", "Coronal projection (z-x)", "Sagittal projection (z-y)"]
        for ax, im, title in zip(axs, projections, titles):
            ax.imshow(im, cmap="gray", vmin=-100, vmax=850, origin="lower")
            ax.set_title(title); ax.axis("off")
        for v in VESSELS:
            p = self.paths[v] - lo
            axs[0].plot(p[:,2], p[:,1], linewidth=2, label=v)
            axs[1].plot(p[:,2], p[:,0], linewidth=2, label=v)
            axs[2].plot(p[:,1], p[:,0], linewidth=2, label=v)
        for ax in axs: ax.legend(loc="best")
        fig.suptitle("OpenPlaque source-volume coronary centerlines — source CCTA series 7", fontsize=15)
        out = self.out / "01_source_centerline_mips.png"
        plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); plt.close(fig)
        return out

    def _plane_at(self, path, frac, half=8.0, pix=0.20):
        p, t, n, b = parallel_transport_frame(path, self.spacing_zyx)
        i = int(np.clip(round(frac * (len(p)-1)), 0, len(p)-1))
        c = np.arange(-half, half+1e-6, pix)
        U, V = np.meshgrid(c, c, indexing="xy")
        pm = p[i] * self.spacing_zyx
        pos_mm = pm[None,None,:] + U[...,None]*n[i][None,None,:] + V[...,None]*b[i][None,None,:]
        vox = pos_mm / self.spacing_zyx
        im = map_coordinates(self.ct, [vox[...,0], vox[...,1], vox[...,2]], order=1, mode="nearest")
        return im, [-half, half, -half, half]

    def _plot_cross_sections(self):
        fracs = (0.08, 0.33, 0.58, 0.83)
        fig, axs = plt.subplots(3, 4, figsize=(14, 11))
        for r, v in enumerate(VESSELS):
            for c, frac in enumerate(fracs):
                im, ext = self._plane_at(self.paths[v], frac)
                ax = axs[r,c]; ax.imshow(im, cmap="gray", vmin=-150, vmax=850, extent=ext, origin="lower")
                ax.plot(0,0,"+",markersize=10); ax.set_aspect("equal"); ax.set_title(f"{v} {frac*100:.0f}%")
                if c == 0: ax.set_ylabel("mm")
                ax.set_xlabel("mm")
        fig.suptitle("Orthogonal source-CCTA cross-sections centered on proposed coronary paths", fontsize=15)
        out = self.out / "02_centerline_cross_sections.png"
        plt.tight_layout(rect=[0,0,1,.96]); plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); plt.close(fig)
        return out

    def _best_cpr_rotation(self, d):
        stack = np.asarray(d["stack"], float)
        cross = np.asarray(d["cross_mm"], float)
        center = np.abs(cross) <= 1.0
        side = (np.abs(cross) >= 4.0) & (np.abs(cross) <= 7.0)
        scores = []
        for k in range(stack.shape[0]):
            core = np.nanmean(stack[k][center])
            flank = np.nanmean(stack[k][side])
            plausible = np.mean((stack[k][center] >= 120) & (stack[k][center] <= 1000))
            scores.append(0.65*plausible + 0.35/(1+np.exp(-(core-flank-20)/80)))
        return int(np.argmax(scores)), np.asarray(scores)

    def _plot_cprs(self):
        fig, axs = plt.subplots(3, 1, figsize=(16, 11))
        rows = []
        for ax, v in zip(axs, VESSELS):
            d = self.cpr[v]; k, scores = self._best_cpr_rotation(d)
            stack = d["stack"]; s = d["arc_mm"]; cross = d["cross_mm"]
            ax.imshow(stack[k], cmap="gray", vmin=-150, vmax=850, aspect="auto", origin="lower", extent=[s[0], s[-1], cross[0], cross[-1]])
            ax.axhline(0, linewidth=1); ax.set_ylabel("Transverse mm"); ax.set_xlabel("Distance along source-volume centerline (mm)")
            ax.set_title(f"{v} generated CPR — rotation {k}/{stack.shape[0]-1}")
            rows.append({"vessel":v, "best_rotation":k, "best_rotation_score":float(scores[k]), "cpr_length_mm":float(s[-1]), "n_rotations":int(stack.shape[0])})
        fig.suptitle("OpenPlaque-generated CPRs from source-volume centerlines", fontsize=15)
        out = self.out / "03_generated_cpr_overview.png"
        plt.tight_layout(rect=[0,0,1,.96]); plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); plt.close(fig)
        pd.DataFrame(rows).to_csv(self.out / "generated_cpr_qc.csv", index=False)
        return out

    def plot_qc(self):
        if "qc_figures" in self._done:
            return [self.out / x for x in ("01_source_centerline_mips.png", "02_centerline_cross_sections.png", "03_generated_cpr_overview.png")]
        self.build_cpr_stacks()
        p = self._paths(); outputs = [self.out / x for x in ("01_source_centerline_mips.png", "02_centerline_cross_sections.png", "03_generated_cpr_overview.png")]
        if self.reuse["qc_figures"] and p["qc_meta"].exists() and all(x.exists() for x in outputs):
            self._record("qc_figures", "reused", self.out)
        else:
            self._plot_mips(); self._plot_cross_sections(); self._plot_cprs(); _json_dump({"algorithm":ALGORITHM_VERSION}, p["qc_meta"]); self._record("qc_figures", "recomputed_and_cached", self.out)
        self._done.add("qc_figures")
        return outputs

    def package_report(self):
        if "report_package" in self._done:
            return self.out / "OPENPLAQUE_SOURCE_CENTERLINES_REPORT_BACK.zip"
        self.plot_qc()
        p = self._paths(); zpath = self.out / "OPENPLAQUE_SOURCE_CENTERLINES_REPORT_BACK.zip"; html = self.out / "OPENPLAQUE_SOURCE_CENTERLINES_REPORT.html"
        files = [
            "01_source_centerline_mips.png", "02_centerline_cross_sections.png", "03_generated_cpr_overview.png",
            "centerline_qc.csv", "generated_cpr_qc.csv", "cache_provenance.csv", "left_route_candidates.csv",
        ]
        for name in ("RCA_source_centerline.csv", "LAD_source_centerline.csv", "LCX_source_centerline.csv"):
            src = self.cache / name
            if src.exists(): shutil.copyfile(src, self.out / name)
            files.append(name)
        if (self.cache / "left_ostium_candidates.csv").exists():
            shutil.copyfile(self.cache / "left_ostium_candidates.csv", self.out / "left_ostium_candidates.csv")
            files.append("left_ostium_candidates.csv")
        def img_tag(name):
            fp = self.out / name
            if not fp.exists(): return ""
            b64 = base64.b64encode(fp.read_bytes()).decode()
            return f'<h2>{name}</h2><img style="max-width:100%" src="data:image/png;base64,{b64}">'
        qc = pd.read_csv(self.out / "centerline_qc.csv") if (self.out / "centerline_qc.csv").exists() else pd.DataFrame()
        prov = pd.read_csv(self.out / "cache_provenance.csv") if (self.out / "cache_provenance.csv").exists() else pd.DataFrame()
        html.write_text(
            "<html><head><meta charset='utf-8'><title>OpenPlaque source-volume centerlines</title></head><body>"
            "<h1>OpenPlaque — source-volume coronary centerlines</h1>"
            "<p><b>Research use only.</b> LAD/LCX are automatic hypotheses and must pass visual anatomical QC before downstream use. Canonical TPV and PCAT are unchanged.</p>"
            + "".join(img_tag(x) for x in files if x.endswith(".png"))
            + "<h2>Centerline QC</h2>" + qc.to_html(index=False)
            + "<h2>Cache provenance</h2>" + prov.to_html(index=False)
            + "</body></html>", encoding="utf-8")
        files += [html.name]
        sig = hashlib.sha256("|".join(f"{x}:{_file_sha(self.out/x) if (self.out/x).exists() else 'missing'}" for x in files).encode()).hexdigest()
        if self.reuse["report_package"] and zpath.exists() and p["report_meta"].exists():
            try:
                if _json_load(p["report_meta"])["signature"] == sig:
                    self._record("report_package", "reused", zpath); self._done.add("report_package"); return zpath
            except Exception:
                pass
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                fp = self.out / name
                if fp.exists(): z.write(fp, arcname=name)
            for v in VESSELS:
                fp = self._paths()["cpr_dir"] / f"{v}_rotating_cpr.npz"
                if fp.exists(): z.write(fp, arcname=f"cpr_stacks/{fp.name}")
        _json_dump({"signature":sig, "algorithm":ALGORITHM_VERSION}, p["report_meta"])
        self._record("report_package", "recomputed_and_cached", zpath)
        self._done.add("report_package")
        return zpath
