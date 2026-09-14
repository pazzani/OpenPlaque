from __future__ import annotations

"""Master coronary anatomy baseline v2.

This is a synthesis workflow, not a vessel search. It freezes only anatomy supported
by prior source-CCTA validation: the canonical RCA and the post-hoc-truncated LAD.
LM and LCX remain unresolved. Historical trunk/secondary candidates are retained
only in a rejection/evidence ledger and never promoted into accepted anatomy.
"""

import json, math, zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ALGORITHM_VERSION = "master-coronary-anatomy-baseline-v2.0"
BASELINE_COMMIT = "0593b453959f5a353d644267fbeef24b514ef4d7"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _load_zyx(path):
    d = pd.read_csv(path)
    if not {"z", "y", "x"}.issubset(d.columns):
        raise ValueError(f"{path} lacks z,y,x columns")
    return d[["z", "y", "x"]].to_numpy(float)


def _arc(path, spacing):
    p = np.asarray(path, float)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.diff(p, axis=0) * np.asarray(spacing, float)[None, :]
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def _find_first(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)


def _zyx_to_lps(path_zyx, meta):
    """Convert fractional Series-7 array z,y,x to DICOM LPS millimetres."""
    p = np.asarray(path_zyx, float)
    ori = np.asarray(meta["image_orientation_patient"], float).reshape(-1)
    if len(ori) < 6:
        raise ValueError("image_orientation_patient must contain 6 values")
    row_dir, col_dir = ori[:3], ori[3:6]
    spacing = np.asarray(meta["spacing_zyx"], float)
    positions = np.asarray(meta["positions_lps_mm"], float)
    z = np.clip(p[:, 0], 0, len(positions) - 1)
    z0 = np.floor(z).astype(int)
    z1 = np.minimum(z0 + 1, len(positions) - 1)
    w = (z - z0)[:, None]
    ipp = (1 - w) * positions[z0] + w * positions[z1]
    return ipp + (p[:, 2] * spacing[2])[:, None] * row_dir[None, :] + (p[:, 1] * spacing[1])[:, None] * col_dir[None, :]


def _curve_min_distance_mm(a_lps, b_lps):
    if len(a_lps) == 0 or len(b_lps) == 0:
        return np.nan
    t = cKDTree(np.asarray(b_lps, float))
    return float(np.min(t.query(np.asarray(a_lps, float))[0]))


def _nearest_pair(a_lps, b_lps):
    t = cKDTree(np.asarray(b_lps, float))
    d, j = t.query(np.asarray(a_lps, float))
    i = int(np.argmin(d))
    return i, int(j[i]), float(d[i])


def _safe_json(path):
    try:
        return _read_json(path) if path and Path(path).exists() else None
    except Exception:
        return None


class MasterCoronaryAnatomyBaselineV2:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque"):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Master_Coronary_Anatomy_Baseline_v2"
        self.out = self.root / "Master_Coronary_Anatomy_Baseline_v2_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.spacing = None
        self.meta = None
        self.rca = self.lad = self.frozen_lad = None
        self.trunk = self.secondary = None
        self.rca_lps = self.lad_lps = None
        self.trunk_lps = self.secondary_lps = None
        self.summary = None

    def load_inputs(self):
        geometry = _find_first([
            self.root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1" / "series7_geometry.json",
            self.root / "Cache" / "Coronary_Anatomy_Reconciliation_v1" / "series7_dicom_geometry.json",
        ])
        if geometry is None:
            raise FileNotFoundError("Verified Series-7 geometry metadata not found")
        self.meta = _read_json(geometry)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)

        rca_fp = self.root / "PCAT_RCA_10_50" / "rca_centerline_smoothed_zyx.csv"
        lad_fp = _find_first([
            self.root / "LAD_Frozen_Proximal_Reacquisition_Report" / "combined_lad_centerline.csv",
            self.root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1" / "combined_lad_centerline.csv",
        ])
        frozen_fp = _find_first([
            self.root / "Cache" / "LAD_Confirmed_Backtrack_v1" / "frozen_lad_centerline.csv",
            self.root / "LAD_Confirmed_Backtrack_Report" / "frozen_lad_centerline.csv",
        ])
        if not rca_fp.exists() or lad_fp is None or frozen_fp is None:
            raise FileNotFoundError("Canonical RCA, validated combined LAD, or frozen LAD input is missing")

        self.rca = _load_zyx(rca_fp)
        self.lad = _load_zyx(lad_fp)
        self.frozen_lad = _load_zyx(frozen_fp)
        self.rca_lps = _zyx_to_lps(self.rca, self.meta)
        self.lad_lps = _zyx_to_lps(self.lad, self.meta)

        trunk_fp = self.root / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv"
        secondary_fp = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv"
        if trunk_fp.exists():
            self.trunk = _load_zyx(trunk_fp)
            self.trunk_lps = _zyx_to_lps(self.trunk, self.meta)
        if secondary_fp.exists():
            self.secondary = _load_zyx(secondary_fp)
            self.secondary_lps = _zyx_to_lps(self.secondary, self.meta)

        prov = {
            "algorithm": ALGORITHM_VERSION,
            "baseline_commit": BASELINE_COMMIT,
            "series7_geometry": str(geometry),
            "rca": str(rca_fp),
            "lad": str(lad_fp),
            "frozen_lad": str(frozen_fp),
            "historical_trunk": str(trunk_fp) if trunk_fp.exists() else None,
            "historical_secondary": str(secondary_fp) if secondary_fp.exists() else None,
        }
        _write_json(prov, self.cache / "input_provenance.json")
        return prov

    def build_evidence(self):
        if self.rca is None:
            self.load_inputs()
        rca_len = float(_arc(self.rca, self.spacing)[-1])
        lad_len = float(_arc(self.lad, self.spacing)[-1])
        frozen_len = float(_arc(self.frozen_lad, self.spacing)[-1])
        proximal_added = max(0.0, lad_len - frozen_len)
        rca_lad_min = _curve_min_distance_mm(self.rca_lps, self.lad_lps)

        global_graph = _safe_json(self.root / "Cache" / "LAD_Global_Left_Coronary_Graph_v1" / "summary.json")
        proximal_summary = _safe_json(self.root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1" / "tracking_summary.json")
        recon = _safe_json(self.root / "Cache" / "Coronary_Anatomy_Reconciliation_v1" / "reconciliation_summary.json")

        accepted = pd.DataFrame([
            {"structure": "RCA", "status": "ACCEPTED", "accepted_length_mm": rca_len, "source": "canonical source-Series-7 RCA centerline / prior PCAT validation", "downstream_policy": "Use only this canonical centerline for RCA plaque/PCAT analyses"},
            {"structure": "LAD", "status": "ACCEPTED", "accepted_length_mm": lad_len, "source": "frozen LAD plus dense final-path-tangent validated proximal extension", "downstream_policy": "Clip LAD downstream analyses to this accepted combined centerline"},
            {"structure": "LM", "status": "UNRESOLVED", "accepted_length_mm": np.nan, "source": "No source-CCTA path established from accepted LAD to aortic root", "downstream_policy": "Do not perform LM-specific plaque/PCAT quantification"},
            {"structure": "LCX", "status": "UNRESOLVED", "accepted_length_mm": np.nan, "source": "No independently validated LCX centerline", "downstream_policy": "Do not perform LCX-specific plaque/PCAT quantification"},
        ])

        ledger = [
            {"evidence": "Canonical RCA", "decision": "ACCEPT", "metric": f"length={rca_len:.3f} mm", "reason": "Established source-Series-7 RCA centerline; retained unchanged."},
            {"evidence": "Frozen LAD core", "decision": "ACCEPT", "metric": f"length={frozen_len:.3f} mm", "reason": "Independently validated compact-lumen LAD segment."},
            {"evidence": "LAD proximal addition", "decision": "ACCEPT", "metric": f"added={proximal_added:.3f} mm; total={lad_len:.3f} mm", "reason": "Dense final-path-tangent QC retained only the defensible proximal extension."},
            {"evidence": "Accepted RCA vs LAD", "decision": "KEEP_SEPARATE", "metric": f"minimum curve distance={rca_lad_min:.3f} mm", "reason": "Accepted segments remain anatomically distinct in common LPS coordinates."},
        ]

        if self.trunk_lps is not None:
            d_lad = _curve_min_distance_mm(self.trunk_lps, self.lad_lps)
            d_rca = _curve_min_distance_mm(self.trunk_lps, self.rca_lps)
            ledger.append({"evidence": "Historical ~8.8 mm trunk candidate", "decision": "RETIRE_AS_LEFT_CORONARY", "metric": f"min to LAD={d_lad:.3f} mm; min to RCA={d_rca:.3f} mm", "reason": "Common-coordinate reconciliation places this candidate with RCA-side anatomy, not accepted LAD."})
        if self.secondary_lps is not None:
            d_lad = _curve_min_distance_mm(self.secondary_lps, self.lad_lps)
            d_rca = _curve_min_distance_mm(self.secondary_lps, self.rca_lps)
            ledger.append({"evidence": "Historical secondary candidate", "decision": "RETIRE_AS_LCX", "metric": f"min to LAD={d_lad:.3f} mm; min to RCA={d_rca:.3f} mm", "reason": "No independent LCX identity; geometry is RCA-associated rather than left-coronary."})

        if proximal_summary:
            raw = proximal_summary.get("raw_proposed_extension_mm", proximal_summary.get("track_length_mm", None))
            val = proximal_summary.get("posthoc_validated_extension_mm", proximal_summary.get("validated_extension_mm", proximal_added))
            if raw is not None:
                ledger.append({"evidence": "Direct proximal LAD reacquisition", "decision": "TRUNCATE", "metric": f"raw proposal={float(raw):.3f} mm; accepted={float(val):.3f} mm", "reason": "Final-path-tangent QC rejected the broad/off-center continuation beyond the accepted proximal addition."})

        if global_graph:
            ledger.append({"evidence": "Global left-coronary graph v1.1", "decision": "NO_FURTHER_EXTENSION", "metric": f"positive_control={global_graph.get('graph_positive_control_passed')}; edge_radius={global_graph.get('adaptive_edge_radius_mm')} mm; candidate_paths={global_graph.get('n_candidate_paths')}; validated_extensions={global_graph.get('n_validated_extensions')}", "reason": "Known-LAD graph positive control passed, yet no novel proximal path left the accepted LAD."})

        if recon:
            ledger.append({"evidence": "Coronary anatomy reconciliation", "decision": "REJECT_OLD_MASTER_LM_LCX_ASSIGNMENT", "metric": str(recon.get("status", "reconciliation completed")), "reason": "Previous master report had selected the wrong LAD/takeoff artifact; common-coordinate reconciliation invalidated that anatomy state."})

        ledger = pd.DataFrame(ledger)
        accepted.to_csv(self.cache / "accepted_anatomy.csv", index=False)
        ledger.to_csv(self.cache / "evidence_ledger.csv", index=False)

        self.summary = {
            "algorithm": ALGORITHM_VERSION,
            "status": "CORONARY_ANATOMY_BASELINE_V2_FROZEN",
            "baseline_commit": BASELINE_COMMIT,
            "accepted": {"RCA_length_mm": rca_len, "LAD_length_mm": lad_len, "LAD_frozen_core_mm": frozen_len, "LAD_validated_proximal_addition_mm": proximal_added},
            "unresolved": ["LM", "LCX"],
            "accepted_RCA_LAD_min_distance_mm": rca_lad_min,
            "downstream_analysis": {"RCA": "allowed only on canonical accepted RCA centerline", "LAD": "allowed only on accepted 24.997-mm-class combined LAD centerline", "LM": "not allowed; unresolved anatomy", "LCX": "not allowed; unresolved anatomy"},
            "historical_trunk_is_LM": False,
            "historical_secondary_is_LCX": False,
            "global_graph_status": global_graph.get("status") if global_graph else None,
            "global_graph_positive_control_passed": global_graph.get("graph_positive_control_passed") if global_graph else None,
        }
        _write_json(self.summary, self.cache / "master_anatomy_summary.json")
        return accepted, ledger, self.summary

    def make_figures(self):
        if self.summary is None:
            self.build_evidence()
        names = []
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.rca_lps[:, 0], self.rca_lps[:, 1], self.rca_lps[:, 2], linewidth=2.4, label="Accepted RCA")
        ax.plot(self.lad_lps[:, 0], self.lad_lps[:, 1], self.lad_lps[:, 2], linewidth=3.0, label="Accepted LAD")
        i, j, d = _nearest_pair(self.rca_lps, self.lad_lps)
        ax.plot([self.rca_lps[i, 0], self.lad_lps[j, 0]], [self.rca_lps[i, 1], self.lad_lps[j, 1]], [self.rca_lps[i, 2], self.lad_lps[j, 2]], linestyle="--", linewidth=1.2)
        ax.set_title(f"Accepted coronary anatomy only — RCA/LAD minimum separation {d:.2f} mm")
        ax.set_xlabel("LPS x (mm)"); ax.set_ylabel("LPS y (mm)"); ax.set_zlabel("LPS z (mm)"); ax.legend()
        fp = self.out / "01_accepted_anatomy_lps.png"
        fig.tight_layout(); fig.savefig(fp, dpi=170); plt.close(fig); names.append(fp.name)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.rca_lps[:, 0], self.rca_lps[:, 1], self.rca_lps[:, 2], linewidth=2.2, label="Accepted RCA")
        ax.plot(self.lad_lps[:, 0], self.lad_lps[:, 1], self.lad_lps[:, 2], linewidth=2.8, label="Accepted LAD")
        if self.trunk_lps is not None:
            ax.plot(self.trunk_lps[:, 0], self.trunk_lps[:, 1], self.trunk_lps[:, 2], linestyle="--", linewidth=1.8, label="Retired trunk candidate")
        if self.secondary_lps is not None:
            ax.plot(self.secondary_lps[:, 0], self.secondary_lps[:, 1], self.secondary_lps[:, 2], linestyle=":", linewidth=1.8, label="Retired secondary candidate")
        ax.set_title("Historical candidates shown for retirement provenance only")
        ax.set_xlabel("LPS x (mm)"); ax.set_ylabel("LPS y (mm)"); ax.set_zlabel("LPS z (mm)"); ax.legend()
        fp = self.out / "02_retired_candidates_lps.png"
        fig.tight_layout(); fig.savefig(fp, dpi=170); plt.close(fig); names.append(fp.name)

        accepted = pd.read_csv(self.cache / "accepted_anatomy.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.axis("off")
        display = accepted[["structure", "status", "accepted_length_mm"]].copy()
        display["accepted_length_mm"] = display["accepted_length_mm"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        tab = ax.table(cellText=display.values, colLabels=["Structure", "Status", "Accepted length (mm)"], loc="center")
        tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 1.6)
        ax.set_title("Frozen coronary anatomy baseline v2", pad=20)
        fp = self.out / "03_anatomy_status.png"
        fig.tight_layout(); fig.savefig(fp, dpi=170); plt.close(fig); names.append(fp.name)

        _write_json(names, self.out / "figure_manifest.json")
        return names

    def build_report(self):
        accepted, ledger, summary = self.build_evidence()
        names = self.make_figures()
        accepted_html = accepted.to_html(index=False, float_format=lambda x: f"{x:.3f}")
        ledger_html = ledger.to_html(index=False)
        imgs = "".join(f'<h3>{n}</h3><img src="{n}" style="max-width:100%">' for n in names)
        html = f"""<html><head><meta charset='utf-8'><title>OpenPlaque Master Coronary Anatomy Baseline v2</title>
<style>body{{font-family:Arial;max-width:1500px;margin:24px auto;padding:0 18px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:6px;border-bottom:1px solid #ddd;text-align:left}} code,pre{{background:#f5f5f5;padding:8px;display:block;white-space:pre-wrap}}</style></head><body>
<h1>OpenPlaque — Master Coronary Anatomy Baseline v2</h1>
<p><b>Status: {summary['status']}</b></p>
<p>This report freezes only independently supported coronary anatomy. RCA and LAD are accepted. LM and LCX remain unresolved. Historical trunk/secondary candidates are retained only as rejected provenance and must not be reintroduced into downstream left-coronary analysis.</p>
<h2>Master summary</h2><pre>{json.dumps(summary, indent=2)}</pre>
<h2>Accepted anatomy</h2>{accepted_html}
<h2>Evidence ledger</h2>{ledger_html}
<h2>Downstream rule</h2><p>Any plaque or PCAT workflow must use the exact accepted centerline artifacts listed here and must not extrapolate beyond their validated endpoints. LM- and LCX-specific analyses remain disabled until independently validated anatomy exists.</p>
<h2>Figures</h2>{imgs}
</body></html>"""
        report = self.out / "OPENPLAQUE_MASTER_CORONARY_ANATOMY_BASELINE_V2_REPORT.html"
        report.write_text(html, encoding="utf-8")
        return report

    def package(self):
        report = self.build_report()
        zip_path = self.out / "OPENPLAQUE_MASTER_CORONARY_ANATOMY_BASELINE_V2_REPORT_BACK.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fp in self.out.iterdir():
                if fp.is_file() and fp != zip_path:
                    z.write(fp, arcname=fp.name)
            for fp in self.cache.iterdir():
                if fp.is_file() and fp.suffix in (".csv", ".json"):
                    z.write(fp, arcname=fp.name)
        return report, zip_path


def synthetic_master_baseline_self_test():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    return {"passed": bool(np.allclose(_arc(p, [1, 1, 1]), [0, 1, 2]) and abs(_curve_min_distance_mm(p, p + [0, 3, 0]) - 3.0) < 1e-8), "algorithm": ALGORITHM_VERSION}
