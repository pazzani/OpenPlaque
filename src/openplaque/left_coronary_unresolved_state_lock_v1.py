from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-coronary-unresolved-state-lock-v1.0"
OUTPUT_DIRNAME = "Left_Coronary_Unresolved_State_Lock_v1"

MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
LAD_PATH = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
CANDIDATE_RANKING = Path("Joint_Three_Vessel_Template_Classifier_v1/LCX_joint_candidate_ranking.csv")

THROUGH = Path("Left_Coronary_Through_Vessel_Continuity_v1/summary.json")
PARENT = Path("Left_Coronary_Bifurcation_Parent_Recovery_v1/summary.json")
OSTIUM_MASK = Path("Left_Coronary_Ostium_Mask_Consensus_v1_1_fixed/summary.json")
OSTIUM_BLIND = Path("Left_Coronary_Source_Ostium_Discovery_v1_2_fast/summary.json")
OSTIUM_MULTI = Path("Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1/summary.json")
GLOBAL_GEO = Path("Left_Coronary_Aorta_Constrained_Geodesic_Bridge_v1/summary.json")
LOCAL_GEO = Path("Left_Coronary_Local_Root_Directed_Bridge_v1/summary.json")
LCX_FREEZE = Path("LCX_Structural_Source_QC_Freeze_v1/summary.json")

EXPECTED_MASTER = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
EXPECTED_THROUGH = "LAD_COMMON_TRUNK_THROUGH_VESSEL_CONTINUITY_CONFIRMED_PARENT_CANDIDATE_REENTERS_LAD"
EXPECTED_PARENT = "BIFURCATION_PARENT_SOURCE_CONTINUATION_QC_POSITIVE_NO_AORTIC_PROGRESS"
EXPECTED_MASK = "SECOND_CORONARY_OSTIAL_EXIT_NOT_ESTABLISHED"
EXPECTED_GLOBAL = "LEFT_CORONARY_AORTA_GEODESIC_NO_STABLE_LAD_ROOT_BRIDGE"
EXPECTED_LOCAL = "LEFT_CORONARY_LOCAL_ROOT_DIRECTED_NO_STABLE_BRIDGE"
EXPECTED_LCX_FREEZE = "LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN"

STATUS = "LEFT_CORONARY_UNRESOLVED_STATE_RESEARCH_LOCKED"

MAX_JUNCTION_GAP_MM = 0.5
MAX_THROUGH_DEFLECTION_DEG = 25.0
MIN_THROUGH_DENSE_QC = 0.90


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _load_lps_csv(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm","lps_y_mm","lps_z_mm"), ("x_mm","y_mm","z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No supported LPS columns in {p}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _angle_deg(a, b):
    a = _unit(a)
    b = _unit(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _orient_at_junction(lad, c6):
    lad = np.asarray(lad, float)
    c6 = np.asarray(c6, float)
    pairs = [
        (0, 0, np.linalg.norm(lad[0]-c6[0])),
        (0, -1, np.linalg.norm(lad[0]-c6[-1])),
        (-1, 0, np.linalg.norm(lad[-1]-c6[0])),
        (-1, -1, np.linalg.norm(lad[-1]-c6[-1])),
    ]
    li, ci, gap = min(pairs, key=lambda x: x[2])
    if li == 0:
        lad = lad[::-1].copy()
    if ci == -1:
        c6 = c6[::-1].copy()
    return lad, c6, float(gap)


def _junction_geometry(lad, c6, window_mm=3.0):
    lad, c6, gap = _orient_at_junction(lad, c6)
    la = _arc(lad)
    ca = _arc(c6)

    lmask = la >= max(0.0, la[-1] - float(window_mm))
    cmask = ca <= min(ca[-1], float(window_mm))

    incoming = lad[-1] - lad[np.flatnonzero(lmask)[0]]
    outgoing = c6[np.flatnonzero(cmask)[-1]] - c6[0]
    deflection = _angle_deg(incoming, outgoing)

    return {
        "junction_gap_mm": gap,
        "through_deflection_deg": deflection,
        "junction_lps_mm": [float(v) for v in 0.5*(lad[-1]+c6[0])],
        "lad_length_mm": float(la[-1]),
        "c6_path_length_mm": float(ca[-1]),
        "lad_oriented": lad,
        "c6_oriented": c6,
    }


def _candidate4_classifier_diagnostic(ranking):
    d = ranking[pd.to_numeric(ranking["candidate_id"], errors="coerce") == 4]
    if d.empty:
        return {"available": False}
    r = d.iloc[0]
    return {
        "available": True,
        "LAD_score": float(r["LAD_score"]),
        "LCX_score": float(r["LCX_score"]),
        "LCX_margin": float(r["LCX_margin"]),
        "RCA_score": float(r["RCA_score"]),
        "length_mm": float(r["length_mm"]),
        "median_hu": float(r["median_hu"]),
        "interpretation": "Classifier separation is descriptive only and is not used as an identity gate.",
    }


def _status_or_none(d):
    return d.get("status") if isinstance(d, dict) else None


def _evidence_table(master, through, parent, ostium_mask, ostium_blind, ostium_multi, global_geo, local_geo, lcx_freeze, geom):
    rows = [
        {
            "evidence": "Frozen master unchanged",
            "observed": master.get("status"),
            "supports_unresolved_lock": master.get("status") == EXPECTED_MASTER,
            "role": "prerequisite",
        },
        {
            "evidence": "Exact LAD-C6 junction",
            "observed": f"gap={geom['junction_gap_mm']:.3f} mm; through deflection={geom['through_deflection_deg']:.1f} deg",
            "supports_unresolved_lock": geom["junction_gap_mm"] <= MAX_JUNCTION_GAP_MM and geom["through_deflection_deg"] <= MAX_THROUGH_DEFLECTION_DEG,
            "role": "topology conflict",
        },
        {
            "evidence": "Dense through-vessel adjudication",
            "observed": through.get("status"),
            "supports_unresolved_lock": (
                through.get("status") == EXPECTED_THROUGH
                and float(through.get("dense_qc",{}).get("pass_fraction",0)) >= MIN_THROUGH_DENSE_QC
            ),
            "role": "topology conflict",
        },
        {
            "evidence": "Parent continuation toward aorta",
            "observed": parent.get("status"),
            "supports_unresolved_lock": parent.get("status") == EXPECTED_PARENT and not bool(parent.get("posthoc",{}).get("meaningful_progress_toward_aorta",True)),
            "role": "negative aortic-origin evidence",
        },
        {
            "evidence": "Mask-consensus second ostial exit",
            "observed": ostium_mask.get("status"),
            "supports_unresolved_lock": ostium_mask.get("status") == EXPECTED_MASK and bool(ostium_mask.get("RCA_interface_control",{}).get("control_pass",False)),
            "role": "interpretable negative",
        },
        {
            "evidence": "Blind source ostium search",
            "observed": ostium_blind.get("status"),
            "supports_unresolved_lock": not bool(ostium_blind.get("second_coronary_candidate",{}).get("accepted",False)),
            "role": "supporting negative; RCA control failed",
        },
        {
            "evidence": "Multiseed source ostium search",
            "observed": ostium_multi.get("status"),
            "supports_unresolved_lock": not bool(ostium_multi.get("second_coronary_candidate",{}).get("accepted",False)),
            "role": "supporting negative; RCA control failed",
        },
        {
            "evidence": "Global aorta-constrained geodesic",
            "observed": global_geo.get("status"),
            "supports_unresolved_lock": global_geo.get("status") == EXPECTED_GLOBAL and bool(global_geo.get("RCA_control",{}).get("pass",False)) and not bool(global_geo.get("LAD_bridge",{}).get("stable_candidate_pass",True)),
            "role": "interpretable negative",
        },
        {
            "evidence": "Local root-directed geodesic",
            "observed": local_geo.get("status"),
            "supports_unresolved_lock": local_geo.get("status") == EXPECTED_LOCAL and bool(local_geo.get("RCA_control",{}).get("pass",False)) and not bool(local_geo.get("LAD_local_bridge",{}).get("stable_candidate_pass",True)),
            "role": "interpretable negative",
        },
        {
            "evidence": "C6/C7 structural source-QC freeze",
            "observed": lcx_freeze.get("status"),
            "supports_unresolved_lock": lcx_freeze.get("status") == EXPECTED_LCX_FREEZE and not bool(lcx_freeze.get("decision",{}).get("clinical_identity_established",False)),
            "role": "research structural labels only",
        },
    ]
    return pd.DataFrame(rows)


def _plot_topology(geom, through, out):
    lad = geom["lad_oriented"]
    c6 = geom["c6_oriented"]
    split = through.get("C67_split_control",{}).get("split_lps_mm")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(lad[:,0], lad[:,1], lad[:,2], linewidth=2.5, label="frozen LAD")
    ax.plot(c6[:,0], c6[:,1], c6[:,2], linewidth=2.5, label="candidate 04 / C6 common path")
    j = np.asarray(geom["junction_lps_mm"], float)
    ax.scatter([j[0]],[j[1]],[j[2]], s=45, label="exact LAD-C6 junction")
    if split is not None:
        s = np.asarray(split,float)
        ax.scatter([s[0]],[s[1]],[s[2]], s=45, label="C6/C7 split")
    ax.set_title("Left-coronary topology conflict audit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_evidence_matrix(evidence, out):
    d = evidence.copy()
    y = np.arange(len(d))
    vals = d["supports_unresolved_lock"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(11,6))
    ax.barh(y, vals)
    ax.set_yticks(y, d["evidence"])
    ax.set_xlim(0,1.15)
    ax.set_xticks([0,1],["no","yes"])
    ax.set_xlabel("Supports unresolved-state lock")
    ax.set_title("Independent evidence audit")
    for i, obs in enumerate(d["observed"]):
        ax.text(1.02, i, str(obs)[:42], va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    lad = np.array([[-2.,0,0],[-1.,0,0],[0.,0,0]])
    c6 = np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
    g = _junction_geometry(lad,c6,window_mm=2.0)
    assert np.isclose(g["junction_gap_mm"],0)
    assert np.isclose(g["through_deflection_deg"],0)
    return {"ok":True}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master = _read_json(root/MASTER)
    through = _read_json(root/THROUGH)
    parent = _read_json(root/PARENT)
    ostium_mask = _read_json(root/OSTIUM_MASK)
    ostium_blind = _read_json(root/OSTIUM_BLIND)
    ostium_multi = _read_json(root/OSTIUM_MULTI)
    global_geo = _read_json(root/GLOBAL_GEO)
    local_geo = _read_json(root/LOCAL_GEO)
    lcx_freeze = _read_json(root/LCX_FREEZE)
    ranking = pd.read_csv(_req(root/CANDIDATE_RANKING))

    lad = _load_lps_csv(root/LAD_PATH)
    c6 = _load_lps_csv(root/C6_PATH)
    geom = _junction_geometry(lad,c6)
    classifier = _candidate4_classifier_diagnostic(ranking)

    evidence = _evidence_table(
        master, through, parent, ostium_mask, ostium_blind, ostium_multi,
        global_geo, local_geo, lcx_freeze, geom
    )
    evidence.to_csv(out/"left_coronary_unresolved_evidence_matrix.csv",index=False)

    required_core = {
        "master_frozen": master.get("status") == EXPECTED_MASTER,
        "exact_junction": geom["junction_gap_mm"] <= MAX_JUNCTION_GAP_MM,
        "through_geometry": geom["through_deflection_deg"] <= MAX_THROUGH_DEFLECTION_DEG,
        "through_dense_qc": (
            through.get("status") == EXPECTED_THROUGH
            and float(through.get("dense_qc",{}).get("pass_fraction",0)) >= MIN_THROUGH_DENSE_QC
        ),
        "parent_no_aortic_progress": (
            parent.get("status") == EXPECTED_PARENT
            and not bool(parent.get("posthoc",{}).get("meaningful_progress_toward_aorta",True))
        ),
        "mask_consensus_no_second_exit_with_RCA_control": (
            ostium_mask.get("status") == EXPECTED_MASK
            and bool(ostium_mask.get("RCA_interface_control",{}).get("control_pass",False))
        ),
        "global_geodesic_negative_with_RCA_control": (
            global_geo.get("status") == EXPECTED_GLOBAL
            and bool(global_geo.get("RCA_control",{}).get("pass",False))
            and not bool(global_geo.get("LAD_bridge",{}).get("stable_candidate_pass",True))
        ),
        "local_geodesic_negative_with_RCA_control": (
            local_geo.get("status") == EXPECTED_LOCAL
            and bool(local_geo.get("RCA_control",{}).get("pass",False))
            and not bool(local_geo.get("LAD_local_bridge",{}).get("stable_candidate_pass",True))
        ),
        "C6_C7_research_only": (
            lcx_freeze.get("status") == EXPECTED_LCX_FREEZE
            and not bool(lcx_freeze.get("decision",{}).get("clinical_identity_established",False))
        ),
    }
    if not all(required_core.values()):
        failed = [k for k,v in required_core.items() if not v]
        raise RuntimeError("Unresolved-state lock prerequisites failed: " + ", ".join(failed))

    decision = {
        "status": STATUS,
        "master_modified": False,
        "clinical_LM_status": "UNRESOLVED",
        "clinical_LCX_OM_identity_established": False,
        "research_structural_labels_retained": {
            "C6":"LCX-like parent continuation",
            "C7":"OM-like daughter",
        },
        "same_scan_heuristic_parameter_tuning": "STOP",
        "reason": (
            "Current source-CCTA evidence contains a topology conflict: the C6/common path joins the accepted LAD with essentially zero gap and through-vessel geometry, "
            "while multiple independent aortic-origin searches with valid RCA controls fail to establish a stable left-coronary connection to the aorta. "
            "The existing C6/C7 labels therefore remain research structural descriptors only and must not be promoted to clinical LCX/OM or LM."
        ),
        "new_evidence_required_to_reopen": [
            "independent expert coronary annotation",
            "validated trained coronary segmentation/centerline model",
            "independent reconstruction or cardiac phase showing proximal left-coronary origin",
            "external clinical centerline or vessel labels",
        ],
        "downstream_policy": {
            "RCA_quantification_allowed": True,
            "frozen_LAD_quantification_allowed": True,
            "C6_C7_research_only_quantification_allowed_on_exact_frozen_paths": True,
            "clinical_LM_quantification_allowed": False,
            "clinical_LCX_OM_quantification_allowed": False,
        },
    }

    _plot_topology(geom, through, out/"01_left_coronary_topology_conflict.png")
    _plot_evidence_matrix(evidence, out/"02_left_coronary_unresolved_evidence_matrix.png")

    summary = {
        "status": STATUS,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "junction_geometry": {
            k:v for k,v in geom.items() if k not in ("lad_oriented","c6_oriented")
        },
        "through_vessel_summary": {
            "status": through.get("status"),
            "dense_qc_pass_fraction": through.get("dense_qc",{}).get("pass_fraction"),
            "through_deflection_deg_prior": through.get("junction",{}).get("through_deflection_deg"),
            "C6_C7_real_split_control_pass": through.get("C67_split_control",{}).get("control_gate_pass"),
            "parent_candidate_reentry_gate_pass": through.get("parent_candidate_reentry",{}).get("reentry_gate_pass"),
        },
        "aortic_origin_evidence": {
            "mask_consensus": _status_or_none(ostium_mask),
            "blind_ostium": _status_or_none(ostium_blind),
            "multiseed_ostium": _status_or_none(ostium_multi),
            "parent_recovery": _status_or_none(parent),
            "global_geodesic": _status_or_none(global_geo),
            "local_geodesic": _status_or_none(local_geo),
        },
        "candidate04_classifier_diagnostic": classifier,
        "core_gates": required_core,
        "decision": decision,
        "scientific_boundary": (
            "This lock is a research workflow decision, not a clinical diagnosis. It preserves the frozen master and prevents repeated same-scan heuristic retuning from converting ambiguous topology into an unsupported clinical vessel label. "
            "RCA and frozen LAD research quantification remain usable under their existing validated/feasibility boundaries; C6/C7 remain exact-path research structures only."
        ),
    }
    _write_json(out/"summary.json",summary)
    _write_json(out/"decision.json",decision)
    _write_json(out/"input_provenance.json",{
        "master":str(root/MASTER),
        "LAD":str(root/LAD_PATH),
        "C6":str(root/C6_PATH),
        "through":str(root/THROUGH),
        "parent":str(root/PARENT),
        "ostium_mask":str(root/OSTIUM_MASK),
        "ostium_blind":str(root/OSTIUM_BLIND),
        "ostium_multiseed":str(root/OSTIUM_MULTI),
        "global_geodesic":str(root/GLOBAL_GEO),
        "local_geodesic":str(root/LOCAL_GEO),
        "LCX_structural_freeze":str(root/LCX_FREEZE),
    })

    report = out/"OPENPLAQUE_LEFT_CORONARY_UNRESOLVED_STATE_LOCK_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Left-Coronary Unresolved-State Lock v1</h1>"
        f"<p><b>Status:</b> {STATUS}</p>"
        f"<p>Exact LAD-C6 junction gap: {geom['junction_gap_mm']:.3f} mm; independently recomputed through deflection: {geom['through_deflection_deg']:.1f}°.</p>"
        f"<p>Prior dense through-vessel QC: {float(through.get('dense_qc',{}).get('pass_fraction',float('nan'))):.3f}. "
        f"Prior parent recovery aortic progress: {float(parent.get('posthoc',{}).get('maximum_aorta_progress_mm',float('nan'))):.2f} mm.</p>"
        "<p>Mask-consensus, global-geodesic, and local-geodesic experiments did not establish a second left-coronary aortic origin; the latter two had valid RCA controls.</p>"
        "<p><b>Decision:</b> LM remains unresolved; C6/C7 remain research structural labels only. Stop same-scan heuristic parameter tuning unless independent new evidence becomes available.</p>"
        "<p><b>Boundary:</b> research workflow lock only; frozen master unchanged.</p>"
        '<img src="01_left_coronary_topology_conflict.png" style="max-width:100%">'
        '<img src="02_left_coronary_unresolved_evidence_matrix.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json",{"status":"COMPLETE","result_status":STATUS,"algorithm":ALGORITHM,"baseline":BASELINE})
    zpath = out/"OPENPLAQUE_LEFT_CORONARY_UNRESOLVED_STATE_LOCK_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p,p.name)

    return {"summary":summary,"report":str(report),"zip":str(zpath)}
