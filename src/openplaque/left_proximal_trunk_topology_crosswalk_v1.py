from __future__ import annotations

"""Geometric topology crosswalk for the source-led proximal-trunk continuation.

This experiment does not search for a new vessel. It compares the already source-QC-positive
long continuation against the frozen LAD, the established C6/C7 structural branches, and the
RCA. The purpose is to determine whether the long path is geometrically rejoining/overlapping a
known left-coronary branch or remains independent. Clinical labels are not assigned and the
frozen master is never modified.
"""

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-proximal-trunk-topology-crosswalk-v1.0"
OUTPUT_DIRNAME = "Left_Proximal_Trunk_Topology_Crosswalk_v1"

MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
SOURCE_META = Path("Cache/Secondary_3D_Vesselness_Topology_v1/series7_int16.json")
LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
RCA = Path("PCAT_RCA_10_50/rca_centerline_smoothed_zyx.csv")
C6 = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7 = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
STRUCTURAL = Path("LCX_Structural_Identity_Adjudication_v1/structural_identity_decision.json")
LOCAL_SUMMARY = Path("Left_Proximal_Trunk_Local_Multidirection_v1/summary.json")
LONG_SUMMARY = Path("Left_Proximal_Trunk_Source_Led_Long_Extension_v1/summary.json")
LONG_PATH = Path("Left_Proximal_Trunk_Source_Led_Long_Extension_v1/best_long_extension_path.csv")
LOCAL_PATH = Path("Left_Proximal_Trunk_Local_Multidirection_v1/best_local_multidirection_path.csv")
PROX_PATH = Path("Left_Proximal_Trunk_Continuation_QC_v1/accepted_proximal_trunk_continuation_candidate.csv")

C67_SPLIT_ARC_MM = 20.25
SAMPLE_STEP_MM = 0.25
ASSOC_NEAR_MM = 1.5
ASSOC_BAND_MM = 2.0
ASSOC_MIN_SPAN_MM = 5.0
ASSOC_MIN_FRACTION = 0.20
ASSOC_MIN_TANGENT_ALIGNMENT = 0.75

STATUS_C6 = "SOURCE_LED_LONG_PATH_GEOMETRICALLY_ASSOCIATED_WITH_C6_LCX_LIKE_PARENT"
STATUS_C7 = "SOURCE_LED_LONG_PATH_GEOMETRICALLY_ASSOCIATED_WITH_C7_OM_LIKE_DAUGHTER"
STATUS_COMMON = "SOURCE_LED_LONG_PATH_GEOMETRICALLY_ASSOCIATED_WITH_C6_C7_COMMON_TRUNK"
STATUS_LAD = "SOURCE_LED_LONG_PATH_GEOMETRICALLY_REJOINS_FROZEN_LAD"
STATUS_INDEPENDENT = "SOURCE_LED_LONG_PATH_GEOMETRICALLY_INDEPENDENT_OF_ESTABLISHED_BRANCHES"
STATUS_AMBIG = "SOURCE_LED_LONG_PATH_TOPOLOGY_AMBIGUOUS"
STATUS_PREREQ = "SOURCE_LED_LONG_PATH_TOPOLOGY_PREREQUISITE_FAILED"


def _req(p: Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p: Path):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p: Path, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _source_ref(meta_path: Path):
    m = _read_json(meta_path)
    shape = tuple(int(x) for x in m.get("shape", m.get("shape_zyx", [1, 1, 1])))
    ref = sitk.Image(int(shape[2]), int(shape[1]), int(shape[0]), sitk.sitkInt16)
    sp = np.asarray(m["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp[::-1]))
    ref.SetOrigin(tuple(np.asarray(m["positions_lps_mm"][0], float)))
    iop = np.asarray(m["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref


def _zyx_to_xyz(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    idx_xyz = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ D.T


def _load_path(path: Path, ref):
    d = pd.read_csv(_req(path))
    for c in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(x in d.columns for x in c):
            return d[list(c)].to_numpy(float)
    for c in (("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")):
        if all(x in d.columns for x in c):
            return _zyx_to_xyz(ref, d[list(c)].to_numpy(float))
    raise ValueError(f"No recognized coordinate columns in {path}: {list(d.columns)}")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _resample(p, step=SAMPLE_STEP_MM):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)]), q


def _tangents(p):
    p = np.asarray(p, float)
    if len(p) == 1:
        return np.zeros_like(p)
    g = np.gradient(p, axis=0)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-9)


def _max_contiguous_span(mask, arc):
    mask = np.asarray(mask, bool)
    arc = np.asarray(arc, float)
    best = 0.0
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask)-1) and start is not None:
            end = i if v and i == len(mask)-1 else i-1
            best = max(best, float(arc[end] - arc[start]))
            start = None
    return best


def _distance_profile(query, reference, label):
    q, qa = _resample(query)
    r, ra = _resample(reference)
    tree = cKDTree(r)
    dist, idx = tree.query(q)
    qt = _tangents(q)
    rt = _tangents(r)
    align = np.abs(np.sum(qt * rt[idx], axis=1))
    near = dist <= ASSOC_BAND_MM
    df = pd.DataFrame({
        "arc_mm": qa,
        f"distance_to_{label}_mm": dist,
        f"tangent_alignment_to_{label}": align,
        f"nearest_{label}_arc_mm": ra[idx],
    })
    j = int(np.argmin(dist))
    metrics = {
        "label": label,
        "min_distance_mm": float(np.min(dist)),
        "p10_distance_mm": float(np.percentile(dist, 10)),
        "median_distance_mm": float(np.median(dist)),
        "p90_distance_mm": float(np.percentile(dist, 90)),
        "endpoint_distance_mm": float(dist[-1]),
        "fraction_within_1mm": float(np.mean(dist <= 1.0)),
        "fraction_within_1_5mm": float(np.mean(dist <= 1.5)),
        "fraction_within_2mm": float(np.mean(near)),
        "max_contiguous_span_within_2mm": float(_max_contiguous_span(near, qa)),
        "median_tangent_alignment_within_2mm": float(np.median(align[near])) if np.any(near) else 0.0,
        "nearest_query_arc_mm": float(qa[j]),
        "nearest_reference_arc_mm": float(ra[idx[j]]),
        "alignment_at_nearest": float(align[j]),
    }
    metrics["association_gate_pass"] = bool(
        metrics["min_distance_mm"] <= ASSOC_NEAR_MM
        and metrics["fraction_within_2mm"] >= ASSOC_MIN_FRACTION
        and metrics["max_contiguous_span_within_2mm"] >= ASSOC_MIN_SPAN_MM
        and metrics["median_tangent_alignment_within_2mm"] >= ASSOC_MIN_TANGENT_ALIGNMENT
    )
    return df, metrics


def _slice_by_arc(p, lo=None, hi=None):
    q, a = _resample(p)
    keep = np.ones(len(q), dtype=bool)
    if lo is not None:
        keep &= a >= float(lo)
    if hi is not None:
        keep &= a <= float(hi)
    return q[keep]


def _join_paths(paths, tol=0.8):
    out = np.asarray(paths[0], float).copy()
    for p in paths[1:]:
        p = np.asarray(p, float)
        if len(p) == 0:
            continue
        d0 = np.linalg.norm(out[-1] - p[0])
        d1 = np.linalg.norm(out[-1] - p[-1])
        if d1 < d0:
            p = p[::-1]
        start = 1 if np.linalg.norm(out[-1]-p[0]) <= tol else 0
        out = np.vstack([out, p[start:]])
    return out


def _classify(m):
    passed = [k for k, v in m.items() if v.get("association_gate_pass")]
    post = [x for x in passed if x in ("C6_post_split", "C7_post_split")]
    if len(post) == 1:
        return STATUS_C6 if post[0] == "C6_post_split" else STATUS_C7
    if len(post) > 1:
        return STATUS_AMBIG
    if "C6_C7_common_trunk" in passed:
        return STATUS_COMMON
    if "LAD" in passed:
        return STATUS_LAD
    if not passed:
        return STATUS_INDEPENDENT
    return STATUS_AMBIG


def synthetic_topology_crosswalk_self_test():
    x = np.column_stack([np.arange(0, 10.25, .25), np.zeros(41), np.zeros(41)])
    y = x + np.array([0, .5, 0])
    _, m = _distance_profile(x, y, "synthetic")
    assert m["association_gate_pass"]
    z = y + np.array([0, 10, 0])
    _, m2 = _distance_profile(x, z, "far")
    assert not m2["association_gate_pass"]
    return {"ok": True, "near_gate": m["association_gate_pass"], "far_gate": m2["association_gate_pass"]}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED", "algorithm":ALGORITHM, "baseline_commit":BASELINE})

    required = [root/MASTER, root/SOURCE_META, root/LAD, root/RCA, root/C6, root/C7,
                root/STRUCTURAL, root/LOCAL_SUMMARY, root/LONG_SUMMARY,
                root/LONG_PATH, root/LOCAL_PATH, root/PROX_PATH]
    for p in required:
        _req(p)

    master = _read_json(root/MASTER)
    structural = _read_json(root/STRUCTURAL)
    local_summary = _read_json(root/LOCAL_SUMMARY)
    long_summary = _read_json(root/LONG_SUMMARY)

    prereq = bool(
        master.get("status") == "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
        and structural.get("all_predeclared_structural_gates_pass") is True
        and str(local_summary.get("status", "")).startswith("PROXIMAL_TRUNK_LOCAL_MULTIDIRECTION_CONTINUATION_QC_POSITIVE")
        and str(long_summary.get("status", "")).startswith("SOURCE_LED_LONG_EXTENSION_QC_POSITIVE")
    )
    if not prereq:
        summary = {"status":STATUS_PREREQ, "algorithm":ALGORITHM, "baseline_commit":BASELINE,
                   "master_status":master.get("status"), "master_modified":False}
        _write_json(out/"summary.json", summary)
        _write_json(out/"run_state.json", {"status":"COMPLETE", "scientific_status":STATUS_PREREQ,
                                            "algorithm":ALGORITHM, "baseline_commit":BASELINE})
        return _finalize(out, summary)

    ref = _source_ref(root/SOURCE_META)
    lad = _load_path(root/LAD, ref)
    rca = _load_path(root/RCA, ref)
    c6 = _load_path(root/C6, ref)
    c7 = _load_path(root/C7, ref)
    longp = _load_path(root/LONG_PATH, ref)
    localp = _load_path(root/LOCAL_PATH, ref)
    proxp = _load_path(root/PROX_PATH, ref)

    refs = {
        "LAD": lad,
        "C6_C7_common_trunk": _slice_by_arc(c6, hi=C67_SPLIT_ARC_MM),
        "C6_post_split": _slice_by_arc(c6, lo=C67_SPLIT_ARC_MM),
        "C7_post_split": _slice_by_arc(c7, lo=C67_SPLIT_ARC_MM),
        "RCA": rca,
    }

    metric_rows = []
    profile = None
    metrics = {}
    for label, rp in refs.items():
        df, m = _distance_profile(longp, rp, label)
        metrics[label] = m
        metric_rows.append(m)
        profile = df if profile is None else profile.merge(df, on="arc_mm", how="outer")
    profile.sort_values("arc_mm").to_csv(out/"long_path_reference_distance_profiles.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out/"topology_crosswalk_metrics.csv", index=False)

    # A secondary local-path crosswalk is retained only as context around the junction.
    local_rows = []
    for label, rp in refs.items():
        _, m = _distance_profile(localp, rp, label)
        local_rows.append(m)
    pd.DataFrame(local_rows).to_csv(out/"local_path_reference_metrics.csv", index=False)

    status = _classify(metrics)
    decision = {
        "status": status,
        "association_thresholds": {
            "nearest_distance_mm": ASSOC_NEAR_MM,
            "band_distance_mm": ASSOC_BAND_MM,
            "minimum_contiguous_span_mm": ASSOC_MIN_SPAN_MM,
            "minimum_fraction_within_band": ASSOC_MIN_FRACTION,
            "minimum_median_tangent_alignment": ASSOC_MIN_TANGENT_ALIGNMENT,
        },
        "long_path_associations": metrics,
        "structural_reference_labels": {
            "C6": structural.get("structural_label_C6"),
            "C7": structural.get("structural_label_C7"),
        },
        "clinical_identity_established": False,
        "master_modified": False,
    }
    _write_json(out/"topology_crosswalk_decision.json", decision)

    composite = _join_paths([proxp, localp, longp])
    pd.DataFrame(composite, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
        out/"investigational_composite_path.csv", index=False)

    # Distance profile plot.
    plt.figure(figsize=(9, 5))
    for label in ("LAD", "C6_C7_common_trunk", "C6_post_split", "C7_post_split", "RCA"):
        col = f"distance_to_{label}_mm"
        if col in profile:
            plt.plot(profile["arc_mm"], profile[col], label=label)
    plt.axhline(ASSOC_BAND_MM, ls="--", lw=.8)
    plt.xlabel("long source-led continuation arc (mm)")
    plt.ylabel("nearest reference-path distance (mm)")
    plt.title("Topology crosswalk: source-led long continuation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out/"01_long_path_reference_distance_profiles.png", dpi=180)
    plt.close()

    # Orthographic geometry projections.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    pairs = [(0,1,"LPS X","LPS Y"),(0,2,"LPS X","LPS Z"),(1,2,"LPS Y","LPS Z")]
    paths = [(longp,"long source-led"),(lad,"frozen LAD"),(refs["C6_C7_common_trunk"],"C6/C7 common"),
             (refs["C6_post_split"],"C6 post-split"),(refs["C7_post_split"],"C7 post-split")]
    for ax, (i,j,xl,yl) in zip(axes, pairs):
        for p, lab in paths:
            ax.plot(p[:,i], p[:,j], lw=1.5, label=lab)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=7)
    fig.suptitle("Geometric topology crosswalk")
    plt.tight_layout()
    plt.savefig(out/"02_topology_geometry_projections.png", dpi=180)
    plt.close()

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "long_source_status": long_summary.get("status"),
        "long_source_best_accepted_arc_mm": long_summary.get("best_hypothesis", {}).get("accepted_arc_mm"),
        "long_source_start_aorta_distance_mm": long_summary.get("posthoc", {}).get("start_aorta_distance_mm"),
        "long_source_end_aorta_distance_mm": long_summary.get("posthoc", {}).get("end_aorta_distance_mm"),
        "association_thresholds": decision["association_thresholds"],
        "associations": metrics,
        "candidate_role": "geometric crosswalk of the source-QC-positive investigational continuation against frozen LAD and established C6/C7 structural paths",
        "scientific_boundary": "This experiment can establish geometric association or independence only. It does not assign clinical LM/LCX/OM identity and does not modify frozen anatomy.",
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"run_state.json", {"status":"COMPLETE", "scientific_status":status,
                                        "algorithm":ALGORITHM, "baseline_commit":BASELINE})
    return _finalize(out, summary)


def _finalize(out, summary):
    report = out/"OPENPLAQUE_PROXIMAL_TRUNK_TOPOLOGY_CROSSWALK_REPORT.html"
    assoc = summary.get("associations", {})
    rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('min_distance_mm')}</td><td>{v.get('max_contiguous_span_within_2mm')}</td><td>{v.get('median_tangent_alignment_within_2mm')}</td><td>{v.get('association_gate_pass')}</td></tr>"
        for k,v in assoc.items())
    report.write_text(
        "<html><body><h1>OpenPlaque Proximal-Trunk Topology Crosswalk v1</h1>"
        f"<p><b>Status:</b> {summary.get('status')}</p>"
        f"<p>Long source-led continuation: {summary.get('long_source_best_accepted_arc_mm')} mm.</p>"
        f"<p>Aortic distance: {summary.get('long_source_start_aorta_distance_mm')} → {summary.get('long_source_end_aorta_distance_mm')} mm.</p>"
        "<table border='1'><tr><th>Reference</th><th>Min distance mm</th><th>Contiguous span ≤2 mm</th><th>Median tangent alignment</th><th>Gate</th></tr>"
        + rows + "</table><p>Frozen master unchanged; clinical identity remains unresolved unless separately adjudicated.</p></body></html>",
        encoding="utf-8")
    zpath = out/"OPENPLAQUE_PROXIMAL_TRUNK_TOPOLOGY_CROSSWALK_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file():
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
