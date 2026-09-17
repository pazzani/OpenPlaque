from __future__ import annotations

"""Identity audit for the previously dense-QC-positive 20 mm 'proximal-trunk continuation'.

This experiment asks whether that path is actually new proximal LAD/LM anatomy or whether it
re-traces the already established C6/C7 common trunk.  It performs geometry only; it does not
search for a new vessel, assign clinical LM/LCX identity, or modify the frozen master.
"""

import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "left-proximal-trunk-continuation-identity-audit-v1.0"
OUTPUT_DIRNAME = "Left_Proximal_Trunk_Continuation_Identity_Audit_v1"

SOURCE_GEOM = Path("Cache/Secondary_3D_Vesselness_Topology_v1/series7_int16.json")
CONT_PATH = Path("Left_Proximal_Trunk_Continuation_QC_v1/accepted_proximal_trunk_continuation_candidate.csv")
CONT_SUMMARY = Path("Left_Proximal_Trunk_Continuation_QC_v1/summary.json")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
FROZEN_LAD = Path("Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv")
CROSSWALK_SUMMARY = Path("Left_Proximal_Trunk_Topology_Crosswalk_v1/summary.json")
STRUCT_DECISION = Path("LCX_Structural_Identity_Adjudication_v1/structural_identity_decision.json")

SPLIT_ARC_MM = 20.25
STEP_MM = 0.25
BAND_MM = 2.0
MIN_FORWARD_FRACTION = 0.80
MIN_REVERSE_FRACTION = 0.80
MIN_CONTIGUOUS_MM = 15.0
MIN_TANGENT_ALIGNMENT = 0.90
MAX_MIN_DISTANCE_MM = 1.0
MAX_ENDPOINT_ANCHOR_MM = 2.0
MIN_MAPPING_ABS_CORR = 0.90
MAX_LENGTH_DIFFERENCE_MM = 4.0

STATUS_REID = "VALIDATED_20MM_CONTINUATION_REIDENTIFIED_AS_C6_C7_COMMON_TRUNK"
STATUS_PARTIAL = "PROXIMAL_CONTINUATION_PARTIAL_C6_C7_COMMON_TRUNK_ASSOCIATION"
STATUS_INDEPENDENT = "PROXIMAL_CONTINUATION_GEOMETRICALLY_INDEPENDENT_OF_C6_C7_COMMON_TRUNK"


def _req(p: Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p: Path):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p: Path, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _arc(p):
    p = np.asarray(p, float)
    if len(p) <= 1:
        return np.zeros(len(p))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _interp(p, q):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])


def _resample(p, step=STEP_MM):
    p = np.asarray(p, float)
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return p.copy(), a
    q = np.arange(0.0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _geometry(meta):
    spacing_zyx = np.asarray(meta["spacing_zyx"], float)
    origin = np.asarray(meta["positions_lps_mm"][0], float)
    iop = np.asarray(meta["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.column_stack([row, col, slc])
    return origin, spacing_zyx[::-1], D


def _zyx_to_lps(zyx, meta):
    zyx = np.atleast_2d(np.asarray(zyx, float))
    origin, spacing_xyz, D = _geometry(meta)
    xyz_index = zyx[:, ::-1]
    return origin + (xyz_index * spacing_xyz) @ D.T


def _load_path(path: Path, meta):
    d = pd.read_csv(_req(path))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    for cols in (("zyx_z", "zyx_y", "zyx_x"), ("source_z", "source_y", "source_x"), ("z", "y", "x")):
        if all(c in d.columns for c in cols):
            return _zyx_to_lps(d[list(cols)].to_numpy(float), meta)
    raise ValueError(f"No recognized path coordinate columns in {path}: {list(d.columns)}")


def _tangents(p):
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.zeros_like(p)
    g = np.gradient(p, axis=0)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return np.divide(g, n, out=np.zeros_like(g), where=n > 1e-9)


def _max_contiguous_span(mask, arcs):
    mask = np.asarray(mask, bool)
    arcs = np.asarray(arcs, float)
    best = 0.0
    start = None
    for i, good in enumerate(mask):
        if good and start is None:
            start = i
        if start is not None and ((not good) or i == len(mask) - 1):
            end = i if good and i == len(mask) - 1 else i - 1
            best = max(best, float(arcs[end] - arcs[start]))
            start = None
    return best


def association_profile(query, reference, band_mm=BAND_MM, step=STEP_MM):
    q, qa = _resample(query, step)
    r, ra = _resample(reference, step)
    tree = cKDTree(r)
    dist, idx = tree.query(q, k=1)
    qt = _tangents(q)
    rt = _tangents(r)
    align = np.abs(np.sum(qt * rt[idx], axis=1))
    nearest_ra = ra[idx]
    within = dist <= band_mm
    if np.sum(within) >= 3 and np.std(qa[within]) > 1e-9 and np.std(nearest_ra[within]) > 1e-9:
        corr = float(np.corrcoef(qa[within], nearest_ra[within])[0, 1])
        slope = float(np.polyfit(qa[within], nearest_ra[within], 1)[0])
    else:
        corr, slope = float("nan"), float("nan")
    metrics = {
        "query_length_mm": float(qa[-1]),
        "reference_length_mm": float(ra[-1]),
        "min_distance_mm": float(np.min(dist)),
        "median_distance_mm": float(np.median(dist)),
        "p90_distance_mm": float(np.percentile(dist, 90)),
        "fraction_within_1mm": float(np.mean(dist <= 1.0)),
        "fraction_within_2mm": float(np.mean(within)),
        "max_contiguous_span_within_2mm": _max_contiguous_span(within, qa),
        "median_tangent_alignment_within_2mm": float(np.median(align[within])) if np.any(within) else 0.0,
        "arc_mapping_correlation_within_2mm": corr,
        "arc_mapping_slope_within_2mm": slope,
    }
    prof = pd.DataFrame({
        "query_arc_mm": qa,
        "distance_mm": dist,
        "tangent_alignment": align,
        "nearest_reference_arc_mm": nearest_ra,
        "within_2mm": within,
    })
    return metrics, prof


def _endpoint_distance_matrix(a, b):
    ea = np.asarray([a[0], a[-1]], float)
    eb = np.asarray([b[0], b[-1]], float)
    return np.linalg.norm(ea[:, None, :] - eb[None, :, :], axis=2)


def identity_gates(forward, reverse, endpoint, length_difference_mm):
    return {
        "forward_min_distance_le_1mm": bool(forward["min_distance_mm"] <= MAX_MIN_DISTANCE_MM),
        "forward_fraction_within_2mm_ge_0_80": bool(forward["fraction_within_2mm"] >= MIN_FORWARD_FRACTION),
        "forward_contiguous_span_ge_15mm": bool(forward["max_contiguous_span_within_2mm"] >= MIN_CONTIGUOUS_MM),
        "forward_tangent_alignment_ge_0_90": bool(forward["median_tangent_alignment_within_2mm"] >= MIN_TANGENT_ALIGNMENT),
        "reverse_fraction_within_2mm_ge_0_80": bool(reverse["fraction_within_2mm"] >= MIN_REVERSE_FRACTION),
        "arc_mapping_abs_correlation_ge_0_90": bool(np.isfinite(forward["arc_mapping_correlation_within_2mm"]) and abs(forward["arc_mapping_correlation_within_2mm"]) >= MIN_MAPPING_ABS_CORR),
        "one_endpoint_anchors_LAD_endpoint_le_2mm": bool(endpoint["best_continuation_to_LAD_endpoint_mm"] <= MAX_ENDPOINT_ANCHOR_MM),
        "other_endpoint_anchors_common_trunk_split_le_2mm": bool(endpoint["best_other_continuation_to_common_split_mm"] <= MAX_ENDPOINT_ANCHOR_MM),
        "endpoint_anchors_are_distinct": bool(endpoint["anchors_distinct"]),
        "length_difference_le_4mm": bool(length_difference_mm <= MAX_LENGTH_DIFFERENCE_MM),
    }


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status": "STARTED", "algorithm": ALGORITHM, "baseline_commit": BASELINE})

    meta = _read_json(root / SOURCE_GEOM)
    cont_summary = _read_json(root / CONT_SUMMARY)
    crosswalk = _read_json(root / CROSSWALK_SUMMARY)
    structural = _read_json(root / STRUCT_DECISION)

    cont = _load_path(root / CONT_PATH, meta)
    c6 = _load_path(root / C6_PATH, meta)
    lad = _load_path(root / FROZEN_LAD, meta)
    c6a = _arc(c6)
    common = _interp(c6, np.arange(0.0, min(SPLIT_ARC_MM, c6a[-1]) + 1e-9, STEP_MM))
    if _arc(common)[-1] < min(SPLIT_ARC_MM, c6a[-1]) - 1e-6:
        common = np.vstack([common, _interp(c6, [min(SPLIT_ARC_MM, c6a[-1])])[0]])

    forward, prof_common = association_profile(cont, common)
    reverse, _ = association_profile(common, cont)
    to_lad, prof_lad = association_profile(cont, lad)

    d_lad = _endpoint_distance_matrix(cont, lad)
    # Find the continuation endpoint that best anchors any LAD endpoint.
    ci_lad, li = np.unravel_index(np.argmin(d_lad), d_lad.shape)
    common_split = common[-1]
    split_d = np.linalg.norm(np.asarray([cont[0], cont[-1]]) - common_split[None, :], axis=1)
    # The split anchor must be the other continuation endpoint for a true bridge identity.
    ci_other = 1 - int(ci_lad)
    endpoint = {
        "continuation_endpoint_nearest_LAD": int(ci_lad),
        "LAD_endpoint_index": int(li),
        "best_continuation_to_LAD_endpoint_mm": float(d_lad[ci_lad, li]),
        "other_continuation_endpoint_index": int(ci_other),
        "best_other_continuation_to_common_split_mm": float(split_d[ci_other]),
        "nearest_any_continuation_to_common_split_mm": float(np.min(split_d)),
        "anchors_distinct": bool(ci_other != ci_lad),
    }

    length_difference = abs(float(_arc(cont)[-1]) - float(_arc(common)[-1]))
    gates = identity_gates(forward, reverse, endpoint, length_difference)
    n_pass = int(sum(gates.values()))
    if all(gates.values()):
        status = STATUS_REID
    elif n_pass >= max(5, len(gates) // 2):
        status = STATUS_PARTIAL
    else:
        status = STATUS_INDEPENDENT

    prof = prof_common.rename(columns={
        "distance_mm": "distance_to_C6_C7_common_trunk_mm",
        "tangent_alignment": "alignment_to_C6_C7_common_trunk",
        "nearest_reference_arc_mm": "nearest_common_trunk_arc_mm",
        "within_2mm": "within_2mm_common_trunk",
    })
    prof["distance_to_LAD_mm"] = prof_lad["distance_mm"].to_numpy()
    prof["nearest_LAD_arc_mm"] = prof_lad["nearest_reference_arc_mm"].to_numpy()
    prof.to_csv(out / "continuation_reference_profiles.csv", index=False)

    rows = []
    for name, m in (("continuation_to_common_trunk", forward), ("common_trunk_to_continuation", reverse), ("continuation_to_frozen_LAD", to_lad)):
        row = {"comparison": name, **m}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "identity_metrics.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": bool(v)} for k, v in gates.items()]).to_csv(out / "identity_gate_matrix.csv", index=False)
    _write_json(out / "endpoint_topology.json", endpoint)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(prof["query_arc_mm"], prof["distance_to_C6_C7_common_trunk_mm"], label="to C6/C7 common trunk")
    ax.plot(prof["query_arc_mm"], prof["distance_to_LAD_mm"], label="to frozen LAD")
    ax.axhline(BAND_MM, linestyle="--", linewidth=1, label="2 mm identity band")
    ax.set_xlabel("20-mm continuation arc (mm)"); ax.set_ylabel("nearest distance (mm)")
    ax.set_title("Continuation identity distance profile"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "01_continuation_identity_distance_profile.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(14, 4.5))
    pairs = ((0,1,"LPS X","LPS Y"), (0,2,"LPS X","LPS Z"), (1,2,"LPS Y","LPS Z"))
    for i, (a,b,xl,yl) in enumerate(pairs, 1):
        ax = fig.add_subplot(1,3,i)
        ax.plot(lad[:,a], lad[:,b], label="frozen LAD")
        ax.plot(common[:,a], common[:,b], label="C6/C7 common trunk")
        ax.plot(cont[:,a], cont[:,b], linewidth=2, label="validated 20-mm continuation")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.axis("equal")
        if i == 1: ax.legend(fontsize=8)
    fig.suptitle("Identity audit: validated continuation vs established anatomy")
    fig.tight_layout(); fig.savefig(out / "02_identity_geometry_projections.png", dpi=180); plt.close(fig)

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": crosswalk.get("master_status", "CORONARY_ANATOMY_BASELINE_V2_FROZEN"),
        "master_modified": False,
        "prior_continuation_status": cont_summary.get("status"),
        "crosswalk_status": crosswalk.get("status"),
        "structural_C6_label": structural.get("structural_label_C6"),
        "structural_C7_label": structural.get("structural_label_C7"),
        "clinical_identity_established": False,
        "continuation_length_mm": float(_arc(cont)[-1]),
        "common_trunk_length_mm": float(_arc(common)[-1]),
        "length_difference_mm": float(length_difference),
        "continuation_to_common_trunk": forward,
        "common_trunk_to_continuation": reverse,
        "continuation_to_frozen_LAD": to_lad,
        "endpoint_topology": endpoint,
        "gates": gates,
        "n_gates_pass": n_pass,
        "n_gates_total": len(gates),
        "interpretation_boundary": "A positive audit re-identifies the prior 20-mm path as already represented C6/C7 common-trunk anatomy. It does not establish clinical LCX/OM identity, LM, an ostium, or modify the frozen master.",
    }
    _write_json(out / "summary.json", summary)

    report = f"""<html><body><h1>OpenPlaque Proximal-Trunk Continuation Identity Audit v1</h1>
<p><b>Status:</b> {status}</p>
<p>Identity gates passed: {n_pass}/{len(gates)}.</p>
<p>20-mm continuation: {summary['continuation_length_mm']:.2f} mm; C6/C7 common trunk: {summary['common_trunk_length_mm']:.2f} mm.</p>
<p>Forward overlap within 2 mm: {forward['fraction_within_2mm']:.3f}; reverse overlap: {reverse['fraction_within_2mm']:.3f}; median tangent alignment: {forward['median_tangent_alignment_within_2mm']:.3f}.</p>
<p>LAD endpoint anchor: {endpoint['best_continuation_to_LAD_endpoint_mm']:.3f} mm; opposite endpoint to C6/C7 split: {endpoint['best_other_continuation_to_common_split_mm']:.3f} mm.</p>
<p>Frozen master unchanged; clinical LM/LCX/OM identity is not assigned.</p></body></html>"""
    (out / "OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_IDENTITY_AUDIT_REPORT.html").write_text(report, encoding="utf-8")

    _write_json(out / "run_state.json", {"status": "COMPLETE", "algorithm": ALGORITHM, "baseline_commit": BASELINE, "result_status": status})
    zip_path = out / "OPENPLAQUE_PROXIMAL_TRUNK_CONTINUATION_IDENTITY_AUDIT_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zip_path:
                z.write(p, p.name)
    return summary
