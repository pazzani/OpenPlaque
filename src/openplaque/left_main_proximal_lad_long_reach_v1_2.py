from __future__ import annotations

"""Feasible long-reach source-CCTA bridge from the accepted proximal LAD to the aortic root.

The previous calibrated proximal-LAD ostial bridge used a 20 mm maximum search length even
though the accepted LAD endpoint was 36.0 mm from the nearest aortic surface.  Reaching the
aorta was therefore geometrically impossible in that experiment, independent of image support.

This follow-up changes only the left-side reachability budget.  It keeps the same source HU,
multiscale vesselness, current/legacy coronary-mask proximity, monotonic aortic-distance beam
search, and RCA calibrated positive-control machinery.  The RCA control is rerun unchanged.
The left search is given a prospective 46 mm budget and 44 mm acceptance-length ceiling.

A source-supported bridge is not called a left-coronary candidate unless its aortic endpoint is
also at least 8 mm from the known RCA proximal endpoint.  That RCA-separation check is post hoc
and does not steer the search.

Research use only.  A positive result nominates a source-supported LAD-to-aorta bridge for
visual adjudication; it does not by itself establish clinical left-main identity or modify the
frozen master anatomy.
"""

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import left_main_proximal_lad_ostial_bridge as b
from . import left_main_proximal_lad_ostial_bridge_v2 as v11

BASELINE = b.BASELINE
ALGORITHM = "left-main-proximal-lad-long-reach-v1.2-feasible"
OUTPUT_DIRNAME = "Left_Main_Proximal_LAD_Long_Reach_v1_2"
PREREQ_DIRNAME = "Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1"

RCA_MAX_MM = 12.0
RCA_GATE_MAX_MM = 10.0
RCA_START_INSIDE_MM = 6.0
LEFT_MAX_MM = 46.0
LEFT_GATE_MAX_MM = 44.0
FEASIBILITY_MARGIN_MM = 6.0
RCA_ENDPOINT_SEPARATION_MM = 8.0

STATUS_RCA_FAIL = "LONG_REACH_RCA_CONTROL_FAILED"
STATUS_NO_BRIDGE = "LONG_REACH_LAD_TO_AORTA_NOT_ESTABLISHED"
STATUS_RCA_ASSOC = "LONG_REACH_PATH_REACHED_AORTA_BUT_RCA_ASSOCIATED"
STATUS_POS = "LONG_REACH_LEFT_CORONARY_BRIDGE_REQUIRES_VISUAL_QC"


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reachability_budget(initial_aorta_distance_mm: float, max_search_mm: float,
                        margin_mm: float = FEASIBILITY_MARGIN_MM):
    initial = float(initial_aorta_distance_mm)
    budget = float(max_search_mm)
    margin = float(margin_mm)
    return {
        "initial_aorta_distance_mm": initial,
        "max_search_mm": budget,
        "required_minimum_mm": initial,
        "prospective_margin_mm": margin,
        "budget_minus_straight_line_mm": budget - initial,
        "geometrically_possible": bool(budget + 1e-9 >= initial),
        "prospectively_adequate": bool(budget + 1e-9 >= initial + margin),
    }


def synthetic_reachability_self_test():
    old = reachability_budget(36.035291878, 20.0)
    new = reachability_budget(36.035291878, LEFT_MAX_MM)
    assert not old["geometrically_possible"]
    assert new["geometrically_possible"]
    assert new["prospectively_adequate"]
    return {"ok": True, "old_budget": old, "new_budget": new}


def _orthogonal_qc(ref, src, path, out_path: Path, title: str, n=8):
    pp, qq = b._resample(path, .20)
    if len(pp) < 5:
        return
    tt = np.gradient(pp, axis=0)
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    picks = np.linspace(0, len(pp) - 1, int(n)).astype(int)
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, ix in zip(axes, picks):
        im, qv = b._plane(ref, src, pp[ix], tt[ix])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=900,
                  extent=[qv[0], qv[-1], qv[-1], qv[0]])
        ax.scatter([0], [0], s=18)
        ax.set_title(f"arc {qq[ix]:.1f} mm")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    b._write_json(out / "run_state.json", {
        "status": "STARTED", "algorithm": ALGORITHM, "baseline_commit": BASELINE,
    })

    prereq = root / PREREQ_DIRNAME / "summary.json"
    required = [
        prereq,
        root / b.SOURCE_CACHE / "series7_int16.npy",
        root / b.SOURCE_CACHE / "series7_int16.json",
        root / b.MASTER,
        root / b.LAD_PATH,
        root / b.RCA_PATH,
        root / b.CUR,
        root / b.LEG,
        root / b.AORTA,
    ]
    for p in required:
        b._req(p)

    prior = _load_json(prereq)
    if prior.get("baseline_commit") != BASELINE:
        raise RuntimeError(f"Unexpected prerequisite baseline: {prior.get('baseline_commit')}")
    if not bool(prior.get("RCA_control_pass", False)):
        raise RuntimeError("Prerequisite v2.1 blind-source RCA control was not recovered")

    master = _load_json(root / b.MASTER)
    ref, src, spacing = b._source(root / b.SOURCE_CACHE)
    lad = b._load_path(root / b.LAD_PATH, ref)
    rca = b._load_path(root / b.RCA_PATH, ref)
    cur = b._resample_mask(root / b.CUR, ref)
    leg = b._resample_mask(root / b.LEG, ref)
    aorta = b._resample_mask(root / b.AORTA, ref)
    aorta_tree, aorta_pts = b._surface_tree(aorta, ref)

    lad_o, lad_prox_dist, lad_flipped = b._orient_endpoint_nearest_tree(lad, aorta_tree)
    budget = reachability_budget(lad_prox_dist, LEFT_MAX_MM)
    b._write_json(out / "left_search_reachability_budget.json", budget)
    print("Accepted LAD endpoint distance to aorta (mm):", round(lad_prox_dist, 3))
    print("Old experiment max search (mm): 20.0")
    print("New max search (mm):", LEFT_MAX_MM)
    print("Budget minus straight-line distance (mm):", round(budget["budget_minus_straight_line_mm"], 3))
    if not budget["prospectively_adequate"]:
        raise RuntimeError(f"Prospective left search budget is inadequate: {budget}")

    print("Running unchanged calibrated RCA proximal retrace control...")
    rpath, rca_summary, rF, rca_o = v11._trace_endpoint(
        ref, src, spacing, cur, leg, aorta_tree, rca,
        max_mm=RCA_MAX_MM, gate_length_max=RCA_GATE_MAX_MM,
        start_inside_mm=RCA_START_INSIDE_MM, require_known_retrace=True,
    )
    b._write_json(out / "RCA_proximal_ostial_control.json", rca_summary)

    control_pass = bool(rca_summary.get("accepted", False))
    print("RCA control accepted:", control_pass)

    lpath = None
    lad_summary = {
        "accepted": False,
        "reason": "RCA_control_failed_before_left_search",
        "proximal_endpoint_aorta_distance_mm": float(lad_prox_dist),
    }
    lF = None
    if control_pass:
        print("Running feasible long-reach accepted-LAD endpoint to aorta search...")
        lpath, lad_summary, lF, _ = v11._trace_endpoint(
            ref, src, spacing, cur, leg, aorta_tree, lad,
            max_mm=LEFT_MAX_MM, gate_length_max=LEFT_GATE_MAX_MM,
            start_inside_mm=0.0, require_known_retrace=False,
        )

    # Post-hoc RCA independence.  This does not steer the left search.
    rca_oriented, _, _ = b._orient_endpoint_nearest_tree(rca, aorta_tree)
    rca_prox = np.asarray(rca_oriented[0], float)
    rca_sep = np.nan
    rca_independent = False
    if lpath is not None:
        rca_sep = float(np.linalg.norm(np.asarray(lpath[-1], float) - rca_prox))
        rca_independent = bool(rca_sep >= RCA_ENDPOINT_SEPARATION_MM)
        lad_summary["posthoc_aortic_endpoint_distance_to_known_RCA_prox_mm"] = rca_sep
        lad_summary["posthoc_RCA_independent"] = rca_independent
        lad_summary["posthoc_RCA_endpoint_separation_gate_mm"] = RCA_ENDPOINT_SEPARATION_MM

    b._write_json(out / "proximal_LAD_long_reach_summary.json", lad_summary)

    left_source_pass = bool(lad_summary.get("accepted", False))
    if not control_pass:
        status = STATUS_RCA_FAIL
    elif not left_source_pass:
        status = STATUS_NO_BRIDGE
    elif not rca_independent:
        status = STATUS_RCA_ASSOC
    else:
        status = STATUS_POS

    if rpath is not None:
        rm = b._path_metrics(ref, src, cur, leg, rF, rpath, rpath[0], aorta_tree)
        pd.DataFrame(rpath, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
            out / "RCA_control_path.csv", index=False
        )
        plt.figure(figsize=(6.5, 4))
        plt.plot(rm["arc_profile_mm"], rm["aorta_distance_profile_mm"], marker=".")
        plt.xlabel("RCA control arc (mm)")
        plt.ylabel("distance to aorta (mm)")
        plt.title("Unchanged RCA calibrated retrace")
        plt.tight_layout(); plt.savefig(out / "01_RCA_control_aorta_distance.png", dpi=180); plt.close()

    if lpath is not None:
        lm = b._path_metrics(ref, src, cur, leg, lF, lpath, lpath[0], aorta_tree)
        pd.DataFrame(lpath, columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
            out / "proximal_LAD_long_reach_candidate.csv", index=False
        )
        b._write_json(out / "proximal_LAD_long_reach_metrics.json",
                      {k: v for k, v in lm.items() if not isinstance(v, np.ndarray)})

        plt.figure(figsize=(7.2, 4.4))
        plt.plot(lm["arc_profile_mm"], lm["aorta_distance_profile_mm"], marker=".", markersize=2)
        plt.axhline(.8, linewidth=.8)
        plt.xlabel("candidate arc (mm)")
        plt.ylabel("distance to aorta (mm)")
        plt.title(f"Long-reach LAD→aorta | RCA endpoint separation={rca_sep:.1f} mm")
        plt.tight_layout(); plt.savefig(out / "02_left_long_reach_aorta_distance.png", dpi=180); plt.close()

        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
        lp, _ = b._resample(lad_o, .3)
        ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], lw=2, label="accepted LAD")
        ax.plot(lpath[:, 0], lpath[:, 1], lpath[:, 2], lw=3, label="long-reach candidate")
        near = aorta_pts[np.linalg.norm(aorta_pts - lpath[-1], axis=1) <= 10]
        if len(near):
            ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=3, alpha=.15, label="local aortic surface")
        ax.scatter([rca_prox[0]], [rca_prox[1]], [rca_prox[2]], s=35, label="known RCA proximal")
        ax.legend(); ax.set_title("Accepted LAD to aortic root: feasible long-reach search")
        plt.tight_layout(); plt.savefig(out / "03_left_long_reach_geometry.png", dpi=180); plt.close()

        _orthogonal_qc(ref, src, lpath, out / "04_left_long_reach_orthogonal_source_qc.png",
                       "Long-reach LAD→aorta source-CCTA orthogonal QC", n=8)

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "prerequisite_source_ostium_status": prior.get("status"),
        "prerequisite_RCA_control_pass": bool(prior.get("RCA_control_pass", False)),
        "left_search_reachability_budget": budget,
        "RCA_calibrated_proximal_retrace": rca_summary,
        "proximal_LAD_long_reach": lad_summary,
        "left_source_gate_pass": left_source_pass,
        "posthoc_RCA_independent": bool(rca_independent),
        "C6_or_LCX_geometry_used_in_search": False,
        "scientific_change": "corrects the previously impossible 20 mm left search budget; source/mask search gates unchanged",
        "scientific_boundary": "A positive result nominates a source-supported bridge from the accepted LAD to an RCA-independent aortic endpoint for visual adjudication. It does not by itself establish clinical LM identity or modify the master baseline.",
    }
    b._write_json(out / "summary.json", summary)
    b._write_json(out / "run_state.json", {
        "status": "COMPLETE", "scientific_status": status,
        "algorithm": ALGORITHM, "baseline_commit": BASELINE,
    })

    report = out / "OPENPLAQUE_LEFT_MAIN_LONG_REACH_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Proximal LAD Long-Reach v1.2</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Prior LAD endpoint-to-aorta distance: {lad_prox_dist:.2f} mm.</p>"
        f"<p>Previous maximum search: 20.0 mm; corrected maximum search: {LEFT_MAX_MM:.1f} mm.</p>"
        f"<p>RCA control accepted: {control_pass}.</p>"
        f"<p>Left source bridge accepted: {left_source_pass}.</p>"
        f"<p>RCA-independent aortic endpoint: {rca_independent}.</p>"
        "<p>Clinical LM identity remains unresolved pending positive-source result and visual adjudication; master unchanged.</p>"
        "</body></html>", encoding="utf-8"
    )
    zpath = out / "OPENPLAQUE_LEFT_MAIN_LONG_REACH_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file():
                z.write(p, p.name)
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
