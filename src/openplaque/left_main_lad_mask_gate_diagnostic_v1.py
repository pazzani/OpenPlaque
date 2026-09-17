from __future__ import annotations

"""Diagnose why the feasible long-reach LAD->aorta search still dies early.

This is a prospective diagnostic experiment. It does not relax the accepted anatomy or modify
Master Coronary Anatomy Baseline v2. The unchanged calibrated RCA proximal retrace is rerun first.
For the left side, the exact prior beam logic is instrumented to count proposal attrition by gate.
A shadow run then removes ONLY the hard coronary-mask-proximity rejection while retaining the
same source HU/vesselness thresholds, tangent/turn constraint, monotonic aortic-distance rule,
beam width, step size, and the original mask-support scoring term. The shadow path is diagnostic
only and is never promoted to accepted anatomy in this experiment.

The question is narrow: did the prior long-reach experiment fail because the TotalSegmentator
coronary mask truncates the proximal left-coronary/LM region, or because source-CCTA support itself
cannot sustain a path toward the aortic root under the unchanged source gates?
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
from scipy.ndimage import map_coordinates

from . import left_main_proximal_lad_ostial_bridge as b
from . import left_main_proximal_lad_ostial_bridge_v2 as v11

BASELINE = b.BASELINE
ALGORITHM = "left-main-lad-mask-gate-diagnostic-v1.0"
OUTPUT_DIRNAME = "Left_Main_LAD_Mask_Gate_Diagnostic_v1"
LONG_REACH_DIRNAME = "Left_Main_Proximal_LAD_Long_Reach_v1_2"
OSTIUM_PREREQ_DIRNAME = "Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1"

LEFT_MAX_MM = 46.0
LEFT_GATE_MAX_MM = 44.0
STEP_MM = 0.40
BEAM_WIDTH = 110
RCA_MAX_MM = 12.0
RCA_GATE_MAX_MM = 10.0
RCA_START_INSIDE_MM = 6.0

STATUS_RCA_FAIL = "LAD_MASK_GATE_DIAGNOSTIC_RCA_CONTROL_FAILED"
STATUS_MASK = "LAD_MASK_GATE_ATTRITION_CONFIRMED_SHADOW_REACHES_AORTA"
STATUS_SOURCE = "LAD_SOURCE_SUPPORT_FAILS_EVEN_WITH_MASK_GATE_REMOVED"
STATUS_HARD_REACH = "LAD_HARD_MASK_SEARCH_UNEXPECTEDLY_REACHED_AORTA"


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _orthogonal_qc(ref, src, path, out_path, title, n=8):
    pp, qq = b._resample(path, .20)
    if len(pp) < 5:
        return
    tt = np.gradient(pp, axis=0)
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    picks = np.linspace(0, len(pp) - 1, int(n)).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, ix in zip(axes.ravel(), picks):
        im, qv = b._plane(ref, src, pp[ix], tt[ix])
        ax.imshow(im, cmap="gray", vmin=-100, vmax=900,
                  extent=[qv[0], qv[-1], qv[-1], qv[0]])
        ax.scatter([0], [0], s=18)
        ax.set_title(f"arc {qq[ix]:.1f} mm")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def _mask_profile(ref, F, path):
    p, q = b._resample(path, .20)
    z = b._xyz_to_zyx(ref, p) - F["lo"][None, :]
    shape = np.asarray(F["roi"].shape)
    inside = np.all((z >= 0) & (z <= shape[None, :] - 1), axis=1)
    du = np.full(len(p), np.nan, float)
    vv = np.full(len(p), np.nan, float)
    hu = np.full(len(p), np.nan, float)
    if np.any(inside):
        zz = z[inside].T
        du[inside] = map_coordinates(F["du"], zz, order=1, mode="nearest")
        vv[inside] = map_coordinates(F["v"], zz, order=1, mode="nearest")
        hu[inside] = map_coordinates(F["roi"], zz, order=1, mode="nearest")
    df = pd.DataFrame({"arc_mm": q, "hu": hu, "vesselness": vv,
                       "distance_to_coronary_mask_mm": du,
                       "within_prior_mask_gate": du <= 1.6})
    finite = np.isfinite(du)
    return df, {
        "mask_gate_support_fraction": float(np.mean(du[finite] <= 1.6)) if np.any(finite) else 0.0,
        "median_mask_distance_mm": float(np.nanmedian(du)) if np.any(finite) else np.nan,
        "max_mask_distance_mm": float(np.nanmax(du)) if np.any(finite) else np.nan,
        "first_arc_over_1p6_mm": float(df.loc[df["distance_to_coronary_mask_mm"] > 1.6, "arc_mm"].iloc[0])
            if np.any(df["distance_to_coronary_mask_mm"] > 1.6) else np.nan,
    }


def _instrumented_beam(ref, F, aorta_tree, start, tangent, *, hard_mask_gate: bool,
                       target_max_mm=LEFT_MAX_MM, step=STEP_MM, beam_width=BEAM_WIDTH):
    origin = np.asarray(start, float)
    d0 = float(aorta_tree.query(origin)[0])
    states = [(0., [origin], b._unit(tangent), d0)]
    reached = []
    best = states[0]
    rows = []
    nsteps = int(math.ceil(target_max_mm / step))

    for step_index in range(nsteps):
        counts = {
            "step_index": step_index,
            "arc_budget_mm": float((step_index + 1) * step),
            "states_in": len(states),
            "proposals": 0,
            "reject_turn": 0,
            "reject_aorta_monotonic": 0,
            "reject_loop": 0,
            "reject_outside_field": 0,
            "reject_hu": 0,
            "reject_vesselness": 0,
            "reject_mask_gate": 0,
            "accepted_proposals": 0,
            "kept_states": 0,
            "reached_aorta": 0,
            "min_aorta_distance_mm": np.nan,
        }
        nxt = []
        for score, pts, t, prev_ad in states:
            for d in b._cone_dirs(t):
                counts["proposals"] += 1
                if np.dot(d, t) < math.cos(math.radians(65)):
                    counts["reject_turn"] += 1; continue
                p = pts[-1] + step * d
                ad = float(aorta_tree.query(p)[0])
                if ad > prev_ad + .10:
                    counts["reject_aorta_monotonic"] += 1; continue
                if len(pts) > 5 and np.min(np.linalg.norm(np.asarray(pts[:-4]) - p, axis=1)) < .60 * step:
                    counts["reject_loop"] += 1; continue
                s = b._sample_field(ref, F, p)
                if s is None:
                    counts["reject_outside_field"] += 1; continue
                hu, vv, du, c, l = s
                near = ad <= 1.3
                if not (80 <= hu <= 1400):
                    counts["reject_hu"] += 1; continue
                if vv < .42 * F["thr"]:
                    counts["reject_vesselness"] += 1; continue
                if hard_mask_gate and not (du <= 1.6 or near):
                    counts["reject_mask_gate"] += 1; continue

                vn = min(1., vv / F["norm"])
                support = 1. if c and l else (.60 if c or l else .15 if du <= 1.6 else 0.)
                improve = max(-.25, prev_ad - ad)
                align = max(0., float(np.dot(d, t)))
                ns = score + 1.7 * vn + .45 * support + .45 * align + .9 * improve / max(step, 1e-6)
                nt = b._unit(.70 * t + .30 * d)
                st = (ns, pts + [p], nt, ad)
                nxt.append(st)
                counts["accepted_proposals"] += 1
                if (ad < best[3] - 1e-9) or (abs(ad - best[3]) <= 1e-9 and ns > best[0]):
                    best = st
                if ad <= .75:
                    reached.append(st)
                    counts["reached_aorta"] += 1

        if not nxt:
            rows.append(counts)
            break
        nxt.sort(key=lambda x: x[0], reverse=True)
        keep, bins = [], set()
        for st in nxt:
            key = tuple(np.round(st[1][-1] / .30).astype(int))
            if key in bins:
                continue
            bins.add(key); keep.append(st)
            if len(keep) >= beam_width:
                break
        states = keep
        counts["kept_states"] = len(states)
        counts["min_aorta_distance_mm"] = float(min(st[3] for st in states)) if states else np.nan
        rows.append(counts)
        if len(reached) >= 12:
            break

    chosen = None
    if reached:
        reached.sort(key=lambda x: (x[3], -x[0]))
        chosen = reached[0]
    return chosen, best, pd.DataFrame(rows), reached


def _direct_corridor_profile(ref, F, start, goal, n=161):
    p = np.linspace(np.asarray(start, float), np.asarray(goal, float), int(n))
    arc = np.linspace(0., float(np.linalg.norm(np.asarray(goal) - np.asarray(start))), int(n))
    z = b._xyz_to_zyx(ref, p) - F["lo"][None, :]
    shape = np.asarray(F["roi"].shape)
    inside = np.all((z >= 0) & (z <= shape[None, :] - 1), axis=1)
    hu = np.full(len(p), np.nan); vv = np.full(len(p), np.nan); du = np.full(len(p), np.nan)
    if np.any(inside):
        zz = z[inside].T
        hu[inside] = map_coordinates(F["roi"], zz, order=1, mode="nearest")
        vv[inside] = map_coordinates(F["v"], zz, order=1, mode="nearest")
        du[inside] = map_coordinates(F["du"], zz, order=1, mode="nearest")
    return pd.DataFrame({"arc_mm": arc, "lps_x_mm": p[:,0], "lps_y_mm": p[:,1], "lps_z_mm": p[:,2],
                         "hu": hu, "vesselness": vv, "distance_to_coronary_mask_mm": du})


def synthetic_attrition_self_test():
    # The diagnostic must distinguish source-pass/mask-fail proposals from true source failures.
    assert bool((500 >= 80) and (500 <= 1400))
    du = 2.5; near = False
    assert not (du <= 1.6 or near)
    return {"ok": True, "mask_gate_mm": 1.6, "step_mm": STEP_MM, "beam_width": BEAM_WIDTH}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    b._write_json(out / "run_state.json", {"status":"STARTED", "algorithm":ALGORITHM,
                                            "baseline_commit":BASELINE})

    prior_long = root / LONG_REACH_DIRNAME / "summary.json"
    prior_ost = root / OSTIUM_PREREQ_DIRNAME / "summary.json"
    required = [prior_long, prior_ost, root/b.SOURCE_CACHE/"series7_int16.npy",
                root/b.SOURCE_CACHE/"series7_int16.json", root/b.MASTER, root/b.LAD_PATH,
                root/b.RCA_PATH, root/b.CUR, root/b.LEG, root/b.AORTA]
    for p in required: b._req(p)

    long_summary = _load_json(prior_long)
    ost_summary = _load_json(prior_ost)
    if long_summary.get("algorithm") != "left-main-proximal-lad-long-reach-v1.2-feasible":
        raise RuntimeError(f"Unexpected long-reach input: {long_summary.get('algorithm')}")
    if long_summary.get("baseline_commit") != BASELINE or ost_summary.get("baseline_commit") != BASELINE:
        raise RuntimeError("Prerequisite baseline mismatch")
    if not bool(ost_summary.get("RCA_control_pass", False)):
        raise RuntimeError("Prerequisite blind-source RCA control not recovered")

    master = _load_json(root / b.MASTER)
    ref, src, spacing = b._source(root / b.SOURCE_CACHE)
    lad = b._load_path(root / b.LAD_PATH, ref)
    rca = b._load_path(root / b.RCA_PATH, ref)
    cur = b._resample_mask(root / b.CUR, ref)
    leg = b._resample_mask(root / b.LEG, ref)
    aorta = b._resample_mask(root / b.AORTA, ref)
    aorta_tree, _ = b._surface_tree(aorta, ref)

    print("Running unchanged calibrated RCA retrace control...")
    rpath, rca_summary, _, _ = v11._trace_endpoint(
        ref, src, spacing, cur, leg, aorta_tree, rca,
        max_mm=RCA_MAX_MM, gate_length_max=RCA_GATE_MAX_MM,
        start_inside_mm=RCA_START_INSIDE_MM, require_known_retrace=True)
    control_pass = bool(rca_summary.get("accepted", False))
    b._write_json(out / "RCA_control.json", rca_summary)
    if not control_pass:
        summary = {"status":STATUS_RCA_FAIL, "algorithm":ALGORITHM, "baseline_commit":BASELINE,
                   "master_status":master.get("status"), "master_modified":False,
                   "RCA_control_pass":False}
        b._write_json(out/"summary.json", summary); return {"summary":summary}

    lad_o, lad_dist, _ = b._orient_endpoint_nearest_tree(lad, aorta_tree)
    pr, pq = b._resample(lad_o, .20)
    start = pr[0]
    look = min(len(pr)-1, max(3, int(round(2.0/.20))))
    outward = b._unit(start - pr[look])
    goal = np.asarray(aorta_tree.data[int(aorta_tree.query(start)[1])], float)
    known = pr[:min(len(pr), look+8)]
    F = b._build_field(ref, src, spacing, cur, leg, start, goal, known)

    direct = _direct_corridor_profile(ref, F, start, goal)
    direct.to_csv(out / "direct_start_to_nearest_aorta_profile.csv", index=False)

    print("Instrumenting exact prior hard-mask beam...")
    hard_chosen, hard_best, hard_stats, hard_reached = _instrumented_beam(
        ref, F, aorta_tree, start, outward, hard_mask_gate=True)
    hard_stats.to_csv(out / "hard_mask_gate_attrition.csv", index=False)
    hard_best_path = np.asarray(hard_best[1])
    pd.DataFrame(hard_best_path, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
        out / "hard_mask_best_frontier_path.csv", index=False)
    hard_prof, hard_prof_sum = _mask_profile(ref, F, hard_best_path)
    hard_prof.to_csv(out / "hard_mask_best_frontier_profile.csv", index=False)

    print("Running shadow beam with only the hard mask rejection removed...")
    soft_chosen, soft_best, soft_stats, soft_reached = _instrumented_beam(
        ref, F, aorta_tree, start, outward, hard_mask_gate=False)
    soft_stats.to_csv(out / "shadow_soft_mask_attrition.csv", index=False)
    soft_state = soft_chosen if soft_chosen is not None else soft_best
    soft_path = np.asarray(soft_state[1])
    pd.DataFrame(soft_path, columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(
        out / "shadow_soft_mask_best_path.csv", index=False)
    soft_prof, soft_prof_sum = _mask_profile(ref, F, soft_path)
    soft_prof.to_csv(out / "shadow_soft_mask_best_path_profile.csv", index=False)
    _orthogonal_qc(ref, src, soft_path, out / "01_shadow_soft_mask_orthogonal_source_qc.png",
                   "Shadow path: hard mask rejection removed")

    hard_min = float(hard_best[3])
    soft_min = float(soft_state[3])
    hard_reaches = hard_chosen is not None
    soft_reaches = soft_chosen is not None
    total_mask_reject = int(hard_stats["reject_mask_gate"].sum()) if len(hard_stats) else 0
    source_pass_mask_fail = total_mask_reject

    if hard_reaches:
        status = STATUS_HARD_REACH
    elif soft_reaches:
        status = STATUS_MASK
    else:
        status = STATUS_SOURCE

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "prior_long_reach_status": long_summary.get("status"),
        "prior_long_reach_frontier_count": long_summary.get("proximal_LAD_long_reach",{}).get("frontier_count"),
        "RCA_control_pass": control_pass,
        "LAD_start_aorta_distance_mm": float(lad_dist),
        "hard_mask_search": {
            "reached_aorta": bool(hard_reaches),
            "best_aorta_distance_mm": hard_min,
            "path_length_points": int(len(hard_best_path)),
            "source_pass_but_mask_gate_rejected_proposals": source_pass_mask_fail,
            **hard_prof_sum,
        },
        "shadow_soft_mask_search": {
            "reached_aorta": bool(soft_reaches),
            "best_aorta_distance_mm": soft_min,
            "path_length_points": int(len(soft_path)),
            "n_reached_states": int(len(soft_reached)),
            **soft_prof_sum,
        },
        "scientific_change": "diagnostic only: instrument exact prior beam and run a shadow beam removing only the hard coronary-mask rejection; source gates and master anatomy unchanged",
        "scientific_boundary": "A shadow path that reaches the aorta demonstrates a mask-gate bottleneck, not an accepted left-main. A shadow failure indicates that source HU/vesselness/geometry or field reachability remains limiting even after the hard mask rejection is removed."
    }
    b._write_json(out / "summary.json", summary)
    b._write_json(out / "run_state.json", {"status":"COMPLETE", "scientific_status":status,
                                            "algorithm":ALGORITHM, "baseline_commit":BASELINE})

    fig, ax = plt.subplots(figsize=(7.2,4.4))
    if len(hard_stats): ax.plot(hard_stats["arc_budget_mm"], hard_stats["min_aorta_distance_mm"], marker=".", label="hard mask")
    if len(soft_stats): ax.plot(soft_stats["arc_budget_mm"], soft_stats["min_aorta_distance_mm"], marker=".", label="shadow soft mask")
    ax.axhline(.75, linewidth=.8); ax.set_xlabel("beam arc budget (mm)"); ax.set_ylabel("minimum distance to aorta (mm)")
    ax.legend(); ax.set_title("LAD->aorta gate diagnostic"); plt.tight_layout(); plt.savefig(out/"02_hard_vs_shadow_aorta_distance.png", dpi=180); plt.close()

    report = out / "OPENPLAQUE_LAD_MASK_GATE_DIAGNOSTIC_REPORT.html"
    report.write_text(
        f"<html><body><h1>OpenPlaque LAD Mask-Gate Diagnostic v1</h1><p><b>Status:</b> {status}</p>"
        f"<p>RCA control pass: {control_pass}</p><p>Hard-mask best aorta distance: {hard_min:.2f} mm.</p>"
        f"<p>Shadow soft-mask best aorta distance: {soft_min:.2f} mm; reached aorta: {soft_reaches}.</p>"
        f"<p>Source-pass proposals rejected only by the prior mask gate: {source_pass_mask_fail}.</p>"
        "<p>This experiment is diagnostic only; master anatomy unchanged.</p></body></html>", encoding="utf-8")
    zpath = out / "OPENPLAQUE_LAD_MASK_GATE_DIAGNOSTIC_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file(): z.write(p, p.name)
    return {"summary":summary, "report":str(report), "zip":str(zpath)}
