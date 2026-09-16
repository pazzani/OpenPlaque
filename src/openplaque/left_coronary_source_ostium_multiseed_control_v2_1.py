from __future__ import annotations

"""Post-hoc adjudication of the blind multi-seed ostium experiment.

This module fixes a control-selection bug in v2.0 without changing any blind
candidate-generation or acceptance threshold.  v2.0 selected the geometrically
closest traced path to the known RCA first and only then asked whether that path
had passed the frozen blind acceptance gate.  With multiple seeds this can make
an unaccepted near-RCA path mask a different blind-accepted RCA path.

The v2.1 adjudication therefore:
  * reads the exact set of hypotheses that v2.0 accepted blindly,
  * deterministically regenerates only those accepted hypotheses,
  * applies the known-RCA geometry strictly post hoc,
  * declares the positive control recovered if ANY blind-accepted hypothesis
    satisfies the frozen RCA geometry limits,
  * then applies the same v2.0 second-exit exclusion rules to the remaining
    blind-accepted hypotheses.

No HU, vesselness, beam-search, serial-QC, radius, or blind-acceptance threshold
is changed.  Known RCA/LAD/C6 geometry never affects blind acceptance.
"""

import json
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import left_coronary_source_ostium_discovery as _base
from . import left_coronary_source_ostium_discovery_v1_2 as _v12
from . import left_coronary_source_ostium_multiseed_v2 as _v2

BASELINE = _base.BASELINE
ALGORITHM = "left-coronary-source-ostium-multiseed-v2.1-control-adjudication"
INPUT_DIRNAME = _v2.OUTPUT_DIRNAME
OUTPUT_DIRNAME = "Left_Coronary_Source_Ostium_Multiseed_Control_Adjudication_v2_1"
CACHE_DIRNAME = _v2.CACHE_DIRNAME

CONTROL_MEDIAN_MM = 1.0
CONTROL_P90_MM = 1.8
CONTROL_SEED_MM = 3.0
RCA_ASSOCIATION_MEDIAN_MM = 4.0
SECOND_SEED_SEPARATION_MM = 8.0

STATUS_CONTROL_FAIL = "SOURCE_OSTIUM_MULTI_SEED_RCA_CONTROL_STILL_FAILED"
STATUS_NO_SECOND = "SOURCE_OSTIUM_MULTI_SEED_RCA_CONTROL_RECOVERED_NO_SECOND_EXIT"
STATUS_POS = "SOURCE_OSTIUM_MULTI_SEED_RCA_CONTROL_RECOVERED_SECOND_EXIT_REQUIRES_VISUAL_QC"


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _accepted_keys(df: pd.DataFrame):
    required = {"component_id", "seed_rank", "accepted"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing v2 hypothesis columns: {sorted(missing)}")
    a = df[df["accepted"].astype(str).str.lower().isin({"true", "1"})].copy()
    return [(int(r.component_id), int(r.seed_rank)) for r in a.itertuples(index=False)]


def _rca_geometry(path_global, rca_o, rca_prox, spacing):
    known, _ = _base._resample(rca_o, spacing, .20)
    known = known[_base._arc(known, spacing) <= 10.0 + 1e-9]
    dd = cKDTree(known * np.asarray(spacing)[None, :]).query(
        np.asarray(path_global) * np.asarray(spacing)[None, :]
    )[0]
    return {
        "postsearch_median_distance_to_known_RCA_mm": float(np.median(dd)),
        "postsearch_p90_distance_to_known_RCA_mm": float(np.percentile(dd, 90)),
        "postsearch_seed_distance_to_known_RCA_prox_mm": float(
            np.linalg.norm((np.asarray(path_global)[0] - np.asarray(rca_prox)) * np.asarray(spacing))
        ),
    }


def _control_geometry_pass(g):
    return bool(
        g["postsearch_median_distance_to_known_RCA_mm"] <= CONTROL_MEDIAN_MM
        and g["postsearch_p90_distance_to_known_RCA_mm"] <= CONTROL_P90_MM
        and g["postsearch_seed_distance_to_known_RCA_prox_mm"] <= CONTROL_SEED_MM
    )


def _make_orthogonal_qc(path_local, src_roi, spacing, out_path, title):
    p, s = _base._resample(path_local, spacing, .25)
    if len(p) < 5:
        return
    ss = np.linspace(.7, max(.7, s[-1] - .7), 6)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, x in zip(axes.ravel(), ss):
        i = int(np.argmin(np.abs(s - x)))
        i0, i1 = max(0, i - 3), min(len(p) - 1, i + 3)
        t = (p[i1] - p[i0]) * np.asarray(spacing)
        im, c = _base._orthogonal_plane(src_roi, p[i], t, spacing, half_mm=5., pix_mm=.20)
        ax.imshow(im, cmap="gray", vmin=0, vmax=900, extent=[c[0], c[-1], c[-1], c[0]])
        ax.axhline(0, linewidth=.5)
        ax.axvline(0, linewidth=.5)
        ax.set_title(f"arc {s[i]:.1f} mm")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    t0 = time.time()
    root = Path(drive_root)
    inp = root / INPUT_DIRNAME
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _base._write_json(out / "run_state.json", {
        "status": "STARTED", "algorithm": ALGORITHM, "baseline_commit": BASELINE,
        "input_dir": str(inp),
    })

    for p in [
        inp / "summary.json",
        inp / "blind_multiseed_root_hypotheses.csv",
        root / _base.SOURCE_CACHE / "series7_int16.npy",
        root / _base.SOURCE_CACHE / "series7_int16.json",
        root / _base.MASTER,
        root / _base.RCA_PATH,
        root / _base.LAD_PATH,
        root / _base.C6_PATH,
        root / _base.AORTA,
        root / CACHE_DIRNAME / "root_vesselness.npy",
        root / CACHE_DIRNAME / "root_vesselness_meta.json",
    ]:
        _base._req(p)

    prior = _load_json(inp / "summary.json")
    if prior.get("algorithm") != _v2.ALGORITHM:
        raise RuntimeError(f"Unexpected input algorithm: {prior.get('algorithm')}")
    if prior.get("baseline_commit") != BASELINE:
        raise RuntimeError(f"Unexpected input baseline: {prior.get('baseline_commit')}")

    hdf = pd.read_csv(inp / "blind_multiseed_root_hypotheses.csv")
    accepted_keys = _accepted_keys(hdf)
    prior_n = int(prior.get("n_accepted_seed_hypotheses", -1))
    if prior_n != len(accepted_keys):
        raise RuntimeError(f"Accepted-count mismatch: summary={prior_n}, csv={len(accepted_keys)}")
    if not accepted_keys:
        raise RuntimeError("v2.0 produced no blind-accepted hypotheses; nothing to adjudicate")
    print("Blind-accepted v2 hypotheses:", accepted_keys)

    master = _load_json(root / _base.MASTER)
    ref, src, spacing = _base._source(root / _base.SOURCE_CACHE)
    rca = _base._load_path(root / _base.RCA_PATH, ref)
    lad = _base._load_path(root / _base.LAD_PATH, ref)
    c6 = _base._load_path(root / _base.C6_PATH, ref)
    aorta = _base._resample_mask(root / _base.AORTA, ref)

    surface = aorta & ~ndi.binary_erosion(aorta, iterations=1, border_value=0)
    stree = cKDTree(np.argwhere(surface) * np.asarray(spacing)[None, :])
    d0 = float(stree.query(rca[0] * spacing)[0])
    d1 = float(stree.query(rca[-1] * spacing)[0])
    rca_o = rca if d0 <= d1 else rca[::-1].copy()
    rca_prox = rca_o[0]

    rca_raw, rca_cal = _base._calibrate_rca(rca_o, src, spacing)
    rca_raw.to_csv(out / "RCA_lumen_calibration_sections.csv", index=False)
    _base._write_json(out / "RCA_lumen_calibration.json", rca_cal)

    lo, hi, sl = _base._root_crop(src.shape, rca_prox, spacing)
    src_roi = np.asarray(src[sl])
    aorta_roi = np.asarray(aorta[sl])
    rca_local = rca_o - lo[None, :]
    vessel = _v12._frangi_3d_cached(
        src_roi, spacing, cache_dir=root / CACHE_DIRNAME, reuse_cache=True
    )
    rp, rs = _base._resample(rca_local, spacing, .25)
    rp = rp[rs <= 8.0 + 1e-9]
    rvals = _base._sample_arr(vessel, rp, order=1, cval=0)
    v_thr = max(.003, min(.12, .35 * float(np.percentile(rvals, 25)))) if len(rvals) else .01
    if abs(float(prior.get("source_vesselness_threshold", np.nan)) - float(v_thr)) > 1e-8:
        raise RuntimeError("vesselness threshold does not reproduce v2.0")

    comps, outside = _base._blind_root_components(
        src_roi, aorta_roi, vessel, spacing, v_thr, rca_prox[0] - lo[0]
    )
    comp_map = {int(c["component_id"]): c for c in comps[:40]}

    adjudicated = []
    for component_id, seed_rank in accepted_keys:
        if component_id not in comp_map:
            raise RuntimeError(f"Accepted component {component_id} not regenerated")
        comp = comp_map[component_id]
        seeds = _v2._surface_contact_seeds(comp, outside, spacing)
        seed_map = {int(s["seed_rank"]): s for s in seeds}
        if seed_rank not in seed_map:
            raise RuntimeError(f"Accepted seed {(component_id, seed_rank)} not regenerated")
        seed = seed_map[seed_rank]
        p, sm, qdf = _v2._trace_seed(
            comp, seed, src_roi, vessel, outside, spacing, v_thr, rca_cal,
            f"component_{component_id}_seed_{seed_rank}_adjudication",
        )
        if p is None or not bool(sm.get("accepted", False)):
            raise RuntimeError(f"Blind-accepted hypothesis {(component_id, seed_rank)} did not reproduce")

        pg = p + lo[None, :]
        geom = _rca_geometry(pg, rca_o, rca_prox, spacing)
        control_match = _control_geometry_pass(geom)
        rec = {
            "component_id": component_id,
            "seed_rank": seed_rank,
            "seed_outside_mm": float(seed["seed_outside_mm"]),
            "blind_accepted": True,
            "selection_score": float(sm.get("selection_score", np.nan)),
            "length_mm": float(sm.get("path_metrics", {}).get("length_mm", np.nan)),
            "plane_pass_fraction": float(sm.get("serial_qc", {}).get("plane_pass_fraction", np.nan)),
            "median_plane_score": float(sm.get("serial_qc", {}).get("median_plane_score", np.nan)),
            "median_radius_mm": float(sm.get("serial_qc", {}).get("median_radius_mm", np.nan)),
            **geom,
            "RCA_control_geometry_pass": bool(control_match),
        }
        adjudicated.append({"record": rec, "path_local": p, "path_global": pg, "summary": sm, "qdf": qdf})
        stem = f"accepted_component_{component_id}_seed_{seed_rank}"
        pd.DataFrame(pg, columns=["source_z", "source_y", "source_x"]).to_csv(
            out / f"{stem}_path.csv", index=False
        )
        qdf.to_csv(out / f"{stem}_serial_qc.csv", index=False)
        print(stem, "median/p90/seed mm =",
              round(geom["postsearch_median_distance_to_known_RCA_mm"], 3),
              round(geom["postsearch_p90_distance_to_known_RCA_mm"], 3),
              round(geom["postsearch_seed_distance_to_known_RCA_prox_mm"], 3),
              "control_match=", control_match)

    table = pd.DataFrame([x["record"] for x in adjudicated])
    table.to_csv(out / "accepted_hypothesis_adjudication.csv", index=False)

    control_matches = [x for x in adjudicated if x["record"]["RCA_control_geometry_pass"]]
    control_matches.sort(key=lambda x: (
        x["record"]["postsearch_median_distance_to_known_RCA_mm"],
        x["record"]["postsearch_p90_distance_to_known_RCA_mm"],
        -x["record"]["selection_score"],
    ))
    control = control_matches[0] if control_matches else None
    rca_control_pass = control is not None

    best = None
    if rca_control_pass:
        control_seed_mm = control["path_global"][0] * np.asarray(spacing)
        eligible = []
        for x in adjudicated:
            r = x["record"]
            med = r["postsearch_median_distance_to_known_RCA_mm"]
            seed_sep = float(np.linalg.norm(x["path_global"][0] * spacing - control_seed_mm))
            r["seed_separation_from_RCA_control_mm"] = seed_sep
            r["RCA_associated_by_v2_exclusion"] = bool(med <= RCA_ASSOCIATION_MEDIAN_MM)
            second_ok = bool(
                med > RCA_ASSOCIATION_MEDIAN_MM
                and seed_sep >= SECOND_SEED_SEPARATION_MM
            )
            r["second_exit_eligible"] = second_ok
            if second_ok:
                eligible.append(x)
        if eligible:
            eligible.sort(key=lambda x: x["record"]["selection_score"], reverse=True)
            best = eligible[0]
            pm = best["path_global"] * np.asarray(spacing)[None, :]
            best["record"]["posthoc_min_distance_to_LAD_mm"] = float(
                np.min(cKDTree(lad * spacing[None, :]).query(pm)[0])
            )
            best["record"]["posthoc_min_distance_to_C6_mm"] = float(
                np.min(cKDTree(c6 * spacing[None, :]).query(pm)[0])
            )
            pd.DataFrame(best["path_global"], columns=["source_z", "source_y", "source_x"]).to_csv(
                out / "second_coronary_ostial_exit_path.csv", index=False
            )
            pd.DataFrame(_base._zyx_to_xyz(ref, best["path_global"]),
                         columns=["lps_x_mm", "lps_y_mm", "lps_z_mm"]).to_csv(
                out / "second_coronary_ostial_exit_lps.csv", index=False
            )
            best["qdf"].to_csv(out / "second_coronary_ostial_exit_serial_qc.csv", index=False)

    # Rewrite table after adding RCA-association / second-exit fields.
    pd.DataFrame([x["record"] for x in adjudicated]).to_csv(
        out / "accepted_hypothesis_adjudication.csv", index=False
    )

    if not rca_control_pass:
        status = STATUS_CONTROL_FAIL
    elif best is None:
        status = STATUS_NO_SECOND
    else:
        status = STATUS_POS

    if control is not None:
        _base._write_json(out / "RCA_control_recovered.json", control["record"])
        pd.DataFrame(control["path_global"], columns=["source_z", "source_y", "source_x"]).to_csv(
            out / "RCA_control_recovered_path.csv", index=False
        )
        control["qdf"].to_csv(out / "RCA_control_recovered_serial_qc.csv", index=False)
        _make_orthogonal_qc(
            control["path_local"], src_roi, spacing,
            out / "02_RCA_control_recovered_orthogonal_qc.png",
            "Recovered blind-accepted RCA control",
        )

    if best is not None:
        _base._write_json(out / "second_coronary_ostial_exit_summary.json", best["record"])
        _make_orthogonal_qc(
            best["path_local"], src_roi, spacing,
            out / "03_second_exit_orthogonal_qc.png",
            "Blind-accepted second coronary ostial exit",
        )

    # Post-hoc visual overlay of the accepted blind hypotheses and known RCA.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    kr, _ = _base._resample(rca_o, spacing, .25)
    krmm = kr * spacing[None, :]
    ax.plot(krmm[:, 2], krmm[:, 1], krmm[:, 0], linewidth=2, label="known RCA (post-hoc)")
    for x in adjudicated:
        q = x["path_global"] * spacing[None, :]
        r = x["record"]
        ax.plot(q[:, 2], q[:, 1], q[:, 0], linewidth=2,
                label=f"accepted c{r['component_id']} s{r['seed_rank']}")
    ax.set_title("Blind-accepted multi-seed hypotheses — post-hoc RCA adjudication")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "01_accepted_hypotheses_RCA_overlay.png", dpi=180)
    plt.close()

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status", "CORONARY_ANATOMY_BASELINE_V2_FROZEN"),
        "master_modified": False,
        "input_algorithm": prior.get("algorithm"),
        "input_status": prior.get("status"),
        "input_blind_accepted_hypotheses": len(accepted_keys),
        "RCA_control_pass": bool(rca_control_pass),
        "RCA_control": control["record"] if control is not None else {"control_pass": False},
        "second_coronary_candidate": best["record"] if best is not None else {"accepted": False},
        "scientific_change": "post-hoc control selection only; blind acceptance unchanged",
        "elapsed_seconds": float(time.time() - t0),
    }
    _base._write_json(out / "summary.json", summary)

    report = out / "OPENPLAQUE_MULTI_SEED_CONTROL_ADJUDICATION_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque Multi-Seed RCA Control Adjudication v2.1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>Blind-accepted v2 hypotheses: {len(accepted_keys)}</p>"
        f"<p>RCA positive control recovered: {rca_control_pass}</p>"
        f"<p>Independent second accepted exit: {best is not None}</p>"
        "<p>No blind search or acceptance threshold changed; the fix is post-hoc control adjudication only.</p>"
        "<p>Master baseline unchanged.</p></body></html>",
        encoding="utf-8",
    )
    zpath = out / "OPENPLAQUE_MULTI_SEED_CONTROL_ADJUDICATION_RESULTS.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p != zpath and p.is_file():
                z.write(p, p.name)

    _base._write_json(out / "run_state.json", {
        "status": "COMPLETE", "scientific_status": status,
        "algorithm": ALGORITHM, "baseline_commit": BASELINE,
    })
    return {"summary": summary, "report": str(report), "zip": str(zpath)}
