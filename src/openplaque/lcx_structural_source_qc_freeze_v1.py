from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-structural-source-qc-freeze-v1.0"
OUTPUT_DIRNAME = "LCX_Structural_Source_QC_Freeze_v1"

SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
STRUCTURAL_SUMMARY = Path("LCX_Structural_Identity_Adjudication_v1/summary.json")
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")

EXPECTED_MASTER_STATUS = "CORONARY_ANATOMY_BASELINE_V2_FROZEN"
EXPECTED_STRUCTURAL_STATUS = "C6_LCX_LIKE_PARENT_C7_OM_LIKE_DAUGHTER_STRUCTURAL_ADJUDICATION_REQUIRES_VISUAL_QC"

SPLIT_ARC_MM = 20.25
BIFURCATION_EXCLUSION_MM = 1.0
ARC_STEP_MM = 0.50
TANGENT_HALF_WINDOW = 2
N_ANGLES = 72
RADIAL_STEP_MM = 0.10
LUMEN_MAX_RADIUS_MM = 4.0
LUMEN_MIN_RADIUS_MM = 0.45

MIN_CENTER_HU = 200.0
MIN_VALID_RADIAL_FRACTION = 0.60
MAX_AXIS_PROXY = 2.50
MIN_STATION_PASS_FRACTION = 0.85
MAX_MEDIAN_AXIS_PROXY = 2.00
MAX_P90_AXIS_PROXY = 2.50

STATUS_PASS = "LCX_RESEARCH_STRUCTURAL_LABELS_FROZEN"
STATUS_FAIL = "LCX_RESEARCH_STRUCTURAL_SOURCE_QC_FAILED"


def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _read_json(p):
    return json.loads(_req(p).read_text(encoding="utf-8"))


def _write_json(p, obj):
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


def _resample(p, step=ARC_STEP_MM):
    p = np.asarray(p, float)
    a = _arc(p)
    q = np.arange(0.0, float(a[-1]) + 1e-9, float(step))
    if len(q) == 0 or q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(p, q), q


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _orth_basis(t):
    t = _unit(t)
    axes = np.eye(3)
    seed = axes[np.argmin(np.abs(axes @ t))]
    u = _unit(np.cross(t, seed))
    v = _unit(np.cross(t, u))
    return u, v


def _smooth_tangents(centers, half=TANGENT_HALF_WINDOW):
    centers = np.asarray(centers, float)
    grad = np.gradient(centers, axis=0)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)
    out = np.zeros_like(centers)
    for i in range(len(centers)):
        lo, hi = max(0, i-half), min(len(centers), i+half+1)
        pts = centers[lo:hi]
        if len(pts) < 3:
            out[i] = grad[i]
            continue
        q = pts - pts.mean(axis=0)
        val, vec = np.linalg.eigh(q.T @ q)
        t = vec[:, int(np.argmax(val))]
        if float(t @ grad[i]) < 0:
            t = -t
        out[i] = _unit(t)
    return out


class SourceGeometry:
    def __init__(self, meta):
        self.spacing_zyx = np.asarray(meta["spacing_zyx"], float)
        self.spacing_xyz = self.spacing_zyx[::-1]
        self.origin = np.asarray(meta["positions_lps_mm"][0], float)
        iop = np.asarray(meta["image_orientation_patient"], float)
        row, col = iop[:3], iop[3:]
        slc = np.cross(row, col)
        self.D = np.array([
            [row[0], col[0], slc[0]],
            [row[1], col[1], slc[1]],
            [row[2], col[2], slc[2]],
        ], float)
        self.invD = np.linalg.inv(self.D)

    def xyz_to_zyx(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        xyz = ((pts - self.origin) @ self.invD.T) / self.spacing_xyz
        return xyz[:, ::-1]


def _load_source(cache):
    cache = Path(cache)
    src = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    meta = _read_json(cache / "series7_int16.json")
    geom = SourceGeometry(meta)
    vv = float(np.prod(geom.spacing_zyx))
    if not (0.001 < vv < 0.5):
        raise RuntimeError(f"Unexpected source voxel volume {vv}")
    return geom, src, vv


def _sample(geom, src, pts, cval=-1024.0):
    z = geom.xyz_to_zyx(pts)
    return map_coordinates(np.asarray(src), z.T, order=1, mode="constant", cval=float(cval))


def _load_path(p):
    d = pd.read_csv(_req(p))
    for cols in (("lps_x_mm", "lps_y_mm", "lps_z_mm"), ("x_mm", "y_mm", "z_mm")):
        if all(c in d.columns for c in cols):
            return d[list(cols)].to_numpy(float)
    raise ValueError(f"No LPS coordinate columns in {p}: {list(d.columns)}")


def _post_split(path, split_arc=SPLIT_ARC_MM):
    a = _arc(path)
    if a[-1] <= split_arc:
        raise RuntimeError(f"Path ends before split arc {split_arc:.2f} mm")
    q = np.arange(split_arc, a[-1] + 1e-9, 0.25)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(path, q), q - split_arc


def _cmedian(x):
    return np.median(np.stack([np.roll(x, k) for k in range(-2, 3)]), axis=0)


def _station_qc(geom, src, c, t):
    u, v = _orth_basis(t)
    th = np.linspace(0.0, 2*np.pi, N_ANGLES, endpoint=False)
    dirs = np.cos(th)[:, None]*u + np.sin(th)[:, None]*v

    center_pts = np.vstack([c, c+.15*u, c-.15*u, c+.15*v, c-.15*v])
    center_hu = float(np.median(_sample(geom, src, center_pts)))
    threshold = float(np.clip(.55*center_hu, 220.0, 500.0))

    rr = np.arange(.2, LUMEN_MAX_RADIUS_MM + 1e-9, RADIAL_STEP_MM)
    P = c[None, None, :] + dirs[:, None, :]*rr[None, :, None]
    hu = _sample(geom, src, P.reshape(-1, 3)).reshape(N_ANGLES, len(rr))

    radii = np.full(N_ANGLES, np.nan)
    for i in range(N_ANGLES):
        b = (hu[i] < threshold) & (rr >= LUMEN_MIN_RADIUS_MM)
        ix = np.flatnonzero(b[:-1] & b[1:])
        if len(ix):
            radii[i] = rr[int(ix[0])]

    valid = np.isfinite(radii)
    vf = float(valid.mean())
    if valid.any():
        med = float(np.median(radii[valid]))
        radii[~valid] = med
        radii = _cmedian(radii)
        radii = np.clip(radii, max(.4, med-.75), min(4.0, med+.75))
        p10, p50, p90 = np.percentile(radii, [10, 50, 90])
        axis = float(p90 / max(p10, 1e-6))
        area = float(.5*np.sum(radii*radii)*(2*np.pi/N_ANGLES))
    else:
        p10 = p50 = p90 = axis = area = np.nan

    pass_qc = bool(
        center_hu >= MIN_CENTER_HU
        and vf >= MIN_VALID_RADIAL_FRACTION
        and np.isfinite(axis)
        and axis <= MAX_AXIS_PROXY
    )
    return {
        "center_hu": center_hu,
        "lumen_threshold_hu": threshold,
        "valid_radial_fraction": vf,
        "lumen_radius_p10_mm": float(p10),
        "lumen_radius_median_mm": float(p50),
        "lumen_radius_p90_mm": float(p90),
        "lumen_axis_proxy": float(axis),
        "lumen_area_mm2": float(area),
        "station_qc_pass": pass_qc,
    }


def _quantify_path(geom, src, p, vessel):
    centers, arcs = _resample(p)
    tangents = _smooth_tangents(centers)
    rows = []
    for i, (c, a, t) in enumerate(zip(centers, arcs, tangents)):
        q = _station_qc(geom, src, c, t)
        rows.append({
            "vessel": vessel,
            "station_index": i,
            "post_split_arc_mm": float(a),
            "lps_x_mm": float(c[0]),
            "lps_y_mm": float(c[1]),
            "lps_z_mm": float(c[2]),
            "tangent_x": float(t[0]),
            "tangent_y": float(t[1]),
            "tangent_z": float(t[2]),
            **q,
        })
    return pd.DataFrame(rows), centers, tangents


def _summarize_vessel(df):
    eval_df = df[df.post_split_arc_mm >= BIFURCATION_EXCLUSION_MM].copy()
    if eval_df.empty:
        raise RuntimeError("No evaluable post-bifurcation stations")
    axis = eval_df.lumen_axis_proxy.to_numpy(float)
    return {
        "evaluated_station_count": int(len(eval_df)),
        "excluded_bifurcation_zone_mm": BIFURCATION_EXCLUSION_MM,
        "station_qc_pass_fraction": float(eval_df.station_qc_pass.mean()),
        "median_center_hu": float(np.median(eval_df.center_hu)),
        "minimum_center_hu": float(np.min(eval_df.center_hu)),
        "median_valid_radial_fraction": float(np.median(eval_df.valid_radial_fraction)),
        "median_axis_proxy": float(np.nanmedian(axis)),
        "p90_axis_proxy": float(np.nanpercentile(axis, 90)),
        "maximum_axis_proxy": float(np.nanmax(axis)),
        "median_lumen_radius_mm": float(np.nanmedian(eval_df.lumen_radius_median_mm)),
    }


def _vessel_gate(s):
    return bool(
        s["station_qc_pass_fraction"] >= MIN_STATION_PASS_FRACTION
        and s["median_axis_proxy"] <= MAX_MEDIAN_AXIS_PROXY
        and s["p90_axis_proxy"] <= MAX_P90_AXIS_PROXY
        and s["median_center_hu"] >= 250.0
    )


def _plane_image(geom, src, c, t, half=5.0, step=.15):
    u, v = _orth_basis(t)
    q = np.arange(-half, half+1e-9, step)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    P = c + xx[..., None]*u + yy[..., None]*v
    im = _sample(geom, src, P.reshape(-1, 3)).reshape(len(q), len(q))
    return im, q


def _informative_indices(df, max_n=8):
    eval_df = df[df.post_split_arc_mm >= BIFURCATION_EXCLUSION_MM].copy()
    ids = []
    for frac in (0.0, .25, .5, .75, 1.0):
        target = float(eval_df.post_split_arc_mm.min() + frac*(eval_df.post_split_arc_mm.max()-eval_df.post_split_arc_mm.min()))
        ids.append(int(eval_df.iloc[np.argmin(np.abs(eval_df.post_split_arc_mm.to_numpy(float)-target))].station_index))
    for _, r in eval_df.nlargest(3, "lumen_axis_proxy").iterrows():
        ids.append(int(r.station_index))
    return list(dict.fromkeys(ids))[:max_n]


def _plot_dense_qc(geom, src, all_df, centers_map, tangents_map, out):
    ids_by = {v: _informative_indices(all_df[all_df.vessel == v]) for v in ("C6", "C7")}
    fig, axes = plt.subplots(2, 8, figsize=(20, 5.7))
    axes = np.asarray(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for row, vessel in enumerate(("C6", "C7")):
        dfv = all_df[all_df.vessel == vessel].copy()
        for col, idx in enumerate(ids_by[vessel][:8]):
            ax = axes[row, col]
            rec = dfv[dfv.station_index == idx].iloc[0]
            im, q = _plane_image(geom, src, centers_map[vessel][idx], tangents_map[vessel][idx])
            ax.imshow(im, cmap="gray", vmin=-100, vmax=900, extent=[q[0], q[-1], q[-1], q[0]])
            ax.scatter([0], [0], s=9)
            ax.set_title(
                f"{vessel} +{rec.post_split_arc_mm:.1f}\\naxis {rec.lumen_axis_proxy:.2f} | HU {rec.center_hu:.0f}",
                fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle("LCX structural dense source-plane QC: representative + worst compactness stations")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_metrics(all_df, out):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for vessel in ("C6", "C7"):
        d = all_df[all_df.vessel == vessel]
        ax.plot(d.post_split_arc_mm, d.lumen_axis_proxy, marker="o", ms=3, label=f"{vessel} axis proxy")
    ax.axhline(MAX_AXIS_PROXY, linestyle="--", linewidth=1)
    ax.axvspan(0, BIFURCATION_EXCLUSION_MM, alpha=.07)
    ax.set_xlabel("Post-split arc (mm)")
    ax.set_ylabel("p90/p10 lumen axis proxy")
    ax.set_title("LCX structural compactness along source paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _plot_geometry(c6, c7, out):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(c6[:,0], c6[:,1], c6[:,2], lw=3, label="C6 LCX-like parent")
    ax.plot(c7[:,0], c7[:,1], c7[:,2], lw=3, label="C7 OM-like daughter")
    ax.scatter([c6[0,0]],[c6[0,1]],[c6[0,2]],s=45,label="split neighborhood")
    ax.set_title("LCX structural research paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def synthetic_self_test():
    s = {
        "station_qc_pass_fraction": .95,
        "median_axis_proxy": 1.4,
        "p90_axis_proxy": 2.0,
        "median_center_hu": 450.0,
    }
    assert _vessel_gate(s)
    s2 = dict(s); s2["p90_axis_proxy"] = 2.7
    assert not _vessel_gate(s2)
    p = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    rp, q = _resample(p, .5)
    assert np.isclose(q[-1], 2.0)
    assert len(rp) == 5
    return {"ok": True, "resampled_points": len(rp)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out/"run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master = _read_json(root/MASTER)
    structural = _read_json(root/STRUCTURAL_SUMMARY)
    if master.get("status") != EXPECTED_MASTER_STATUS:
        raise RuntimeError(f"Frozen master prerequisite failed: {master.get('status')}")
    if structural.get("status") != EXPECTED_STRUCTURAL_STATUS:
        raise RuntimeError(f"Structural adjudication prerequisite failed: {structural.get('status')}")
    if not bool(structural.get("decision",{}).get("all_predeclared_structural_gates_pass",False)):
        raise RuntimeError("Structural adjudication gates are no longer all passing")

    geom, src, vv = _load_source(root/SOURCE_CACHE)
    c6full = _load_path(root/C6_PATH)
    c7full = _load_path(root/C7_PATH)
    c6, _ = _post_split(c6full)
    c7, _ = _post_split(c7full)

    c6df, c6centers, c6t = _quantify_path(geom, src, c6, "C6")
    c7df, c7centers, c7t = _quantify_path(geom, src, c7, "C7")
    all_df = pd.concat([c6df, c7df], ignore_index=True)
    all_df.to_csv(out/"LCX_structural_dense_source_QC.csv", index=False)

    c6s = _summarize_vessel(c6df)
    c7s = _summarize_vessel(c7df)
    c6_pass = _vessel_gate(c6s)
    c7_pass = _vessel_gate(c7s)
    source_qc_pass = bool(c6_pass and c7_pass)

    gates = {
        "prior_structural_adjudication_all_gates_pass": True,
        "C6_dense_source_qc_pass": c6_pass,
        "C7_dense_source_qc_pass": c7_pass,
        "master_still_frozen": master.get("status") == EXPECTED_MASTER_STATUS,
        "clinical_identity_not_promoted": True,
        "LM_remains_unresolved": structural.get("decision",{}).get("LM_status") == "UNRESOLVED",
    }
    status = STATUS_PASS if all(gates.values()) and source_qc_pass else STATUS_FAIL

    pd.DataFrame([{"gate":k,"pass":bool(v)} for k,v in gates.items()]).to_csv(
        out/"LCX_structural_source_QC_gates.csv", index=False
    )

    _plot_dense_qc(
        geom, src, all_df,
        {"C6":c6centers,"C7":c7centers},
        {"C6":c6t,"C7":c7t},
        out/"01_LCX_structural_dense_source_planes.png"
    )
    _plot_metrics(all_df, out/"02_LCX_structural_compactness.png")
    _plot_geometry(c6, c7, out/"03_LCX_structural_geometry.png")

    decision = {
        "status": status,
        "research_structural_label_C6": "LCX-like parent continuation" if status == STATUS_PASS else "UNRESOLVED",
        "research_structural_label_C7": "OM-like daughter" if status == STATUS_PASS else "UNRESOLVED",
        "research_structural_labels_frozen": bool(status == STATUS_PASS),
        "clinical_LCX_OM_identity_established": False,
        "LM_status": "UNRESOLVED",
        "master_anatomy_modified": False,
        "downstream_quantification_allowed": bool(status == STATUS_PASS),
        "downstream_scope": (
            "Research-only source-space quantification on the exact frozen C6/C7 structural paths. "
            "Do not relabel the master anatomy and do not call these clinically established LCX/OM."
            if status == STATUS_PASS else
            "No downstream LCX-like/OM-like quantification."
        ),
    }
    _write_json(out/"structural_freeze_decision.json", decision)

    summary = {
        "status": status,
        "algorithm": ALGORITHM,
        "baseline_commit": BASELINE,
        "master_status": master.get("status"),
        "master_modified": False,
        "source_voxel_volume_mm3": vv,
        "prior_structural_status": structural.get("status"),
        "C6_post_split_length_mm": float(_arc(c6)[-1]),
        "C7_post_split_length_mm": float(_arc(c7)[-1]),
        "bifurcation_exclusion_mm": BIFURCATION_EXCLUSION_MM,
        "C6_dense_source_QC": c6s,
        "C7_dense_source_QC": c7s,
        "gates": gates,
        "decision": decision,
        "scientific_boundary": (
            "This freezes OpenPlaque research structural labels only if dense source-plane QC passes. "
            "C6 remains 'LCX-like parent continuation' and C7 remains 'OM-like daughter'. Clinical LCX/OM identity is not established, "
            "LM remains unresolved, and the frozen master anatomy is unchanged."
        ),
    }
    _write_json(out/"summary.json", summary)
    _write_json(out/"input_provenance.json", {
        "source_cache": str(root/SOURCE_CACHE),
        "master": str(root/MASTER),
        "structural_adjudication_summary": str(root/STRUCTURAL_SUMMARY),
        "C6_path": str(root/C6_PATH),
        "C7_path": str(root/C7_PATH),
    })

    report = out/"OPENPLAQUE_LCX_STRUCTURAL_SOURCE_QC_FREEZE_REPORT.html"
    report.write_text(
        "<html><body><h1>OpenPlaque LCX Structural Source-QC Freeze v1</h1>"
        f"<p><b>Status:</b> {status}</p>"
        f"<p>C6 dense source QC pass: {c6_pass}; station pass fraction {c6s['station_qc_pass_fraction']:.3f}; "
        f"median axis {c6s['median_axis_proxy']:.3f}; p90 axis {c6s['p90_axis_proxy']:.3f}.</p>"
        f"<p>C7 dense source QC pass: {c7_pass}; station pass fraction {c7s['station_qc_pass_fraction']:.3f}; "
        f"median axis {c7s['median_axis_proxy']:.3f}; p90 axis {c7s['p90_axis_proxy']:.3f}.</p>"
        "<p><b>Boundary:</b> research structural labels only; clinical identity not established; LM unresolved; master unchanged.</p>"
        "<h2>C6</h2><pre>"+json.dumps(c6s,indent=2,default=str)+"</pre>"
        "<h2>C7</h2><pre>"+json.dumps(c7s,indent=2,default=str)+"</pre>"
        '<img src="01_LCX_structural_dense_source_planes.png" style="max-width:100%">'
        '<img src="02_LCX_structural_compactness.png" style="max-width:100%">'
        '<img src="03_LCX_structural_geometry.png" style="max-width:100%">'
        "</body></html>",
        encoding="utf-8",
    )

    _write_json(out/"run_state.json", {
        "status":"COMPLETE","result_status":status,"algorithm":ALGORITHM,"baseline":BASELINE
    })
    zpath = out/"OPENPLAQUE_LCX_STRUCTURAL_SOURCE_QC_FREEZE_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p != zpath:
                z.write(p,p.name)

    return {"summary":summary,"report":str(report),"zip":str(zpath)}
