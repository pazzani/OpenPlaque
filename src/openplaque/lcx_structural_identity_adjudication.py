from __future__ import annotations

"""Gate-based structural identity adjudication for the established C6/C7 left-coronary branches.

No new vessel search is performed. The module combines only pre-existing validated evidence:
consensus-trunk establishment, parent/daughter topology with C6/C9 truncation, calibrated
direct chamber-interface behavior, accepted distal C7 source support, and direct source-CCTA
support of the established C6 segment. C6 distal-extension failures are recorded only as a
limitation and do not count for or against identity.

Research use only. A positive result freezes structural labels "C6 LCX-like parent continuation"
and "C7 OM-like daughter" for OpenPlaque research bookkeeping; it does not establish clinical
vessel identity, does not resolve LM, and does not alter the frozen master anatomy baseline.
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
from scipy.ndimage import map_coordinates

BASELINE = "0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM = "lcx-structural-identity-adjudication-v1.0"
OUTPUT_DIRNAME = "LCX_Structural_Identity_Adjudication_v1"

MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
SOURCE_CACHE = Path("Cache/Secondary_3D_Vesselness_Topology_v1")
TS = Path("TotalSegmentator_Cardiovascular_Cache_v1")
CUR = TS / "coronary_arteries/coronary_arteries.nii.gz"
LEG = TS / "coronary_arteries_LEGACY/coronary_arteries.nii.gz"
C6_PATH = Path("Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv")
C7_PATH = Path("LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv")
CONSENSUS = Path("LCX_Consensus_Trunk_AV_Groove_v1/summary.json")
TOPOLOGY = Path("LCX_Parent_Continuation_Topology_v1/summary.json")
CHAMBER = Path("LCX_Chamber_Interface_Midpoint_v1/summary.json")
DISTAL = Path("LCX_Distal_Reacquisition_v1_fixed/summary.json")
MONOTONIC = Path("LCX_C6_Monotonic_Distal_v1/summary.json")

STATUS_POS = "C6_LCX_LIKE_PARENT_C7_OM_LIKE_DAUGHTER_STRUCTURAL_ADJUDICATION_REQUIRES_VISUAL_QC"
STATUS_FAIL = "STRUCTURAL_IDENTITY_ADJUDICATION_GATE_FAILED"

def _req(p):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p

def _read_json(p):
    return json.loads(_req(p).read_text())

def _write_json(p, x):
    Path(p).write_text(json.dumps(x, indent=2, default=str, allow_nan=True), encoding="utf-8")

def _find(obj, key, default=None):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            out = _find(v, key, None)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for v in obj:
            out = _find(v, key, None)
            if out is not None:
                return out
    return default

def _source(cache):
    a = np.load(_req(cache / "series7_int16.npy"), mmap_mode="r")
    m = _read_json(cache / "series7_int16.json")
    ref = sitk.GetImageFromArray(np.asarray(a))
    sp_zyx = np.asarray(m["spacing_zyx"], float)
    ref.SetSpacing(tuple(sp_zyx[::-1]))
    ref.SetOrigin(tuple(np.asarray(m["positions_lps_mm"][0], float)))
    iop = np.asarray(m["image_orientation_patient"], float)
    row, col = iop[:3], iop[3:]
    slc = np.cross(row, col)
    D = np.array([[row[0], col[0], slc[0]],
                  [row[1], col[1], slc[1]],
                  [row[2], col[2], slc[2]]], float)
    ref.SetDirection(tuple(D.ravel()))
    return ref, a

def _xyz_to_zyx(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    idx_xyz = ((pts - o) @ np.linalg.inv(D).T) / sp
    return idx_xyz[:, ::-1]

def _zyx_to_xyz(img, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    idx_xyz = pts[:, ::-1]
    o = np.asarray(img.GetOrigin(), float)
    sp = np.asarray(img.GetSpacing(), float)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    return o + (idx_xyz * sp) @ D.T

def _load_path(path, ref):
    d = pd.read_csv(_req(path))
    for c in (("lps_x_mm","lps_y_mm","lps_z_mm"), ("x_mm","y_mm","z_mm")):
        if all(x in d.columns for x in c):
            return d[list(c)].to_numpy(float)
    for c in (("zyx_z","zyx_y","zyx_x"), ("source_z","source_y","source_x"), ("z","y","x")):
        if all(x in d.columns for x in c):
            return _zyx_to_xyz(ref, d[list(c)].to_numpy(float))
    raise ValueError(f"No recognized path coordinates in {path}: {list(d.columns)}")

def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] if len(p) > 1 else np.zeros(len(p))

def _interp(p, q):
    a = _arc(p); q = np.asarray(q, float)
    return np.column_stack([np.interp(q, a, p[:, k]) for k in range(3)])

def _resample(p, step=.25):
    a = _arc(p)
    if len(p) < 2 or a[-1] <= 0:
        return np.asarray(p, float), a
    q = np.arange(0, a[-1] + 1e-9, step)
    if q[-1] < a[-1] - 1e-6:
        q = np.r_[q, a[-1]]
    return _interp(np.asarray(p, float), q), q

def _resample_mask(path, ref):
    im = sitk.ReadImage(str(_req(path)))
    same = (im.GetSize() == ref.GetSize() and np.allclose(im.GetSpacing(), ref.GetSpacing()) and
            np.allclose(im.GetOrigin(), ref.GetOrigin()) and np.allclose(im.GetDirection(), ref.GetDirection()))
    if not same:
        im = sitk.Resample(im, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(im) > 0

def _sample_source(ref, arr, pts):
    z = _xyz_to_zyx(ref, pts)
    return map_coordinates(np.asarray(arr), z.T, order=1, mode="constant", cval=-1024.0)

def _path_support(ref, arr, cur, leg, p):
    p, _ = _resample(p, .25)
    h = _sample_source(ref, arr, p)
    z = np.rint(_xyz_to_zyx(ref, p)).astype(int)
    z = np.clip(z, [0,0,0], np.asarray(arr.shape) - 1)
    cs = cur[tuple(z.T)]
    ls = leg[tuple(z.T)]
    return {
        "length_mm": float(_arc(p)[-1]),
        "median_hu": float(np.median(h)),
        "robust_hu_fraction": float(np.mean((h >= 120) & (h <= 1200))),
        "current_support_fraction": float(np.mean(cs)),
        "legacy_support_fraction": float(np.mean(ls)),
        "union_support_fraction": float(np.mean(cs | ls)),
        "dual_support_fraction": float(np.mean(cs & ls)),
    }

def adjudicate_evidence(e):
    gates = {
        "consensus_trunk_established": bool(e["consensus_trunk_established"]),
        "topology_parent_daughter_established": bool(e["topology_parent_daughter_established"]),
        "c6_c9_truncation_established": bool(e["c6_c9_truncation_established"]),
        "chamber_method_replicated_control_pass": bool(e["chamber_method_replicated_control_pass"]),
        "c6_chamber_winner": bool(e["chamber_winner_id"] == 6),
        "chamber_score_margin_ge_0_08": bool(e["chamber_score_margin"] >= .08),
        "chamber_distance_or_alignment_advantage": bool(e["median_distance_advantage_mm"] >= 1.5 or e["alignment_advantage"] >= .15),
        "chamber_departure_advantage": bool(e["departure_slope_advantage_mm_per_mm"] >= .20 or e["endpoint_distance_advantage_mm"] >= 2.0),
        "c7_distal_source_extension_accepted": bool(e["c7_distal_accepted"]),
        "c7_distal_length_ge_5mm": bool(e["c7_distal_new_length_mm"] >= 5.0),
        "c7_distal_source_quality": bool(e["c7_distal_robust_hu_fraction"] >= .90 and e["c7_distal_union_support_fraction"] >= .90 and e["c7_distal_tortuosity"] <= 1.8),
        "c6_established_segment_source_quality": bool(e["c6_robust_hu_fraction"] >= .90 and e["c6_current_support_fraction"] >= .90 and e["c6_legacy_support_fraction"] >= .90),
        "c7_combined_segment_source_quality": bool(e["c7_robust_hu_fraction"] >= .90 and e["c7_current_support_fraction"] >= .90 and e["c7_union_support_fraction"] >= .90),
        "c6_extension_controls_replicated": bool(e["c6_dijkstra_control_pass"] and e["c6_monotonic_control_pass"]),
        "master_still_frozen": bool(e["master_status"] == "CORONARY_ANATOMY_BASELINE_V2_FROZEN"),
    }
    passed = all(gates.values())
    return gates, passed

def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    root = Path(drive_root)
    out = Path(output_dir) if output_dir else root / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_state.json", {"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})

    required = [root/MASTER, root/SOURCE_CACHE/"series7_int16.npy", root/SOURCE_CACHE/"series7_int16.json",
                root/CUR, root/LEG, root/C6_PATH, root/C7_PATH, root/CONSENSUS, root/TOPOLOGY,
                root/CHAMBER, root/DISTAL, root/MONOTONIC]
    for p in required: _req(p)

    master = _read_json(root/MASTER); consensus = _read_json(root/CONSENSUS); topology = _read_json(root/TOPOLOGY)
    chamber = _read_json(root/CHAMBER); distal = _read_json(root/DISTAL); mono = _read_json(root/MONOTONIC)

    ref, src = _source(root/SOURCE_CACHE); cur = _resample_mask(root/CUR, ref); leg = _resample_mask(root/LEG, ref)
    c6 = _load_path(root/C6_PATH, ref); c7 = _load_path(root/C7_PATH, ref)
    split_arc = 20.25; c6_len = _arc(c6)[-1]; c7_len = _arc(c7)[-1]
    c6_post = _interp(c6, np.arange(split_arc, c6_len + 1e-9, .25))
    c7_post = _interp(c7, np.arange(split_arc, c7_len + 1e-9, .25))
    c6q = _path_support(ref, src, cur, leg, c6_post); c7q = _path_support(ref, src, cur, leg, c7_post)

    dsrc = distal.get("source_reacquisition", {}); d7 = dsrc.get("C7", {}); d7best = d7.get("best", {})
    dctrl = distal.get("chamber_interface_controls", {}); ddec = distal.get("decision", {})
    msrc = mono.get("C6_source_reacquisition", {}); mctrl = msrc.get("control", {})
    csrc = distal.get("source_reacquisition", {}).get("C6", {}).get("control", {})

    evidence = {
        "master_status": master.get("status"),
        "consensus_trunk_established": str(consensus.get("status","")).startswith("CONSENSUS_TRUNK_ESTABLISHED"),
        "topology_parent_daughter_established": str(topology.get("status","")).startswith("C6_PARENT_CONTINUATION_C7_SIDE_BRANCH"),
        "c6_c9_truncation_established": bool(_find(topology, "truncation_gate_pass", False)),
        "chamber_method_replicated_control_pass": bool(chamber.get("chamber_interface_controls",{}).get("control_pass",False) and dctrl.get("control_pass",False)),
        "chamber_winner_id": int(ddec.get("winner_source_candidate_id", -1)),
        "chamber_score_margin": float(ddec.get("score_margin", 0.0)),
        "median_distance_advantage_mm": float(ddec.get("median_interface_distance_advantage_mm", 0.0)),
        "alignment_advantage": float(ddec.get("tangent_alignment_advantage", 0.0)),
        "departure_slope_advantage_mm_per_mm": float(ddec.get("departure_slope_advantage_mm_per_mm", 0.0)),
        "endpoint_distance_advantage_mm": float(ddec.get("endpoint_interface_distance_advantage_mm", 0.0)),
        "c7_distal_accepted": bool(d7.get("accepted", False)),
        "c7_distal_new_length_mm": float(d7best.get("new_length_mm", 0.0)),
        "c7_distal_robust_hu_fraction": float(d7best.get("robust_hu_fraction", 0.0)),
        "c7_distal_union_support_fraction": float(d7best.get("union_support_fraction", 0.0)),
        "c7_distal_tortuosity": float(d7best.get("tortuosity", 99.0)),
        "c6_robust_hu_fraction": c6q["robust_hu_fraction"],
        "c6_current_support_fraction": c6q["current_support_fraction"],
        "c6_legacy_support_fraction": c6q["legacy_support_fraction"],
        "c7_robust_hu_fraction": c7q["robust_hu_fraction"],
        "c7_current_support_fraction": c7q["current_support_fraction"],
        "c7_union_support_fraction": c7q["union_support_fraction"],
        "c6_dijkstra_control_pass": bool(csrc.get("control_pass", False)),
        "c6_monotonic_control_pass": bool(mctrl.get("control_pass", False)),
    }

    gates, passed = adjudicate_evidence(evidence); status = STATUS_POS if passed else STATUS_FAIL
    rows = [{"gate": k, "pass": bool(v)} for k,v in gates.items()]
    pd.DataFrame(rows).to_csv(out/"evidence_gate_matrix.csv", index=False)
    _write_json(out/"evidence_values.json", evidence)
    _write_json(out/"direct_source_support.json", {"C6_post_split":c6q,"C7_post_split_extended":c7q})

    limitation = {"C6_distal_extension_established": bool(msrc.get("accepted", False)),
                  "C6_monotonic_failure_reason": msrc.get("reason"),
                  "interpretation": "C6 distal-extension failure is recorded as a limitation only; it is not a positive or negative identity vote."}
    _write_json(out/"C6_extension_limitation.json", limitation)

    decision = {"status": status,"all_predeclared_structural_gates_pass": passed,
                "structural_label_C6": "LCX-like parent continuation" if passed else "UNRESOLVED",
                "structural_label_C7": "OM-like daughter" if passed else "UNRESOLVED",
                "clinical_LCX_OM_identity_established": False,"LM_status":"UNRESOLVED","master_anatomy_modified":False,
                "visual_QC_required_before_freeze": bool(passed)}
    _write_json(out/"structural_identity_decision.json", decision)

    plt.figure(figsize=(10,6)); y=np.arange(len(rows)); vals=[1 if r["pass"] else 0 for r in rows]
    plt.barh(y,vals); plt.yticks(y,[r["gate"].replace("_"," ") for r in rows],fontsize=8); plt.xlim(0,1.05)
    plt.xlabel("gate pass"); plt.title("Structural identity adjudication gates"); plt.tight_layout(); plt.savefig(out/"01_evidence_gate_matrix.png",dpi=180); plt.close()

    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d")
    ax.plot(c6_post[:,0],c6_post[:,1],c6_post[:,2],lw=3,label="C6 established"); ax.plot(c7_post[:,0],c7_post[:,1],c7_post[:,2],lw=3,label="C7 frozen extended")
    ax.scatter([c6_post[0,0]],[c6_post[0,1]],[c6_post[0,2]],s=45,label="F2 split"); ax.legend(); ax.set_title("Established source-space C6/C7 geometry")
    plt.tight_layout(); plt.savefig(out/"02_structural_geometry.png",dpi=180); plt.close()

    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for r,(name,p) in enumerate((("C6",c6_post),("C7",c7_post))):
        pp,qq=_resample(p,.25); picks=np.linspace(0,len(pp)-1,4).astype(int)
        for j,ix in enumerate(picks):
            center=pp[ix]
            tangent=(pp[min(2,len(pp)-1)]-pp[0]) if ix==0 else ((pp[-1]-pp[max(0,len(pp)-3)]) if ix==len(pp)-1 else (pp[min(len(pp)-1,ix+1)]-pp[max(0,ix-1)]))
            t=tangent/max(np.linalg.norm(tangent),1e-9); axes3=np.eye(3); seed=axes3[np.argmin(np.abs(axes3@t))]
            u=np.cross(t,seed); u/=max(np.linalg.norm(u),1e-9); v=np.cross(t,u); v/=max(np.linalg.norm(v),1e-9)
            qv=np.arange(-7,7.0001,.2); yy,xx=np.meshgrid(qv,qv,indexing="ij"); P=center+xx[...,None]*u+yy[...,None]*v
            im=_sample_source(ref,src,P.reshape(-1,3)).reshape(len(qv),len(qv)); ax=axes[r,j]
            ax.imshow(im,cmap="gray",vmin=-100,vmax=900,extent=[qv[0],qv[-1],qv[-1],qv[0]]); ax.scatter([0],[0],s=18); ax.set_title(f"{name} +{qq[ix]:.1f} mm")
    plt.tight_layout(); plt.savefig(out/"03_source_orthogonal_qc.png",dpi=180); plt.close()

    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status"),"LCX_master_status":"UNRESOLVED",
             "evidence":evidence,"gates":gates,"direct_source_support":{"C6_post_split":c6q,"C7_post_split_extended":c7q},"C6_extension_limitation":limitation,
             "decision":decision,"scientific_boundary":"Positive adjudication freezes research structural labels only: C6 LCX-like parent continuation and C7 OM-like daughter. It does not establish clinical vessel identity, does not resolve LM, and does not alter Master Coronary Anatomy Baseline v2.1."}
    _write_json(out/"summary.json",summary); _write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE})
    report=out/"OPENPLAQUE_LCX_STRUCTURAL_IDENTITY_ADJUDICATION_REPORT.html"
    report.write_text(f"<html><body><h1>OpenPlaque LCX Structural Identity Adjudication</h1><p><b>Status:</b> {status}</p><p>All predeclared gates pass: {passed}.</p><p>C6 structural label: {decision['structural_label_C6']}; C7 structural label: {decision['structural_label_C7']}.</p><p>Clinical identity remains unestablished; LM unresolved; master unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LCX_STRUCTURAL_IDENTITY_ADJUDICATION_REPORT_BACK.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    return {"summary":summary,"report":str(report),"zip":str(zpath)}

def synthetic_adjudication_self_test():
    e={"master_status":"CORONARY_ANATOMY_BASELINE_V2_FROZEN","consensus_trunk_established":True,"topology_parent_daughter_established":True,
       "c6_c9_truncation_established":True,"chamber_method_replicated_control_pass":True,"chamber_winner_id":6,"chamber_score_margin":.12,
       "median_distance_advantage_mm":3.0,"alignment_advantage":.16,"departure_slope_advantage_mm_per_mm":.5,"endpoint_distance_advantage_mm":5.0,
       "c7_distal_accepted":True,"c7_distal_new_length_mm":8.0,"c7_distal_robust_hu_fraction":1.0,"c7_distal_union_support_fraction":.95,
       "c7_distal_tortuosity":1.2,"c6_robust_hu_fraction":1.0,"c6_current_support_fraction":.95,"c6_legacy_support_fraction":.95,
       "c7_robust_hu_fraction":1.0,"c7_current_support_fraction":.95,"c7_union_support_fraction":.95,"c6_dijkstra_control_pass":True,"c6_monotonic_control_pass":True}
    gates,ok=adjudicate_evidence(e); assert ok and all(gates.values()); e["chamber_score_margin"]=.07; _,bad=adjudicate_evidence(e); assert not bad
    return {"ok":True,"n_gates":len(gates)}
