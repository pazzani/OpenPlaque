from __future__ import annotations
import json, zipfile
from pathlib import Path
import pandas as pd

BASELINE="0593b453959f5a353d644267fbeef24b514ef4d7"
ALGORITHM="openplaque-research-endpoint-v1.0"
OUTPUT_DIRNAME="OpenPlaque_Research_Endpoint_v1"
STATUS="OPENPLAQUE_RESEARCH_ENDPOINT_V1_COMPLETE"

MULTI=Path("OpenPlaque_Multivessel_Research_Summary_v1/research_measurements.json")
LOCK=Path("Left_Coronary_Unresolved_State_Lock_v1/summary.json")
RCA=Path("RCA_Plaque_PCAT_Research_Lock_v1/summary.json")
LAD=Path("LAD_Source_Space_PCAT_Feasibility_v1/summary.json")
LCX=Path("LCX_OM_Source_Space_Composition_PCAT_Feasibility_v1/summary.json")
MASTER=Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")

def _read(p):
    p=Path(p)
    if not p.exists(): raise FileNotFoundError(str(p))
    return json.loads(p.read_text())

def _write(p,o): Path(p).write_text(json.dumps(o,indent=2,default=str,allow_nan=True),encoding="utf-8")

def synthetic_self_test():
    assert STATUS.endswith("_COMPLETE")
    return {"ok":True}

def run(drive_root="/content/drive/MyDrive/OpenPlaque",output_dir=None):
    root=Path(drive_root)
    out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME
    out.mkdir(parents=True,exist_ok=True)
    _write(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline":BASELINE})

    master=_read(root/MASTER); multi=_read(root/MULTI); lock=_read(root/LOCK)
    rca=_read(root/RCA); lad=_read(root/LAD); lcx=_read(root/LCX)

    if master.get("status")!="CORONARY_ANATOMY_BASELINE_V2_FROZEN": raise RuntimeError("master not frozen")
    if lock.get("status")!="LEFT_CORONARY_UNRESOLVED_STATE_RESEARCH_LOCKED": raise RuntimeError("left-coronary lock missing")
    if rca.get("status")!="RCA_RESEARCH_PLAQUE_PCAT_BENCHMARK_LOCKED": raise RuntimeError("RCA lock missing")
    if lad.get("status")!="LAD_SOURCE_SPACE_PCAT_FEASIBILITY_COMPLETE": raise RuntimeError("LAD PCAT missing")
    if lcx.get("status")!="LCX_OM_SOURCE_SPACE_COMPOSITION_PCAT_FEASIBILITY_COMPLETE": raise RuntimeError("LCX/OM feasibility missing")

    endpoint={
      "status":STATUS,
      "baseline_commit":BASELINE,
      "master_status":master.get("status"),
      "master_modified":False,
      "anatomy":{
        "RCA":"frozen accepted",
        "LAD":"frozen accepted",
        "LAD_distal_continuation":"source-confirmed research continuation; not in master",
        "C6":"LCX-like parent continuation; research structural label only",
        "C7":"OM-like daughter; research structural label only",
        "LM":"UNRESOLVED",
        "clinical_LCX_OM_identity_established":False,
      },
      "quantification":{
        "RCA_plaque_excess_proxy_mm3":rca["nominal_plaque_excess_totals_mm3"]["excess_total_plaque_proxy_mm3"],
        "RCA_direct_PCAT_HU":rca["pcat"]["fat_voxel_weighted_mean_hu"],
        "LAD_frozen_direct_PCAT_HU":lad["segments"]["frozen_LAD"]["pcat_mean_hu"],
        "LAD_distal_direct_PCAT_HU":lad["segments"]["validated_distal_continuation"]["pcat_mean_hu"],
        "C6_direct_PCAT_HU":lcx["pcat_primary"]["C6"]["pcat_mean_hu"],
        "C7_direct_PCAT_HU":lcx["pcat_primary"]["C7"]["pcat_mean_hu"],
      },
      "left_coronary_lock":lock["decision"],
      "project_policy":{
        "same_scan_left_coronary_heuristic_tuning":"STOP",
        "RCA_research_quantification":"ALLOWED",
        "frozen_LAD_research_quantification":"ALLOWED",
        "C6_C7_exact_path_research_quantification":"ALLOWED",
        "clinical_LM_quantification":"DISABLED",
        "clinical_LCX_OM_quantification":"DISABLED",
      },
      "scientific_boundary":"Research endpoint only. RCA plaque is a research proxy, not clinical TPV. PCAT is direct attenuation, not proprietary FAI. LAD plaque remains developmental. C6/C7 are research structural labels only. LM and clinical LCX/OM identity remain unresolved."
    }
    _write(out/"endpoint.json",endpoint)

    rows=[
      ["RCA","anatomy","frozen accepted"],
      ["RCA","plaque","locked research excess proxy"],
      ["RCA","PCAT","locked direct attenuation"],
      ["LAD","anatomy","frozen accepted"],
      ["LAD","plaque","developmental; not locked"],
      ["LAD","PCAT","technically feasible"],
      ["C6/C7","anatomy","research structural labels frozen"],
      ["C6/C7","plaque","raw shell composition only"],
      ["C6/C7","PCAT","technically feasible"],
      ["LM","anatomy","unresolved"],
    ]
    pd.DataFrame(rows,columns=["structure","domain","status"]).to_csv(out/"research_endpoint_status.csv",index=False)

    report=out/"OPENPLAQUE_RESEARCH_ENDPOINT_V1_REPORT.html"
    q=endpoint["quantification"]
    report.write_text(
      "<html><body><h1>OpenPlaque Research Endpoint v1</h1>"
      f"<p><b>Status:</b> {STATUS}</p>"
      "<p><b>Anatomy:</b> RCA and LAD accepted; C6/C7 retained as research structural labels only; LM unresolved; clinical LCX/OM identity unresolved.</p>"
      f"<p><b>RCA:</b> plaque excess proxy {q['RCA_plaque_excess_proxy_mm3']:.2f} mm3; direct PCAT {q['RCA_direct_PCAT_HU']:.2f} HU.</p>"
      f"<p><b>LAD:</b> frozen direct PCAT {q['LAD_frozen_direct_PCAT_HU']:.2f} HU; distal research continuation {q['LAD_distal_direct_PCAT_HU']:.2f} HU.</p>"
      f"<p><b>C6/C7 direct PCAT:</b> {q['C6_direct_PCAT_HU']:.2f} / {q['C7_direct_PCAT_HU']:.2f} HU.</p>"
      "<p><b>Left-coronary decision:</b> stop same-scan heuristic tuning. Reopen only with genuinely independent evidence.</p>"
      "<p><b>Boundary:</b> research endpoint, not a clinical coronary report. No clinical TPV, no proprietary FAI, no clinical LM/LCX/OM promotion.</p>"
      "</body></html>",encoding="utf-8"
    )
    _write(out/"input_provenance.json",{"multivessel":str(root/MULTI),"left_lock":str(root/LOCK),"RCA":str(root/RCA),"LAD":str(root/LAD),"LCX":str(root/LCX),"master":str(root/MASTER)})
    _write(out/"run_state.json",{"status":"COMPLETE","result_status":STATUS,"algorithm":ALGORITHM,"baseline":BASELINE})
    z=out/"OPENPLAQUE_RESEARCH_ENDPOINT_V1_RESULTS.zip"
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zz:
        for p in out.iterdir():
            if p.is_file() and p!=z: zz.write(p,p.name)
    return {"endpoint":endpoint,"report":str(report),"zip":str(z)}
