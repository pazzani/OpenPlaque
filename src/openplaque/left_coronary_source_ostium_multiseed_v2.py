from __future__ import annotations

"""Blind multi-seed source-CCTA coronary ostium recovery.

Follow-up to the failed v1.2 RCA positive control. Frozen HU/vesselness thresholds,
RCA lumen calibration, serial-plane acceptance gates, and post-hoc RCA control criteria
are retained. The prospective change is candidate parameterization: each connected root
component may produce several spatially separated near-surface seed hypotheses. Known
RCA/LAD/C6 coordinates never affect seed generation or blind path selection.
"""

import json, math, time, zipfile
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

BASELINE = _base.BASELINE
ALGORITHM = "left-coronary-source-ostium-multiseed-v2.0-control-recovery"
OUTPUT_DIRNAME = "Left_Coronary_Source_Ostium_Multiseed_v2"
CACHE_DIRNAME = _v12.CACHE_DIRNAME
CONTACT_MAX_MM = 2.0
SEED_MIN_SEPARATION_MM = 2.5
MAX_SEEDS_PER_COMPONENT = 10
CHECKPOINTS_MM = (6.0, 9.0, 12.0)
TOP_PER_SEED_CHECKPOINT = 3
BEAM_WIDTH = 48

STATUS_CONTROL_FAIL = "SOURCE_OSTIUM_MULTI_SEED_RCA_CONTROL_FAILED"
STATUS_NO_SECOND = "SOURCE_OSTIUM_MULTI_SEED_CONTROL_PASSED_NO_SECOND_EXIT"
STATUS_POS = "SOURCE_OSTIUM_MULTI_SEED_SECOND_EXIT_REQUIRES_VISUAL_QC"


def _surface_contact_seeds(comp, outside, spacing, contact_max_mm=CONTACT_MAX_MM,
                           min_sep_mm=SEED_MIN_SEPARATION_MM, max_seeds=MAX_SEEDS_PER_COMPONENT):
    pts = np.asarray(comp["points"], dtype=np.intp)
    od = np.asarray(outside)[tuple(pts.T)]
    keep = od <= float(contact_max_mm)
    if not np.any(keep):
        return []
    contact = pts[keep].astype(float); cod = od[keep].astype(float)
    order = np.argsort(cod, kind="stable"); contact, cod = contact[order], cod[order]
    sp = np.asarray(spacing, float); selected = [0]
    while len(selected) < int(max_seeds):
        d = np.full(len(contact), np.inf)
        for j in selected:
            d = np.minimum(d, np.linalg.norm((contact - contact[j]) * sp[None, :], axis=1))
        d[selected] = -np.inf
        i = int(np.argmax(d))
        if not np.isfinite(d[i]) or d[i] < float(min_sep_mm):
            break
        selected.append(i)
    return [{"seed_rank": r, "seed_zyx_local": contact[i].copy(), "seed_outside_mm": float(cod[i])}
            for r, i in enumerate(selected)]


def _outside_gradient(seed, outside, spacing, delta_mm=.8):
    seed = np.asarray(seed, float); sp = np.asarray(spacing, float); g = np.zeros(3, float)
    for a in range(3):
        dp = np.zeros(3); dp[a] = float(delta_mm) / sp[a]
        g[a] = (float(_base._sample_arr(outside, [seed + dp])[0]) -
                float(_base._sample_arr(outside, [seed - dp])[0])) / (2 * float(delta_mm))
    return _base._unit(g)


def _initial_directions(comp, seed, outside, spacing):
    pts = np.asarray(comp["points"], float); seed = np.asarray(seed, float); sp = np.asarray(spacing, float)
    near = pts[np.linalg.norm((pts - seed[None, :]) * sp[None, :], axis=1) <= 6.0]
    if len(near) >= 3:
        q = (near - seed[None, :]) * sp[None, :]; q -= q.mean(axis=0)
        _, vecs = np.linalg.eigh(q.T @ q / max(len(q) - 1, 1)); pca = _base._unit(vecs[:, -1])
    else:
        pca = np.array([0., 1., 0.])
    grad = _outside_gradient(seed, outside, spacing)
    if np.linalg.norm(grad) < .5: grad = pca.copy()
    if np.dot(pca, grad) < 0: pca = -pca
    raw = [pca, grad, _base._unit(.65*pca + .35*grad), _base._unit(.35*pca + .65*grad)]
    out = []
    for d in raw:
        if np.linalg.norm(d) >= .5 and all(abs(float(np.dot(d, u))) < .985 for u in out): out.append(d)
    return out[:4]


def _beam_checkpoints(seed, tangent, src_roi, vessel, outside, spacing, v_thr,
                      max_mm=12., step=.45, beam_width=BEAM_WIDTH, checkpoints=CHECKPOINTS_MM):
    sp = np.asarray(spacing, float); origin = np.asarray(seed, float)
    d0 = float(_base._sample_arr(outside, [origin])[0])
    norm = max(float(np.median(vessel[vessel >= v_thr]))*1.5, .02) if np.any(vessel >= v_thr) else .02
    states = [(0., [origin], _base._unit(tangent), d0)]; captures = []; captured = set(); shape = np.asarray(src_roi.shape)
    for istep in range(1, int(math.ceil(max_mm/step))+1):
        nxt = []
        for score, pts, t, prev_od in states:
            for d in _base._cone_dirs(t):
                if np.dot(d, t) < math.cos(math.radians(58)): continue
                p = pts[-1] + step*d/sp
                if np.any(p < 1) or np.any(p > shape-2): continue
                od = float(_base._sample_arr(outside, [p])[0])
                if od < prev_od-.10: continue
                hu = float(_base._sample_arr(src_roi, [p], order=1, cval=-1024)[0]); vv = float(_base._sample_arr(vessel, [p], order=1, cval=0)[0])
                if not (120 <= hu <= 1200 and vv >= .40*v_thr): continue
                if len(pts) > 5 and np.min(np.linalg.norm((np.asarray(pts[:-4])-p)*sp[None, :], axis=1)) < .55: continue
                vn = min(1., vv/norm); progress = max(-.1, od-prev_od); align = max(0., float(np.dot(d,t)))
                hs = float(np.exp(-.5*((hu-560.)/360.)**2)); ns = score + 1.75*vn + .42*align + .55*progress/max(step,1e-6) + .18*hs
                nxt.append((ns, pts+[p], _base._unit(.72*t+.28*d), od))
        if not nxt: break
        nxt.sort(key=lambda x:x[0], reverse=True); keep=[]; bins=set()
        for st in nxt:
            key=tuple(np.round(st[1][-1]*sp/.35).astype(int))
            if key in bins: continue
            bins.add(key); keep.append(st)
            if len(keep) >= int(beam_width): break
        states=keep; length=istep*float(step)
        for cp in checkpoints:
            if cp not in captured and length >= cp:
                captures.extend({"checkpoint_mm":float(cp), "beam_score":float(st[0]), "path":np.asarray(st[1],float)} for st in states)
                captured.add(cp)
    return captures


def _trace_seed(comp, seed_info, src_roi, vessel, outside, spacing, v_thr, rca_cal, label):
    seed=np.asarray(seed_info["seed_zyx_local"],float); candidates=[]; sp=np.asarray(spacing,float)
    for idir,tangent in enumerate(_initial_directions(comp,seed,outside,spacing)):
        for x in _beam_checkpoints(seed,tangent,src_roi,vessel,outside,spacing,v_thr):
            m=_base._path_metrics(x["path"],src_roi,vessel,outside,spacing,v_thr)
            if _v12._cheap_path_gate(m,v_thr): candidates.append({**x,"direction_index":idir,"metrics":m})
    selected=[]
    for cp in CHECKPOINTS_MM:
        grp=sorted([x for x in candidates if abs(x["checkpoint_mm"]-cp)<1e-6], key=lambda x:x["beam_score"], reverse=True)
        bins=set(); n=0
        for x in grp:
            key=tuple(np.round(x["path"][-1]*sp/.8).astype(int))
            if key in bins: continue
            bins.add(key); selected.append(x); n+=1
            if n>=TOP_PER_SEED_CHECKPOINT: break
    rows=[]
    for j,x in enumerate(selected):
        qdf,qsum=_base._serial_qc(x["path"],src_roi,spacing,rca_cal,n=9,label=f"{label}_q{j}")
        gate=_v12._serial_gate(qsum,rca_cal); score=_v12._selection_score(x["metrics"],qsum,v_thr)
        rows.append((bool(gate),float(score),x,qdf,qsum))
    if not rows:
        return None,{**seed_info,"accepted":False,"reason":"no_candidate_passed_non_qc_gates","n_cheap_pass":len(candidates),"n_serial_qc":0},pd.DataFrame()
    rows.sort(key=lambda r:(r[0],r[1]),reverse=True); gate,score,x,qdf,qsum=rows[0]
    sm={**seed_info,"accepted":gate,"selection_score":score,"beam_score":float(x["beam_score"]),"checkpoint_mm":float(x["checkpoint_mm"]),
        "direction_index":int(x["direction_index"]),"path_metrics":{k:v for k,v in x["metrics"].items() if not isinstance(v,np.ndarray)},
        "serial_qc":qsum,"n_cheap_pass":len(candidates),"n_serial_qc":len(rows)}
    return x["path"],sm,qdf


def synthetic_seed_coverage_self_test():
    outside=np.ones((20,20,20),float)*9; pts=[]
    for x in range(3,17):
        pts.append([10,6,x]); outside[10,6,x]=.8+.05*abs(x-5)
        pts.append([10,13,x]); outside[10,13,x]=.8+.05*abs(x-14)
    seeds=_surface_contact_seeds({"points":np.asarray(pts,np.intp)},outside,np.ones(3),2.0,2.5,10)
    s=np.asarray([x["seed_zyx_local"] for x in seeds]); assert len(seeds)>=4
    assert np.min(np.linalg.norm(s-np.array([10.,6.,5.]),axis=1))<=2.5
    assert np.min(np.linalg.norm(s-np.array([10.,13.,14.]),axis=1))<=2.5
    return {"ok":True,"n_seeds":len(seeds)}


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_dir=None):
    t0=time.time(); root=Path(drive_root); out=Path(output_dir) if output_dir else root/OUTPUT_DIRNAME; out.mkdir(parents=True,exist_ok=True)
    _base._write_json(out/"run_state.json",{"status":"STARTED","algorithm":ALGORITHM,"baseline_commit":BASELINE})
    for p in [root/_base.SOURCE_CACHE/"series7_int16.npy",root/_base.SOURCE_CACHE/"series7_int16.json",root/_base.MASTER,root/_base.RCA_PATH,root/_base.LAD_PATH,root/_base.C6_PATH,root/_base.AORTA]: _base._req(p)
    master=json.loads((root/_base.MASTER).read_text()); ref,src,spacing=_base._source(root/_base.SOURCE_CACHE)
    rca=_base._load_path(root/_base.RCA_PATH,ref); lad=_base._load_path(root/_base.LAD_PATH,ref); c6=_base._load_path(root/_base.C6_PATH,ref); aorta=_base._resample_mask(root/_base.AORTA,ref)
    surface=aorta&~ndi.binary_erosion(aorta,iterations=1,border_value=0); stree=cKDTree(np.argwhere(surface)*spacing[None,:])
    d0=float(stree.query(rca[0]*spacing)[0]); d1=float(stree.query(rca[-1]*spacing)[0]); rca_o=rca if d0<=d1 else rca[::-1].copy(); rca_prox=rca_o[0]
    rca_raw,rca_cal=_base._calibrate_rca(rca_o,src,spacing); rca_raw.to_csv(out/"RCA_lumen_calibration_sections.csv",index=False); _base._write_json(out/"RCA_lumen_calibration.json",rca_cal)
    lo,hi,sl=_base._root_crop(src.shape,rca_prox,spacing); src_roi=np.asarray(src[sl]); aorta_roi=np.asarray(aorta[sl]); rca_local=rca_o-lo[None,:]
    vessel=_v12._frangi_3d_cached(src_roi,spacing,cache_dir=root/CACHE_DIRNAME,reuse_cache=True)
    rp,rs=_base._resample(rca_local,spacing,.25); rp=rp[rs<=8.0+1e-9]; rvals=_base._sample_arr(vessel,rp,order=1,cval=0)
    v_thr=max(.003,min(.12,.35*float(np.percentile(rvals,25)))) if len(rvals) else .01
    comps,outside=_base._blind_root_components(src_roi,aorta_roi,vessel,spacing,v_thr,rca_prox[0]-lo[0]); print("Blind root components:",len(comps),"v_thr:",v_thr)
    objects=[]; rows=[]
    for ic,comp in enumerate(comps[:40],1):
        seeds=_surface_contact_seeds(comp,outside,spacing); print(f"component {ic}/{min(40,len(comps))} id={comp['component_id']} seeds={len(seeds)}")
        for seed in seeds:
            p,sm,qdf=_trace_seed(comp,seed,src_roi,vessel,outside,spacing,v_thr,rca_cal,f"component_{comp['component_id']}_seed_{seed['seed_rank']}")
            sm.update(component_id=comp["component_id"],component_n_voxels=comp["n_voxels"],component_span_mm=comp["physical_span_mm"])
            if p is not None:
                pg=p+lo[None,:]; known,_=_base._resample(rca_o,spacing,.20); known=known[_base._arc(known,spacing)<=10.0+1e-9]; dd=cKDTree(known*spacing[None,:]).query(pg*spacing[None,:])[0]
                sm["postsearch_median_distance_to_known_RCA_mm"]=float(np.median(dd)); sm["postsearch_p90_distance_to_known_RCA_mm"]=float(np.percentile(dd,90)); sm["postsearch_seed_distance_to_known_RCA_prox_mm"]=float(np.linalg.norm((pg[0]-rca_prox)*spacing))
            else:
                sm["postsearch_median_distance_to_known_RCA_mm"]=sm["postsearch_p90_distance_to_known_RCA_mm"]=sm["postsearch_seed_distance_to_known_RCA_prox_mm"]=float("inf")
            rows.append({"component_id":comp["component_id"],"seed_rank":seed["seed_rank"],"seed_outside_mm":seed["seed_outside_mm"],"accepted":sm.get("accepted",False),"selection_score":sm.get("selection_score",np.nan),
                         "length_mm":sm.get("path_metrics",{}).get("length_mm",np.nan),"plane_pass_fraction":sm.get("serial_qc",{}).get("plane_pass_fraction",np.nan),"median_radius_mm":sm.get("serial_qc",{}).get("median_radius_mm",np.nan),
                         "median_distance_to_RCA_mm":sm.get("postsearch_median_distance_to_known_RCA_mm",np.nan),"seed_distance_to_RCA_prox_mm":sm.get("postsearch_seed_distance_to_known_RCA_prox_mm",np.nan),"n_serial_qc":sm.get("n_serial_qc",0)})
            objects.append((p,sm,qdf,comp))
    pd.DataFrame(rows).to_csv(out/"blind_multiseed_root_hypotheses.csv",index=False); traced=[x for x in objects if x[0] is not None]
    rca_obj=min(traced,key=lambda x:x[1].get("postsearch_median_distance_to_known_RCA_mm",np.inf)) if traced else None; rca_control=False
    if rca_obj is not None:
        s=rca_obj[1]; rca_control=bool(s.get("accepted",False) and s["postsearch_median_distance_to_known_RCA_mm"]<=1.0 and s["postsearch_p90_distance_to_known_RCA_mm"]<=1.8 and s["postsearch_seed_distance_to_known_RCA_prox_mm"]<=3.0)
        s["control_pass"]=rca_control; rca_obj[2].to_csv(out/"RCA_multiseed_discovery_serial_qc.csv",index=False); _base._write_json(out/"RCA_multiseed_discovery_control.json",s)
        pd.DataFrame(rca_obj[0]+lo[None,:],columns=["source_z","source_y","source_x"]).to_csv(out/"RCA_multiseed_discovery_path.csv",index=False)
    else: _base._write_json(out/"RCA_multiseed_discovery_control.json",{"control_pass":False,"reason":"no_traced_seed_hypotheses"})
    best=None
    if rca_control:
        rseed=(rca_obj[0][0]+lo)*spacing; eligible=[]
        for obj in objects:
            p,sm,qdf,comp=obj
            if p is None or not sm.get("accepted",False) or obj is rca_obj: continue
            seed_sep=float(np.linalg.norm((p[0]+lo)*spacing-rseed))
            if sm.get("postsearch_median_distance_to_known_RCA_mm",np.inf)>4.0 and seed_sep>=8.0: eligible.append(obj)
        if eligible:
            eligible.sort(key=lambda x:x[1].get("selection_score",0),reverse=True); best=eligible[0]; p,bsum,bqdf,bcomp=best; pg=p+lo[None,:]; pm=pg*spacing[None,:]
            bsum["posthoc_min_distance_to_LAD_mm"]=float(np.min(cKDTree(lad*spacing[None,:]).query(pm)[0])); bsum["posthoc_min_distance_to_C6_mm"]=float(np.min(cKDTree(c6*spacing[None,:]).query(pm)[0]))
            pd.DataFrame(pg,columns=["source_z","source_y","source_x"]).to_csv(out/"second_coronary_ostial_exit_path.csv",index=False); pd.DataFrame(_base._zyx_to_xyz(ref,pg),columns=["lps_x_mm","lps_y_mm","lps_z_mm"]).to_csv(out/"second_coronary_ostial_exit_lps.csv",index=False)
            bqdf.to_csv(out/"second_coronary_ostial_exit_serial_qc.csv",index=False); _base._write_json(out/"second_coronary_ostial_exit_summary.json",bsum)
    status=STATUS_CONTROL_FAIL if not rca_control else STATUS_POS if best is not None else STATUS_NO_SECOND; accepted=sum(bool(x[1].get("accepted",False)) for x in objects)
    summary={"status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE,"master_status":master.get("status","CORONARY_ANATOMY_BASELINE_V2_FROZEN"),"master_modified":False,"RCA_lumen_calibration":rca_cal,"source_vesselness_threshold":v_thr,
             "n_blind_root_components":len(comps),"n_surface_seed_hypotheses":len(objects),"n_traced_seed_hypotheses":len(traced),"n_accepted_seed_hypotheses":accepted,"RCA_blind_control":rca_obj[1] if rca_obj else {"control_pass":False},"second_coronary_candidate":best[1] if best else {"accepted":False},"elapsed_seconds":float(time.time()-t0)}
    _base._write_json(out/"summary.json",summary)
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection="3d")
    for comp in comps[:12]:
        pts=(comp["points"][::max(1,len(comp["points"])//100)]+lo[None,:])*spacing[None,:]; ax.scatter(pts[:,2],pts[:,1],pts[:,0],s=4,alpha=.25)
    if rca_obj is not None:
        q=(rca_obj[0]+lo[None,:])*spacing[None,:]; ax.plot(q[:,2],q[:,1],q[:,0],linewidth=3,label="closest blind seed hypothesis")
    if best is not None:
        q=(best[0]+lo[None,:])*spacing[None,:]; ax.plot(q[:,2],q[:,1],q[:,0],linewidth=3,label="second accepted exit")
    ax.set_title("Blind multi-seed source-CCTA root hypotheses"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/"01_multiseed_root_hypotheses.png",dpi=180); plt.close()
    for obj,name,title in [(rca_obj,"02_RCA_multiseed_orthogonal_qc.png",f"Closest blind seed hypothesis | RCA control={rca_control}"),(best,"03_second_exit_orthogonal_qc.png","Second accepted root exit")]:
        if obj is None: continue
        p,s=_base._resample(obj[0],spacing,.25); ss=np.linspace(.7,max(.7,s[-1]-.7),6); fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,x in zip(axes.ravel(),ss):
            i=int(np.argmin(np.abs(s-x))); i0=max(0,i-3); i1=min(len(p)-1,i+3); t=(p[i1]-p[i0])*spacing; im,c=_base._orthogonal_plane(src_roi,p[i],t,spacing,half_mm=5.,pix_mm=.20); ax.imshow(im,cmap="gray",vmin=0,vmax=900,extent=[c[0],c[-1],c[-1],c[0]]); ax.set_title(f"arc {s[i]:.1f} mm"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title); plt.tight_layout(); plt.savefig(out/name,dpi=180); plt.close()
    report=out/"OPENPLAQUE_LEFT_CORONARY_MULTI_SEED_OSTIUM_REPORT.html"; report.write_text(f"<html><body><h1>OpenPlaque blind multi-seed ostium recovery</h1><p><b>Status:</b> {status}</p><p>RCA control pass: {rca_control}</p><p>Accepted seed hypotheses: {accepted}</p><p>Second accepted exit: {best is not None}</p><p>Master baseline unchanged.</p></body></html>",encoding="utf-8")
    zpath=out/"OPENPLAQUE_LEFT_CORONARY_MULTI_SEED_OSTIUM_RESULTS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p!=zpath and p.is_file(): z.write(p,p.name)
    _base._write_json(out/"run_state.json",{"status":"COMPLETE","scientific_status":status,"algorithm":ALGORITHM,"baseline_commit":BASELINE}); return {"summary":summary,"report":str(report),"zip":str(zpath)}
