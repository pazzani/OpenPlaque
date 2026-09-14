from __future__ import annotations

"""OpenPlaque frozen coronary anatomy + source-CCTA QC master report.

This module does not discover or extend coronary anatomy. It consolidates frozen
geometry from prior validated experiments, encodes support classes explicitly,
samples source-CCTA orthogonal planes for QC, and creates an integrated report.

Research use only.
"""

import html
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ALGORITHM_VERSION = "master-coronary-anatomy-qc-v1.0"
SECONDARY_CONTINUOUS_SUPPORT_MM = 12.2
SECONDARY_TRANSITION_ONSET_MM = 12.4
SECONDARY_ISLAND_START_MM = 13.2
SECONDARY_ISLAND_END_MM = 13.6


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) == 0:
        return np.array([], float)
    if len(p) == 1:
        return np.array([0.0], float)
    d = np.linalg.norm(np.diff(p, axis=0) * np.asarray(spacing, float), axis=1)
    return np.r_[0.0, np.cumsum(d)]


def _resample_path(path, spacing, step_mm=0.20):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return p.copy()
    s = _arc_mm(p, spacing)
    x = np.arange(0.0, float(s[-1]) + 1e-9, float(step_mm))
    if len(x) == 0 or x[-1] < s[-1] - 0.03:
        x = np.r_[x, s[-1]]
    return np.column_stack([np.interp(x, s, p[:, j]) for j in range(3)])


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def _plane_memmap(ct, point_zyx, tangent_mm, spacing, half_mm=4.0, pix_mm=0.18):
    t = _unit(tangent_mm)
    u, v = _orth_basis(t)
    grid = np.arange(-half_mm, half_mm + 1e-9, pix_mm)
    vv, uu = np.meshgrid(grid, grid, indexing="ij")
    center_mm = np.asarray(point_zyx, float) * np.asarray(spacing, float)
    zyx_mm = center_mm[None, None, :] + uu[..., None] * u + vv[..., None] * v
    zyx = zyx_mm / np.asarray(spacing, float)
    im = ndi.map_coordinates(
        ct,
        [zyx[..., 0], zyx[..., 1], zyx[..., 2]],
        output=np.float32,
        order=1,
        mode="nearest",
    )
    return im, grid, u, v


def _component_metrics(im, grid, comp):
    yy, xx = np.nonzero(comp)
    pix = float(abs(grid[1] - grid[0]))
    area = float(comp.sum()) * pix * pix
    radius = math.sqrt(area / math.pi)
    cu = float(np.mean(grid[xx])); cv = float(np.mean(grid[yy]))
    shift = float(math.hypot(cu, cv))
    er = ndi.binary_erosion(comp)
    perimeter = max(float(np.logical_and(comp, ~er).sum()) * pix, pix)
    circularity = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0, 1))
    return {
        "radius_mm": radius,
        "centroid_shift_mm": shift,
        "circularity": circularity,
        "component_median_hu": float(np.median(im[comp])),
        "offset_u_mm": cu,
        "offset_v_mm": cv,
    }


def _nearest_component(im, grid, threshold_hu, max_shift_mm=1.15):
    bright = (im >= float(threshold_hu)) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    cands = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        if int(comp.sum()) < 7:
            continue
        m = _component_metrics(im, grid, comp)
        yy, xx = np.nonzero(comp)
        dmin = float(np.min(np.hypot(grid[xx], grid[yy])))
        if m["centroid_shift_mm"] > max_shift_mm and dmin > 0.45:
            continue
        cands.append((m["centroid_shift_mm"] + 0.25*dmin, m))
    return None if not cands else min(cands, key=lambda x: x[0])[1]


def _read_xyz(path):
    df = pd.read_csv(path)
    variants = [
        ("z","y","x"),
        ("source_z","source_y","source_x"),
        ("voxel_z","voxel_y","voxel_x"),
        ("center_z","center_y","center_x"),
    ]
    for cols in variants:
        if all(c in df.columns for c in cols):
            arr = df[list(cols)].to_numpy(float)
            ok = np.all(np.isfinite(arr), axis=1)
            return arr[ok], df.loc[ok].reset_index(drop=True)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) >= 3:
        arr = df[num[:3]].to_numpy(float)
        ok = np.all(np.isfinite(arr), axis=1)
        return arr[ok], df.loc[ok].reset_index(drop=True)
    raise ValueError(f"No usable z,y,x columns in {path}")


def _first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def _tangent(path, spacing, i, radius=3):
    i0 = max(0, i-radius)
    i1 = min(len(path)-1, i+radius)
    if i1 == i0:
        return np.array([0.0,1.0,0.0])
    return _unit((path[i1]-path[i0]) * np.asarray(spacing,float))


def _figsave(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def synthetic_master_qc_self_test():
    spacing = np.array([0.5,0.5,0.5])
    p = np.array([[10.,10.,10.],[10.,12.,10.],[10.,14.,10.]])
    s = _arc_mm(p, spacing)
    rp = _resample_path(p, spacing, 0.25)
    sec = np.array([0.0, 12.2, 12.4, 13.3, 13.7])
    cls = [
        "accepted_compact" if a <= SECONDARY_CONTINUOUS_SUPPORT_MM + 1e-6
        else ("local_island_unaccepted" if SECONDARY_ISLAND_START_MM <= a <= SECONDARY_ISLAND_END_MM
              else "ambiguous_geometric")
        for a in sec
    ]
    return {
        "passed": bool(abs(s[-1]-2.0)<1e-6 and len(rp) >= 8 and cls[0]=="accepted_compact"
                       and cls[2]=="ambiguous_geometric" and cls[3]=="local_island_unaccepted"),
        "arc_length_mm": float(s[-1]),
        "resampled_points": int(len(rp)),
        "support_classes": cls,
    }


class MasterCoronaryAnatomyQCWorkflow:
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Master_Coronary_Anatomy_QC_v1"
        self.out = self.root / "Master_Coronary_Anatomy_QC_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {"inputs":True, "qc":True, "figures":True, "report":True}
        if reuse:
            self.reuse.update({k:bool(v) for k,v in reuse.items()})
        self.ct = None
        self.spacing = None
        self.paths = {}
        self.source_files = {}
        self.master_centerlines = None
        self.artery_summary = None
        self.qc = None
        self.history = None
        self.summary = None
        self.threshold_hu = None

    def cache_status(self):
        names = {
            "inputs":"frozen_anatomy.json",
            "qc":"selected_qc.csv",
            "figures":"figures.done",
            "report":"report.done",
        }
        return pd.DataFrame([
            {"component":k, "reuse":self.reuse[k], "cache_exists":(self.cache/v).exists()}
            for k,v in names.items()
        ])

    def _resolve_inputs(self):
        r = self.root
        cands = {
            "RCA":[r/"Source_Volume_Coronary_Centerlines"/"RCA_source_centerline.csv"],
            "LAD":[
                r/"LAD_Takeoff_Root_Alternatives_Report"/"alternative_01_centerline.csv",
                r/"Source_Volume_Coronary_Centerlines"/"LAD_source_centerline.csv",
            ],
            "LM":[r/"LAD_Takeoff_Confirmation_Report"/"trunk_centerline.csv"],
            "SECONDARY":[r/"Secondary_Branch_Lateral_Divergence_Report"/"branch_centerline.csv"],
        }
        resolved = {k:_first_existing(v) for k,v in cands.items()}
        missing = [k for k,p in resolved.items() if p is None]
        if missing:
            raise FileNotFoundError(
                "Missing frozen centerline source(s): " + ", ".join(missing) +
                ". Expected prior accepted OpenPlaque result folders in MyDrive/OpenPlaque."
            )
        ct = r/"Cache"/"Secondary_3D_Vesselness_Topology_v1"/"series7_int16.npy"
        meta = r/"Cache"/"Secondary_3D_Vesselness_Topology_v1"/"series7_int16.json"
        if not ct.exists() or not meta.exists():
            raise FileNotFoundError(f"Missing source CCTA cache: {ct} / {meta}")
        return resolved, ct, meta

    def load_frozen_anatomy(self):
        resolved, ct_path, meta_path = self._resolve_inputs()
        meta = json.loads(meta_path.read_text())
        self.spacing = np.asarray(meta["spacing_zyx"], float)
        self.ct = np.load(ct_path, mmap_mode="r")
        self.source_files = {k:str(v) for k,v in resolved.items()}
        for k,p in resolved.items():
            arr,_ = _read_xyz(p)
            self.paths[k] = arr

        sec = _resample_path(self.paths["SECONDARY"], self.spacing, 0.20)
        ssec = _arc_mm(sec, self.spacing)
        cal = sec[(ssec>=9.5)&(ssec<=12.0)]
        if len(cal) < 3:
            cal = sec
        hu = ndi.map_coordinates(self.ct, [cal[:,0],cal[:,1],cal[:,2]],
                                 output=np.float32, order=1, mode="nearest")
        ref_hu = float(np.median(hu))
        self.threshold_hu = float(max(170.0, min(300.0, 0.38*ref_hu)))

        all_rows=[]
        summary_rows=[]
        frozen = {
            "algorithm":ALGORITHM_VERSION,
            "source_ct":str(ct_path),
            "spacing_zyx_mm":self.spacing.tolist(),
            "bright_threshold_hu":self.threshold_hu,
            "arteries":{},
            "secondary_frozen_decision":{
                "continuous_compact_support_mm":SECONDARY_CONTINUOUS_SUPPORT_MM,
                "transition_onset_mm":SECONDARY_TRANSITION_ONSET_MM,
                "compact_island_mm":[SECONDARY_ISLAND_START_MM,SECONDARY_ISLAND_END_MM],
                "compact_island_status":"short_local_appearance_not_accepted",
                "identity_label":"secondary_coronary_like_branch",
                "lcx_identity_assigned":False,
            },
        }

        for vessel in ["RCA","LM","LAD","SECONDARY"]:
            p = _resample_path(self.paths[vessel], self.spacing, 0.20)
            s = _arc_mm(p, self.spacing)
            if vessel=="SECONDARY":
                support=[]
                for a in s:
                    if a <= SECONDARY_CONTINUOUS_SUPPORT_MM + 1e-6:
                        support.append("accepted_compact")
                    elif SECONDARY_ISLAND_START_MM <= a <= SECONDARY_ISLAND_END_MM:
                        support.append("local_island_unaccepted")
                    else:
                        support.append("ambiguous_geometric")
                accepted_len = min(float(s[-1]), SECONDARY_CONTINUOUS_SUPPORT_MM)
                confidence = "high through 12.2 mm; unresolved beyond"
                status = "accepted proximal segment; distal geometry unaccepted"
            else:
                support=["accepted_frozen"]*len(s)
                accepted_len=float(s[-1])
                confidence="high"
                status="accepted/frozen"
            for i,(q,a,c) in enumerate(zip(p,s,support)):
                all_rows.append({
                    "vessel":vessel, "point_index":i, "arc_mm":float(a),
                    "z":float(q[0]),"y":float(q[1]),"x":float(q[2]),
                    "support_class":c, "source_file":str(resolved[vessel]),
                })
            summary_rows.append({
                "vessel":vessel,
                "geometric_length_mm":float(s[-1]),
                "accepted_lumen_supported_length_mm":accepted_len,
                "status":status,
                "confidence":confidence,
                "source_file":str(resolved[vessel]),
            })
            frozen["arteries"][vessel] = {
                "source_file":str(resolved[vessel]),
                "geometric_length_mm":float(s[-1]),
                "accepted_lumen_supported_length_mm":accepted_len,
                "status":status,
                "confidence":confidence,
            }

        self.master_centerlines = pd.DataFrame(all_rows)
        self.artery_summary = pd.DataFrame(summary_rows)
        self.master_centerlines.to_csv(self.cache/"master_centerlines.csv", index=False)
        self.artery_summary.to_csv(self.cache/"artery_summary.csv", index=False)
        _write_json(frozen, self.cache/"frozen_anatomy.json")
        return frozen

    def _measure_plane(self, vessel, point, tangent, arc):
        im,g,_,_ = _plane_memmap(self.ct, point, tangent, self.spacing, 4.0, 0.18)
        m = _nearest_component(im,g,self.threshold_hu,1.15)
        if m is None:
            return {
                "vessel":vessel,"arc_mm":float(arc),"component_found":False,
                "radius_mm":np.nan,"centroid_shift_mm":np.nan,"circularity":np.nan,
                "component_median_hu":np.nan,"plane_score":0.0,
            }
        rref = 1.9 if vessel=="SECONDARY" else 2.1
        rscore = math.exp(-0.5*((m["radius_mm"]-rref)/1.15)**2)
        sscore = math.exp(-0.5*(m["centroid_shift_mm"]/0.70)**2)
        cscore = float(np.clip(m["circularity"]/0.65,0,1))
        score = float(0.40*rscore+0.36*sscore+0.24*cscore)
        return {
            "vessel":vessel,"arc_mm":float(arc),"component_found":True,
            "radius_mm":float(m["radius_mm"]),
            "centroid_shift_mm":float(m["centroid_shift_mm"]),
            "circularity":float(m["circularity"]),
            "component_median_hu":float(m["component_median_hu"]),
            "plane_score":score,
        }

    def build_qc(self, candidate_step_mm=1.0, max_selected_per_vessel=12):
        if self.master_centerlines is None:
            self.load_frozen_anatomy()
        rows=[]
        for vessel in ["RCA","LM","LAD","SECONDARY"]:
            sub=self.master_centerlines[self.master_centerlines.vessel==vessel].reset_index(drop=True)
            if vessel=="SECONDARY":
                target_arcs=list(np.arange(0.0, min(SECONDARY_CONTINUOUS_SUPPORT_MM,float(sub.arc_mm.max()))+1e-9,
                                              candidate_step_mm))
                target_arcs += [12.2,12.4,13.0,13.3,13.5,13.7,float(sub.arc_mm.max())]
            else:
                target_arcs=list(np.arange(0.0,float(sub.arc_mm.max())+1e-9,candidate_step_mm))
                target_arcs += [float(sub.arc_mm.max())]
            idxs=sorted(set(int(np.argmin(np.abs(sub.arc_mm.to_numpy()-a))) for a in target_arcs))
            path=sub[["z","y","x"]].to_numpy(float)
            for i in idxs:
                tangent=_tangent(path,self.spacing,i,3)
                rec=self._measure_plane(vessel,path[i],tangent,float(sub.arc_mm.iloc[i]))
                rec["point_index"]=int(i)
                rec["support_class"]=str(sub.support_class.iloc[i])
                rows.append(rec)

        cand=pd.DataFrame(rows)
        selected=[]
        reasons={}
        for vessel in ["RCA","LM","LAD","SECONDARY"]:
            d=cand[cand.vessel==vessel].reset_index(drop=True)
            if len(d)==0: continue
            pick=set()
            def add_idx(idx, reason):
                if idx is None: return
                idx=int(idx); pick.add(idx); reasons[(vessel,int(d.loc[idx,"point_index"]))]=reason
            add_idx(0,"proximal")
            add_idx(len(d)-1,"distal/end")
            if d.component_found.any():
                good=d[d.component_found]
                add_idx(good.plane_score.idxmin(),"lowest QC score")
                add_idx(good.centroid_shift_mm.idxmax(),"largest recenter shift")
                add_idx(good.radius_mm.idxmin(),"smallest apparent radius")
                add_idx(good.radius_mm.idxmax(),"largest apparent radius")
            for frac,lab in [(0.25,"25% arc"),(0.5,"50% arc"),(0.75,"75% arc")]:
                target=frac*float(d.arc_mm.max())
                add_idx(int(np.argmin(np.abs(d.arc_mm.to_numpy()-target))),lab)
            if vessel=="SECONDARY":
                for a,lab in [(12.2,"last robust support"),(12.4,"transition onset"),
                              (13.3,"local island"),(13.5,"local island"),
                              (13.7,"post-island broadening")]:
                    if a <= float(d.arc_mm.max())+0.2:
                        add_idx(int(np.argmin(np.abs(d.arc_mm.to_numpy()-a))),lab)
            for idx in d.sort_values("plane_score").index:
                if len(pick)>=max_selected_per_vessel: break
                add_idx(idx,"informative low-score")
            for idx in sorted(pick, key=lambda j: float(d.loc[j,"arc_mm"]))[:max_selected_per_vessel]:
                rec=d.loc[idx].to_dict()
                rec["selection_reason"]=reasons.get((vessel,int(rec["point_index"])),"representative")
                selected.append(rec)

        self.qc=pd.DataFrame(selected)
        self.qc.to_csv(self.cache/"selected_qc.csv", index=False)
        cand.to_csv(self.cache/"candidate_qc.csv", index=False)
        _write_json({"n_candidates":int(len(cand)),"n_selected":int(len(self.qc))},
                    self.cache/"qc_summary.json")
        return self.qc

    def build_history(self):
        entries=[
            ("RCA source-volume centerline / PCAT validation","accepted","RCA anatomy and PCAT already validated; frozen."),
            ("LAD source-volume tracking","accepted","~57 mm source-volume LAD backbone with strong orthogonal-plane quality."),
            ("Left-main takeoff confirmation","accepted","~8.8 mm proximal trunk from left coronary origin into bifurcation."),
            ("Secondary lateral divergence","accepted proximal","Independent coronary-like branch from LM neighborhood; no LCX identity assignment."),
            ("Secondary target-free continuation","negative","No evaluable forward compact-lumen continuation."),
            ("Secondary rejection diagnostics","negative","Nearby structure became broad/off-center rather than coronary-sized."),
            ("Secondary 3-D vesselness topology","mechanical-only","Tubular corridor found but source-CCTA lumen validation rejected coronary identity."),
            ("Secondary anchored lumen validation","negative","Candidate corridor radius ~4.4 mm; compact-lumen gate failed."),
            ("Secondary endpoint fan-out","negative","No compact-lumen survivor; first-step radii already broad."),
            ("Secondary distal reacquisition","negative","No distal compact lumen reacquired with vesselness proposals."),
            ("Secondary dense source reacquisition","negative","No distal compact lumen reacquired without vesselness proposal dependence."),
            ("Secondary terminal transition","characterized","Continuous compact support ends near 12.2 mm; transition begins ~12.3-12.4 mm."),
            ("Secondary compact-island bidirectional validation","negative/local","13.2-13.6 mm island is only a short local compact appearance (~0.62 mm total)."),
        ]
        df=pd.DataFrame(entries,columns=["experiment","outcome","master_interpretation"])
        self.history=df
        df.to_csv(self.cache/"evidence_history.csv",index=False)
        return df

    def make_figures(self):
        if self.master_centerlines is None: self.load_frozen_anatomy()
        if self.qc is None: self.build_qc()
        if self.history is None: self.build_history()
        names=[]

        fig=plt.figure(figsize=(10,8))
        ax=fig.add_subplot(111,projection="3d")
        for vessel in ["RCA","LM","LAD","SECONDARY"]:
            d=self.master_centerlines[self.master_centerlines.vessel==vessel]
            if vessel!="SECONDARY":
                ax.plot(d.x,d.y,d.z,label=vessel,linewidth=2.5)
            else:
                for cls,g in d.groupby("support_class",sort=False):
                    style="-" if cls=="accepted_compact" else "--"
                    ax.plot(g.x,g.y,g.z,style,label=f"SECONDARY: {cls}",linewidth=2.5 if cls=="accepted_compact" else 1.8)
        ax.set_xlabel("x voxel"); ax.set_ylabel("y voxel"); ax.set_zlabel("z voxel")
        ax.set_title("Frozen coronary anatomy — 3-D geometry")
        ax.legend(fontsize=8)
        p=self.out/"01_frozen_coronary_tree_3d.png"; _figsave(fig,p); names.append(p.name)

        fig,axs=plt.subplots(1,3,figsize=(16,5))
        projections=[("x","y","XY"),("x","z","XZ"),("y","z","YZ")]
        for ax,(a,b,title) in zip(axs,projections):
            for vessel in ["RCA","LM","LAD","SECONDARY"]:
                d=self.master_centerlines[self.master_centerlines.vessel==vessel]
                if vessel!="SECONDARY":
                    ax.plot(d[a],d[b],label=vessel,linewidth=2)
                else:
                    for cls,g in d.groupby("support_class",sort=False):
                        ax.plot(g[a],g[b],"-" if cls=="accepted_compact" else "--",
                                label=f"SECONDARY {cls}" if title=="XY" else None,linewidth=2)
            ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(title); ax.invert_yaxis()
        axs[0].legend(fontsize=7)
        fig.suptitle("Frozen coronary centerlines — orthogonal projections")
        p=self.out/"02_frozen_coronary_tree_projections.png"; _figsave(fig,p); names.append(p.name)

        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,(a,b,title) in zip(axs,projections):
            for vessel in ["LM","LAD","SECONDARY"]:
                d=self.master_centerlines[self.master_centerlines.vessel==vessel]
                if vessel=="LAD":
                    d=d[d.arc_mm<=18.0]
                if vessel=="SECONDARY":
                    for cls,g in d.groupby("support_class",sort=False):
                        ax.plot(g[a],g[b],"-" if cls=="accepted_compact" else "--",
                                label=f"{vessel}: {cls}" if title=="XY" else None,linewidth=2.4)
                else:
                    ax.plot(d[a],d[b],label=vessel if title=="XY" else None,linewidth=2.4)
            ax.set_title(title); ax.set_xlabel(a); ax.set_ylabel(b); ax.invert_yaxis()
        axs[0].legend(fontsize=7)
        fig.suptitle("Left-main bifurcation and secondary-branch confidence")
        p=self.out/"03_left_main_bifurcation_zoom.png"; _figsave(fig,p); names.append(p.name)

        trans_path=self.root/"Cache"/"Secondary_Terminal_Transition_v1"/"transition_profile.csv"
        if not trans_path.exists():
            trans_path=self.root/"Secondary_Terminal_Transition_Report"/"transition_profile.csv"
        fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
        if trans_path.exists():
            td=pd.read_csv(trans_path)
            axs[0].plot(td.arc_mm,td.radius_mm,marker="o",markersize=3)
            axs[1].plot(td.arc_mm,td.centroid_shift_mm,marker="o",markersize=3)
        else:
            d=pd.read_csv(self.cache/"candidate_qc.csv")
            d=d[d.vessel=="SECONDARY"].sort_values("arc_mm")
            axs[0].plot(d.arc_mm,d.radius_mm,marker="o",markersize=3)
            axs[1].plot(d.arc_mm,d.centroid_shift_mm,marker="o",markersize=3)
        for ax in axs:
            ax.axvline(12.2,linestyle="--"); ax.axvline(12.4,linestyle=":")
            ax.axvspan(13.2,13.6,alpha=0.12)
            ax.grid(alpha=0.25)
        axs[0].axhline(2.65,linestyle="--")
        axs[0].set_ylabel("Apparent radius (mm)")
        axs[1].set_ylabel("Centroid shift (mm)"); axs[1].set_xlabel("Secondary branch arc (mm)")
        fig.suptitle("Secondary branch — compact support, transition, and unaccepted local island")
        p=self.out/"04_secondary_transition_summary.png"; _figsave(fig,p); names.append(p.name)

        atlas_groups=[("RCA",["RCA"]),("LAD",["LAD"]),("LM_secondary",["LM","SECONDARY"])]
        for tag,vessels in atlas_groups:
            q=self.qc[self.qc.vessel.isin(vessels)].copy()
            if len(q)==0: continue
            n=min(len(q),18)
            q=q.iloc[:n]
            cols=3; rows=int(math.ceil(n/cols))
            fig,axs=plt.subplots(rows,cols,figsize=(12,3.6*rows))
            axs=np.atleast_1d(axs).ravel()
            for ax in axs[n:]: ax.axis("off")
            for ax,(_,rec) in zip(axs,q.iterrows()):
                sub=self.master_centerlines[self.master_centerlines.vessel==rec.vessel].reset_index(drop=True)
                i=int(rec.point_index); path=sub[["z","y","x"]].to_numpy(float)
                tang=_tangent(path,self.spacing,i,3)
                im,g,_,_=_plane_memmap(self.ct,path[i],tang,self.spacing,3.8,0.16)
                ax.imshow(im,cmap="gray",vmin=0,vmax=850,extent=[g[0],g[-1],g[-1],g[0]])
                ax.axhline(0,linewidth=.5); ax.axvline(0,linewidth=.5)
                ax.set_title(f"{rec.vessel} {rec.arc_mm:.1f} mm\n"
                             f"r={rec.radius_mm:.2f}, shift={rec.centroid_shift_mm:.2f}, score={rec.plane_score:.2f}\n"
                             f"{rec.selection_reason}",fontsize=8)
                ax.set_xlabel("mm"); ax.set_ylabel("mm")
            fig.suptitle(f"Automatically selected source-CCTA QC atlas — {tag.replace('_',' + ')}")
            p=self.out/f"05_{tag}_qc_atlas.png" if tag=="RCA" else (
                self.out/f"06_{tag}_qc_atlas.png" if tag=="LAD" else self.out/f"07_{tag}_qc_atlas.png")
            _figsave(fig,p); names.append(p.name)

        fig,ax=plt.subplots(figsize=(13,8))
        ax.axis("off")
        table_data=self.history[["experiment","outcome","master_interpretation"]].values.tolist()
        tbl=ax.table(cellText=table_data,
                     colLabels=["Experiment","Outcome","Master interpretation"],
                     loc="center",cellLoc="left",colLoc="left",colWidths=[.30,.14,.56])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1,1.55)
        ax.set_title("Evidence ledger supporting frozen coronary anatomy",pad=20)
        p=self.out/"08_evidence_ledger.png"; _figsave(fig,p); names.append(p.name)

        (self.cache/"figures.done").write_text("done")
        _write_json({"figures":names},self.cache/"figure_manifest.json")
        return names

    def _html_table(self, df, cols=None):
        if cols is not None: df=df[cols]
        return df.to_html(index=False,escape=True,border=0,classes="dataframe")

    def build_report(self):
        if self.master_centerlines is None: self.load_frozen_anatomy()
        if self.qc is None: self.build_qc()
        if self.history is None: self.build_history()
        figs=json.loads((self.cache/"figure_manifest.json").read_text())["figures"] if (self.cache/"figure_manifest.json").exists() else self.make_figures()

        status = {
            "algorithm":ALGORITHM_VERSION,
            "status":"FROZEN_CORONARY_ANATOMY_QC_COMPLETE",
            "secondary_continuous_compact_support_mm":SECONDARY_CONTINUOUS_SUPPORT_MM,
            "secondary_transition_onset_mm":SECONDARY_TRANSITION_ONSET_MM,
            "secondary_island_status":"short local compact appearance; not accepted continuation",
            "secondary_lcx_identity_assigned":False,
            "n_selected_qc_planes":int(len(self.qc)),
            "figures":figs,
        }
        self.summary=status
        _write_json(status,self.cache/"summary.json")

        style="""
        <style>
        body{font-family:Arial,sans-serif;max-width:1400px;margin:24px auto;padding:0 18px;color:#222}
        h1,h2{margin-top:28px}.note{background:#f3f4f6;padding:12px;border-left:4px solid #666}
        .good{background:#eef8ee;padding:12px;border-left:4px solid #2f7d32}
        table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;vertical-align:top}
        th{background:#f5f5f5;text-align:left}img{max-width:100%;margin:10px 0 22px 0}
        code{background:#f3f3f3;padding:2px 4px}
        </style>
        """
        sec_note = """
        <div class='note'><b>Secondary branch frozen interpretation.</b>
        Continuous source-CCTA compact-lumen support is accepted through approximately 12.2 mm.
        Transition begins around 12.3–12.4 mm. The compact-looking 13.2–13.6 mm island was
        independently tracked bidirectionally and proved to be only a short local appearance,
        not an accepted distal segment. The historical ~13.8 mm geometry is retained for
        visualization only. No LCX identity is assigned.</div>
        """
        fig_html="".join(f"<h3>{html.escape(name)}</h3><img src='{html.escape(name)}'>" for name in figs)
        report=f"""<html><head><meta charset='utf-8'><title>OpenPlaque Master Coronary Anatomy + QC</title>{style}</head>
        <body>
        <h1>OpenPlaque — Master Coronary Anatomy + QC</h1>
        <div class='good'><b>Status:</b> FROZEN_CORONARY_ANATOMY_QC_COMPLETE<br>
        This report consolidates prior validated anatomy. It does not perform new vessel discovery.</div>
        <h2>Frozen artery summary</h2>
        {self._html_table(self.artery_summary,["vessel","geometric_length_mm","accepted_lumen_supported_length_mm","status","confidence"])}
        {sec_note}
        <h2>Automatically selected source-CCTA QC planes</h2>
        {self._html_table(self.qc,["vessel","arc_mm","support_class","radius_mm","centroid_shift_mm","plane_score","selection_reason"])}
        <h2>Evidence ledger</h2>
        {self._html_table(self.history)}
        <h2>Visualizations</h2>{fig_html}
        <h2>Machine-readable outputs</h2>
        <p><code>frozen_anatomy.json</code>, <code>artery_summary.csv</code>,
        <code>master_centerlines.csv</code>, <code>selected_qc.csv</code>,
        <code>evidence_history.csv</code>, and <code>summary.json</code>.</p>
        <p><b>Research use only.</b></p>
        </body></html>"""
        html_path=self.out/"OPENPLAQUE_MASTER_CORONARY_ANATOMY_QC_REPORT.html"
        html_path.write_text(report)

        for name in ["frozen_anatomy.json","artery_summary.csv","master_centerlines.csv","selected_qc.csv",
                     "candidate_qc.csv","evidence_history.csv","summary.json","qc_summary.json","figure_manifest.json"]:
            src=self.cache/name
            if src.exists():
                (self.out/name).write_bytes(src.read_bytes())
        (self.cache/"report.done").write_text("done")
        return html_path

    def package(self):
        html_path=self.build_report()
        zip_path=self.out/"OPENPLAQUE_MASTER_CORONARY_ANATOMY_QC_REPORT_BACK.zip"
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file() and p.name != zip_path.name:
                    z.write(p,arcname=p.name)
        return zip_path

    def run(self):
        self.load_frozen_anatomy()
        self.build_qc()
        self.build_history()
        names=self.make_figures()
        html_path=self.build_report()
        zip_path=self.package()
        return {
            "status":"FROZEN_CORONARY_ANATOMY_QC_COMPLETE",
            "algorithm":ALGORITHM_VERSION,
            "report":str(html_path),
            "zip":str(zip_path),
            "figures":names,
            "artery_summary":self.artery_summary.to_dict(orient="records"),
            "secondary_continuous_compact_support_mm":SECONDARY_CONTINUOUS_SUPPORT_MM,
        }
