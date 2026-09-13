from __future__ import annotations

"""Continue a validated LAD proximally with source-resolution lumen recentering.

Starts from the established LAD produced by LAD_Confirmed_Backtrack_v1. No global
LAD rediscovery and no LCX search are performed. Each proximal step is proposed
in a broadened directional cone and then recentered onto a coronary-sized,
contrast-filled component in a true orthogonal source-CCTA plane before entering
the beam. The validated RCA calibration is reused as a subject-specific positive
reference. TotalSegmentator contributes only the aorta exclusion/distance mask.

Research use only.
"""

import base64
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from .lad_confirmed_backtrack import (
    PINNED_VALIDATED_RCA,
    _json_read,
    _json_write,
    _orth_basis,
    _ram,
    _unit,
    arc_mm,
    orthogonal_plane,
    resample_path,
    source_zyx_to_lps,
    stream_source_ct_to_memmap,
)

ALGORITHM_VERSION = "lad-proximal-recenter-backtrack-v1.0"


def _component_metrics(im, c, comp, cu, cv):
    pix = float(abs(c[1] - c[0]))
    area = int(comp.sum())
    radius = math.sqrt(area * pix * pix / math.pi)
    er = ndi.binary_erosion(comp, structure=np.ones((3, 3), bool))
    per = max(1, int((comp & ~er).sum())) * pix
    circ = float(np.clip(4 * math.pi * (area * pix * pix) / max(per * per, 1e-8), 0, 1.2))
    U, V = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(U - cu, V - cv)
    center = float(np.median(im[R <= 0.70]))
    core = float(np.median(im[R <= 1.0]))
    ring = float(np.median(im[(R >= 2.5) & (R <= 4.0)]))
    return {
        "radius_mm": radius,
        "circularity": circ,
        "center_hu": center,
        "core_minus_ring_hu": core - ring,
        "centroid_u_mm": float(cu),
        "centroid_v_mm": float(cv),
        "recenter_shift_mm": float(math.hypot(cu, cv)),
    }


def find_recenter_component(im, c, rca, search_mm=2.25, rescue=False):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    lo_hu = max(170.0, min(260.0, 0.38 * rh))
    lab, nlab = ndi.label((im >= lo_hu) & (im <= 1200.0), structure=np.ones((3, 3), np.uint8))
    pix = float(abs(c[1] - c[0]))
    limit = 2.85 if rescue else float(search_mm)
    best = None
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        area = int(comp.sum())
        if area < 7:
            continue
        yy, xx = np.nonzero(comp)
        cu, cv = float(np.mean(c[xx])), float(np.mean(c[yy]))
        shift = float(math.hypot(cu, cv))
        radius = math.sqrt(area * pix * pix / math.pi)
        if shift > limit or not (0.65 <= radius <= 3.35):
            continue
        m = _component_metrics(im, c, comp, cu, cv)
        rscore = math.exp(-0.5 * ((m["radius_mm"] - 1.05 * rr) / max(0.62 * rr, 0.72)) ** 2)
        pscore = math.exp(-0.5 * (m["recenter_shift_mm"] / (1.15 if rescue else 0.95)) ** 2)
        cscore = float(np.clip(m["circularity"] / 0.50, 0, 1))
        hscore = math.exp(-0.5 * ((m["center_hu"] - rh) / 350.0) ** 2)
        xscore = 1.0 / (1.0 + math.exp(-(m["core_minus_ring_hu"] - 10.0) / 85.0))
        choose = 0.34 * rscore + 0.27 * pscore + 0.17 * cscore + 0.12 * hscore + 0.10 * xscore
        m["component_choose_score"] = float(choose)
        if best is None or choose > best[0]:
            best = (choose, m)
    return None if best is None else best[1]


def score_recentered(m, rca, rescue=False):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    r, sh, circ, hu, con = (float(m[k]) for k in ("radius_mm", "recenter_shift_mm", "circularity", "center_hu", "core_minus_ring_hu"))
    rs = math.exp(-0.5 * ((r - 1.05 * rr) / max(0.55 * rr, 0.68)) ** 2)
    ss = math.exp(-0.5 * (sh / (1.20 if rescue else 0.92)) ** 2)
    cs = float(np.clip(circ / 0.55, 0, 1))
    hs = math.exp(-0.5 * ((hu - rh) / 330.0) ** 2)
    xs = 1.0 / (1.0 + math.exp(-(con - 15.0) / 80.0))
    score = 0.34 * rs + 0.23 * ss + 0.18 * cs + 0.14 * hs + 0.11 * xs
    hard = bool(0.58 * rr <= r <= min(3.30, 2.00 * rr + 0.10) and sh <= (1.80 if rescue else 1.45) and circ >= 0.18 and 120 <= hu <= 1150 and score >= 0.60)
    soft = bool(0.48 * rr <= r <= 3.35 and sh <= (2.65 if rescue else 2.15) and circ >= 0.10 and 100 <= hu <= 1200 and score >= 0.50)
    return float(score), hard, soft


def serial_qc(path, ct, spacing, rca, step_sample_mm=1.35, label="path"):
    p = resample_path(path, spacing, 0.35)
    s = arc_mm(p, spacing)
    rows = []
    if len(p) >= 4:
        ss = np.arange(min(0.5, 0.10 * s[-1]), s[-1] + 1e-6, step_sample_mm)
        if len(ss) == 0 or ss[-1] < s[-1] - 0.5:
            ss = np.r_[ss, s[-1]]
        for x in ss:
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 4), min(len(p) - 1, i + 4)
            tangent = (p[i1] - p[i0]) * np.asarray(spacing, float)
            im, c = orthogonal_plane(ct, p[i], tangent, spacing)
            m = find_recenter_component(im, c, rca, search_mm=1.15, rescue=False)
            if m is None:
                rows.append({"label": label, "arc_mm": float(s[i]), "radius_mm": np.nan, "circularity": np.nan, "center_hu": np.nan, "core_minus_ring_hu": np.nan, "recenter_shift_mm": np.inf, "plane_score": 0.0, "plane_pass": False, "soft_pass": False})
                continue
            sc, hp, sp = score_recentered(m, rca, rescue=False)
            rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_score": sc, "plane_pass": hp, "soft_pass": sp})
    df = pd.DataFrame(rows)
    return df, {
        "label": label,
        "length_mm": float(s[-1]) if len(s) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "soft_pass_fraction": float(df.soft_pass.mean()) if len(df) else 0.0,
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else np.nan,
        "median_recenter_shift_mm": float(df.recenter_shift_mm.replace([np.inf], np.nan).median()) if len(df) else np.nan,
        "median_circularity": float(df.circularity.median()) if len(df) else np.nan,
        "median_center_hu": float(df.center_hu.median()) if len(df) else np.nan,
    }


def _cone(base, angles=(0.0, 18.0, 36.0), n_az=8):
    base = _unit(base)
    u, v = _orth_basis(base)
    out = []
    for deg in angles:
        if deg == 0:
            out.append(base)
            continue
        a = math.radians(float(deg))
        for k in range(int(n_az)):
            ph = 2 * math.pi * k / int(n_az)
            out.append(_unit(math.cos(a) * base + math.sin(a) * (math.cos(ph) * u + math.sin(ph) * v)))
    return out


class LADProximalRecenterWorkflow:
    COMPONENTS = ("source_ct", "validated_prior", "aorta_constraint", "continuation", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LAD_Proximal_Recenter_v1"
        self.out = self.root / "LAD_Proximal_Recenter_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = self.rca_cal = None
        self.established = self.established_qc = self.established_summary = None
        self.aorta_lo = self.aorta_crop = self.aorta_dist = None
        self.continuation = self.continuation_qc = self.continuation_summary = None
        self.combined = self.endpoint = self.step_log = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {"source_ct":"series7_int16.npy","validated_prior":"established_lad_summary.json","aorta_constraint":"aorta_constraint.npz","continuation":"continuation_summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        priors = [self.root/"Cache"/"LAD_Confirmed_Backtrack_v1"/"series7_int16.npy", self.root/"Cache"/"LAD_Origin_Backtrack_v1"/"series7_int16.npy"]
        if not self.reuse["source_ct"]:
            for p in (own, own.with_suffix(".json")):
                if p.exists(): p.unlink()
            priors = []
        self.ct, self.meta, action = stream_source_ct_to_memmap(self.root, own, reuse_sources=priors)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", action, own if action in ("reused_own_cache","recomputed_and_cached") else action.split(":",1)[-1])
        _ram("After source CT")
        return self.ct

    def validate_established_lad(self):
        pf,qf,sf,rf = (self.cache/n for n in ("established_lad_centerline.csv","established_lad_qc.csv","established_lad_summary.json","validated_rca_calibration.json"))
        if self.ct is None: self.load_source_ct()
        if self.reuse["validated_prior"] and all(p.exists() for p in (pf,qf,sf,rf)):
            self.established = pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.established_qc = pd.read_csv(qf)
            self.established_summary = _json_read(sf)
            self.rca_cal = _json_read(rf)
            self._record("validated_prior","reused",sf)
            return self.established_summary
        prior = self.root/"Cache"/"LAD_Confirmed_Backtrack_v1"
        ppath, prca = prior/"combined_lad_centerline.csv", prior/"validated_rca_calibration.json"
        if not ppath.exists(): raise FileNotFoundError(f"Required established LAD path not found: {ppath}")
        x = _json_read(prca) if prca.exists() else dict(PINNED_VALIDATED_RCA)
        self.rca_cal = x if 1.0 <= float(x.get("median_radius_mm",np.nan)) <= 2.3 and float(x.get("plane_pass_fraction",0)) >= 0.75 else dict(PINNED_VALIDATED_RCA)
        _json_write(self.rca_cal,rf)
        d = pd.read_csv(ppath)
        p = resample_path(d[["z","y","x"]].to_numpy(float),self.spacing,0.35)
        qdf,qsum = serial_qc(p,self.ct,self.spacing,self.rca_cal,1.35,"ESTABLISHED_LAD")
        qsum.update({"accepted":bool(qsum["length_mm"]>=20 and qsum["plane_pass_fraction"]>=0.78 and qsum["median_plane_score"]>=0.80),"source_prior_path":str(ppath)})
        self.established,self.established_qc,self.established_summary=p,qdf,qsum
        pd.DataFrame({"arc_mm":arc_mm(p,self.spacing),"z":p[:,0],"y":p[:,1],"x":p[:,2]}).to_csv(pf,index=False)
        qdf.to_csv(qf,index=False)
        _json_write(qsum,sf)
        self._record("validated_prior","recomputed_and_cached",sf,f"accepted={qsum['accepted']}, pass={qsum['plane_pass_fraction']:.3f}")
        return qsum

    def load_aorta_constraint(self):
        fp = self.cache/"aorta_constraint.npz"
        if self.established is None: self.validate_established_lad()
        source = fp if fp.exists() else self.root/"Cache"/"LAD_Confirmed_Backtrack_v1"/"aorta_constraint.npz"
        if not source.exists(): raise FileNotFoundError("Prior TotalSegmentator aorta constraint cache not found")
        z=np.load(source,allow_pickle=False)
        self.aorta_lo=z["lo"].astype(int)
        self.aorta_crop=z["aorta"].astype(bool)
        self.aorta_dist=z["dist"].astype(np.float32)
        if source != fp:
            np.savez_compressed(fp,lo=self.aorta_lo.astype(np.int32),aorta=self.aorta_crop.astype(np.uint8),dist=self.aorta_dist)
        self._record("aorta_constraint","reused" if source==fp else "imported_prior_constraint",fp)
        return self.aorta_dist

    def _aorta_distance(self,p):
        q=np.asarray(p,float)-self.aorta_lo
        if np.any(q<0) or np.any(q>=np.asarray(self.aorta_dist.shape)-1): return float("inf"),False
        d=float(map_coordinates(self.aorta_dist,q[:,None],order=1,mode="nearest",prefilter=False)[0])
        qi=np.rint(q).astype(int)
        return d,bool(self.aorta_crop[tuple(qi)])

    def _propose(self,cur,direction,rescue=False):
        guess=cur+(0.80*_unit(direction))/self.spacing
        if np.any(guess<2) or np.any(guess>=np.asarray(self.ct.shape,float)-3): return None
        im,c=orthogonal_plane(self.ct,guess,direction,self.spacing)
        u,v=_orth_basis(direction)
        m=find_recenter_component(im,c,self.rca_cal,2.25,rescue)
        if m is None: return None
        p=(guess*self.spacing+m["centroid_u_mm"]*u+m["centroid_v_mm"]*v)/self.spacing
        vec=(p-cur)*self.spacing
        L=float(np.linalg.norm(vec))
        if not (0.55<=L<=(2.45 if rescue else 1.90)): return None
        ad,inside=self._aorta_distance(p)
        if inside or ad<0.65: return None
        sc,hp,sp=score_recentered(m,self.rca_cal,rescue)
        if not sp: return None
        return {"point":p,"direction":_unit(vec),"plane_score":sc,"hard":hp,"soft":sp,"aorta_distance_mm":ad,"step_length_mm":L,**m}

    def continue_proximally(self,max_extension_mm=42.0,beam_width=9):
        pf,qf,sf,cf,ef,lf=(self.cache/n for n in ("continuation_centerline.csv","continuation_qc.csv","continuation_summary.json","combined_lad_centerline.csv","proximal_endpoint.json","continuation_step_log.csv"))
        if self.established is None: self.validate_established_lad()
        if self.aorta_dist is None: self.load_aorta_constraint()
        if self.reuse["continuation"] and all(p.exists() for p in (pf,qf,sf,cf,ef,lf)):
            self.continuation=pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.continuation_qc=pd.read_csv(qf)
            self.continuation_summary=_json_read(sf)
            self.combined=pd.read_csv(cf)[["z","y","x"]].to_numpy(float)
            self.endpoint=_json_read(ef)
            self.step_log=pd.read_csv(lf)
            self._record("continuation","reused",sf)
            return self.continuation_summary
        if not self.established_summary.get("accepted",False): raise RuntimeError("Established LAD failed corrected validation gate")
        p=resample_path(self.established,self.spacing,0.35)
        anchor=p[0].copy()
        j=min(len(p)-1,max(4,int(round(3.5/0.35))))
        init_dir=-_unit((p[j]-p[0])*self.spacing)
        anchor_aorta,_=self._aorta_distance(anchor)
        beams=[{"point":anchor,"direction":init_dir,"path":[anchor.copy()],"scores":[],"hard":[],"soft":[],"shifts":[],"aorta":[anchor_aorta],"obj":0.0}]
        completed=[]
        logs=[]
        stop="max_extension"
        for step in range(int(math.ceil(max_extension_mm/0.80))):
            props=[]
            for bi,st in enumerate(beams):
                local=[]
                for d in _cone(st["direction"],(0,18,36),8):
                    r=self._propose(st["point"],d,False)
                    if r is not None: local.append(r)
                if len(local)<2:
                    for d in _cone(st["direction"],(48,60),8):
                        r=self._propose(st["point"],d,True)
                        if r is not None: local.append(r)
                for r in local:
                    smooth=float(np.clip(np.dot(_unit(st["direction"]),_unit(r["direction"])),-1,1))
                    prev=st["aorta"][-1]
                    aprog=float(np.clip((prev-r["aorta_distance_mm"])/1.5,-1,1)) if np.isfinite(prev) and np.isfinite(r["aorta_distance_mm"]) else 0.0
                    obj=0.61*r["plane_score"]+0.13*float(r["hard"])+0.12*((smooth+1)/2)+0.09*math.exp(-0.5*(r["recenter_shift_mm"]/1.0)**2)+0.05*((aprog+1)/2)
                    ns={"point":r["point"],"direction":r["direction"],"path":st["path"]+[r["point"].copy()],"scores":st["scores"]+[r["plane_score"]],"hard":st["hard"]+[r["hard"]],"soft":st["soft"]+[r["soft"]],"shifts":st["shifts"]+[r["recenter_shift_mm"]],"aorta":st["aorta"]+[r["aorta_distance_mm"]],"obj":st["obj"]+obj}
                    props.append(ns)
                    logs.append({"step":step+1,"parent_beam":bi,"plane_score":r["plane_score"],"hard_pass":r["hard"],"recenter_shift_mm":r["recenter_shift_mm"],"radius_mm":r["radius_mm"],"circularity":r["circularity"],"center_hu":r["center_hu"],"aorta_distance_mm":r["aorta_distance_mm"],"step_length_mm":r["step_length_mm"]})
            if not props:
                completed.extend(beams)
                stop="no_plausible_recentered_proposals"
                break
            def rank(st):
                n=len(st["scores"])
                L=arc_mm(np.asarray(st["path"]),self.spacing)[-1]
                return 0.50*np.mean(st["scores"])+0.20*np.mean(st["hard"])+0.18*min(L/24,1)+0.12*math.exp(-0.5*(np.median(st["shifts"])/1.0)**2)
            props.sort(key=rank,reverse=True)
            keep=[]
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=0.55 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            completed.extend(beams)
            beams=keep
            near=[st for st in beams if np.isfinite(st["aorta"][-1]) and st["aorta"][-1]<=4.0 and np.mean(st["hard"][-4:])>=0.75]
            if near:
                completed.extend(near)
                stop="near_aortic_root_constraint"
                break
        completed.extend(beams)
        viable=[]
        for st in completed:
            if len(st["path"])<4: continue
            L=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
            hf=float(np.mean(st["hard"]))
            sfv=float(np.mean(st["soft"]))
            ms=float(np.mean(st["scores"]))
            sh=float(np.median(st["shifts"]))
            gain=float(anchor_aorta-st["aorta"][-1]) if np.isfinite(anchor_aorta) and np.isfinite(st["aorta"][-1]) else 0.0
            fr=0.42*min(L/32,1)+0.27*hf+0.22*ms+0.06*sfv+0.03*np.clip(gain/20,-1,1)-0.04*min(sh/2,1)
            viable.append((fr,L,hf,sfv,ms,sh,gain,st))
        if not viable: raise RuntimeError("No proximal continuation survived the recentered lumen gate")
        fr,L,hf,sfv,ms,sh,gain,best=sorted(viable,key=lambda x:x[0],reverse=True)[0]
        cont=resample_path(np.asarray(best["path"],float)[::-1],self.spacing,0.35)
        qdf,qsum=serial_qc(cont,self.ct,self.spacing,self.rca_cal,1.15,"RECENTER_CONTINUATION")
        est=resample_path(self.established,self.spacing,0.35)
        combined=resample_path(np.vstack([cont[:-1],est]),self.spacing,0.35)
        ep=cont[0]
        eplps=source_zyx_to_lps(self.meta,ep[None,:])[0]
        epa,_=self._aorta_distance(ep)
        good=qsum["plane_pass_fraction"]>=0.72 and qsum["median_plane_score"]>=0.78
        status="PASS" if L>=12 and good else ("REVIEW" if L>=5 and qsum["soft_pass_fraction"]>=0.75 else "PARTIAL")
        interp="validated continuation reached the aortic-root constraint; inspect for left-main transition/bifurcation" if epa<=4 and good else ("validated proximal LAD continuation; endpoint is not yet called the LAD origin" if L>=12 and good else "short but locally vessel-like continuation; endpoint remains unvalidated as an LAD origin")
        self.continuation,self.continuation_qc,self.combined,self.step_log=cont,qdf,combined,pd.DataFrame(logs)
        self.endpoint={"status":"PROXIMAL_ENDPOINT_ONLY","source_zyx":[float(x) for x in ep],"lps_mm":[float(x) for x in eplps],"distance_to_TotalSegmentator_aorta_mm":float(epa),"note":interp}
        self.continuation_summary={"algorithm":ALGORITHM_VERSION,"status":status,"stop_reason":stop,"continuation_length_mm":float(L),"beam_rank_score":float(fr),"beam_hard_pass_fraction":hf,"beam_soft_pass_fraction":sfv,"beam_mean_plane_score":ms,"beam_median_recenter_shift_mm":sh,"aorta_distance_reduction_mm":gain,"endpoint_aorta_distance_mm":float(epa),**{f"serial_{k}":v for k,v in qsum.items() if k!="label"},"established_length_mm":float(self.established_summary["length_mm"]),"established_pass_fraction":float(self.established_summary["plane_pass_fraction"]),"combined_length_mm":float(arc_mm(combined,self.spacing)[-1]),"interpretation":interp}
        pd.DataFrame({"arc_mm":arc_mm(cont,self.spacing),"z":cont[:,0],"y":cont[:,1],"x":cont[:,2]}).to_csv(pf,index=False)
        qdf.to_csv(qf,index=False)
        pd.DataFrame({"arc_mm":arc_mm(combined,self.spacing),"z":combined[:,0],"y":combined[:,1],"x":combined[:,2]}).to_csv(cf,index=False)
        self.step_log.to_csv(lf,index=False)
        _json_write(self.continuation_summary,sf)
        _json_write(self.endpoint,ef)
        self._record("continuation","recomputed_and_cached",sf,f"status={status}, length={L:.1f} mm, stop={stop}")
        _ram("After proximal continuation")
        return self.continuation_summary

    def _sections(self,path,axs,label,n=8):
        p=resample_path(path,self.spacing,0.35)
        s=arc_mm(p,self.spacing)
        if len(p)<4:
            for ax in axs: ax.axis("off")
            return
        for ax,x in zip(axs,np.linspace(0.4,max(0.4,s[-1]-0.3),n)):
            i=int(np.argmin(abs(s-x)))
            i0,i1=max(0,i-4),min(len(p)-1,i+4)
            im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing)
            ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower")
            ax.plot(0,0,"+",ms=8)
            ax.set_aspect("equal")
            ax.set_title(f"{label}\n{x:.1f} mm")

    def plot_qc(self):
        done=self.cache/"figures.done"
        figs=[self.out/f for f in ("01_established_lad_revalidation.png","02_recenter_continuation_cross_sections.png","03_combined_lad_source_mips.png","04_recenter_profiles.png","05_proximal_endpoint_neighborhood.png","06_combined_lad_lps_course.png")]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs):
            self._record("figures","reused",done)
            return figs
        if self.continuation_summary is None: self.continue_proximally()
        fig,axs=plt.subplots(2,4,figsize=(11,5.5))
        self._sections(self.established,axs.ravel(),"Established LAD")
        fig.suptitle(f"Established LAD revalidation — pass {self.established_summary['plane_pass_fraction']:.2f}")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[0],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(2,4,figsize=(11,5.5))
        self._sections(self.continuation,axs.ravel(),"Recenter continuation")
        fig.suptitle(f"Proximal recentered LAD continuation — {self.continuation_summary['status']} ({self.continuation_summary['continuation_length_mm']:.1f} mm)")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[1],dpi=180,bbox_inches="tight"); plt.close(fig)
        pts=self.combined
        pad=np.ceil(10/self.spacing).astype(int)
        lo=np.maximum(0,np.floor(pts.min(0)).astype(int)-pad)
        hi=np.minimum(np.asarray(self.ct.shape),np.ceil(pts.max(0)).astype(int)+pad+1)
        sl=tuple(slice(int(lo[d]),int(hi[d])) for d in range(3))
        crop=np.asarray(self.ct[sl],dtype=np.int16)
        q=pts-lo
        fig,axs=plt.subplots(1,3,figsize=(16,5))
        ims=[crop.max(0),crop.max(1),crop.max(2)]
        for ax,im,title in zip(axs,ims,("Axial MIP","Coronal MIP","Sagittal MIP")):
            ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower")
            ax.set_title(title)
            ax.axis("off")
        axs[0].plot(q[:,2],q[:,1],lw=2)
        axs[1].plot(q[:,2],q[:,0],lw=2)
        axs[2].plot(q[:,1],q[:,0],lw=2)
        fig.suptitle("Combined source-resolution LAD: new proximal continuation + established segment")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[2],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(4,1,figsize=(9,10),sharex=True)
        d=self.continuation_qc
        if len(d):
            axs[0].plot(d.arc_mm,d.radius_mm,marker="o"); axs[0].axhline(self.rca_cal["median_radius_mm"],ls="--"); axs[0].set_ylabel("radius mm")
            axs[1].plot(d.arc_mm,d.recenter_shift_mm,marker="o"); axs[1].axhline(1.45,ls="--"); axs[1].set_ylabel("QC recenter mm")
            axs[2].plot(d.arc_mm,d.plane_score,marker="o"); axs[2].axhline(.60,ls="--"); axs[2].set_ylabel("plane score")
        if self.step_log is not None and len(self.step_log):
            g=self.step_log.groupby("step",as_index=False).agg(aorta_distance_mm=("aorta_distance_mm","min"))
            axs[3].plot(g.step*.8,g.aorta_distance_mm,marker=".")
            axs[3].set_ylabel("min aorta dist mm")
            axs[3].set_xlabel("search step nominal mm")
        fig.suptitle("Proximal continuation source-resolution profiles")
        fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(figs[3],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(2,4,figsize=(11,5.5))
        p=resample_path(self.combined,self.spacing,.35)
        s=arc_mm(p,self.spacing)
        for ax,x in zip(axs.ravel(),np.linspace(0,min(12,s[-1]),8)):
            i=int(np.argmin(abs(s-x)))
            i0,i1=max(0,i-4),min(len(p)-1,i+4)
            im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing)
            ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower")
            ax.plot(0,0,"+",ms=8)
            ax.set_aspect("equal")
            ax.set_title(f"endpoint +{x:.1f} mm")
        fig.suptitle("Current proximal LAD endpoint neighborhood — not automatically called the LAD origin")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[4],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(15,4.8))
        l=source_zyx_to_lps(self.meta,self.combined)
        for ax,(a,b,title) in zip(axs,[(0,2,"L-R vs S-I"),(1,2,"P-A vs S-I"),(0,1,"L-R vs P-A")]):
            ax.plot(l[:,a],l[:,b],".-")
            ax.plot(l[0,a],l[0,b],"x",ms=9,label="proximal endpoint")
            ax.set_title(title)
            ax.set_aspect("equal",adjustable="datalim")
            ax.legend(fontsize=8)
        fig.suptitle("Combined LAD course in patient LPS coordinates")
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[5],dpi=180,bbox_inches="tight"); plt.close(fig)
        done.write_text(ALGORITHM_VERSION)
        self._record("figures","recomputed_and_cached",done)
        return figs

    def package(self):
        done=self.cache/"report.done"
        zpath=self.out/"OPENPLAQUE_LAD_RECENTER_CONTINUATION_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists():
            self._record("report","reused",zpath)
            return zpath
        figs=self.plot_qc()
        html=self.out/"OPENPLAQUE_LAD_RECENTER_CONTINUATION_REPORT.html"
        def img(fp):
            return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(Path(fp).read_bytes()).decode()}'>"
        html.write_text("<html><body><h1>OpenPlaque — LAD proximal recenter continuation</h1><p><b>Research use only.</b> Starts from the previously validated LAD and continues only proximally. Each step is recentered on a coronary-sized source-resolution lumen component. TotalSegmentator contributes only the aorta constraint. No LCX or plaque analysis.</p>"+"".join(img(f) for f in figs)+"<h2>Validated RCA calibration</h2><pre>"+json.dumps(self.rca_cal,indent=2)+"</pre><h2>Established LAD</h2><pre>"+json.dumps(self.established_summary,indent=2)+"</pre><h2>Continuation</h2><pre>"+json.dumps(self.continuation_summary,indent=2)+"</pre><h2>Current proximal endpoint</h2><pre>"+json.dumps(self.endpoint,indent=2)+"</pre></body></html>",encoding="utf-8")
        names=[f.name for f in figs]+["cache_provenance.csv",html.name]
        for n in ("validated_rca_calibration.json","established_lad_centerline.csv","established_lad_qc.csv","established_lad_summary.json","continuation_centerline.csv","continuation_qc.csv","continuation_summary.json","combined_lad_centerline.csv","proximal_endpoint.json","continuation_step_log.csv"):
            p=self.cache/n
            if p.exists():
                shutil.copyfile(p,self.out/n)
                names.append(n)
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for n in dict.fromkeys(names):
                p=self.out/n
                if p.exists(): z.write(p,arcname=n)
        done.write_text(ALGORITHM_VERSION)
        self._record("report","recomputed_and_cached",zpath)
        return zpath
