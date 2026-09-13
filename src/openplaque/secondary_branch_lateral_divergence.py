from __future__ import annotations

"""Focused secondary-coronary branch test using lateral divergence.

This workflow freezes the previously established LAD/takeoff/proximal-trunk
geometry and tests only whether a second coronary-sized tube separates
progressively from that reference. It does not rediscover the LAD and does not
assign an LCX label automatically. Research use only.
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

ALGORITHM_VERSION = "secondary-branch-lateral-divergence-v1.0"
PINNED_RCA = {
    "median_radius_mm": 1.616906,
    "median_center_hu": 551.061478,
    "plane_pass_fraction": 0.916667,
}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))


def arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) <= 1:
        return np.zeros(len(p), float)
    d = np.diff(p, axis=0) * np.asarray(spacing, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def resample_path(path, spacing, step_mm=0.3):
    p = np.asarray(path, float)
    if len(p) < 2:
        return p.copy()
    s = arc_mm(p, spacing)
    if s[-1] <= step_mm:
        return p.copy()
    q = np.arange(0.0, s[-1], step_mm)
    if len(q) == 0 or q[-1] < s[-1] - 1e-6:
        q = np.r_[q, s[-1]]
    return np.column_stack([np.interp(q, s, p[:, k]) for k in range(3)])


def _orth_basis(t):
    t = _unit(t)
    ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.82 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(t, ref))
    v = _unit(np.cross(t, u))
    return u, v


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=5.5, pix_mm=0.18):
    u, v = _orth_basis(tangent_zyx_mm)
    c = np.arange(-half_mm, half_mm + 1e-9, pix_mm, dtype=np.float32)
    U, V = np.meshgrid(c, c, indexing="xy")
    sp = np.asarray(spacing_zyx, np.float32)
    pm = np.asarray(point_zyx, np.float32) * sp
    pos = pm[None, None, :] + U[..., None] * u.astype(np.float32) + V[..., None] * v.astype(np.float32)
    vox = pos / sp[None, None, :]
    out = np.empty(U.shape, dtype=np.float32)
    map_coordinates(ct, [vox[..., 0], vox[..., 1], vox[..., 2]], output=out, order=1, mode="nearest", prefilter=False)
    return out, c


def _component_metrics(im, c, comp, cu, cv):
    pix = float(abs(c[1] - c[0]))
    area = int(comp.sum())
    radius = math.sqrt(area * pix * pix / math.pi)
    er = ndi.binary_erosion(comp, structure=np.ones((3, 3), bool))
    per = max(1, int((comp & ~er).sum())) * pix
    circ = float(np.clip(4 * math.pi * area * pix * pix / max(per * per, 1e-8), 0, 1.2))
    U, V = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(U - cu, V - cv)
    center = float(np.median(im[R <= 0.70]))
    core = float(np.median(im[R <= 1.0]))
    rv = im[(R >= 2.5) & (R <= 4.0)]
    ring = float(np.median(rv)) if rv.size else center
    return {
        "radius_mm": radius,
        "circularity": circ,
        "center_hu": center,
        "core_minus_ring_hu": core - ring,
        "centroid_u_mm": float(cu),
        "centroid_v_mm": float(cv),
        "recenter_shift_mm": float(math.hypot(cu, cv)),
    }


def find_component(im, c, rca, search_mm=1.7, rescue=False):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    lo_hu = max(170.0, min(260.0, 0.38 * rh))
    bright = (im >= lo_hu) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    pix = float(abs(c[1] - c[0]))
    lim = 2.8 if rescue else float(search_mm)
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
        if shift > lim or not (0.60 <= radius <= 3.2):
            continue
        m = _component_metrics(im, c, comp, cu, cv)
        rscore = math.exp(-0.5 * ((m["radius_mm"] - 1.08 * rr) / max(0.70 * rr, 0.82)) ** 2)
        pscore = math.exp(-0.5 * (m["recenter_shift_mm"] / (1.30 if rescue else 0.95)) ** 2)
        cscore = float(np.clip(m["circularity"] / 0.48, 0, 1))
        hscore = math.exp(-0.5 * ((m["center_hu"] - rh) / 360.0) ** 2)
        xscore = 1.0 / (1.0 + math.exp(-(m["core_minus_ring_hu"] - 5.0) / 90.0))
        choose = 0.33 * rscore + 0.28 * pscore + 0.16 * cscore + 0.12 * hscore + 0.11 * xscore
        if best is None or choose > best[0]:
            m["choose_score"] = float(choose)
            best = (choose, m)
    return None if best is None else best[1]


def score_component(m, rca, rescue=False):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    r = float(m["radius_mm"]); sh = float(m["recenter_shift_mm"]); circ = float(m["circularity"])
    hu = float(m["center_hu"]); con = float(m["core_minus_ring_hu"])
    rs = math.exp(-0.5 * ((r - 1.05 * rr) / max(0.70 * rr, 0.80)) ** 2)
    ss = math.exp(-0.5 * (sh / (1.30 if rescue else 0.95)) ** 2)
    cs = float(np.clip(circ / 0.52, 0, 1))
    hs = math.exp(-0.5 * ((hu - rh) / 350.0) ** 2)
    xs = 1.0 / (1.0 + math.exp(-(con - 10.0) / 85.0))
    score = 0.32 * rs + 0.25 * ss + 0.17 * cs + 0.14 * hs + 0.12 * xs
    hard = bool(0.56 * rr <= r <= min(3.25, 2.0 * rr + 0.15) and sh <= (1.9 if rescue else 1.5) and circ >= 0.16 and 120 <= hu <= 1150 and score >= 0.58)
    soft = bool(0.45 * rr <= r <= 3.25 and sh <= (2.8 if rescue else 2.2) and circ >= 0.08 and 90 <= hu <= 1200 and score >= 0.48)
    return float(score), hard, soft


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))


def _nearest_reference_distance(point, reference, spacing, takeoff, exclude_mm=1.1):
    p = np.asarray(reference, float)
    sp = np.asarray(spacing, float)
    d0 = np.linalg.norm((p - np.asarray(takeoff, float)) * sp, axis=1)
    keep = d0 >= float(exclude_mm)
    q = p[keep] if np.any(keep) else p
    return float(np.min(np.linalg.norm((q - np.asarray(point, float)) * sp, axis=1)))


def _serial_qc(path, ct, spacing, rca, step=0.75):
    p = resample_path(path, spacing, 0.28)
    s = arc_mm(p, spacing)
    rows = []
    if len(p) >= 4:
        for x in np.arange(min(0.28, 0.08 * s[-1]), s[-1] + 1e-6, step):
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 5), min(len(p) - 1, i + 5)
            t = (p[i1] - p[i0]) * np.asarray(spacing, float)
            im, c = orthogonal_plane(ct, p[i], t, spacing)
            m = find_component(im, c, rca, 1.55, False)
            if m is None:
                rows.append({"arc_mm": float(s[i]), "radius_mm": np.nan, "circularity": np.nan, "center_hu": np.nan, "core_minus_ring_hu": np.nan, "recenter_shift_mm": np.inf, "plane_score": 0.0, "plane_pass": False, "soft_pass": False})
            else:
                sc, hp, sp = score_component(m, rca, False)
                rows.append({"arc_mm": float(s[i]), **m, "plane_score": sc, "plane_pass": hp, "soft_pass": sp})
    df = pd.DataFrame(rows)
    return df, {
        "length_mm": float(s[-1]) if len(s) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "soft_pass_fraction": float(df.soft_pass.mean()) if len(df) else 0.0,
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else np.nan,
        "median_recenter_shift_mm": float(df.recenter_shift_mm.replace([np.inf], np.nan).median()) if len(df) else np.nan,
        "median_circularity": float(df.circularity.median()) if len(df) else np.nan,
    }


class SecondaryBranchLateralDivergenceWorkflow:
    COMPONENTS = ("source_ct", "frozen_geometry", "branch_search", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "Secondary_Branch_Lateral_Divergence_v1"
        self.out = self.root / "Secondary_Branch_Lateral_Divergence_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse: self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.spacing = self.meta = None
        self.reference = self.trunk = self.takeoff = self.lad_distal = self.rca = None
        self.branch = self.branch_qc = self.candidates = self.summary = None

    def _record(self, c, a, p="", n=""):
        self.prov.append({"component": c, "reuse_requested": self.reuse[c], "action": a, "path": str(p), "note": n})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {"source_ct":"series7_int16.npy","frozen_geometry":"frozen_geometry.json","branch_search":"branch_summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def _first(self, rels):
        for r in rels:
            p = self.root / r
            if p.exists(): return p
        return None

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"; om = self.cache / "series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and om.exists():
            self.ct = np.load(own, mmap_mode="r"); self.meta = _json(om); self.spacing = np.asarray(self.meta["spacing_zyx"], float); self._record("source_ct","reused",own); return self.ct
        src = self._first(["Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy","Cache/LAD_Proximal_Recenter_v1/series7_int16.npy","Cache/LAD_Confirmed_Backtrack_v1/series7_int16.npy"])
        if src is None: raise FileNotFoundError("Disk-backed series-7 CCTA cache not found")
        sm = src.with_suffix(".json")
        shutil.copyfile(src, own); shutil.copyfile(sm, om)
        self.ct = np.load(own, mmap_mode="r"); self.meta = _json(om); self.spacing = np.asarray(self.meta["spacing_zyx"], float); self._record("source_ct","imported_prior_cache",src); return self.ct

    def load_frozen_geometry(self):
        if self.ct is None: self.load_source_ct()
        alt = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "alternative_01_centerline.csv"
        cand = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "takeoff_candidate.json"
        trunk = self.root / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv"
        tsum = self.root / "LAD_Takeoff_Confirmation_Report" / "trunk_summary.json"
        rca = self.root / "LAD_Takeoff_Confirmation_Report" / "validated_rca_calibration.json"
        if not all(p.exists() for p in (alt,cand,trunk,tsum)): raise FileNotFoundError("Required prior takeoff-confirmation outputs not found")
        ts = _json(tsum)
        if not bool(ts.get("accepted", False)): raise RuntimeError("Prior proximal trunk did not pass; refusing lateral-divergence search")
        self.reference = pd.read_csv(alt)[["z","y","x"]].to_numpy(float)
        self.trunk = pd.read_csv(trunk)[["z","y","x"]].to_numpy(float)
        info = _json(cand)
        self.rca = _json(rca) if rca.exists() else dict(PINNED_RCA)
        if not (1.0 <= float(self.rca.get("median_radius_mm", np.nan)) <= 2.3): self.rca = dict(PINNED_RCA)
        p = resample_path(self.reference, self.spacing, 0.20); s = arc_mm(p, self.spacing)
        i = int(np.argmin(abs(s - float(info["arc_from_proximal_mm"]))))
        self.takeoff = p[i].copy(); self.lad_distal = p[i:].copy()
        _write_json({"candidate":info,"trunk_summary":ts,"algorithm":ALGORITHM_VERSION}, self.cache / "frozen_geometry.json")
        self._record("frozen_geometry","loaded_prior_confirmed_geometry",trunk)
        return info

    def _propose(self, cur, direction, rescue=False):
        step = 0.64
        guess = cur + (step * _unit(direction)) / self.spacing
        if np.any(guess < 2) or np.any(guess >= np.asarray(self.ct.shape) - 3): return None
        im, c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        m = find_component(im, c, self.rca, 2.5 if rescue else 1.65, rescue)
        if m is None: return None
        u, v = _orth_basis(direction)
        p = (guess * self.spacing + m["centroid_u_mm"] * u + m["centroid_v_mm"] * v) / self.spacing
        vec = (p - cur) * self.spacing; L = float(np.linalg.norm(vec))
        if not (0.35 <= L <= (2.4 if rescue else 1.65)): return None
        sc, hp, sp = score_component(m, self.rca, rescue)
        if not sp: return None
        sep = _nearest_reference_distance(p, self.reference, self.spacing, self.takeoff, 1.1)
        return {"point":p,"direction":_unit(vec),"score":sc,"hard":hp,"soft":sp,"sep":sep,"step":L,**m}

    def search_branch(self, max_length_mm=11.0, beam_width=28):
        sf=self.cache/"branch_summary.json"; pf=self.cache/"branch_centerline.csv"; qf=self.cache/"branch_qc.csv"; cf=self.cache/"branch_candidates.csv"
        if self.takeoff is None: self.load_frozen_geometry()
        if self.reuse["branch_search"] and all(p.exists() for p in (sf,pf,qf,cf)):
            self.summary=_json(sf); self.branch=pd.read_csv(pf)[["z","y","x"]].to_numpy(float); self.branch_qc=pd.read_csv(qf); self.candidates=pd.read_csv(cf); self._record("branch_search","reused",sf); return self.summary

        td = resample_path(self.lad_distal,self.spacing,.25); tr = resample_path(self.trunk,self.spacing,.25)
        lad_dir = _unit((td[min(len(td)-1,10)]-td[0])*self.spacing)
        trunk_dir = _unit((tr[0]-tr[-1])*self.spacing)
        u,v = _orth_basis(lad_dir)
        seeds=[]
        for ph in np.linspace(0,2*math.pi,64,endpoint=False):
            for deg in (50,65,80,95,110,125,140):
                a=math.radians(deg); d=_unit(math.cos(a)*lad_dir+math.sin(a)*(math.cos(ph)*u+math.sin(ph)*v))
                if _angle(d,trunk_dir)<35: continue
                r=self._propose(self.takeoff,d,True)
                if r is None: continue
                seeds.append({"point":r["point"],"direction":r["direction"],"path":[self.takeoff.copy(),r["point"].copy()],"scores":[r["score"]],"hard":[r["hard"]],"sep":[r["sep"]],"gain":[0.0],"obj":r["score"]})
        seeds.sort(key=lambda st:st["obj"],reverse=True)
        beams=[]
        for st in seeds:
            if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>.45 for q in beams): beams.append(st)
            if len(beams)>=beam_width: break
        pool=list(beams)
        nsteps=int(math.ceil(max_length_mm/.64))
        for _ in range(1,nsteps):
            props=[]
            for st in beams:
                L0=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1]); prev_sep=float(st["sep"][-1])
                local=[]
                for ang in (0,16,28,40,52):
                    if ang==0: dirs=[st["direction"]]
                    else:
                        uu,vv=_orth_basis(st["direction"]); aa=math.radians(ang)
                        dirs=[_unit(math.cos(aa)*st["direction"]+math.sin(aa)*(math.cos(ph)*uu+math.sin(ph)*vv)) for ph in np.linspace(0,2*math.pi,14,endpoint=False)]
                    for d in dirs:
                        r=self._propose(st["point"],d,False)
                        if r is None: continue
                        gain=float(r["sep"]-prev_sep)
                        if L0>=1.4 and gain < -0.12: continue
                        if L0>=2.5 and r["sep"] < 0.85: continue
                        local.append((r,gain))
                if not local:
                    for ang in (60,75):
                        uu,vv=_orth_basis(st["direction"]); aa=math.radians(ang)
                        for ph in np.linspace(0,2*math.pi,16,endpoint=False):
                            d=_unit(math.cos(aa)*st["direction"]+math.sin(aa)*(math.cos(ph)*uu+math.sin(ph)*vv))
                            r=self._propose(st["point"],d,True)
                            if r is None: continue
                            gain=float(r["sep"]-prev_sep)
                            if L0>=1.4 and gain < -0.18: continue
                            local.append((r,gain))
                for r,gain in local:
                    smooth=float(np.clip(np.dot(st["direction"],r["direction"]),-1,1))
                    lateral=min(max(r["sep"]-0.5,0)/3.0,1.0)
                    positive_gain=min(max(gain+0.05,0)/0.45,1.0)
                    step_obj=0.48*r["score"]+0.10*float(r["hard"])+0.10*((smooth+1)/2)+0.18*lateral+0.14*positive_gain
                    props.append({"point":r["point"],"direction":r["direction"],"path":st["path"]+[r["point"].copy()],"scores":st["scores"]+[r["score"]],"hard":st["hard"]+[r["hard"]],"sep":st["sep"]+[r["sep"]],"gain":st["gain"]+[gain],"obj":st["obj"]+step_obj})
            if not props: break
            def rank(st):
                L=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1]); sep=np.asarray(st["sep"],float); gains=np.diff(sep) if len(sep)>1 else np.array([0.0]); mono=float(np.mean(gains>=-0.08))
                return 0.32*np.mean(st["scores"])+0.13*np.mean(st["hard"])+0.20*min(L/8,1)+0.20*min(sep[-1]/3.5,1)+0.15*mono
            props.sort(key=rank,reverse=True)
            keep=[]
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=0.55 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            pool.extend(keep); beams=keep

        rows=[]; evaluated=[]
        for st in pool:
            path=np.asarray(st["path"],float); L=float(arc_mm(path,self.spacing)[-1])
            if L<3.0: continue
            qdf,qsum=_serial_qc(path,self.ct,self.spacing,self.rca,.72)
            sep=np.asarray(st["sep"],float); gains=np.diff(sep) if len(sep)>1 else np.array([0.0])
            mono=float(np.mean(gains>=-0.08)); net=float(sep[-1]-sep[0]); end=float(sep[-1])
            sep5=float(sep[min(len(sep)-1, max(1,int(round(5.0/max(L/(len(sep)-1),.1)))) )]) if len(sep)>1 else end
            rank=0.24*min(L/8,1)+0.25*qsum["plane_pass_fraction"]+0.21*qsum["median_plane_score"]+0.18*min(end/3.5,1)+0.12*mono
            row={"candidate":len(rows)+1,"length_mm":L,"plane_pass_fraction":qsum["plane_pass_fraction"],"soft_pass_fraction":qsum["soft_pass_fraction"],"median_plane_score":qsum["median_plane_score"],"median_radius_mm":qsum["median_radius_mm"],"endpoint_reference_separation_mm":end,"initial_reference_separation_mm":float(sep[0]),"net_separation_gain_mm":net,"monotonic_separation_fraction":mono,"approx_separation_at_5mm":sep5,"rank_score":rank}
            rows.append(row); evaluated.append((rank,row,qsum,qdf,path,st))
        if not evaluated:
            self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.candidates=pd.DataFrame(rows); self.summary={"status":"FAIL","accepted":False,"reason":"no_lateral_candidate_persisted_beyond_3_mm"}
        else:
            evaluated.sort(key=lambda x:x[0],reverse=True); rank,row,qsum,qdf,path,st=evaluated[0]
            accepted=bool(row["length_mm"]>=6.0 and row["plane_pass_fraction"]>=0.75 and row["soft_pass_fraction"]>=0.85 and row["median_plane_score"]>=0.78 and row["endpoint_reference_separation_mm"]>=3.0 and row["net_separation_gain_mm"]>=2.0 and row["monotonic_separation_fraction"]>=0.70)
            self.branch=resample_path(path,self.spacing,.30); self.branch_qc=qdf
            self.summary={**qsum,**{k:v for k,v in row.items() if k not in qsum},"status":"PASS" if accepted else "REVIEW","accepted":accepted,"interpretation":"persistent laterally diverging coronary-like secondary branch; not automatically labeled LCX" if accepted else "coronary-like trajectory did not satisfy independent monotonic-divergence gate"}
        self.candidates=pd.DataFrame(rows).sort_values("rank_score",ascending=False) if rows else pd.DataFrame()
        pd.DataFrame({"arc_mm":arc_mm(self.branch,self.spacing),"z":self.branch[:,0],"y":self.branch[:,1],"x":self.branch[:,2]}).to_csv(pf,index=False)
        self.branch_qc.to_csv(qf,index=False); self.candidates.to_csv(cf,index=False); _write_json(self.summary,sf)
        self._record("branch_search","recomputed_and_cached",sf,f"status={self.summary.get('status')}, length={self.summary.get('length_mm',0):.2f}")
        return self.summary

    def make_figures(self):
        done=self.cache/"figures.done"
        if self.reuse["figures"] and done.exists() and all((self.out/f).exists() for f in ("01_lateral_branch_cross_sections.png","02_branch_vs_reference_mips.png","03_monotonic_separation_profile.png","04_candidate_leaderboard.png")):
            self._record("figures","reused",done); return
        if self.summary is None: self.search_branch()
        p=resample_path(self.branch,self.spacing,.30); s=arc_mm(p,self.spacing)
        # Cross sections
        xs=np.linspace(min(.3,s[-1]),s[-1],min(12,max(2,int(s[-1]/.7)+1))) if s[-1]>0 else np.array([0])
        fig,axs=plt.subplots(3,4,figsize=(14,10)); axs=axs.ravel()
        for ax in axs: ax.axis("off")
        for ax,x in zip(axs,xs):
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); t=(p[i1]-p[i0])*self.spacing; im,c=orthogonal_plane(self.ct,p[i],t,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=900,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,lw=.7); ax.axvline(0,lw=.7); ax.set_title(f"{s[i]:.1f} mm")
        fig.suptitle(f"Secondary branch lateral-divergence QC — {self.summary.get('status')}"); fig.tight_layout(); fig.savefig(self.out/"01_lateral_branch_cross_sections.png",dpi=170); plt.close(fig)
        # MIPs
        pts=np.vstack([self.reference,self.branch]); lo=np.floor(pts.min(0)-18/np.asarray(self.spacing)).astype(int); hi=np.ceil(pts.max(0)+18/np.asarray(self.spacing)).astype(int); lo=np.maximum(lo,0); hi=np.minimum(hi,np.asarray(self.ct.shape)-1); crop=np.asarray(self.ct[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1],dtype=np.float32)
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,axis,title in zip(axs,(0,1,2),("axial-axis MIP","coronal-axis MIP","sagittal-axis MIP")):
            ax.imshow(np.max(crop,axis=axis),cmap="gray",vmin=-100,vmax=900); ax.set_title(title); ax.axis("off")
        fig.suptitle("Frozen LAD/root reference vs laterally diverging secondary trajectory"); fig.tight_layout(); fig.savefig(self.out/"02_branch_vs_reference_mips.png",dpi=170); plt.close(fig)
        # Separation profile recomputed from branch points
        bs=arc_mm(self.branch,self.spacing); sep=[_nearest_reference_distance(q,self.reference,self.spacing,self.takeoff,1.1) for q in self.branch]
        fig,ax=plt.subplots(figsize=(9,5)); ax.plot(bs,sep,marker="o",ms=3); ax.axhline(3.0,ls="--"); ax.set_xlabel("Branch arc length (mm)"); ax.set_ylabel("Distance from LAD/root reference (mm)"); ax.set_title("Lateral separation profile"); fig.tight_layout(); fig.savefig(self.out/"03_monotonic_separation_profile.png",dpi=170); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5));
        if len(self.candidates):
            top=self.candidates.head(20).copy(); ax.scatter(top.endpoint_reference_separation_mm,top.plane_pass_fraction,s=40+140*top.rank_score); ax.axvline(3.0,ls="--"); ax.axhline(.75,ls="--"); ax.set_xlabel("Endpoint separation (mm)"); ax.set_ylabel("Plane pass fraction")
        ax.set_title("Candidate leaderboard"); fig.tight_layout(); fig.savefig(self.out/"04_candidate_leaderboard.png",dpi=170); plt.close(fig)
        done.write_text(ALGORITHM_VERSION); self._record("figures","recomputed_and_cached",done)

    def make_report(self):
        done=self.cache/"report.done"
        if self.summary is None: self.search_branch()
        self.make_figures()
        if self.reuse["report"] and done.exists() and (self.out/"OPENPLAQUE_SECONDARY_BRANCH_LATERAL_DIVERGENCE_REPORT_BACK.zip").exists():
            self._record("report","reused",done); return self.out/"OPENPLAQUE_SECONDARY_BRANCH_LATERAL_DIVERGENCE_REPORT_BACK.zip"
        _write_json(self.summary,self.out/"branch_summary.json"); self.branch_qc.to_csv(self.out/"branch_qc.csv",index=False); self.candidates.to_csv(self.out/"branch_candidates.csv",index=False); pd.DataFrame({"arc_mm":arc_mm(self.branch,self.spacing),"z":self.branch[:,0],"y":self.branch[:,1],"x":self.branch[:,2]}).to_csv(self.out/"branch_centerline.csv",index=False)
        imgs=["01_lateral_branch_cross_sections.png","02_branch_vs_reference_mips.png","03_monotonic_separation_profile.png","04_candidate_leaderboard.png"]
        html=["<html><body><h1>OpenPlaque — Secondary Branch Lateral Divergence</h1>",f"<p><b>Algorithm:</b> {ALGORITHM_VERSION}</p>",f"<pre>{json.dumps(self.summary,indent=2)}</pre>"]
        for fn in imgs:
            b64=base64.b64encode((self.out/fn).read_bytes()).decode(); html.append(f"<h2>{fn}</h2><img style='max-width:100%' src='data:image/png;base64,{b64}'>")
        html.append("<p>PASS supports an independent coronary-like secondary branch only; it is not an automatic LCX label.</p></body></html>")
        (self.out/"OPENPLAQUE_SECONDARY_BRANCH_LATERAL_DIVERGENCE_REPORT.html").write_text("\n".join(html))
        zip_path=self.out/"OPENPLAQUE_SECONDARY_BRANCH_LATERAL_DIVERGENCE_REPORT_BACK.zip"
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
            for p in self.out.iterdir():
                if p.is_file() and p.name!=zip_path.name: z.write(p,p.name)
        done.write_text(ALGORITHM_VERSION); self._record("report","recomputed_and_cached",zip_path)
        return zip_path
