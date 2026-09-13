from __future__ import annotations

"""Focused confirmation of a candidate LAD takeoff/bifurcation neighborhood.

This workflow does not rediscover the LAD. It reuses the previously validated
root-directed alternative 1 from ``LAD_Takeoff_Root_Alternatives_v1`` and tests
only two hypotheses:

1. The short segment from the candidate takeoff point proximally toward the
   aortic root is a continuous coronary-sized lumen (left-main-like trunk).
2. At least one non-LAD branch emerging from the candidate persists as an
   independent coronary-sized tube for 6--10 mm.

The accepted RCA calibration is used as a subject-specific positive reference.
TotalSegmentator contributes only the existing aorta distance/exclusion mask.
No LCX label is assigned automatically. Research use only.
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

ALGORITHM_VERSION = "lad-takeoff-confirmation-v1.0"
PINNED_VALIDATED_RCA = {
    "length_mm": 52.1978,
    "median_plane_score": 0.932688,
    "plane_pass_fraction": 0.916667,
    "median_radius_mm": 1.616906,
    "median_offset_mm": 0.317649,
    "median_circularity": 1.2,
    "median_center_hu": 551.061478,
    "median_core_minus_ring_hu": 503.91744,
}


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def arc_mm(path, spacing):
    p = np.asarray(path, float)
    if len(p) == 0:
        return np.zeros(0, float)
    if len(p) == 1:
        return np.zeros(1, float)
    d = np.diff(p, axis=0) * np.asarray(spacing, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(d, axis=1))]


def resample_path(path, spacing, step_mm=0.35):
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


def _json_read(path):
    return json.loads(Path(path).read_text())


def _json_write(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x))


def _orth_basis(tangent_zyx_mm):
    t = _unit(tangent_zyx_mm)
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
    circ = float(np.clip(4 * math.pi * (area * pix * pix) / max(per * per, 1e-8), 0, 1.2))
    U, V = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(U - cu, V - cv)
    center = float(np.median(im[R <= 0.70]))
    core = float(np.median(im[R <= 1.0]))
    ring_vals = im[(R >= 2.5) & (R <= 4.0)]
    ring = float(np.median(ring_vals)) if ring_vals.size else center
    return {
        "radius_mm": radius,
        "circularity": circ,
        "center_hu": center,
        "core_minus_ring_hu": core - ring,
        "centroid_u_mm": float(cu),
        "centroid_v_mm": float(cv),
        "recenter_shift_mm": float(math.hypot(cu, cv)),
    }


def find_component(im, c, rca, search_mm=1.35, rescue=False):
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
        if shift > lim or not (0.60 <= radius <= 3.45):
            continue
        m = _component_metrics(im, c, comp, cu, cv)
        rscore = math.exp(-0.5 * ((m["radius_mm"] - 1.12 * rr) / max(0.72 * rr, 0.82)) ** 2)
        pscore = math.exp(-0.5 * (m["recenter_shift_mm"] / (1.25 if rescue else 0.95)) ** 2)
        cscore = float(np.clip(m["circularity"] / 0.48, 0, 1))
        hscore = math.exp(-0.5 * ((m["center_hu"] - rh) / 360.0) ** 2)
        xscore = 1.0 / (1.0 + math.exp(-(m["core_minus_ring_hu"] - 5.0) / 90.0))
        choose = 0.33 * rscore + 0.28 * pscore + 0.16 * cscore + 0.12 * hscore + 0.11 * xscore
        if best is None or choose > best[0]:
            m["component_choose_score"] = float(choose)
            best = (choose, m)
    return None if best is None else best[1]


def score_component(m, rca, allow_larger=False, rescue=False):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    r, sh, circ, hu, con = (float(m[k]) for k in ("radius_mm", "recenter_shift_mm", "circularity", "center_hu", "core_minus_ring_hu"))
    target = 1.20 * rr if allow_larger else 1.05 * rr
    rs = math.exp(-0.5 * ((r - target) / max(0.70 * rr, 0.80)) ** 2)
    ss = math.exp(-0.5 * (sh / (1.25 if rescue else 0.95)) ** 2)
    cs = float(np.clip(circ / 0.52, 0, 1))
    hs = math.exp(-0.5 * ((hu - rh) / 350.0) ** 2)
    xs = 1.0 / (1.0 + math.exp(-(con - 10.0) / 85.0))
    score = 0.32 * rs + 0.25 * ss + 0.17 * cs + 0.14 * hs + 0.12 * xs
    upper = min(3.45, (2.15 if allow_larger else 2.00) * rr + 0.15)
    hard = bool(0.56 * rr <= r <= upper and sh <= (1.85 if rescue else 1.50) and circ >= 0.16 and 120 <= hu <= 1150 and score >= 0.58)
    soft = bool(0.45 * rr <= r <= 3.45 and sh <= (2.70 if rescue else 2.20) and circ >= 0.08 and 90 <= hu <= 1200 and score >= 0.48)
    return float(score), hard, soft


def serial_qc(path, ct, spacing, rca, step_sample_mm=0.9, allow_larger=False, label="PATH"):
    p = resample_path(path, spacing, 0.30)
    s = arc_mm(p, spacing)
    rows = []
    if len(p) >= 4 and s[-1] > 0:
        ss = np.arange(min(0.3, 0.08 * s[-1]), s[-1] + 1e-6, step_sample_mm)
        if len(ss) == 0 or ss[-1] < s[-1] - 0.35:
            ss = np.r_[ss, s[-1]]
        for x in ss:
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 5), min(len(p) - 1, i + 5)
            tangent = (p[i1] - p[i0]) * np.asarray(spacing, float)
            im, c = orthogonal_plane(ct, p[i], tangent, spacing)
            m = find_component(im, c, rca, search_mm=1.45, rescue=False)
            if m is None:
                rows.append({"label": label, "arc_mm": float(s[i]), "radius_mm": np.nan, "circularity": np.nan, "center_hu": np.nan, "core_minus_ring_hu": np.nan, "recenter_shift_mm": np.inf, "plane_score": 0.0, "plane_pass": False, "soft_pass": False})
                continue
            sc, hp, sp = score_component(m, rca, allow_larger=allow_larger, rescue=False)
            rows.append({"label": label, "arc_mm": float(s[i]), **m, "plane_score": sc, "plane_pass": hp, "soft_pass": sp})
    df = pd.DataFrame(rows)
    summary = {
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
    return df, summary


def _cone(base, angles=(0, 20, 35, 50), n_az=12):
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


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))


def _nearest_path_distance_mm(point, path, spacing):
    p = np.asarray(path, float)
    d = np.linalg.norm((p - np.asarray(point, float)) * np.asarray(spacing, float), axis=1)
    return float(d.min()) if len(d) else float("inf")


class LADTakeoffConfirmationWorkflow:
    COMPONENTS = ("source_ct", "prior_candidate", "aorta_constraint", "trunk_qc", "secondary_branch", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LAD_Takeoff_Confirmation_v1"
        self.out = self.root / "LAD_Takeoff_Confirmation_Report"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.rca = None
        self.alt1 = self.takeoff = self.trunk = self.lad_distal = None
        self.aorta_lo = self.aorta_crop = self.aorta_dist = None
        self.trunk_qc = self.trunk_summary = None
        self.branch = self.branch_qc = self.branch_summary = self.branch_candidates = None
        self.final_summary = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action, "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {"source_ct":"series7_int16.npy","prior_candidate":"prior_candidate.json","aorta_constraint":"aorta_constraint.npz","trunk_qc":"trunk_summary.json","secondary_branch":"secondary_branch_summary.json","figures":"figures.done","report":"report.done"}
        return pd.DataFrame([{"component":k,"reuse":self.reuse[k],"cache_exists":(self.cache/v).exists()} for k,v in names.items()])

    def _find_existing(self, rels):
        for r in rels:
            p = self.root / r
            if p.exists():
                return p
        return None

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"
        own_meta = self.cache / "series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and own_meta.exists():
            self.ct = np.load(own, mmap_mode="r")
            self.meta = _json_read(own_meta)
            self.spacing = np.asarray(self.meta["spacing_zyx"], float)
            self._record("source_ct", "reused", own)
            return self.ct
        src = self._find_existing([
            "Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy",
            "Cache/LAD_Proximal_Recenter_v1/series7_int16.npy",
            "Cache/LAD_Confirmed_Backtrack_v1/series7_int16.npy",
            "Cache/LAD_Origin_Backtrack_v1/series7_int16.npy",
        ])
        if src is None:
            raise FileNotFoundError("No prior disk-backed series-7 CCTA cache found. Run the previous LAD root-alternatives notebook first.")
        meta = src.with_suffix(".json")
        if not meta.exists():
            raise FileNotFoundError(f"Source CT metadata missing beside {src}")
        shutil.copyfile(src, own)
        shutil.copyfile(meta, own_meta)
        self.ct = np.load(own, mmap_mode="r")
        self.meta = _json_read(own_meta)
        self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", "imported_prior_cache", src)
        return self.ct

    def load_prior_candidate(self):
        sf = self.cache / "prior_candidate.json"
        pf = self.cache / "alternative_01_centerline.csv"
        rf = self.cache / "validated_rca_calibration.json"
        if self.ct is None:
            self.load_source_ct()
        if self.reuse["prior_candidate"] and all(p.exists() for p in (sf,pf,rf)):
            info = _json_read(sf)
            self.alt1 = pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.rca = _json_read(rf)
            self._split_prior(info)
            self._record("prior_candidate", "reused", sf)
            return info
        prior = self.root / "Cache" / "LAD_Takeoff_Root_Alternatives_v1"
        palt = prior / "alternative_01_centerline.csv"
        pcand = prior / "takeoff_candidate.json"
        prca = prior / "validated_rca_calibration.json"
        if not palt.exists():
            palt = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "alternative_01_centerline.csv"
        if not pcand.exists():
            pcand = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "takeoff_candidate.json"
        if not prca.exists():
            prca = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "validated_rca_calibration.json"
        if not palt.exists() or not pcand.exists():
            raise FileNotFoundError("Required alternative 1 and takeoff candidate from prior run were not found")
        info = _json_read(pcand)
        self.alt1 = pd.read_csv(palt)[["z","y","x"]].to_numpy(float)
        x = _json_read(prca) if prca.exists() else dict(PINNED_VALIDATED_RCA)
        self.rca = x if 1.0 <= float(x.get("median_radius_mm",np.nan)) <= 2.3 and float(x.get("plane_pass_fraction",0)) >= 0.75 else dict(PINNED_VALIDATED_RCA)
        pd.DataFrame({"z":self.alt1[:,0],"y":self.alt1[:,1],"x":self.alt1[:,2]}).to_csv(pf,index=False)
        _json_write(info, sf)
        _json_write(self.rca, rf)
        self._split_prior(info)
        self._record("prior_candidate", "imported_prior_result", pcand, f"alternative_id={info.get('alternative_id')}")
        return info

    def _split_prior(self, info):
        p = resample_path(self.alt1, self.spacing, 0.20)
        s = arc_mm(p, self.spacing)
        target_arc = float(info["arc_from_proximal_mm"])
        i = int(np.argmin(abs(s - target_arc)))
        self.takeoff = p[i].copy()
        self.trunk = p[:i+1].copy()
        self.lad_distal = p[i:].copy()

    def load_aorta_constraint(self):
        fp = self.cache / "aorta_constraint.npz"
        if self.takeoff is None:
            self.load_prior_candidate()
        src = fp if (self.reuse["aorta_constraint"] and fp.exists()) else self._find_existing([
            "Cache/LAD_Takeoff_Root_Alternatives_v1/aorta_constraint.npz",
            "Cache/LAD_Proximal_Recenter_v1/aorta_constraint.npz",
            "Cache/LAD_Confirmed_Backtrack_v1/aorta_constraint.npz",
        ])
        if src is None:
            raise FileNotFoundError("Prior TotalSegmentator aorta constraint cache not found")
        z = np.load(src, allow_pickle=False)
        self.aorta_lo = z["lo"].astype(int)
        self.aorta_crop = z["aorta"].astype(bool)
        self.aorta_dist = z["dist"].astype(np.float32)
        if src != fp:
            np.savez_compressed(fp, lo=self.aorta_lo.astype(np.int32), aorta=self.aorta_crop.astype(np.uint8), dist=self.aorta_dist)
        self._record("aorta_constraint", "reused" if src==fp else "imported_prior_constraint", src)
        return self.aorta_dist

    def _aorta_distance(self, p):
        q = np.asarray(p,float) - self.aorta_lo
        if np.any(q<0) or np.any(q>=np.asarray(self.aorta_dist.shape)-1):
            return float("inf"), False
        d = float(map_coordinates(self.aorta_dist, q[:,None], order=1, mode="nearest", prefilter=False)[0])
        qi = np.rint(q).astype(int)
        return d, bool(self.aorta_crop[tuple(qi)])

    def evaluate_trunk(self):
        sf = self.cache / "trunk_summary.json"
        qf = self.cache / "trunk_qc.csv"
        pf = self.cache / "trunk_centerline.csv"
        if self.takeoff is None:
            self.load_prior_candidate()
        if self.aorta_dist is None:
            self.load_aorta_constraint()
        if self.reuse["trunk_qc"] and all(p.exists() for p in (sf,qf,pf)):
            self.trunk_summary = _json_read(sf)
            self.trunk_qc = pd.read_csv(qf)
            self.trunk = pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self._record("trunk_qc", "reused", sf)
            return self.trunk_summary
        qdf, qsum = serial_qc(self.trunk, self.ct, self.spacing, self.rca, 0.75, allow_larger=True, label="PROXIMAL_TRUNK")
        d0,_ = self._aorta_distance(self.trunk[0])
        d1,_ = self._aorta_distance(self.trunk[-1])
        qsum.update({
            "aorta_distance_at_proximal_end_mm": float(d0),
            "aorta_distance_at_takeoff_mm": float(d1),
        })
        accepted = bool(6.0 <= qsum["length_mm"] <= 13.0 and qsum["plane_pass_fraction"] >= 0.78 and qsum["soft_pass_fraction"] >= 0.88 and qsum["median_plane_score"] >= 0.80 and 1.1 <= qsum["median_radius_mm"] <= 3.0)
        qsum["status"] = "PASS" if accepted else "REVIEW"
        qsum["accepted"] = accepted
        qsum["interpretation"] = "continuous coronary-sized proximal trunk from takeoff candidate toward aortic root" if accepted else "proximal trunk requires review; do not assign left-main label automatically"
        self.trunk_qc, self.trunk_summary = qdf, qsum
        qdf.to_csv(qf,index=False)
        pd.DataFrame({"arc_mm":arc_mm(self.trunk,self.spacing),"z":self.trunk[:,0],"y":self.trunk[:,1],"x":self.trunk[:,2]}).to_csv(pf,index=False)
        _json_write(qsum,sf)
        self._record("trunk_qc", "recomputed_and_cached", sf, f"status={qsum['status']}, pass={qsum['plane_pass_fraction']:.3f}")
        return qsum

    def _propose_branch(self, cur, direction, ref_path, rescue=False):
        step = 0.72
        guess = cur + (step * _unit(direction)) / self.spacing
        if np.any(guess<2) or np.any(guess>=np.asarray(self.ct.shape,float)-3):
            return None
        im,c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        u,v = _orth_basis(direction)
        m = find_component(im,c,self.rca,search_mm=2.1,rescue=rescue)
        if m is None:
            return None
        p = (guess*self.spacing + m["centroid_u_mm"]*u + m["centroid_v_mm"]*v) / self.spacing
        vec = (p-cur)*self.spacing
        L = float(np.linalg.norm(vec))
        if not (0.45 <= L <= (2.3 if rescue else 1.75)):
            return None
        ad, inside = self._aorta_distance(p)
        if inside or ad < 0.6:
            return None
        sc,hp,sp = score_component(m,self.rca,allow_larger=False,rescue=rescue)
        if not sp:
            return None
        sep = _nearest_path_distance_mm(p,ref_path,self.spacing)
        return {"point":p,"direction":_unit(vec),"plane_score":sc,"hard":hp,"soft":sp,"step_length_mm":L,"aorta_distance_mm":ad,"reference_separation_mm":sep,**m}

    def trace_secondary_branch(self, max_length_mm=10.0, beam_width=12):
        sf = self.cache / "secondary_branch_summary.json"
        pf = self.cache / "secondary_branch_centerline.csv"
        qf = self.cache / "secondary_branch_qc.csv"
        cf = self.cache / "secondary_branch_candidates.csv"
        if self.trunk_summary is None:
            self.evaluate_trunk()
        if self.reuse["secondary_branch"] and all(p.exists() for p in (sf,pf,qf,cf)):
            self.branch_summary = _json_read(sf)
            self.branch = pd.read_csv(pf)[["z","y","x"]].to_numpy(float)
            self.branch_qc = pd.read_csv(qf)
            self.branch_candidates = pd.read_csv(cf)
            self._record("secondary_branch", "reused", sf)
            return self.branch_summary

        td = resample_path(self.lad_distal,self.spacing,0.25)
        tr = resample_path(self.trunk,self.spacing,0.25)
        lad_dir = _unit((td[min(len(td)-1,8)]-td[0])*self.spacing)
        trunk_prox_dir = _unit((tr[0]-tr[-1])*self.spacing)
        seeds=[]
        for d in _cone(lad_dir,(35,50,65,80,95,110,125),16):
            if _angle_deg(d,lad_dir) < 32: continue
            if _angle_deg(d,trunk_prox_dir) < 28: continue
            r=self._propose_branch(self.takeoff,d,self.alt1,rescue=True)
            if r is None or r["reference_separation_mm"] < 0.75: continue
            seeds.append({"point":r["point"],"direction":r["direction"],"path":[self.takeoff.copy(),r["point"].copy()],"scores":[r["plane_score"]],"hard":[r["hard"]],"soft":[r["soft"]],"sep":[r["reference_separation_mm"]],"obj":r["plane_score"]})
        if not seeds:
            self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.branch_candidates=pd.DataFrame()
            self.branch_summary={"status":"FAIL","accepted":False,"reason":"no_distinct_coronary_sized_secondary_seed"}
            pd.DataFrame({"z":[self.takeoff[0]],"y":[self.takeoff[1]],"x":[self.takeoff[2]]}).to_csv(pf,index=False)
            self.branch_qc.to_csv(qf,index=False); self.branch_candidates.to_csv(cf,index=False); _json_write(self.branch_summary,sf)
            self._record("secondary_branch","recomputed_and_cached",sf,self.branch_summary["reason"])
            return self.branch_summary
        seeds.sort(key=lambda st:np.mean(st["scores"]),reverse=True)
        beams=seeds[:beam_width]
        pool=list(beams)
        nsteps=int(math.ceil(max_length_mm/0.72))
        for step_idx in range(1,nsteps):
            props=[]
            for st in beams:
                local=[]
                for d in _cone(st["direction"],(0,18,32,46),10):
                    r=self._propose_branch(st["point"],d,self.alt1,rescue=False)
                    if r is None: continue
                    cur_len=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
                    minsep=1.15 if cur_len>=1.4 else 0.75
                    if r["reference_separation_mm"] < minsep: continue
                    local.append(r)
                if not local:
                    for d in _cone(st["direction"],(55,70),10):
                        r=self._propose_branch(st["point"],d,self.alt1,rescue=True)
                        if r is None: continue
                        cur_len=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
                        if r["reference_separation_mm"] < (1.15 if cur_len>=1.4 else 0.75): continue
                        local.append(r)
                for r in local:
                    smooth=float(np.clip(np.dot(_unit(st["direction"]),_unit(r["direction"])),-1,1))
                    obj=0.62*r["plane_score"]+0.14*float(r["hard"])+0.10*((smooth+1)/2)+0.14*min(r["reference_separation_mm"]/4.0,1.0)
                    props.append({"point":r["point"],"direction":r["direction"],"path":st["path"]+[r["point"].copy()],"scores":st["scores"]+[r["plane_score"]],"hard":st["hard"]+[r["hard"]],"soft":st["soft"]+[r["soft"]],"sep":st["sep"]+[r["reference_separation_mm"]],"obj":st["obj"]+obj})
            if not props: break
            def rank(st):
                L=float(arc_mm(np.asarray(st["path"]),self.spacing)[-1])
                return 0.42*np.mean(st["scores"])+0.22*np.mean(st["hard"])+0.20*min(L/8.0,1)+0.16*min(np.median(st["sep"])/3.0,1)
            props.sort(key=rank,reverse=True)
            keep=[]
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing)>=0.65 for q in keep): keep.append(st)
                if len(keep)>=beam_width: break
            pool.extend(keep); beams=keep

        rows=[]; evaluated=[]
        for st in pool:
            path=np.asarray(st["path"],float); L=float(arc_mm(path,self.spacing)[-1])
            if L<2.5: continue
            qdf,qsum=serial_qc(path,self.ct,self.spacing,self.rca,0.75,allow_larger=False,label="SECONDARY_BRANCH")
            sep=float(np.median(st["sep"])); endsep=float(st["sep"][-1]); initial_ang=_angle_deg((path[1]-path[0])*self.spacing,lad_dir)
            rank=0.34*min(L/8,1)+0.28*qsum["plane_pass_fraction"]+0.24*qsum["median_plane_score"]+0.14*min(endsep/4,1)
            rows.append({"candidate":len(rows)+1,"length_mm":L,"plane_pass_fraction":qsum["plane_pass_fraction"],"median_plane_score":qsum["median_plane_score"],"median_radius_mm":qsum["median_radius_mm"],"median_reference_separation_mm":sep,"endpoint_reference_separation_mm":endsep,"initial_angle_from_lad_deg":initial_ang,"rank_score":rank})
            evaluated.append((rank,L,qsum,qdf,path,st))
        if not evaluated:
            self.branch=np.asarray([self.takeoff]); self.branch_qc=pd.DataFrame(); self.branch_candidates=pd.DataFrame(rows)
            self.branch_summary={"status":"FAIL","accepted":False,"reason":"no_secondary_branch_persisted_beyond_2_5_mm"}
        else:
            evaluated.sort(key=lambda x:x[0],reverse=True)
            rank,L,qsum,qdf,path,st=evaluated[0]
            self.branch=resample_path(path,self.spacing,0.30); self.branch_qc=qdf
            accepted=bool(L>=6.0 and qsum["plane_pass_fraction"]>=0.72 and qsum["soft_pass_fraction"]>=0.85 and qsum["median_plane_score"]>=0.78 and st["sep"][-1]>=2.3)
            self.branch_summary={**qsum,"status":"PASS" if accepted else "REVIEW","accepted":accepted,"rank_score":float(rank),"endpoint_reference_separation_mm":float(st["sep"][-1]),"median_reference_separation_mm":float(np.median(st["sep"])),"initial_angle_from_lad_deg":float(_angle_deg((path[1]-path[0])*self.spacing,lad_dir)),"interpretation":"persistent independent coronary-like secondary branch from candidate; branch is not automatically labeled LCX" if accepted else "secondary direction did not meet persistence/independence gate"}
        self.branch_candidates=pd.DataFrame(rows).sort_values("rank_score",ascending=False) if rows else pd.DataFrame()
        pd.DataFrame({"arc_mm":arc_mm(self.branch,self.spacing),"z":self.branch[:,0],"y":self.branch[:,1],"x":self.branch[:,2]}).to_csv(pf,index=False)
        self.branch_qc.to_csv(qf,index=False); self.branch_candidates.to_csv(cf,index=False); _json_write(self.branch_summary,sf)
        self._record("secondary_branch","recomputed_and_cached",sf,f"status={self.branch_summary.get('status')}, length={self.branch_summary.get('length_mm',0):.1f} mm")
        return self.branch_summary

    def summarize(self):
        if self.branch_summary is None: self.trace_secondary_branch()
        supported=bool(self.trunk_summary.get("accepted",False) and self.branch_summary.get("accepted",False))
        self.final_summary={"algorithm":ALGORITHM_VERSION,"status":"SUPPORTED_TAKEOFF_NEIGHBORHOOD" if supported else "CANDIDATE_REQUIRES_REVIEW","trunk_status":self.trunk_summary.get("status"),"trunk_length_mm":self.trunk_summary.get("length_mm"),"trunk_plane_pass_fraction":self.trunk_summary.get("plane_pass_fraction"),"trunk_median_plane_score":self.trunk_summary.get("median_plane_score"),"secondary_branch_status":self.branch_summary.get("status"),"secondary_branch_length_mm":self.branch_summary.get("length_mm",0.0),"secondary_branch_plane_pass_fraction":self.branch_summary.get("plane_pass_fraction",0.0),"secondary_branch_endpoint_separation_mm":self.branch_summary.get("endpoint_reference_separation_mm",0.0),"note":"This supports a LAD takeoff/bifurcation neighborhood but does not constitute a manual anatomical label and does not assign the secondary branch as LCX." if supported else "One or both focused confirmation gates failed; retain candidate status only."}
        _json_write(self.final_summary,self.cache/"confirmation_summary.json")
        return self.final_summary

    def _sections(self,path,axs,label,n=8):
        p=resample_path(path,self.spacing,0.30); s=arc_mm(p,self.spacing)
        if len(p)<4: return
        for ax,x in zip(np.ravel(axs),np.linspace(0.25,max(0.25,s[-1]-0.2),n)):
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5)
            im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing)
            ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"{label}\n{x:.1f} mm")

    def plot_qc(self):
        done=self.cache/"figures.done"
        figs=[self.out/f for f in ("01_proximal_trunk_cross_sections.png","02_secondary_branch_cross_sections.png","03_local_takeoff_mips.png","04_confirmation_profiles.png","05_takeoff_neighborhood.png")]
        if self.reuse["figures"] and done.exists() and all(f.exists() for f in figs): self._record("figures","reused",done); return figs
        self.summarize()
        fig,axs=plt.subplots(2,4,figsize=(11,5.5)); self._sections(self.trunk,axs,"Proximal trunk",8); fig.suptitle(f"Candidate-to-root trunk — {self.trunk_summary['status']} ({self.trunk_summary['length_mm']:.1f} mm)"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[0],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(2,4,figsize=(11,5.5)); self._sections(self.branch,axs,"Secondary branch",8); fig.suptitle(f"Independent secondary branch — {self.branch_summary.get('status')} ({self.branch_summary.get('length_mm',0):.1f} mm)"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[1],dpi=180,bbox_inches="tight"); plt.close(fig)
        pts=np.vstack([self.trunk,self.lad_distal[:max(2,min(len(self.lad_distal),80))],self.branch]); pad=np.ceil(8/self.spacing).astype(int); lo=np.maximum(0,np.floor(pts.min(0)).astype(int)-pad); hi=np.minimum(np.asarray(self.ct.shape),np.ceil(pts.max(0)).astype(int)+pad+1); crop=np.asarray(self.ct[tuple(slice(int(lo[d]),int(hi[d])) for d in range(3))],dtype=np.int16)
        fig,axs=plt.subplots(1,3,figsize=(15,5)); ims=[crop.max(0),crop.max(1),crop.max(2)]
        for ax,im,title in zip(axs,ims,("Axial MIP","Coronal MIP","Sagittal MIP")): ax.imshow(im,cmap="gray",vmin=-100,vmax=850,origin="lower"); ax.set_title(title); ax.axis("off")
        def plot_path(path,style,label):
            q=np.asarray(path)-lo; axs[0].plot(q[:,2],q[:,1],style,label=label); axs[1].plot(q[:,2],q[:,0],style,label=label); axs[2].plot(q[:,1],q[:,0],style,label=label)
        plot_path(self.trunk,"-","proximal trunk"); plot_path(self.lad_distal[:max(2,min(len(self.lad_distal),80))],"--","LAD distal"); plot_path(self.branch,"-.","secondary")
        for ax in axs: ax.legend(fontsize=8)
        fig.suptitle("Focused takeoff neighborhood: trunk, LAD, independent secondary branch"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[2],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        if len(self.trunk_qc): axs[0,0].plot(self.trunk_qc.arc_mm,self.trunk_qc.radius_mm,marker="o"); axs[0,1].plot(self.trunk_qc.arc_mm,self.trunk_qc.plane_score,marker="o")
        if len(self.branch_qc): axs[1,0].plot(self.branch_qc.arc_mm,self.branch_qc.radius_mm,marker="o"); axs[1,1].plot(self.branch_qc.arc_mm,self.branch_qc.plane_score,marker="o")
        axs[0,0].set_title("Trunk radius"); axs[0,1].set_title("Trunk plane score"); axs[1,0].set_title("Secondary radius"); axs[1,1].set_title("Secondary plane score")
        for ax in axs.ravel(): ax.set_xlabel("arc mm")
        fig.suptitle("Focused confirmation profiles"); fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(figs[3],dpi=180,bbox_inches="tight"); plt.close(fig)
        fig,axs=plt.subplots(2,4,figsize=(11,5.5)); p=resample_path(np.vstack([self.trunk,self.lad_distal[1:]]),self.spacing,0.25); s=arc_mm(p,self.spacing); tc=float(arc_mm(self.trunk,self.spacing)[-1])
        for ax,off in zip(axs.ravel(),np.linspace(-3.5,3.5,8)):
            x=np.clip(tc+off,0,s[-1]); i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-5),min(len(p)-1,i+5); im,c=orthogonal_plane(self.ct,p[i],(p[i1]-p[i0])*self.spacing,self.spacing); ax.imshow(im,cmap="gray",vmin=-150,vmax=850,extent=[c[0],c[-1],c[0],c[-1]],origin="lower"); ax.plot(0,0,"+",ms=8); ax.set_aspect("equal"); ax.set_title(f"takeoff {off:+.1f} mm")
        fig.suptitle("Candidate LAD takeoff neighborhood — source-resolution orthogonal planes"); fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(figs[4],dpi=180,bbox_inches="tight"); plt.close(fig)
        done.write_text(ALGORITHM_VERSION); self._record("figures","recomputed_and_cached",done); return figs

    def package(self):
        done=self.cache/"report.done"; zpath=self.out/"OPENPLAQUE_LAD_TAKEOFF_CONFIRMATION_REPORT_BACK.zip"
        if self.reuse["report"] and done.exists() and zpath.exists(): self._record("report","reused",zpath); return zpath
        figs=self.plot_qc(); summary=self.summarize(); html=self.out/"OPENPLAQUE_LAD_TAKEOFF_CONFIRMATION_REPORT.html"
        def img(fp): return f"<h2>{Path(fp).name}</h2><img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(Path(fp).read_bytes()).decode()}'>"
        html.write_text("<html><body><h1>OpenPlaque — LAD takeoff focused confirmation</h1><p><b>Research use only.</b> Reuses the validated LAD/root alternative 1 and tests only proximal trunk continuity plus persistence of an independent secondary branch. No automatic LCX label.</p>"+"".join(img(f) for f in figs)+"<h2>Confirmation summary</h2><pre>"+json.dumps(summary,indent=2)+"</pre><h2>Trunk</h2><pre>"+json.dumps(self.trunk_summary,indent=2)+"</pre><h2>Secondary branch</h2><pre>"+json.dumps(self.branch_summary,indent=2)+"</pre></body></html>",encoding="utf-8")
        names=[f.name for f in figs]+[html.name,"cache_provenance.csv"]
        for n in ("prior_candidate.json","validated_rca_calibration.json","trunk_centerline.csv","trunk_qc.csv","trunk_summary.json","secondary_branch_centerline.csv","secondary_branch_qc.csv","secondary_branch_summary.json","secondary_branch_candidates.csv","confirmation_summary.json"):
            p=self.cache/n
            if p.exists(): shutil.copyfile(p,self.out/n); names.append(n)
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            for n in names:
                p=self.out/n
                if p.exists(): z.write(p,arcname=n)
        done.write_text(ALGORITHM_VERSION); self._record("report","recomputed_and_cached",zpath); return zpath
