from __future__ import annotations

"""Continuous source-CCTA bridge test across the distal secondary-branch gap.

The previously accepted 13.8-mm secondary coronary branch is frozen.  The
rollback-relaunch experiment showed a good coronary-sized segment through about
12-13 mm, then several serial detector failures around 14-17 mm, followed by a
recovered good-looking point near 18 mm.  This workflow asks one narrow
question: can a *continuous* sequence of source-CCTA coronary-sized components
connect the last unequivocally good proximal point to that recovered distal
neighborhood without jumps between disconnected bright structures?

Every accepted search step must contain a qualifying contrast-filled component,
recenter only locally, and remain coronary-sized.  The old relaunch trajectory
is used only to define a distal target neighborhood and for comparison; it is
not forced into the bridge.  No LCX label is assigned automatically.
Research use only.
"""

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

ALGORITHM_VERSION = "lcx-gap-bridge-v1.0"
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


def resample_path(path, spacing, step_mm=0.25):
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


def orthogonal_plane(ct, point_zyx, tangent_zyx_mm, spacing_zyx, half_mm=5.0, pix_mm=0.18):
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


def find_component(im, c, rca, search_mm=1.35):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    lo_hu = max(170.0, min(260.0, 0.38 * rh))
    bright = (im >= lo_hu) & (im <= 1200.0)
    lab, nlab = ndi.label(bright, structure=np.ones((3, 3), np.uint8))
    pix = float(abs(c[1] - c[0]))
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
        if shift > float(search_mm) or not (0.65 <= radius <= 2.65):
            continue
        m = _component_metrics(im, c, comp, cu, cv)
        rscore = math.exp(-0.5 * ((m["radius_mm"] - 1.10 * rr) / max(0.62 * rr, 0.72)) ** 2)
        pscore = math.exp(-0.5 * (m["recenter_shift_mm"] / 0.72) ** 2)
        cscore = float(np.clip(m["circularity"] / 0.50, 0, 1))
        hscore = math.exp(-0.5 * ((m["center_hu"] - rh) / 330.0) ** 2)
        xscore = 1.0 / (1.0 + math.exp(-(m["core_minus_ring_hu"] - 10.0) / 80.0))
        choose = 0.33 * rscore + 0.30 * pscore + 0.16 * cscore + 0.11 * hscore + 0.10 * xscore
        if best is None or choose > best[0]:
            m["choose_score"] = float(choose)
            best = (choose, m)
    return None if best is None else best[1]


def score_component(m, rca):
    rr, rh = float(rca["median_radius_mm"]), float(rca["median_center_hu"])
    r = float(m["radius_mm"]); sh = float(m["recenter_shift_mm"]); circ = float(m["circularity"])
    hu = float(m["center_hu"]); con = float(m["core_minus_ring_hu"])
    rs = math.exp(-0.5 * ((r - 1.08 * rr) / max(0.64 * rr, 0.75)) ** 2)
    ss = math.exp(-0.5 * (sh / 0.78) ** 2)
    cs = float(np.clip(circ / 0.52, 0, 1))
    hs = math.exp(-0.5 * ((hu - rh) / 330.0) ** 2)
    xs = 1.0 / (1.0 + math.exp(-(con - 10.0) / 82.0))
    score = 0.33 * rs + 0.27 * ss + 0.17 * cs + 0.13 * hs + 0.10 * xs
    hard = bool(0.60 * rr <= r <= 2.60 and sh <= 1.20 and circ >= 0.18 and 120 <= hu <= 1150 and score >= 0.62)
    soft = bool(0.50 * rr <= r <= 2.65 and sh <= 1.35 and circ >= 0.12 and 100 <= hu <= 1200 and score >= 0.54)
    return float(score), hard, soft


def _nearest_dist(point, ref, spacing):
    p = np.asarray(ref, float)
    sp = np.asarray(spacing, float)
    return float(np.min(np.linalg.norm((p - np.asarray(point, float)) * sp, axis=1)))


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1, 1))))


def _serial_qc(path, ct, spacing, rca, step=0.42):
    p = resample_path(path, spacing, 0.22)
    s = arc_mm(p, spacing)
    rows = []
    if len(p) >= 4:
        xs = np.arange(min(0.22, 0.06 * s[-1]), s[-1] + 1e-6, step)
        if len(xs) == 0 or xs[-1] < s[-1] - 0.15:
            xs = np.r_[xs, s[-1]]
        for x in xs:
            i = int(np.argmin(abs(s - x)))
            i0, i1 = max(0, i - 5), min(len(p) - 1, i + 5)
            t = (p[i1] - p[i0]) * np.asarray(spacing, float)
            im, c = orthogonal_plane(ct, p[i], t, spacing)
            m = find_component(im, c, rca, 1.25)
            if m is None:
                rows.append({"arc_mm": float(s[i]), "radius_mm": np.nan, "circularity": np.nan, "center_hu": np.nan,
                             "core_minus_ring_hu": np.nan, "recenter_shift_mm": np.inf, "plane_score": 0.0,
                             "plane_pass": False, "soft_pass": False})
            else:
                sc, hp, sp = score_component(m, rca)
                rows.append({"arc_mm": float(s[i]), **m, "plane_score": sc, "plane_pass": hp, "soft_pass": sp})
    df = pd.DataFrame(rows)
    fails = (~df.soft_pass.astype(bool)).to_numpy() if len(df) else np.array([], bool)
    max_run = run = 0
    for f in fails:
        run = run + 1 if f else 0
        max_run = max(max_run, run)
    return df, {
        "length_mm": float(s[-1]) if len(s) else 0.0,
        "plane_pass_fraction": float(df.plane_pass.mean()) if len(df) else 0.0,
        "soft_pass_fraction": float(df.soft_pass.mean()) if len(df) else 0.0,
        "median_plane_score": float(df.plane_score.median()) if len(df) else 0.0,
        "median_radius_mm": float(df.radius_mm.median()) if len(df) else np.nan,
        "median_recenter_shift_mm": float(df.recenter_shift_mm.replace([np.inf], np.nan).median()) if len(df) else np.nan,
        "max_consecutive_soft_failures": int(max_run),
    }


class LCXGapBridgeWorkflow:
    COMPONENTS = ("source_ct", "frozen_geometry", "bridge_search", "figures", "report")

    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        self.root = Path(root)
        self.cache = self.root / "Cache" / "LCX_Gap_Bridge_v1"
        self.out = self.root / "LCX_Gap_Bridge_Report"
        self.cache.mkdir(parents=True, exist_ok=True); self.out.mkdir(parents=True, exist_ok=True)
        self.reuse = {k: True for k in self.COMPONENTS}
        if reuse:
            self.reuse.update({k: bool(v) for k, v in reuse.items()})
        self.prov = []
        self.ct = self.meta = self.spacing = None
        self.seed = self.reference = self.trunk = self.rca = None
        self.prior_course = self.prior_qc = None
        self.source_point = self.target_point = None
        self.bridge = self.bridge_qc = self.candidates = self.summary = self.diag = None

    def _record(self, component, action, path="", note=""):
        self.prov.append({"component": component, "reuse_requested": self.reuse[component], "action": action,
                          "path": str(path), "note": note})
        pd.DataFrame(self.prov).to_csv(self.out / "cache_provenance.csv", index=False)

    def cache_status(self):
        names = {"source_ct": "series7_int16.npy", "frozen_geometry": "frozen_geometry.json",
                 "bridge_search": "bridge_summary.json", "figures": "figures.done", "report": "report.done"}
        return pd.DataFrame([{"component": k, "reuse": self.reuse[k], "cache_exists": (self.cache / v).exists()} for k, v in names.items()])

    def _first(self, rels):
        for r in rels:
            p = self.root / r
            if p.exists():
                return p
        return None

    def load_source_ct(self):
        own = self.cache / "series7_int16.npy"; om = self.cache / "series7_int16.json"
        if self.reuse["source_ct"] and own.exists() and om.exists():
            self.ct = np.load(own, mmap_mode="r"); self.meta = _json(om); self.spacing = np.asarray(self.meta["spacing_zyx"], float)
            self._record("source_ct", "reused", own); return self.ct
        src = self._first(["Cache/LCX_Distal_Relaunch_v1/series7_int16.npy", "Cache/LCX_Course_Confirmation_v2/series7_int16.npy",
                           "Cache/LAD_Takeoff_Root_Alternatives_v1/series7_int16.npy"])
        if src is None:
            raise FileNotFoundError("Disk-backed series-7 CCTA cache not found")
        sm = src.with_suffix(".json")
        shutil.copyfile(src, own); shutil.copyfile(sm, om)
        self.ct = np.load(own, mmap_mode="r"); self.meta = _json(om); self.spacing = np.asarray(self.meta["spacing_zyx"], float)
        self._record("source_ct", "imported_prior_cache", src); return self.ct

    def load_frozen_geometry(self):
        if self.ct is None:
            self.load_source_ct()
        seed = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv"
        ss = self.root / "Secondary_Branch_Lateral_Divergence_Report" / "branch_summary.json"
        ref = self.root / "LAD_Takeoff_Root_Alternatives_Report" / "alternative_01_centerline.csv"
        trunk = self.root / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv"
        rca = self.root / "LAD_Takeoff_Confirmation_Report" / "validated_rca_calibration.json"
        prior = self.root / "LCX_Distal_Relaunch_Report" / "best_combined_centerline.csv"
        prior_qc = self.root / "LCX_Distal_Relaunch_Report" / "best_combined_qc.csv"
        if not all(p.exists() for p in (seed, ss, ref, trunk, prior, prior_qc)):
            raise FileNotFoundError("Required accepted branch and distal-relaunch outputs not found")
        info = _json(ss)
        if not bool(info.get("accepted", False)):
            raise RuntimeError("Accepted lateral-divergence branch missing")
        self.seed = pd.read_csv(seed)[["z", "y", "x"]].to_numpy(float)
        self.reference = pd.read_csv(ref)[["z", "y", "x"]].to_numpy(float)
        self.trunk = pd.read_csv(trunk)[["z", "y", "x"]].to_numpy(float)
        self.rca = _json(rca) if rca.exists() else dict(PINNED_RCA)
        if not (1.0 <= float(self.rca.get("median_radius_mm", np.nan)) <= 2.3):
            self.rca = dict(PINNED_RCA)
        self.prior_course = pd.read_csv(prior)[["z", "y", "x"]].to_numpy(float)
        self.prior_qc = pd.read_csv(prior_qc)

        seed_qc, _ = _serial_qc(self.seed, self.ct, self.spacing, self.rca, 0.50)
        good = seed_qc[(seed_qc.plane_pass == True) & (seed_qc.plane_score >= 0.90) &
                       (seed_qc.radius_mm <= 2.25) & (seed_qc.recenter_shift_mm <= 0.80) &
                       (seed_qc.arc_mm >= 11.4) & (seed_qc.arc_mm <= 13.1)]
        if len(good):
            source_arc = float(good.arc_mm.max())
        else:
            source_arc = 12.2
        sp = resample_path(self.seed, self.spacing, 0.18); ssarc = arc_mm(sp, self.spacing)
        si = int(np.argmin(abs(ssarc - source_arc))); self.source_point = sp[si].copy(); source_arc = float(ssarc[si])

        pq = self.prior_qc.copy()
        recovered = pq[(pq.arc_mm >= 17.0) & (pq.plane_pass == True) & (pq.plane_score >= 0.85) &
                       (pq.radius_mm <= 2.50) & (pq.recenter_shift_mm <= 0.90)]
        if not len(recovered):
            recovered = pq[(pq.arc_mm >= 16.5) & (pq.plane_pass == True) & (pq.plane_score >= 0.80)]
        if not len(recovered):
            raise RuntimeError("No recovered distal coronary-like target found in prior relaunch QC")
        target_arc = float(recovered.iloc[0].arc_mm)
        pp = resample_path(self.prior_course, self.spacing, 0.18); parc = arc_mm(pp, self.spacing)
        ti = int(np.argmin(abs(parc - target_arc))); self.target_point = pp[ti].copy(); target_arc = float(parc[ti])

        snap = {"algorithm": ALGORITHM_VERSION, "source_arc_mm": source_arc, "target_prior_arc_mm": target_arc,
                "straight_line_source_to_target_mm": float(np.linalg.norm((self.target_point - self.source_point) * self.spacing)),
                "source_zyx": self.source_point.tolist(), "target_zyx": self.target_point.tolist()}
        _write_json(snap, self.cache / "frozen_geometry.json")
        seed_qc.to_csv(self.out / "accepted_seed_qc.csv", index=False)
        self.prior_qc.to_csv(self.out / "prior_relaunch_qc.csv", index=False)
        self._record("frozen_geometry", "loaded_and_fixed_source_target", self.cache / "frozen_geometry.json")
        return snap

    def _source_direction_hypotheses(self):
        p = resample_path(self.seed, self.spacing, 0.18); s = arc_mm(p, self.spacing)
        i = int(np.argmin(np.linalg.norm((p - self.source_point) * self.spacing, axis=1)))
        out = []
        for back in (0.8, 1.4, 2.2, 3.2):
            j = int(np.argmin(abs(s - max(0.0, s[i] - back))))
            if j >= i:
                continue
            d = _unit((p[i] - p[j]) * self.spacing)
            if all(_angle(d, q) >= 8 for q in out):
                out.append(d)
        td = _unit((self.target_point - self.source_point) * self.spacing)
        if all(_angle(td, q) >= 8 for q in out):
            out.append(td)
        return out

    def _propose(self, cur, direction, step_mm):
        guess = np.asarray(cur, float) + (float(step_mm) * _unit(direction)) / self.spacing
        if np.any(guess < 2) or np.any(guess >= np.asarray(self.ct.shape) - 3):
            return None
        im, c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        m = find_component(im, c, self.rca, 1.30)
        if m is None:
            return None
        u, v = _orth_basis(direction)
        p = (guess * self.spacing + m["centroid_u_mm"] * u + m["centroid_v_mm"] * v) / self.spacing
        vec = (p - cur) * self.spacing; L = float(np.linalg.norm(vec))
        if not (0.18 <= L <= 1.15):
            return None
        sc, hp, sp = score_component(m, self.rca)
        if not sp:
            return None
        refsep = _nearest_dist(p, np.vstack([self.reference, self.trunk]), self.spacing)
        if refsep < 1.75:
            return None
        return {"point": p, "direction": _unit(vec), "score": sc, "hard": hp, "soft": sp,
                "refsep": refsep, "step": L, **m}

    def search_bridge(self, max_bridge_mm=8.5, beam_width=40):
        sf = self.cache / "bridge_summary.json"; pf = self.cache / "bridge_centerline.csv"
        qf = self.cache / "bridge_qc.csv"; cf = self.cache / "bridge_candidates.csv"; df = self.cache / "step_diagnostics.csv"
        if self.source_point is None:
            self.load_frozen_geometry()
        if self.reuse["bridge_search"] and all(p.exists() for p in (sf, pf, qf, cf)):
            self.summary = _json(sf); self.bridge = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.bridge_qc = pd.read_csv(qf); self.candidates = pd.read_csv(cf); self.diag = pd.read_csv(df) if df.exists() else pd.DataFrame()
            self._record("bridge_search", "reused", sf); return self.summary

        beams = []
        for d in self._source_direction_hypotheses():
            beams.append({"point": self.source_point.copy(), "direction": d, "path": [self.source_point.copy()],
                          "scores": [], "hard": [], "target_d": [float(np.linalg.norm((self.target_point-self.source_point)*self.spacing))],
                          "refsep": [_nearest_dist(self.source_point, np.vstack([self.reference,self.trunk]), self.spacing)], "turns": []})
        pool = []; diag = []; nsteps = int(math.ceil(max_bridge_mm / 0.30))
        for step_i in range(nsteps):
            step_mm = 0.28 if step_i < 8 else 0.32
            props = []
            for bi, st in enumerate(beams):
                target_dir = _unit((self.target_point - st["point"]) * self.spacing)
                dirs = []
                for base_w in (0.0, 0.18, 0.32):
                    base = _unit((1-base_w)*st["direction"] + base_w*target_dir)
                    for ang in (0, 10, 20, 30, 40):
                        if ang == 0:
                            dirs.append((base, 0.0))
                        else:
                            u, v = _orth_basis(base); aa = math.radians(ang)
                            for ph in np.linspace(0, 2*math.pi, 12, endpoint=False):
                                d = _unit(math.cos(aa)*base + math.sin(aa)*(math.cos(ph)*u + math.sin(ph)*v))
                                dirs.append((d, float(ang)))
                local = []
                for d, ang in dirs:
                    r = self._propose(st["point"], d, step_mm)
                    if r is None:
                        continue
                    td = float(np.linalg.norm((self.target_point - r["point"]) * self.spacing))
                    # Mild target attraction only; continuous source evidence remains primary.
                    if step_i >= 4 and td > st["target_d"][-1] + 0.55:
                        continue
                    smooth = float(np.clip(np.dot(st["direction"], r["direction"]), -1, 1))
                    obj = (0.48*r["score"] + 0.16*float(r["hard"]) + 0.15*((smooth+1)/2)
                           + 0.11*min(r["refsep"]/4.5,1) + 0.10*max(0.0, 1.0-td/7.0))
                    local.append((obj, r, ang, td))
                for obj, r, ang, td in local:
                    ns = {"point": r["point"], "direction": r["direction"], "path": st["path"]+[r["point"].copy()],
                          "scores": st["scores"]+[r["score"]], "hard": st["hard"]+[r["hard"]],
                          "target_d": st["target_d"]+[td], "refsep": st["refsep"]+[r["refsep"]], "turns": st["turns"]+[ang]}
                    props.append((obj, ns))
                    diag.append({"step_index": step_i, "beam_index": bi, "turn_deg": ang, "step_mm": r["step"],
                                 "plane_score": r["score"], "hard": r["hard"], "target_distance_mm": td,
                                 "reference_separation_mm": r["refsep"], "recenter_shift_mm": r["recenter_shift_mm"],
                                 "radius_mm": r["radius_mm"]})
            if not props:
                break
            def rank(item):
                _, st = item; path = np.asarray(st["path"]); L = float(arc_mm(path, self.spacing)[-1]); td = st["target_d"][-1]
                return 0.38*np.mean(st["scores"]) + 0.16*np.mean(st["hard"]) + 0.22*min(L/6.0,1) + 0.16*max(0,1-td/6.0) + 0.08*min(st["refsep"][-1]/4.5,1)
            props.sort(key=rank, reverse=True)
            keep = []
            for _, st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing) >= 0.30 for q in keep):
                    keep.append(st)
                if len(keep) >= beam_width:
                    break
            pool.extend(keep); beams = keep
            if min(st["target_d"][-1] for st in beams) <= 0.65:
                break

        rows = []; evaluated = []
        for st in pool:
            path = np.asarray(st["path"], float); L = float(arc_mm(path, self.spacing)[-1])
            if L < 2.0:
                continue
            qdf, qsum = _serial_qc(path, self.ct, self.spacing, self.rca, 0.36)
            td = float(st["target_d"][-1]); ref_end = float(st["refsep"][-1]); hard_steps = float(np.mean(st["hard"])) if st["hard"] else 0.0
            target_reached = bool(td <= 1.15)
            continuous = bool(L >= 4.5 and qsum["soft_pass_fraction"] >= 0.90 and qsum["plane_pass_fraction"] >= 0.82 and
                              qsum["median_plane_score"] >= 0.80 and qsum["max_consecutive_soft_failures"] <= 1 and hard_steps >= 0.78)
            score = (0.24*min(L/6.0,1) + 0.24*qsum["plane_pass_fraction"] + 0.20*qsum["median_plane_score"] +
                     0.15*max(0,1-td/5.0) + 0.09*hard_steps + 0.08*min(ref_end/4.0,1))
            row = {"candidate": len(rows)+1, "bridge_length_mm": L, "target_distance_mm": td, "target_reached": target_reached,
                   "plane_pass_fraction": qsum["plane_pass_fraction"], "soft_pass_fraction": qsum["soft_pass_fraction"],
                   "median_plane_score": qsum["median_plane_score"], "median_radius_mm": qsum["median_radius_mm"],
                   "median_recenter_shift_mm": qsum["median_recenter_shift_mm"], "max_consecutive_soft_failures": qsum["max_consecutive_soft_failures"],
                   "step_hard_fraction": hard_steps, "endpoint_reference_separation_mm": ref_end,
                   "max_turn_deg": max(st["turns"]) if st["turns"] else 0.0, "continuous_gate": continuous, "rank_score": score}
            rows.append(row); evaluated.append((score, row, qdf, qsum, path, st))
        self.diag = pd.DataFrame(diag); self.diag.to_csv(df, index=False)
        if not evaluated:
            self.bridge = np.asarray([self.source_point]); self.bridge_qc = pd.DataFrame(); self.candidates = pd.DataFrame()
            self.summary = {"status": "NO_CONTINUOUS_BRIDGE_FOUND", "accepted_bridge": False,
                            "interpretation": "No continuous coronary-like path longer than 2 mm could be traced from the fixed proximal source toward the recovered distal neighborhood."}
        else:
            evaluated.sort(key=lambda x: (x[1]["continuous_gate"], x[1]["target_reached"], x[0]), reverse=True)
            score, row, qdf, qsum, path, st = evaluated[0]
            self.bridge = resample_path(path, self.spacing, 0.20); self.bridge_qc = qdf
            self.candidates = pd.DataFrame(rows).sort_values(["continuous_gate", "target_reached", "rank_score"], ascending=[False,False,False])
            bridge_supported = bool(row["continuous_gate"] and row["target_reached"])
            alternative_supported = bool(row["continuous_gate"] and row["bridge_length_mm"] >= 5.5 and not row["target_reached"])
            status = "CONTINUOUS_GAP_BRIDGE_SUPPORTED" if bridge_supported else ("CONTINUOUS_ALTERNATIVE_COURSE_FOUND" if alternative_supported else "BRIDGE_REQUIRES_REVIEW")
            self.summary = {**row, "status": status, "accepted_bridge": bridge_supported,
                            "accepted_continuous_alternative": alternative_supported,
                            "interpretation": "A continuous component-by-component coronary-like path bridges the prior 14-17 mm gap to the recovered distal neighborhood." if bridge_supported else ("A continuous coronary-like alternative distal path was found, but it did not connect to the prior recovered distal target neighborhood." if alternative_supported else "Search found partial continuous trajectories but not enough evidence to bridge the prior serial-QC gap.")}
        pd.DataFrame({"arc_mm":arc_mm(self.bridge,self.spacing),"z":self.bridge[:,0],"y":self.bridge[:,1],"x":self.bridge[:,2]}).to_csv(pf,index=False)
        self.bridge_qc.to_csv(qf,index=False); self.candidates.to_csv(cf,index=False); _write_json(self.summary,sf)
        self._record("bridge_search", "recomputed_and_cached", sf, f"status={self.summary.get('status')}")
        return self.summary

    def make_figures(self):
        done = self.cache / "figures.done"
        names = ("01_source_target_context.png","02_bridge_cross_sections.png","03_bridge_vs_prior_course.png","04_bridge_diagnostics.png")
        if self.reuse["figures"] and done.exists() and all((self.out/n).exists() for n in names):
            self._record("figures","reused",done); return
        if self.summary is None:
            self.search_bridge()
        pq = self.prior_qc
        fig, ax = plt.subplots(figsize=(10,4)); ax.plot(pq.arc_mm, pq.plane_score, 'o-', label='prior relaunch plane score')
        ax.axvline(14.5, ls='--', lw=1); ax.axvline(17.1, ls='--', lw=1); ax.axhline(.80, ls=':', lw=1)
        ax.set_xlabel('prior relaunch arc (mm)'); ax.set_ylabel('plane score'); ax.set_title('Prior serial-QC gap and recovered distal neighborhood'); ax.legend(); fig.tight_layout(); fig.savefig(self.out/names[0],dpi=150); plt.close(fig)
        p = self.bridge; s = arc_mm(p,self.spacing); xs = np.linspace(0, s[-1], 12) if len(p)>1 else np.array([0.])
        fig, axs = plt.subplots(3,4,figsize=(14,10)); axs=axs.ravel()
        for ax,x in zip(axs,xs):
            i=int(np.argmin(abs(s-x))); i0,i1=max(0,i-4),min(len(p)-1,i+4); t=(p[i1]-p[i0])*self.spacing if i1>i0 else np.array([1.,0.,0.])
            im,c=orthogonal_plane(self.ct,p[i],t,self.spacing); ax.imshow(im,cmap='gray',vmin=-100,vmax=850,extent=[c[0],c[-1],c[-1],c[0]]); ax.axhline(0,lw=.5); ax.axvline(0,lw=.5); ax.set_title(f'{s[i]:.1f} mm'); ax.axis('off')
        fig.suptitle('Continuous gap-bridge candidate — source CCTA'); fig.tight_layout(); fig.savefig(self.out/names[1],dpi=150); plt.close(fig)
        ref=np.vstack([self.reference,self.trunk]); old=self.prior_course
        fig,axs=plt.subplots(1,3,figsize=(16,5))
        for ax,(a,b,title) in zip(axs,[(1,2,'axial projection'),(0,2,'coronal projection'),(0,1,'sagittal projection')]):
            ax.plot(ref[:,b],ref[:,a],lw=1,label='LAD/trunk'); ax.plot(old[:,b],old[:,a],lw=1,label='prior relaunch'); ax.plot(p[:,b],p[:,a],lw=2,label='continuous bridge')
            ax.scatter([self.source_point[b]],[self.source_point[a]],s=35,label='fixed source'); ax.scatter([self.target_point[b]],[self.target_point[a]],s=35,label='recovered target'); ax.set_title(title); ax.set_aspect('equal'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(self.out/names[2],dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4))
        if self.diag is not None and len(self.diag):
            beststep=self.diag.groupby('step_index').agg(plane_score=('plane_score','max'),target_distance_mm=('target_distance_mm','min'),recenter_shift_mm=('recenter_shift_mm','min')).reset_index()
            ax.plot(beststep.step_index,beststep.plane_score,label='best step plane score'); ax2=ax.twinx(); ax2.plot(beststep.step_index,beststep.target_distance_mm,label='min target distance'); ax.set_xlabel('search step'); ax.set_ylabel('plane score'); ax2.set_ylabel('target distance (mm)')
        ax.set_title('Bridge search continuity diagnostics'); fig.tight_layout(); fig.savefig(self.out/names[3],dpi=150); plt.close(fig)
        done.write_text('ok'); self._record('figures','recomputed_and_cached',done)

    def make_report(self):
        done=self.cache/'report.done'; html=self.out/'OPENPLAQUE_LCX_GAP_BRIDGE_REPORT.html'
        if self.reuse['report'] and done.exists() and html.exists():
            self._record('report','reused',html); return html
        if self.summary is None:
            self.search_bridge()
        if not (self.out/'01_source_target_context.png').exists():
            self.make_figures()
        _write_json(self.summary,self.out/'bridge_summary.json')
        self.bridge_qc.to_csv(self.out/'bridge_qc.csv',index=False); self.candidates.to_csv(self.out/'bridge_candidates.csv',index=False); self.diag.to_csv(self.out/'step_diagnostics.csv',index=False)
        pd.DataFrame({"arc_mm":arc_mm(self.bridge,self.spacing),"z":self.bridge[:,0],"y":self.bridge[:,1],"x":self.bridge[:,2]}).to_csv(self.out/'bridge_centerline.csv',index=False)
        rows=''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in self.summary.items() if not isinstance(v,(dict,list)))
        body=f'''<html><body><h1>OpenPlaque LCX Gap Bridge</h1><p>Algorithm: {ALGORITHM_VERSION}</p><p><b>Status: {self.summary.get('status')}</b></p><table border="1" cellspacing="0" cellpadding="4">{rows}</table><h2>Prior gap context</h2><img src="01_source_target_context.png" width="90%"><h2>Bridge cross-sections</h2><img src="02_bridge_cross_sections.png" width="95%"><h2>Bridge versus prior course/reference</h2><img src="03_bridge_vs_prior_course.png" width="95%"><h2>Search diagnostics</h2><img src="04_bridge_diagnostics.png" width="90%"><p>Research use only. This experiment tests continuity through the previously observed serial-QC gap and does not assign an LCX label automatically.</p></body></html>'''
        html.write_text(body); done.write_text('ok'); self._record('report','recomputed_and_cached',html); return html

    def package(self):
        self.make_report(); z=self.out/'OPENPLAQUE_LCX_GAP_BRIDGE_REPORT_BACK.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
            for p in self.out.iterdir():
                if p.is_file() and p != z:
                    q.write(p,p.name)
        return z
