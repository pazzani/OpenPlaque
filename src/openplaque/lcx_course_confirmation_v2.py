from __future__ import annotations

"""Robust continuation of the accepted secondary coronary branch.

This is a refinement of the LCX-course confirmation experiment.  The accepted
13.8-mm lateral-divergence branch is immutable.  Only its distal continuation is
searched.  Compared with v1, endpoint direction is estimated from several
look-back windows, the first few millimetres use smaller steps and a wider
turning cone, and a temporary reduction in LAD/trunk separation is permitted.
The workflow always returns a diagnostic result instead of raising when no
extension reaches an arbitrary minimum length.
"""

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .lcx_course_confirmation import (
    LCXCourseConfirmationWorkflow as _Base,
    _explicit_affine,
    _nearest_dist,
    _orth_basis,
    _serial_qc,
    _to_lps,
    _unit,
    _write_json,
    arc_mm,
    find_component,
    orthogonal_plane,
    resample_path,
    score_component,
)

ALGORITHM_VERSION = "lcx-course-confirmation-v1.1-endpoint-recovery"


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


class LCXCourseConfirmationWorkflow(_Base):
    def __init__(self, root="/content/drive/MyDrive/OpenPlaque", reuse=None):
        super().__init__(root=root, reuse=reuse)
        self.cache = self.root / "Cache" / "LCX_Course_Confirmation_v2"
        self.out = self.root / "LCX_Course_Confirmation_Report_v2"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.step_diagnostics = pd.DataFrame()

    def _propose_step(self, cur, direction, step_mm, rescue=False):
        guess = np.asarray(cur, float) + (float(step_mm) * _unit(direction)) / self.spacing
        if np.any(guess < 2) or np.any(guess >= np.asarray(self.ct.shape) - 3):
            return None
        im, c = orthogonal_plane(self.ct, guess, direction, self.spacing)
        m = find_component(im, c, self.rca, 3.0 if rescue else 2.0, rescue)
        if m is None:
            return None
        u, v = _orth_basis(direction)
        p = (guess * self.spacing + m["centroid_u_mm"] * u + m["centroid_v_mm"] * v) / self.spacing
        vec = (p - cur) * self.spacing
        L = float(np.linalg.norm(vec))
        if not (0.25 <= L <= (2.8 if rescue else 1.9)):
            return None
        sc, hp, sp = score_component(m, self.rca, rescue)
        if not sp:
            return None
        sep = _nearest_dist(p, np.vstack([self.reference, self.trunk]), self.spacing)
        return {"point": p, "direction": _unit(vec), "score": sc, "hard": hp,
                "soft": sp, "sep": sep, "step": L, **m}

    def _endpoint_direction_hypotheses(self, seed):
        s = arc_mm(seed, self.spacing)
        end = seed[-1]
        out = []
        for lookback_mm in (1.0, 1.8, 2.8, 4.2, 6.0):
            target = max(0.0, float(s[-1] - lookback_mm))
            i = int(np.argmin(abs(s - target)))
            if i >= len(seed) - 1:
                continue
            d = _unit((end - seed[i]) * self.spacing)
            if all(_angle_deg(d, q) >= 8.0 for q in out):
                out.append(d)
        # local PCA direction is useful when the last few points wobble around the lumen centroid
        keep = s >= max(0.0, s[-1] - 4.0)
        pts = seed[keep] * self.spacing
        if len(pts) >= 4:
            x = pts - pts.mean(axis=0)
            _, _, vh = np.linalg.svd(x, full_matrices=False)
            d = _unit(vh[0])
            if np.dot(d, _unit((seed[-1] - seed[max(0, len(seed)-4)]) * self.spacing)) < 0:
                d = -d
            if all(_angle_deg(d, q) >= 8.0 for q in out):
                out.append(d)
        return out or [_unit((seed[-1] - seed[-2]) * self.spacing)]

    def extend_course(self, max_extra_mm=24.0, beam_width=24):
        sf = self.cache / "course_summary.json"
        pf = self.cache / "combined_branch_centerline.csv"
        qf = self.cache / "combined_branch_qc.csv"
        cf = self.cache / "course_candidates.csv"
        dfp = self.cache / "course_step_diagnostics.csv"
        if self.seed is None:
            self.load_frozen_seed()
        if self.reuse["course_extension"] and all(p.exists() for p in (sf, pf, qf, cf)):
            self.summary = json.loads(sf.read_text())
            self.combined = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.qc = pd.read_csv(qf)
            self.candidates = pd.read_csv(cf)
            self.step_diagnostics = pd.read_csv(dfp) if dfp.exists() else pd.DataFrame()
            self._record("course_extension", "reused", sf)
            return self.summary

        seed = resample_path(self.seed, self.spacing, 0.22)
        ref = np.vstack([self.reference, self.trunk])
        sep0 = _nearest_dist(seed[-1], ref, self.spacing)
        dirs0 = self._endpoint_direction_hypotheses(seed)
        beams = [{"point": seed[-1].copy(), "direction": d, "ext": [seed[-1].copy()],
                  "scores": [], "hard": [], "seps": [sep0], "obj": 0.0,
                  "turns": []} for d in dirs0]
        pool = []
        diag = []
        nsteps = int(math.ceil(max_extra_mm / 0.45))
        termination = "MAX_STEPS"

        for step_i in range(nsteps):
            step_mm = 0.42 if step_i < 7 else (0.52 if step_i < 14 else 0.62)
            props = []
            for bi, st in enumerate(beams):
                # Broad early cone: the accepted branch can turn sharply at its distal endpoint.
                angles = (0, 18, 32, 48, 64) if step_i < 8 else (0, 14, 28, 42, 56)
                local = []
                for ang in angles:
                    if ang == 0:
                        dirs = [st["direction"]]
                    else:
                        u, v = _orth_basis(st["direction"])
                        a = math.radians(float(ang))
                        naz = 16 if step_i < 8 else 12
                        dirs = [_unit(math.cos(a) * st["direction"] + math.sin(a) *
                                      (math.cos(ph) * u + math.sin(ph) * v))
                                for ph in np.linspace(0, 2*math.pi, naz, endpoint=False)]
                    for d in dirs:
                        r = self._propose_step(st["point"], d, step_mm, rescue=False)
                        if r is None:
                            continue
                        extra0 = float(arc_mm(np.asarray(st["ext"]), self.spacing)[-1])
                        # Once a branch is independently established, a genuine circumflex course
                        # need not continue monotonically away from the LAD in Euclidean distance.
                        floor = 1.9 if extra0 < 2.0 else 2.15
                        if r["sep"] < floor:
                            continue
                        local.append((r, ang, False))
                if not local:
                    u, v = _orth_basis(st["direction"])
                    for ang in ((72, 90) if step_i < 10 else (68,)):
                        a = math.radians(float(ang))
                        for ph in np.linspace(0, 2*math.pi, 18, endpoint=False):
                            d = _unit(math.cos(a) * st["direction"] + math.sin(a) *
                                      (math.cos(ph) * u + math.sin(ph) * v))
                            r = self._propose_step(st["point"], d, max(0.34, step_mm-0.08), rescue=True)
                            if r is not None and r["sep"] >= 1.8:
                                local.append((r, ang, True))
                for r, ang, rescued in local:
                    smooth = float(np.clip(np.dot(st["direction"], r["direction"]), -1, 1))
                    sep_gain = float(r["sep"] - st["seps"][-1])
                    obj = (0.56*r["score"] + 0.13*float(r["hard"]) + 0.11*((smooth+1)/2)
                           + 0.14*min(r["sep"]/5.0, 1.0)
                           + 0.06*min(max(sep_gain+0.50, 0)/0.9, 1.0))
                    ns = {"point": r["point"], "direction": r["direction"],
                          "ext": st["ext"] + [r["point"].copy()],
                          "scores": st["scores"] + [r["score"]],
                          "hard": st["hard"] + [r["hard"]],
                          "seps": st["seps"] + [r["sep"]],
                          "obj": st["obj"] + obj, "turns": st["turns"] + [float(ang)]}
                    props.append(ns)
                    diag.append({"step_index": step_i, "beam_index": bi, "accepted_proposal": True,
                                 "rescue": rescued, "turn_deg": float(ang), "step_mm": float(r["step"]),
                                 "plane_score": float(r["score"]), "hard": bool(r["hard"]),
                                 "reference_separation_mm": float(r["sep"]),
                                 "recenter_shift_mm": float(r["recenter_shift_mm"])})
            if not props:
                termination = "NO_PROPOSALS"
                break

            def rank(st):
                ext = np.asarray(st["ext"])
                L = float(arc_mm(ext, self.spacing)[-1])
                return (0.38*np.mean(st["scores"]) + 0.16*np.mean(st["hard"])
                        + 0.27*min(L/16.0, 1.0) + 0.19*min(st["seps"][-1]/5.0, 1.0))

            props.sort(key=rank, reverse=True)
            keep = []
            for st in props:
                if all(np.linalg.norm((st["point"]-q["point"])*self.spacing) >= 0.45 for q in keep):
                    keep.append(st)
                if len(keep) >= beam_width:
                    break
            pool.extend(keep)
            beams = keep
            if not beams:
                termination = "BEAM_EMPTY"
                break
            if max(float(arc_mm(np.asarray(st["ext"]), self.spacing)[-1]) for st in beams) >= max_extra_mm:
                termination = "MAX_LENGTH"
                break

        self.step_diagnostics = pd.DataFrame(diag)
        self.step_diagnostics.to_csv(dfp, index=False)

        rows, evaluated = [], []
        for st in pool:
            ext = np.asarray(st["ext"], float)
            extra = float(arc_mm(ext, self.spacing)[-1])
            if extra < 1.0:
                continue
            combined = np.vstack([seed[:-1], ext])
            qdf, qsum = _serial_qc(combined, self.ct, self.spacing, self.rca, 0.85)
            eqdf, eqsum = _serial_qc(ext, self.ct, self.spacing, self.rca, 0.65)
            sep_end = float(st["seps"][-1]); sep_med = float(np.median(st["seps"]))
            score = (0.26*min(extra/12.0,1) + 0.24*eqsum["plane_pass_fraction"]
                     + 0.20*eqsum["median_plane_score"] + 0.14*min(sep_end/5.0,1)
                     + 0.08*min(sep_med/4.0,1) + 0.08*qsum["median_plane_score"])
            row = {"candidate": len(rows)+1, "extra_length_mm": extra,
                   "total_length_mm": qsum["length_mm"],
                   "combined_plane_pass_fraction": qsum["plane_pass_fraction"],
                   "combined_soft_pass_fraction": qsum["soft_pass_fraction"],
                   "combined_median_plane_score": qsum["median_plane_score"],
                   "extension_plane_pass_fraction": eqsum["plane_pass_fraction"],
                   "extension_soft_pass_fraction": eqsum["soft_pass_fraction"],
                   "extension_median_plane_score": eqsum["median_plane_score"],
                   "extension_median_radius_mm": eqsum["median_radius_mm"],
                   "endpoint_reference_separation_mm": sep_end,
                   "median_extension_reference_separation_mm": sep_med,
                   "max_turn_deg": float(max(st["turns"])) if st["turns"] else 0.0,
                   "rank_score": score}
            rows.append(row)
            evaluated.append((score, row, qsum, qdf, eqsum, combined, st))

        if not evaluated:
            qdf, qsum = _serial_qc(seed, self.ct, self.spacing, self.rca, 0.85)
            self.combined = seed
            self.qc = qdf
            self.candidates = pd.DataFrame()
            self.summary = {**qsum, "extra_length_mm": 0.0, "total_length_mm": qsum["length_mm"],
                            "status": "NO_PERSISTENT_EXTENSION", "accepted_topology": False,
                            "termination_reason": termination,
                            "interpretation": "The accepted 13.8-mm secondary branch remains valid, but this continuation search found no additional coronary-like course."}
        else:
            evaluated.sort(key=lambda x: x[0], reverse=True)
            score, row, qsum, qdf, eqsum, combined, st = evaluated[0]
            self.combined = resample_path(combined, self.spacing, 0.30)
            self.qc = qdf
            self.candidates = pd.DataFrame(rows).sort_values("rank_score", ascending=False)
            explicit_aff, key = _explicit_affine(self.meta or {})
            orient = {"patient_coordinate_status": "UNAVAILABLE", "affine_key": None}
            if explicit_aff is not None:
                lps = _to_lps(self.combined, explicit_aff)
                ss = arc_mm(self.combined, self.spacing)
                j = int(np.argmin(abs(ss - min(3.0, ss[-1]))))
                delta = lps[-1] - lps[j]
                orient = {"patient_coordinate_status": "AVAILABLE", "affine_key": key,
                          "delta_left_mm": float(delta[0]), "delta_posterior_mm": float(delta[1]),
                          "delta_superior_mm": float(delta[2]), "leftward_component": bool(delta[0] > 0),
                          "posterior_component": bool(delta[1] > -2.0)}
            continuity = bool(row["total_length_mm"] >= 20.0 and row["extra_length_mm"] >= 6.0
                              and row["extension_plane_pass_fraction"] >= 0.72
                              and row["extension_soft_pass_fraction"] >= 0.82
                              and row["extension_median_plane_score"] >= 0.76)
            independence = bool(row["endpoint_reference_separation_mm"] >= 3.0
                                and row["median_extension_reference_separation_mm"] >= 2.25)
            orientation_support = None
            if orient["patient_coordinate_status"] == "AVAILABLE":
                orientation_support = bool(orient["leftward_component"] and orient["posterior_component"])
            if continuity and independence and orientation_support is True:
                status = "LCX_COMPATIBLE_COURSE_SUPPORTED"
            elif continuity and independence:
                status = "LCX_TOPOLOGY_SUPPORTED_ORIENTATION_UNVERIFIED"
            else:
                status = "COURSE_REQUIRES_REVIEW"
            self.summary = {**qsum, **row, **orient, "continuity_gate": continuity,
                            "independence_gate": independence, "orientation_gate": orientation_support,
                            "status": status, "accepted_topology": bool(continuity and independence),
                            "termination_reason": termination,
                            "algorithm": ALGORITHM_VERSION,
                            "interpretation": "Accepted secondary branch continuation tested with endpoint-turn recovery; no LCX label is assigned solely from geometry."}

        pd.DataFrame({"arc_mm": arc_mm(self.combined, self.spacing), "z": self.combined[:,0],
                      "y": self.combined[:,1], "x": self.combined[:,2]}).to_csv(pf, index=False)
        self.qc.to_csv(qf, index=False)
        self.candidates.to_csv(cf, index=False)
        _write_json(self.summary, sf)
        self._record("course_extension", "recomputed_and_cached", sf,
                     f"status={self.summary.get('status')}; termination={termination}")
        return self.summary

    def make_report(self):
        done = self.cache / "report.done"
        html = self.out / "OPENPLAQUE_LCX_COURSE_CONFIRMATION_REPORT.html"
        if self.reuse["report"] and done.exists() and html.exists():
            self._record("report", "reused", html)
            return html
        if self.summary is None:
            self.extend_course()
        if not (self.out / "01_course_cross_sections.png").exists():
            self.make_figures()
        _write_json(self.summary, self.out / "course_summary.json")
        pd.DataFrame({"arc_mm": arc_mm(self.combined,self.spacing), "z":self.combined[:,0],
                      "y":self.combined[:,1], "x":self.combined[:,2]}).to_csv(self.out/"combined_branch_centerline.csv",index=False)
        self.qc.to_csv(self.out/"combined_branch_qc.csv",index=False)
        self.candidates.to_csv(self.out/"course_candidates.csv",index=False)
        if len(self.step_diagnostics): self.step_diagnostics.to_csv(self.out/"course_step_diagnostics.csv",index=False)
        rows = ''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in self.summary.items() if not isinstance(v,(dict,list)))
        body = f'''<html><body><h1>OpenPlaque LCX Course Confirmation</h1><p>Algorithm: {ALGORITHM_VERSION}</p><p><b>Status: {self.summary.get('status')}</b></p><table border="1" cellspacing="0" cellpadding="4">{rows}</table><h2>Orthogonal source-CCTA planes</h2><img src="01_course_cross_sections.png" width="95%"><h2>Course versus frozen LAD/trunk</h2><img src="02_course_vs_lad_mips.png" width="95%"><h2>Independence profile</h2><img src="03_course_profiles.png" width="80%"><p>Research use only. The workflow does not automatically assign an LCX label.</p></body></html>'''
        html.write_text(body); done.write_text('ok'); self._record('report','recomputed_and_cached',html); return html

    def package(self):
        self.make_report()
        z = self.out / "OPENPLAQUE_LCX_COURSE_CONFIRMATION_REPORT_BACK.zip"
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as q:
            for p in self.out.iterdir():
                if p.is_file() and p != z:
                    q.write(p, p.name)
        return z
