from __future__ import annotations

"""Safe multi-objective selection for LAD takeoff root alternatives v1.1.

The v1 search generated the intended root-directed trajectories but used Python
container equality on dictionaries containing NumPy arrays while assembling the
shortlist. This subclass preserves the algorithm and replaces shortlist
membership with identity-safe bookkeeping. All downstream QC/report behavior is
inherited unchanged.
"""

import math
import numpy as np
import pandas as pd

from .lad_takeoff_root_alternatives import (
    LADTakeoffRootAlternativesWorkflow as _Base,
    _angle_deg,
    _unit,
    arc_mm,
    resample_path,
    serial_qc,
)

ALGORITHM_VERSION = "lad-takeoff-root-alternatives-v1.1-selection-fix"


class LADTakeoffRootAlternativesWorkflow(_Base):
    def search_alternatives(self, max_extension_mm=28.0, beam_width=14, n_alternatives=4):
        sf = self.cache / "alternative_summary.csv"
        if self.lad is None:
            self.validate_lad_backbone()
        if self.aorta_dist is None:
            self.build_aorta_constraint()
        if self.reuse["alternatives"] and sf.exists():
            summary = pd.read_csv(sf)
            alts = []
            for _, r in summary.iterrows():
                fp = self.cache / f"alternative_{int(r.alternative_id):02d}_centerline.csv"
                qf = self.cache / f"alternative_{int(r.alternative_id):02d}_qc.csv"
                if fp.exists() and qf.exists():
                    alts.append({"id": int(r.alternative_id), "path": pd.read_csv(fp)[["z", "y", "x"]].to_numpy(float), "qc": pd.read_csv(qf), "summary": r.to_dict()})
            if alts:
                self.alternatives, self.alternative_summary = alts, summary
                self._record("alternatives", "reused", sf)
                return summary
        if not bool(self.lad_summary.get("accepted", False)):
            raise RuntimeError("Prior LAD backbone failed revalidation gate")

        p = resample_path(self.lad, self.spacing, 0.35)
        anchor = p[0].copy()
        j = min(len(p) - 1, max(5, int(round(4.0 / 0.35))))
        init_dir = -_unit((p[j] - p[0]) * self.spacing)
        ad0, _ = self._aorta_distance(anchor)
        init = {"point": anchor, "direction": init_dir, "path": [anchor.copy()], "scores": [], "hard": [], "soft": [], "shifts": [], "aorta": [ad0], "obj": 0.0}
        beams = [init]
        pool = []
        nsteps = int(math.ceil(float(max_extension_mm) / 0.75))

        from .lad_takeoff_root_alternatives import _cone
        for _step in range(nsteps):
            props = []
            for st in beams:
                cur = st["point"]
                ad = st["aorta"][-1]
                root_dir = self._aorta_gradient_dir(cur)
                bases = [st["direction"]]
                if root_dir is not None:
                    for w in (0.18, 0.35, 0.52, 0.68):
                        bases.append(_unit((1.0 - w) * _unit(st["direction"]) + w * root_dir))
                dirs = []
                for b in bases:
                    dirs.extend(_cone(b, (0, 12, 24, 36 if ad > 10 else 48), 6))
                unique = []
                for d in dirs:
                    if all(_angle_deg(d, q) >= 7.0 for q in unique):
                        unique.append(d)
                near_root = bool(np.isfinite(ad) and ad <= 12.0)
                for d in unique:
                    for r in self._propose(cur, d, near_root):
                        smooth = float(np.clip(np.dot(_unit(st["direction"]), _unit(r["direction"])), -1, 1))
                        prev_ad = st["aorta"][-1]
                        prog = float(np.clip((prev_ad - r["aorta_distance_mm"]) / 1.2, -1, 1)) if np.isfinite(prev_ad) and np.isfinite(r["aorta_distance_mm"]) else 0.0
                        root_weight = 0.18 if near_root else 0.11
                        local = (0.54 - root_weight / 2) * r["plane_score"] + 0.13 * float(r["hard_pass"]) + 0.12 * ((smooth + 1) / 2) + 0.10 * math.exp(-0.5 * (r["recenter_shift_mm"] / 1.05) ** 2) + root_weight * ((prog + 1) / 2)
                        props.append({
                            "point": r["point"], "direction": r["direction"], "path": st["path"] + [r["point"].copy()],
                            "scores": st["scores"] + [r["plane_score"]], "hard": st["hard"] + [r["hard_pass"]], "soft": st["soft"] + [r["soft_pass"]],
                            "shifts": st["shifts"] + [r["recenter_shift_mm"]], "aorta": st["aorta"] + [r["aorta_distance_mm"]], "obj": st["obj"] + local,
                        })
            pool.extend(beams)
            if not props:
                break

            def rank(st):
                n = max(1, len(st["scores"]))
                L = arc_mm(np.asarray(st["path"]), self.spacing)[-1]
                gain = ad0 - st["aorta"][-1] if np.isfinite(ad0) and np.isfinite(st["aorta"][-1]) else 0.0
                return st["obj"] / n + 0.10 * np.mean(st["hard"]) + 0.005 * min(L, 24) + 0.004 * max(gain, 0)

            props.sort(key=rank, reverse=True)
            keep = []
            for st in props:
                if all(np.linalg.norm((st["point"] - q["point"]) * self.spacing) >= 0.7 or _angle_deg(st["direction"], q["direction"]) >= 13 for q in keep):
                    keep.append(st)
                if len(keep) >= int(beam_width):
                    break
            beams = keep
            if any(np.isfinite(st["aorta"][-1]) and st["aorta"][-1] <= 3.0 and np.mean(st["hard"][-4:]) >= 0.75 for st in beams):
                pool.extend(beams)
                break
        pool.extend(beams)

        cand = []
        for st in pool:
            if len(st["path"]) < 8:
                continue
            path = resample_path(np.asarray(st["path"], float)[::-1], self.spacing, 0.35)
            qdf, qsum = serial_qc(path, self.ct, self.spacing, self.rca_cal, 1.1, "ROOT_ALT")
            L = float(qsum["length_mm"])
            if L < 5.0:
                continue
            end_ad, _ = self._aorta_distance(path[0])
            gain = ad0 - end_ad if np.isfinite(ad0) and np.isfinite(end_ad) else 0.0
            quality = 0.55 * qsum["plane_pass_fraction"] + 0.45 * qsum["median_plane_score"]
            shift = float(qsum["median_recenter_shift_mm"])
            if not np.isfinite(shift):
                shift = 3.0
            balanced = 0.46 * quality + 0.25 * min(max(gain, 0) / 20.0, 1) + 0.18 * min(L / 20.0, 1) + 0.11 * math.exp(-0.5 * (shift / 0.75) ** 2)
            cand.append({"state": st, "path": path, "qc": qdf, "qsum": qsum, "endpoint_aorta_distance_mm": end_ad, "aorta_gain_mm": gain, "quality": quality, "balanced": balanced})
        if not cand:
            raise RuntimeError("No root-directed LAD alternative survived serial source-resolution QC")

        # Identity-safe multi-objective shortlist.
        ordered = []
        seen = set()
        for key, rev in (("balanced", True), ("quality", True), ("endpoint_aorta_distance_mm", False), ("aorta_gain_mm", True)):
            for x in sorted(cand, key=lambda z: z[key], reverse=rev):
                xid = id(x)
                if xid not in seen:
                    ordered.append(x)
                    seen.add(xid)
                    break
        for x in sorted(cand, key=lambda z: z["balanced"], reverse=True):
            xid = id(x)
            if xid not in seen:
                ordered.append(x)
                seen.add(xid)

        selected = []
        for x in ordered:
            ep = x["path"][0]
            tan = _unit((x["path"][min(len(x["path"]) - 1, 8)] - x["path"][0]) * self.spacing)
            distinct = True
            for y in selected:
                ytan = _unit((y["path"][min(len(y["path"]) - 1, 8)] - y["path"][0]) * self.spacing)
                if np.linalg.norm((ep - y["path"][0]) * self.spacing) < 1.6 and _angle_deg(tan, ytan) < 18:
                    distinct = False
                    break
            if distinct:
                selected.append(x)
            if len(selected) >= int(n_alternatives):
                break
        if not selected:
            selected = [max(cand, key=lambda z: z["balanced"])]

        rows, alts = [], []
        for i, x in enumerate(selected, 1):
            fp = self.cache / f"alternative_{i:02d}_centerline.csv"
            qf = self.cache / f"alternative_{i:02d}_qc.csv"
            pd.DataFrame({"arc_mm": arc_mm(x["path"], self.spacing), "z": x["path"][:, 0], "y": x["path"][:, 1], "x": x["path"][:, 2]}).to_csv(fp, index=False)
            x["qc"].to_csv(qf, index=False)
            row = {
                "alternative_id": i,
                "length_mm": x["qsum"]["length_mm"],
                "plane_pass_fraction": x["qsum"]["plane_pass_fraction"],
                "soft_pass_fraction": x["qsum"]["soft_pass_fraction"],
                "median_plane_score": x["qsum"]["median_plane_score"],
                "median_radius_mm": x["qsum"]["median_radius_mm"],
                "median_recenter_shift_mm": x["qsum"]["median_recenter_shift_mm"],
                "endpoint_aorta_distance_mm": x["endpoint_aorta_distance_mm"],
                "aorta_distance_reduction_mm": x["aorta_gain_mm"],
                "quality_score": x["quality"],
                "balanced_score": x["balanced"],
            }
            rows.append(row)
            alts.append({"id": i, "path": x["path"], "qc": x["qc"], "summary": row})
        summary = pd.DataFrame(rows)
        summary.to_csv(sf, index=False)
        self.alternatives, self.alternative_summary = alts, summary
        self._record("alternatives", "recomputed_and_cached", sf, f"selected {len(alts)} distinct root-directed alternatives; {ALGORITHM_VERSION}")
        return summary
