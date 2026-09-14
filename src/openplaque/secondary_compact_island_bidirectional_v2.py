"""Runtime bugfix for compact-island bidirectional validation.

Scientific logic is unchanged from secondary_compact_island_bidirectional.py.
This wrapper only fixes a local variable named `pd` that shadowed the pandas alias
at the return from _track_one().
"""

import json
import math

import numpy as np
import pandas as pd

from .secondary_compact_island_bidirectional import (
    SecondaryCompactIslandBidirectionalWorkflow as _V1,
    _angle_deg,
    _compact_component,
    _direction_fan,
    _plane_memmap,
    _unit,
)


class SecondaryCompactIslandBidirectionalWorkflow(_V1):
    def _track_one(self, sign, max_length_mm=3.0, step_mm=0.20):
        label = "forward" if sign > 0 else "backward"
        cur = self.seed_point.copy()
        cur_dir = self.seed_tangent.copy() * float(sign)
        pts = [cur.copy()]
        rows = []
        terminal_rows = []
        cum = 0.0
        arc0, dist0 = self._posthoc_arc(cur)
        rows.append({
            "direction": label, "step_index": 0, "track_length_mm": 0.0,
            "z": cur[0], "y": cur[1], "x": cur[2],
            "radius_mm": np.nan, "centroid_shift_mm": np.nan,
            "plane_score": np.nan, "turn_deg": 0.0, "actual_step_mm": 0.0,
            "posthoc_nearest_arc_mm": arc0,
            "posthoc_distance_to_accepted_mm": dist0,
        })
        step_index = 0
        while cum + 0.5 * step_mm <= max_length_mm:
            proposals = []
            reject = {}
            for d in _direction_fan(cur_dir):
                nominal_phys = cur * self.spacing + float(step_mm) * d
                nominal = nominal_phys / self.spacing
                if np.any(nominal < 2.0) or np.any(nominal >= np.asarray(self.ct.shape) - 3.0):
                    reject["outside_volume"] = reject.get("outside_volume", 0) + 1
                    continue
                im, g, u, v = _plane_memmap(self.ct, nominal, d, self.spacing, 3.8, 0.16)
                m0 = _compact_component(im, g, self.calibration["bright_threshold_hu"], 0.95, compact_only=False)
                if m0 is None:
                    reject["no_component"] = reject.get("no_component", 0) + 1
                    continue
                if not (0.65 <= m0["radius_mm"] <= 2.65):
                    reject["radius_too_large_or_small"] = reject.get("radius_too_large_or_small", 0) + 1
                    continue
                corrected_phys = nominal_phys + m0["offset_u_mm"] * u + m0["offset_v_mm"] * v
                corrected = corrected_phys / self.spacing
                vec = corrected_phys - cur * self.spacing
                actual_step = float(np.linalg.norm(vec))
                actual_dir = _unit(vec)
                if not (0.10 <= actual_step <= 0.45):
                    reject["step_length"] = reject.get("step_length", 0) + 1
                    continue
                turn = _angle_deg(cur_dir, actual_dir)
                if turn > 48.0:
                    reject["turn_gt_48"] = reject.get("turn_gt_48", 0) + 1
                    continue
                if float(np.dot(actual_dir, cur_dir)) < 0.60:
                    reject["insufficient_forward_progress"] = reject.get("insufficient_forward_progress", 0) + 1
                    continue
                im2, g2, _, _ = _plane_memmap(self.ct, corrected, actual_dir, self.spacing, 3.8, 0.16)
                m = _compact_component(im2, g2, self.calibration["bright_threshold_hu"], 0.70, compact_only=False)
                passed, score = self._score_measure(m)
                if m is None:
                    reject["no_component_after_recenter"] = reject.get("no_component_after_recenter", 0) + 1
                    continue
                if not passed:
                    reason = "radius_too_large_or_small" if not (0.65 <= m["radius_mm"] <= 2.65) else "lumen_gate"
                    reject[reason] = reject.get(reason, 0) + 1
                    continue
                continuity = math.exp(-turn / 35.0)
                rank = 0.82 * score + 0.18 * continuity
                proposals.append((rank, corrected, actual_dir, m, score, turn, actual_step))
            if not proposals:
                terminal_rows.append({
                    "direction": label,
                    "terminal_step_index": step_index + 1,
                    "track_length_mm": cum,
                    "rejection_counts_json": json.dumps(reject, sort_keys=True),
                    "n_directions_tested": len(_direction_fan(cur_dir)),
                })
                break
            _, pnew, dnew, m, score, turn, actual_step = max(proposals, key=lambda x: x[0])
            if len(pts) > 4:
                old = np.asarray(pts[:-2]) * self.spacing
                if np.min(np.linalg.norm(old - pnew * self.spacing, axis=1)) < 0.30:
                    terminal_rows.append({
                        "direction": label,
                        "terminal_step_index": step_index + 1,
                        "track_length_mm": cum,
                        "rejection_counts_json": json.dumps({"loop_guard": 1}),
                        "n_directions_tested": len(_direction_fan(cur_dir)),
                    })
                    break
            cur = pnew
            cur_dir = dnew
            cum += actual_step
            step_index += 1
            pts.append(cur.copy())
            posthoc_arc, posthoc_dist = self._posthoc_arc(cur)
            rows.append({
                "direction": label, "step_index": step_index,
                "track_length_mm": cum, "z": cur[0], "y": cur[1], "x": cur[2],
                "radius_mm": m["radius_mm"],
                "centroid_shift_mm": m["centroid_shift_mm"],
                "circularity": m["circularity"],
                "component_median_hu": m["component_median_hu"],
                "plane_score": score, "turn_deg": turn,
                "actual_step_mm": actual_step,
                "posthoc_nearest_arc_mm": posthoc_arc,
                "posthoc_distance_to_accepted_mm": posthoc_dist,
            })
        return pd.DataFrame(rows), pd.DataFrame(terminal_rows), np.asarray(pts, float)
