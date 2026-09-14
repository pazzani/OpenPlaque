from __future__ import annotations

"""Bug-fixed 3-D vesselness/topology continuation.

This module preserves the scientific logic and thresholds of
secondary_3d_vesselness_topology.py and fixes only the Dijkstra
flat-index -> (z,y,x) conversion used to enumerate reachable nodes.
"""

import json
import numpy as np
import pandas as pd

from . import secondary_3d_vesselness_topology as base

ALGORITHM_VERSION = "secondary-3d-vesselness-topology-v1.1-indexfix"


def finite_dist_coords(dist, shape):
    """Return reachable Dijkstra nodes as integer (z,y,x) coordinates."""
    flat_ids = np.flatnonzero(np.isfinite(np.asarray(dist)))
    if flat_ids.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    coords = np.column_stack(np.unravel_index(flat_ids, tuple(shape))).astype(np.int64, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"Expected reachable coordinates with shape (N,3), got {coords.shape}")
    shape_arr = np.asarray(shape, dtype=np.int64)
    if np.any(coords < 0) or np.any(coords >= shape_arr[None, :]):
        raise RuntimeError("Unravelled reachable coordinate fell outside ROI bounds")
    return coords


class Secondary3DVesselnessTopologyWorkflow(base.Secondary3DVesselnessTopologyWorkflow):
    """Same workflow as v1.0, with corrected reachable-node coordinate conversion."""

    def search_topology(self, min_candidate_projection_mm=1.5, max_cost=150.0):
        sf = self.cache / "topology_summary.json"
        cf = self.cache / "topology_candidates.csv"
        pf = self.cache / "control_path.csv"
        ef = self.cache / "extension_path.csv"
        ff = self.cache / "full_path.csv"

        if self.vesselness is None:
            self.compute_vesselness()

        if self.reuse["topology_search"] and all(p.exists() for p in (sf, cf, pf, ef, ff)):
            self.summary = json.loads(sf.read_text())
            self.candidates = pd.read_csv(cf)
            self.control_path = pd.read_csv(pf)[["z", "y", "x"]].to_numpy(float)
            self.extension_path = pd.read_csv(ef)[["z", "y", "x"]].to_numpy(float)
            self.full_path = pd.read_csv(ff)[["z", "y", "x"]].to_numpy(float)
            self._record("topology_search", "reused", sf)
            return self.summary

        mask, cost = self._build_mask_and_cost()
        self.control_path, csum = self._positive_control(mask, cost)

        if not csum["control_pass"]:
            self.extension_path = np.asarray([self.control_point])
            self.full_path = self.control_path if self.control_path is not None else np.asarray([self.start_point])
            self.candidates = pd.DataFrame()
            self.summary = {
                "algorithm": ALGORITHM_VERSION,
                "status": "POSITIVE_CONTROL_FAILED",
                "accepted_continuation": False,
                **csum,
                "interpretation": (
                    "3-D vesselness topology search could not reliably rediscover the known branch "
                    "segment; distal inference is invalid."
                ),
            }
        else:
            start = self._nearest_voxel(self.control_point)
            ext_mask = mask.copy()
            ext_mask[self._sphere_mask(self.control_point, 0.60)] = True
            dist, prev, _ = self._dijkstra(ext_mask, cost, start, goal_mask=None, max_cost=max_cost)

            # v1.0 bug: np.nonzero(np.isfinite(dist)) returns one column of FLAT IDs
            # because dist is 1-D. Those IDs were then used as axis-0 indices into
            # a 3-D vesselness array. Convert flat IDs explicitly back to z,y,x.
            coords = finite_dist_coords(dist, self.roi.shape)

            if len(coords):
                phys = (coords + self.roi_lo) * self.spacing
                cp = self.control_point * self.spacing
                proj = (phys - cp) @ self.terminal_tangent
                disp = np.linalg.norm(phys - cp, axis=1)
                vals = self.vesselness[tuple(coords.T)]
                ok = (
                    (proj >= min_candidate_projection_mm)
                    & (disp >= 1.8)
                    & (vals >= self.vessel_threshold)
                )
                coords = coords[ok]
                proj = proj[ok]
                vals = vals[ok]

            rows = []
            paths = []
            if len(coords):
                score = proj + 1.5 * vals
                order = np.argsort(score)[::-1]
                chosen = []
                for ii in order:
                    g = coords[ii] + self.roi_lo
                    if all(np.linalg.norm((g - q) * self.spacing) >= 1.0 for q in chosen):
                        chosen.append(g.copy())
                        flat = np.ravel_multi_index(tuple(coords[ii]), self.roi.shape)
                        loc = self._reconstruct(prev, flat, self.roi.shape)
                        if loc is None or len(loc) < 2:
                            continue

                        path = base.resample_path(self._local_to_global(loc), self.spacing, 0.20)
                        m = self._candidate_metrics(path)

                        continuous = bool(
                            m["new_length_mm"] >= 2.0
                            and m["endpoint_displacement_mm"] >= 1.8
                            and m["forward_projection_mm"] >= 1.5
                            and m["tortuosity"] <= 2.0
                            and m["max_turn_deg"] <= 75.0
                            and m["p10_vesselness"] >= 0.65 * self.vessel_threshold
                            and m["endpoint_reference_separation_mm"] >= 1.7
                        )
                        supported = bool(
                            continuous
                            and m["new_length_mm"] >= 4.0
                            and m["endpoint_displacement_mm"] >= 3.0
                            and m["forward_projection_mm"] >= 2.5
                            and m["endpoint_old_seed_separation_mm"] >= 1.0
                            and m["p90_best_scale_mm"] <= 1.50
                        )
                        rank = (
                            0.24 * min(m["new_length_mm"] / 6.0, 1.0)
                            + 0.22 * min(m["forward_projection_mm"] / 5.0, 1.0)
                            + 0.20
                            * min(
                                m["median_vesselness"] / max(self.vessel_threshold * 2.0, 1e-4),
                                1.0,
                            )
                            + 0.12 * min(m["endpoint_displacement_mm"] / 5.0, 1.0)
                            + 0.12 * (1.0 / max(m["tortuosity"], 1.0))
                            + 0.10 * min(m["endpoint_old_seed_separation_mm"] / 2.0, 1.0)
                        )
                        row = {
                            "candidate": len(rows) + 1,
                            **m,
                            "continuous_gate": continuous,
                            "supported_continuation": supported,
                            "rank_score": float(rank),
                        }
                        rows.append(row)
                        paths.append((row, path))
                        if len(rows) >= 24:
                            break

            self.candidates = pd.DataFrame(rows)

            if len(self.candidates):
                self.candidates = self.candidates.sort_values(
                    ["supported_continuation", "continuous_gate", "rank_score"],
                    ascending=[False, False, False],
                )
                best_id = int(self.candidates.iloc[0]["candidate"])
                row, path = next(
                    (r, p) for r, p in paths if int(r["candidate"]) == best_id
                )
                self.extension_path = path
                self.full_path = base.resample_path(
                    np.vstack([self.control_path, self.extension_path[1:]]),
                    self.spacing,
                    0.20,
                )

                if bool(row["supported_continuation"]):
                    status = "SUPPORTED_3D_TOPOLOGIC_EXTENSION"
                    interp = (
                        "Positive control passed and a >=4 mm geometrically new 3-D "
                        "vesselness-supported continuation was found; vessel identity remains unassigned."
                    )
                elif bool(row["continuous_gate"]):
                    status = "PARTIAL_3D_TOPOLOGIC_EXTENSION"
                    interp = (
                        "Positive control passed and a multi-mm 3-D tubular corridor was found, "
                        "but it did not satisfy the full new-course support gate."
                    )
                else:
                    status = "NO_SUPPORTED_3D_TOPOLOGIC_EXTENSION"
                    interp = (
                        "Positive control passed, but reachable distal 3-D vesselness paths "
                        "failed continuity/geometry gates."
                    )

                self.summary = {
                    "algorithm": ALGORITHM_VERSION,
                    "status": status,
                    "accepted_continuation": bool(row["supported_continuation"]),
                    **csum,
                    **row,
                    "interpretation": interp,
                }
            else:
                self.extension_path = np.asarray([self.control_point])
                self.full_path = self.control_path
                self.summary = {
                    "algorithm": ALGORITHM_VERSION,
                    "status": "NO_3D_TOPOLOGIC_EXTENSION",
                    "accepted_continuation": False,
                    **csum,
                    "interpretation": (
                        "Positive control passed, but no target-free distal node >=1.5 mm "
                        "forward was reachable through the adaptive 3-D vesselness graph."
                    ),
                }

        self.candidates.to_csv(cf, index=False)
        for path_file, arr in [
            (pf, self.control_path),
            (ef, self.extension_path),
            (ff, self.full_path),
        ]:
            arr = np.asarray(arr, float)
            pd.DataFrame(
                {
                    "arc_mm": base.arc_mm(arr, self.spacing),
                    "z": arr[:, 0],
                    "y": arr[:, 1],
                    "x": arr[:, 2],
                }
            ).to_csv(path_file, index=False)

        base._write_json(self.summary, sf)
        self._record(
            "topology_search",
            "computed_indexfix_v1_1",
            sf,
            note="Fixed Dijkstra flat-index to 3-D ROI-coordinate conversion; scientific gates unchanged.",
        )
        return self.summary
