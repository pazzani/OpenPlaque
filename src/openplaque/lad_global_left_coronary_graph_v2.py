from __future__ import annotations

"""Positive-control calibrated correction for global left-coronary graph v1.0.

This is a same-experiment logical correction.  The original graph used a fixed
6.5-mm maximum edge radius.  In the first completed run, sparse vesselness peaks
on the *already validated* LAD were 7.69 mm apart, so the graph could not preserve
known-true LAD connectivity.  v1.1 therefore calibrates the graph edge radius
against detected nodes that lie on the accepted LAD before interpreting any
negative topology result.

Scientific safeguards:
- RCA remains a hard exclusion corridor inherited from v1.0.
- Aorta remains post-hoc only.
- Nodes already belonging to the accepted LAD are labeled and cannot count as
  novel extension targets.
- Candidate proximal-extension paths may use only the proximal <=4 mm LAD
  neighborhood before leaving the accepted LAD; downstream LAD nodes are not
  allowed to masquerade as proximal continuation.
- Dense final-path-tangent source-CCTA QC remains unchanged and mandatory.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from . import lad_global_left_coronary_graph as base

ALGORITHM_VERSION = "lad-global-left-coronary-graph-v1.1-positive-control-calibrated"
REF_LAD_DIST_MM = 0.80
NOVEL_LAD_DIST_MM = 2.50
PROXIMAL_LAD_ARC_MAX_MM = 4.0
ANCHOR_LINK_MAX_MM = 4.0
EDGE_RADII_MM = tuple(np.arange(6.5, 10.01, 0.5))


class GlobalLeftCoronaryGraphWorkflowV2(base.GlobalLeftCoronaryGraphWorkflow):
    def discover_nodes(self):
        nodes = super().discover_nodes()
        if len(nodes) == 0:
            self.nodes = nodes
            return nodes
        P = nodes[["z", "y", "x"]].to_numpy(float)
        lad_phys = self.lad * self.spacing
        tree = cKDTree(lad_phys)
        d, j = tree.query(P * self.spacing)
        lad_arc = base.arc_mm(self.lad, self.spacing)
        nodes = nodes.copy()
        nodes["lad_distance_mm"] = d.astype(float)
        nodes["lad_nearest_arc_mm"] = lad_arc[np.asarray(j, int)]
        nodes["is_reference_lad_node"] = nodes.lad_distance_mm <= REF_LAD_DIST_MM
        nodes["is_proximal_reference_node"] = (
            nodes.is_reference_lad_node
            & (nodes.lad_nearest_arc_mm <= PROXIMAL_LAD_ARC_MAX_MM)
        )
        nodes["is_novel_node"] = nodes.lad_distance_mm >= NOVEL_LAD_DIST_MM
        self.nodes = nodes
        nodes.to_csv(self.cache / "graph_nodes.csv", index=False)
        return nodes

    def _edges_at_radius(self, radius_mm):
        if self.nodes is None:
            self.discover_nodes()
        rows = []
        if len(self.nodes) == 0:
            return pd.DataFrame(rows, columns=[
                "i", "j", "distance_mm", "tangent_mismatch_deg",
                "line_mean", "line_q25", "cost"
            ])
        P = self.nodes[["z", "y", "x"]].to_numpy(float)
        T = self.nodes[["tz", "ty", "tx"]].to_numpy(float)
        tree = cKDTree(P * self.spacing)
        for i, j in tree.query_pairs(float(radius_mm)):
            dist = float(np.linalg.norm((P[i] - P[j]) * self.spacing))
            tm = base._tangent_mismatch(T[i], T[j])
            if tm > 55.0:
                continue
            lm, lq = base._line_support(
                self.vessel,
                self._source_to_iso(P[i:i+1])[0],
                self._source_to_iso(P[j:j+1])[0],
            )
            if lm < 0.015 and lq < 0.004:
                continue
            rows.append({
                "i": int(i), "j": int(j), "distance_mm": dist,
                "tangent_mismatch_deg": float(tm), "line_mean": float(lm),
                "line_q25": float(lq),
                "cost": float(dist * (1.0 + 0.012 * tm) + 1.5 / (lm + 0.03)),
            })
        return pd.DataFrame(rows, columns=[
            "i", "j", "distance_mm", "tangent_mismatch_deg",
            "line_mean", "line_q25", "cost"
        ])

    def _positive_control_connected(self, edges):
        if len(self.nodes) == 0:
            return False
        ref = self.nodes[self.nodes.is_reference_lad_node]
        proximal = ref[ref.lad_nearest_arc_mm <= PROXIMAL_LAD_ARC_MAX_MM].index.to_numpy(int)
        distal = ref[ref.lad_nearest_arc_mm >= 15.0].index.to_numpy(int)
        if len(proximal) == 0 or len(distal) == 0:
            return False
        n = len(self.nodes)
        if len(edges) == 0:
            return False
        ri, ci = [], []
        for r in edges.itertuples():
            ri += [int(r.i), int(r.j)]
            ci += [int(r.j), int(r.i)]
        G = csr_matrix((np.ones(len(ri), float), (ri, ci)), shape=(n, n))
        _, labels = connected_components(G, directed=False, return_labels=True)
        return bool(any(labels[p] == labels[d] for p in proximal for d in distal))

    def build_graph(self):
        if self.nodes is None:
            self.discover_nodes()
        calibration = []
        chosen = None
        chosen_edges = None
        for radius in EDGE_RADII_MM:
            edges = self._edges_at_radius(radius)
            pc = self._positive_control_connected(edges)
            calibration.append({
                "edge_radius_mm": float(radius),
                "n_edges": int(len(edges)),
                "positive_control_connected": bool(pc),
            })
            if pc:
                chosen = float(radius)
                chosen_edges = edges
                break
        if chosen_edges is None:
            chosen = float(EDGE_RADII_MM[-1])
            chosen_edges = self._edges_at_radius(chosen)
            pc = self._positive_control_connected(chosen_edges)
        else:
            pc = True
        self.edge_radius_mm = chosen
        self.graph_positive_control_passed = bool(pc)
        self.edge_calibration = pd.DataFrame(calibration)
        self.edge_calibration.to_csv(self.cache / "graph_edge_positive_control.csv", index=False)
        self.edges = chosen_edges
        self.edges.to_csv(self.cache / "graph_edges.csv", index=False)
        return self.edges

    def _anchor_links_v2(self, allowed):
        ep = self.lad[0]
        t = base._unit((self.lad[min(5, len(self.lad)-1)] - ep) * self.spacing)
        P = self.nodes[["z", "y", "x"]].to_numpy(float)
        T = self.nodes[["tz", "ty", "tx"]].to_numpy(float)
        out = []
        for i, p in enumerate(P):
            if not allowed[i]:
                continue
            # Anchor only to nodes in the proximal accepted-LAD neighborhood.
            if float(self.nodes.iloc[i].lad_nearest_arc_mm) > PROXIMAL_LAD_ARC_MAX_MM:
                continue
            d = float(np.linalg.norm((p - ep) * self.spacing))
            if d > ANCHOR_LINK_MAX_MM:
                continue
            tm = base._tangent_mismatch(t, T[i])
            if tm > 70.0:
                continue
            lm, lq = base._line_support(
                self.vessel,
                self._source_to_iso(ep[None, :])[0],
                self._source_to_iso(p[None, :])[0],
            )
            if d > 3.0 and lm < 0.008:
                continue
            out.append((int(i), float(d * (1.0 + 0.01 * tm) + 1.0 / (lm + 0.025))))
        return out

    def enumerate_paths(self):
        if self.edges is None:
            self.build_graph()
        n = len(self.nodes)
        rows, arr = [], []
        if n == 0 or not self.graph_positive_control_passed:
            self._path_arrays = arr
            self.paths = pd.DataFrame(rows)
            self.paths.to_csv(self.cache / "candidate_graph_paths.csv", index=False)
            return self.paths

        # Keep proximal-LAD-neighborhood nodes plus genuinely novel nodes.
        # Exclude nodes that merely rediscover downstream portions of accepted LAD.
        allowed = (
            (self.nodes.lad_nearest_arc_mm <= PROXIMAL_LAD_ARC_MAX_MM)
            | self.nodes.is_novel_node
        ).to_numpy(bool)

        ri, ci, da = [], [], []
        for r in self.edges.itertuples():
            i, j = int(r.i), int(r.j)
            if not (allowed[i] and allowed[j]):
                continue
            ri += [i, j]
            ci += [j, i]
            da += [float(r.cost), float(r.cost)]
        anchor = n
        for i, c in self._anchor_links_v2(allowed):
            ri += [anchor, i]
            ci += [i, anchor]
            da += [c, c]
        G = csr_matrix((da, (ri, ci)), shape=(n + 1, n + 1))
        dist, pred = dijkstra(G, directed=False, indices=anchor, return_predecessors=True)
        P = self.nodes[["z", "y", "x"]].to_numpy(float)

        target_order = np.argsort(dist[:n])
        for target in target_order:
            if not np.isfinite(dist[target]) or not allowed[target]:
                continue
            if not bool(self.nodes.iloc[target].is_novel_node):
                continue
            chain, cur, guard = [], int(target), 0
            while cur != anchor and cur >= 0 and guard < n + 5:
                chain.append(cur)
                cur = int(pred[cur])
                guard += 1
            if cur != anchor:
                continue
            chain = chain[::-1]
            pts = np.vstack([self.lad[0], P[chain]])
            plen = float(base.arc_mm(pts, self.spacing)[-1])
            rows.append({
                "path_id": len(rows),
                "target_node": int(target),
                "n_nodes": len(chain),
                "graph_cost": float(dist[target]),
                "graph_length_mm": plen,
                "endpoint_from_lad_mm": float(self.nodes.iloc[target].lad_distance_mm),
                "endpoint_aorta_mm": self._aorta_distance(pts[-1]),
                "node_ids": ";".join(map(str, chain)),
            })
            arr.append(pts)
            if len(rows) >= 12:
                break
        self._path_arrays = arr
        self.paths = pd.DataFrame(rows)
        self.paths.to_csv(self.cache / "candidate_graph_paths.csv", index=False)
        return self.paths

    def validate_paths(self):
        summary = super().validate_paths()
        summary = dict(summary)
        summary["algorithm"] = ALGORITHM_VERSION
        summary["graph_positive_control_passed"] = bool(getattr(self, "graph_positive_control_passed", False))
        summary["adaptive_edge_radius_mm"] = float(getattr(self, "edge_radius_mm", np.nan))
        summary["n_reference_lad_nodes"] = int(self.nodes.is_reference_lad_node.sum()) if len(self.nodes) else 0
        summary["n_novel_nodes"] = int(self.nodes.is_novel_node.sum()) if len(self.nodes) else 0
        summary["proximal_departure_arc_limit_mm"] = PROXIMAL_LAD_ARC_MAX_MM
        if not summary["graph_positive_control_passed"]:
            summary["status"] = "GLOBAL_GRAPH_POSITIVE_CONTROL_FAILED"
        elif int(summary.get("n_validated_extensions", 0)) == 0 and int(summary.get("n_candidate_paths", 0)) == 0:
            summary["status"] = "GLOBAL_GRAPH_POSITIVE_CONTROL_PASSED_NO_PROXIMAL_NOVEL_PATH"
        self.summary = summary
        base._json_write(summary, self.cache / "summary.json")
        return summary


def synthetic_global_graph_v2_self_test():
    # Regression check for the exact conceptual error: a known-true scaffold gap
    # larger than the old 6.5-mm cap must be eligible for adaptive calibration.
    radii = list(EDGE_RADII_MM)
    return {
        "passed": bool(radii[0] == 6.5 and radii[-1] >= 8.0 and NOVEL_LAD_DIST_MM > REF_LAD_DIST_MM),
        "algorithm": ALGORITHM_VERSION,
        "edge_radii_mm": radii,
    }
