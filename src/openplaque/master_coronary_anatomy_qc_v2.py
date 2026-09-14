from __future__ import annotations

"""Drive-layout-safe wrapper for the master frozen coronary anatomy QC workflow.

Scientific anatomy decisions, QC thresholds, and report logic are unchanged from
master_coronary_anatomy_qc.py. This patch only makes RCA input discovery robust to
older OpenPlaque Drive layouts by recursively locating RCA_source_centerline.csv
when the canonical folder path is absent.
"""

from pathlib import Path

from .master_coronary_anatomy_qc import (
    MasterCoronaryAnatomyQCWorkflow as _BaseWorkflow,
    synthetic_master_qc_self_test,
)


ALGORITHM_VERSION = "master-coronary-anatomy-qc-v1.1-rca-resolver"


class MasterCoronaryAnatomyQCWorkflow(_BaseWorkflow):
    def _resolve_inputs(self):
        r = self.root

        # Preserve the exact accepted paths for left-coronary anatomy.
        preferred = {
            "RCA": r / "Source_Volume_Coronary_Centerlines" / "RCA_source_centerline.csv",
            "LAD": r / "LAD_Takeoff_Root_Alternatives_Report" / "alternative_01_centerline.csv",
            "LM": r / "LAD_Takeoff_Confirmation_Report" / "trunk_centerline.csv",
            "SECONDARY": r / "Secondary_Branch_Lateral_Divergence_Report" / "branch_centerline.csv",
        }

        resolved = {}
        for key, path in preferred.items():
            if path.exists():
                resolved[key] = path
            else:
                resolved[key] = None

        # LAD has one established fallback from the original workflow.
        if resolved["LAD"] is None:
            lad_fallback = r / "Source_Volume_Coronary_Centerlines" / "LAD_source_centerline.csv"
            if lad_fallback.exists():
                resolved["LAD"] = lad_fallback

        # RCA source centerline predates several later result folders and can live
        # one or more levels deeper in MyDrive/OpenPlaque. Search by its unique
        # accepted filename only if the canonical path is absent.
        if resolved["RCA"] is None:
            matches = sorted(
                p for p in r.rglob("RCA_source_centerline.csv")
                if p.is_file()
            )
            if matches:
                # Prefer a path whose parent hierarchy explicitly identifies the
                # source-volume centerline package; otherwise require uniqueness.
                preferred_matches = [
                    p for p in matches
                    if "Source_Volume_Coronary_Centerlines" in p.parts
                ]
                if len(preferred_matches) == 1:
                    resolved["RCA"] = preferred_matches[0]
                elif len(matches) == 1:
                    resolved["RCA"] = matches[0]
                elif len(preferred_matches) > 1:
                    # Multiple copies of the same accepted package can arise from
                    # nested archival folders. Use the shallowest such copy.
                    resolved["RCA"] = sorted(preferred_matches, key=lambda p: (len(p.parts), str(p)))[0]
                else:
                    raise FileNotFoundError(
                        "Found multiple RCA_source_centerline.csv files but none in a "
                        "Source_Volume_Coronary_Centerlines path: "
                        + "; ".join(str(p) for p in matches)
                    )

        missing = [k for k, p in resolved.items() if p is None]
        if missing:
            raise FileNotFoundError(
                "Missing frozen centerline source(s): " + ", ".join(missing) +
                ". Expected prior accepted OpenPlaque result folders under MyDrive/OpenPlaque."
            )

        ct = r / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.npy"
        meta = r / "Cache" / "Secondary_3D_Vesselness_Topology_v1" / "series7_int16.json"
        if not ct.exists() or not meta.exists():
            raise FileNotFoundError(f"Missing source CCTA cache: {ct} / {meta}")

        return resolved, ct, meta
