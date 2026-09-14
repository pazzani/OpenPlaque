from __future__ import annotations

"""Bookkeeping-corrected Master Coronary Anatomy Baseline v2.1.

This wrapper preserves the v2 geometry/anatomy decisions, but takes the frozen-LAD
core length and post-hoc validated proximal-extension length from the authoritative
LAD proximal-reacquisition tracking summary.  Those two values must sum to the
accepted combined LAD length within tolerance.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .master_coronary_anatomy_baseline_v2 import (
    MasterCoronaryAnatomyBaselineV2,
    _safe_json,
    _write_json,
)

ALGORITHM_VERSION = "master-coronary-anatomy-baseline-v2.1-authoritative-lad-partition"


class MasterCoronaryAnatomyBaselineV21(MasterCoronaryAnatomyBaselineV2):
    def build_evidence(self):
        accepted, ledger, summary = super().build_evidence()

        tracking_fp = self.root / "Cache" / "LAD_Frozen_Proximal_Reacquisition_v1" / "tracking_summary.json"
        tracking = _safe_json(tracking_fp)
        if not tracking:
            raise FileNotFoundError(f"Authoritative LAD tracking summary missing: {tracking_fp}")

        required = [
            "frozen_lad_length_mm",
            "posthoc_validated_extension_mm",
            "combined_lad_length_mm",
        ]
        missing = [k for k in required if k not in tracking]
        if missing:
            raise RuntimeError(f"Authoritative LAD tracking summary lacks fields: {missing}")

        core = float(tracking["frozen_lad_length_mm"])
        extension = float(tracking["posthoc_validated_extension_mm"])
        combined = float(tracking["combined_lad_length_mm"])
        measured_combined = float(summary["accepted"]["LAD_length_mm"])

        if abs((core + extension) - combined) > 0.02:
            raise RuntimeError(
                f"LAD partition inconsistent: core {core:.6f} + extension {extension:.6f} != combined {combined:.6f} mm"
            )
        if abs(combined - measured_combined) > 0.02:
            raise RuntimeError(
                f"Tracking-summary combined LAD {combined:.6f} disagrees with accepted centerline {measured_combined:.6f} mm"
            )

        summary["algorithm"] = ALGORITHM_VERSION
        summary["accepted"]["LAD_length_mm"] = combined
        summary["accepted"]["LAD_frozen_core_mm"] = core
        summary["accepted"]["LAD_validated_proximal_addition_mm"] = extension
        summary["LAD_length_partition_source"] = str(tracking_fp)
        summary["LAD_partition_sum_check_mm"] = float(core + extension)

        ledger = ledger.copy()
        m = ledger["evidence"].eq("Frozen LAD core")
        if m.any():
            ledger.loc[m, "metric"] = f"length={core:.3f} mm"
            ledger.loc[m, "reason"] = "Independently validated compact-lumen LAD segment; length imported from authoritative post-hoc tracking summary."

        m = ledger["evidence"].eq("LAD proximal addition")
        if m.any():
            ledger.loc[m, "metric"] = f"added={extension:.3f} mm; total={combined:.3f} mm"
            ledger.loc[m, "reason"] = "Dense final-path-tangent QC retained only the defensible proximal extension; partition imported from authoritative tracking summary."

        raw = tracking.get("proposal_time_raw_extension_mm")
        direct_mask = ledger["evidence"].eq("Direct proximal LAD reacquisition")
        if raw is not None:
            row = {
                "evidence": "Direct proximal LAD reacquisition",
                "decision": "TRUNCATE",
                "metric": f"raw proposal={float(raw):.3f} mm; accepted={extension:.3f} mm; first sustained failure={float(tracking.get('first_sustained_posthoc_failure_mm', np.nan)):.3f} mm",
                "reason": "Final-path-tangent QC rejected the broad/off-center continuation beyond the accepted proximal addition.",
            }
            if direct_mask.any():
                for k, v in row.items():
                    ledger.loc[direct_mask, k] = v
            else:
                ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)

        accepted.to_csv(self.cache / "accepted_anatomy.csv", index=False)
        ledger.to_csv(self.cache / "evidence_ledger.csv", index=False)
        _write_json(summary, self.cache / "master_anatomy_summary.json")
        self.summary = summary
        return accepted, ledger, summary


def synthetic_master_baseline_v21_self_test():
    core = 22.632845080314638
    extension = 2.3638076169482742
    total = 24.996652697262913
    return {
        "passed": bool(abs(core + extension - total) < 1e-9),
        "algorithm": ALGORITHM_VERSION,
    }
