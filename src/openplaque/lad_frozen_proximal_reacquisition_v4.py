from __future__ import annotations

"""Calibration-safe wrapper for frozen-LAD proximal reacquisition v1.2.

This is a same-experiment runtime correction.  The scientific search and final-
path-tangent validator are inherited unchanged from v1.2.  The only change is
calibration handling: reuse the immediately preceding successful frozen-LAD
calibration when it is valid; otherwise recompute it with denser sampling and
conservative source-plane QC rather than aborting after a sparse sample failure.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .lad_frozen_proximal_reacquisition import (
    _json_write,
    _unit,
    arc_mm,
    lumen_metrics,
    orthogonal_plane,
)
from .lad_frozen_proximal_reacquisition_v3 import (
    LADFrozenProximalReacquisitionWorkflowV3,
)

ALGORITHM_VERSION = "lad-frozen-proximal-reacquisition-v1.2.1-calibration-safe"


def _valid_calibration(ref):
    try:
        return bool(
            int(ref.get("n_planes", 0)) >= 4
            and 0.7 <= float(ref["median_radius_mm"]) <= 3.0
            and 150.0 <= float(ref["median_center_hu"]) <= 1200.0
            and 0.0 <= float(ref.get("median_shift_mm", 0.0)) <= 1.2
            and 0.05 <= float(ref.get("median_circularity", 0.5)) <= 1.2
        )
    except Exception:
        return False


class LADFrozenProximalReacquisitionWorkflowV4(LADFrozenProximalReacquisitionWorkflowV3):
    def calibrate_from_frozen_lad(self, end_mm=8.0):
        if self.ct is None:
            self.load_source_ct()
        if self.frozen is None:
            self.load_frozen_lad()

        cal_fp = self.cache / "lad_lumen_calibration.json"
        planes_fp = self.cache / "frozen_lad_calibration_planes.csv"

        # The previous successful run used this exact frozen LAD and exact source
        # volume.  Reuse that calibration when explicitly requested and when its
        # values pass basic plausibility checks.
        if self.reuse.get("calibration", True) and cal_fp.exists():
            try:
                ref = json.loads(cal_fp.read_text(encoding="utf-8"))
                n_rows = 0
                if planes_fp.exists():
                    try:
                        n_rows = len(pd.read_csv(planes_fp))
                    except Exception:
                        n_rows = 0
                if _valid_calibration(ref) and (n_rows == 0 or n_rows >= 4):
                    ref = dict(ref)
                    ref["reuse_note"] = "validated calibration reused from prior successful run on identical frozen LAD/source CCTA"
                    self.ref = ref
                    self._record("calibration", "reused_valid_prior_calibration", cal_fp, f"n_planes={ref.get('n_planes')}")
                    return self.ref
            except Exception:
                pass

        # Robust fallback: sample twice as densely as v1.0.  This does not alter
        # the lumen definition; it only avoids an all-or-nothing sparse-sample
        # calibration failure.
        p = self.frozen
        s = arc_mm(p, self.spacing)
        stop = min(float(end_mm), float(s[-1]) - 0.35)
        if stop <= 0.45:
            raise RuntimeError("Frozen LAD too short for proximal calibration")

        rows = []
        for ss in np.arange(0.4, stop + 1e-9, 0.4):
            i = int(np.argmin(np.abs(s - ss)))
            a = max(0, i - 4)
            b = min(len(p) - 1, i + 4)
            if b <= a:
                continue
            t = _unit((p[b] - p[a]) * self.spacing)
            im, g, _, _ = orthogonal_plane(self.ct, p[i], t, self.spacing)
            m = lumen_metrics(im, g)
            rows.append({"arc_mm": float(s[i]), **m})

        q = pd.DataFrame(rows)
        if q.empty:
            raise RuntimeError("Could not obtain any source-CCTA calibration planes from frozen LAD")

        good = q[
            np.isfinite(q.radius_mm)
            & np.isfinite(q.component_median_hu)
            & np.isfinite(q.centroid_shift_mm)
            & (q.centroid_shift_mm <= 1.2)
            & (q.radius_mm >= 0.7)
            & (q.radius_mm <= 3.0)
            & (q.component_median_hu >= 150.0)
            & (q.component_median_hu <= 1200.0)
        ]

        q.to_csv(planes_fp, index=False)
        if len(good) < 4:
            # Preserve the diagnostic table so a failure is inspectable.
            self._record("calibration", "robust_recalibration_failed", planes_fp, f"usable={len(good)}/{len(q)}")
            raise RuntimeError(
                f"Could not calibrate proximal tracking from frozen LAD: only {len(good)} of {len(q)} dense source planes usable"
            )

        self.ref = {
            "median_radius_mm": float(good.radius_mm.median()),
            "median_center_hu": float(good.component_median_hu.median()),
            "median_shift_mm": float(good.centroid_shift_mm.median()),
            "median_circularity": float(good.circularity.median()),
            "n_planes": int(len(good)),
            "source": "dense source-CCTA calibration from first ~8 mm of independently frozen LAD",
        }
        _json_write(self.ref, cal_fp)
        self._record("calibration", "robust_dense_recalibration", cal_fp, f"usable={len(good)}/{len(q)}")
        return self.ref
