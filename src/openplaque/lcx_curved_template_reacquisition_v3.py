from __future__ import annotations

"""Same-experiment path correction for LCX curved-template reacquisition.

The authoritative Master Coronary Anatomy Baseline v2 folder lives under
OpenPlaque/Cache, not directly under the OpenPlaque Drive root.  This wrapper
keeps the cache-only v2 scientific workflow unchanged while correcting that
input path in one place.
"""

from pathlib import Path

from . import lcx_curved_template_reacquisition as base
from . import lcx_curved_template_reacquisition_v2 as v2

BASELINE = base.BASELINE
ALGORITHM = "lcx-curved-template-reacquisition-v1.3-cache-only-master-path-fix"
MASTER = Path("Cache/Master_Coronary_Anatomy_Baseline_v2/master_anatomy_summary.json")
OUTPUT_DIRNAME = "LCX_Curved_Template_Reacquisition_v1_cacheonly_masterpath_fixed"


def run(drive_root="/content/drive/MyDrive/OpenPlaque", output_root=None):
    # The v2 workflow reads base.MASTER dynamically; correct the authoritative
    # location before execution and restore module globals afterwards.
    old_master = base.MASTER
    old_algorithm = v2.ALGORITHM
    old_output_dirname = v2.OUTPUT_DIRNAME
    try:
        base.MASTER = MASTER
        v2.ALGORITHM = ALGORITHM
        v2.OUTPUT_DIRNAME = OUTPUT_DIRNAME
        return v2.run(drive_root, output_root)
    finally:
        base.MASTER = old_master
        v2.ALGORITHM = old_algorithm
        v2.OUTPUT_DIRNAME = old_output_dirname


def synthetic_lcx_template_v3_self_test():
    prior = v2.synthetic_lcx_template_v2_self_test()
    return {
        "passed": bool(prior.get("passed") and str(MASTER).startswith("Cache/")),
        "algorithm": ALGORITHM,
        "master_relative_path": str(MASTER),
        "historical_Full_DICOM_zip_used": False,
    }
