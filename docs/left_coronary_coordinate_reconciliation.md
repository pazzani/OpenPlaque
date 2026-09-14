# Left-coronary coordinate reconciliation

This experiment is intentionally diagnostic. It compares stored left-coronary centerline coordinates against source CCTA and independent TotalSegmentator coronary masks. RCA is used as a positive-control anchor.

The workflow does **not** permit an arbitrary free rigid registration or translation to manufacture agreement. It only tests evidence-backed coordinate interpretations already present in OpenPlaque data, including stored LPS coordinates, source voxel coordinates recomputed through source DICOM geometry, and an explicit RAS-to-LPS sign-flip audit.

The notebook compares current and legacy TotalSegmentator masks separately, plus their union and intersection, samples source-CCTA HU along candidate centerlines, evaluates proximal prefixes, and ranks multiple discovered LAD/secondary candidates to distinguish coordinate errors, stale/wrong centerline selection, and left-coronary under-segmentation.

Outputs are written to `MyDrive/OpenPlaque/Left_Coronary_Coordinate_Reconciliation_v1/` and include candidate inventory, prefix metrics, per-point diagnostics, reconciliation figures, a JSON conclusion, HTML report, and report-back ZIP.
