# Left proximal-trunk topology crosswalk v1

## Question

The source-led long continuation is strongly validated by dense source-CCTA plane QC but moves monotonically away from the nearest aortic surface. This experiment asks whether that path geometrically overlaps or rejoins an already established coronary path rather than representing a direct LM-to-aorta bridge.

## Inputs

- Frozen LAD centerline from `Cache/LAD_Frozen_Proximal_Reacquisition_v1/combined_lad_centerline.csv`
- C6 path from `Joint_Three_Vessel_Template_Classifier_v1/candidate_04_source_path.csv`
- C7 extended path from `LCX_Distal_Reacquisition_v1_fixed/C7_extended_path.csv`
- Frozen RCA centerline
- Source-QC-positive local and long proximal-trunk continuation paths
- Prior C6/C7 structural adjudication decision

C6/C7 are used only with their previously established structural labels: C6 = LCX-like parent continuation, C7 = OM-like daughter. Clinical identity is not assumed.

## Prospective geometric association gate

A reference association requires all of:

- minimum path distance <= 1.5 mm;
- at least 20% of the long path within 2.0 mm;
- at least 5.0 mm contiguous arc within 2.0 mm;
- median absolute tangent alignment >= 0.75 among samples within 2.0 mm.

C6/C7 are evaluated separately after their established split arc at 20.25 mm. Their shared pre-split segment is evaluated independently.

## Interpretation

The experiment may identify geometric association with C6, C7, their common trunk, the frozen LAD, or no established reference path. It cannot by itself establish clinical LM/LCX/OM identity and never modifies the frozen coronary anatomy baseline.

## Outputs

`summary.json`, `topology_crosswalk_decision.json`, `topology_crosswalk_metrics.csv`, `long_path_reference_distance_profiles.csv`, `local_path_reference_metrics.csv`, `investigational_composite_path.csv`, projection/distance-profile PNGs, HTML report, and ZIP archive.
