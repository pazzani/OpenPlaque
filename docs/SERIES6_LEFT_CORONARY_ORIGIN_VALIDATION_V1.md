# Series 6 Left Coronary Origin Validation v1

Focused same-exam phase comparison:

- reference: Series 7, **BestDiast 63%**
- alternate: Series 6, **BestSyst 32%**

The experiment is intentionally limited to these two like-for-like high-resolution coronary source reconstructions. Series 8, Series 10 and the large multiphase Series 11 are excluded.

## Scientific question

Does Series 6 independently show a **distinct left aortic contact** associated with the frozen LAD/C6 structure, separated from the known RCA ostium?

## Fixed gates

- registration translation magnitude <= 6 mm
- RCA support >= 0.60
- RCA aortic-contact cluster within 5 mm of frozen RCA root
- LAD support >= 0.60
- C6 support >= 0.50
- LAD and C6 share an independent-model connected component
- left candidate contact has >=10 voxels
- left contact is >=8 mm from RCA root
- left contact is <=25 mm from frozen proximal LAD anchor

No threshold is adjusted after observing the result.

## Final statuses

- `SERIES6_DISTINCT_LEFT_OSTIUM_PRESENT`
- `SERIES6_NO_DISTINCT_LEFT_OSTIUM`
- `SERIES6_RCA_CONTROL_FAILED`
- `SERIES6_REGISTRATION_FAILED`

Even a positive result does not automatically establish clinical LM/LCX/OM identity and does not modify the frozen master.

## Outputs

`/content/drive/MyDrive/OpenPlaque/Series6_Left_Coronary_Origin_Validation_v1/`

Expected outputs include preparation/provenance JSON, registration CSV, path support, aortic-contact clusters, gates, QC PNGs, HTML report, summary/decision/run-state JSON and ZIP archive.
