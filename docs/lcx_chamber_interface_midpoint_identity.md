# LCX chamber-interface midpoint identity experiment

This experiment is a fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

Goal: adjudicate C6 versus C7 as LCX-like versus OM-like using direct atrial–ventricular chamber-surface geometry, without myocardium-derived seams/ribbons, curved-template similarity, or prior AV-groove scores.

Method:
- preserve the established C6/C9 truncation and ~20 mm F2 split topology prerequisite;
- build local LA↔LV and RA↔RV interface clouds from symmetric mutual nearest-neighbor surface pairs;
- use pair midpoints as the daughter-blind anatomical interface representation;
- calibrate with accepted RCA as right-sided positive control and accepted LAD as left-sided negative control;
- compare C6/C7 by interface distance, local PCA-tangent alignment, downstream retention/departure, dual coronary-mask support, and source-CCTA HU support.

The notebook uses the single-kernel execution pattern: Drive mount first, controls second, fresh clone, normal non-editable package install, same-kernel import/test, preflight, and same-kernel scientific run. No Python subprocess is used.

A positive result nominates an LCX-like interface-following daughter and an OM-like interface-departing daughter, but does not establish clinical identity automatically. LM remains unresolved.
