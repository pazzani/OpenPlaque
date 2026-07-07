# Plaque characterization notes

The characterization step consumes a refined plaque mask and the original CT volume. Every voxel with plaque label `2` is classified by HU threshold into:

- low attenuation plaque
- non-calcified plaque
- mixed/intermediate plaque
- calcified plaque

The module reports volumes and fractions per vessel and per connected lesion. It also writes color-coded overlay PNGs and optional NIfTI class maps.

This is a transparent rule-based first version. It should be calibrated and validated before clinical use.
