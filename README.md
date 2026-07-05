# OpenPlaque Volume-Based Boundary Tuning

This update replaces absolute voxel-count component filtering with physical
minimum component volume in mm³.

## Why

The previous default `min_component_voxels=80` was too aggressive for small-vessel
findings and removed all LCX plaque in the UCLA test.

## New tuning parameters

- `min_component_volume_mm3`
- `lumen_distance_voxels`
- optional `low_hu_threshold`
- optional `high_hu_threshold`

## Files

```text
src/openplaque/boundary_volume.py
notebooks/08_Volume_Based_Boundary_Tuning.ipynb
```
