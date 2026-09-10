# Experimental RCA centerline prototype

This branch adds a deliberately narrow first step toward standardized PCAT analysis.

## Scope

`src/openplaque/centerline.py` extracts one proximal-to-distal path from an **isolated RCA mask**. It does not yet identify the RCA automatically from a whole coronary tree.

Inputs:

- 3-D boolean RCA mask
- voxel spacing in `(x, y, z)` mm
- approximate RCA ostium in NumPy `(z, y, x)` voxel coordinates
- optional distal RCA hint

The mask is reduced to the connected component containing the ostium, skeletonized in 3-D, converted to a 26-connected physical-space graph, and traversed with Dijkstra shortest paths. If a distal hint is supplied, the path ends at the skeleton point nearest that hint. Otherwise the farthest geodesically reachable skeleton point is used.

The resulting path is parameterized by physical arc length. By default the utility interpolates landmarks at **0, 10, and 50 mm** from the ostium. These are intended to support a later standardized proximal RCA PCAT region corresponding to the 10-50 mm segment.

## Example

```python
from openplaque.centerline import extract_rca_centerline, show_centerline_mip

result = extract_rca_centerline(
    rca_mask,
    spacing_xyz_mm=spacing,
    ostium_zyx=(z0, y0, x0),
    distal_hint_zyx=(z1, y1, x1),
)

print(result.length_mm)
print(result.landmarks_xyz_mm[10.0])
print(result.landmarks_xyz_mm[50.0])
show_centerline_mip(ct_volume, result)
```

## Why require an ostium and optional distal hint?

This prototype is intended to validate the geometry before adding automatic coronary-tree labeling. A raw coronary skeleton may contain side branches and bifurcations; automatically choosing the farthest endpoint can therefore select the wrong branch. Supplying a distal RCA hint makes branch selection explicit and testable.

## Validation before PCAT

Before using this path for PCAT sampling, inspect several cases and verify:

1. the centerline remains inside the intended RCA lumen;
2. no branch jumps occur;
3. the 0-mm landmark is at the RCA ostium;
4. the 10-mm and 50-mm landmarks lie on the expected proximal RCA segment;
5. arc-length changes are stable with anisotropic voxel spacing;
6. results are robust to small perturbations of the supplied ostium and distal hint.

The subsequent PCAT implementation should use the centerline mainly for **longitudinal position**. The radial PCAT boundary should be derived from the vessel/outer-wall geometry rather than from a simple circular tube around the centerline.

Research use only. Not clinically validated. Not for diagnosis.
