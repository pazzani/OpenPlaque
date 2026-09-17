# Left-Coronary Backbone Branch Discovery v1.1 — coordinate-space fix

The original v1.0 implementation attempted to load both:

- `series7_int16.npy` as the full Series-7 source CCTA, and
- `vesselness.npy` from `Cache/Secondary_3D_Vesselness_Topology_v1`

and assumed they shared the same voxel lattice. They do not. In the current case the full source is `(524, 512, 512)`, while the cached vesselness is an older cropped ROI `(63, 79, 59)`. Therefore the v1.0 assertion correctly stopped the run rather than silently sampling the wrong coordinate space.

v1.1 is a technical implementation correction on the **same conceptual experiment branch**. It does not alter the prospective branch-discovery design or acceptance gates.

## Correction

`left_coronary_backbone_branch_discovery_v1_1.py`:

1. loads only the full-resolution Series-7 source image and geometry;
2. waits until the complete label-neutral backbone is known;
3. creates a one-time source-voxel ROI enclosing that backbone with a 14-mm margin;
4. recomputes multiscale 3-D Frangi vesselness directly from source CCTA in that ROI;
5. samples that ROI using an explicit full-source-to-local-ROI voxel transform.

The 14-mm margin is larger than the 9-mm branch search plus preview reach, so the source-led search is not artificially clipped by the vesselness field.

## Unchanged prospective rules

- same validated backbone construction;
- same known C6/C7 side-branch positive control;
- same 1.5-mm blind seed spacing;
- same 35–145 degree branch-direction proposals;
- same 9-mm maximum branch search;
- same backbone re-entry exclusion;
- same HU gate;
- same vesselness threshold calibration rule, now measured on correctly aligned source-space vesselness;
- same dense 0.4-mm orthogonal source-plane QC;
- same >=5-mm, >=80% plane-pass, >=30-degree branch-angle, and >=3-mm endpoint-separation acceptance gates;
- no aortic geometry or clinical vessel labels used in discovery;
- frozen anatomy remains unchanged.

The output folder remains `Left_Coronary_Backbone_Branch_Discovery_v1`. A completed v1.1 run additionally writes `vesselness_coordinate_fix_v1_1.json` for provenance.
