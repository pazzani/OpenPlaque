# Boundary Refinement Notes

## Motivation

The sample-dataset validation suggested high recall but lower precision, meaning
the model may include extra voxels. These may represent:

- normal vessel wall
- lumen-adjacent voxels
- calcification blooming
- motion artifacts
- interpolation artifacts

## Refinement strategies

1. Remove small connected plaque components.
2. Remove plaque voxels adjacent to the predicted normal-vessel label.
3. Optionally erode the plaque mask to estimate a high-confidence core.
4. Optionally remove voxels outside an HU range.

## Recommended reporting

OpenPlaque should eventually report both:

- Raw AI TPV
- Refined TPV
- Removed boundary volume
- Core TPV

This makes uncertainty explicit rather than hiding it.
