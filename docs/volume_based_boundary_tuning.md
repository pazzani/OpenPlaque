# Volume-Based Boundary Tuning

The previous boundary refinement used an absolute connected-component voxel
threshold. This can be misleading when image spacing varies and can be too
aggressive for small plaques.

This version tunes a physical minimum component volume in mm³.

The objective score also penalizes parameter sets that remove all predicted
plaque from a case or remove an excessive fraction of the raw prediction.
