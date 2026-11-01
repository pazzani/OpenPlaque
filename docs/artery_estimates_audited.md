# Corrected notebook 13 execution path

Use `notebooks/13_Plaque_Labels_And_Candidates_Audited.ipynb` for the corrected
version of the uploaded older notebook. The newer upstream notebook 13 includes
additional workflows and is intentionally preserved. Other notebooks and shared
pipeline defaults are unchanged. Only this notebook imports `artery_estimates`.

The corrected notebook pins the library commit and nnU-Net version, checks
installation failures, and saves a complete installed-environment snapshot.
Scientific dependency ranges let pip select wheels compatible with the current
Python runtime, including Python 3.13.16. Setup constrains PyTorch to the runtime's
installed build, preserving Colab's CUDA installation instead of downgrading it.
nnU-Net 2.6.2 explicitly loads model checkpoints with the behavior required by
newer PyTorch. The runtime must supply PyTorch >=2.6. Restart if setup reports that
an already imported package changed. CUDA/driver compatibility and actual private
model inference still require validation in Colab. Dependency resolution is not a
fixed cross-runtime lock; exact versions and their fingerprint are saved per run.

## Inputs and migration

- Study/model extraction caches are keyed by ZIP SHA-256 and marked complete only
  after extraction succeeds. A different ZIP cannot reuse a prior study directory.
- DICOM selection uses the exact SeriesInstanceUID, requires a unique series number,
  and checks the number of loaded files. Ambiguous metadata selections stop.
- Inspect `study.summary()` before setting `SERIES_OVERRIDES`. There is no automatic
  fallback to the old study's series numbers.
- Reused masks require a sidecar with study, series, model, inference-version and
  geometry identity plus the mask checksum. Legacy masks are regenerated, not
  silently assigned provenance. All image/mask geometry must agree.
- All five requested fold checkpoints and model metadata must be present. Model
  labels must identify background, vessel/lumen and plaque; an unrecognized label
  vocabulary fails closed and must be reviewed against the actual trained model.

## Measurement semantics

`model_*` measures label 2. `candidate_*` measures a separate exterior search shell
around labels 1 and 2, with radius in physical millimeters and Euclidean voxel-center
distance. A zero radius or empty anatomy produces an empty shell. The default
0.5 mm is an explicit exploratory setting, not a validated vessel-wall thickness.

Both regions use the same configured HU bins, including the high-attenuation bin.
Intermediate intensity does not establish a mixed lesion. Candidates are never
added to model plaque. Below-range and nonfinite voxels are accounted for separately.
`*_region_mm3` includes every voxel in that region; `*_classified_mm3` includes the
four configured HU bins. Fractions use classified volume and are blank if undefined.
TOTAL fractions use summed physical volumes, not summed voxel counts.

Volumes use the validated image grid. No other series' spacing is substituted.
Whether a curved/derived reformat represents physical tissue volume without repeated
sampling must be established externally; this code cannot infer that from spacing.
TOTAL is a sum of the selected artery grids, with anatomical non-overlap requiring review.

CSV, charts and HTML all preserve the model/candidate distinction. HTML embeds its
chart and derives thresholds from configuration. A manifest records input/model
checksums, series, geometry, mask source/checksum, commit and measurement settings.

## Artery visualization

Section 7 provides one interactive tab per artery, with CT-only and overlaid slices,
image-axis selection, slice navigation, CT window controls, opacity and vessel/plaque
toggles. Each artery starts at its largest plaque slice (or largest vessel slice
when plaque is absent). Cyan represents vessel label 1. Plaque uses Low attenuation
(red), Noncalcified (orange), Mixed/intermediate (blue), Calcified (white), and
Unclassified plaque (purple for below-range/nonfinite label-2 values).

Rotatable Plotly surfaces preserve each image's spacing, origin and direction.
They display model labels only, never candidates. Each artery is shown separately;
derived-image grids are not assumed to share a common anatomical frame.
`MESH_STEP_SIZE=1` retains native grid sampling; larger steps can omit fine detail
and affect visualization only. Measurement arrays are unchanged by the viewers.

PNG overviews are embedded in the main HTML report. Standalone, offline-capable
3D HTML viewers are linked from it; download these beside the report to retain
working links. The interactive slice controls require the running notebook.
Set `SHOW_INTERACTIVE_VIEWS=False` for export-only operation, or
`CREATE_3D_VIEWS=False` to skip surface generation. Empty masks are shown explicitly.

## Validation

Run `PYTHONPATH=src python -m pytest tests/test_artery_estimates.py
tests/test_artery_detection.py tests/test_plaque_context.py`.
The new tests cover real synthetic DICOM series sharing a folder, stale extraction,
cache identity/integrity, geometry mismatches, detection ambiguity, incomplete models,
HU boundaries/nonfinite values, empty masks, physical radii and volume-weighted totals.
The notebook's measurement/export cells can be exercised with synthetic reports;
patient inference still requires the private DICOM/model ZIPs and a Colab GPU.
