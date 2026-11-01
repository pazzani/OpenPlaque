# Corrected notebook 13 execution path

Use `notebooks/13_Plaque_Labels_And_Candidates_Audited.ipynb` for the corrected
version of the uploaded older notebook. The newer upstream notebook 13 includes
additional workflows and is intentionally preserved. Other notebooks and shared
pipeline defaults are unchanged. Only this notebook imports `artery_estimates`.

The corrected notebook pins the library commit and direct dependencies, checks
installation failures, and saves a complete installed-environment snapshot.
Use a fresh Python 3.10-3.12 Colab GPU runtime. Restart if setup reports that an
already imported package changed. CUDA/driver compatibility remains runtime-dependent.

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

## Validation

Run `PYTHONPATH=src python -m pytest tests/test_artery_estimates.py
tests/test_artery_detection.py tests/test_plaque_context.py`.
The new tests cover real synthetic DICOM series sharing a folder, stale extraction,
cache identity/integrity, geometry mismatches, detection ambiguity, incomplete models,
HU boundaries/nonfinite values, empty masks, physical radii and volume-weighted totals.
The notebook's measurement/export cells can be exercised with synthetic reports;
patient inference still requires the private DICOM/model ZIPs and a Colab GPU.
