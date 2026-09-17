# LAD Source-Space Plaque Transfer Validation v1

## Question

Does the RCA reference-normalized source-space plaque-excess model transfer to the LAD without LAD-specific fitting?

## Scientific prerequisites

- Frozen master baseline: `0593b453959f5a353d644267fbeef24b514ef4d7`.
- RCA excess-specificity experiment must have status `RCA_SOURCE_SPACE_PLAQUE_EXCESS_SPECIFICITY_PASS`.
- Distal LAD extension must have status `LAD_DISTAL_EXTENSION_BIDIRECTIONALLY_SOURCE_CONFIRMED`.
- The frozen master and clinical labels are not modified.

## Design

The experiment refits the already-prespecified RCA normal-wall model only from the RCA 20-50 mm reference zone, using the same four shell thicknesses (0.75, 1.0, 1.25, 1.5 mm), component HU bins, robust Huber regression, and 90th-percentile reference residual threshold used in the successful RCA specificity experiment. Those frozen RCA coefficients are then applied to the LAD without any LAD refitting.

The research LAD path consists of the frozen 24.997-mm LAD plus the independently traced and bidirectionally confirmed distal extension. The existing frozen LAD arc is preserved: arc 0 is the previously frozen distal endpoint and the added distal extension is represented at negative arc values. Prior LAD ensemble plaque votes are used only on the frozen LAD arc for cross-modal validation. The added distal extension is quantified but treated as exploratory because it has no corresponding prespecified ensemble-validation axis.

## Prospective transfer gates

The LAD transfer passes only if all of the following hold:

- frozen-LAD source station QC >= 0.90;
- confirmed distal-extension source station QC >= 0.85;
- at least 4 majority-positive and 5 vote-free frozen-LAD validation bins;
- majority-positive versus vote-free AUROC >= 0.75;
- if at least two strict 5/5 bins are present, strict 5/5 versus vote-free AUROC >= 0.85;
- median excess in majority-positive bins is at least 1.5 times the vote-free median;
- minimum non-nominal shell Spearman correlation with the nominal 1.0-mm profile >= 0.75.

## Outputs

The run writes station-level source-space quantification, the frozen RCA reference model coefficients, LAD excess profiles for all shell thicknesses, cross-modal validation bins, shell-consistency results, automatically selected source-CCTA QC figures, an HTML report, summary/run-state JSON, and a ZIP archive.

## Boundary

This is a cross-vessel transfer validation of a reference-normalized source-space plaque proxy. It is not an independently segmented outer-wall measurement and is not validated clinical total plaque volume. Distal-extension plaque quantities remain research/exploratory until independently cross-validated.
