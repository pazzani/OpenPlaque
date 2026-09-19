# Plaque + Inflammation Publication Visualization v1

This is a visualization/export layer for the existing plaque + inflammation best-estimate endpoint.

It is intentionally on the same branch because it does not change the scientific plaque or PCAT estimates.

## Run-all behavior

The Colab:

1. mounts Google Drive first;
2. clones the pinned OpenPlaque branch;
3. runs synthetic tests;
4. regenerates the plaque/inflammation endpoint into a subfolder;
5. loads cached longitudinal PCAT files for RCA, frozen LAD, C6 (LCX-like parent), and C7 (OM-like alternate);
6. generates figures;
7. displays all figures inline;
8. writes an HTML report and CSV tables;
9. creates one ZIP containing every output and the endpoint subfolder.

No GPU or model inference is required.

## Figures

- publication-style stacked plaque composition by LAD/RCA/LCX/LM
- mean raw PCAT attenuation by RCA/LAD/LCX
- stacked PCAT attenuation-band distribution
- longitudinal PCAT profiles
- coverage-aware longitudinal PCAT heatmap
- RCA radial PCAT gradient

## PCAT band decomposition

The band chart is **not** a voxel-level histologic tissue decomposition.

For each measured 1-mm longitudinal bin, its local mean PCAT HU is assigned to one of four descriptive bands:

- -190 to <-150 HU
- -150 to <-110 HU
- -110 to <-70 HU
- -70 to -30 HU

The bin contributes its measured fat-voxel count to that band. Bars therefore represent a **fat-voxel-weighted distribution of local mean PCAT attenuation**.

This preserves the actual longitudinal measurements without pretending that summary mean/SD values reveal the voxel-level HU histogram.

## Boundaries

- Direct PCAT attenuation is not proprietary Caristo FAI-Score.
- LM inflammation is not standardized and is not estimated.
- LCX primary visualization uses C6, the frozen LCX-like parent structural segment. C7 is retained only as an alternate research branch.
- Plaque volumes remain OpenPlaque best-estimate research proxies, not Cleerly outputs.
