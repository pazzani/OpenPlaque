# Plaque + Inflammation Final Visualization v2

This is the single corrected visualization/export endpoint requested after audit of v1.

## What changed from v1

The prior visualization correctly reproduced the locked mean PCAT values and longitudinal profiles, but its attenuation-band stacked bars classified **1-mm bin means**. Because all local means fell between -110 and -70 HU, the chart misleadingly showed 100% in one band.

Version 2 fixes that by reconstructing the actual source-space PCAT fat voxels for:

- RCA 10-50 mm, canonical +0.75 mm modeled wall margin
- frozen LAD, accepted 1-24 mm arc range
- C6 LCX-like parent segment after 1 mm bifurcation exclusion

The fat window remains -190 to -30 HU.

The source-space voxel reconstructions are accepted only if they reproduce the locked endpoint mean HU and fat-voxel counts within tight prespecified tolerances.

## Corrected figures

1. Corrected publication-style plaque composition
   - LAP/necrotic-core-like: red
   - fibro-fatty: orange
   - fibrous/intermediate: blue
   - dense calcium: white/light
   - clean non-overlapping legend
   - LM footnote explicitly states calcium-anchor-only status

2. Locked mean PCAT bar
   - RCA - locked 40 mm
   - LAD - exact locked 22.5 mm
   - LCX/C6 - exact locked 5.5297 mm

3. True voxel-level PCAT HU band decomposition
   - -190 to <-150 HU
   - -150 to <-110 HU
   - -110 to <-70 HU
   - -70 to -30 HU

4. Source-space PCAT voxel HU histograms

5. Longitudinal PCAT profiles

6. Coverage-aware longitudinal heatmap

7. RCA radial profile, explicitly labeled exploratory geometry QC

## Scientific boundaries

- Plaque values remain OpenPlaque research best-estimate proxies, not Cleerly outputs.
- The 54 mm3 LM value is a calcium-volume anchor/floor, not complete LM TPV.
- Direct PCAT is not proprietary Caristo FAI-Score.
- The voxel bands are attenuation distributions, not histologic tissue classes.
- LM inflammation is not standardized or estimated.
- C6 remains an LCX-like research structural parent and is a low-confidence clinical-LCX proxy.
- The RCA radial gradient is descriptive geometry QC, not a validated clinical inflammation gradient.

## Run-all behavior

The final notebook:
- mounts Drive first;
- validates all required caches;
- runs tests;
- regenerates the accepted plaque/inflammation endpoint;
- reconstructs and validates actual PCAT voxels;
- displays every figure inline;
- writes PNG, CSV, NPZ, JSON and HTML outputs;
- exports the entire output directory to one ZIP.
