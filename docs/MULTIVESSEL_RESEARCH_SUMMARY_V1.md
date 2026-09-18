# OpenPlaque Multivessel Research Summary v1

## Purpose

This experiment consolidates the current coronary research state into one machine-readable and human-readable report without changing any science.

It does not fit a new plaque model, alter anatomy, retune PCAT, or promote provisional labels.

## Inputs

- Master Coronary Anatomy Baseline v2
- RCA Plaque + PCAT Research Lock v1
- LAD Source-Space PCAT Feasibility v1
- LAD Distal-Reference Plaque Self-Calibration v1
- LCX Structural Source-QC Freeze v1
- LCX-like / OM-like Source-Space Composition + PCAT Feasibility v1

All inputs must retain their expected status.

## Current quantitative interpretation

### RCA

RCA is the only artery with a locked plaque-burden research benchmark.

- Plaque quantity: reference-normalized source-space excess-wall proxy
- Nominal shell: 1.0 mm
- Research total: approximately 23.2 mm3
- Direct PCAT: locked RCA 10-50 mm endpoint
- Neither the plaque quantity nor the PCAT result is a commercial clinical metric

### LAD

The accepted frozen LAD anatomy is strong.

Direct PCAT is technically feasible on the frozen LAD.

The independently source-confirmed distal continuation also supports direct PCAT, but it is not part of the frozen master.

The developmental LAD plaque self-calibration produced a plausible numeric estimate but did not pass all prespecified specificity/stability gates. It remains developmental and is not locked.

### C6 / C7

Research structural labels are frozen:

- C6 = LCX-like parent continuation
- C7 = OM-like daughter

Clinical LCX/OM identity is not established.

Direct PCAT is technically feasible on both paths.

Raw 1-mm peri-luminal shell HU composition is reported, but raw shell volume is not plaque burden or TPV.

### Left main

Left main remains unresolved.

No quantitative LM measurement is allowed.

## Outputs

- coronary_research_measurements.csv
- coronary_anatomy_research_status.csv
- coronary_direct_pcat_summary.csv
- research_measurements.json
- input_provenance.json
- run_state.json
- 01_multivessel_direct_PCAT.png
- 02_coronary_research_readiness.png
- OPENPLAQUE_MULTIVESSEL_RESEARCH_SUMMARY_REPORT.html
- ZIP archive

## Scientific boundary

This is an integrated research report, not a clinical coronary report.

- RCA plaque is a locked research proxy, not independently segmented clinical TPV.
- LAD plaque remains developmental.
- C6/C7 raw shell composition is not plaque burden.
- PCAT is direct attenuation and not proprietary FAI.
- C6/C7 clinical labels and left main remain unresolved.
- Frozen anatomy is not modified.
