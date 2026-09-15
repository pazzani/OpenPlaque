# LCX consensus trunk and AV-groove adjudication

Algorithm: `lcx-consensus-trunk-av-groove-v1.0`

This experiment is a fresh branch from the frozen OpenPlaque baseline
`0593b453959f5a353d644267fbeef24b514ef4d7`.

## Purpose

The previous source-anatomy experiment showed that the five strongest LCX hypotheses are not five independent arteries. They share a long source-space trajectory and diverge only distally. This experiment:

1. extracts the longest consensus prefix shared by all five source paths;
2. identifies the portion after sustained separation from the accepted LAD;
3. freezes that segment as a **consensus left-coronary branch trunk** if source-space support is strong;
4. clusters the distal continuations into branch families;
5. uses TotalSegmentator high-resolution chamber masks to determine which continuation best follows the left atrioventricular groove between the left atrium and left ventricle.

## Patient-specific control

The accepted RCA is scored against the **right atrium/right ventricle** surfaces as a positive AV-groove control. The accepted LAD is scored against the **left atrium/left ventricle** surfaces as a left-sided negative control. If the RCA-vs-LAD control does not separate, no LCX continuation is promoted.

Curved LCX-template similarity is retained only as metadata and contributes **zero** to the AV-groove decision.

## Scientific boundary

A passing result nominates an LCX-like distal continuation for visual/anatomical QC. It does not establish LM identity, circumferential registration, plaque volume, or a clinically validated vessel label. LM and LCX remain unresolved in Master Coronary Anatomy Baseline v2 until explicitly promoted in a later frozen-baseline experiment.
