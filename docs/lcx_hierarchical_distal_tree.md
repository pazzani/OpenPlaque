# LCX hierarchical distal-tree adjudication

Research-only source-space experiment for OpenPlaque.

## Purpose

The previous local bifurcation experiment showed that the five LCX candidate paths are hierarchical rather than five peer alternatives. This experiment reconstructs that hierarchy automatically and adjudicates each bifurcation separately.

The expected topology is discovered from the paths, not hard-coded. At every split, child branches are compared using:

- local LA-LV atrioventricular-groove alignment,
- continuity with the incoming parent trunk,
- left-atrial retention,
- current and legacy coronary-mask support,
- source-CCTA lumen attenuation.

Curved-template LCX similarity is metadata only and has zero decision weight.

## Decision policy

A branch gate requires coronary-mask and HU support >= 0.90, local score >= the LAD negative control + 0.05, groove alignment >= 0.35, atrial retention >= 0.55, and an incoming continuation angle <= 50 degrees.

A sibling decision is considered decisive if the gated top child has either:

1. local geometry score margin >= 0.05, or
2. incoming-continuity angle advantage >= 20 degrees **and** groove-alignment advantage >= 0.10.

An LCX-like path is nominated only when the root split and at least one downstream split are decisive and the traversal terminates in a singleton source candidate.

## Scientific boundary

The result is a research-only LCX-like continuation nomination. It does not establish clinical vessel identity, does not update the frozen Master Coronary Anatomy baseline, and does not resolve the left main.
