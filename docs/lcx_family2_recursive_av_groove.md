# LCX Family-2 recursive AV-groove adjudication

Fresh experiment from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.

Purpose: resolve the favored distal Family-2 subtree from the prior local-bifurcation experiment. Only source candidates C6, C7, and C9 are carried forward. C5/C11 are excluded because they belong to the less-favored Family 1.

Method:
- reconstruct the common prefix shared by C6/C7/C9;
- locate the next sustained internal split after the five-path consensus trunk;
- validate the local AV-groove tangent signal using accepted RCA as positive control and accepted LAD as negative control;
- compare only the daughter subfamilies downstream of the Family-2 split;
- preserve current + legacy TotalSegmentator support and source-CCTA HU gates;
- use the same `0.05` family-level separation margin as the preceding experiment, rather than loosening it;
- prior LCX-template similarity remains metadata only and has zero decision weight.

The run is low-memory by construction: source CCTA is memory-mapped and masks are converted one at a time to sparse physical-space KD-trees. The standalone Colab executes the scientific run in a fresh Python subprocess.

A positive result is only `LCX_F2_INTERNAL_CONTINUATION_CANDIDATE_REQUIRES_VISUAL_QC`; it does not establish clinical vessel identity. LM remains unresolved.
