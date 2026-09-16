# Proximal LAD ostial bridge

Fresh-baseline experiment from `0593b453959f5a353d644267fbeef24b514ef4d7`.

Purpose: test whether the independently accepted proximal LAD endpoint can be connected directly to the aortic surface by a short source-CCTA-supported path. This deliberately excludes C6/LCX geometry from the search so the earlier artificial common graph anchor cannot define the seed.

Calibration: the same source-search machinery starts 6 mm inside the accepted RCA and must retrace the known proximal RCA to the aortic root. The RCA control additionally requires close agreement with the known proximal RCA path.

Left search: automatically orient the accepted LAD so the endpoint nearest the aorta is proximal, use the reverse local LAD tangent as the initial direction, and require monotonic approach to the aorta with source HU, multiscale vesselness, coronary-mask proximity, low tortuosity, and source support.

A positive result is only a **left-coronary ostial/proximal-trunk candidate** for visual QC. It does not establish LM, does not prove where the LCX-like C6 joins, and does not modify Master Coronary Anatomy Baseline v2.1.
