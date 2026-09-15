# Final integrated OpenPlaque research report

This experiment is intentionally created directly from the frozen baseline
`0593b453959f5a353d644267fbeef24b514ef4d7`.

It consolidates only completed, validated upstream artifacts:

- canonical source-CCTA RCA/LAD/secondary anatomy,
- 5-fold native curved-series plaque-confidence atlas,
- canonical curved-series registration,
- locked RCA PCAT 10–50 mm analysis,
- corrected longitudinal plaque + PCAT fusion.

Scientific boundary:

- plaque ensemble volumes remain native curved-series vote volumes, not source-space TPV;
- validated registration is longitudinal only;
- no circumferential/radial or 3-D source-space plaque localization is asserted;
- no LAD plaque-PCAT relationship is asserted;
- OpenPlaque PCAT attenuation is not Caristo FAI-Score.

The output is a self-contained HTML research report, summary JSON, metrics CSV,
provenance JSON, generated figures, and ZIP package.
