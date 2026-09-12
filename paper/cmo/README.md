# Diagnostic manuscript

Current article: **Ray-field residuals as a diagnostic for compact stereo-microscope calibration**.

- [Main PDF](manuscript.pdf): 8 pages, 4 figures, 2 tables.
- [Supplement](supplementary.pdf): 4 pages.
- [French assessment and editorial decisions](REORIENTATION.md).
- [Executed results](results/diagnostic_audit.json).

## Reproduce from the repository root

```bash
python3 -m pip install -r paper/cmo/analysis/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 paper/cmo/analysis/diagnostic_audit.py
bash paper/cmo/build_pdflatex.sh
```

The build requires a working LaTeX installation with latexmk, lmodern, amsmath,
booktabs, natbib, microtype and hyperref. It regenerates the scientific figures
and numeric tables, checks them, and compiles both documents. The numerical
experiment itself takes only seconds on a CPU with the cached inputs.

The 17 x 13 training and 32 x 24 reserved grids assess compression of the SAME
canonical ray field. They do not provide independent physical observations.
The inverse-pixel check uses processed/completed fitted observations with common
archived poses. Forty synthetic cases test a deliberately specified rigid /
quadratic perturbation. No new physical measurements or raw-image detections
are included in this analysis.

`analysis/diagnostic_audit.py` records input SHA-256 hashes and the base revision.
`results/fitted_models.npz` contains the fitted compact coefficients;
`results/diagnostic_maps.npz` contains the reserved-grid and plotting data.
The test `test_full_grid_not_counted_twice_without_distinct_support` protects the
fixed residual-duplication bug. The original likelihood-based family ranking
is not used in the new paper.

Historical figures with names other than `diagnostic_*` remain as provenance
for the earlier manuscript and its notebook generators. They are not current
article figures. The original long text and obsolete results remain accessible
in Git history; `REORIENTATION.md` explains why their principal claims changed.
The separate `codex/strain-rayfield-validation` branch contains the earlier
proposal about reconstruction and false-strain errors.
