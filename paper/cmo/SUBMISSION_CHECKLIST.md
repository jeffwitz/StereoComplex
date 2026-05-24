# Submission checklist — Rayfield-Based CMO Microscope Calibration

## Manuscript: `paper/cmo/manuscript.tex`

- [x] PDF compiles clean (0 LaTeX warnings, 0 undefined refs, 0 undefined citations)
- [x] All figures present and embedded (7 figures: cmo_physical, pipeline, bic_bars,
      BIC_vs_order, dy_profile_comparison, residual_evolution, subpupil_3d,
      schur_singular_values, pareto_gauge_regularization)
- [x] All `\cite{}` resolve (43 references, bibtex compiles)
- [x] All `\ref{}` match `\label{}` (verified via LaTeX clean build)
- [x] Author affiliation: CNRS, Univ. Lille, Centrale Lille, LaMcube
- [x] ORCID: 0000-0002-7240-9476
- [x] Corresponding author email: jean-francois.witz@centralelille.fr
- [x] Data availability statement: GitHub + Zenodo DOI placeholder
- [x] Reproducibility statement: commit hash 60272b7, intermediate_state.npz
- [x] Funding: CNRS (no specific grant number)
- [x] No undefined acronyms: CMO, BA, BIC, AIC, TPS, SE(3), RMS defined on first use
- [x] Pages: 23 (within limits)
- [x] Word count: ~10,500 words (abstract + body)
- [x] Cover letter: `paper/cmo/cover_letter.txt`

## Figures

| Figure | File | DPI | Caption |
|---|---|---|---|
| 1 | cmo_physical.png | static | CMO optical layout |
| 2 | pipeline.png | 200 | Pipeline overview |
| 3 | dy_profile_comparison.png | 200 | dy profiles |
| 4 | bic_bars.png | 200 | BIC bar chart (2 panels) |
| 5 | BIC_vs_order.png | 200 | Zernike order selection |
| 6 | residual_evolution.png | 200 | Residual evolution (3 panels) |
| 7 | subpupil_3d.png | 200 | Sub-pupil 3D reconstruction |
| 8 | schur_singular_values.png | 200 | Schur complement singular values |
| 9 | pareto_gauge_regularization.png | 200 | Pareto frontier |

## Reproducibility

- [x] Code: https://github.com/jeffwitz/StereoComplex (commit 60272b7)
- [x] Dataset: Pycaso (https://github.com/LaboratoireMecaniqueLille/Pycaso)
- [x] Pre-computed state: intermediate_state.npz (restart without raw images)
- [x] Zenodo DOI: 10.5281/zenodo.XXXXXXX (to be filled after archival)

## Pre-submission to-do (human)

- [ ] Fill Zenodo DOI (archive repository)
- [ ] Confirm contact email is active
- [ ] Optional: contact Pycaso authors for nominal specs
- [ ] Archive paper/cmo/ with Zenodo
