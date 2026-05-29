# Submission checklist — Rayfield-Based CMO Microscope Calibration

## Manuscript: `paper/cmo/manuscript.tex`

- [x] PDF compiles clean (0 LaTeX warnings, 0 overfull hboxes, 0 undefined citations)
- [x] All figures present and embedded (12 figures across 31 pages)
- [x] All `\cite{}` resolve (36 references in bibtex, all cited)
- [x] All `\ref{}` match `\label{}` (verified via LaTeX clean build)
- [x] Author affiliation: CNRS, Univ. Lille, Centrale Lille, LaMcube
- [x] ORCID: 0000-0002-7240-9476
- [x] Corresponding author email: jean-francois.witz@centralelille.fr
- [x] Data availability statement: GitHub + Zenodo DOI placeholder
- [x] Reproducibility statement: commit hash 93110db, intermediate_state.npz
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
| 3 | BIC_vs_order.png | 200 | Zernike order selection (BIC vs N) |
| 4 | subpupil_3d.png | 200 | Sub-pupil 3D reconstruction |
| 5 | dy_profile_comparison.png | 200 | dy residual profiles |
| 6 | residual_evolution.png | 200 | Residual evolution (3 panels) |
| 7 | pareto_gauge_regularization.png | 200 | Pareto frontier |
| 8 | schur_singular_values.png | 200 | Schur complement singular values |
| 9 | specimen_reconstruction.png | 200 | Specimen sanity check |
| 10 | zernike_cmo_rigid_removed.png | 200 | Rigid-gauge removal comparison |
| 11 | specimen_schur_regularized.png | 200 | Specimen reconstruction (Schur regularised) |
| 12 | bic_bars.png | 200 | BIC bar chart (model selection) |

## Reproducibility

- [x] Code: https://github.com/jeffwitz/StereoComplex (commit 93110db)
- [x] Dataset: Pycaso (https://github.com/LaboratoireMecaniqueLille/Pycaso)
- [x] Pre-computed state: intermediate_state.npz (restart without raw images)
- [x] Zenodo DOI: 10.5281/zenodo.20444216 (90 files, manuscript + all reproducibility assets)

## Pre-submission to-do (human)

- [x] Fill Zenodo DOI (archive repository) — 10.5281/zenodo.20444216
- [ ] Confirm contact email is active
- [ ] Optional: contact Pycaso authors for nominal specs
- [ ] Archive paper/cmo/ with Zenodo
