# Paper workspace

This folder contains working drafts of journal articles describing the
StereoComplex project.

## Multi-paper layout

```
paper/
├── compression/         ← "Compression-Robust Stereo Calibration"
│   ├── manuscript.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── cmo/                 ← "Rayfield-Based CMO Microscope Calibration"
│   ├── manuscript.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── figures/             ← shared figures (symlinks or copies)
├── tables/              ← shared tables
└── README.md
```

## Build

Each paper has its own `build_pdflatex.sh`:

```bash
cd paper/compression && bash build_pdflatex.sh
cd paper/cmo && bash build_pdflatex.sh
```

No external LaTeX template is required — all papers use standard
`article` class with common packages (booktabs, siunitx, amsmath, graphicx).

## Target journals

- `compression/`: Experimental Mechanics
- `cmo/`: Experimental Mechanics or Optics Express

## CMO paper status (2026-05-17)

24 pages, 43 references, 9 figures, 7 tables. Phases 1-6 complete.

The CMO paper now includes:
- Full methodological pipeline (Ray2D preprocessing → Zernike rayfield BA → BIC model selection)
- 26-parameter CMO+SE(3) physical model achieving 1.06 px on real Pycaso data
- Operational BIC with 1.5 px reprojection guard
- Internal validation (cross-val, bootstrap, fx sensitivity)
- **Specimen reconstruction sanity check**: dense stereo on independent speckled
  coin specimen (DIS optical flow, 2M correspondences, 0.107 mm median ray gap).
  Not absolute metrology — confirms rayfield coherence on unseen data.

Tags: `v0.6.0-cmo-paper`, `v1.0-submission-ready`
