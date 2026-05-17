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
