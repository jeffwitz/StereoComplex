# Paper workspace (target: Experimental Mechanics)

This folder contains a working draft of a journal article describing the project and the current implemented baseline.

## Contents

- `paper/manuscript.tex`: main LaTeX draft (single-file for now)
- `paper/references.bib`: BibTeX database (initial curated refs + placeholders)
- `paper/figures/`: figures (exported plots/screenshots), keep filenames stable
- `paper/tables/`: tables (CSV or LaTeX snippets)

## Build (optional)

If you have a LaTeX distribution installed (pdflatex + bibtex):

```bash
bash paper/build_pdflatex.sh
```

## Notes

We are not using the official Springer/Experimental Mechanics template yet (no network access here); once the target template is chosen locally, `manuscript.tex` can be migrated.

