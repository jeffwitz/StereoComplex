#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p build

pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex

# When using -output-directory, BibTeX reads build/manuscript.aux but needs to find references.bib in paper/.
# Provide a local BIBINPUTS so "references.bib" resolves.
if command -v bibtex >/dev/null 2>&1; then
  BIBINPUTS="${ROOT_DIR}:" bibtex build/manuscript || true
fi

pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build manuscript.tex

echo "Built: ${ROOT_DIR}/build/manuscript.pdf"

# Keep the standalone PDF in sync with the build. The Zenodo bundle
# (make_zenodo_bundle.py) and the upload command reference paper/cmo/manuscript.pdf,
# NOT build/manuscript.pdf — without this copy the archived standalone PDF can go stale.
cp build/manuscript.pdf manuscript.pdf
echo "Synced: ${ROOT_DIR}/manuscript.pdf"

