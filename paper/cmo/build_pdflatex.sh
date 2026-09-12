#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 analysis/build_diagnostic_assets.py
python3 check_manuscript_numbers.py
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
