#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if [[ "${VALIDATE_INSTALL:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e '.[dev]'
fi
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m ruff check src tests
if [[ -f docs/_build/html/index.html ]]; then
  "${PYTHON_BIN}" scripts/check_docs_nav.py
fi
