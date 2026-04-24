#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if [[ "${VALIDATE_INSTALL:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install --no-build-isolation -e '.[docs]'
fi
make -C docs html
