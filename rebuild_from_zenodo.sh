#!/usr/bin/env bash
# Rebuild the CMO paper from its self-contained Zenodo archive.
#
# From v5 on, the archive is a STRUCTURED bundle (cmo_paper_bundle.zip) that
# preserves the repository layout, plus the standalone manuscript.pdf. Rebuilding
# is therefore just "unzip and run make" — no file-renaming dance, no hard-coded
# figure list (the old flat 100-file layout drifted every time a figure changed).
# The bundle is produced by paper/cmo/make_zenodo_bundle.py and is self-describing
# via BUNDLE_MANIFEST.json (repo-relative path + sha256 for every file).
#
# Usage:
#   A) From an already-extracted bundle (BUNDLE_MANIFEST.json in CWD):
#        unzip cmo_paper_bundle.zip && bash rebuild_from_zenodo.sh   # verifies hashes
#   B) From an EMPTY directory (downloads the bundle from Zenodo):
#        bash rebuild_from_zenodo.sh                                  # concept DOI = latest
#        ZENODO_RECORD=20574710 bash rebuild_from_zenodo.sh           # a specific version
#   C) From a git clone (uses the local checkout, does NOT download):
#        cd StereoComplex && bash rebuild_from_zenodo.sh
#      To fetch the archived bundle instead of the checkout, force the download:
#        FORCE_ZENODO=1 ZENODO_RECORD=20574710 bash rebuild_from_zenodo.sh
#
# After running:
#   cd paper/cmo && make repro    # rebuild manuscript.pdf + audit every number
#   make figures                  # regenerate every figure and table

set -euo pipefail

# Concept-DOI record: the Zenodo landing page that always resolves to the latest
# version. Override with ZENODO_RECORD=<id> to pin a specific version.
ZENODO_RECORD="${ZENODO_RECORD:-20444215}"
BUNDLE="cmo_paper_bundle.zip"

# ── Acquire the bundle ────────────────────────────────────────────────
# Precedence: an already-extracted bundle wins; then FORCE_ZENODO forces a
# download even inside a git checkout (for exact-version reproduction); then a
# local zip; then a git checkout uses its own files; otherwise download.
if [ -f BUNDLE_MANIFEST.json ]; then
    echo "=== Extracted bundle already present ==="
elif [ "${FORCE_ZENODO:-0}" = "1" ]; then
    echo "=== FORCE_ZENODO=1: downloading $BUNDLE from Zenodo record $ZENODO_RECORD ==="
    curl -sL "https://zenodo.org/records/${ZENODO_RECORD}/files/${BUNDLE}" -o "$BUNDLE"
    unzip -qo "$BUNDLE"
elif [ -f "$BUNDLE" ]; then
    echo "=== Extracting local $BUNDLE ==="
    unzip -qo "$BUNDLE"
elif [ -f paper/cmo/manuscript.tex ]; then
    echo "=== Git checkout present; using local files ==="
    echo "    (this rebuilds the CHECKOUT, not a Zenodo version; to fetch the"
    echo "     archived bundle instead, run in an empty dir or set FORCE_ZENODO=1)"
else
    echo "=== Downloading $BUNDLE from Zenodo record $ZENODO_RECORD ==="
    curl -sL "https://zenodo.org/records/${ZENODO_RECORD}/files/${BUNDLE}" -o "$BUNDLE"
    unzip -qo "$BUNDLE"
fi

# ── Verify against the manifest (sha256), if Python is available ───────
if [ -f BUNDLE_MANIFEST.json ] && command -v python3 >/dev/null 2>&1; then
    echo "=== Verifying files against BUNDLE_MANIFEST.json ==="
    python3 - <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path("BUNDLE_MANIFEST.json").read_text())["files"]
bad = 0
for e in m:
    p = Path(e["path"])
    if not p.is_file():
        print(f"  MISSING: {e['path']}"); bad += 1; continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
        print(f"  HASH MISMATCH: {e['path']}"); bad += 1
print(f"  {len(m)} files, {bad} problem(s)")
sys.exit(1 if bad else 0)
PY
fi

# ── Heavy specimen reconstructions (separate data record) ─────────────
# The five-variant Schur-BA .npz (~120 MB) are NOT in this bundle: the manuscript
# no longer references the five-variant figure, so no paper artefact needs them.
# They remain archived for completeness at https://doi.org/10.5281/zenodo.20369312
# (fetch with examples/zenodo_fetch.py if you want to re-run that analysis script).

echo ""
echo "=== Done. Next steps: ==="
echo "  cd paper/cmo && make repro      # rebuild the PDF + run the numerical audit"
echo "  make figures                    # regenerate every figure and table"
