"""Helper to fetch the schur_ba/ archive from Zenodo when not present locally.

Usage:
    from stereocomplex_examples.zenodo_fetch import ensure_schur_ba
    ensure_schur_ba()

If the directory ``docs/assets/pycaso_real_data/schur_ba/`` exists locally,
nothing happens.  Otherwise the archive is downloaded from Zenodo DOI
10.5281/zenodo.20369312 and extracted.
"""

import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

ZENODO_DOI = "10.5281/zenodo.20369312"
ZENODO_RECORD = ZENODO_DOI.split('.')[-1]
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

SCHUR_BA_FILES = [
    "specimen_cmo_26p_initial.npz",
    "specimen_cmo_26p_ba_unregularized.npz",
    "specimen_cmo_26p_ba_iso_a1e-2.npz",
    "specimen_cmo_26p_ba_schur_a1e-3.npz",
    "specimen_zernike_rayfield_57p.npz",
]
# Find repo root: go up until we find CLAUDE.md
REPO_ROOT = Path(__file__).resolve().parent
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "CLAUDE.md").exists():
        break
    REPO_ROOT = REPO_ROOT.parent
SCHUR_BA_DIR = REPO_ROOT / "docs" / "assets" / "pycaso_real_data" / "schur_ba"


def ensure_schur_ba(quiet: bool = False) -> bool:
    """Ensure schur_ba/ exists locally, downloading from Zenodo if needed.

    Returns True if the directory is now available (already existed or was
    downloaded successfully).
    """
    if SCHUR_BA_DIR.exists():
        if not quiet:
            print(f"schur_ba/ already present at {SCHUR_BA_DIR}")
        return True

    print(f"schur_ba/ not found — downloading from Zenodo (DOI {ZENODO_DOI})…")
    try:
        SCHUR_BA_DIR.mkdir(parents=True, exist_ok=True)
        for fname in SCHUR_BA_FILES:
            url = f"{ZENODO_URL}/{fname}"
            print(f"  downloading {fname}…")
            req = Request(url)
            with urlopen(req, timeout=120) as resp:
                (SCHUR_BA_DIR / fname).write_bytes(resp.read())
        print(f"Downloaded and extracted to {SCHUR_BA_DIR}")
        return True
    except Exception as exc:
        print(f"Failed to download schur_ba/ from Zenodo: {exc}", file=sys.stderr)
        print(f"Computations that depend on pre-computed schur_ba/ artefacts will not run.", file=sys.stderr)
        print(f"You can still run the full computation with --mode ba (2 hours).", file=sys.stderr)
        return False


if __name__ == "__main__":
    ensure_schur_ba()
