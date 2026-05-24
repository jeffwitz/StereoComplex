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
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request

ZENODO_DOI = "10.5281/zenodo.20369312"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_DOI.split('.')[-1]}/files/schur_ba.zip"
REPO_ROOT = Path(__file__).resolve().parents[2]  # examples/ → repo root
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
        req = Request(ZENODO_URL)
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(BytesIO(data)) as zf:
            SCHUR_BA_DIR.mkdir(parents=True, exist_ok=True)
            zf.extractall(SCHUR_BA_DIR)
        print(f"Downloaded and extracted to {SCHUR_BA_DIR}")
        return True
    except Exception as exc:
        print(f"Failed to download schur_ba/ from Zenodo: {exc}", file=sys.stderr)
        print(f"Computations that depend on pre-computed schur_ba/ artefacts will not run.", file=sys.stderr)
        print(f"You can still run the full computation with --mode ba (2 hours).", file=sys.stderr)
        return False


if __name__ == "__main__":
    ensure_schur_ba()
