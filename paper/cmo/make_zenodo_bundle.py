#!/usr/bin/env python3
"""Assemble the self-contained Zenodo archive for the CMO paper.

Collects every file needed to rebuild the manuscript PDF and regenerate every
figure and table *from this repository alone* into a structured staging tree
that mirrors the repo layout, writes a ``BUNDLE_MANIFEST.json`` (repo-relative
path + size + sha256 of each file) and zips it to
``paper/cmo/build/cmo_paper_bundle.zip``.

Because the zip preserves the repository structure, the companion
``rebuild_from_zenodo.sh`` only has to unzip it: a third party can then run
``cd paper/cmo && make repro`` with no git clone and no file-renaming dance.
This replaces the previous flat 100-file layout, which hard-coded a figure list
that drifts every time the figure set changes.

The bundle ships the package source (``src/``) and ``pyproject.toml`` so it is
self-contained for *both* gates: ``make repro`` (PDF + audit) needs no package,
but ``make figures`` does (e.g. ``generate_fig_cmo_physical.py`` imports
``stereocomplex.viz`` at module level). A bundle user therefore runs
``python -m venv .venv && .venv/bin/pip install -e ".[notebooks]"`` (the
``notebooks`` extra pulls in matplotlib) and then ``make ... PY=.venv/bin/python``.

Scope note. The heavy five-variant Schur-BA reconstructions
(``docs/assets/pycaso_real_data/schur_ba/specimen_*.npz``, ~120 MB) are **not**
bundled: the manuscript no longer references the five-variant figure, so no
paper artefact depends on them. They are handled, with the other large data, by
the separate data record (Zenodo 10.5281/zenodo.20369312) via
``examples/zenodo_upload.py``.

Run
---
    rtk .venv/bin/python paper/cmo/make_zenodo_bundle.py
    rtk .venv/bin/python paper/cmo/make_zenodo_bundle.py --verify   # round-trip check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "paper/cmo/build/zenodo_bundle"
ZIP = REPO / "paper/cmo/build/cmo_paper_bundle.zip"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"

# Repo-relative include set. ``dir/**`` includes a directory recursively; a glob
# with ``*`` is expanded relative to the repo root; a plain path is a single
# file. Everything is staged at its repo-relative path (structure preserved).
INCLUDE = [
    # --- package source + packaging (self-contained: several figure scripts
    #     import stereocomplex at module level, e.g. generate_fig_cmo_physical.py;
    #     ship src so `pip install -e .` rebuilds everything from the bundle alone) ---
    "src/**",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    # --- manuscript sources + built PDF ---
    "paper/cmo/manuscript.tex",
    "paper/cmo/manuscript.pdf",
    "paper/cmo/references.bib",
    "paper/cmo/build_pdflatex.sh",
    "paper/cmo/Makefile",
    "paper/cmo/REPRODUCE.md",
    "paper/cmo/SUBMISSION_CHECKLIST.md",
    "paper/cmo/cover_letter.txt",
    "paper/cmo/make_zenodo_bundle.py",
    "paper/cmo/figures/*",
    "paper/cmo/tables/*.tex",
    # --- figure asset folders (data + manifests + READMEs) ---
    "docs/assets/cmo_paper/**",
    # --- reproducibility scripts ---
    "examples/notebooks/generate_fig_*.py",
    "examples/notebooks/generate_tab_*.py",
    "examples/notebooks/audit_paper_numbers.py",
    "examples/notebooks/sensitivity_coupling_norm.py",
    "examples/notebooks/parameter_identifiability.py",
    "examples/notebooks/regenerate_specimen_reconstructions.py",
    "examples/zenodo_fetch.py",
    "examples/zenodo_upload.py",
    "examples/pycaso_schur_regularized_ba.py",
    "rebuild_from_zenodo.sh",
    # --- key data needed by the figures + numerical audit ---
    "docs/assets/pycaso_real_data/intermediate_state.npz",
    "docs/assets/pycaso_real_data/specimen_correspondences.npz",
    "docs/assets/pycaso_real_data/specimen_reconstruction_cmo26.npz",
    "docs/assets/pycaso_real_data/specimen_reconstruction_zernike.npz",
    "docs/assets/pycaso_real_data/*.json",
    "docs/assets/pycaso_real_data/schur_ba/*.json",
]


def _expand(pattern: str) -> list[Path]:
    """Expand one INCLUDE entry into the list of existing files it matches."""
    if pattern.endswith("/**"):
        base = REPO / pattern[:-3]
        return sorted(p for p in base.rglob("*") if p.is_file())
    if "*" in pattern:
        return sorted(p for p in REPO.glob(pattern) if p.is_file())
    p = REPO / pattern
    return [p] if p.is_file() else []


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[Path]:
    """Return the de-duplicated, sorted list of files to bundle (repo paths)."""
    files: set[Path] = set()
    for pat in INCLUDE:
        files.update(_expand(pat))
    return sorted(files)


def build() -> None:
    """Stage the files, write the manifest, and zip the structured bundle."""
    files = collect()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    manifest = []
    total = 0
    for src in files:
        rel = src.relative_to(REPO)
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = src.stat().st_size
        total += size
        manifest.append({"path": str(rel), "bytes": size, "sha256": _sha256(src)})

    (STAGE / MANIFEST_NAME).write_text(json.dumps(
        {"n_files": len(manifest), "total_bytes": total, "files": manifest}, indent=2))

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(STAGE.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(STAGE))

    print(f"  staged {len(manifest)} files, {total / 1e6:.1f} MB -> {STAGE.relative_to(REPO)}")
    print(f"  wrote  {ZIP.relative_to(REPO)} ({ZIP.stat().st_size / 1e6:.1f} MB)")
    print(f"  manifest: {(STAGE / MANIFEST_NAME).relative_to(REPO)}")


def verify() -> int:
    """Round-trip check: every manifest entry is in the zip with the right hash."""
    if not ZIP.exists():
        raise SystemExit("bundle zip missing — run without --verify first")
    manifest = json.loads((STAGE / MANIFEST_NAME).read_text())["files"]
    bad = 0
    with zipfile.ZipFile(ZIP) as zf:
        names = set(zf.namelist())
        for entry in manifest:
            rel = entry["path"]
            if rel not in names:
                print(f"  MISSING in zip: {rel}")
                bad += 1
                continue
            h = hashlib.sha256(zf.read(rel)).hexdigest()
            if h != entry["sha256"]:
                print(f"  HASH MISMATCH: {rel}")
                bad += 1
    # also re-hash the source to confirm the manifest matches the working tree
    for entry in manifest:
        src = REPO / entry["path"]
        if src.is_file() and _sha256(src) != entry["sha256"]:
            print(f"  WORKING-TREE DRIFT: {entry['path']}")
            bad += 1
    print(f"  verify: {len(manifest)} entries, {bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check the existing zip against the manifest and working tree")
    args = ap.parse_args()
    if args.verify:
        return verify()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
