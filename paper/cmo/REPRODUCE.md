# Reproducing the CMO paper

Everything is driven by the `Makefile` in this directory. Run targets from
`paper/cmo/` (they `cd` to the repo root themselves).

## One command

```bash
make repro
```

Compiles the manuscript to `build/manuscript.pdf` from the **versioned figures**
and runs the numerical audit (every story-level number is cross-checked against
the versioned assets). Minutes, offline, no raw images, no 2-hour BA.

```bash
make pdf        # just compile the manuscript
make audit      # cross-check every manuscript number against the assets
make figures    # REGENERATE all manuscript figures (needs the heavy specimen .npz)
make bundle     # assemble the self-contained Zenodo archive (see ZENODO_UPDATE.md)
make clean      # remove build/
```

The figures are versioned artefacts, so `make repro` builds the paper without
any heavy data. **`make figures` also runs from a fresh clone alone**: the
specimen reconstructions it needs (`specimen_reconstruction_*.npz`,
`specimen_correspondences.npz`) are versioned in git (verified end-to-end on a
clean `git clone`). `make specimen` is only needed to *re-derive* those `.npz`
from the raw Pycaso images.

## Reproduce the exact submitted version

The repository `develop` branch tracks the latest state. To rebuild the precise
version archived with the submission (v8 = `10.5281/zenodo.21025322`), fetch its
bundle into an **empty directory** (a git checkout would rebuild itself, not the
archived version). The v8 bundle is self-contained — it ships `src/` and
`pyproject.toml`:

```bash
mkdir -p /tmp/cmo_v8 && cd /tmp/cmo_v8
curl -sL "https://zenodo.org/records/21025322/files/cmo_paper_bundle.zip" -o bundle.zip
unzip -q bundle.zip                  # extracts the v8 tree + BUNDLE_MANIFEST.json
bash rebuild_from_zenodo.sh          # verifies every file against the manifest

# environment (the bundle has no .venv / rtk; override PY):
python -m venv .venv
.venv/bin/pip install -e ".[notebooks]"   # core deps + matplotlib for the figures
make -C paper/cmo repro   PY=.venv/bin/python   # PDF + audit
make -C paper/cmo figures PY=.venv/bin/python   # regenerate every figure + table
```

> **Earlier versions.** v5 (`20574710`) and below do **not** ship `src/`: from
> them `make repro` (PDF + audit) works standalone, but `make figures` needs the
> codebase (`pip install stereocomplex` or clone at the version's pinned commit).
> Use v6 or later for a fully self-contained rebuild.

From *inside* an existing clone, force the download instead of using the local
checkout with `FORCE_ZENODO=1 ZENODO_RECORD=21025322 bash rebuild_from_zenodo.sh`.
Without an override, `rebuild_from_zenodo.sh` in a clone rebuilds the checkout,
and in an empty dir it resolves the concept DOI `10.5281/zenodo.20444215` (the
latest published version).

## What is versioned vs. fetched vs. recomputed

| Artefact | Where it lives | How to (re)create |
|---|---|---|
| Calibration state `intermediate_state.npz`, `zernike_pose_variants.json`, order/pose sweeps, corner-BA JSONs | git (`docs/assets/pycaso_real_data/`) | versioned input |
| Corner Schur-BA **calibrations & diagnostics** `schur_ba/*.json` (thetas, sweeps, coupling sensitivity) | git (≈70 KB) | versioned input; `make schur` recomputes (~2 h) |
| Manuscript-figure **reconstructions** `specimen_reconstruction_*.npz`, `specimen_correspondences.npz` (Fig. specimen + rigid-removal) | git (~62 MB) | versioned input; `make specimen` re-derives them from the raw images |
| Stage-prior experiment `figure10b_zernike_stage_prior/{stage_prior_results.json, near_hard_zernike_rayfield.npz}` (Fig. 10b + the near-hard rayfield consumed by the profilometry figure) | git (~6 KB) | versioned input; `make stageprior` re-derives them from the raw images |
| Orphaned five-variant `schur_ba/specimen_*.npz` (~120 MB) | **not** in git | `make fetch` (Zenodo `20369312`) — **optional**, no manuscript figure uses them |
| Raw Pycaso coin/ChArUco images | external (git-ignored) | [Pycaso dataset](https://github.com/LaboratoireMecaniqueLille/Pycaso) |
| Figures `figures/*.{pdf,png}` | git | `make figures` |

The small calibration/diagnostic JSONs are kept **in git** precisely so that no
manuscript number depends on the 2-hour bundle adjustment or on a Zenodo round
trip: `make figures` + `make audit` run from the repository alone.

## Full reproduction from raw inputs

```bash
make repro-full
```

Runs, in order: `schur` (corner Schur-BA: diagnostic + free-pose BA, ~2 h, needs
`intermediate_state.npz`), `specimen` (full-image DIS optical flow + the five
calibration reconstructions, needs the Pycaso raw images), `stageprior` (the
hierarchical translation-stage prior experiment producing the Fig. 10b JSON and
the near-hard Zernike rayfield, needs the raw images), then `figures`, `pdf`,
`audit`. Use this only to rebuild the heavy `.npz` from scratch; the default
`make repro` reuses the versioned/fetched data.

The calibration state uses the ten explicit stage positions
`2.65, 2.72, 2.79, 2.86, 2.93, 3.00, 3.07, 3.21, 3.28, 3.35` mm. The auxiliary
`3.014` mm image remains available for preprocessing validation but is not a
calibration pair.

## Notes

- The specimen reconstructions are produced by
  `examples/notebooks/regenerate_specimen_reconstructions.py` — DIS flow on the
  **full** image, ROI extracted from the field (no holes). `--dis ultrafast`
  reproduces the original ROI configuration for validation.
- `make audit` skips a check when its heavy source JSON is absent (it prints
  `SKIP` and the command to regenerate it) instead of failing.

## Bibliography style

`manuscript.tex` uses `plainurl.bst` (urlbst-generated, GPL, shipped in
TeX Live's `texlive-bibtex-extra` on Debian/Ubuntu). A copy is committed
here (`paper/cmo/plainurl.bst`) and picked up via `BSTINPUTS` in
`build_pdflatex.sh`, so the build is self-contained on a minimal TeX Live.
