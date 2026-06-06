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
make clean      # remove build/
```

The figures are versioned artefacts, so `make repro` builds the paper without
any heavy data. To *regenerate* them, `make figures` needs the specimen `.npz`
present — produce them with `make specimen` (the Zenodo copy predates the
full-image regeneration and is stale for the current numbers).

## What is versioned vs. fetched vs. recomputed

| Artefact | Where it lives | How to (re)create |
|---|---|---|
| Calibration state `intermediate_state.npz`, `zernike_pose_variants.json`, order/pose sweeps, corner-BA JSONs | git (`docs/assets/pycaso_real_data/`) | versioned input |
| Corner Schur-BA **calibrations & diagnostics** `schur_ba/*.json` (thetas, sweeps, coupling sensitivity) | git (≈70 KB) | versioned input; `make schur` recomputes (~2 h) |
| Dense specimen **reconstructions** `schur_ba/specimen_*.npz`, `specimen_reconstruction_*.npz` | heavy, **not** in git | `make fetch` (Zenodo) **or** `make specimen` (local) |
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
calibration reconstructions, needs the Pycaso raw images), then `figures`,
`pdf`, `audit`. Use this only to rebuild the heavy `.npz` from scratch; the
default `make repro` reuses the versioned/fetched data.

## Notes

- The specimen reconstructions are produced by
  `examples/notebooks/regenerate_specimen_reconstructions.py` — DIS flow on the
  **full** image, ROI extracted from the field (no holes). `--dis ultrafast`
  reproduces the original ROI configuration for validation.
- `make audit` skips a check when its heavy source JSON is absent (it prints
  `SKIP` and the command to regenerate it) instead of failing.
