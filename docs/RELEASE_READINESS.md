# Release readiness — v0.1.0-alpha

**Target:** snapshot `origin/develop` as a publishable, merge-ready baseline.
**Version:** v0.1.0-alpha (unless JFW changes it).

## Stable APIs (semver — no breaking changes in patch/minor)

These are the validated production paths.  A user following the tutorial
(`docs/TUTORIAL.md`) stays entirely inside this surface.

| API surface | Entry point | Validated |
|---|---|---|
| OpenCV stereo path | `api.calibration.fit_opencv_stereo_from_image_dirs` | ✅ slow tests |
| Ray2D corner refinement | `eval.predictors.predict_charuco_points` | ✅ benchmark |
| Central stereo rayfield | `api.calibration.fit_stereo_central_rayfield_from_image_dirs` | ✅ slow tests |
| Non-central Zernike origin field | `calibration.fit_stereo_zernike_origin_field` | ✅ benchmark |
| Model import / export | `api.model_io.{save,load}_stereo_central_rayfield` | ✅ round-trip test |
| Reconstruction diagnostics | `metrics.reconstruction_metrics.reconstruct_points_central_stereo` | ✅ benchmark |

## Advanced APIs (stable interface, expert use)

These are validated but require domain knowledge (optical models, gauge
conventions, parameter layout).  Breaking changes may happen after deprecation
notice.

| API surface | Entry point |
|---|---|
| Physical model selection | `physics.model_selection.fit_physical_model_to_rayfield` |
| CMO physical model | `physics.cmo_physical.fit_cmo_physical_stereo_model_to_rayfields` |
| CMO telecentric model | `physics.cmo_physical.fit_cmo_telecentric_model_to_rayfields` |
| CMO warped model | `physics.cmo_physical.fit_cmo_warped_model_to_rayfields` |
| Schur diagnostics | `optical_ba.regularized_ba` |
| Paper reproduction helpers | `benchmarks/`, `synthetic/` |

## Experimental APIs (no stability promise)

These exist as scaffolding for future work.  Signatures, parameter layouts,
and return types will change without notice.

| API surface | Status |
|---|---|
| N-camera facades (`api.calibration.calibrate(cameras=...)`) | Routes only `("left","right")`; other topologies raise `NotImplementedError` |
| `MultiCameraCharucoObservationSet` | Container only; real N-rayfield BA not implemented |
| `MultiCameraZernikeRayField` | Container only |
| `CMOTelecentricNModel` | Wraps stereo model via `from_stereo()` |
| `fit_zernike_rayfields_from_multi_camera_observations` | Routes only `("left","right")` |
| N-camera oracles / simulators | Benchmark scaffolding only |
| `optical_ba/` scripts | Paper-specific, not general-purpose |

**Explicit non-goals for v0.1.0-alpha:**
- True N-camera bundle adjustment (Phase 2 — specification exists, not implemented)
- Shared-rig SE(3) constraints across >2 channels
- Production-grade multi-camera calibration from real images
- GUI / web dashboard

## Paper-only code

This code exists to reproduce the CMO paper.  It is not part of the library
API and may use hard-coded paths, deprecated helpers, or paper-specific
conventions.

| Location | Purpose |
|---|---|
| `paper/cmo/` | Manuscript source, build scripts, standalone figure generators |
| `docs/assets/pycaso_real_data/` | JSON/NPZ artefacts for paper figures |
| `examples/notebooks/09_pycaso_real_data.py` | Full paper analysis walkthrough |
| `examples/notebooks/generate_fig_*.py` | Standalone figure generators |

## Quality gates (all passing @ 2026-05-24)

| Gate | Command | Status |
|---|---|---|
| Lint | `rtk .venv/bin/python -m ruff check src/` | 0 errors |
| Docstring contract | `rtk .venv/bin/python examples/notebooks/check_docstring_params.py` | 0 alarms, 88 files |
| Fast tests | `rtk .venv/bin/python -m pytest` | 159 passed, 39 deselected |
| Slow tests | `rtk .venv/bin/python -m pytest -m slow` | 39 passed, 0 failures |
| Doc coverage | `100%` (425/425 public functions) | ✅ |

## Known heavy artefacts

| Artefact | Size | Strategy |
|---|---|---|
| `docs/assets/pycaso_real_data/` | ~50 MB | Keep in repo (needed for paper reproduction) |
| `paper/cmo/figures/*.pdf` | <2 MB total | Keep (vector, versioned) |
| `examples/notebooks/*.ipynb` | ~5 MB total | Keep (Colab badges point to develop) |
| `paper/cmo/build/manuscript.pdf` | ~2 MB | Keep (submission artefact) |

No Git LFS candidate identified at this stage.

## Pending before merge to main

1. Switch Colab links from `blob/develop/` to `blob/main/` (see CLAUDE.md § Colab branch hack)
2. Tag `v0.1.0-alpha` on the merge commit
3. Update `docs/START_HERE.md` to reflect that `develop` → `main` has happened
4. Archive paper artefacts on Zenodo and link from README
