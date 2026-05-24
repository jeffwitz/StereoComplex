# Changelog

## [0.7.0-alpha] - 2026-05-24

### Added

- **Publishable snapshot** (`docs/RELEASE_READINESS.md`): stable/advanced/experimental API boundaries, quality gates, known artefacts.
- **Zenodo archival**: Schur BA snapshots archived at DOI 10.5281/zenodo.20369312 with auto-download helper (`examples/zenodo_fetch.py`).
- **CI docstring guard**: `check_docstring_params.py` wired into `.github/workflows/ci.yml`.
- **Diátaxis tutorial**: 6-step walkthrough from install to CMO calibration.

### Changed

- **100% docstring coverage** (425/425 public functions) — all scientific core, instrumentation, CLI, and API modules documented with Parameters/Returns, units, and shapes.
- Repository weight reduced from 234 MB to 114 MB by archiving `schur_ba/` on Zenodo.
- **Ruff**: E/F=0, E501=0, cosmetic rules cleared.

## [0.7.0] - 2026-05-17

### Added

- **Pycaso specimen reconstruction sanity check** (Phase 4-6 of CMO paper):
  - Dense stereo reconstruction on independent speckled coin specimen (no ChArUco)
  - DIS optical flow: 2M correspondences over 1448×1448 ROI (95.4% valid)
  - Triangulation with CMO 26p and Zernike 57p models: median ray gap 0.107 mm
  - CMO 26p surface roughness (Z MAD) = 0.072 mm; Zernike = 0.176 mm
  - Honest caveat: internal consistency check, not absolute metrological validation
  - Saved: `specimen_correspondences.npz`, `specimen_reconstruction_cmo26.npz`,
    `specimen_reconstruction_zernike.npz`, `specimen_reconstruction_metrics.json`
  - New notebook: `10_pycaso_specimen_reconstruction.py`
  - New paper figure: `specimen_reconstruction.png` (2×3 panel)

## [0.5.3] - 2026-05-17

### Added

- **Phase 8 — Direct vs Rayfield Inversion infrastructure**:
  - Phase 1: ChArUco simulator densification (`min_corners_per_frame`, `max_pose_attempts`, `pose_jitter_deg`, `SamplingDiagnostics`) - 8 tests
  - Phase 2: Pipeline A direct inversion on CMO oracle (`estimate_initial_poses_from_central_pinhole`, `fit_direct_model_from_observations`) - 2 new tests
  - Phase 3: Zernike rayfield from ChArUco observations (`fit_zernike_rayfield_from_charuco_observations`) - 3 tests
  - Phase 4: Schur-complement diagnostics (`compute_inverse_problem_diagnostics`) - 5 tests
- **Notebook 08**: direct vs rayfield comparison pipeline, FAST mode support
- **CLAUDE.md**: Phase 8 status and deep-claude session reference

## [0.6.0] - 2026-05-11

### Added

- **OpenCV-user façade**:
  - Short function aliases: `sc.calibrate_opencv`, `sc.calibrate_central`,
    `sc.calibrate_noncentral`, `sc.identify_optics`.
  - `sc.compare_opencv_stereo_calibration()`: one-call raw vs Ray2D-refined comparison.
  - `sc.assess_calibration(result)` → `CalibrationAssessment(status, messages, recommendations)`.
  - `StereoOpenCVCalibrationResult.to_dict()` and `.to_opencv()` export methods.
- **Documentation**:
  - `docs/FROM_OPENCV_TO_STEREOCOMPLEX.md` — 3-minute quickstart for OpenCV users.
  - `docs/VALIDATION_STATUS.md` — synthetic/exploratory/real-data status matrix.
  - `docs/ROADMAP.md` — 5-phase user-facing API roadmap.
  - `docs/DIRECT_VS_RAYFIELD_INVERSION.md` reframed as methodological page.
  - "This page answers" callouts on 4 main doc pages.
  - `examples/reproduce_docs_results.py` — lightweight reproduction check.
  - `examples/notebooks/00_getting_started.py` — OpenCV onboarding notebook.
- **Analytic `project_point`** on `CentralPinholeModel`, `CentralBrownConradyModel`,
  and `CMOPhysicalChannelModel`, making pipeline A fast (~2–10 s per oracle).
- **6-oracle sweep**: pipeline B correctly classifies all 6 families;
  pipeline A converges on 4/6 (fails on CMO and Greenough due to pinhole init).

## [0.5.3] - 2026-05-10

### Added

- **Direct-vs-rayfield inversion study** (Phase 8):
  - `stereocomplex.benchmarks.charuco_observation_simulator` with rejection
    sampling and `SamplingDiagnostics`.
  - `stereocomplex.benchmarks.direct_inversion`: pipeline A with cv2.solvePnP
    pose initialisation and joint optical+pose BA.
  - `stereocomplex.benchmarks.rayfield_from_observations`: image-based Zernike
    rayfield fit (pipeline B) from the same ChArUco observations.
  - `compute_pipeline_condition_number` for Schur-complement conditioning
    comparison between pipelines A and B.
  - Notebook 08 (`examples/notebooks/08_direct_vs_rayfield_inversion.py`)
    comparing both pipelines on a CMO oracle.
  - Documentation page `docs/DIRECT_VS_RAYFIELD_INVERSION.md`.

### Changed

- Test suite split: 76 fast tests (default, ~28 s) + 37 slow tests
  (`-m ""` for all, ~9 min). 113 total.
- `simulate_charuco_observations_from_rayfield`: min_corners_per_frame
  rejection sampling (default 30), pose centring for better coverage.
- `DirectFitResult`: `success` → `converged`, added `n_iterations`.

## [0.5.2] - 2026-05-09

### Added

- Optical diagrams (pinhole, CMO, Greenough) in `stereocomplex.viz`.
- Clean alias-free API, zero warnings, 77 tests.

## [0.5.0] - 2026-05-09

### Added

- **`ZernikeCandidate`**: compact Zernike rayfield candidate (origin + direction,
  default `max_order=2`, ~36 params per stereo pair) as a model-selection fallback.
  Wins BIC when no physical model in the catalogue explains the rayfield, signalling
  that the optics fall outside the known families.
- **Classification matrix** (`examples/notebooks/07_model_selection_matrix.py`):
  validates BIC-based architecture identification on six oracles (pinhole, Brown,
  inclined plate, CMO, Greenough, uncatalogued Zernike), both noiseless and under
  20 µm Gaussian origin noise.
- **ΔBIC heatmaps** in `docs/assets/cmo_model_selection/` (noiseless and noisy
  variants), referenced as figures in `docs/CMO_MODEL_SELECTION.md`.
- **Three-regime noise analysis**: parsimony, structural mismatch, and exotic
  detection regimes identified under measurement noise.
- **Real microscope mapping table** in `docs/CMO_PHYSICAL_MODEL.md` covering Leica,
  Evident/Olympus, Zeiss, and Nikon CMO models, plus Greenough and other architectures.

### Changed

- **BREAKING**: `CMOPolynomialChannelModel` renamed to
  `NonCentralPolynomialChannelModel`. The old name no longer exists.
- **BREAKING**: `cmo_polynomial_channel_parameters_from_spec` renamed to
  `polynomial_channel_parameters_from_spec`. The old name no longer exists.
- `origin_z_mm` added as a free parameter in the polynomial surrogate
  (was hardcoded to 0), increasing parameter count from 17 to 18 per channel.
- `ZernikeOriginFieldConfig` now has a `modes()` convenience method.
- Candidate model name string changed from `"cmo_polynomial_channel"` to
  `"polynomial_surrogate_channel"` in reports and specs.

### Removed

- Top-level `__getattr__` deprecation shim: symbols outside Tier 1+2 must now be
  imported from their canonical sub-namespace (`stereocomplex.advanced`,
  `stereocomplex.physics`, `stereocomplex.synthetic`, `stereocomplex.rayfields`).
- All backward-compatibility aliases from v0.4 → v0.5 transitions.
- `test_public_api.py` now only checks Tier 1+2 symbols.

### Policy

- v0.x is explicitly unstable. Renames ship as real renames, no aliases.
  Stability commitments start at the 1.0 release.
- `filterwarnings = ["error::DeprecationWarning"]` is active in `pyproject.toml`
  (with numpy/scipy exceptions). Any new deprecation fails CI.

## [0.4.0] - 2026-05-06

### Added

- **`CMOPhysicalStereoModel`**: geometrically constrained CMO microscope model
  (17 shared parameters in default mode, 19 in aligned-sensor mode). Implements
  `PhysicalRayFieldModel` with `is_stereo_shared = True`.
- **`fit_cmo_physical_stereo_model_to_rayfields`**: joint stereo fit dispatching
  to a shared-rig optimization.
- **`is_stereo_shared`** attribute on `PhysicalRayFieldModel`; `select_physical_model_from_rayfield`
  dispatches on this flag and accepts `target_right` for stereo-shared candidates.
- **`share_principal_point`** mode on the physical CMO with transverse-gauge
  enforcement on principal-point offsets.
- **CMO rayfield bundle adjustment** (`fit_cmo_stereo_model_and_poses_from_zernike_rayfields`)
  jointly fitting per-channel polynomial models and board poses.
- **Greenough oracle classification test**: Brown-Conrady × 2 wins on independent channels.
- **Brown-Conrady oracle test**: parameter recovery and BIC selection on a dedicated Brown oracle.
- **Physical CMO vs polynomial surrogate comparison test** with BIC-based selection.

### Changed

- `pixel_pitch_mm` is a fixed kwarg on the physical CMO, excluded from the
  parameter vector. Only `f_tube_mm` is fitted; `p` and `f_tube` are not
  separately identifiable from ray geometry.
- `select_physical_model_from_rayfield` signature extended with `target_right=None`
  (per-channel mode preserved when None).
- `PhysicalRayFieldModel` protocol now includes `is_stereo_shared: bool`.

### Fixed

- `_aic_bic` documentation now states that `n` counts residual scalars
  (6 × n_pixels for two-plane residuals), not independent pixel observations.

## [0.3.0] - 2026-05-02

### Added

- **Optical model identification**: `select_physical_model_from_rayfield` fits
  and scores candidate physical models (central pinhole, Brown-Conrady, inclined
  parallel plate) against a measured Zernike rayfield using BIC/AIC.
- `fit_physical_model_to_rayfield`: lower-level single-model fitter with
  configurable support pixels, weights, and robust loss.
- `CentralBrownConradyModel`: five-parameter radial+tangential distortion model
  as a central physical candidate.
- `OpticalModelSelectionReport` / `PhysicalModelFitResult` dataclasses with
  `rows()` helper for pandas/JSON export.
- `default_physical_model_specs()`: standard three-candidate set.
- New tests: Brown distort/undistort roundtrip, Brown parameter recovery, and
  discrimination test proving model selection is not plate-biased.
- Tutorial `docs/IDENTIFY_MY_OPTICS.md` covering the identification workflow,
  report interpretation, custom candidates, and pitfalls.
- **API reorganization** (v0.3): `stereocomplex.__all__` slimmed to 24 Tier 1 +
  Tier 2 symbols. Sub-namespaces: `stereocomplex.advanced` (Tier 3 composition),
  `stereocomplex.synthetic`, `stereocomplex.physics`, `stereocomplex.rayfields`.

### Changed

- All 53 displaced symbols remain importable at the top level with a
  `DeprecationWarning` pointing to their new sub-namespace, until v1.0.
- `PhysicalModelFitResult` no longer has a `mdl_score` field (was always
  `None`); this is a breaking change only if code used `result.mdl_score`.
- `fit_physical_model_to_rayfield` gains an optional `name: str | None`
  parameter; when supplied, it overrides the model-derived name in the result.

### Fixed

- `select_physical_model_from_rayfield` no longer reconstructs a full
  `PhysicalModelFitResult` to patch the model name; `spec.name` is now
  threaded directly into `fit_physical_model_to_rayfield`.
- Redundant `rayfield_two_plane_residuals` recomputation after convergence in
  `fit_physical_model_to_rayfield` eliminated by slicing `combined`.

## [0.2.0] - 2026-04-30

### Added

- Non-central parallel-plate benchmark: full end-to-end pipeline from synthetic dataset
  generation to 3D reconstruction, with oracle, central, and Zernike origin-field models.
- `ZernikeRayField`: joint origin + direction Zernike model with transverse-gauge direction
  perturbation and full BA (`optimize_directions=True`).
- Theoretical writeup of Bundle Adjustment in `docs/`.
- `StereoZernikeOriginFieldFitResult` with health indicators: residual RMS, median, P95,
  and per-frame statistics surfaced from the BA solver.

### Changed

- BA closure (`fit_stereo_zernike_origin_field`) now pre-computes Zernike design matrices
  and pinhole directions once before `least_squares`; ~5× speedup on BA-dominated tests.
- `_triangulate_many` vectorised with numpy batch operations; eliminates Python loop over
  correspondences (~50-100× faster for large N).
- Transverse-gauge projection extracted into `_project_transverse` shared by model classes
  and the BA hot path, preventing silent divergence if the gauge definition changes.

### Fixed

- `_render_one_view` was already fully vectorised (meshgrid + single
  `parallel_plate_ray_from_pixel` call); no pixel-level loop existed.

## [0.1.0] - 2026-04-24

### Added

- Public calibration API for OpenCV stereo and central ray-field stereo workflows.
- Public ChArUco detection/refinement API with `CharucoBoardSpec`.
- Bring-your-own-data documentation for left/right stereo image folders.
- Guided notebooks and committed sample scenes for first-run examples.
- Ray-field virtual rectification demo for classic dense stereo matchers.

### Changed

- `opencv-contrib-python-headless` is installed by default for ArUco/ChArUco support.
- Jupyter notebook dependencies are available through the optional `notebooks` extra.
