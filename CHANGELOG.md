# Changelog

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
