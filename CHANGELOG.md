# Changelog

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
