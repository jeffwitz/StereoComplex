# Changelog

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
