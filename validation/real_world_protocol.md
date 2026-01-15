# Real-world protocol (MVP)

Goal: start sim→real validation as early as possible with a small set, without waiting for OptiX/ML.

## Acquisition

- 10–30 stereo pairs of a target (ChArUco or textured), with diverse poses (tilt + translation + distance).
- Keep acquisition metadata: `pitch_um`, W/H, binning, crop/ROI, resize, bit depth, gamma if known.

## Metrics to compute (initially)

- Reprojection error (px) on detectable points (even if the ML calibration is not ready).
- Stability vs blur/noise (at least 2 levels of focus/ISO/exposure).
- Out-of-domain detection: a minimal “quality” score (to be defined later).
