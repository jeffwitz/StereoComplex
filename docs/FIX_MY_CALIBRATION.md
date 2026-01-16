# Fix my calibration (OpenCV + ChArUco)

This is the **user-facing** entry point: you have ChArUco images, OpenCV calibration “plateaus” (blur, distortion, compression), and you want a practical improvement.

StereoComplex provides a simple idea:

1. detect ArUco/ChArUco as usual,
2. replace raw ChArUco corners by **refined corners** (`rayfield_tps_robust`) using a planar geometric prior,
3. feed these refined points to OpenCV (`calibrateCamera`, `stereoCalibrate`) or your own pipeline.

## Step 1 — Export refined 2D corners

On a dataset v0 scene:

```bash
.venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png \
  --split train --scene scene_0000 \
  --max-frames 16 \
  --method rayfield_tps_robust \
  --out-json paper/tables/refined_corners_scene0000.json \
  --out-npz paper/tables/refined_corners_scene0000_opencv.npz
```

Outputs:

- `refined_corners_scene0000.json`: per-frame raw + refined corners (for inspection/debug),
- `refined_corners_scene0000_opencv.npz`: per-frame object points + image points (ready for OpenCV calibration scripts).

## Step 2 — Calibrate with OpenCV (example)

StereoComplex includes a reproducible comparison script (raw vs ray-field):

```bash
.venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/v0_png \
  --split train --scene scene_0000 \
  --out paper/tables/opencv_calibration_rayfield.json
```

This script shows the expected impact on:

- mono RMS (px),
- stereo RMS (px),
- baseline error in “pixel-equivalent disparity” (px),
- triangulation error vs GT (mm) on synthetic datasets.

## What to do on real data?

The current `refine-corners` command expects a dataset v0 scene (it uses `meta.json` to build the board definition).

For real data, the next step is to add a folder-based mode:

- provide your ChArUco board parameters (`squares_x/y`, `square_size_mm`, `marker_size_mm`, dictionary),
- run corner refinement on all images,
- export the resulting 2D points for OpenCV calibration.

This is planned as a follow-up once we lock the output format and validation workflow.

