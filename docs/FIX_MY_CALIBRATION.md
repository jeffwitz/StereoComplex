# Fix my calibration (OpenCV + ChArUco)

This is the **user-facing** entry point: you have ChArUco images, OpenCV calibration “plateaus” (blur, distortion, compression), and you want a practical improvement.

StereoComplex provides a simple idea:

1. detect ArUco/ChArUco as usual,
2. replace raw ChArUco corners by **refined corners** (`rayfield_tps_robust`) using a planar geometric prior,
3. feed these refined points to OpenCV (`calibrateCamera`, `stereoCalibrate`) or your own pipeline.

## Fast path on your own images

If you do **not** use the synthetic dataset layout, you can still use the public API
directly on your own folders:

```python
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

detections = sc.detect_charuco_corners(image="left/000000.png", board=board)
refined_xy = sc.refine_charuco_corners(
    method="rayfield_tps_robust",
    board=board,
    marker_ids=detections.marker_ids,
    marker_corners=detections.marker_corners,
    charuco_ids=detections.charuco_ids,
    charuco_xy=detections.charuco_xy,
)
```

This is the smallest public API example when you only want the refined 2D corners.
For the full stereo path from two folders, see :doc:`BRING_YOUR_OWN_DATA`.

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

Use the public Python path described in :doc:`BRING_YOUR_OWN_DATA`.

The dataset v0 CLI remains useful for:

- reproducible synthetic benchmarks,
- paper figures,
- regression tests.

But it is no longer the only way to try StereoComplex on real stereo folders.
