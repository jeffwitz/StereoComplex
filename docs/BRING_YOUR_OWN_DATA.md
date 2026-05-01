# Bring your own data

This page is the shortest path from:

- `left/*.png`
- `right/*.png`
- a known ChArUco board

to:

- a baseline **OpenCV raw vs Ray2D + OpenCV** comparison,
- a calibrated StereoComplex model,
- optionally an experimental non-central Zernike origin-field model,
- an exported `models/<name>/model.json + weights.npz`,
- and then `model.triangulate(...)` in Python.

## What you need

StereoComplex does **not** require the full dataset v0 format for user data anymore.

For a first trial you only need:

1. one folder of left images,
2. one folder of right images,
3. the ChArUco board definition:
   - `squares_x`, `squares_y`
   - `square_size_mm`, `marker_size_mm`
   - `aruco_dictionary`

The current public API assumes:

- one left image matches one right image,
- images are paired by sorted filename order,
- all images have the same resolution,
- the board is planar and visible in several poses.

## Minimal calibration from two folders

### Option A: stay in OpenCV and compare raw vs Ray2D

```python
from pathlib import Path

import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

raw = sc.fit_opencv_stereo_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    method2d="raw",
)
refined = sc.fit_opencv_stereo_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    method2d="rayfield_tps_robust",
)

print(raw.report)
print(refined.report)
```

Use this when your goal is:

- standard OpenCV pinhole calibration,
- a quick before/after comparison,
- and no 3D ray-field model yet.

### Option B: fit the StereoComplex 3D backend

```python
from pathlib import Path

import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

result = sc.fit_stereo_central_rayfield_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    method2d="rayfield_tps_robust",
    nmax=10,
    export_model_dir=Path("models/my_first_calibration"),
)

print(result.report)
model = result.model
```

What this does:

1. detect ArUco / ChArUco in each stereo pair,
2. optionally refine ChArUco corners with the planar second pass,
3. initialize poses from homographies,
4. run the central stereo ray-field bundle adjustment,
5. optionally export a reusable model directory.

### Option C: fit an experimental non-central origin field

Use this when a central/pinhole model leaves systematic ray gaps or
reconstruction bias, for example with protective glass, an inclined window, a
thick optical stack, or a diopter-like setup.

```python
from pathlib import Path

import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=9,
    squares_y=6,
    square_size_mm=20.0,
    marker_size_mm=15.0,
    aruco_dictionary="DICT_4X4_50",
)

fit = sc.fit_stereo_zernike_origin_field_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    max_order=4,
    method2d="rayfield_tps_robust",
)

print(fit.residual_rms)
left_field = fit.left_field
right_field = fit.right_field
```

This path fits a Zernike origin field `O(u,v)` initialized from an OpenCV stereo
calibration. It is experimental and should be checked with diverse board poses,
train/test pose splits, and support-aware rayfield metrics before deployment.

For the practical walkthrough, see :doc:`NONCENTRAL_FROM_IMAGES`. For the
controlled physical oracle and complete BA discussion, see
:doc:`PARALLEL_PLATE_ORIGIN_FIELD`.

## Later: load and triangulate

```python
import numpy as np
import stereocomplex as sc

model = sc.load_stereo_central_rayfield("models/my_first_calibration")

uv_left_px = np.array([[320.0, 240.0]], dtype=float)
uv_right_px = np.array([[318.5, 240.0]], dtype=float)
XYZ_mm, skew_mm = model.triangulate(uv_left_px, uv_right_px)
```

`XYZ_mm` is expressed in the **left camera frame**.

## If you already have a dataset v0 scene

Use the dataset-specific public wrapper:

```python
import stereocomplex as sc

result = sc.fit_stereo_central_rayfield_from_dataset(
    dataset_root="dataset/v0_png",
    split="train",
    scene="scene_0000",
    max_frames=5,
    method2d="rayfield_tps_robust",
    export_model_dir="models/scene0000_rayfield3d",
)
```

This is the stable API equivalent of the older internal calibration scripts.

## If you only want to improve OpenCV corners

You do **not** need the 3D ray-field backend for that.

The low-level public workflow is:

```python
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

detections = sc.detect_charuco_corners(image="my_data/left/000000.png", board=board)
refined_xy = sc.refine_charuco_corners(
    method="rayfield_tps_robust",
    board=board,
    detections=detections,
)
```

You can then feed the refined points into your usual OpenCV calibration code.

## Practical notes

- `method2d="raw"` keeps OpenCV ChArUco corners as-is.
- `method2d="rayfield_tps_robust"` applies the planar second pass before 3D calibration.
- `min_common_corners` controls how many common left/right ChArUco corners are required per stereo pair.
- `max_nfev` is the main knob for a quicker smoke test versus a more converged optimization.

## Recommended first experiment

Do this first:

1. run with `max_pairs=3` or `max_frames=3`,
2. keep `nmax=4` or `nmax=6`,
3. confirm that `result.report.n_initialized_frames >= 2`,
4. inspect `result.report.train_skew_p95_mm` and `result.report.train_point_to_ray_p95_mm`,
5. save the model,
6. only then increase the number of poses and the basis size.

`n_initialized_frames >= 2` only means that the optimizer could start. It is not a
quality criterion. Large training skew or point-to-ray residuals mean the exported
model may be reloadable but geometrically unusable.
