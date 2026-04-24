# Public API contract

StereoComplex is a research prototype, but it exposes a small **public API** meant to be usable in downstream code.

## Stability promise

- Everything under `stereocomplex.api` is considered **public** and should remain backward compatible within the `0.x` series as much as possible.
- Everything else (`stereocomplex.core`, `stereocomplex.eval`, `paper/`, `docs/examples/`) is **internal** and may change without notice.

## Recommended imports

Top-level re-exports (stable):

```python
import stereocomplex as sc

model = sc.load_stereo_central_rayfield("models/my_model")
XYZ_mm, skew_mm = model.triangulate(uv_left_px, uv_right_px)
```

Direct API imports (stable):

```python
from stereocomplex.api import (
    CharucoBoardSpec,
    StereoOpenCVCalibrationResult,
    StereoCentralRayFieldModel,
    build_charuco_board,
    detect_charuco_corners,
    fit_opencv_stereo_from_image_dirs,
    fit_stereo_central_rayfield_from_dataset,
    fit_stereo_central_rayfield_from_image_dirs,
    load_stereo_central_rayfield,
    refine_charuco_corners,
    save_stereo_central_rayfield,
)
```

## Calibrate from your own images

### Standard OpenCV stereo, raw vs Ray2D

If you want to stay in a classic OpenCV pinhole workflow and only compare the
2D preprocessing stage, the stable public entry point is:

```python
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

raw = sc.fit_opencv_stereo_from_image_dirs(
    left_dir="my_data/left",
    right_dir="my_data/right",
    board=board,
    method2d="raw",
)
refined = sc.fit_opencv_stereo_from_image_dirs(
    left_dir="my_data/left",
    right_dir="my_data/right",
    board=board,
    method2d="rayfield_tps_robust",
)

print(raw.report)
print(refined.report)
```

This is the exact path used at the beginning of notebook `01_ray2d_vs_opencv`.

### StereoComplex 3D ray-field calibration

The main high-level calibration entry points are:

```python
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.0713,
    marker_size_mm=27.3499,
    aruco_dictionary="DICT_4X4_1000",
)

result = sc.fit_stereo_central_rayfield_from_image_dirs(
    left_dir="my_data/left",
    right_dir="my_data/right",
    board=board,
    method2d="rayfield_tps_robust",
    export_model_dir="models/my_calibration",
)

model = result.model
print(result.report)
```

If you already use a versioned dataset v0 scene, the stable dataset wrapper is:

```python
result = sc.fit_stereo_central_rayfield_from_dataset(
    dataset_root="dataset/v0_png",
    split="train",
    scene="scene_0000",
    max_frames=5,
    method2d="rayfield_tps_robust",
    export_model_dir="models/scene0000_rayfield3d",
)
```

## Corner refinement API

For ChArUco refinement, the stable entry point is:

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
    detections=detections,
)
```

This keeps the workflow inside the public API:

1. detect ArUco / ChArUco,
2. refine corners with the planar geometric prior,
3. send the refined points to OpenCV or to the public StereoComplex calibration wrappers.

Note: the default package install already includes OpenCV ArUco support through
`opencv-contrib-python-headless`.

## Public `method2d` values

| Method | Status | Description |
| --- | --- | --- |
| `raw` | public stable | OpenCV ChArUco corners are used unchanged. |
| `rayfield_tps_robust` | public stable | Robust planar TPS/ray-field 2D correction before calibration. |
| other methods | internal benchmark | Experimental or historical methods are not guaranteed by the public API. |
