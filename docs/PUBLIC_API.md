# Public API contract

StereoComplex is a research prototype, but it exposes a small **public API** meant to be usable in downstream code.

## Stability promise

- Everything under `stereocomplex.api` is considered **public** and should remain backward compatible within the `0.x` series as much as possible.
- The inclined-plate / `O(u,v)` origin-field functions are **public experimental**:
  they are exposed for reproducible research examples, but may still evolve
  before a 1.0 API.
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

Experimental non-central origin-field imports:

```python
from stereocomplex.api import (
    ParallelPlateSyntheticParams,
    PinholeParallelPlateFitParams,
    ZernikeOriginFieldConfig,
    fit_parallel_plate_to_zernike_rayfield,
    fit_stereo_zernike_origin_field_from_image_dirs,
    fit_stereo_zernike_origin_field,
    compare_3d_reconstruction_with_without_origin_field,
    oracle_reconstruction_floor_report,
    compare_rayfields_on_planes,
    run_parallel_plate_origin_field_benchmark,
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

### Experimental non-central calibration from image directories

If you suspect the rig is not well represented by a central/pinhole model, the
public experimental entry point is:

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
    left_dir=Path("left"),
    right_dir=Path("right"),
    board=board,
    max_order=4,
    method2d="rayfield_tps_robust",
)

print(fit.residual_rms)
left_field = fit.left_field
right_field = fit.right_field
```

This API currently fits a Zernike origin field `O(u,v)` initialized from an
OpenCV stereo calibration. It is intended for experimental non-central stereo
systems where a pinhole model leaves systematic ray gaps or reconstruction bias.

The complete `O(u,v), d(u,v), poses, rig` BA remains an advanced benchmark path
and is documented separately from this practical origin-field API in
:doc:`PARALLEL_PLATE_ORIGIN_FIELD`. For the guided practical workflow, use
`examples/notebooks/05_noncentral_calibration_from_images.ipynb`.

### Experimental physical plate fit from a measured rayfield

Notebook 04 also exposes a second-stage interpretation tool:

```python
plate_fit = sc.fit_parallel_plate_to_zernike_rayfield(
    zernike_field=fit.left_field,
    K=fit.left_field.K,
    image_size=fit.left_field.config.image_size,
    eta=1.5,
    support_pixels=observed_pixels,
)

print(plate_fit.params)
print(plate_fit.rayfield_rms_support_mm)
```

This does **not** replace the generic Zernike fit. It treats the fitted rayfield
as a measured geometric object, then asks whether a compact pinhole + inclined
parallel-plate model can explain it in ray space. The residual is computed from
intersections with two z-planes, not by comparing raw ray origins.

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

## Experimental non-central rayfield API

The inclined parallel-plate benchmark is exposed as a public experimental API.
It is meant for synthetic non-central ray-field experiments, not for calibrating
the physical parameters of a glass plate.

The shortest complete benchmark is:

```python
import stereocomplex as sc

report = sc.run_parallel_plate_origin_field_benchmark(
    max_order=4,
    noise_std_px=0.05,
)

print(report.reconstruction_comparison.central.rms_3d)
print(report.oracle_floor.oracle_observed_pixels.rms_3d)
print(report.reconstruction_comparison.with_origin_field.rms_3d)
print(report.reconstruction_comparison.improvement_rms_factor)
```

The rendered-image front-end check is also exposed:

```python
from pathlib import Path

import stereocomplex as sc

image_report = sc.run_parallel_plate_rendered_image_benchmark(
    out_dir=Path("outputs/parallel_plate_rendered"),
    max_order=3,
    method2d="raw",
)
refined_image_report = sc.run_parallel_plate_rendered_image_benchmark(
    out_dir=Path("outputs/parallel_plate_rendered_ray2d"),
    max_order=3,
    method2d="rayfield_tps_robust",
)

print(image_report.n_frames)
print(image_report.n_common_corners)
print(image_report.n_points_total)
print(image_report.reconstruction_comparison.central.rms_3d)
print(image_report.oracle_detected.rms_3d)
print(image_report.reconstruction_comparison.with_origin_field.rms_3d)
print(refined_image_report.reconstruction_comparison.with_origin_field.rms_3d)
```

This renders non-central ChArUco images with vignetting, edge blur and noise,
detects the corners with OpenCV, optionally applies the public
`rayfield_tps_robust` planar refinement, then runs the complete geometric BA
over `O(u,v)`, `d(u,v)`, board poses and stereo rig. The `oracle_detected`
report is an evaluation-only floor: it triangulates the front-end pixel
coordinates with the exact synthetic rayfield, so it separates image/detector
error from BA error.

The lower-level path is:

```python
import stereocomplex as sc

dataset = sc.make_default_parallel_plate_dataset(noise_std_px=0.05)
dataset_clean = sc.make_default_parallel_plate_dataset(noise_std_px=0.0)
config = sc.ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=4)

fit = sc.fit_stereo_zernike_origin_field(
    observations=dataset,
    K_left=dataset.K_left,
    K_right=dataset.K_right,
    T_right_left_initial=dataset.T_right_left,
    board_poses_initial=dataset.board_poses,
    config_left=config,
    config_right=config,
    regularization=1e-3,
)

comparison = sc.compare_3d_reconstruction_with_without_origin_field(
    dataset=dataset,
    central_model_result=None,
    origin_field_result=fit,
)

oracle_floor = sc.oracle_reconstruction_floor_report(
    dataset_observed=dataset,
    dataset_clean=dataset_clean,
)

left_rayfield = sc.compare_rayfields_on_planes(
    fitted_field=fit.left_field,
    oracle_ray_function=dataset.oracle_left_ray_function,
    image_size=dataset.image_size,
    z_planes=(100.0, 1000.0),
)

fit_ba = sc.fit_stereo_zernike_origin_field(
    observations=dataset,
    K_left=dataset.K_left,
    K_right=dataset.K_right,
    T_right_left_initial=dataset.T_right_left,
    board_poses_initial=dataset.board_poses,
    config_left=sc.ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=3),
    config_right=sc.ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=3),
    optimize_directions=True,
    optimize_board_poses=True,
    optimize_stereo_extrinsics=True,
    regularization=1e-5,
    direction_regularization=1e-2,
    pose_regularization=10.0,
    rig_regularization=100.0,
)
```

This API deliberately separates:

- oracle generation: `ParallelPlateSyntheticParams`,
  `generate_parallel_plate_stereo_dataset`;
- rendered image generation: `render_parallel_plate_charuco_images`,
  `detected_observations_from_rendered_parallel_plate`,
  `run_parallel_plate_rendered_image_benchmark`;
- generic identified models: `ZernikeOriginField` for O-only fits and
  `ZernikeRayField` for experimental `O(u,v)+d(u,v)` fits;
- fitting: `fit_stereo_zernike_origin_field`, with optional
  `optimize_directions=True`, `optimize_board_poses=True`, and
  `optimize_stereo_extrinsics=True` for the complete geometric BA mode over
  `O(u,v)`, `d(u,v)`, poses, and rig;
- evaluation: `compare_3d_reconstruction_with_without_origin_field`,
  `oracle_reconstruction_floor_report`, `compare_rayfields_on_planes`.

See the theory and figures in :doc:`PARALLEL_PLATE_ORIGIN_FIELD`, and the
guided notebook `examples/notebooks/04_parallel_plate_origin_field.ipynb`.

## Public `method2d` values

| Method | Status | Description |
| --- | --- | --- |
| `raw` | public stable | OpenCV ChArUco corners are used unchanged. |
| `rayfield_tps_robust` | public stable | Robust planar TPS/ray-field 2D correction before calibration. |
| other methods | internal benchmark | Experimental or historical methods are not guaranteed by the public API. |
