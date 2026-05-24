# Non-central stereo calibration from image folders

This page is the practical entry point for the non-central
calibration path. Use it when you already have two folders of calibration images
and a known ChArUco board.

For the scientific validation of the model, including the inclined
parallel-plate oracle and complete `O(u,v), d(u,v), poses, rig` BA benchmark,
see :doc:`PARALLEL_PLATE_ORIGIN_FIELD`. The present page intentionally hides
that oracle machinery and focuses on the user-facing API.

## What this fits

The practical non-central API fits a Zernike origin field:

```{math}
(u,v) \longmapsto (O(u,v), d_\mathrm{pinhole}(u,v)).
```

This means each pixel defines a 3D line rather than a ray emitted from one fixed
camera center. The direction is initialized from the OpenCV pinhole calibration;
the fitted `O(u,v)` field captures systematic non-centrality.

The API starts from a standard image workflow:

1. detect ChArUco corners in left/right image folders;
2. optionally apply the Ray2D planar refinement `rayfield_tps_robust`;
3. run OpenCV mono and stereo calibration to initialize `K`, poses and rig;
4. fit the Zernike origin field by point-to-ray bundle adjustment.

## Minimal example

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

print(f"success      : {fit.success}")
print(f"residual RMS : {fit.residual_rms:.3f} mm")
print(f"observations : {fit.n_observations}")

left_field = fit.left_field
right_field = fit.right_field
T_right_left = fit.stereo_transform
```

For any pixel, the fitted fields return the line representation:

```python
O, d = left_field.ray(u, v)
```

The companion notebook is:

```text
examples/notebooks/05_noncentral_calibration_from_images.ipynb
```

It renders a small synthetic non-central image set so the notebook runs from a
fresh clone, then shows where to replace `left_dir`, `right_dir` and `board` for
your own data.

## When to use this path

Use this path when a pinhole stereo model leaves systematic geometric residuals,
for example with:

- protective glass or an inclined window;
- a thick optical stack;
- a diopter or refractive plate;
- microscope-like or CMO-like geometry;
- a stereo rig where ray gaps remain structured after OpenCV calibration.

If you only want a better OpenCV calibration, start with:

```python
sc.fit_opencv_stereo_from_image_dirs(
    left_dir=Path("left"),
    right_dir=Path("right"),
    board=board,
    method2d="rayfield_tps_robust",
)
```

If you want a central ray-based model, use
`fit_stereo_central_rayfield_from_image_dirs(...)` instead.

## Reading the result

The returned object is a `StereoZernikeOriginFieldFitResult`:

- `left_field`, `right_field`: fitted `ZernikeOriginField` objects;
- `stereo_transform`: transform from the left camera frame to the right camera
  frame;
- `board_poses`: fitted board poses;
- `residual_rms`, `residual_median`, `residual_p95`: point-to-ray residuals in
  millimetres;
- `n_observations`: number of common ChArUco correspondences used by the BA.

Useful first checks:

| Metric | Read as | Warning sign |
| --- | --- | --- |
| `residual_rms` | point-to-ray fit error | large values: detection failure, poor initialization, too few poses |
| `n_observations` | amount of geometric support | too small for chosen `max_order` |
| mean `|O(u,v)|` | strength of non-centrality | near zero means pinhole may already be enough; huge values suggest instability |

## Status and limitations

The non-central Zernike rayfield backend is **validated on real CMO microscope
data** (Pycaso, see Notebook 09: 0.47 px RMS, CMO-consistent descriptors
recovered).  It is useful for controlled benchmarks, for testing systems where
the central-camera assumption is visibly biased, and for identifying physical
optical architectures from measured rayfields.

Current limitations:

- requires several diverse board poses;
- is sensitive to 2D detection quality;
- higher Zernike orders require more observations and better field coverage;
- the practical API exposes the origin-field model first;
- the complete `O(u,v), d(u,v), poses, rig` BA is documented as an advanced
  benchmark path in :doc:`PARALLEL_PLATE_ORIGIN_FIELD`;
- train/test pose splits and support-aware rayfield metrics should be checked
  before claiming deployment-grade calibration.

## Relation to Ray2D

Ray2D / `rayfield_tps_robust` is a 2D planar refinement of observed ChArUco
corners. It is not itself a 3D non-central camera model. In this workflow it
improves the image observations before the non-central 3D line model is fitted.
