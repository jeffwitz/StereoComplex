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
from stereocomplex.api import StereoCentralRayFieldModel, load_stereo_central_rayfield, save_stereo_central_rayfield
```

## Corner refinement API

For ChArUco refinement, the stable entry point is:

```python
from stereocomplex.api import refine_charuco_corners
```

This refines ChArUco corners using only geometric priors on the board plane (`rayfield_tps_robust`).

Note: `opencv-contrib-python` is required for `cv2.aruco` and homography estimation.

