# Reference: result objects

Every public result dataclass returned by a calibration or metric function.
All fields are listed with type, unit, shape, and physical meaning.

## `StereoOpenCVCalibrationResult`

Returned by `fit_opencv_stereo_*`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `K_left` | `(3,3) ndarray` | px, px | Left camera matrix |
| `K_right` | `(3,3) ndarray` | px, px | Right camera matrix |
| `d_left` | `(5,) ndarray` | — | Left distortion coeffs (k1,k2,p1,p2,k3) |
| `d_right` | `(5,) ndarray` | — | Right distortion coeffs |
| `R` | `(3,3) ndarray` | — | Rotation from left to right camera |
| `T` | `(3,) ndarray` | mm | Translation from left to right |
| `rms_px` | `float` | px | Mean reprojection error |
| `image_size` | `(int,int)` | px | Sensor width, height |
| `n_frames` | `int` | — | Number of frames used |

## `StereoCentralRayFieldFitResult`

Returned by `fit_stereo_central_rayfield_*`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `model` | `StereoCentralRayFieldModel` | — | Fitted central rayfield |
| `rvecs` | `dict[int, (3,) ndarray]` | rad | Per‑frame rotation vectors |
| `tvecs` | `dict[int, (3,) ndarray]` | mm | Per‑frame translation vectors |
| `n_input_pairs` | `int` | — | Number of stereo pairs loaded |
| `n_detected_pairs` | `int` | — | Pairs with enough common corners |
| `n_observation_frames` | `int` | — | Frames with observations |
| `n_initialized_frames` | `int` | — | Frames with initial poses |
| `method2d` | `str` | — | 2‑D refinement method used |

## `ReconstructionResult`

Returned by `reconstruct_points_central_stereo` and
`reconstruct_points_with_origin_fields`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `points_3d_mm` | `(N,3) ndarray` | mm | Triangulated 3‑D points |
| `ray_gap_mm` | `(N,) ndarray` | mm | Minimum ray distance per point |
| `mask_valid` | `(N,) ndarray` | — | True where triangulation succeeded |

## `CMOPhysicalStereoFitResult`

Returned by `fit_cmo_physical_stereo_model_to_rayfields` and its variants.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `x` | `(P,) ndarray` | — | Optimal parameter vector |
| `model` | `CMOPhysicalStereoModel` | — | Fitted CMO rig |
| `message` | `str` | — | Optimiser exit message |
| `success` | `bool` | — | Whether the optimiser converged |
| `rms_mm` | `float` | mm | RMS ray‑gap after fitting |
| `nfev` | `int` | — | Number of function evaluations |
| `n_parameters` | `int` | — | Number of free parameters (19 or 21) |

## `PhysicalModelFitResult`

Returned by `fit_physical_model_to_rayfield`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `x` | `(P,) ndarray` | — | Optimal parameters |
| `model` | `physical model instance` | — | Fitted instance |
| `rms_mm` | `float` | mm | RMS ray‑space residual |
| `success` | `bool` | — | Convergence flag |

## `ReconstructionComparisonReport`

Returned by `compare_3d_reconstruction_with_without_origin_field`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `central` | `ReconstructionReport` | — | Central‑model error statistics |
| `noncentral` | `ReconstructionReport` | — | Non‑central error statistics |
