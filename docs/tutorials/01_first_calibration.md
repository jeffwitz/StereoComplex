# Tutorial 1 — Your first calibration (OpenCV, 10 minutes)

This is the simplest entry point into StereoComplex.  You will calibrate a
stereo camera pair from ChArUco images and get a numerical quality score.
No 3-D rayfields, no optical models, no paper references.  If you already
use OpenCV, this is a drop‑in improvement.

**Time to complete:** < 10 minutes on the bundled dataset.

## What you need

- Python 3.10+ with `pip install stereocomplex[full]` (or `pip install -e .`
  from a local clone).
- The bundled dataset `dataset/v0_png/train/scene_0000/` (shipped in the repo).

## Step 1 — Import and define your board

```python
from stereocomplex import CharucoBoardSpec

board = CharucoBoardSpec(
    squares_x=5, squares_y=7,
    square_size_mm=20.0,
    marker_size_mm=15.0,
)
```

## Step 2 — Calibrate with the OpenCV stereo path

```python
from stereocomplex.api import fit_opencv_stereo_from_dataset

result = fit_opencv_stereo_from_dataset(
    dataset_root="dataset/v0_png",
    split="train",
    scene="scene_0000",
    board=board,
    max_frames=20,
    method2d="raw",     # raw = standard OpenCV corner detection
)
```

This returns a `StereoOpenCVCalibrationResult` with per‑camera intrinsics,
distortion coefficients, and the stereo rig transform.

## Step 3 — Check the quality

```python
from stereocomplex.api import assess_calibration

report = assess_calibration(result)
print(report)          # ok / warning / failed + per‑channel RMS
```

## Step 4 — Export back to OpenCV

```python
K1, d1, K2, d2, R, T = result.to_opencv()
# → you're back in classical OpenCV land with a better-calibrated result
```

## What you just did

| Step | What happened |
|---|---|
| 1 | Defined a ChArUco board geometry (same as you would in OpenCV) |
| 2 | Stereo‑calibrated from pre‑captured images with standard OpenCV corners |
| 3 | Assessed quality (RMS reprojection, skew, point‑to‑ray distance) |
| 4 | Exported back to native OpenCV format |

## Where to go next

- [Tutorial 2 — Improve with Ray2D](02_opencv_plus_ray2d) — refine the same calibration with sub‑pixel corrections.
- [How‑to: Bring your own data](../how_to/bring_your_own_data) — calibrate from your own image folders.
- [How‑to: Fix my calibration](../how_to/fix_my_calibration) — diagnose what's wrong with an existing result.
