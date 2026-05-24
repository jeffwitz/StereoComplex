# From OpenCV to StereoComplex

You already know `cv2.calibrateCamera` and `cv2.stereoCalibrate`?
Here is the equivalent path in StereoComplex, plus what you gain.

## 3-minute quickstart

```python
from pathlib import Path
import stereocomplex as sc

# 1. Define your ChArUco board (same as OpenCV)
board = sc.CharucoBoardSpec(
    squares_x=11, squares_y=7,
    square_size_mm=39.07, marker_size_mm=27.35,
    aruco_dictionary="DICT_4X4_1000",
)

# 2. Compare OpenCV raw vs Ray2D-refined in one call
report = sc.compare_opencv_stereo_calibration(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
)
print(f"Raw stereo RMS:    {report['raw']['stereo_rms_px']:.3f} px")
print(f"Refined stereo RMS: {report['refined']['stereo_rms_px']:.3f} px")
print(f"Improvement:        {report['improvement_px']:.3f} px")

# 3. Export to OpenCV format
K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()
```

## Translation table

| OpenCV | StereoComplex |
|---|---|
| `cv2.aruco.CharucoBoard` | `sc.CharucoBoardSpec(...)` |
| `cv2.aruco.detectMarkers` | `sc.detect_charuco_corners(image, board)` |
| `cv2.calibrateCamera` | `sc.calibrate_opencv(left_dir, right_dir, board)` |
| Raw corners | `method2d="raw"` |
| Refined corners (Ray2D) | `method2d="rayfield_tps_robust"` |
| `K1, d1, K2, d2, R, T` | `result.to_opencv()` |
| Compare raw vs refined | `sc.compare_opencv_stereo_calibration(...)` |

## What you gain

| Feature | OpenCV alone | With StereoComplex |
|---|---|---|
| Corner accuracy | Raw ChArUco | Ray2D-refined (~2× better) |
| Non-central optics | Not supported | Full Zernike rayfield pipeline |
| Model identification | Manual | Automatic BIC-based selection |
| Optics catalogue | None | 6 families (pinhole→CMO→Greenough→exotic) |

## Next steps

- **I want better corners before OpenCV** → `sc.compare_opencv_stereo_calibration()`
- **I want a central ray-based model** → `sc.calibrate_central(left_dir, right_dir, board)`
- **I suspect non-central optics** → `sc.calibrate_noncentral(left_dir, right_dir, board)`
- **I want to identify my microscope optics** → `sc.identify_optics(target_field, candidate_specs, K, image_size)`

## See also

- [Bring Your Own Data](BRING_YOUR_OWN_DATA.md) — detailed walkthrough
- [Identify My Optics](IDENTIFY_MY_OPTICS.md) — model catalogue and interpretation
- [START HERE](START_HERE.md) — three workflows overview
