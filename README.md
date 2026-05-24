# StereoComplex

Lightweight Python toolkit for robust stereo calibration and ray-based 3D reconstruction.

StereoComplex is designed for users who already know OpenCV calibration but need more diagnostic power when ChArUco localization, blur, compression, optical distortion, microscope optics, protective glass, or non-central effects make a standard pinhole workflow plateau.

## Start here

The recommended first entry point is:

```text
examples/notebooks/00_getting_started.ipynb
examples/notebooks/00_getting_started.py
```

It gives the shortest OpenCV-to-StereoComplex path:

1. define a ChArUco board;
2. compare raw OpenCV calibration against Ray2D-refined calibration;
3. inspect calibration quality with `assess_calibration`;
4. export the result back to OpenCV format with `result.to_opencv()`.

Companion guide:

```text
docs/FROM_OPENCV_TO_STEREOCOMPLEX.md
```

Full notebook map:

```text
docs/NOTEBOOKS.md
```

## What StereoComplex does

StereoComplex has three distinct layers. Keeping them separate is important.

| Layer | Meaning | Current status |
|---|---|---|
| **Ray2D / planar ray-field** | Homography plus smooth residual field on the calibration board plane | stable practical preprocessing |
| **Central 3D ray-field** | Pixel → 3D direction with a shared camera center | research prototype |
| **Non-central 3D rayfield** | Pixel → 3D line `(O(u,v), d(u,v))` | research-grade, validated on synthetic oracles and one real CMO case study |

Ray2D is **not** a non-central 3D camera model. It is a robust 2D observation refinement stage that improves the image points fed to calibration.

## Main user paths

| I want to… | Start with |
|---|---|
| Learn the basic workflow | `examples/notebooks/00_getting_started.ipynb` |
| Improve ChArUco corners before OpenCV | `sc.refine_charuco_corners(...)` |
| Compare OpenCV raw vs Ray2D-refined | `sc.compare_opencv_stereo_calibration(...)` |
| Calibrate like OpenCV with refined corners | `sc.calibrate_opencv(...)` |
| Switch to a central ray-based model | `sc.calibrate_central(...)` |
| Test non-central optics | `sc.calibrate_noncentral(...)` |
| Identify CMO / plate / Brown / other optical families | `sc.identify_optics(...)` |
| Check whether a calibration is usable | `sc.assess_calibration(result)` |
| Export to OpenCV | `result.to_opencv()` |

## Why use it?

StereoComplex targets two practical problems:

1. **The camera model is not always the first limitation.** In many stereo systems, calibration is limited by 2D localization quality: blur, compression, noise, low contrast, defocus, or ChArUco detection failures.
2. **Some optics are not well described by a central pinhole model.** Microscopes, inclined plates, thick optical stacks, protective windows, and CMO systems can leave structured ray residuals that a central model cannot explain.

The package therefore combines OpenCV-compatible workflows with ray-based diagnostics.

## Key results

### 1. OpenCV-compatible Ray2D refinement

On the bundled synthetic ChArUco examples, Ray2D refinement improves corner localization and can substantially reduce OpenCV mono/stereo reprojection residuals. See:

```text
docs/RAYFIELD_WORKED_EXAMPLE.md
docs/FIX_MY_CALIBRATION.md
examples/notebooks/01_ray2d_vs_opencv.ipynb
```

### 2. Non-central rendered-image benchmark

On the inclined-plate benchmark, raw OpenCV ChArUco detections impose a high reconstruction floor. With Ray2D-refined observations, the same non-central bundle adjustment reaches sub-millimetric reconstruction accuracy:

| Front-end | Central RMS | Oracle-detected RMS | Non-central BA RMS |
|---|---:|---:|---:|
| OpenCV raw | ~4.21 mm | ~3.44 mm | ~3.36 mm |
| Ray2D refined | ~2.50 mm | ~0.76 mm | ~0.66 mm |

Interpretation: the non-central model works when the 2D observations are good enough; front-end quality is the limiting factor on rendered or real images.

### 3. Optical model identification

A measured Zernike rayfield can be used as a geometric diagnostic instrument. StereoComplex can compare compact physical hypotheses such as central pinhole, Brown-Conrady, inclined plate, CMO-like models, and generic fallback rayfields.

On the inclined-plate oracle, the physical plate model is selected unambiguously by ray-space BIC:

| Candidate model | Support RMS | Full-grid RMS | BIC |
|---|---:|---:|---:|
| Central pinhole | 2.99 mm | 3.71 mm | +1 052 |
| Central Brown-Conrady | 2.14 mm | 2.65 mm | −10 772 |
| Pinhole + inclined plate | **0.00026 mm** | **0.00335 mm** | **−306 399** |

See:

```text
docs/IDENTIFY_MY_OPTICS.md
docs/CMO_MODEL_SELECTION.md
examples/notebooks/06_cmo_model_selection.ipynb
```

### 4. Real CMO microscope case study

The Pycaso CMO microscope case study is the main real-data validation of the non-central workflow.

Measured results:

| Model / method | Role | RMS |
|---|---|---:|
| Standard OpenCV stereo on the tested setup | central baseline | >300 px |
| Perspective CMO physical model | wrong optical family | ~86 px |
| Telecentric CMO 14p | correct family but not usable | ~14.6 px |
| **Telecentric CMO + per-arm SE(3), 26p** | compact usable physical model | **1.06 px** |
| **Zernike O(0)+d(2), 57p** | flexible rayfield reference | **0.47 px** |

The key result is not only the subpixel Zernike fit. The case study shows a complete diagnostic chain:

```text
measure rayfield → analyze Pluecker residuals → reject wrong hypotheses → add per-arm SE(3) → obtain a compact physical CMO model
```

See:

```text
docs/REAL_CMO_PYCASO_RAYFIELD.md
docs/CMO_PHYSICAL_MODEL.md
examples/notebooks/09_pycaso_real_data.ipynb
```

The generated R3XA metadata file for the case study is:

```text
docs/assets/pycaso_real_data/pycaso_cmo_calibration.r3xa.json
```

## Installation

Editable install:

```bash
.venv/bin/python -m pip install -e .
```

With notebook support:

```bash
.venv/bin/python -m pip install -e '.[notebooks]'
```

With documentation dependencies:

```bash
.venv/bin/python -m pip install -e '.[docs]'
```

If you do not install the package, most examples can also be run with `PYTHONPATH=src`.

## Quickstart from OpenCV

```python
from pathlib import Path
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.07,
    marker_size_mm=27.35,
    aruco_dictionary="DICT_4X4_1000",
)

report = sc.compare_opencv_stereo_calibration(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
)

print(report["raw"]["stereo_rms_px"])
print(report["refined"]["stereo_rms_px"])

K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()
```

For the guided version, use:

```text
examples/notebooks/00_getting_started.ipynb
```

## Documentation map

First use:

- `examples/notebooks/00_getting_started.ipynb`
- `docs/FROM_OPENCV_TO_STEREOCOMPLEX.md`
- `docs/START_HERE.md`
- `docs/NOTEBOOKS.md`

Practical calibration:

- `docs/BRING_YOUR_OWN_DATA.md`
- `docs/FIX_MY_CALIBRATION.md`
- `docs/CHARUCO_IDENTIFICATION.md`
- `docs/RAYFIELD_WORKED_EXAMPLE.md`

Ray-based and non-central calibration:

- `docs/RAYFIELD3D_RECONSTRUCTION.md`
- `docs/NONCENTRAL_FROM_IMAGES.md`
- `docs/PARALLEL_PLATE_ORIGIN_FIELD.md`
- `docs/IDENTIFY_MY_OPTICS.md`

Real-data CMO case study:

- `docs/REAL_CMO_PYCASO_RAYFIELD.md`
- `docs/CMO_PHYSICAL_MODEL.md`
- `examples/notebooks/09_pycaso_real_data.ipynb`

Reference:

- `docs/PUBLIC_API.md`
- `docs/ARCHITECTURE.md`
- `docs/DATASET_SPEC.md`
- `docs/CONVENTIONS.md`
- `docs/ALTERNATIVES_POSITIONING.md`

## Build documentation

```bash
make -C docs html
```

Run the local validation suite before committing:

```bash
bash scripts/validate_local.sh
```

## Scope and non-goals

StereoComplex is not a SLAM, VIO, camera-IMU, ROS, or Structure-from-Motion framework. It focuses on stereo calibration, ChArUco observation quality, ray-based diagnostics, and compact physical interpretation of measured rayfields.

## License

- Code: GNU General Public License v2.0 or later (GPL-2.0-or-later), see `LICENSE`.
- Documentation (`docs/`): Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), see `docs/LICENSE.md`.
