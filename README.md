# StereoComplex

Stereo calibration and 3D reconstruction research prototype, built around:

- a CPU synthetic-data generator (digital twins) for stereo + ChArUco,
- a 2D “ray-field” correction (homography + smooth residual field) to improve ChArUco corner localization,
- an experimental 3D ray-based calibration prototype (central ray-field, Zernike basis) designed as a stepping stone towards complex/non-pinhole optics.

## Why would you use this?

If your OpenCV calibration plateaus because of blur / distortion / compression, StereoComplex provides a practical lever:
**refine the ChArUco corners before calibration** (without assuming a global pinhole model for the refinement).

Visual proof (green = GT, red = OpenCV raw, blue = ray-field):

- `docs/assets/rayfield_worked_example/micro_overlays/left_best_frame000000.png`
- `docs/assets/rayfield_worked_example/micro_overlays/right_best_frame000000.png`

## Highlights (from the provided examples)

- **2D ChArUco accuracy improvement (example)**: RMS corner error drops from ~0.357 px → ~0.219 px (left) and ~0.356 px → ~0.153 px (right) with the 2D ray-field correction.
- **OpenCV stereo calibration impact (example)**: feeding OpenCV with ray-field-corrected corners improves mono RMS (~0.306/0.302 px → ~0.079/0.061 px), improves stereo RMS (~0.381 px → ~0.163 px), and reduces baseline error in disparity-equivalent pixels (~0.424 px → ~0.205 px).

See `docs/RAYFIELD_WORKED_EXAMPLE.md` and `docs/STEREO_RECONSTRUCTION.md` for full methodology, plots, and definitions.

## Installation

Core dependencies are in `pyproject.toml` (NumPy, Pillow, SciPy). For ChArUco/ArUco features, you also need OpenCV with `aruco`:

- recommended: `opencv-contrib-python` (provides `cv2.aruco`).

Editable install:

```bash
.venv/bin/python -m pip install -e .
```

## Quickstart (CPU dataset generator)

CLI help:

```bash
.venv/bin/python -m stereocomplex.cli --help
```

Generate a minimal synthetic dataset:

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/v0 --scenes 2 --frames-per-scene 16 --width 640 --height 480
```

ChArUco + blur (e.g., 8 µm FWHM):

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_blur --pattern charuco --blur-fwhm-um 8
```

Stronger edge blur (variable PSF approximation):

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_edgeblur --pattern charuco --blur-fwhm-um 6 --blur-edge-factor 3 --blur-edge-start 0.5
```

Texture interpolation (anti-alias):

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_interp --pattern charuco --tex-interp lanczos4
```

Geometric aberrations (distortion):

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_dist --pattern charuco --distort brown --distort-strength 0.5
```

Black background outside the board + lossless WebP:

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco_webp_black --pattern charuco --image-format webp --outside-mask hard
```

Validate dataset consistency:

```bash
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/v0
```

Oracle eval (synthetic sanity check: very small reprojection/triangulation errors expected):

```bash
.venv/bin/python -m stereocomplex.cli eval-oracle dataset/v0
```

Note: if you prefer not to install the package, you can prefix commands with `PYTHONPATH=src`.

## Quickstart (fix OpenCV calibration on a dataset scene)

Export refined ChArUco corners (JSON + an OpenCV-ready NPZ):

```bash
.venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png --split train --scene scene_0000 \
  --method rayfield_tps_robust \
  --out-json paper/tables/refined_corners_scene0000.json \
  --out-npz paper/tables/refined_corners_scene0000_opencv.npz
```

## Documentation

Start here:

- `docs/START_HERE.md`
- `docs/ARCHITECTURE.md`
- `docs/DATASET_SPEC.md`
- `docs/CONVENTIONS.md`

Core method pages:

- `docs/CHARUCO_IDENTIFICATION.md`
- `docs/FIX_MY_CALIBRATION.md`
- `docs/RAYFIELD_WORKED_EXAMPLE.md`
- `docs/STEREO_RECONSTRUCTION.md`
- `docs/RAYFIELD3D_RECONSTRUCTION.md`
- `docs/RECONSTRUCTION_API.md`

### Sphinx / ReadTheDocs

Build local HTML docs:

```bash
.venv/bin/python -m pip install -e .[docs]
make -C docs html
```

Build PDF (LaTeX):

```bash
make -C docs latexpdf
```
