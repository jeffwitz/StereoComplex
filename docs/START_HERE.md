# Start Here

Goal: establish a **reproducible** and **measurable** baseline for stereo calibration, ChArUco identification, and ray-based reconstruction.

## Installation (recommended)

This repository is a Python package. For consistent CLI and imports, use an editable install:

```bash
.venv/bin/python -m pip install -e .
```

The default editable install brings the core runtime dependencies. If you want the walkthrough notebooks,
install the notebook extra as well:

```bash
.venv/bin/python -m pip install -e '.[notebooks]'
```

This adds the Jupyter runtime used by the walkthroughs, so the examples can run directly from a fresh clone.
If you do not install the package, you can still run most commands by prefixing them with `PYTHONPATH=src`.

## What StereoComplex does today

StereoComplex is built around:

- a synthetic stereo dataset generator (CPU) with GT correspondences,
- ChArUco detection evaluation against GT,
- a 2D “ray-field” correction (`rayfield_tps_robust`) to improve corner localization,
- a stereo calibration / triangulation evaluation pipeline (OpenCV),
- an experimental central 3D ray-field (Zernike) calibrated by point↔ray bundle adjustment.

## What is already implemented

- Dataset v0: `docs/DATASET_SPEC.md` + validator (`validate-dataset`)
- Conventions (pixel centers, frames): `docs/CONVENTIONS.md`
- CPU dataset generator: `src/stereocomplex/sim/cpu/generate_dataset.py`
- Evaluations: `src/stereocomplex/eval/` and `paper/experiments/`
- Worked example (end-to-end 2D ray-field): `docs/RAYFIELD_WORKED_EXAMPLE.md`
- Virtual rectification demo (ray-field -> rectified dense stereo): `docs/examples/rayfield_virtual_rectification_demo.py`
- Stereo calibration + reconstruction study: `docs/STEREO_RECONSTRUCTION.md`
- Robustness sweep: `docs/ROBUSTNESS_SWEEP.md`
- Central 3D ray-field + point↔ray bundle adjustment: `docs/RAYFIELD3D_RECONSTRUCTION.md`

## Notebook walkthroughs

If you want a more guided, executable entry point, open the notebook overview
page first:

- :doc:`NOTEBOOKS`

The three notebooks are then available as:

- `examples/notebooks/01_ray2d_vs_opencv.ipynb` and `examples/notebooks/01_ray2d_vs_opencv.py`
- `examples/notebooks/02_ray3d.ipynb` and `examples/notebooks/02_ray3d.py`
- `examples/notebooks/03_rayfield_virtual_rectification.ipynb` and `examples/notebooks/03_rayfield_virtual_rectification.py`

They reuse the committed synthetic images, overlays, and JSON summaries already stored in this repository.
The two sample scenes used by the notebooks are versioned in Git, so the notebooks open
without having to regenerate the benchmark data first.

## Quickstart

### Quickstart for your own stereo folders

If you already have `left/*.png`, `right/*.png`, and a known ChArUco board, the
shortest public API path is:

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
    export_model_dir=Path("models/my_calibration"),
)
```

The full walkthrough is here:

- :doc:`BRING_YOUR_OWN_DATA`

### Quickstart for the versioned synthetic dataset

Generate a minimal dataset and validate it:

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco --pattern charuco --frames-per-scene 16
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/charuco
```

Measure raw vs ray-field ChArUco identification against GT (synthetic datasets only):

```bash
.venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/charuco --method rayfield_tps_robust
```

Export refined corners for OpenCV calibration (JSON + NPZ):

```bash
.venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png --split train --scene scene_0000 --max-frames 5 \
  --method rayfield_tps_robust \
  --out-json paper/tables/refined_corners_scene0000.json \
  --out-npz paper/tables/refined_corners_scene0000_opencv.npz
```

Run the end-to-end worked example (plots + overlays):

```bash
.venv/bin/python docs/examples/rayfield_charuco_end_to_end.py dataset/v0_png --split train --scene scene_0000 --out docs/assets/rayfield_worked_example --save-overlays
```

Run the stereo calibration/reconstruction comparison (OpenCV):

```bash
.venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/v0_png --split train --scene scene_0000 --out paper/tables/opencv_calibration_rayfield.json
```

## Next steps (recommended)

- Try the public calibration wrappers on your own stereo folders: :doc:`BRING_YOUR_OWN_DATA`
- Extend the simulator and add more varied scenes/optics (including non-central behavior).
- Generalize the 3D ray-field from **central** to **non-central** (pixel → 3D line).
