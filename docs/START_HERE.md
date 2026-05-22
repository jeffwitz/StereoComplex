# Start Here

Goal: choose the right StereoComplex workflow without reading the whole research documentation first.

StereoComplex has a deliberately layered design:

| Layer | Use it for | Status |
|---|---|---|
| **Ray2D / planar ray-field** | improve ChArUco observations before OpenCV calibration | stable practical preprocessing |
| **Central 3D ray-field** | ray-based stereo calibration/reconstruction without a pinhole matrix | research prototype |
| **Non-central Zernike rayfield** | diagnose and model systems where each pixel maps to a 3D line | research-grade, validated on synthetic oracles and one real CMO case study |

Ray2D is a 2D correction of board-plane observations. It is not itself a 3D non-central camera model.

## Recommended first step

Start with notebook 00 — [open it directly in Google Colab](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/00_getting_started.ipynb),
or run it locally (see the *Notebook walkthroughs* section below):

```text
examples/notebooks/00_getting_started.ipynb
examples/notebooks/00_getting_started.py
```

This is the shortest path for users who already know OpenCV:

1. define a ChArUco board;
2. compare raw OpenCV calibration against Ray2D-refined calibration;
3. call `assess_calibration`;
4. export the refined result back to OpenCV format.

Companion page:

- :doc:`FROM_OPENCV_TO_STEREOCOMPLEX`

## Installation

Editable install:

```bash
.venv/bin/python -m pip install -e .
```

For notebooks:

```bash
.venv/bin/python -m pip install -e '.[notebooks]'
```

If you do not install the package, most commands can still be run with `PYTHONPATH=src`.

## Which path should I use?

| Situation | Recommended path |
|---|---|
| I know OpenCV and want a quick start | `examples/notebooks/00_getting_started.ipynb` |
| I just want better OpenCV calibration | `sc.compare_opencv_stereo_calibration(...)` or `sc.calibrate_opencv(..., method2d="rayfield_tps_robust")` |
| My OpenCV calibration plateaus on blur/noise/compression | :doc:`FIX_MY_CALIBRATION` |
| I have my own left/right folders | :doc:`BRING_YOUR_OWN_DATA` |
| I want to understand Ray2D | `examples/notebooks/01_ray2d_vs_opencv.ipynb` and :doc:`RAYFIELD_WORKED_EXAMPLE` |
| I want a central ray-based 3D model | `sc.calibrate_central(...)` and :doc:`RAYFIELD3D_RECONSTRUCTION` |
| I suspect non-central optics | `sc.calibrate_noncentral(...)` and :doc:`NONCENTRAL_FROM_IMAGES` |
| I want to identify a physical optical family | `sc.identify_optics(...)` and :doc:`IDENTIFY_MY_OPTICS` |
| I want the real CMO validation | :doc:`REAL_CMO_PYCASO_RAYFIELD` and `examples/notebooks/09_pycaso_real_data.ipynb` |

## What StereoComplex does today

StereoComplex currently provides:

- a CPU synthetic stereo dataset generator with ChArUco targets;
- ChArUco detection and Ray2D refinement tools;
- OpenCV-compatible calibration wrappers and diagnostics;
- central 3D ray-field calibration and triangulation prototypes;
- non-central Zernike rayfield calibration from image folders;
- physical optical model identification from measured rayfields;
- a real-data Pycaso CMO microscope case study.

## Main validated results

### OpenCV-compatible calibration improvement

Ray2D can improve ChArUco localization before feeding points into OpenCV calibration. Start with:

- `examples/notebooks/00_getting_started.ipynb`
- `examples/notebooks/01_ray2d_vs_opencv.ipynb`
- :doc:`FROM_OPENCV_TO_STEREOCOMPLEX`
- :doc:`FIX_MY_CALIBRATION`

### Non-central rendered-image benchmark

On the inclined-plate benchmark, the non-central bundle adjustment benefits strongly from Ray2D-refined observations and reaches sub-millimetric reconstruction accuracy. Read:

- :doc:`PARALLEL_PLATE_ORIGIN_FIELD`
- `examples/notebooks/04_parallel_plate_origin_field.ipynb`

### Real CMO microscope case study

The Pycaso CMO case study is the main real-data validation of the non-central workflow. The final documented comparison is:

| Model / method | Role | RMS |
|---|---|---:|
| Standard OpenCV stereo on the tested setup | central baseline | >300 px |
| Perspective CMO physical model | wrong optical family | ~86 px |
| Telecentric CMO 14p | correct family but not usable | ~14.6 px |
| **Telecentric CMO + per-arm SE(3), 26p** | compact usable physical model | **1.06 px** |
| **Zernike O(0)+d(2), 57p** | flexible rayfield reference | **0.47 px** |

Read:

- :doc:`REAL_CMO_PYCASO_RAYFIELD`
- :doc:`CMO_PHYSICAL_MODEL`
- `examples/notebooks/09_pycaso_real_data.ipynb`

## Notebook walkthroughs

The ten numbered notebooks are the guided teaching sequence; see
:doc:`NOTEBOOKS` for the detailed overview.

**Run them locally** — install the notebook extras, then launch JupyterLab.
Each notebook's first cell moves to the repository root on its own, so the
relative `dataset/` paths resolve wherever JupyterLab is started:

```bash
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/jupyter lab examples/notebooks
```

**Run them on Google Colab** — click a link below. The first cell installs
StereoComplex automatically; no local setup is needed.

Recommended order:

1. [`00_getting_started.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/00_getting_started.ipynb) — OpenCV-to-StereoComplex onboarding.
2. [`01_ray2d_vs_opencv.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/01_ray2d_vs_opencv.ipynb) — why Ray2D improves ChArUco observations.
3. [`02_ray3d.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/02_ray3d.ipynb) — central 3D ray-field reconstruction prototype.
4. [`03_rayfield_virtual_rectification.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/03_rayfield_virtual_rectification.ipynb) — bridge back to dense stereo.
5. [`04_parallel_plate_origin_field.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/04_parallel_plate_origin_field.ipynb) — non-central synthetic oracle.
6. [`05_noncentral_calibration_from_images.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/05_noncentral_calibration_from_images.ipynb) — practical non-central image-folder path.
7. [`06_cmo_model_selection.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/06_cmo_model_selection.ipynb) — physical model selection on CMO-like optics.
8. [`07_model_selection_matrix.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/07_model_selection_matrix.ipynb) — 6-oracle classification matrix.
9. [`08_direct_vs_rayfield_inversion.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/08_direct_vs_rayfield_inversion.ipynb) — direct vs rayfield-mediated inversion.
10. [`09_pycaso_real_data.ipynb`](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/09_pycaso_real_data.ipynb) — real Pycaso CMO case study.

## Quickstart for your own stereo folders

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

report = sc.compare_opencv_stereo_calibration(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
)

assessment = sc.assess_calibration(report["refined_result"])
K1, d1, K2, d2, R, T = report["refined_result"].to_opencv()
```

For the detailed version:

- :doc:`BRING_YOUR_OWN_DATA`

## Quickstart for the versioned synthetic dataset

Generate and validate a minimal dataset:

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco --pattern charuco --frames-per-scene 16
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/charuco
```

Evaluate Ray2D ChArUco identification against synthetic ground truth:

```bash
.venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/charuco --method rayfield_tps_robust
```

## Documentation map

- First use: :doc:`FROM_OPENCV_TO_STEREOCOMPLEX`, :doc:`BRING_YOUR_OWN_DATA`, :doc:`NOTEBOOKS`.
- ChArUco and Ray2D: :doc:`CHARUCO_IDENTIFICATION`, :doc:`RAYFIELD_WORKED_EXAMPLE`.
- Ray-based calibration and reconstruction: :doc:`STEREO_RECONSTRUCTION`, :doc:`RECONSTRUCTION_API`, :doc:`RAYFIELD3D_RECONSTRUCTION`, :doc:`RAYFIELD_VIRTUAL_RECTIFY`.
- Non-central and optical model identification: :doc:`NONCENTRAL_FROM_IMAGES`, :doc:`IDENTIFY_MY_OPTICS`, :doc:`PARALLEL_PLATE_ORIGIN_FIELD`, :doc:`CMO_MODEL_SELECTION`.
- Real-data CMO: :doc:`REAL_CMO_PYCASO_RAYFIELD`, :doc:`CMO_PHYSICAL_MODEL`.
- Project status and reference: :doc:`VALIDATION_STATUS`, :doc:`ROADMAP`, :doc:`PUBLIC_API`, :doc:`ARCHITECTURE`, :doc:`CONVENTIONS`.

## Next steps

- Start with `examples/notebooks/00_getting_started.ipynb`.
- Use :doc:`FROM_OPENCV_TO_STEREOCOMPLEX` if you want the OpenCV migration path.
- Use :doc:`REAL_CMO_PYCASO_RAYFIELD` if you want the advanced scientific validation.
- Add more real datasets before claiming deployment-grade generality for non-central optics.
