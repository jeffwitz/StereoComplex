# Notebook walkthroughs

The notebooks are the most guided entry point into StereoComplex. They are
written as teaching material, not just as runnable code dumps:

- each notebook starts with a short motivation,
- the relevant equations are repeated inline in Markdown/LaTeX,
- the figures are narrated step by step,
- and the companion `.py` export is kept next to the notebook for quick review
  in a plain editor.

All notebooks read committed synthetic images and JSON summaries already stored
in the repository, so you can inspect the workflow without rerunning the full
benchmarks first. The two small scene folders used by the walkthroughs are also
versioned directly in Git:

- `dataset/compression_sweep_pnp/png_lossless/train/scene_0000`
- `dataset/v0_png/train/scene_0000`

## Recommended reading order

| Notebook | What it teaches | Companion script |
|---|---|---|
| **`00_getting_started.py`** | **Start here if you know OpenCV.**  Compare raw vs Ray2D-refined calibration, check quality, export to OpenCV format. | `00_getting_started.py` |
| `01_ray2d_vs_opencv.ipynb` | How the 2D planar refinement (`rayfield_tps_robust`) changes ChArUco detection and why that matters for standard OpenCV calibration. | `01_ray2d_vs_opencv.py` |
| `02_ray3d.ipynb` | How the compact central 3D ray-field is calibrated, how the Pycaso-style sweeps are organized, and how the compression experiments are read. | `02_ray3d.py` |
| `03_rayfield_virtual_rectification.ipynb` | How a ray-field can be turned into virtual rectification maps and then reused with a classic dense stereo pipeline. | `03_rayfield_virtual_rectification.py` |
| `04_parallel_plate_origin_field.ipynb` | How an inclined parallel-plate oracle creates non-central stereo data, renders ChArUco images with vignetting/blur/noise, and runs Zernike rayfield BA (`O(u,v)`, `d(u,v)`, poses, rig). | `04_parallel_plate_origin_field.py` |
| `05_noncentral_calibration_from_images.ipynb` | Practical workflow: fit a non-central Zernike origin-field model from two image folders and a ChArUco board definition. | `05_noncentral_calibration_from_images.py` |
| `06_cmo_model_selection.ipynb` | CMO workflow: generate a ChArUco CMO scene, measure generic Zernike `O(u,v), d(u,v)` rayfields, then select among pinhole, Brown-Conrady, plate, and CMO physical candidates. | `06_cmo_model_selection.py` |
| `07_model_selection_matrix.py` | Run the complete 6-oracle classification matrix (pinhole through exotic), noiseless and under noise. | `07_model_selection_matrix.py` |
| `08_direct_vs_rayfield_inversion.py` | Compare direct ChArUco inversion (pipeline A) against rayfield-mediated selection (pipeline B) on a CMO oracle. | `08_direct_vs_rayfield_inversion.py` |

Use notebook 04 if you want to understand why and how the non-central model
works on a controlled physical oracle. Use notebook 05 if you want the shortest
practical path from left/right calibration images to a fitted non-central
origin-field model.

## What to look for

### 01 Ray2D vs OpenCV

Start here if you want the intuition behind the 2D preprocessing stage. The
notebook now begins with the real onboarding path:

- define `left_dir`,
- define `right_dir`,
- define `CharucoBoardSpec`,
- run `fit_opencv_stereo_from_image_dirs(..., method2d="raw")`,
- run `fit_opencv_stereo_from_image_dirs(..., method2d="rayfield_tps_robust")`.

Only after that does it move to the synthetic GT overlays and then to the
released benchmark summaries.

### 02 ray3D

Use this notebook if you want the 3D backend story. It walks through the
dataset parameters, the Z-sweep, the compression stress test, and the
comparison against the Pycaso-style baselines.

### 03 Virtual rectification

Use this notebook if you want the bridge back to classical dense stereo. It
shows how the ray-field is converted into dense remap tables and how the
rectified pairs can be fed to a standard matcher such as StereoSGBM.

### 04 Parallel plate origin field

Use this notebook if you want the non-central story. It walks through the
inclined-plate oracle, the central-stereo failure mode, the staged Zernike
`O(u,v)` fit, and the complete geometric BA over `O(u,v)`, `d(u,v)`, poses, and
the stereo rig. It also renders ChArUco images with vignetting/blur/noise,
detects them with OpenCV, and feeds those detections to the same non-central BA.
It is the executable companion to :doc:`PARALLEL_PLATE_ORIGIN_FIELD`.

### 05 Non-central calibration from images

Use this notebook if you want the practical user-facing path. It starts with two
image directories and a `CharucoBoardSpec`, then calls
`fit_stereo_zernike_origin_field_from_image_dirs(...)`. The inclined-plate oracle
is only used to generate self-contained example images; the notebook is written
so you can replace `left_dir`, `right_dir`, and `board` with your own data.

This notebook intentionally hides most of the research benchmark details. For
the scientific validation and complete BA discussion, use notebook 04.

### 06 CMO model selection

Use this notebook if you want the ray-space model-selection story on a CMO-like
optical system. It generates a ChArUco CMO scene from the shared
`stereocomplex.physics` model, fits generic Zernike `O(u,v), d(u,v)` rayfields
to the generated rays, and then compares physical candidates in ray space. The
notebook compares a shared physical CMO candidate against generic per-channel
candidates, including a polynomial surrogate that uses the same low-level
Brown-Conrady and polynomial angular primitives.

Scientific companion page: [CMO model selection](CMO_MODEL_SELECTION.md).

### 00 Getting started — from OpenCV to StereoComplex

`00_getting_started.py` is the recommended first notebook for OpenCV users:
- define a ChArUco board,
- compare OpenCV raw vs Ray2D-refined calibration,
- check quality with `assess_calibration`,
- export to OpenCV format with `result.to_opencv()`.

Companion guide: [From OpenCV to StereoComplex](FROM_OPENCV_TO_STEREOCOMPLEX.md).

### 07 Model selection classification matrix

`07_model_selection_matrix.py` runs the complete 6-oracle classification
matrix (pinhole, Brown, plate, CMO, Greenough, exotic), both noiseless
and under 20 µm measurement noise.

Scientific companion page: [CMO model selection](CMO_MODEL_SELECTION.md).

### 08 Direct vs rayfield-mediated inversion

`08_direct_vs_rayfield_inversion.py` compares fitting optical models
directly to ChArUco corners (pipeline A) against the rayfield-mediated
strategy (pipeline B) on a CMO oracle.

Scientific companion page: [Rayfield mediation](DIRECT_VS_RAYFIELD_INVERSION.md).

## Open locally

Open the notebooks from the repository root so relative paths resolve cleanly:

```bash
jupyter lab examples/notebooks
```

If Jupyter is not installed, the notebook files can still be opened directly in
VS Code or another notebook viewer.
