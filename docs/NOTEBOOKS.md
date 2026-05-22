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

## Open in Colab

> **Temporary — these links pin the `develop` branch.** `main` is stale and its
> `blob/main/...` Colab links 404. Revert these to `main` after the next
> `develop` → `main` merge (see the *Colab branch hack* note in `CLAUDE.md`).

- [00_getting_started](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/00_getting_started.ipynb)
- [01_ray2d_vs_opencv](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/01_ray2d_vs_opencv.ipynb)
- [02_ray3d](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/02_ray3d.ipynb)
- [03_rayfield_virtual_rectification](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/03_rayfield_virtual_rectification.ipynb)
- [04_parallel_plate_origin_field](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/04_parallel_plate_origin_field.ipynb)
- [05_noncentral_calibration_from_images](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/05_noncentral_calibration_from_images.ipynb)
- [06_cmo_model_selection](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/06_cmo_model_selection.ipynb)
- [07_model_selection_matrix](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/07_model_selection_matrix.ipynb)
- [08_direct_vs_rayfield_inversion](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/08_direct_vs_rayfield_inversion.ipynb)
- [09_pycaso_real_data](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/09_pycaso_real_data.ipynb)

## Recommended reading order

### A. Onboarding

| Notebook | What it teaches | Companion script |
|---|---|---|
| **`00_getting_started.ipynb`** | **Start here if you know OpenCV.** Compare raw vs Ray2D-refined calibration, check quality, export to OpenCV format. | `00_getting_started.py` |
| `01_ray2d_vs_opencv.ipynb` | How the 2D planar refinement (`rayfield_tps_robust`) changes ChArUco detection and why that matters for standard OpenCV calibration. | `01_ray2d_vs_opencv.py` |

### B. Central rayfield and reconstruction

| Notebook | What it teaches | Companion script |
|---|---|---|
| `02_ray3d.ipynb` | How the compact central 3D ray-field is calibrated, how the Pycaso-style sweeps are organized, and how the compression experiments are read. | `02_ray3d.py` |
| `03_rayfield_virtual_rectification.ipynb` | How a ray-field can be turned into virtual rectification maps and then reused with a classic dense stereo pipeline. | `03_rayfield_virtual_rectification.py` |

### C. Non-central and physical model identification

| Notebook / script | What it teaches | Companion script |
|---|---|---|
| `04_parallel_plate_origin_field.ipynb` | How an inclined parallel-plate oracle creates non-central stereo data, renders ChArUco images with vignetting/blur/noise, and runs Zernike rayfield BA (`O(u,v)`, `d(u,v)`, poses, rig). | `04_parallel_plate_origin_field.py` |
| `05_noncentral_calibration_from_images.ipynb` | Practical workflow: fit a non-central Zernike origin-field model from two image folders and a ChArUco board definition. | `05_noncentral_calibration_from_images.py` |
| `06_cmo_model_selection.ipynb` | CMO workflow: generate a ChArUco CMO scene, measure generic Zernike `O(u,v), d(u,v)` rayfields, then select among pinhole, Brown-Conrady, plate, and CMO physical candidates. | `06_cmo_model_selection.py` |
| `07_model_selection_matrix.py` | Script: run the complete 6-oracle classification matrix, noiseless and under noise. | `07_model_selection_matrix.py` |
| `08_direct_vs_rayfield_inversion.py` | Script: compare direct ChArUco inversion (pipeline A) against rayfield-mediated selection (pipeline B) on a CMO oracle. | `08_direct_vs_rayfield_inversion.py` |

### D. Real-data case study

| Notebook | What it teaches | Companion script |
|---|---|---|
| **`09_pycaso_real_data.ipynb`** | **Real-data demonstration on a Pycaso CMO microscope.** Legacy ChArUco detection, Hessian completion, Ray2D TPS, constrained Zernike rayfield, CMO descriptor extraction, telecentricity diagnosis, physical CMO+SE(3) model, and final reproducible artifacts. | `09_pycaso_real_data.py` |

## What to look for

### 00 Getting started — from OpenCV to StereoComplex

`00_getting_started.ipynb` is the recommended first notebook for OpenCV users:

- define a ChArUco board;
- compare OpenCV raw vs Ray2D-refined calibration;
- check quality with `assess_calibration`;
- export to OpenCV format with `result.to_opencv()`.

Companion guide: [From OpenCV to StereoComplex](FROM_OPENCV_TO_STEREOCOMPLEX.md).

### 01 Ray2D vs OpenCV

Use this notebook if you want the intuition behind the 2D preprocessing stage.
It starts from the practical onboarding path:

- define `left_dir`;
- define `right_dir`;
- define `CharucoBoardSpec`;
- run `fit_opencv_stereo_from_image_dirs(..., method2d="raw")`;
- run `fit_opencv_stereo_from_image_dirs(..., method2d="rayfield_tps_robust")`.

Only after that does it move to the synthetic ground-truth overlays and the
released benchmark summaries.

### 02 Ray3D

Use this notebook if you want the central 3D backend story. It walks through the
dataset parameters, the Z-sweep, the compression stress test, and the comparison
against the Pycaso-style baselines.

### 03 Virtual rectification

Use this notebook if you want the bridge back to classical dense stereo. It
shows how the ray-field is converted into dense remap tables and how the
rectified pairs can be fed to a standard matcher such as StereoSGBM.

### 04 Parallel plate origin field

Use this notebook if you want the controlled non-central story. It walks through
the inclined-plate oracle, the central-stereo failure mode, the staged Zernike
`O(u,v)` fit, and the complete geometric BA over `O(u,v)`, `d(u,v)`, poses, and
the stereo rig. It also renders ChArUco images with vignetting/blur/noise,
detects them with OpenCV, and feeds those detections to the same non-central BA.
It is the executable companion to [Parallel plate origin field](PARALLEL_PLATE_ORIGIN_FIELD.md).

### 05 Non-central calibration from images

Use this notebook if you want the practical user-facing non-central path. It
starts with two image directories and a `CharucoBoardSpec`, then calls
`fit_stereo_zernike_origin_field_from_image_dirs(...)`.

The inclined-plate oracle is only used to generate self-contained example
images. The notebook is written so you can replace `left_dir`, `right_dir`, and
`board` with your own data.

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

### 07 Model selection classification matrix

`07_model_selection_matrix.py` runs the complete 6-oracle classification matrix
(pinhole, Brown, plate, CMO, Greenough, exotic), both noiseless and under
20 µm measurement noise.

Scientific companion page: [CMO model selection](CMO_MODEL_SELECTION.md).

### 08 Direct vs rayfield-mediated inversion (Synthèse)

`08_direct_vs_rayfield_inversion.py` compares fitting optical models directly to
ChArUco corners (pipeline A) against the rayfield-mediated strategy (pipeline B)
on synthetic oracles (CMO, Brown, pinhole, parallel-plate).

Scientific companion page: [Rayfield mediation](DIRECT_VS_RAYFIELD_INVERSION.md).

### 09 Pycaso real data — validation on a real CMO microscope

`09_pycaso_real_data.ipynb` is the real-data counterpart of notebook 08: the
rayfield-mediated pipeline (B) validated on real Pycaso CMO calibration images.
While notebook 08 asks "is the rayfield better than direct inversion?", notebook 09
asks "does it work on a real microscope?" — and answers yes (1.06 px with the
26-parameter CMO+SE(3) physical model).

The two notebooks form a pair: **synthetic validation** (08) followed by
**real-data demonstration** (09).
StereoComplex pipeline on a physical CMO stereo microscope (Pycaso):

- **ChArUco detection** with `legacy_pattern=True` for the older Pycaso board
  convention (`DICT_6X6_250`, 16×12 squares, 0.3 mm).
- **Hessian corner completion** (`|det H|` + Otsu + barycentre) fills missing
  corners and yields 165/165 points on every frame.
- **Ray2D TPS denoising** (`predict_points_rayfield_tps_robust`) smooths the
  corner positions.
- **Zernike rayfield fit** with constrained poses achieves **0.47 px** local
  pixel-equivalent RMS for `O(0)+d(2)` with 57 parameters.
- **CMO-consistent geometric descriptors** are read directly from the rayfield:
  baseline about 24.9 mm, effective objective focal length about 62 mm, working
  distance about 65 mm, and convergence angle about 22.6 degrees.
- **Telecentricity diagnosis and physical modelling** show that a perspective
  CMO model is the wrong family, a 14p telecentric model identifies the family
  but is not usable in reprojection, and a compact telecentric CMO + per-arm
  SE(3) model reaches **1.06 px** RMS with 26 parameters.

Key result: StereoComplex separates a flexible subpixel rayfield reference
(**0.47 px**, 57p) from a compact physically interpretable CMO model
(**1.06 px**, 26p). Standard central OpenCV stereo calibration does not converge
to a usable model under the tested configuration.

Scientific companion pages: [Real CMO Pycaso rayfield](REAL_CMO_PYCASO_RAYFIELD.md),
[CMO Physical Model](CMO_PHYSICAL_MODEL.md), and [Identify My Optics](IDENTIFY_MY_OPTICS.md).

## Open locally

Open the notebooks from the repository root so relative paths resolve cleanly:

```bash
jupyter lab examples/notebooks
```

If Jupyter is not installed, the notebook files can still be opened directly in
VS Code or another notebook viewer.