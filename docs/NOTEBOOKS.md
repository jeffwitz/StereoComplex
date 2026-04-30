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
| `01_ray2d_vs_opencv.ipynb` | How the 2D planar refinement (`rayfield_tps_robust`) changes ChArUco detection and why that matters for standard OpenCV calibration. | `01_ray2d_vs_opencv.py` |
| `02_ray3d.ipynb` | How the compact central 3D ray-field is calibrated, how the Pycaso-style sweeps are organized, and how the compression experiments are read. | `02_ray3d.py` |
| `03_rayfield_virtual_rectification.ipynb` | How a ray-field can be turned into virtual rectification maps and then reused with a classic dense stereo pipeline. | `03_rayfield_virtual_rectification.py` |
| `04_parallel_plate_origin_field.ipynb` | How an inclined parallel-plate oracle creates non-central stereo data, renders ChArUco images with vignetting/blur/noise, and runs Zernike rayfield BA (`O(u,v)`, `d(u,v)`, poses, rig). | `04_parallel_plate_origin_field.py` |

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

## Open locally

Open the notebooks from the repository root so relative paths resolve cleanly:

```bash
jupyter lab examples/notebooks
```

If Jupyter is not installed, the notebook files can still be opened directly in
VS Code or another notebook viewer.
