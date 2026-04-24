# Example notebooks

These notebooks are intentionally lightweight:

- the **local workflow** parts now use the public StereoComplex API,
- the **global benchmark** plots still read versioned JSON summaries and
  committed synthetic images already present in the repository.

This keeps the notebooks pedagogical without asking you to rerun the full
experiments first.

The default `pip install -e .` now brings in the Jupyter stack used by these
examples (`jupyterlab`, `nbconvert`, `ipykernel`), so the walkthroughs open
directly from the repository environment.

## 01 Ray2D vs OpenCV

`01_ray2d_vs_opencv.ipynb` shows, with the companion script `01_ray2d_vs_opencv.py`:

- the exact onboarding path `left_dir + right_dir + CharucoBoardSpec`,
- raw OpenCV stereo calibration vs `Ray2D + OpenCV`,
- then a synthetic stereo example used for controlled GT overlays,
- the visual effect of `rayfield_tps_robust`,
- the impact on OpenCV pinhole calibration,
- and a robustness sweep comparing results with and without Ray2D.

It now starts from the same three user inputs you would have on real data:
`left_dir`, `right_dir`, and `board`. The first executed cell runs
`fit_opencv_stereo_from_image_dirs(..., method2d="raw")` and
`fit_opencv_stereo_from_image_dirs(..., method2d="rayfield_tps_robust")`
before moving to the synthetic GT-specific visual explanations.

## 02 ray3D

`02_ray3d.ipynb` shows, with the companion script `02_ray3d.py`:

- the compact central 3D ray-field workflow,
- the Pycaso-style depth-sweep comparison,
- the compression comparison where ray3D is the backend of interest,
- and the public API entry point for loading an exported model.

It now begins with the same synthetic-setup summary, so the Z-sweep and pose-sweep plots can be
read in the context of the exact camera model and simulator degradations, and it
demonstrates `fit_stereo_central_rayfield_from_dataset(...)` as the public
calibration path.

## 03 Virtual rectification

`03_rayfield_virtual_rectification.ipynb` shows, with the companion script `03_rayfield_virtual_rectification.py`:

- how to build virtual rectification maps from a calibrated ray-field,
- how to recover scanline-aligned stereo pairs with `cv2.remap`,
- how to run a standard dense matcher after rectification,
- and how to inspect the vertical-disparity sanity checks before/after rectification.

It now obtains the calibration model through the public API before building the
virtual rectification maps.

## Open locally

Open the notebooks from the repository root so relative paths resolve cleanly:

```bash
jupyter lab examples/notebooks
```

If you do not have Jupyter installed, open the `.ipynb` files directly in VS
Code or another notebook viewer.
