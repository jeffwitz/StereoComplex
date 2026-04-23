# Example notebooks

These notebooks are intentionally lightweight: they read the versioned JSON
summaries and a few committed synthetic images already present in the
repository, so you can understand the workflows without rerunning the full
experiments first.

The default `pip install -e .` now brings in the Jupyter stack used by these
examples (`jupyterlab`, `nbconvert`, `ipykernel`), so the walkthroughs open
directly from the repository environment.

## 01 Ray2D vs OpenCV

`01_ray2d_vs_opencv.ipynb` shows:

- a synthetic stereo example,
- the visual effect of `rayfield_tps_robust`,
- the impact on OpenCV pinhole calibration,
- and a robustness sweep comparing results with and without Ray2D.

## 02 ray3D

`02_ray3d.ipynb` shows:

- the compact central 3D ray-field workflow,
- the Pycaso-style depth-sweep comparison,
- the compression comparison where ray3D is the backend of interest,
- and the public API entry point for loading an exported model.

## Open locally

Open the notebooks from the repository root so relative paths resolve cleanly:

```bash
jupyter lab examples/notebooks
```

If you do not have Jupyter installed, open the `.ipynb` files directly in VS
Code or another notebook viewer.
