# Tutorial: From zero to CMO calibration in 15 minutes

Each notebook link below opens directly in **Google Colab**.
The very first cell handles installation — run it once,
then **Runtime → Restart runtime**, then run all cells.
On your own machine, the first cell is harmless (it detects Colab and skips).
physical calibration of a Common Main Objective stereo microscope.  You will
learn the three layers of the library (Ray2D, Central 3D, Non-Central 3D) by
using them, not by reading about them.

!!! warning "Research prototype"
    StereoComplex v0.x is unstable.  Names may change.  The non‑central path is
    validated on one real CMO dataset — treat it as research‑grade.

## 1.  Installation (2 minutes)

```bash
git clone https://github.com/jeffwitz/StereoComplex.git
cd StereoComplex
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Verify:

```bash
.venv/bin/python -c "import stereocomplex as sc; print(sc.__version__)"
# → 0.1.0
```

## 2.  Calibrate a standard stereo rig with OpenCV (3 minutes)

StereoComplex wraps OpenCV so you can compare raw detections against its own
Ray2D refinement in one call.

```python
import stereocomplex as sc
from pathlib import Path

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

print(report["raw_result"].report)       # OpenCV raw
print(report["refined_result"].report)   # Ray2D-refined
```

**What happened:** the library detected ChArUco corners with OpenCV, refined them
with a smooth 2D planar model (Ray2D TPS), and ran stereo calibration on both.
The report shows reprojection RMS, epipolar error, and reconstruction quality.

**Key take-away:** the Ray2D front-end is a drop‑in improvement over raw OpenCV
detection.  It does not require a camera model — it works on the 2D board plane.

> 📓 **Notebook:** [00_getting_started.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/00_getting_started.ipynb) and
> [01_ray2d_vs_opencv.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/01_ray2d_vs_opencv.ipynb) walk through this step in
> detail with visual comparisons.

## 3.  Move to a central 3D ray‑field (3 minutes)

If your rig is well approximated by a pinhole model, you can replace the
classical `(K1,d1,K2,d2,R,T)` tuple with a compact ray‑field: a learned 3D
direction per pixel that shares a single camera centre.

```python
result = sc.fit_stereo_central_rayfield_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    method2d="rayfield_tps_robust",
    export_model_dir="models/my_rayfield",
)

model = result.model
print(result.report)
```

Triangulate a 3D point from pixel matches:

```python
points_3d = sc.reconstruct_points_central_stereo(
    left_pixels=np.array([[512, 512]]),
    right_pixels=np.array([[520, 512]]),
    model=model,
)
print(points_3d)  # shape (N, 3) in mm
```

**Key take-away:** the central ray‑field is a drop‑in replacement for the
classical OpenCV model.  It triangulates from rays instead of projecting from
pinhole matrices.

> 📓 **Notebook:** [02_ray3d.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/02_ray3d.ipynb) covers the central
> ray‑field and compression sweeps.

## 4.  Detect non‑centrality with a Zernike rayfield (5 minutes)

If you suspect your rig is NOT well modelled by a single optical centre
(microscopes, endoscopes, protective glass, Scheimpflug), the Zernike rayfield
fits a 3D line `(O(u,v), d(u,v))` for every pixel.

```python
fit = sc.fit_stereo_zernike_origin_field_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    max_order=2,
    method2d="rayfield_tps_robust",
)

print(f"Zernike BA RMS: {fit.residual_rms:.3f} mm")
left_field = fit.left_field   # ZernikeRayField
```

Read geometric descriptors directly from the fitted rayfield:

```python
O_L = left_field.origin(1024, 1024)   # 3D origin at centre pixel
O_R = fit.right_field.origin(1024, 1024)
d_L = left_field.direction(1024, 1024)
d_R = fit.right_field.direction(1024, 1024)

import numpy as np
baseline = np.linalg.norm(O_R - O_L)
convergence = np.degrees(np.arccos(np.clip(np.dot(d_L, d_R), -1, 1)))
print(f"Baseline: {baseline:.1f} mm, Convergence: {convergence:.1f}°")
```

**Key take-away:** the Zernike rayfield is a diagnostic instrument.  You read
physical descriptors directly from it without fitting any optical model.

> 📓 **Notebook:** [05_noncentral_calibration_from_images.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/05_noncentral_calibration_from_images.ipynb)
> and [04_parallel_plate_origin_field.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/04_parallel_plate_origin_field.ipynb) demonstrate
> Zernike fitting from image directories and on synthetic oracles.

## 5.  Identify the physical optics (2 minutes)

Once you have a measured Zernike rayfield, ask which physical model best
compresses it:

```python
report = sc.select_physical_model_from_rayfield(
    target_field=left_field,
    K=left_field.K,
    image_size=left_field.config.image_size,
)

print(report.best_by_bic)   # e.g. "central_brown_conrady"
for row in report.rows():
    print(row)              # model, n_params, rms_mm, bic
```

The report compares pinhole, Brown‑Conrady, parallel‑plate, and telecentric CMO
candidates in ray‑space using the Bayesian Information Criterion.

**Key take-away:** model selection happens in ray‑space, without re‑projecting
corners.  The BIC identifies the correct optical family.

> 📓 **Notebook:** [06_cmo_model_selection.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/06_cmo_model_selection.ipynb) and
> [07_model_selection_matrix.ipynb](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/07_model_selection_matrix.ipynb) demonstrate model
> selection on CMO-like and multi-oracle data.

## 6.  Build a compact CMO model (with real data)

The complete non‑central pipeline on real Pycaso CMO data is demonstrated in
`examples/notebooks/09_pycaso_real_data.py`.  The final 26‑parameter
CMO+SE(3) model reaches **1.06 px** reprojection on a microscope where OpenCV
fails (>300 px).

```bash
cd examples/notebooks
python3 09_pycaso_real_data.py
```

This notebook walks through:

1. ChArUco detection + double TPS denoising
2. Zernike rayfield BA (57 parameters, 0.47 px)
3. Residual analysis → discovery of telecentricity
4. Per‑channel SE(3) arm alignment → 26p model at 1.06 px
5. BIC model selection + operational usability score

> 📓 **Full walkthrough:** [09_pycaso_real_data.py](https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/main/examples/notebooks/09_pycaso_real_data.py) (1676 lines).
> The compiled CMO paper is at [paper/cmo/build/manuscript.pdf](../paper/cmo/build/manuscript.pdf).

## Where to go next

- **Fix your calibration:** :doc:`FIX_MY_CALIBRATION`
- **Bring your own images:** :doc:`BRING_YOUR_OWN_DATA`
- **Understand the optics:** :doc:`CMO_PHYSICAL_MODEL`
- **API reference:** :doc:`API`
- **All notebooks:** :doc:`NOTEBOOKS`
