# Reconstruction API (load model, triangulate points, optional image maps)

This page documents a small, **usage-oriented** API to reconstruct 3D points from stereo correspondences, and a **file format** that is compatible with future ML usage (small JSON + NPZ weights).

The goal is to make it easy to:

- export a calibrated stereo model to disk,
- load it back in Python with a single import,
- triangulate points (and optionally precompute per-pixel ray maps).

## File format: `model.json` + `weights.npz`

Stereo models are stored in a directory:

```
models/<name>/
  model.json
  weights.npz
```

- `model.json`: small metadata (schema, image size, disk mapping, rig parameters, NPZ keys).
- `weights.npz`: NumPy arrays (Zernike coefficients + stereo rig).

This structure is “ML friendly”: a training loop can treat `weights.npz` as learnable parameters while keeping the rest in JSON.

## API: load + triangulate

The following code is a minimal, commented example using the public API:

```{literalinclude} examples/reconstruction_api_demo.py
:language: python
:start-after: from __future__ import annotations
:end-before: def build_charuco_from_meta
```

Optionally, precompute ray directions over the full image grid (useful for real-time pipelines):

```python
dL_map, dR_map = model.ray_direction_maps()  # (H,W,3) float32
```

## End-to-end demo on a dataset scene

This demo calibrates a **central 3D ray-field** from a subset of frames, exports the model, then evaluates it on the scene (triangulation error against GT).

### Prerequisites

- `opencv-contrib-python` (required for `cv2.aruco`)

### 1) Calibrate + export a reusable model

```bash
.venv/bin/python paper/experiments/calibrate_central_rayfield3d_from_images.py dataset/v0_png \
  --split train --scene scene_0000 \
  --max-frames 5 \
  --method2d rayfield_tps_robust \
  --nmax 10 --lam-coeff 1e-3 --outer-iters 3 \
  --out paper/tables/rayfield3d_ba_scene0000.json \
  --export-model models/scene0000_rayfield3d
```

### 2) Apply the model (reconstruction on detected corners)

```bash
.venv/bin/python docs/examples/reconstruction_api_demo.py dataset/v0_png \
  --split train --scene scene_0000 --max-frames 5 \
  --model models/scene0000_rayfield3d
```

Notes:

- This evaluation uses OpenCV ChArUco detections to obtain correspondences in each frame.
- GT comparison requires synthetic data with `gt_charuco_corners.npz`.
- If you only want to export refined 2D corners for OpenCV calibration (without ray-field 3D), see `stereocomplex refine-corners` in `docs/START_HERE.md`.

## Code references

- API classes + triangulation: `src/stereocomplex/api/stereo_reconstruction.py`
- Save/load model: `src/stereocomplex/api/model_io.py`
- Export model from calibration script: `paper/experiments/calibrate_central_rayfield3d_from_images.py`
- Evaluate an exported model on a scene (API demo): `docs/examples/reconstruction_api_demo.py`
- Evaluate an exported model on a scene (JSON report): `paper/experiments/eval_exported_stereo_model.py`
