# Virtual rectification for dense stereo

This page shows how to use the **ray-field virtual rectification** to obtain scanline-aligned stereo
pairs (horizontal epipolars) so you can run a **standard dense pipeline** (BM/SGM/Census) exactly
as in a pinhole setup.

The key idea is to build two dense warps (left/right) that map a **virtual rectified pinhole**
grid back to the source images using the calibrated **ray-field** and the stereo rig `R,t`.
Downstream code (rectification → disparity → depth) remains unchanged; only the remap is different.

## Drop-in workflow (pinhole-compatible)

1. Calibrate a ray-field stereo rig (or any backend exposing `pixel -> direction`).
2. Build rectification maps with :mod:`stereocomplex.ray3d.rayfield_rectify`:

   ```
   from stereocomplex.ray3d.rayfield_rectify import RectifyParams, build_virtual_rectify_maps, rectify_pair
   mapx_L, mapy_L, mapx_R, mapy_R, R_rect = build_virtual_rectify_maps(
       ray_L, ray_R, R_lr, t_lr, RectifyParams(width=W, height=H)
   )
   I_L_rect, I_R_rect = rectify_pair(I_L, I_R, mapx_L, mapy_L, mapx_R, mapy_R)
   ```

3. Run any 1D dense matcher (BM/SGM/Census) on the rectified pair.
4. Convert disparity to depth either with a virtual pinhole `Q` or with ray-intersection
   (the latter is more consistent with the ray-field).

The virtual camera intrinsics are chosen to maximize valid coverage; by default
`fx'=fy'=0.9*W'` and `cx'=W'/2`, `cy'=H'/2`. The rectified axes are built from the
baseline direction, so the epipolars become horizontal.

## Implementation overview (maps for `cv2.remap`)

The rectification is implemented as a **dense warp** toward a **virtual rectified pinhole camera**.
For each pixel `(u', v')` in the rectified images:

1. **Virtual ray**: build a unit direction `d_rect = normalize([(u'-cx')/fx', (v'-cy')/fy', 1])`.
2. **Physical rays**: rotate `d_rect` into the left and right camera frames using the rectified axes
   derived from the baseline.
3. **Inverse mapping (direction → pixel)**: since the calibrated model is **forward** (`pixel → direction`),
   each target direction must be inverted back to a source pixel `(u,v)`:
   - coarse init via a **quantized inverse LUT** (`lut_use/lut_quant`),
   - fallback init via a coarse image grid (`coarse_step`),
   - refinement via a few Gauss–Newton iterations minimizing the angular error between
     `dir(u,v)` and the target direction.
4. **Fill maps**: write `mapx/mapy` for left and right; invalid pixels (direction outside the model FOV)
   are marked as `-1` and filled by `cv2.remap` border policy.

The LUT and Newton refinement are internal details of `build_virtual_rectify_maps`, but they are the key
reason virtual rectification is practical: the dense maps are computed once per model, cached, then reused.

## End-to-end example

The repository now ships a runnable demo that:

1. loads an exported ray-field stereo model,
2. builds virtual rectification maps,
3. rectifies one stereo pair,
4. runs a standard dense matcher on the rectified pair,
5. and reports the residual ChArUco vertical disparity before/after rectification.

If you already have an exported model (`model.json` + `weights.npz`), the demo is:

```bash
PYTHONPATH=src .venv/bin/python docs/examples/rayfield_virtual_rectification_demo.py \
  dataset/v0_png --split train --scene scene_0000 --frame-id 0 \
  --model models/scene0000_rayfield3d \
  --out docs/assets/rayfield_virtual_rectify_demo
```

If you do not have an exported model yet, the same demo can calibrate a small one first and
store it under `--out/model`:

```bash
PYTHONPATH=src .venv/bin/python docs/examples/rayfield_virtual_rectification_demo.py \
  dataset/v0_png --split train --scene scene_0000 --frame-id 0 \
  --out docs/assets/rayfield_virtual_rectify_demo
```

The output directory contains:

- `left_raw.png`, `right_raw.png`
- `left_rectified.png`, `right_rectified.png`
- `disparity.png` and `disparity.npy`
- `rectify_maps.npz`
- `summary.json` with rectification validity and residual vertical disparity statistics

For your own code, the core usage pattern is:

```python
from pathlib import Path

from stereocomplex.api import load_stereo_central_rayfield
from stereocomplex.ray3d.rayfield_rectify import RectifyParams, build_virtual_rectify_maps, rectify_pair

model = load_stereo_central_rayfield(Path("models/scene0000_rayfield3d"))
ray_L = ...  # any object exposing dir(u, v), width, height
ray_R = ...
params = RectifyParams(width=model.image_width_px, height=model.image_height_px)
mapx_L, mapy_L, mapx_R, mapy_R, _ = build_virtual_rectify_maps(ray_L, ray_R, model.R_RL, model.t_RL, params)
I_L_rect, I_R_rect = rectify_pair((I_L, I_R), (mapx_L, mapy_L, mapx_R, mapy_R), params)
```

The shipped demo script wraps `StereoCentralRayFieldModel.left/right` into that tiny adapter
automatically; you only need to write the adapter yourself if you integrate the rectifier into
custom code.

For a guided, cell-by-cell walkthrough, see the companion notebook:
`examples/notebooks/03_rayfield_virtual_rectification.ipynb` (with the linear export
`examples/notebooks/03_rayfield_virtual_rectification.py`).

Once you have `I_L_rect` and `I_R_rect`, you can drop them into any standard scanline matcher
such as `cv2.StereoSGBM_create(...)`, then convert disparity to depth as usual.

## Visual sanity check (synthetic pinhole)

Below is a synthetic check on a pinhole rig (baseline 10 cm, random 3D points, 640×480):
the virtual rectification collapses vertical disparity to ~0 px, making the pair compatible
with standard 1D matching.

.. figure:: assets/virtual_rectify_hist.png
   :alt: Histogram of vertical disparity before/after virtual rectification (synthetic pinhole)
   :width: 80%

   Histogram of vertical disparity (v_L - v_R, pixels) before/after virtual rectification on a synthetic pinhole rig.

## Notes and limits

- The inversion `direction -> pixel` is solved by a small Newton loop with finite-difference
  Jacobians; maps are meant to be **precomputed and cached** once per model. A coarse inverse LUT
  (quantized directions) is used as an initial guess before Newton, with a fallback on a coarse
  image grid if the LUT bin is empty.
- If a direction is outside the FOV of the ray-field, the corresponding rectified pixel is
  marked invalid (cv2.remap fills it with the border value).
- For depth recovery, you can either use a virtual `Q` (pinhole-like) or intersect the two
  rays associated with the rectified disparity (more exact for ray-field).
