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
