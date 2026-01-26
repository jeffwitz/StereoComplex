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
  Jacobians; maps are meant to be **precomputed and cached** once per model.
- If a direction is outside the FOV of the ray-field, the corresponding rectified pixel is
  marked invalid (cv2.remap fills it with the border value).
- For depth recovery, you can either use a virtual `Q` (pinhole-like) or intersect the two
  rays associated with the rectified disparity (more exact for ray-field).

