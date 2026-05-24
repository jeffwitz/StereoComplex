# %% [markdown]
# # 05 - Non-central stereo calibration from images
#
# **Goal**: go from two directories of raw calibration images to a fitted
# Zernike origin-field model, show that it outperforms a standard pinhole model
# on a non-central stereo system, then explain how to read the quality metrics.
#
# The public entry-point is:
#
# ```python
# stereocomplex.fit_stereo_zernike_origin_field_from_image_dirs(
#     left_dir=..., right_dir=..., board=...,
# )
# ```
#
# It runs internally in three stages:
#
# 1. ChArUco corner detection on every image pair.
# 2. OpenCV mono + stereo calibration → K_left, K_right, T_right_left.
# 3. Zernike bundle adjustment → per-camera origin field O(u,v).

# %% [markdown]
# ## 0. Calibration images
#
# We render synthetic calibration images from the parallel-plate oracle so the
# notebook is self-contained.
#
# **To use your own images**, replace the three lines marked `← replace` below:
#
# ```python
# left_dir = Path("/path/to/left")   # ← replace
# right_dir = Path("/path/to/right") # ← replace
# board = sc.CharucoBoardSpec(        # ← replace with your board
#     squares_x=9, squares_y=6,
#     square_size_mm=20.0,
#     marker_size_mm=15.0,
#     aruco_dictionary="DICT_4X4_50",
# )
# ```
#
# The board description (`squares_x/y`, `square_size_mm`, `marker_size_mm`,
# `aruco_dictionary`) must exactly match the physical board you printed.
# Image filenames inside each directory are sorted lexicographically; left and
# right directories must contain the same number of files in matching order.

# %%
import tempfile
from pathlib import Path

import numpy as np

import stereocomplex as sc
from stereocomplex.advanced import (
    reconstruct_points_central_stereo,
    reconstruct_points_with_origin_fields,
)
from stereocomplex.benchmarks.parallel_plate_origin_field import (
    make_default_parallel_plate_charuco_board,
    make_default_parallel_plate_charuco_dataset,
)

tmp = Path(tempfile.mkdtemp())
board = make_default_parallel_plate_charuco_board()
dataset_gt = make_default_parallel_plate_charuco_dataset()
rendered = sc.render_parallel_plate_charuco_images(dataset_gt, board, tmp / "calib")

left_dir = rendered.left_images[0].parent
right_dir = rendered.right_images[0].parent
print(f"left  images : {left_dir}  ({len(rendered.left_images)} files)")
print(f"right images : {right_dir}  ({len(rendered.right_images)} files)")

# %% [markdown]
# ## 1. Fit the non-central model
#
# One function call: detect corners, calibrate K, fit the Zernike origin field.
#
# `max_order=4` means Zernike polynomials up to radial order 4 (15 modes per
# camera per component).  Higher orders give more flexibility but need more
# frames with diverse poses to be identifiable.  `max_order=4` is a safe
# default for a first run; try 6 or 8 once the basic fit converges well.

# %%
fit = sc.fit_stereo_zernike_origin_field_from_image_dirs(
    left_dir=left_dir,
    right_dir=right_dir,
    board=board,
    max_order=4,
)

print(f"success        : {fit.success}")
print(f"message        : {fit.message}")
print(f"residual RMS   : {fit.residual_rms:.4f} mm")
print(f"n observations : {fit.n_observations}")

# %% [markdown]
# ## 2. Non-central vs pinhole comparison
#
# This is the key result: the non-central origin field should produce smaller
# ray gaps than the pinhole model when the stereo system is genuinely
# non-central (e.g., caused by glass plates, diopters, or a thick lens).
#
# We use the same K matrices and stereo transform for both models so the only
# variable is the origin field.  The **ray gap** is the 3D distance between the
# two closest points on the left and right rays for each correspondence —
# a perfectly central system with perfect data would give gap = 0.
#
# With 10 rendered frames and realistic detection noise, the improvement is
# visible but modest: detection noise limits how well the Zernike coefficients
# can be identified.  With ≥ 20 frames, better pose diversity, or cleaner
# detections, the improvement grows larger.  See notebook `04` for the
# controlled benchmark on noise-free synthetic observations, which shows > 2×.

# %%
detected = sc.detected_observations_from_rendered_parallel_plate(rendered)
uv_L = np.concatenate(detected.left_pixels)
uv_R = np.concatenate(detected.right_pixels)

K_left = fit.left_field.K
K_right = fit.right_field.K
T_RL = fit.stereo_transform

result_pinhole = reconstruct_points_central_stereo(uv_L, uv_R, K_left, K_right, T_RL)
result_nc = reconstruct_points_with_origin_fields(uv_L, uv_R, fit.left_field, fit.right_field, T_RL)

v_ph = result_pinhole.valid_mask
v_nc = result_nc.valid_mask

gap_ph = float(np.sqrt(np.mean(result_pinhole.ray_gap[v_ph] ** 2)))
gap_nc = float(np.sqrt(np.mean(result_nc.ray_gap[v_nc] ** 2)))

print(f"Ray-gap RMS — pinhole    : {gap_ph:.4f} mm")
print(f"Ray-gap RMS — non-central: {gap_nc:.4f} mm")
print(f"Improvement factor       : {gap_ph / gap_nc:.1f}×")

# %% [markdown]
# ## 3. Inspect the fitted origin field
#
# `left_field` and `right_field` are `ZernikeOriginField` objects.
# For any pixel `(u, v)` they return the physical ray `(O, d)`.
# A non-zero `O` is what distinguishes a non-central camera from a pinhole.
# For a pinhole camera, `O` would be exactly zero everywhere by construction.

# %%
left_field = fit.left_field
right_field = fit.right_field
W, H = left_field.config.image_size

u_c, v_c = W / 2.0, H / 2.0
O_L, d_L = left_field.ray(u_c, v_c)
print(f"Left ray at image centre:")
print(f"  origin    O = {O_L}")
print(f"  direction d = {d_L}")
print(f"  |O|         = {np.linalg.norm(O_L):.4f} mm  (non-zero → non-central)")

# %% [markdown]
# ## 4. Reconstruct 3D points
#
# Use the fitted model to triangulate detected correspondences into 3D points
# in the left-camera coordinate frame.

# %%
valid = result_nc.valid_mask
print(f"reconstructed {valid.sum()} / {len(valid)} points")
print(f"ray-gap RMS   : {float(np.sqrt(np.mean(result_nc.ray_gap[valid] ** 2))):.4f} mm")
print(f"first point   : {result_nc.points_3d[valid][0]}")

# %% [markdown]
# ## 5. Reading the quality metrics
#
# After every calibration run, check these three quantities in order:
#
# | Metric | Typical range | Flag if… |
# |--------|---------------|----------|
# | `residual_rms` (BA) | < 1 mm on synthetics; 1–5 mm on real data | > 10 mm: bad init, too few frames, or detection failure |
# | ray-gap RMS | similar to `residual_rms` | >> `residual_rms`: K and ray-field are inconsistent |
# | mean `\|O(u,v)\|` | 0–50 mm | < 0.1 mm: system is essentially pinhole (non-central overkill); > 100 mm: likely a numerical issue |
#
# A large discrepancy between `residual_rms` and ray-gap usually points to a
# coordinate-frame mismatch, which can happen when K from OpenCV and K used
# inside the Zernike field have drifted during the BA.

# %%
print(f"BA residual:    RMS {fit.residual_rms:.4f}  median {fit.residual_median:.4f}  p95 {fit.residual_p95:.4f} mm")
print(f"ray-gap (nc):   RMS {gap_nc:.4f} mm")

us = np.linspace(0, W - 1, 9)
vs = np.linspace(0, H - 1, 7)
UU, VV = np.meshgrid(us, vs)
O_grid, _ = left_field.ray(UU.ravel(), VV.ravel())
norms = np.linalg.norm(O_grid, axis=-1)
print(f"origin field:   mean |O| {norms.mean():.4f}  max |O| {norms.max():.4f} mm")

# %% [markdown]
# ## 6. Troubleshooting
#
# **`RuntimeError: no usable stereo frames after ChArUco detection`**
# — The board spec does not match the images (wrong square size, wrong
# dictionary) or the images are blurry / too dark.  Check that `aruco_dictionary`
# matches the markers printed on your board.
#
# **`RuntimeError: not enough frames for mono calibration (need ≥ 2 per side)`**
# — Fewer than 2 frames per camera passed detection.  Same root cause as above.
#
# **`RuntimeError: only N corners visible in every frame`**
# — Detection succeeded per frame, but the intersection of visible corners
# across ALL frames is too small.  The current implementation requires every
# calibration corner to be visible in every frame; frames with partial board
# coverage are automatically restricted to the global intersection.
# Fix: add frames that cover the full board, or reduce `min_common_corners`.
#
# **`fit.success` is `False` / `residual_rms` is large**
# — The Zernike BA did not converge.  Common causes:
# (1) too few frames (use ≥ 8 for `max_order=4`);
# (2) poor pose diversity (move the board to different angles and distances);
# (3) `regularization` too small (try `1e-3` instead of the default `1e-6`);
# (4) `max_nfev` too low (increase to 500).
#
# **`solvePnP failed for all frames`**
# — Pose estimation failed, usually because `min_common_corners` corners are
# detected but the board is too close to the image boundary or nearly
# fronto-parallel.  Add more diverse frames.
