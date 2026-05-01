# %% [markdown]
# # 05 - Non-central stereo calibration from images
#
# **Goal**: go from two directories of raw calibration images to a fitted
# Zernike origin-field model in one function call, then use that model to
# reconstruct 3D points and compare against a standard pinhole model.
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
#
# Everything below is runnable as a plain `.py` script or as a Jupyter notebook
# (cells are marked with `# %%`).

# %% [markdown]
# ## 0. Render synthetic calibration images
#
# We render synthetic images from the parallel-plate oracle so the notebook is
# self-contained.  Replace `left_dir` / `right_dir` with your own image
# directories to run on real data — only the board description needs to match.

# %%
import tempfile
from pathlib import Path

import numpy as np

import stereocomplex as sc

tmp = Path(tempfile.mkdtemp())
board = sc.make_default_parallel_plate_charuco_board()
dataset_gt = sc.make_default_parallel_plate_charuco_dataset()
rendered = sc.render_parallel_plate_charuco_images(dataset_gt, board, tmp / "calib")

left_dir = rendered.left_images[0].parent
right_dir = rendered.right_images[0].parent
print(f"left  images : {left_dir}  ({len(rendered.left_images)} files)")
print(f"right images : {right_dir}  ({len(rendered.right_images)} files)")

# %% [markdown]
# ## 1. Fit the non-central model
#
# One function call: detect corners, calibrate K, fit the Zernike origin field.

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
# ## 2. Inspect the fitted origin field
#
# `left_field` and `right_field` are `ZernikeOriginField` objects.
# For any pixel `(u, v)` they return the physical ray `(O, d)`.
# A non-zero `O` is what distinguishes a non-central camera from a pinhole.

# %%
left_field = fit.left_field
right_field = fit.right_field
T_right_left = fit.stereo_transform

W, H = left_field.config.image_size
u_c, v_c = W / 2.0, H / 2.0
O_L, d_L = left_field.ray(u_c, v_c)
print(f"Left ray at image centre:")
print(f"  origin    O = {O_L}")
print(f"  direction d = {d_L}")
print(f"  |O|         = {np.linalg.norm(O_L):.4f} mm  (non-zero → non-central)")

# %% [markdown]
# ## 3. Reconstruct 3D points
#
# Detect corners in the rendered images, then triangulate using the fitted
# non-central model.  In production, pass your own detected correspondences.

# %%
detected = sc.detected_observations_from_rendered_parallel_plate(rendered)
uv_L = np.concatenate(detected.left_pixels)
uv_R = np.concatenate(detected.right_pixels)

result = sc.reconstruct_points_with_origin_fields(
    uv_L, uv_R, left_field, right_field, T_right_left
)

valid = result.valid_mask
print(f"reconstructed {valid.sum()} / {len(valid)} points")
print(f"ray-gap RMS   : {np.sqrt(np.mean(result.ray_gap[valid] ** 2)):.4f} mm")
print(f"first point   : {result.points_3d[valid][0]}")

# %% [markdown]
# ## 4. Quality summary
#
# The residual RMS from the fit is the primary quality indicator: it measures
# the average point-to-ray distance across all observations.  A low residual
# means the fitted origin field explains the detected correspondences well.
#
# The ray gap from triangulation is an independent measure: rays from the two
# cameras that come closest to each other should nearly intersect; a large gap
# indicates inconsistent calibration.

# %%
print(f"Fit quality:")
print(f"  residual RMS (BA)   : {fit.residual_rms:.4f} mm")
print(f"  residual median     : {fit.residual_median:.4f} mm")
print(f"  residual p95        : {fit.residual_p95:.4f} mm")
print()
print(f"Triangulation quality on detected corners:")
valid = result.valid_mask
print(f"  ray-gap RMS         : {np.sqrt(np.mean(result.ray_gap[valid] ** 2)):.4f} mm")
print(f"  ray-gap median      : {float(np.median(result.ray_gap[valid])):.4f} mm")
print()
print(f"Origin-field peak displacement (left camera):")
us = np.linspace(0, W - 1, 9)
vs = np.linspace(0, H - 1, 7)
UU, VV = np.meshgrid(us, vs)
O_grid, _ = left_field.ray(UU.ravel(), VV.ravel())
norms = np.linalg.norm(O_grid, axis=-1)
print(f"  mean |O(u,v)|       : {norms.mean():.4f} mm")
print(f"  max  |O(u,v)|       : {norms.max():.4f} mm")
