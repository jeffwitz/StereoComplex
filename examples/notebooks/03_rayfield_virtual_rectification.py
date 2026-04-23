"""Exported companion script for 03_rayfield_virtual_rectification.ipynb.

This file mirrors the notebook content in a linear, readable Python form.
"""

# %% [markdown]
# # Ray-field virtual rectification
#
# This notebook shows the bridge from a ray-field backend to a **standard dense stereo pipeline**.
# The goal is simple:
#
# - start from a calibrated central ray-field model,
# - build a **virtual rectified pinhole camera**,
# - produce dense `mapx/mapy` arrays for `cv2.remap`,
# - and run a classic 1D matcher such as `StereoSGBM` on the rectified pair.
#
# The key implementation detail is that the ray-field is forward in image space:
#
# $$
# (u, v) \mapsto d(u, v)
# $$
#
# Rectification therefore needs an inverse mapping from a desired ray direction back to a source pixel.
# StereoComplex handles that inverse with a coarse LUT plus a short Gauss-Newton refinement.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np

def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    while True:
        if (cur / 'pyproject.toml').exists():
            return cur
        if cur == cur.parent:
            raise RuntimeError('Could not locate repository root from notebook')
        cur = cur.parent

ROOT = find_repo_root()
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(ROOT / 'docs' / 'examples') not in sys.path:
    sys.path.insert(0, str(ROOT / 'docs' / 'examples'))

import rayfield_virtual_rectification_demo as demo
from stereocomplex.api import load_stereo_central_rayfield
from stereocomplex.ray3d.rayfield_rectify import RectifyParams, build_virtual_rectify_maps, rectify_pair

plt.rcParams.update({'figure.dpi': 120, 'image.cmap': 'gray'})
print('Repo root:', ROOT)

# %% [markdown]
# ## 1. Load one synthetic scene and obtain a small reusable ray-field model
#
# If no exported model is already present, this notebook will fit a small one automatically
# from the same synthetic scene. The resulting `model.json` + `weights.npz` pair can be reused elsewhere.

# %%
dataset_root = ROOT / 'dataset' / 'v0_png'
scene_dir = dataset_root / 'train' / 'scene_0000'
frame_id = 0

out_dir = ROOT / 'docs' / 'assets' / 'rayfield_virtual_rectify_demo'
out_dir.mkdir(parents=True, exist_ok=True)

model_dir = ROOT / 'models' / 'scene0000_rayfield3d'
if not (model_dir / 'model.json').exists():
    auto = SimpleNamespace(
        dataset_root=dataset_root,
        split='train',
        scene='scene_0000',
        frame_id=frame_id,
        out=out_dir,
        model=None,
        export_model=model_dir,
        max_frames=5,
        method2d='rayfield_tps_robust',
        max_points_per_frame=0,
        nmax=10,
        lam_coeff=1e-3,
        outer_iters=3,
        tps_lam=10.0,
        tps_huber=1.0,
        tps_iters=3,
        rect_fx=None,
        rect_fy=None,
        rect_cx=None,
        rect_cy=None,
        sgbm_num_disparities=128,
        sgbm_block_size=5,
    )
    print('No exported model found, calibrating a small one first...')
    model_dir = demo.ensure_model(auto, out_dir)

model = load_stereo_central_rayfield(model_dir)
meta = demo.load_json(scene_dir / 'meta.json')
imgL, imgR, frame = demo.load_scene_images(scene_dir, frame_id)
dictionary, board, detector_params, aruco_detector, charuco_detector = demo.build_charuco_from_meta(meta)

raw_L = demo.detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, imgL)
raw_R = demo.detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, imgR)

print('Scene:', frame)
print('Model dir:', model_dir)
print('Raw ChArUco detections:', len(raw_L or {}), len(raw_R or {}))
print('Raw vertical disparity stats:', json.dumps(demo.vertical_disparity_stats(raw_L, raw_R), indent=2))

# %% [markdown]
# ## 2. Build the virtual rectification maps
#
# The rectifier takes a target rectified pinhole camera and produces two dense remap tables:
#
# - `mapx_L`, `mapy_L`
# - `mapx_R`, `mapy_R`
#
# These maps can be fed directly to `cv2.remap`.

# %%
rect_params = RectifyParams(width=model.image_width_px, height=model.image_height_px)
rayL = demo.RayModelAdapter(model.left, model.image_width_px, model.image_height_px)
rayR = demo.RayModelAdapter(model.right, model.image_width_px, model.image_height_px)

mapx_L, mapy_L, mapx_R, mapy_R, R_rect = build_virtual_rectify_maps(
    rayL,
    rayR,
    model.R_RL,
    model.t_RL,
    rect_params,
)
I_L_rect, I_R_rect = rectify_pair((imgL, imgR), (mapx_L, mapy_L, mapx_R, mapy_R), rect_params)

valid_L = np.mean((mapx_L >= 0) & (mapx_L < model.image_width_px) & (mapy_L >= 0) & (mapy_L < model.image_height_px))
valid_R = np.mean((mapx_R >= 0) & (mapx_R < model.image_width_px) & (mapy_R >= 0) & (mapy_R < model.image_height_px))
print('Valid map coverage left/right:', float(valid_L), float(valid_R))

# %% [markdown]
# ## 3. Look at the raw pair versus the rectified pair
#
# The top row shows the raw images. The bottom row shows the rectified pair.
# After rectification, the same physical points should sit on nearly the same scanline.

# %%
def draw_detection(ax, img, det, title):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    if det:
        pts = np.asarray(list(det.values()), dtype=np.float64)
        ax.scatter(pts[:, 0], pts[:, 1], s=18, facecolors='none', edgecolors='lime', linewidths=1.2)
    ax.set_title(title)
    ax.set_axis_off()

rect_L_det = demo.detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, I_L_rect)
rect_R_det = demo.detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, I_R_rect)

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
draw_detection(axs[0, 0], imgL, raw_L, 'Raw left')
draw_detection(axs[0, 1], imgR, raw_R, 'Raw right')
draw_detection(axs[1, 0], I_L_rect, rect_L_det, 'Rectified left')
draw_detection(axs[1, 1], I_R_rect, rect_R_det, 'Rectified right')
fig.suptitle('Ray-field virtual rectification', fontsize=14)
fig.tight_layout()
plt.show()

rect_stats = demo.vertical_disparity_stats(rect_L_det, rect_R_det)
print('Rectified vertical disparity stats:', json.dumps(rect_stats, indent=2))

# %% [markdown]
# ## 4. Run a classic dense matcher on the rectified pair
#
# Once the scanlines are horizontal, a standard 1D matcher works as in the pinhole case.
# Here we use `StereoSGBM`, but the same rectified pair can be passed to BM, Census, or any other
# block-matching method.

# %%
block_size = 5
num_disp = 128
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size * block_size,
    P2=32 * block_size * block_size,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=1,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
disp = sgbm.compute(I_L_rect, I_R_rect).astype(np.float32) / 16.0
np.savez_compressed(out_dir / 'virtual_rectification_demo_outputs.npz', mapx_L=mapx_L, mapy_L=mapy_L, mapx_R=mapx_R, mapy_R=mapy_R, disp=disp)

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
valid = disp > 0
if np.any(valid):
    lo = float(np.quantile(disp[valid], 0.02))
    hi = float(np.quantile(disp[valid], 0.98))
else:
    lo, hi = 0.0, 1.0
im = ax.imshow(disp, cmap='turbo', vmin=lo, vmax=hi)
ax.set_title('SGBM disparity on rectified images')
ax.set_axis_off()
fig.colorbar(im, ax=ax, shrink=0.8, label='disparity (px)')
fig.tight_layout()
plt.show()

print('Disparity valid fraction:', float(np.mean(valid)))

# %% [markdown]
# ## 5. What to do in your own code
#
# The core pattern is always the same:
#
# 1. load the exported model,
# 2. build rectification maps once,
# 3. cache the maps,
# 4. call `cv2.remap`,
# 5. run your favorite dense matcher.
#
# If you want to integrate it into an existing project, the low-level API is the same as in the demo script:
# `build_virtual_rectify_maps(...)` and `rectify_pair(...)`.

# %%
print(json.dumps({
    'model_dir': str(model_dir),
    'demo_output_dir': str(out_dir),
    'rectification': {
        'raw_vertical_disparity_px': demo.vertical_disparity_stats(raw_L, raw_R),
        'rectified_vertical_disparity_px': rect_stats,
    },
    'dense_matcher': {
        'algorithm': 'StereoSGBM',
        'num_disparities': num_disp,
        'block_size': block_size,
    },
}, indent=2, sort_keys=True))
