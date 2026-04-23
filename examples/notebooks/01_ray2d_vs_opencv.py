"""Exported companion script for 01_ray2d_vs_opencv.ipynb.

This file mirrors the notebook content in a linear, readable Python form.
"""

# %% [markdown]
# # Ray2D vs OpenCV pinhole on synthetic data
#
# This notebook is the best place to understand the **2D** contribution of StereoComplex before
# looking at the 3D ray-field backend. The setting is intentionally narrow:
#
# - start from one synthetic stereo scene with ground truth,
# - compare the raw OpenCV ChArUco detection against the `rayfield_tps_robust` second pass,
# - then move from a local visual example to the **global released sweep**.
#
# The central idea is that calibration quality is often limited upstream by the image measurements
# themselves. If the detected corners are biased by blur, compression, or local distortions, the
# camera model is asked to explain the wrong geometry.
#
# Ray2D does **not** change the downstream pinhole optimizer. It only regularizes the 2D points on
# the calibration plane:
#
# $$
# \mathbf{p}_{\mathrm{refined}}
# =
# \mathbf{p}_{\mathrm{raw}}
# +
# \Delta \mathbf{p}_{\mathrm{ray2D}}(\mathbf{p}_{\mathrm{raw}})
# $$
#
# The practical consequence is measured downstream with stereo quantities, not only with a mono
# reprojection RMS. For a rectified stereo pair, a useful diagnostic is the residual vertical
# disparity
#
# $$
# |\Delta y| = |y_L^{\mathrm{rect}} - y_R^{\mathrm{rect}}|,
# $$
#
# which should stay close to zero if the estimated geometry is coherent.
#
# Reading guide:
#
# - Section 0 explains where the committed example data come from.
# - Sections 1 and 1b show **what physically moves** in the corner detections.
# - Sections 2 and 3 answer the real question: does that local correction improve the stereo
#   pipeline consistently over the whole benchmark?
#
# Everything loaded here is already versioned in the repository: images, overlays, and JSON
# summaries. The notebook is therefore a teaching front-end, not a long-running experiment.

# %%
from pathlib import Path
import json
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

from stereocomplex.api.corner_refinement import refine_charuco_corners
from stereocomplex.core.image_io import load_gray_u8
from stereocomplex.eval.charuco_detection import _make_charuco_detector

plt.style.use('seaborn-v0_8-whitegrid')

def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for p in [start, *start.parents]:
        if (p / 'pyproject.toml').exists() and (p / 'README.md').exists():
            return p
    raise RuntimeError('Repository root not found')

ROOT = find_repo_root()
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def load_json(rel_path):
    return json.loads((ROOT / rel_path).read_text(encoding='utf-8'))

def show_image_grid(items, ncols=2, figsize=(14, 8), cmap=None):
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (title, image) in zip(axes, items):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    return fig

print('ROOT =', ROOT)

# %% [markdown]
# ## 0. Where the example comes from
#
# The notebook reads the synthetic compression benchmark already committed to the repository.
# This is important for two reasons:
#
# - the images are realistic enough to show the visual effect of the refinement,
# - the ground truth corners are known exactly, so we can separate *measurement quality* from
#   *camera-model fitting*.
#
# In other words, this notebook is not a toy illustration with invented points. It is a slice
# through the same dataset family used in the paper and in the CLI experiments.
#
# The benchmark scene provides:
#
# - the left/right images,
# - the raw OpenCV detections,
# - the board metadata,
# - and the GT corner positions used later for zoomed overlays.
#
# That lets us explain the method in the right order: first the 2D measurements, then the stereo
# consequences.

# %%
scene_dir = ROOT / 'dataset' / 'compression_sweep_pnp' / 'png_lossless' / 'train' / 'scene_0000'
meta = load_json(scene_dir / 'meta.json')
frames = [json.loads(line) for line in (scene_dir / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
gt = np.load(scene_dir / 'gt_charuco_corners.npz')

def print_synthetic_setup(meta):
    board = meta.get('board', {})
    sim = meta.get('sim_params', {})
    stereo = meta.get('stereo', {})
    left = stereo.get('left', {})
    sensor = left.get('sensor', {})
    image = left.get('image', {})
    f_um = sim.get('f_um')
    pixel_pitch_um = sensor.get('pixel_pitch_um')
    fx_px = f_um / pixel_pitch_um if f_um and pixel_pitch_um else None
    print('\nSynthetic setup summary')
    print('  board:')
    print(f"    type={board.get('type')}  dictionary={board.get('aruco_dictionary')}")
    print(f"    grid={board.get('squares_x')}x{board.get('squares_y')}  square={board.get('square_size_mm'):.2f} mm  marker={board.get('marker_size_mm'):.2f} mm")
    print('  intrinsics / optics:')
    print(f"    camera_model={sim.get('camera_model')}  f={f_um:.1f} um  pixel_pitch={pixel_pitch_um:.4f} um  f~{fx_px:.1f} px")
    print('  extrinsics / rig:')
    print(f"    baseline={sim.get('baseline_mm'):.2f} mm")
    print('  aberrations / degradations:')
    print(f"    distortion_model={sim.get('distortion_model')}  image_format={sim.get('image_format')}  outside_mask={sim.get('outside_mask')}")
    print(f"    blur_fwhm_um={sim.get('blur_fwhm_um'):.2f}  blur_edge_factor={sim.get('blur_edge_factor'):.2f}  blur_edge_start={sim.get('blur_edge_start'):.2f}  noise_std={sim.get('noise_std'):.3f}")
    print(f"    left_distortion={json.dumps(sim.get('distortion_left', {}), sort_keys=True)}")
    print(f"    right_distortion={json.dumps(sim.get('distortion_right', {}), sort_keys=True)}")
    print('  image setup:')
    print(f"    size={image.get('width_px')}x{image.get('height_px')} px  bit_depth={image.get('bit_depth')}  gamma={image.get('gamma')}  texture_interp={board.get('texture_interp')}")

cv2_obj, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = _make_charuco_detector(meta['board'])

def detect_charuco_view(img_gray):
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img_gray)
    else:
        if aruco_detector is not None:
            marker_corners, marker_ids, _rej = aruco_detector.detectMarkers(img_gray)
        else:
            marker_corners, marker_ids, _rej = aruco.detectMarkers(img_gray, dictionary, parameters=detector_params)
        charuco_corners, charuco_ids = None, None
        if hasattr(aruco, 'interpolateCornersCharuco') and marker_ids is not None and len(marker_ids) > 0:
            ret = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if ret is not None:
                if len(ret) == 3:
                    charuco_corners, charuco_ids, _ = ret
                elif len(ret) == 4:
                    _, charuco_corners, charuco_ids, _ = ret

    marker_ids_arr = np.asarray(marker_ids if marker_ids is not None else [], dtype=np.int32).reshape(-1)
    marker_corners_arr = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners] if marker_corners is not None else []
    charuco_ids_arr = np.asarray(charuco_ids if charuco_ids is not None else [], dtype=np.int32).reshape(-1)
    if charuco_corners is None or charuco_ids_arr.size == 0:
        charuco_xy = np.zeros((0, 2), dtype=np.float64)
    else:
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
    return {
        'marker_ids': marker_ids_arr,
        'marker_corners': marker_corners_arr,
        'charuco_ids': charuco_ids_arr,
        'charuco_xy': charuco_xy,
    }

def make_refined_points(det):
    refined = refine_charuco_corners(
        method='rayfield_tps_robust',
        board=board,
        marker_ids=det['marker_ids'],
        marker_corners=det['marker_corners'],
        charuco_ids=det['charuco_ids'],
        charuco_xy=det['charuco_xy'],
        tps_lam=10.0,
        huber_c=3.0,
        iters=3,
    )
    return np.asarray(refined, dtype=np.float64).reshape(-1, 2)

def gt_map_for_frame_id(fid, side):
    frame_ids = np.asarray(gt['frame_id'], dtype=np.int32).reshape(-1)
    corner_ids = np.asarray(gt['corner_id'], dtype=np.int32).reshape(-1)
    uv = np.asarray(gt['uv_left_px'] if side == 'left' else gt['uv_right_px'], dtype=np.float64).reshape(-1, 2)
    mask = frame_ids == int(fid)
    return {int(cid): uv[k] for k, cid in enumerate(corner_ids[mask].tolist())}

print('scene_dir =', scene_dir)
print('frames =', len(frames))
print('board type =', meta['board']['type'])
print_synthetic_setup(meta)

# %% [markdown]
# ## 1. A synthetic example to look at
#
# The first figure answers a very pragmatic question:
#
# **before looking at metrics, what point set is actually entering the calibration code?**
#
# Each panel shows the same board support, cropped around the useful region, with all detected
# ChArUco corners:
#
# - raw OpenCV detections in red,
# - Ray2D-refined detections in blue.
#
# At this stage we deliberately avoid plotting the GT on top of everything. When all three point
# sets are nearly coincident, the markers visually collapse and the reader learns nothing.
#
# What to inspect in this figure:
#
# - whether both views detect the full grid,
# - whether the refined points stay on the same geometric support,
# - whether the method is a gentle correction or a drastic rewrite of the detections.
#
# The expected interpretation is: Ray2D keeps the same board structure and corner count, but makes
# the point set more geometrically regular before OpenCV sees it.

# %%
frame0 = frames[0]
left_img = load_gray_u8(scene_dir / 'left' / frame0['left'])
right_img = load_gray_u8(scene_dir / 'right' / frame0['right'])

det_left = detect_charuco_view(left_img)
det_right = detect_charuco_view(right_img)
ref_left = make_refined_points(det_left)
ref_right = make_refined_points(det_right)

def crop_box(points, shape, pad=45):
    pts = np.asarray(points, dtype=np.float64)
    x0 = max(0.0, float(pts[:, 0].min() - pad))
    x1 = min(float(shape[1]), float(pts[:, 0].max() + pad))
    y0 = max(0.0, float(pts[:, 1].min() - pad))
    y1 = min(float(shape[0]), float(pts[:, 1].max() + pad))
    return x0, x1, y0, y1

def plot_detected_panel(ax, img, title, pts, color, marker, pad=45):
    if len(pts) == 0:
        raise RuntimeError('No ChArUco corner detected')
    x0, x1, y0, y1 = crop_box(pts, img.shape, pad=pad)
    ax.imshow(img, cmap='gray', origin='upper')
    ax.scatter(pts[:, 0], pts[:, 1], s=42, c=color, marker=marker, linewidths=1.6)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='gold', linewidth=1.0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect('equal')
    ax.set_title(f'{title}  |  detected corners = {len(pts)}')
    ax.grid(False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
plot_detected_panel(axes[0, 0], left_img, 'Left image - raw OpenCV', det_left['charuco_xy'], 'crimson', 'x')
plot_detected_panel(axes[0, 1], left_img, 'Left image - Ray2D', ref_left, 'dodgerblue', '+')
plot_detected_panel(axes[1, 0], right_img, 'Right image - raw OpenCV', det_right['charuco_xy'], 'crimson', 'x')
plot_detected_panel(axes[1, 1], right_img, 'Right image - Ray2D', ref_right, 'dodgerblue', '+')

legend_handles = [
    Line2D([0], [0], marker='x', linestyle='None', markersize=10, color='crimson', label='raw OpenCV corners'),
    Line2D([0], [0], marker='+', linestyle='None', markersize=10, color='dodgerblue', label='Ray2D refined corners'),
]
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, fontsize=12, handletextpad=0.8, columnspacing=1.8)
plt.show()

# %% [markdown]
# ## 1b. Zoom on one corner
#
# This is where the geometric story becomes visible.
# We take one corner id that is visible in both views, zoom around it, and compare:
#
# - the GT corner (green),
# - the raw OpenCV measurement (red),
# - the Ray2D-refined point (blue).
#
# The chosen corner is **not random**: it is the one with the largest average raw-to-refined
# displacement across the left and right images, so the zoom highlights a case where the second
# pass has something meaningful to correct.
#
# A useful way to read this overlay is through the displacement vector
#
# $$
# \delta \mathbf{p} = \mathbf{p}_{\mathrm{refined}} - \mathbf{p}_{\mathrm{raw}}.
# $$
#
# Even when $\|\delta \mathbf{p}\|$ is only a fraction of a pixel, stereo can be sensitive to it:
# the baseline, rectification, and triangulation all depend on coherent point geometry across both
# cameras. A tiny local bias can therefore produce a visible 3D effect once calibration is run.
#
# The purpose of this zoom is not to claim that every corner moves a lot. It is to show, on a
# concrete target, the kind of subpixel correction that later accumulates into better stereo
# behavior.

# %%
frame0 = frames[0]
left_img = load_gray_u8(scene_dir / 'left' / frame0['left'])
right_img = load_gray_u8(scene_dir / 'right' / frame0['right'])

det_left = detect_charuco_view(left_img)
det_right = detect_charuco_view(right_img)
ref_left = make_refined_points(det_left)
ref_right = make_refined_points(det_right)
gt_left = gt_map_for_frame_id(int(frame0['frame_id']), 'left')
gt_right = gt_map_for_frame_id(int(frame0['frame_id']), 'right')

def choose_common_corner(det_left, ref_left, gt_left, det_right, ref_right, gt_right):
    ids_left = {int(cid): i for i, cid in enumerate(det_left['charuco_ids'].tolist())}
    ids_right = {int(cid): i for i, cid in enumerate(det_right['charuco_ids'].tolist())}
    common = sorted(set(ids_left).intersection(ids_right).intersection(gt_left).intersection(gt_right))
    if not common:
        raise RuntimeError('No common corner visible in both views')
    scored = []
    for cid in common:
        l = float(np.linalg.norm(ref_left[ids_left[cid]] - det_left['charuco_xy'][ids_left[cid]]))
        r = float(np.linalg.norm(ref_right[ids_right[cid]] - det_right['charuco_xy'][ids_right[cid]]))
        scored.append((0.5 * (l + r), cid))
    return max(scored)[1]

corner_id = choose_common_corner(det_left, ref_left, gt_left, det_right, ref_right, gt_right)
print('Chosen common corner id =', corner_id)

def plot_zoom(ax, img, title, gt_pt, raw_pt, ref_pt, radius=45):
    h, w = img.shape[:2]
    x, y = float(gt_pt[0]), float(gt_pt[1])
    x0 = max(0, int(round(x - radius)))
    x1 = min(w, int(round(x + radius)))
    y0 = max(0, int(round(y - radius)))
    y1 = min(h, int(round(y + radius)))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.scatter([gt_pt[0]], [gt_pt[1]], s=110, c='lime', marker='o', edgecolors='black', linewidths=1.0, label='GT')
    ax.scatter([raw_pt[0]], [raw_pt[1]], s=95, c='crimson', marker='x', linewidths=2.6, label='raw')
    ax.scatter([ref_pt[0]], [ref_pt[1]], s=95, c='dodgerblue', marker='+', linewidths=2.6, label='Ray2D')
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='gold', linewidth=1.2))
    ax.set_title(title)
    ax.title.set_color('white')
    ax.title.set_path_effects([])
    ax.grid(False)

def plot_full(ax, img, title, gt_pt, raw_pt, ref_pt):
    ax.imshow(img, cmap='gray', origin='upper')
    ax.scatter([gt_pt[0]], [gt_pt[1]], s=70, c='lime', marker='o', edgecolors='black', linewidths=0.8, label='GT')
    ax.scatter([raw_pt[0]], [raw_pt[1]], s=55, c='crimson', marker='x', linewidths=2.0, label='raw')
    ax.scatter([ref_pt[0]], [ref_pt[1]], s=55, c='dodgerblue', marker='+', linewidths=2.0, label='Ray2D')
    ax.set_title(title)
    ax.title.set_color('white')
    ax.title.set_path_effects([])
    ax.grid(False)

id_to_idx_L = {int(cid): i for i, cid in enumerate(det_left['charuco_ids'].tolist())}
id_to_idx_R = {int(cid): i for i, cid in enumerate(det_right['charuco_ids'].tolist())}
raw_L = det_left['charuco_xy'][id_to_idx_L[corner_id]]
ref_L = ref_left[id_to_idx_L[corner_id]]
gt_L = gt_left[corner_id]
raw_R = det_right['charuco_xy'][id_to_idx_R[corner_id]]
ref_R = ref_right[id_to_idx_R[corner_id]]
gt_R = gt_right[corner_id]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
plot_full(axes[0, 0], left_img, f'Left image, frame {int(frame0["frame_id"])}', gt_L, raw_L, ref_L)
plot_full(axes[0, 1], right_img, f'Right image, frame {int(frame0["frame_id"])}', gt_R, raw_R, ref_R)
plot_zoom(axes[1, 0], left_img, f'Left zoom around corner {corner_id}', gt_L, raw_L, ref_L)
plot_zoom(axes[1, 1], right_img, f'Right zoom around corner {corner_id}', gt_R, raw_R, ref_R)

legend_handles = [
    Line2D([0], [0], marker='o', linestyle='None', markersize=10, markerfacecolor='lime', markeredgecolor='black', label='GT'),
    Line2D([0], [0], marker='x', linestyle='None', markersize=10, color='crimson', label='raw'),
    Line2D([0], [0], marker='+', linestyle='None', markersize=10, color='dodgerblue', label='Ray2D'),
]
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, fontsize=13, handletextpad=0.8, columnspacing=1.8)
plt.show()

# %% [markdown]
# ## 2. Aggregate metrics on the released sweep
#
# The previous figure is local and intuitive; this section is where the method should really be
# judged.
#
# We aggregate all successful cases from the released robustness sweep and compare the median metric
# values for:
#
# - raw OpenCV pinhole calibration,
# - `Ray2D + OpenCV`, meaning the same OpenCV backend fed with refined corners.
#
# The metrics shown here answer different questions:
#
# - mono RMS: how well each camera reprojects its calibration data,
# - stereo RMS: how well the two-camera fit explains the paired observations,
# - baseline error: how much the recovered rig drifts in a disparity-equivalent unit,
# - `Tri %Zbar`: how much the reconstructed depth deviates relative to the average scene depth.
#
# This distinction matters because a low mono reprojection error does not automatically imply good
# stereo geometry. The most important reading of the bar chart is therefore not “which bar is
# smaller in isolation?”, but “does the refined 2D point set improve the whole chain consistently?”

# %%
sweep = load_json('paper/tables/robustness_sweep/summary.json')
cases = [c for c in sweep['cases'] if c.get('status') == 'ok']

metric_map = [
    ('raw.mono_L_rms_px', 'rayfield_tps_robust.mono_L_rms_px', 'mono RMS left (px)'),
    ('raw.mono_R_rms_px', 'rayfield_tps_robust.mono_R_rms_px', 'mono RMS right (px)'),
    ('raw.stereo_rms_px', 'rayfield_tps_robust.stereo_rms_px', 'stereo RMS (px)'),
    ('raw.baseline_delta_px_rms', 'rayfield_tps_robust.baseline_delta_px_rms', 'baseline err (px @ Zbar)'),
    ('raw.tri_rms_mm', 'rayfield_tps_robust.tri_rms_mm', 'triangulation RMS (mm)'),
]

summary_rows = []
for raw_key, ray_key, label in metric_map:
    raw_vals = np.array([float(c[raw_key]) for c in cases], dtype=float)
    ray_vals = np.array([float(c[ray_key]) for c in cases], dtype=float)
    summary_rows.append({
        'label': label,
        'raw_median': float(np.median(raw_vals)),
        'ray_median': float(np.median(ray_vals)),
        'improved_count': int((ray_vals < raw_vals).sum()),
        'n': int(len(raw_vals)),
    })

print(f"{'metric':32s} {'raw median':>12s} {'ray2d median':>14s} {'improved':>12s}")
for row in summary_rows:
    print(f"{row['label']:32s} {row['raw_median']:12.4f} {row['ray_median']:14.4f} {row['improved_count']:5d}/{row['n']:<6d}")

labels = [row['label'] for row in summary_rows]
raw_vals = np.array([row['raw_median'] for row in summary_rows], dtype=float)
ray_vals = np.array([row['ray_median'] for row in summary_rows], dtype=float)
x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(1, 1, figsize=(12, 4.8), constrained_layout=True)
ax.bar(x - w/2, raw_vals, width=w, label='raw OpenCV', color='crimson', alpha=0.85)
ax.bar(x + w/2, ray_vals, width=w, label='Ray2D + OpenCV', color='dodgerblue', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha='right')
ax.set_title('Median metrics across the released robustness sweep')
ax.set_ylabel('value (mixed units, smaller is better)')
ax.legend()
plt.show()

# %% [markdown]
# ## 3. Per-case scatter: every point should go below the diagonal
#
# Median bars are compact, but they hide dispersion. The next figure keeps the **per-case**
# information: each point is one successful benchmark case, with
#
# - raw OpenCV on the x-axis,
# - `Ray2D + OpenCV` on the y-axis.
#
# The diagonal is the “no change” line. A point below it means the refined pipeline performs better
# for that exact case; a point above it means worse.
#
# This is the right plot to answer two practical concerns:
#
# - is the gain broad or carried by only a few easy cases?
# - does the refinement introduce catastrophic regressions on difficult images?
#
# For a usable preprocessing stage, the cloud should move below the diagonal in a stable way rather
# than showing a fragile trade-off.

# %%
sweep = load_json('paper/tables/robustness_sweep/summary.json')
cases = [c for c in sweep['cases'] if c.get('status') == 'ok']

raw_stereo = np.array([float(c['raw.stereo_rms_px']) for c in cases], dtype=float)
ray_stereo = np.array([float(c['rayfield_tps_robust.stereo_rms_px']) for c in cases], dtype=float)
raw_tri = np.array([float(c['raw.tri_rms_mm']) for c in cases], dtype=float)
ray_tri = np.array([float(c['rayfield_tps_robust.tri_rms_mm']) for c in cases], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
pairs = [
    (axes[0], raw_stereo, ray_stereo, 'Stereo RMS (px)'),
    (axes[1], raw_tri, ray_tri, 'Triangulation RMS (mm)'),
]
for ax, x, y, title in pairs:
    lim = float(max(np.max(x), np.max(y)) * 1.05)
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.scatter(x, y, s=44, alpha=0.85, color='royalblue')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('raw OpenCV')
    ax.set_ylabel('Ray2D + OpenCV')
    ax.set_title(title)

plt.show()
print(f'Cases plotted: {len(cases)}')
print(f'Triangulation improved in {(ray_tri < raw_tri).sum()} / {len(cases)} cases')
print(f'Stereo RMS improved in {(ray_stereo < raw_stereo).sum()} / {len(cases)} cases')

# %% [markdown]
# ## How to rerun
#
# This notebook is deliberately lightweight, but it stays traceable to the production scripts.
# If you want to regenerate the underlying outputs instead of reading the committed JSON files, run:
#
# ```bash
# .venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/compression_sweep_pnp/png_lossless --split train --scene scene_0000 --out paper/tables/compression_compare/png_lossless.raw.json
# .venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/compression_sweep_pnp/png_lossless --split train --scene scene_0000 --out paper/tables/compression_compare/png_lossless.rayfield2d.json
# .venv/bin/python paper/experiments/sweep_robustness_board_focal_aberrations.py --seeds 0,1 --frames 16 --run-rayfield3d
# ```
#
# A reasonable reading strategy is:
#
# 1. use this notebook to understand the method and the metrics,
# 2. inspect the committed JSON summaries if you want the raw numbers,
# 3. rerun the heavy scripts only when you want to regenerate the benchmark from scratch.
# 
# That separation is intentional: the notebook teaches, the experiment scripts produce.

# %%
pass

# %%
pass
