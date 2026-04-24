"""Exported companion script for 01_ray2d_vs_opencv.ipynb.

This file mirrors the notebook content in a linear, readable Python form.
"""

# %% [markdown]
# # Ray2D + OpenCV from two image folders
#
# This notebook is the best place to understand the **2D** contribution of StereoComplex before
# looking at the 3D ray-field backend. It is written around the exact onboarding workflow a new
# user expects:
#
# - start from one folder of **left** images and one folder of **right** images,
# - define the ChArUco board once with `CharucoBoardSpec`,
# - run a standard OpenCV stereo calibration on the **raw** detections,
# - run the same OpenCV calibration again after the `rayfield_tps_robust` second pass,
# - then move from this local workflow to the **global released sweep**.
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
# - Section 0 is the workflow you should copy on your own data:
#   `left_dir`, `right_dir`, `board`, then raw OpenCV vs `Ray2D + OpenCV`.
# - Section 1 explains where the committed sample images come from.
# - Sections 2 and 3 show **what physically moves** in the corner detections.
# - Sections 4 and 5 answer the global question: does that local correction improve the stereo
#   pipeline consistently over the released benchmark?
#
# The notebook now uses the **public StereoComplex API** both for the calibration path and for the
# image-space steps:
#
# - `CharucoBoardSpec`
# - `detect_charuco_corners`
# - `refine_charuco_corners`
# - `fit_opencv_stereo_from_image_dirs`
#
# The only remaining JSON files are the committed sweep summaries used for the aggregate plots.
# Everything up to the raw-vs-Ray2D calibration comparison is therefore the same API path a user
# can apply to their own folders of real images.

# %%
from pathlib import Path
import json
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

import stereocomplex as sc
from stereocomplex.core.image_io import load_gray_u8

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
# ## 0. Minimal workflow: two folders + one board definition
#
# If you arrive with your own stereo captures, the **minimum useful input** is:
#
# - one folder `left_dir/` with the left images,
# - one folder `right_dir/` with the matching right images,
# - the ChArUco geometry and dictionary.
#
# The code cell below does exactly that:
#
# 1. define the board,
# 2. point to the two image folders,
# 3. run **raw OpenCV** calibration,
# 4. run **Ray2D + OpenCV** calibration,
# 5. compare the resulting stereo reports.
#
# This is the part to copy first when adapting StereoComplex to your own data. The repository
# sample is synthetic, but the API call itself is not synthetic-specific.
#
# The only three lines you would replace on your side are:
#
# - `left_dir = ...`
# - `right_dir = ...`
# - `board = sc.CharucoBoardSpec(...)`

# %%
scene_dir = ROOT / 'dataset' / 'compression_sweep_pnp' / 'png_lossless' / 'train' / 'scene_0000'
left_dir = scene_dir / 'left'
right_dir = scene_dir / 'right'
frame0 = {'frame_id': 0, 'left': '000000.png', 'right': '000000.png'}
board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.07131716473686,
    marker_size_mm=27.3499220153158,
    aruco_dictionary='DICT_4X4_1000',
)
raw_calib = sc.fit_opencv_stereo_from_image_dirs(
    left_dir=left_dir,
    right_dir=right_dir,
    board=board,
    method2d='raw',
)
ray2d_calib = sc.fit_opencv_stereo_from_image_dirs(
    left_dir=left_dir,
    right_dir=right_dir,
    board=board,
    method2d='rayfield_tps_robust',
)
gt = np.load(scene_dir / 'gt_charuco_corners.npz')
synthetic_setup = {
    'camera_model': 'pinhole',
    'f_um': 5937.556676880016,
    'pixel_pitch_um': 4.879994502565531,
    'baseline_mm': 170.78787723319832,
    'distortion_model': 'brown',
    'distortion_left': {
        'k1': 0.12700349940811692,
        'k2': -0.04021660839652088,
        'p1': 0.00767410947823205,
        'p2': 0.009357424634668461,
        'k3': 0.0007782273137719147,
    },
    'distortion_right': {
        'k1': -0.07651061967936106,
        'k2': 0.058363487297534024,
        'p1': -0.0068696888789977085,
        'p2': 0.005795209253633032,
        'k3': 0.007796412275380848,
    },
    'image_format': 'png',
    'outside_mask': 'hard',
    'blur_fwhm_um': 6.0,
    'blur_edge_factor': 2.5,
    'blur_edge_start': 0.55,
    'noise_std': 0.01,
    'image_width_px': 800,
    'image_height_px': 600,
    'bit_depth': 8,
    'gamma': 1.0,
    'texture_interp': 'lanczos4',
}

def print_synthetic_setup(board, setup):
    f_um = setup.get('f_um')
    pixel_pitch_um = setup.get('pixel_pitch_um')
    fx_px = f_um / pixel_pitch_um if f_um and pixel_pitch_um else None
    print('\nSynthetic setup summary')
    print('  board:')
    print(f"    type=charuco  dictionary={board.aruco_dictionary}")
    print(f"    grid={board.squares_x}x{board.squares_y}  square={board.square_size_mm:.2f} mm  marker={board.marker_size_mm:.2f} mm")
    print('  intrinsics / optics:')
    print(f"    camera_model={setup.get('camera_model')}  f={f_um:.1f} um  pixel_pitch={pixel_pitch_um:.4f} um  f~{fx_px:.1f} px")
    print('  extrinsics / rig:')
    print(f"    baseline={setup.get('baseline_mm'):.2f} mm")
    print('  aberrations / degradations:')
    print(f"    distortion_model={setup.get('distortion_model')}  image_format={setup.get('image_format')}  outside_mask={setup.get('outside_mask')}")
    print(f"    blur_fwhm_um={setup.get('blur_fwhm_um'):.2f}  blur_edge_factor={setup.get('blur_edge_factor'):.2f}  blur_edge_start={setup.get('blur_edge_start'):.2f}  noise_std={setup.get('noise_std'):.3f}")
    print(f"    left_distortion={json.dumps(setup.get('distortion_left', {}), sort_keys=True)}")
    print(f"    right_distortion={json.dumps(setup.get('distortion_right', {}), sort_keys=True)}")
    print('  image setup:')
    print(f"    size={setup.get('image_width_px')}x{setup.get('image_height_px')} px  bit_depth={setup.get('bit_depth')}  gamma={setup.get('gamma')}  texture_interp={setup.get('texture_interp')}")


def print_calibration_report(label, result):
    report = result.report
    print(
        f"{label:18s}  monoL={report.mono_left_rms_px:7.4f} px  "
        f"monoR={report.mono_right_rms_px:7.4f} px  stereo={report.stereo_rms_px:7.4f} px  "
        f"baseline={report.baseline_mm:9.3f} mm  stereo_frames={report.n_stereo_frames}"
    )

def detect_charuco_view(img_gray):
    det = sc.detect_charuco_corners(image=img_gray, board=board)
    if det is None:
        return {
            'marker_ids': np.zeros((0,), dtype=np.int32),
            'marker_corners': [],
            'charuco_ids': np.zeros((0,), dtype=np.int32),
            'charuco_xy': np.zeros((0, 2), dtype=np.float64),
        }
    return {
        'marker_ids': det.marker_ids,
        'marker_corners': det.marker_corners,
        'charuco_ids': det.charuco_ids,
        'charuco_xy': det.charuco_xy,
    }

def make_refined_points(det):
    refined = sc.refine_charuco_corners(
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

print('left_dir  =', left_dir)
print('right_dir =', right_dir)
print('frame_id  =', frame0['frame_id'])
print('board type = charuco')
print()
print('OpenCV stereo calibration from image folders')
print_calibration_report('raw OpenCV', raw_calib)
print_calibration_report('Ray2D + OpenCV', ray2d_calib)
print()
print('To reuse this on your own images, change only: left_dir, right_dir, board')
print_synthetic_setup(board, synthetic_setup)

# %% [markdown]
# ## 1. Why this sample is still synthetic
#
# The worked example above already uses the same public API as a real user workflow. The reason we
# keep a **synthetic committed sample** in the notebook is different: it gives us controlled ground
# truth for the later zooms and aggregate plots.
#
# This matters because it lets us separate three questions cleanly:
#
# - what OpenCV detected,
# - what Ray2D changed,
# - and where the geometric target was actually supposed to be.
#
# On your own real data, you would keep the first calibration cell and skip the GT-specific visual
# comparisons. Here we can afford both, so the notebook can teach the method rather than only run
# it.

# %% [markdown]
# ## 2. A synthetic example to look at
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
# ## 3. Zoom on one corner
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
#
# The zoomed ROI is computed automatically from the three plotted locations (`GT`, `raw`,
# `Ray2D`). We first measure their spread, then build a tight square crop with a small safety
# margin. This avoids the previous failure mode where a large fixed crop made the three symbols
# visually collapse.

# %%
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

def make_zoom_roi(gt_pt, raw_pt, ref_pt, shape, *, min_half_width=9.0, max_half_width=24.0, pad_px=5.0):
    pts = np.stack(
        [
            np.asarray(gt_pt, dtype=np.float64),
            np.asarray(raw_pt, dtype=np.float64),
            np.asarray(ref_pt, dtype=np.float64),
        ],
        axis=0,
    )
    h, w = shape[:2]
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    spread = float(np.max(bbox_max - bbox_min))
    pairwise = [
        float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    ]
    max_disp = max(pairwise) if pairwise else 0.0
    half_width = min(max_half_width, max(min_half_width, 0.5 * spread + 3.0 * max_disp + pad_px))
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    x0 = max(0.0, cx - half_width)
    x1 = min(float(w), cx + half_width)
    y0 = max(0.0, cy - half_width)
    y1 = min(float(h), cy + half_width)
    return x0, x1, y0, y1, max_disp


def plot_zoom(ax, img, title, gt_pt, raw_pt, ref_pt):
    x0, x1, y0, y1, max_disp = make_zoom_roi(gt_pt, raw_pt, ref_pt, img.shape)
    h, w = img.shape[:2]
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.scatter([gt_pt[0]], [gt_pt[1]], s=170, c='lime', marker='o', edgecolors='black', linewidths=1.4, label='GT', zorder=3)
    ax.scatter([raw_pt[0]], [raw_pt[1]], s=140, c='crimson', marker='x', linewidths=3.0, label='raw', zorder=4)
    ax.scatter([ref_pt[0]], [ref_pt[1]], s=140, c='dodgerblue', marker='+', linewidths=3.0, label='Ray2D', zorder=5)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='gold', linewidth=1.2))
    ax.set_title(f'{title}  |  max disp = {max_disp:.3f} px')
    ax.title.set_color('white')
    ax.title.set_path_effects([])
    ax.grid(False)
    ax.set_aspect('equal')

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
# ## 4. Aggregate metrics on the released sweep
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
# ## 5. Per-case scatter: every point should go below the diagonal
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
# If you want to regenerate the underlying outputs instead of reading the committed JSON files, use
# the public API for the local 2D steps and the benchmark scripts only for the heavy released sweep:
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
