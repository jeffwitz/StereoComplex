"""Exported companion script for 02_ray3d.ipynb.

This file mirrors the notebook content in a linear, readable Python form.
"""

# %% [markdown]
# # ray3D: central ray-field workflow
#
# This notebook is the 3D continuation of the Ray2D walkthrough. The first notebook explained how
# better 2D correspondences help a classical pinhole stereo fit. This one asks the next question:
#
# **what happens if we also replace the 3D camera model by a compact central ray-field backend?**
#
# The underlying geometric object is a ray through a fixed camera center:
#
# $$
# \ell_p(t) = C + t\,\hat{\mathbf d}(u,v), \qquad
# \hat{\mathbf d}(u,v) = \frac{[x(u,v),\,y(u,v),\,1]^\top}{\|[x(u,v),\,y(u,v),\,1]^\top\|}
# $$
#
# Compared with a pinhole model, the unknown is no longer only a matrix of intrinsics and a small
# distortion vector. The mapping from pixel to direction is represented in a compact Zernike basis,
# then optimized with point-to-ray consistency.
#
# The notebook is organized as a progression from the cleanest benchmark to the most realistic one:
#
# - Section 1: a **true Z-sweep** tailored to compare 3D backends fairly,
# - Section 2: a **pose-sweep compression benchmark** where Pycaso becomes stressed,
# - Section 3: the **full StereoComplex compression sweep**, which combines 2D and 3D effects,
# - Section 4: one local corner overlay to reconnect the global curves to image-space measurements,
# - Section 5: the public API entry point used to load an exported ray3D model.
#
# As in the first notebook, we read only versioned JSON summaries and committed example assets. The
# goal is explanation, not long recomputation.
#
# The important change compared with the older version of this walkthrough is that the **local**
# operations now use the public API:
#
# - `CharucoBoardSpec`
# - `detect_charuco_corners`
# - `refine_charuco_corners`
# - `fit_stereo_central_rayfield_from_dataset`
# - `load_stereo_central_rayfield`
#
# The committed JSON files remain only for the published global sweeps.

# %%
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

print('ROOT =', ROOT)

# %% [markdown]
# ## 0. What ray3D is trying to do
#
# The ray3D backend is designed for a specific gap in standard stereo workflows:
#
# - pinhole calibration is convenient and efficient,
# - but under strong photometric stress or more complex optics, a rigid pinhole parameterization may
#   absorb the wrong signal.
#
# StereoComplex therefore keeps the pipeline modular:
#
# 1. first improve the 2D observations with Ray2D,
# 2. then optionally replace the classical pinhole 3D backend with a central ray-field model.
#
# The fit itself is driven by **point-to-ray** consistency. Intuitively, the optimizer asks whether
# the observed board points are explained by a coherent bundle of rays across both cameras, rather
# than only by a small parametric projection model.
#
# The local overlay shown later in the notebook still matters, because ray3D is not magic: it is
# fed by 2D corners. The reason to start from global plots is simply that the 3D claim must be
# assessed on benchmark trends, not on one hand-picked image.
#
# For that reason the notebook separates two layers very explicitly:
#
# - public API calls for the per-image and per-model workflow,
# - benchmark JSON summaries for the multi-case plots already released in the repo.

# %%
scene_dir = ROOT / 'dataset' / 'compression_sweep_pnp' / 'png_lossless' / 'train' / 'scene_0000'
frame0 = {'frame_id': 0, 'left': '000000.png', 'right': '000000.png'}
gt = np.load(scene_dir / 'gt_charuco_corners.npz')
board = sc.CharucoBoardSpec(
    squares_x=11,
    squares_y=7,
    square_size_mm=39.07131716473686,
    marker_size_mm=27.3499220153158,
    aruco_dictionary='DICT_4X4_1000',
)
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

def gt_map_for_frame_id(fid, side):
    frame_id = np.asarray(gt['frame_id'], dtype=np.int32).reshape(-1)
    corner_id = np.asarray(gt['corner_id'], dtype=np.int32).reshape(-1)
    uv = np.asarray(gt['uv_left_px'] if side == 'left' else gt['uv_right_px'], dtype=np.float64).reshape(-1, 2)
    mask = frame_id == int(fid)
    return {int(cid): uv[k] for k, cid in enumerate(corner_id[mask].tolist())}

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

print('scene_dir =', scene_dir)
print('frame_id =', frame0['frame_id'])
print('board type = charuco')
print_synthetic_setup(board, synthetic_setup)

# %% [markdown]
# ## 1. True Z-sweep benchmark: OpenCV, Pycaso, and ray3D
#
# We start from the dedicated `PYCASO_Z_SWEEP` benchmark, where the board translates only along
# $Z$ while the two cameras stay fixed. This protocol is intentionally simple:
#
# - the board keeps the same in-plane geometry,
# - only the depth changes,
# - and the camera pair itself does not move.
#
# This is the cleanest place to compare OpenCV, Pycaso-like polynomial backends, and ray3D because
# the experiment is naturally aligned with Pycaso's preferred operating regime. If ray3D is
# competitive here, the comparison is fair; if it is better later under harder perturbations, that
# extra gain cannot be dismissed as a protocol mismatch.
#
# What to look for in the next plots:
#
# - the left panel shows how the RMS Z error evolves with depth,
# - the right panel compresses that into one global RMS value per backend.
#
# This section answers the baseline question: on clean, well-conditioned data, does ray3D stay in
# the same accuracy range as the other methods?

# %%
z_sweep = load_json('validation/z_sweep/pycaso_z_sweep_metrics_refined.json')

methods = [
    ('opencv_pinhole', 'OpenCV pinhole', 'crimson', 'o'),
    ('pycaso_direct_poly', 'Pycaso direct', 'darkorange', 's'),
    ('pycaso_soloff_lm', 'Pycaso Soloff LM', 'purple', '^'),
    ('rayfield3d_fixed', 'ray3D fixed', 'royalblue', 'D'),
]

print(f"{'method':24s} {'global RMS Z (mm)':>18s}")
for key, label, _color, _marker in methods:
    print(f"{label:24s} {float(z_sweep[key]['rms_z_mm']):18.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

for key, label, color, marker in methods:
    by_depth = z_sweep[key]['by_depth']
    depth_mm = np.asarray(by_depth['depth_mm'], dtype=float)
    rms_mm = np.asarray(by_depth['rms'], dtype=float)
    axes[0].plot(depth_mm, rms_mm, marker=marker, color=color, linewidth=2, label=label)

axes[0].set_xlabel('board depth Z (mm)')
axes[0].set_ylabel('RMS Z error (mm)')
axes[0].set_title('Z-sweep: per-depth RMS Z')
axes[0].legend()

labels = [label for _key, label, _color, _marker in methods]
values = [float(z_sweep[key]['rms_z_mm']) for key, _label, _color, _marker in methods]
colors = [color for _key, _label, color, _marker in methods]
axes[1].bar(labels, values, color=colors, alpha=0.85)
axes[1].set_ylabel('global RMS Z (mm)')
axes[1].set_title('Z-sweep: global summary')
axes[1].tick_params(axis='x', rotation=18)

plt.show()

# %% [markdown]
# ## 2. Global compression pose-sweep: Pycaso vs ray3D
#
# The next plot switches to the published **pose-sweep compression benchmark**, which is harder and
# more revealing.
#
# Here the board is no longer limited to a pure depth translation. The pose set explores a broader
# family of configurations, while codec degradations perturb the underlying 2D observations.
#
# This regime is especially informative for polynomial direct mappings such as Pycaso:
#
# - they can be very strong in a well-matched sweep protocol,
# - but they become more fragile when the observation manifold broadens and the image evidence is
#   perturbed.
#
# Each point in the figure is one codec setting. The ratio plot on the right is important: it tells
# you not only that ray3D is smaller, but **by how much** the error is reduced for each degradation
# level.

# %%
pose = load_json('paper/tables/pycaso_pose_sweep_sweep_summary.json')

codec_order = ['png_lossless', 'webp_q70', 'webp_q80', 'webp_q90', 'webp_q95', 'jpeg_q80', 'jpeg_q90', 'jpeg_q95', 'jpeg_q98']
labels = [c.replace('_', ' ') for c in codec_order]
pycaso_rel = np.array([float(pose['cases'][c]['metrics']['pycaso_direct_poly']['rms_z_rel_percent']) for c in codec_order], dtype=float)
ray3d_rel = np.array([float(pose['cases'][c]['metrics']['rayfield3d_fixed']['rms_z_rel_percent']) for c in codec_order], dtype=float)
gain = pycaso_rel / ray3d_rel

print(f"{'codec':16s} {'Pycaso direct (%)':>18s} {'ray3D fixed (%)':>18s} {'gain x':>12s}")
for label, p, r, g in zip(labels, pycaso_rel, ray3d_rel, gain, strict=True):
    print(f"{label:16s} {p:18.6f} {r:18.6f} {g:12.2f}")

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
axes[0].plot(x, pycaso_rel, marker='o', color='darkorange', linewidth=2, label='Pycaso direct')
axes[0].plot(x, ray3d_rel, marker='D', color='royalblue', linewidth=2, label='ray3D fixed')
axes[0].set_yscale('log')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=25, ha='right')
axes[0].set_ylabel('RMS Z (% mean depth, log scale)')
axes[0].set_title('Pose-sweep compression benchmark')
axes[0].legend()

axes[1].bar(x, gain, color='slateblue', alpha=0.85)
axes[1].axhline(1.0, color='k', linestyle='--', linewidth=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=25, ha='right')
axes[1].set_ylabel('Pycaso / ray3D improvement factor')
axes[1].set_title('How much ray3D reduces RMS Z')

plt.show()

# %% [markdown]
# ## 3. Full compression sweep in the stereo pipeline
#
# This last global plot puts ray3D back into the full StereoComplex story.
# We now compare three full pipelines across all released codec settings:
#
# - raw pinhole,
# - pinhole + Ray2D,
# - ray3D + Ray2D.
#
# This section is useful because it separates two questions that are easy to confuse:
#
# - how much is already gained by improving the 2D inputs?
# - what additional robustness comes from changing the 3D backend itself?
#
# In practice, the expected reading is:
#
# - raw pinhole gives the reference baseline,
# - pinhole + Ray2D shows what better correspondences alone buy you,
# - ray3D + Ray2D shows whether the central ray-field backend remains more stable under aggressive
#   codec perturbations.

# %%
sweep3d = load_json('docs/assets/compression_sweep/sweep_metrics.json')
codec_order = ['png_lossless', 'webp_q70', 'webp_q80', 'webp_q90', 'webp_q95', 'jpeg_q80', 'jpeg_q90', 'jpeg_q95', 'jpeg_q98']
labels = [c.replace('_', ' ') for c in codec_order]
methods = [
    ('opencv_pinhole_raw', 'raw pinhole', 'crimson', 'o'),
    ('opencv_pinhole_rayfield2d', 'pinhole + Ray2D', 'dodgerblue', 's'),
    ('rayfield3d_ba_rayfield2d', 'ray3D + Ray2D', 'forestgreen', 'D'),
]

fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
x = np.arange(len(codec_order))
for key, label, color, marker in methods:
    vals = np.array([float(sweep3d['cases'][c][key]['tri_rms_rel_depth_percent']) for c in codec_order], dtype=float)
    ax.plot(x, vals, marker=marker, color=color, linewidth=2, label=label)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylabel('Triangulation RMS (% mean depth, log scale)')
ax.set_title('Full compression sweep: 3D backends across all codecs')
ax.legend()
plt.show()

print(f"{'codec':16s} {'raw pinhole':>14s} {'pinhole+Ray2D':>16s} {'ray3D+Ray2D':>14s}")
for c, label in zip(codec_order, labels, strict=True):
    row = sweep3d['cases'][c]
    print(f"{label:16s} {float(row['opencv_pinhole_raw']['tri_rms_rel_depth_percent']):14.6f} {float(row['opencv_pinhole_rayfield2d']['tri_rms_rel_depth_percent']):16.6f} {float(row['rayfield3d_ba_rayfield2d']['tri_rms_rel_depth_percent']):14.6f}")

# %% [markdown]
# ## 4. Local corner view: what enters the 3D fit
#
# The global curves above are the real evidence, but it is still useful to reconnect them to the
# raw image domain.
#
# The ray3D fit is ultimately driven by 2D corners. This local view therefore shows one stereo pair
# and one corner where the Ray2D preprocessing moves the observation before the 3D backend sees it.
#
# Read this panel as a reminder of the full logic:
#
# - ray3D is a 3D model,
# - but its stability still depends on the quality of the image-space measurements,
# - which is why Ray2D and ray3D are complementary rather than competing ideas.

# %%
left_img = load_gray_u8(scene_dir / 'left' / frame0['left'])
right_img = load_gray_u8(scene_dir / 'right' / frame0['right'])
det_left = detect_charuco_view(left_img)
det_right = detect_charuco_view(right_img)
ref_left = make_refined_points(det_left)
ref_right = make_refined_points(det_right)
gt_left = gt_map_for_frame_id(int(frame0['frame_id']), 'left')
gt_right = gt_map_for_frame_id(int(frame0['frame_id']), 'right')

def choose_corner(det, ref, gt_map):
    ids = det['charuco_ids']
    raw = det['charuco_xy']
    common = [int(cid) for cid in ids.tolist() if int(cid) in gt_map]
    id_to_idx = {int(cid): i for i, cid in enumerate(ids.tolist())}
    shifts = []
    for cid in common:
        i = id_to_idx[cid]
        shifts.append((float(np.linalg.norm(ref[i] - raw[i])), cid))
    return max(shifts)[1]

corner_id_L = choose_corner(det_left, ref_left, gt_left)
corner_id_R = choose_corner(det_right, ref_right, gt_right)
print('Chosen left corner id =', corner_id_L)
print('Chosen right corner id =', corner_id_R)

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
    ax.scatter([gt_pt[0]], [gt_pt[1]], s=80, c='lime', marker='o', edgecolors='black', linewidths=0.8, label='GT')
    ax.scatter([raw_pt[0]], [raw_pt[1]], s=70, c='crimson', marker='x', linewidths=2.0, label='raw')
    ax.scatter([ref_pt[0]], [ref_pt[1]], s=70, c='dodgerblue', marker='+', linewidths=2.0, label='ray3D fit')
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='gold', linewidth=1.2))
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(False)

id_to_idx_L = {int(cid): i for i, cid in enumerate(det_left['charuco_ids'].tolist())}
id_to_idx_R = {int(cid): i for i, cid in enumerate(det_right['charuco_ids'].tolist())}
raw_L = det_left['charuco_xy'][id_to_idx_L[corner_id_L]]
ref_L = ref_left[id_to_idx_L[corner_id_L]]
gt_L = gt_left[corner_id_L]
raw_R = det_right['charuco_xy'][id_to_idx_R[corner_id_R]]
ref_R = ref_right[id_to_idx_R[corner_id_R]]
gt_R = gt_right[corner_id_R]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
plot_zoom(axes[0], left_img, f'Left zoom around corner {corner_id_L}', gt_L, raw_L, ref_L)
plot_zoom(axes[1], right_img, f'Right zoom around corner {corner_id_R}', gt_R, raw_R, ref_R)
plt.show()

# %% [markdown]
# ## 5. API entry point
#
# The ray3D backend is meant to be used as a compact exported model, not only as an experiment
# script result.
#
# The code below is therefore intentionally small and safe:
#
# - it loads an exported ray-field model if one exists in your checkout,
# - it inspects the available metadata,
# - and it shows the public entry point that downstream code should rely on.
#
# This is the bridge from the benchmark world to actual use: once the model is exported, the rest of
# the pipeline can interact with a stable API instead of reconstructing the calibration state by
# hand.
#
# The notebook therefore uses the **public** dataset wrapper first. The older `paper/experiments`
# script is still valid for paper reproducibility, but it is no longer the recommended onboarding
# path.

# %%
model_dir = ROOT / 'models' / 'scene0000_rayfield3d'
if model_dir.exists():
    model = sc.load_stereo_central_rayfield(model_dir)
    uvL = np.array([[320.0, 240.0]], dtype=float)
    uvR = np.array([[318.5, 240.0]], dtype=float)
    xyz_mm, skew_mm = model.triangulate(uvL, uvR)
    print('XYZ_mm =', xyz_mm)
    print('skew_mm =', skew_mm)
else:
    print('No exported model found at', model_dir, '- fitting one with the public API...')
    fit = sc.fit_stereo_central_rayfield_from_dataset(
        dataset_root=ROOT / 'dataset' / 'compression_sweep_pnp' / 'png_lossless',
        split='train',
        scene='scene_0000',
        max_frames=5,
        method2d='rayfield_tps_robust',
        nmax=10,
        export_model_dir=model_dir,
    )
    model = fit.model
    print(fit.report)

print('\nEquivalent public API for your own folders:')
print('  sc.fit_stereo_central_rayfield_from_image_dirs(left_dir=..., right_dir=..., board=..., export_model_dir=...)')

# %%
pass

# %%
pass
