#!/usr/bin/env python3
"""Pycaso coin specimen — dense stereo reconstruction via ray intersection."""

import json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10})

# ═══════════════════════════════════════════════════════════
# 1 — Load images
# ═══════════════════════════════════════════════════════════
PYCASO = Path('/home/jeff/StereoComplex/examples/pycaso_data/Exemple/Images_example')
OUT = Path('docs/assets/pycaso_real_data')
OUT.mkdir(parents=True, exist_ok=True)

left_path = PYCASO / 'left_identification' / 'coin.tif'
right_path = PYCASO / 'right_identification2' / 'coin_1.tif'
print(f"Loading: {left_path.name}, {right_path.name}")

imgL = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
imgR = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
H, W = imgL.shape
print(f"  Image size: {W}x{H}")

# ── ROI: avoid borders where flow breaks ──
roi_x0, roi_x1 = 300, W - 300
roi_y0, roi_y1 = 300, H - 300
roiL = imgL[roi_y0:roi_y1, roi_x0:roi_x1]
roiR = imgR[roi_y0:roi_y1, roi_x0:roi_x1]
h_roi, w_roi = roiL.shape
print(f"  ROI: [{roi_x0}:{roi_x1}, {roi_y0}:{roi_y1}] -> {w_roi}x{h_roi}")

# ═══════════════════════════════════════════════════════════
# 2 — Dense optical flow (DIS)
# ═══════════════════════════════════════════════════════════
print("\nComputing dense optical flow (DIS, Pycaso DIC parameters)...")
t0 = time.time()

# Pycaso DIC-equivalent parameters mapped to OpenCV DISOpticalFlow:
#   pyram_levels=3           → finest_scale=0 (full res, pyramid has 3 levels built-in)
#   ordre_inter=3            → cubic interpolation (DIS default)
#   max_iter=10              → gradient_descent_iterations
#   max_linear_iter=1        → variational_refinement_iterations
#   lmbda=20000              → variational_refinement_alpha
#   lambda2=0.001            → variational_refinement_gamma
#   lambda3=1.0              → variational_refinement_delta
#   size_median_filter=3     → post-processing (applied separately)
#   factor=2.0               → implicit in pyramid construction

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setFinestScale(0)        # full-resolution finest scale
dis.setGradientDescentIterations(10)      # max_iter=10
dis.setVariationalRefinementIterations(1) # max_linear_iter=1
dis.setVariationalRefinementAlpha(20000.0)  # lmbda=20000
dis.setVariationalRefinementGamma(0.001)    # lambda2=0.001
dis.setVariationalRefinementDelta(1.0)      # lambda3=1.0

flow = dis.calc((roiL * 255).astype(np.uint8), (roiR * 255).astype(np.uint8), None)
print(f"  DIS optical flow: {time.time()-t0:.1f}s")

# Post-processing: median filter (size_median_filter=3 in Pycaso)
dx = cv2.medianBlur(flow[..., 0].astype(np.float32), 3)
dy = cv2.medianBlur(flow[..., 1].astype(np.float32), 3)

# Right pixels = left_pixel + flow (BOTH U and V components — images NOT rectified)
xL, yL = np.meshgrid(np.arange(w_roi), np.arange(h_roi))
xR = xL + dx
yR = yL + dy

# Valid mask: correspondences within image bounds
valid = (xR >= 0) & (xR < w_roi) & (yR >= 0) & (yR < h_roi)
# Disparity magnitude filter (both components matter)
disp_mag = np.sqrt(dx**2 + dy**2)
valid &= (disp_mag < min(w_roi, h_roi) * 0.3)

# Flow gives dx = pixel displacement from left to right at each (x,y) in left image
dx = flow[..., 0]  # horizontal disparity
dy = flow[..., 1]

# Right pixels = left_pixel + flow
xL, yL = np.meshgrid(np.arange(w_roi), np.arange(h_roi))
xR = xL + dx
yR = yL + dy

# Valid mask: correspondences within image bounds and reasonable disparity
valid = (xR >= 0) & (xR < w_roi) & (yR >= 0) & (yR < h_roi)
# Disparity magnitude filter
disp_mag = np.sqrt(dx**2 + dy**2)
valid &= (disp_mag < w_roi * 0.3)  # max 30% of width
n_valid = valid.sum()
print(f"  Valid correspondences: {n_valid}/{w_roi*h_roi} ({100*n_valid/(w_roi*h_roi):.1f}%)")

# Map back to full image coordinates
uL = xL[valid] + roi_x0
vL = yL[valid] + roi_y0
uR = xR[valid] + roi_x0
vR = yR[valid] + roi_y0

# Save correspondences
np.savez_compressed(OUT / 'specimen_correspondences.npz',
    uL=uL.astype(np.float32), vL=vL.astype(np.float32),
    uR=uR.astype(np.float32), vR=vR.astype(np.float32),
    image_size=(W, H), roi=[roi_x0, roi_x1, roi_y0, roi_y1])

summary = {
    'image_size': [W, H], 'roi': [roi_x0, roi_x1, roi_y0, roi_y1],
    'n_correspondences': int(n_valid),
    'pct_valid': float(100*n_valid/(w_roi*h_roi)),
    'median_dx_px': float(np.median(dx[valid])),
    'median_dy_px': float(np.median(dy[valid])),
    'flow_method': 'DIS' if 'dis' in dir() else 'Farneback',
}
with open(OUT / 'specimen_correspondences_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved correspondences: {OUT / 'specimen_correspondences.npz'}")

# ═══════════════════════════════════════════════════════════
# 3 — Triangulation with CMO 26p model
# ═══════════════════════════════════════════════════════════
print("\nTriangulating with CMO 26p model...")
data = np.load('docs/assets/pycaso_real_data/intermediate_state.npz')
IMG_SIZE = tuple(data['image_size'])

def load_cmo_26p():
    with open('docs/assets/pycaso_real_data/aligned_cmo_fit.json') as f:
        ad = json.load(f)
    a26 = ad['aligned_26p']
    # Build from 14 telecentric params
    # The x_26p vector is in intermediate_state.npz
    x_26p = data['x_26p']
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(x_26p[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
    rv_L = x_26p[14:17]; t_L = x_26p[17:20]
    rv_R = x_26p[20:23]; t_R = x_26p[23:26]
    return m_tel, rv_L, t_L, rv_R, t_R

def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    O_new = (R @ O.T).T + t[None, :]
    d_new = _normalize((R @ d.T).T)
    return O_new, d_new

def triangulate_rays(O1, d1, O2, d2):
    """Find closest point between two skew lines. Returns midpoint and gap."""
    n = np.cross(d1, d2, axis=1)
    n_norm = np.linalg.norm(n, axis=1)
    valid_line = n_norm > 1e-12

    # Solve: O1 + t1*d1 = O2 + t2*d2 (approximately)
    # t1 = det(O2-O1, d2, d1×d2) / |d1×d2|²
    w = O2 - O1
    cross_d2_n = np.cross(d2, n, axis=1)
    cross_d1_n = np.cross(d1, n, axis=1)

    t1 = np.sum(w * cross_d2_n, axis=1) / n_norm**2
    t2 = np.sum(w * cross_d1_n, axis=1) / n_norm**2

    P1 = O1 + t1[:, None] * d1
    P2 = O2 + t2[:, None] * d2
    midpoint = (P1 + P2) / 2
    gap = np.linalg.norm(P1 - P2, axis=1)

    return midpoint, gap, valid_line

# CMO 26p triangulation
m_tel, rv_L, t_L, rv_R, t_R = load_cmo_26p()
OL_raw, dL_raw = m_tel.ray(uL, vL, "left")
OR_raw, dR_raw = m_tel.ray(uR, vR, "right")
OL, dL = apply_se3(OL_raw, dL_raw, rv_L, t_L)
OR, dR = apply_se3(OR_raw, dR_raw, rv_R, t_R)

P_cmo, gap_cmo, valid_cmo = triangulate_rays(OL, dL, OR, dR)
Z_cmo = P_cmo[:, 2]
n_valid_cmo = valid_cmo.sum()
print(f"  CMO 26p: {n_valid_cmo}/{len(uL)} valid ({100*n_valid_cmo/len(uL):.1f}%), median gap={np.median(gap_cmo[valid_cmo]):.4f} mm")

np.savez_compressed(OUT / 'specimen_reconstruction_cmo26.npz',
    X=P_cmo[:,0].astype(np.float32), Y=P_cmo[:,1].astype(np.float32), Z=Z_cmo.astype(np.float32),
    gap=gap_cmo.astype(np.float32), valid=valid_cmo)

# Zernike triangulation
print("Triangulating with Zernike rayfield...")
from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients
with open('docs/assets/pycaso_real_data/zernike_pose_variants.json') as f:
    zv = json.load(f)
cc = zv['zernike_constrained']

def arr(x): return np.asarray(x, dtype=np.float64).reshape(-1, 3)
config = ZernikeOriginFieldConfig(image_size=IMG_SIZE, max_order=2)
lf = ZernikeRayField(K=np.array([[25600,0,1024],[0,25600,1024],[0,0,1]], dtype=np.float64), config=config,
    coefficients=ZernikeRayFieldCoefficients(origin_coeffs=arr(cc['left_origin_coeffs']), direction_coeffs=arr(cc['left_direction_coeffs'])))
rf = ZernikeRayField(K=np.array([[25600,0,1024],[0,25600,1024],[0,0,1]], dtype=np.float64), config=config,
    coefficients=ZernikeRayFieldCoefficients(origin_coeffs=arr(cc['right_origin_coeffs']), direction_coeffs=arr(cc['right_direction_coeffs'])))

Oz_L, dz_L = lf.ray(uL, vL)
Oz_R, dz_R = rf.ray(uR, vR)
P_zer, gap_zer, valid_zer = triangulate_rays(Oz_L, dz_L, Oz_R, dz_R)
Z_zer = P_zer[:, 2]
n_valid_zer = valid_zer.sum()
print(f"  Zernike: {n_valid_zer}/{len(uL)} valid ({100*n_valid_zer/len(uL):.1f}%), median gap={np.median(gap_zer[valid_zer]):.4f} mm")

np.savez_compressed(OUT / 'specimen_reconstruction_zernike.npz',
    X=P_zer[:,0].astype(np.float32), Y=P_zer[:,1].astype(np.float32), Z=Z_zer.astype(np.float32),
    gap=gap_zer.astype(np.float32), valid=valid_zer)

# ── Metrics ──
Z_valid_cmo = Z_cmo[valid_cmo]
Z_valid_zer = Z_zer[valid_zer]
roughness_cmo = float(np.median(np.abs(Z_valid_cmo - np.median(Z_valid_cmo))))
roughness_zer = float(np.median(np.abs(Z_valid_zer - np.median(Z_valid_zer))))

metrics = {
    'cmo_26p': {
        'valid_fraction': float(valid_cmo.sum()/len(valid_cmo)),
        'median_ray_gap_mm': float(np.median(gap_cmo[valid_cmo])),
        'median_Z_mm': float(np.median(Z_valid_cmo)),
        'Z_mad_mm': roughness_cmo,
        'Z_range_mm': [float(Z_valid_cmo.min()), float(Z_valid_cmo.max())],
    },
    'zernike_57p': {
        'valid_fraction': float(valid_zer.sum()/len(valid_zer)),
        'median_ray_gap_mm': float(np.median(gap_zer[valid_zer])),
        'median_Z_mm': float(np.median(Z_valid_zer)),
        'Z_mad_mm': roughness_zer,
        'Z_range_mm': [float(Z_valid_zer.min()), float(Z_valid_zer.max())],
    },
}
with open(OUT / 'specimen_reconstruction_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics: CMO Z_MAD={roughness_cmo:.4f} mm, Zernike Z_MAD={roughness_zer:.4f} mm")

# ═══════════════════════════════════════════════════════════
# 4 — Figure: 2×3 reconstruction
# ═══════════════════════════════════════════════════════════
print("\nGenerating figure...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 0: left image, disparity, valid mask
axes[0,0].imshow(imgL, cmap='gray'); axes[0,0].set_title('Left image (coin specimen)')
axes[0,0].set_xlabel('u (px)'); axes[0,0].set_ylabel('v (px)')

disp_map = np.full((h_roi, w_roi), np.nan)
disp_map[valid.reshape(h_roi, w_roi)] = dx[valid.reshape(h_roi, w_roi)]
im1 = axes[0,1].imshow(disp_map, cmap='RdYlBu', origin='lower')
axes[0,1].set_title('Horizontal disparity (px)'); plt.colorbar(im1, ax=axes[0,1])

mask_viz = np.zeros((h_roi, w_roi))
mask_viz[valid.reshape(h_roi, w_roi)] = 1
axes[0,2].imshow(mask_viz, cmap='gray', origin='lower')
axes[0,2].set_title(f'Valid mask ({100*valid.sum()/valid.size:.0f}%)')

# Row 1: Z maps
for ax, Z_data, val_mask, label in [
    (axes[1,0], Z_cmo, valid_cmo, 'CMO 26p'),
    (axes[1,1], Z_zer, valid_zer, 'Zernike 57p'),
]:
    z_map = np.full(valid.shape, np.nan)
    z_map[valid] = np.nan  # init all with nan
    z_vals = np.full(valid.shape, np.nan)
    # valid is 1D boolean, Z_data is 1D
    z_vals[valid] = Z_data  # fill where flow was valid
    z_map = z_vals.reshape(h_roi, w_roi)
    z_med = np.nanmedian(z_map)
    vmin = max(z_med - 2, np.nanmin(z_map))
    vmax = min(z_med + 2, np.nanmax(z_map))
    im = ax.imshow(z_map, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f'{label} Z (mm)'); plt.colorbar(im, ax=ax)
    ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')

# Gap histogram
axes[1,2].hist(gap_cmo[valid_cmo], bins=50, alpha=0.6, label=f'CMO 26p (med={np.median(gap_cmo[valid_cmo]):.3f} mm)')
axes[1,2].hist(gap_zer[valid_zer], bins=50, alpha=0.6, label=f'Zernike (med={np.median(gap_zer[valid_zer]):.3f} mm)')
axes[1,2].set_xlabel('Ray gap (mm)'); axes[1,2].set_ylabel('Count')
axes[1,2].set_title('Triangulation ray gap'); axes[1,2].legend()
axes[1,2].set_xlim(0, 0.2)

fig.suptitle('Pycaso coin specimen — dense stereo reconstruction', fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(Path('paper/cmo/figures/specimen_reconstruction.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Figure saved: paper/cmo/figures/specimen_reconstruction.png")
print("Done!")
