#!/usr/bin/env python3
"""Compare Zernike 57p vs CMO 26p specimen reconstructions after removing rigid-body effects.

7 phases:
  1. Load data, subsample 100k common-valid correspondences
  2. Kabsch SE(3) Zernike→CMO on matched 3D points
  3. Affine plane fit to ΔZ before/after SE(3)
  4. 2×4 figure: Z_CMO, Z_Zernike, ΔZ raw, affine plane, Z_aligned, ΔZ post-SE3, ΔZ post-SE3+plane, histogram
  5. Plücker ray bundle comparison: common + per-arm SE(3)
  6. Save JSON metrics
  7. Interpretation: A (global frame), B (per-arm), C (affine ramp), D (non-rigid residual)
"""

import json, time, sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, 'src')
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients,
)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10})

ASSETS = Path('docs/assets/pycaso_real_data')
FIGDIR = Path('paper/cmo/figures')
SEED = 42

# ═══════════════════════════════════════════════════════════════════
# Phase 1 — Load data
# ═══════════════════════════════════════════════════════════════════
print('=' * 60)
print('Phase 1 — Loading data')
print('=' * 60)

cmo = dict(np.load(ASSETS / 'specimen_reconstruction_cmo26.npz', allow_pickle=True))
zer = dict(np.load(ASSETS / 'specimen_reconstruction_zernike.npz', allow_pickle=True))
corr = dict(np.load(ASSETS / 'specimen_correspondences.npz', allow_pickle=True))

P_cmo = np.column_stack([cmo['X'], cmo['Y'], cmo['Z']])
P_zer = np.column_stack([zer['X'], zer['Y'], zer['Z']])
# The CMO model fits the Pycaso rayfield with an inverted v->Y sign (rho_y, s_y < 0),
# so its reconstruction is Y-mirrored relative to the Zernike reference. Apply the
# documented -Y correction (cf. pycaso_schur_regularized_ba.py) BEFORE the rigid
# alignment: without it the CMO<->Zernike relation is a reflection that the proper-
# rotation Kabsch (det=+1) disguises as a ~180 deg rotation, which spuriously inverts
# the Z relief. With the correction the alignment is a genuine ~8 deg rotation and the
# reliefs agree in sign (only the depth-scale s_z=0.67 amplitude difference remains).
P_cmo[:, 1] *= -1.0
valid_cmo = cmo['valid']
valid_zer = zer['valid']

W_img, H_img = int(corr['image_size'][0]), int(corr['image_size'][1])
roi_x0, roi_x1, roi_y0, roi_y1 = [int(x) for x in corr['roi']]
h_roi, w_roi = roi_y1 - roi_y0, roi_x1 - roi_x0

valid_both = valid_cmo & valid_zer
n_common = valid_both.sum()
print(f'  Common valid points: {n_common}/{len(valid_both)} ({100*n_common/len(valid_both):.1f}%)')

# Subsample to 100k
rng = np.random.default_rng(SEED)
idx_all = np.where(valid_both)[0]
n_sample = min(100_000, len(idx_all))
idx_sample = rng.choice(idx_all, size=n_sample, replace=False)
print(f'  Subsample: {n_sample} points')

P_cmo_s = P_cmo[idx_sample]
P_zer_s = P_zer[idx_sample]

# Pixel coordinates for sampled points (for 2D maps and ray queries)
uL_s = corr['uL'][idx_sample].astype(np.float64)
vL_s = corr['vL'][idx_sample].astype(np.float64)
uR_s = corr['uR'][idx_sample].astype(np.float64)
vR_s = corr['vR'][idx_sample].astype(np.float64)
u_roi = (uL_s - roi_x0).astype(np.int32)
v_roi = (vL_s - roi_y0).astype(np.int32)

# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Kabsch SE(3) Zernike→CMO on matched 3D points
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 2 — Kabsch SE(3) Zernike→CMO')
print('=' * 60)


def kabsch(a, b):
    """Fit no-scale SE(3) mapping a→b. Returns R (3,3), t (3,)."""
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_c = a - a_mean
    b_c = b - b_mean
    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = b_mean - R @ a_mean
    return R, t


R_kabsch, t_kabsch = kabsch(P_zer_s, P_cmo_s)
rotvec_kabsch = Rotation.from_matrix(R_kabsch).as_rotvec()
angle_kabsch = float(np.linalg.norm(rotvec_kabsch))
trans_norm_kabsch = float(np.linalg.norm(t_kabsch))

dP_before = np.linalg.norm(P_zer_s - P_cmo_s, axis=1)
P_zer_aligned = (R_kabsch @ P_zer_s.T).T + t_kabsch
dP_after = np.linalg.norm(P_zer_aligned - P_cmo_s, axis=1)

med_before = float(np.median(dP_before))
med_after = float(np.median(dP_after))

print(f'  Rotation angle: {np.degrees(angle_kabsch):.4f}°')
print(f'  Translation norm: {trans_norm_kabsch:.4f} mm')
print(f'  Median ||ΔP|| before SE(3): {med_before:.4f} mm')
print(f'  Median ||ΔP|| after SE(3):  {med_after:.4f} mm')
print(f'  Reduction: {100*(1 - med_after/med_before):.1f}%')

# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Affine plane fit to ΔZ
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 3 — Affine plane fit to ΔZ')
print('=' * 60)

Z_cmo_s = P_cmo_s[:, 2]
Z_zer_s = P_zer_s[:, 2]
Z_zer_aligned_s = P_zer_aligned[:, 2]
XY_s = P_cmo_s[:, :2]


def fit_affine_plane(xy, z):
    """Fit z = a*x + b*y + c. Returns (a,b,c), r2, mad, residuals."""
    A = np.column_stack([xy, np.ones(len(xy))])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    z_pred = A @ coeffs
    resid = z - z_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((z - z.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mad = float(np.median(np.abs(resid)))
    return coeffs, float(r2), mad, resid


dZ_before = Z_zer_s - Z_cmo_s
coeffs_before, r2_before, mad_before_plane, resid_before = fit_affine_plane(XY_s, dZ_before)

dZ_after_se3 = Z_zer_aligned_s - Z_cmo_s
coeffs_after, r2_after, mad_after_plane, resid_after_se3 = fit_affine_plane(XY_s, dZ_after_se3)

# Remove affine plane from post-SE(3) ΔZ
dZ_after_se3_plane = resid_after_se3

print(f'  Before SE(3): R²={r2_before:.4f}, MAD={mad_before_plane:.4f} mm')
print(f'  After SE(3):  R²={r2_after:.4f}, MAD={mad_after_plane:.4f} mm')

# ═══════════════════════════════════════════════════════════════════
# Phase 4 — 2×4 figure
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 4 — 2×4 figure')
print('=' * 60)


def make_2d_map(values_1d, fill=np.nan):
    """Map 1D sampled values back to (h_roi, w_roi) grid."""
    grid = np.full((h_roi, w_roi), fill, dtype=np.float64)
    grid[v_roi, u_roi] = values_1d
    return grid


fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# Row 0
im00 = axes[0, 0].imshow(make_2d_map(Z_cmo_s), cmap='viridis', origin='lower')
axes[0, 0].set_title('$Z_\\mathrm{CMO}$ (mm)')
plt.colorbar(im00, ax=axes[0, 0])

im01 = axes[0, 1].imshow(make_2d_map(Z_zer_s), cmap='viridis', origin='lower')
axes[0, 1].set_title('$Z_\\mathrm{Zernike}$ (mm)')
plt.colorbar(im01, ax=axes[0, 1])

im02 = axes[0, 2].imshow(make_2d_map(dZ_before), cmap='RdBu_r', origin='lower',
                          vmin=-np.percentile(np.abs(dZ_before), 99),
                          vmax=np.percentile(np.abs(dZ_before), 99))
axes[0, 2].set_title('$\\Delta Z$ raw (mm)')
plt.colorbar(im02, ax=axes[0, 2])

# Affine plane fit visualization
Z_plane_pred = XY_s @ coeffs_before[:2] + coeffs_before[2]
im03 = axes[0, 3].imshow(make_2d_map(Z_plane_pred), cmap='RdBu_r', origin='lower',
                          vmin=-np.percentile(np.abs(Z_plane_pred), 99),
                          vmax=np.percentile(np.abs(Z_plane_pred), 99))
axes[0, 3].set_title(f'Affine plane ($R^2={r2_before:.3f}$)')

# Row 1
im10 = axes[1, 0].imshow(make_2d_map(Z_zer_aligned_s), cmap='viridis', origin='lower')
axes[1, 0].set_title('$Z_\\mathrm{aligned}$ (mm)')
plt.colorbar(im10, ax=axes[1, 0])

im11 = axes[1, 1].imshow(make_2d_map(dZ_after_se3), cmap='RdBu_r', origin='lower',
                          vmin=-np.percentile(np.abs(dZ_after_se3), 99),
                          vmax=np.percentile(np.abs(dZ_after_se3), 99))
axes[1, 1].set_title('$\\Delta Z$ after SE(3) (mm)')
plt.colorbar(im11, ax=axes[1, 1])

im12 = axes[1, 2].imshow(make_2d_map(dZ_after_se3_plane), cmap='RdBu_r', origin='lower',
                          vmin=-np.percentile(np.abs(dZ_after_se3_plane), 99),
                          vmax=np.percentile(np.abs(dZ_after_se3_plane), 99))
axes[1, 2].set_title('$\\Delta Z$ after SE(3)+plane (mm)')
plt.colorbar(im12, ax=axes[1, 2])

# Histogram of 3D residuals
ax_hist = axes[1, 3]
ax_hist.hist(dP_before, bins=80, alpha=0.5, density=True,
             label=f'Before SE(3) (med={med_before:.3f})')
ax_hist.hist(dP_after, bins=80, alpha=0.5, density=True,
             label=f'After SE(3) (med={med_after:.3f})')
ax_hist.set_xlabel('$||\\Delta P||_2$ (mm)')
ax_hist.set_ylabel('Density')
ax_hist.set_title('3D point residuals')
ax_hist.legend()
xlim99 = np.percentile(np.concatenate([dP_before, dP_after]), 99.5)
ax_hist.set_xlim(0, xlim99)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel('u (px)')
        ax.set_ylabel('v (px)')

fig.suptitle('Zernike 57p vs CMO 26p — rigid-body removal', fontweight='bold', fontsize=14)
fig.tight_layout()
fig.savefig(FIGDIR / 'zernike_cmo_rigid_removed.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  Saved: {FIGDIR / "zernike_cmo_rigid_removed.png"}')

# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Plücker ray bundle comparison
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 5 — Plücker ray bundle comparison')
print('=' * 60)

# Load CMO 26p model
data = np.load(ASSETS / 'intermediate_state.npz', allow_pickle=True)
IMG_SIZE = tuple(int(x) for x in data['image_size'])
x_26p = data['x_26p']

with open(ASSETS / 'aligned_cmo_fit.json') as f:
    ad = json.load(f)
a26 = ad['aligned_26p']

m_tel = CMOTelecentricStereoModel.from_parameter_vector(
    x_26p[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
rv_L = x_26p[14:17]
t_L_cmo = x_26p[17:20]
rv_R = x_26p[20:23]
t_R_cmo = x_26p[23:26]

# Load Zernike model
with open(ASSETS / 'zernike_pose_variants.json') as f:
    zv = json.load(f)
cc = zv['zernike_constrained']


def arr(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 3)


K = np.array([[25600, 0, 1024], [0, 25600, 1024], [0, 0, 1]], dtype=np.float64)
config = ZernikeOriginFieldConfig(image_size=IMG_SIZE, max_order=2)
lf = ZernikeRayField(K=K, config=config,
                     coefficients=ZernikeRayFieldCoefficients(
                         origin_coeffs=arr(cc['left_origin_coeffs']),
                         direction_coeffs=arr(cc['left_direction_coeffs'])))
rf = ZernikeRayField(K=K, config=config,
                     coefficients=ZernikeRayFieldCoefficients(
                         origin_coeffs=arr(cc['right_origin_coeffs']),
                         direction_coeffs=arr(cc['right_direction_coeffs'])))

# Compute rays for sampled points
print('  Computing CMO rays...')
t0 = time.time()
OL_raw, dL_raw = m_tel.ray(uL_s, vL_s, 'left')
OR_raw, dR_raw = m_tel.ray(uR_s, vR_s, 'right')

RL_cmo = Rotation.from_rotvec(rv_L).as_matrix()
RR_cmo = Rotation.from_rotvec(rv_R).as_matrix()
OL_cmo = (RL_cmo @ OL_raw.T).T + t_L_cmo[None, :]
dL_cmo = _normalize((RL_cmo @ dL_raw.T).T)
OR_cmo = (RR_cmo @ OR_raw.T).T + t_R_cmo[None, :]
dR_cmo = _normalize((RR_cmo @ dR_raw.T).T)
print(f'    {time.time() - t0:.1f}s')

print('  Computing Zernike rays...')
t0 = time.time()
OL_zer, dL_zer = lf.ray(uL_s, vL_s)
OR_zer, dR_zer = rf.ray(uR_s, vR_s)
print(f'    {time.time() - t0:.1f}s')


def plucker(O, d):
    """Plücker coordinates: (direction, moment = O × d)."""
    m = np.cross(O, d)
    return d, m


def fit_se3_plucker(d_src, m_src, d_tgt, m_tgt):
    """Fit SE(3) minimizing direction + moment error in Plücker space.

    Returns R (3,3), t (3,), angle_deg, trans_norm, dir_resid_before, dir_resid_after,
    moment_resid_before, moment_resid_after.
    """
    # Rotation from directions (Kabsch on unit vectors)
    H = d_src.T @ d_tgt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    dir_resid_before = float(np.median(np.linalg.norm(d_src - d_tgt, axis=1)))
    d_src_rot = (R @ d_src.T).T
    dir_resid_after = float(np.median(np.linalg.norm(d_src_rot - d_tgt, axis=1)))

    # Translation from moments: minimize ||R*m_src + t × (R*d_src) - m_tgt||
    # t × v = -[v]_× t, so residual is R*m_src - m_tgt - [R*d_src]_× t
    # => [R*d_src]_× t ≈ R*m_src - m_tgt
    m_src_rot = (R @ m_src.T).T
    d_src_rot = (R @ d_src.T).T  # already computed above, but keep for clarity

    # Build normal equations for 3×3 system
    ATA = np.zeros((3, 3))
    ATb = np.zeros(3)
    residual = m_src_rot - m_tgt
    for i in range(len(d_src_rot)):
        skew = np.array([[0, -d_src_rot[i, 2], d_src_rot[i, 1]],
                         [d_src_rot[i, 2], 0, -d_src_rot[i, 0]],
                         [-d_src_rot[i, 1], d_src_rot[i, 0], 0]])
        ATA += skew.T @ skew
        ATb += skew.T @ residual[i]

    try:
        t = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        t = np.linalg.lstsq(ATA, ATb, rcond=None)[0]

    # Compute residuals
    moment_resid_before = float(np.median(np.linalg.norm(m_src - m_tgt, axis=1)))
    m_pred = m_src_rot + np.cross(t[None, :], d_src_rot)
    moment_resid_after = float(np.median(np.linalg.norm(m_pred - m_tgt, axis=1)))

    angle_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
    trans_norm = float(np.linalg.norm(t))

    return R, t, angle_deg, trans_norm, dir_resid_before, dir_resid_after, moment_resid_before, moment_resid_after


# Combine left and right rays
d_zer_all = np.vstack([dL_zer, dR_zer])
m_zer_all = np.vstack([plucker(OL_zer, dL_zer)[1], plucker(OR_zer, dR_zer)[1]])
d_cmo_all = np.vstack([dL_cmo, dR_cmo])
m_cmo_all = np.vstack([plucker(OL_cmo, dL_cmo)[1], plucker(OR_cmo, dR_cmo)[1]])

# Left Plücker
dL_zer_m, mL_zer = plucker(OL_zer, dL_zer)
dL_cmo_m, mL_cmo = plucker(OL_cmo, dL_cmo)
# Right Plücker
dR_zer_m, mR_zer = plucker(OR_zer, dR_zer)
dR_cmo_m, mR_cmo = plucker(OR_cmo, dR_cmo)

# Common SE(3)
print('  Fitting common SE(3)...')
(R_common, t_common, ang_common, tnorm_common,
 dir_before_common, dir_after_common, mom_before_common, mom_after_common) = \
    fit_se3_plucker(d_zer_all, m_zer_all, d_cmo_all, m_cmo_all)

print(f'  Common SE(3):')
print(f'    Rotation: {ang_common:.4f}°')
print(f'    Translation: {tnorm_common:.4f} mm')
print(f'    Direction residual: {dir_before_common:.6f} → {dir_after_common:.6f}')
print(f'    Moment residual:    {mom_before_common:.4f} → {mom_after_common:.4f} mm')

# Per-arm SE(3)
print('  Fitting per-arm SE(3)...')
(R_L, t_L, ang_L, tnorm_L, dir_before_L, dir_after_L, mom_before_L, mom_after_L) = \
    fit_se3_plucker(dL_zer_m, mL_zer, dL_cmo_m, mL_cmo)

(R_R, t_R, ang_R, tnorm_R, dir_before_R, dir_after_R, mom_before_R, mom_after_R) = \
    fit_se3_plucker(dR_zer_m, mR_zer, dR_cmo_m, mR_cmo)

print(f'  Left SE(3):  rot={ang_L:.4f}°, trans={tnorm_L:.4f} mm, dir_resid={dir_before_L:.6f}→{dir_after_L:.6f}, mom_resid={mom_before_L:.4f}→{mom_after_L:.4f} mm')
print(f'  Right SE(3): rot={ang_R:.4f}°, trans={tnorm_R:.4f} mm, dir_resid={dir_before_R:.6f}→{dir_after_R:.6f}, mom_resid={mom_before_R:.4f}→{mom_after_R:.4f} mm')

# Per-arm combined metrics
per_arm_dir_after = float(np.median(np.concatenate([
    np.linalg.norm((R_L @ dL_zer_m.T).T - dL_cmo_m, axis=1),
    np.linalg.norm((R_R @ dR_zer_m.T).T - dR_cmo_m, axis=1),
])))
per_arm_mom_after = float(np.median(np.concatenate([
    np.linalg.norm((R_L @ mL_zer.T).T + np.cross(t_L[None, :], (R_L @ dL_zer_m.T).T) - mL_cmo, axis=1),
    np.linalg.norm((R_R @ mR_zer.T).T + np.cross(t_R[None, :], (R_R @ dR_zer_m.T).T) - mR_cmo, axis=1),
])))

print(f'  Per-arm combined: dir_resid_after={per_arm_dir_after:.6f}, mom_resid_after={per_arm_mom_after:.4f} mm')

# ═══════════════════════════════════════════════════════════════════
# Phase 6 — Save JSON metrics
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 6 — Save JSON metrics')
print('=' * 60)

results = {
    'n_common_valid': int(n_common),
    'n_subsample': int(n_sample),
    'kabsch_se3': {
        'rotation_angle_deg': float(np.degrees(angle_kabsch)),
        'translation_norm_mm': trans_norm_kabsch,
        'rotation_vec': rotvec_kabsch.tolist(),
        'translation': t_kabsch.tolist(),
        'median_dP_before_mm': med_before,
        'median_dP_after_mm': med_after,
        'reduction_pct': float(100 * (1 - med_after / med_before)),
    },
    'affine_plane': {
        'before_se3': {
            'coeffs_a_b_c': coeffs_before.tolist(),
            'r2': r2_before,
            'mad_mm': mad_before_plane,
        },
        'after_se3': {
            'coeffs_a_b_c': coeffs_after.tolist(),
            'r2': r2_after,
            'mad_mm': mad_after_plane,
        },
    },
    'plucker_common_se3': {
        'rotation_angle_deg': ang_common,
        'translation_norm_mm': tnorm_common,
        'direction_residual_before': dir_before_common,
        'direction_residual_after': dir_after_common,
        'moment_residual_before_mm': mom_before_common,
        'moment_residual_after_mm': mom_after_common,
    },
    'plucker_per_arm_se3': {
        'left': {
            'rotation_angle_deg': ang_L,
            'translation_norm_mm': tnorm_L,
            'direction_residual_before': dir_before_L,
            'direction_residual_after': dir_after_L,
            'moment_residual_before_mm': mom_before_L,
            'moment_residual_after_mm': mom_after_L,
        },
        'right': {
            'rotation_angle_deg': ang_R,
            'translation_norm_mm': tnorm_R,
            'direction_residual_before': dir_before_R,
            'direction_residual_after': dir_after_R,
            'moment_residual_before_mm': mom_before_R,
            'moment_residual_after_mm': mom_after_R,
        },
        'combined_direction_residual_after': per_arm_dir_after,
        'combined_moment_residual_after_mm': per_arm_mom_after,
    },
}

# ── Interpretation metrics ──
# Ratio: per-arm vs common direction improvement
dir_improvement_common = dir_before_common - dir_after_common
dir_improvement_per_arm = dir_before_common - per_arm_dir_after  # approximate
per_arm_vs_common_ratio = per_arm_dir_after / dir_after_common if dir_after_common > 1e-15 else 1.0

# Residual after all corrections
dZ_final_mad = float(np.median(np.abs(dZ_after_se3_plane)))

results['interpretation_metrics'] = {
    'kabsch_dP_reduction_pct': float(100 * (1 - med_after / med_before)),
    'affine_r2_after_se3': r2_after,
    'affine_mad_after_se3_mm': mad_after_plane,
    'dZ_final_mad_mm': dZ_final_mad,
    'common_se3_moment_residual_mm': mom_after_common,
    'per_arm_vs_common_direction_ratio': float(per_arm_vs_common_ratio),
    'per_arm_moment_residual_mm': per_arm_mom_after,
}

# ═══════════════════════════════════════════════════════════════════
# Phase 7 — Interpretation
# ═══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Phase 7 — Interpretation')
print('=' * 60)

# Case scores (0-1, higher = more consistent with case)
case_scores = {}

# Case A: global frame effect — common SE(3) explains most difference
# If per-arm doesn't help much vs common → strong Case A
case_a_score = 1.0 - min(1.0, max(0.0, (per_arm_vs_common_ratio - 0.5) / 0.5))
case_scores['A_global_frame'] = float(case_a_score)

# Case B: per-arm — per-arm SE(3) is significantly better than common
# If common-to-per-arm ratio is small → strong Case B
mom_improvement_ratio = mom_after_common / max(mom_after_common, per_arm_mom_after)
if per_arm_mom_after < mom_after_common * 0.7:
    case_b_score = 1.0 - per_arm_mom_after / mom_after_common
else:
    case_b_score = 0.0
case_scores['B_per_arm'] = float(max(0.0, min(1.0, case_b_score)))

# Case C: affine ramp — plane fit captures ΔZ
# High R² before SE(3) → planar trend
case_scores['C_affine_ramp'] = float(max(0.0, min(1.0, r2_before)))

# Case D: non-rigid residual — significant residual after all corrections
# Large residual relative to specimen depth range
Z_range = float(np.ptp(Z_cmo_s))
residual_fraction = dZ_final_mad / max(Z_range, 0.001)
case_d_score = min(1.0, residual_fraction / 0.05)  # normalize: 5% of depth range → score 1
case_scores['D_non_rigid'] = float(min(1.0, max(0.0, case_d_score)))

# Primary interpretation
primary = max(case_scores, key=case_scores.get)
interpretation_names = {
    'A_global_frame': 'Case A — Global frame effect (common SE(3) suffices)',
    'B_per_arm': 'Case B — Per-arm misalignment (separate SE(3) per arm needed)',
    'C_affine_ramp': 'Case C — Affine ramp (planar Z trend dominates residual)',
    'D_non_rigid': 'Case D — Non-rigid residual (genuine rayfield differences remain)',
}

results['case_scores'] = case_scores
results['primary_interpretation'] = primary

print(f'\n  Case scores:')
for case, score in case_scores.items():
    bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
    print(f'    {case}: {bar} {score:.3f}')
print(f'\n  Primary: {interpretation_names[primary]}')

out_path = ASSETS / 'zernike_cmo_rigid_comparison.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Saved: {out_path}')

# ── Detailed interpretation narrative ──
print('\n' + '─' * 60)
print('Interpretation narrative:')
print('─' * 60)

if med_after < 0.02:
    print(f'  • Kabsch SE(3) removes {results["kabsch_se3"]["reduction_pct"]:.1f}% of 3D point error')
    print(f'    → Zernike and CMO reconstructions agree to {med_after:.3f} mm after alignment')
else:
    print(f'  • Kabsch SE(3) reduces error by {results["kabsch_se3"]["reduction_pct"]:.1f}%')
    print(f'    → Residual {med_after:.3f} mm remains after rigid alignment')

if r2_before > 0.5:
    print(f'  • ΔZ shows strong planar trend (R²={r2_before:.3f}) → affine ramp present')
elif r2_before > 0.2:
    print(f'  • ΔZ shows weak planar trend (R²={r2_before:.3f})')
else:
    print(f'  • ΔZ shows no planar trend (R²={r2_before:.3f})')

if per_arm_vs_common_ratio < 0.5:
    print(f'  • Per-arm SE(3) is {per_arm_vs_common_ratio:.1f}× common residual → per-arm differences important')
elif per_arm_vs_common_ratio < 0.9:
    print(f'  • Per-arm SE(3) provides modest improvement ({per_arm_vs_common_ratio:.2f}×) over common')
else:
    print(f'  • Common SE(3) nearly optimal → global frame effect dominates')

print(f'\n  CONCLUSION: {interpretation_names[primary]}')
print(f'  (scores: {", ".join(f"{k}={v:.2f}" for k, v in case_scores.items())})')

print('\nDone.')
