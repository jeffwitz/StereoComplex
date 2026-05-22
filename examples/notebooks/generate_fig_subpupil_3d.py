#!/usr/bin/env python3
"""3D sub-pupil reconstruction from CMO rayfield (Figure 4).

Reconstructs the effective sub-pupil positions O_L and O_R from the fitted 26p
CMO model, draws the chief rays converging at the working plane, and annotates
the baseline b=||O_R-O_L||, working distance WD, and convergence angle theta.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, 'src')
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

# Load data
data = np.load('docs/assets/pycaso_real_data/intermediate_state.npz')
x_26p = data['x_26p']
IMG_SIZE = tuple(data['image_size'])
W, H = IMG_SIZE

# Build CMO model
m_tel = CMOTelecentricStereoModel.from_parameter_vector(x_26p[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)

# Get centre-pixel rays (sub-pupil definitions)
OL, dL = m_tel.ray(np.array([W/2]), np.array([H/2]), "left")
OR, dR = m_tel.ray(np.array([W/2]), np.array([H/2]), "right")

# Apply SE(3) arm alignment
rv_L = x_26p[14:17]; t_L = x_26p[17:20]
rv_R = x_26p[20:23]; t_R = x_26p[23:26]

def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

OL_a, dL_a = apply_se3(OL, dL, rv_L, t_L)
OR_a, dR_a = apply_se3(OR, dR, rv_R, t_R)

OcL = OL_a[0]
OcR = OR_a[0]
dcL = dL_a[0]
dcR = dR_a[0]

# Descriptors
b = float(np.linalg.norm(OcR - OcL))
WD_est = float(np.mean([float(data["opt_t"][i][2]) for i in range(10)]))
theta = float(np.degrees(np.arctan2(b/2, WD_est)))

# Chief ray intersection with working plane
tL_w = (WD_est - OcL[2]) / dcL[2] if abs(dcL[2]) > 1e-12 else WD_est
tR_w = (WD_est - OcR[2]) / dcR[2] if abs(dcR[2]) > 1e-12 else WD_est
PL_w = OcL + tL_w * dcL
PR_w = OcR + tR_w * dcR

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sub-pupil positions
ax.scatter(*OcL, c='#2a55c7', s=200, marker='o', edgecolors='black', linewidth=0.5, label=f'O$_L$ ({OcL[0]:.1f}, {OcL[1]:.1f}, {OcL[2]:.1f})')
ax.scatter(*OcR, c='#c4332a', s=200, marker='o', edgecolors='black', linewidth=0.5, label=f'O$_R$ ({OcR[0]:.1f}, {OcR[1]:.1f}, {OcR[2]:.1f})')

# Chief rays (from sub-pupils toward working plane)
ray_len = 65
ray_L_end = OcL + ray_len * dcL
ray_R_end = OcR + ray_len * dcR
ax.plot([OcL[0], ray_L_end[0]], [OcL[1], ray_L_end[1]], [OcL[2], ray_L_end[2]],
        color='#2a55c7', lw=1.5, alpha=0.7, label='Chief ray L')
ax.plot([OcR[0], ray_R_end[0]], [OcR[1], ray_R_end[1]], [OcR[2], ray_R_end[2]],
        color='#c4332a', lw=1.5, alpha=0.7, label='Chief ray R')

# Working plane intersections
ax.scatter(*PL_w, c='green', s=120, marker='x', linewidth=1.5, label=f'Working plane hit L (z={WD_est:.1f})')
ax.scatter(*PR_w, c='green', s=120, marker='x', linewidth=1.5)

# Baseline line
ax.plot([OcL[0], OcR[0]], [OcL[1], OcR[1]], [OcL[2], OcR[2]],
        'k--', lw=2, alpha=0.6)
# Baseline midpoint annotation
bc = (OcL + OcR) / 2
ax.text(bc[0], bc[1], bc[2] - 2, f'b = {b:.1f} mm', fontsize=12, ha='center', fontweight='bold')

# Working plane (horizontal line at z=WD_est, spanning the x-range of chief ray hits)
wp_x = np.linspace(min(PL_w[0], PR_w[0]) - 5, max(PL_w[0], PR_w[0]) + 5, 2)
wp_y = np.linspace(min(PL_w[1], PR_w[1]) - 5, max(PL_w[1], PR_w[1]) + 5, 2)
wp_z = WD_est * np.ones(2)
ax.plot(wp_x, wp_y, wp_z, 'gray', lw=1, alpha=0.3, ls='--')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'3D sub-pupil reconstruction\nb={b:.1f} mm, WD={WD_est:.1f} mm, $\\theta$={theta:.1f}°')

ax.legend(fontsize=9, loc='upper left')

# Set reasonable viewing angle
ax.view_init(elev=20, azim=-60)

fig.tight_layout()
fig.savefig(OUT / 'subpupil_3d.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('subpupil_3d.pdf OK')
