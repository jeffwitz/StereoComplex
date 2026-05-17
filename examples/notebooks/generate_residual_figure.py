#!/usr/bin/env python3
"""Generate residual evolution figure for CMO paper (Figure 3)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, 'src')

# Journal-quality: serif font, larger sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

data = np.load('docs/assets/pycaso_real_data/intermediate_state.npz')
left_pixels, right_pixels = data["left_pixels"], data["right_pixels"]
obj_pts = data["obj_pts"]
opt_R_z, opt_t_z = data["opt_R"], data["opt_t"]
IMG_SIZE = tuple(data["image_size"]); W, H = IMG_SIZE
FX = float(data["FX"])

import cv2; from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)
n_frames = 10

# ── Zernike fit ──
left_px = [left_pixels[pi] for pi in range(n_frames)]
right_px = [right_pixels[pi] for pi in range(n_frames)]
rvecs, tvecs = [], []
for lp in left_px:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3)); tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0.,0.,65.]))
obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
                             left_pixels=left_px, right_pixels=right_px, point_indices=[np.arange(165,dtype=int) for _ in range(n_frames)],
                             noise_std_px=0.0, image_size=IMG_SIZE)
lf, rf, zd, _, _ = fit_constrained_zernike_rayfield(obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(), max_order_d=2, max_nfev=300, origin_reg_weight=0.0)

# ── Grid ──
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf)
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
WD_est = float(np.mean([float(opt_t_z[i][2]) for i in range(n_frames)]))
z_p = float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
f_obj_est = WD_est - z_p

def angle_map(d_model):
    dot = np.clip(np.sum(dzL * d_model, axis=1), -1, 1)
    return np.degrees(np.arccos(dot)).reshape(41, 41)

# ── Stage 1: Perspective CMO ──
pp = 0.0055 / f_obj_est
dL_persp = _normalize(np.column_stack([
    (uf - 1024)*pp - b_est/(2*f_obj_est), (vf - 1024)*pp, np.ones_like(uf)]))
r1 = angle_map(dL_persp)

# ── Stage 2: Telecentric CMO ──
theta_fixed = float(np.arctan2(b_est/2, f_obj_est))
x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dcL[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)

def res_tel(x):
    m = CMOTelecentricStereoModel.from_parameter_vector(x, pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
    OL, dL = m.ray(uf, vf, "left"); OR, dR = m.ray(uf, vf, "right")
    OzR, dzR = rf.ray(uf, vf)
    blocks = []
    for z in [50.,80.]:
        for O_a,d_a,O_r,d_r in [(OL,dL,OzL,dzL),(OR,dR,OzR,dzR)]:
            tz_ref=(z-O_r[:,2])/d_r[:,2];P_ref=O_r+tz_ref[:,None]*d_r
            tz_mod=(z-O_a[:,2])/d_a[:,2];P_mod=O_a+tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)

sol_tel = least_squares(res_tel, x0=x0_tel, bounds=(lo_tel,hi_tel), loss="huber",max_nfev=300,xtol=1e-10)
m_tel = CMOTelecentricStereoModel.from_parameter_vector(sol_tel.x[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
_, dL_tel = m_tel.ray(uf, vf, "left")
r2 = angle_map(dL_tel)

# ── Stage 3: CMO + SE(3) ──
def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None,:], _normalize((R @ d.T).T)

rot_lo=np.full(3,-0.08);rot_hi=np.full(3,0.08);trans_lo=np.full(3,-3.);trans_hi=np.full(3,3.)
arm_lo=np.concatenate([rot_lo,trans_lo,rot_lo,trans_lo]);arm_hi=np.concatenate([rot_hi,trans_hi,rot_hi,trans_hi])

def res_se3(x):
    m=CMOTelecentricStereoModel.from_parameter_vector(x[:14],pixel_pitch_mm=0.0055,image_size=IMG_SIZE)
    OL,dL=m.ray(uf,vf,"left");OR,dR=m.ray(uf,vf,"right")
    OL_a,dL_a=apply_se3(OL,dL,x[14:17],x[17:20]);OR_a,dR_a=apply_se3(OR,dR,x[20:23],x[23:26])
    OzR,dzR=rf.ray(uf,vf)
    blocks=[]
    for z in [50.,80.]:
        for O_a,d_a,O_r,d_r in [(OL_a,dL_a,OzL,dzL),(OR_a,dR_a,OzR,dzR)]:
            tz_ref=(z-O_r[:,2])/d_r[:,2];P_ref=O_r+tz_ref[:,None]*d_r
            tz_mod=(z-O_a[:,2])/d_a[:,2];P_mod=O_a+tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)

sol_se3=least_squares(res_se3,x0=np.concatenate([sol_tel.x[:14],np.zeros(12)]),
    bounds=(np.concatenate([lo_tel,arm_lo]),np.concatenate([hi_tel,arm_hi])),
    loss="huber",max_nfev=300,xtol=1e-10)
m_se3=CMOTelecentricStereoModel.from_parameter_vector(sol_se3.x[:14],pixel_pitch_mm=0.0055,image_size=IMG_SIZE)
OL_s,dL_s=m_se3.ray(uf,vf,"left")
_,dL_se3=apply_se3(OL_s,dL_s,sol_se3.x[14:17],sol_se3.x[17:20])
r3=angle_map(dL_se3)

# ── Stage 0: Zernike self ──
r0=angle_map(dzL)

# ── Plot: 2x2 with individual colorbars ──
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
stages = [
    (axes[0,0], r1, "(a) Perspective CMO", 0, 30,
     rf"RMS={np.sqrt(np.mean(r1**2)):.1f}°"),
    (axes[0,1], r2, "(b) Telecentric CMO", 0, 0.5,
     rf"RMS={np.sqrt(np.mean(r2**2)):.3f}°"),
    (axes[1,0], r3, "(c) CMO + SE(3)", 0, 0.01,
     rf"RMS={np.sqrt(np.mean(r3**2)):.3f}°"),
    (axes[1,1], r0, "(d) Zernike reference (self)", 0, 0.001,
     rf"RMS={np.sqrt(np.mean(r0**2)):.0e}°"),
]

for ax, arr, title, vmin, vmax, rms_label in stages:
    im = ax.imshow(arr, origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                    extent=[0, W, 0, H], aspect='auto')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('u (px)')
    ax.set_ylabel('v (px)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Direction error (deg)', fontsize=10)
    ax.text(0.98, 0.02, rms_label, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Residual-guided model construction: direction error '
             r'$\Delta d(u,v) = \angle(d_{\mathrm{Zernike}}, d_{\mathrm{model}})$',
             fontsize=15, y=1.01)
fig.tight_layout()

out = Path('paper/cmo/figures/residual_evolution.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
print(f"  Persp RMS={np.sqrt(np.mean(r1**2)):.1f}°  "
      f"Telecentric RMS={np.sqrt(np.mean(r2**2)):.3f}°  "
      f"CMO+SE3 RMS={np.sqrt(np.mean(r3**2)):.4f}°")
plt.close()
