#!/usr/bin/env python3
"""Residual evolution across model stages (Figure 6).

Three-panel heatmap showing direction error Δd (degrees) between the
Zernike rayfield (reference) and each candidate model on a 31×31 grid:
(a) Perspective CMO — large structured error (RMS ~23 deg),
(b) Telecentric CMO — RMS drops to ~0.27 deg (dominated by Z0 piston),
(c) CMO+SE(3) — RMS drops to ~0.003 deg, near the noise floor.

Requires re-running the optimization pipeline (~2-3 minutes).
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, 'src')
import cv2
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

# ── Load data ──
print("Loading data...")
data = np.load('docs/assets/pycaso_real_data/intermediate_state.npz')
W, H = tuple(data["image_size"])
FX = float(data["FX"])
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)
n_frames = 10

left_pixels = data["left_pixels"]
right_pixels = data["right_pixels"]
obj_pts = data["obj_pts"]
opt_t = data["opt_t"]

# ── Zernike fit (reference rayfield) ──
print("Fitting Zernike rayfield...")
t0 = time.time()
left_px = [left_pixels[pi] for pi in range(n_frames)]
right_px = [right_pixels[pi] for pi in range(n_frames)]
rvecs, tvecs = [], []
for lp in left_px:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32),
                              K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))
obs = CharucoObservationSet(
    object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
    left_pixels=left_px, right_pixels=right_px,
    point_indices=[np.arange(165, dtype=int) for _ in range(n_frames)],
    noise_std_px=0.0, image_size=tuple(data["image_size"]))
lf, rf, zd, _, _ = fit_constrained_zernike_rayfield(
    obs, image_size=tuple(data["image_size"]), K_left=K, K_right=K.copy(),
    max_order_d=2, max_nfev=300, origin_reg_weight=0.0)
print(f"  Zernike fit: {time.time()-t0:.1f}s")

# ── Grid ──
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 31), np.linspace(0, H-1, 31))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf)

# Centre rays for descriptors
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(n_frames)]))
z_p = float((abs(OcL[0, 2]) + abs(OcR[0, 2])) / 2)
f_obj_est = WD_est - z_p

def angle_map(d_model):
    dot = np.clip(np.sum(dzL * d_model, axis=1), -1, 1)
    return np.degrees(np.arccos(dot)).reshape(31, 31)

# ── Stage 1: Perspective CMO ──
print("Stage 1: Perspective CMO...")
pp = 0.0055 / f_obj_est
dL_persp = _normalize(np.column_stack([
    (uf - 1024)*pp - b_est/(2*f_obj_est),
    (vf - 1024)*pp,
    np.ones_like(uf)
]))
r1 = angle_map(dL_persp)

# ── Stage 2: Telecentric CMO ──
print("Stage 2: Telecentric CMO fit...")
theta_fixed = float(np.arctan2(b_est/2, f_obj_est))
x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est,
                    theta_fixed, dcL[0, 1], 0., 0., 0., 0., 0., 0.], dtype=np.float64)
lo_tel = np.array([1., 1., 0., 0., 0., 20., 0., -0.3, -10., -10., -10., -10., -10., -10.], dtype=np.float64)
hi_tel = np.array([500., 1000., 200., 2048., 2048., 200., 0.5, 0.3, 10., 10., 10., 10., 10., 10.], dtype=np.float64)

def res_tel(x):
    m = CMOTelecentricStereoModel.from_parameter_vector(x, pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
    OL, dL = m.ray(uf, vf, "left")
    OR, dR = m.ray(uf, vf, "right")
    OzR, dzR = rf.ray(uf, vf)
    blocks = []
    for z in [50., 80.]:
        for O_a, d_a, O_r, d_r in [(OL, dL, OzL, dzL), (OR, dR, OzR, dzR)]:
            tz_ref = (z - O_r[:, 2]) / d_r[:, 2]
            P_ref = O_r + tz_ref[:, None] * d_r
            tz_mod = (z - O_a[:, 2]) / d_a[:, 2]
            P_mod = O_a + tz_mod[:, None] * d_a
            blocks.append((P_ref - P_mod).reshape(-1))
    return np.concatenate(blocks)

sol_tel = least_squares(res_tel, x0=x0_tel, bounds=(lo_tel, hi_tel), loss="huber", max_nfev=300, xtol=1e-10)
m_tel = CMOTelecentricStereoModel.from_parameter_vector(sol_tel.x[:14], pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
_, dL_tel = m_tel.ray(uf, vf, "left")
r2 = angle_map(dL_tel)

# ── Stage 3: CMO + SE(3) ──
print("Stage 3: CMO + SE(3) fit...")
def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

rot_lo = np.full(3, -0.08); rot_hi = np.full(3, 0.08)
trans_lo = np.full(3, -3.); trans_hi = np.full(3, 3.)
arm_lo = np.concatenate([rot_lo, trans_lo, rot_lo, trans_lo])
arm_hi = np.concatenate([rot_hi, trans_hi, rot_hi, trans_hi])

def res_se3(x):
    m = CMOTelecentricStereoModel.from_parameter_vector(x[:14], pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
    OL, dL = m.ray(uf, vf, "left")
    OR, dR = m.ray(uf, vf, "right")
    OL_a, dL_a = apply_se3(OL, dL, x[14:17], x[17:20])
    OR_a, dR_a = apply_se3(OR, dR, x[20:23], x[23:26])
    OzR, dzR = rf.ray(uf, vf)
    blocks = []
    for z in [50., 80.]:
        for O_a, d_a, O_r, d_r in [(OL_a, dL_a, OzL, dzL), (OR_a, dR_a, OzR, dzR)]:
            tz_ref = (z - O_r[:, 2]) / d_r[:, 2]
            P_ref = O_r + tz_ref[:, None] * d_r
            tz_mod = (z - O_a[:, 2]) / d_a[:, 2]
            P_mod = O_a + tz_mod[:, None] * d_a
            blocks.append((P_ref - P_mod).reshape(-1))
    return np.concatenate(blocks)

sol_se3 = least_squares(res_se3, x0=np.concatenate([sol_tel.x[:14], np.zeros(12)]),
    bounds=(np.concatenate([lo_tel, arm_lo]), np.concatenate([hi_tel, arm_hi])),
    loss="huber", max_nfev=300, xtol=1e-10)
m_se3 = CMOTelecentricStereoModel.from_parameter_vector(sol_se3.x[:14], pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
OL_s, dL_s = m_se3.ray(uf, vf, "left")
_, dL_se3 = apply_se3(OL_s, dL_s, sol_se3.x[14:17], sol_se3.x[17:20])
r3 = angle_map(dL_se3)

# ── Plot: 1×3 ──
print("Plotting...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
stages = [
    (axes[0], r1, "(a) Perspective CMO", 0, 30,
     rf"RMS = {np.sqrt(np.mean(r1**2)):.1f}°"),
    (axes[1], r2, "(b) Telecentric CMO", 0, 0.5,
     rf"RMS = {np.sqrt(np.mean(r2**2)):.3f}°"),
    (axes[2], r3, "(c) CMO + SE(3)", 0, 0.01,
     rf"RMS = {np.sqrt(np.mean(r3**2)):.3f}°"),
]

for ax, arr, title, vmin, vmax, rms_label in stages:
    im = ax.imshow(arr, origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                    extent=[0, W, 0, H], aspect='auto')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('u (px)')
    ax.set_ylabel('v (px)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Direction error (deg)', fontsize=10)
    ax.text(0.98, 0.02, rms_label, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle(r'Residual-guided model construction: $\Delta d$ (deg) vs Zernike rayfield',
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / 'residual_evolution.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'residual_evolution.pdf OK (RMS: {np.sqrt(np.mean(r1**2)):.1f}° / '
      f'{np.sqrt(np.mean(r2**2)):.3f}° / {np.sqrt(np.mean(r3**2)):.4f}°)')
print('Done.')
