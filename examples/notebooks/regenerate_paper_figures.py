#!/usr/bin/env python3
"""Regenerate all CMO paper figures with serif fonts."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys, json
sys.path.insert(0, 'src')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

OUT = Path('paper/cmo/figures')
data = np.load('docs/assets/pycaso_real_data/intermediate_state.npz')
W, H = tuple(data["image_size"])

# ═══════════════════════════════════════════════════════════
# Figure 1: residual_evolution.png — 3 panels (1×3), no Zernike-vs-self
# ═══════════════════════════════════════════════════════════
print("Generating residual evolution (1×3)...")

import cv2; from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

FX = float(data["FX"]); K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)
n_frames = 10
left_pixels, right_pixels = data["left_pixels"], data["right_pixels"]
obj_pts = data["obj_pts"]

left_px = [left_pixels[pi] for pi in range(n_frames)]
right_px = [right_pixels[pi] for pi in range(n_frames)]
rvecs, tvecs = [], []
for lp in left_px:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3)); tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0.,0.,65.]))
obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
                             left_pixels=left_px, right_pixels=right_px, point_indices=[np.arange(165,dtype=int) for _ in range(n_frames)],
                             noise_std_px=0.0, image_size=tuple(data["image_size"]))
lf, rf, zd, _, _ = fit_constrained_zernike_rayfield(obs, image_size=tuple(data["image_size"]), K_left=K, K_right=K.copy(), max_order_d=2, max_nfev=300, origin_reg_weight=0.0)

u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf)
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
WD_est = float(np.mean([float(data["opt_t"][i][2]) for i in range(n_frames)]))
z_p = float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
f_obj_est = WD_est - z_p

def angle_map(d_model):
    dot = np.clip(np.sum(dzL * d_model, axis=1), -1, 1)
    return np.degrees(np.arccos(dot)).reshape(41, 41)

# Stage 1: Perspective
pp = 0.0055 / f_obj_est
dL_persp = _normalize(np.column_stack([
    (uf - 1024)*pp - b_est/(2*f_obj_est), (vf - 1024)*pp, np.ones_like(uf)]))
r1 = angle_map(dL_persp)

# Stage 2: Telecentric
theta_fixed = float(np.arctan2(b_est/2, f_obj_est))
x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dcL[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
def res_tel(x):
    m = CMOTelecentricStereoModel.from_parameter_vector(x, pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
    OL,dL=m.ray(uf,vf,"left");OR,dR=m.ray(uf,vf,"right");OzR,dzR=rf.ray(uf,vf)
    blocks=[]
    for z in [50.,80.]:
        for O_a,d_a,O_r,d_r in [(OL,dL,OzL,dzL),(OR,dR,OzR,dzR)]:
            tz_ref=(z-O_r[:,2])/d_r[:,2];P_ref=O_r+tz_ref[:,None]*d_r
            tz_mod=(z-O_a[:,2])/d_a[:,2];P_mod=O_a+tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)
sol_tel = least_squares(res_tel, x0=x0_tel, bounds=(lo_tel,hi_tel), loss="huber",max_nfev=300,xtol=1e-10)
m_tel = CMOTelecentricStereoModel.from_parameter_vector(sol_tel.x[:14], pixel_pitch_mm=0.0055, image_size=tuple(data["image_size"]))
_, dL_tel = m_tel.ray(uf, vf, "left"); r2 = angle_map(dL_tel)

# Stage 3: CMO + SE(3)
def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None,:], _normalize((R @ d.T).T)
rot_lo=np.full(3,-0.08);rot_hi=np.full(3,0.08);trans_lo=np.full(3,-3.);trans_hi=np.full(3,3.)
arm_lo=np.concatenate([rot_lo,trans_lo,rot_lo,trans_lo]);arm_hi=np.concatenate([rot_hi,trans_hi,rot_hi,trans_hi])
def res_se3(x):
    m=CMOTelecentricStereoModel.from_parameter_vector(x[:14],pixel_pitch_mm=0.0055,image_size=tuple(data["image_size"]))
    OL,dL=m.ray(uf,vf,"left");OR,dR=m.ray(uf,vf,"right");OzR,dzR=rf.ray(uf,vf)
    OL_a,dL_a=apply_se3(OL,dL,x[14:17],x[17:20]);OR_a,dR_a=apply_se3(OR,dR,x[20:23],x[23:26])
    blocks=[]
    for z in [50.,80.]:
        for O_a,d_a,O_r,d_r in [(OL_a,dL_a,OzL,dzL),(OR_a,dR_a,OzR,dzR)]:
            tz_ref=(z-O_r[:,2])/d_r[:,2];P_ref=O_r+tz_ref[:,None]*d_r
            tz_mod=(z-O_a[:,2])/d_a[:,2];P_mod=O_a+tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)
sol_se3=least_squares(res_se3,x0=np.concatenate([sol_tel.x[:14],np.zeros(12)]),
    bounds=(np.concatenate([lo_tel,arm_lo]),np.concatenate([hi_tel,arm_hi])),loss="huber",max_nfev=300,xtol=1e-10)
m_se3=CMOTelecentricStereoModel.from_parameter_vector(sol_se3.x[:14],pixel_pitch_mm=0.0055,image_size=tuple(data["image_size"]))
OL_s,dL_s=m_se3.ray(uf,vf,"left")
_,dL_se3=apply_se3(OL_s,dL_s,sol_se3.x[14:17],sol_se3.x[17:20]);r3=angle_map(dL_se3)

# Plot 1×3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
stages = [
    (axes[0], r1, "(a) Perspective CMO", 0, 30, rf"RMS={np.sqrt(np.mean(r1**2)):.1f}°"),
    (axes[1], r2, "(b) Telecentric CMO", 0, 0.5, rf"RMS={np.sqrt(np.mean(r2**2)):.3f}°"),
    (axes[2], r3, "(c) CMO + SE(3)", 0, 0.01, rf"RMS={np.sqrt(np.mean(r3**2)):.3f}°"),
]
for ax, arr, title, vmin, vmax, rms_label in stages:
    im = ax.imshow(arr, origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                    extent=[0, W, 0, H], aspect='auto')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Direction error (deg)', fontsize=10)
    ax.text(0.98, 0.02, rms_label, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle(r'Residual-guided model construction: $\Delta d$ (deg) vs Zernike rayfield', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(OUT / 'residual_evolution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  residual_evolution.png OK")

# ═══════════════════════════════════════════════════════════
# Figure 2: dy_profile_comparison.png — regenerate from data
# ═══════════════════════════════════════════════════════════
print("Generating dy_profile_comparison...")
with open('docs/assets/pycaso_real_data/dy_profile_data.json') as f:
    dd = json.load(f)
v_px = np.array(dd['v_px'])
zernike = np.array(dd['zernike'])
telecentric = np.array(dd['telecentric'])
perspective = np.array(dd['perspective'])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(v_px, zernike, 'ko-', label='Zernike (measured)', ms=6)
ax.plot(v_px, telecentric, 's--', color='darkgreen', label='Telecentric CMO', ms=7)
ax.plot(v_px, perspective, '^:', color='darkred', label='Perspective CMO', ms=7)
ax.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel('v (px)'); ax.set_ylabel('$d_y$')
ax.set_title('$d_y(u,v)$ profiles across sensor centre column')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.text(0.98, 0.98, 'Range: Zernike=0.079, Telecentric=0.073, Perspective=0.232',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
fig.tight_layout()
fig.savefig(OUT / 'dy_profile_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  dy_profile_comparison.png OK")

# ═══════════════════════════════════════════════════════════
# Figure 3: pareto_gauge_regularization.png — regenerate from data
# ═══════════════════════════════════════════════════════════
print("Generating Pareto plot...")
with open('docs/assets/pycaso_real_data/zernike_gauge_regularization_sweep.json') as f:
    sweep = json.load(f)

sweep_results = sweep.get('sweep', [])
# Find Pareto-optimal points
pareto = []
for i, r in enumerate(sweep_results):
    dominated = False
    for j, r2 in enumerate(sweep_results):
        if i == j: continue
        if r2["ray_rms_mm"] <= r["ray_rms_mm"] and r2["drift_z0_deg"] <= r["drift_z0_deg"]:
            if r2["ray_rms_mm"] < r["ray_rms_mm"] or r2["drift_z0_deg"] < r["drift_z0_deg"]:
                dominated = True; break
    if not dominated:
        pareto.append(r)

all_rms = [r["ray_rms_mm"] for r in sweep_results]
all_z0 = [r["drift_z0_deg"] for r in sweep_results]
pareto_rms = [r["ray_rms_mm"] for r in pareto]
pareto_z0 = [r["drift_z0_deg"] for r in pareto]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
# RMS vs Z0 drift
ax = axes[0]
ax.scatter(all_rms, all_z0, c="steelblue", s=60, zorder=3, label="all runs")
ax.scatter(pareto_rms, pareto_z0, c="darkorange", s=120, zorder=4, edgecolors="black", linewidth=0.5, label="Pareto-optimal")
po_sorted = sorted(pareto, key=lambda r_: r_["ray_rms_mm"])
if len(po_sorted) >= 2:
    ax.plot([r_["ray_rms_mm"] for r_ in po_sorted], [r_["drift_z0_deg"] for r_ in po_sorted],
            "darkorange", lw=1.5, alpha=0.6, zorder=2)
constrained_rms = sweep.get('constrained_rms_mm', 0.000653)
ax.axvline(x=constrained_rms, color="gray", ls="--", alpha=0.5, label="constrained ref")
ax.set_xlabel("Ray RMS (mm)"); ax.set_ylabel("Z₀ drift (°)")
ax.set_title("Pareto frontier: RMS vs gauge drift")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Baseline
ax = axes[1]
for r in sweep_results:
    ax.plot(r["drift_z0_deg"], r.get("baseline_mm", 0), "o", ms=8, alpha=0.7)
ax.set_xlabel("Z₀ drift (°)"); ax.set_ylabel("Baseline (mm)")
ax.set_title("Baseline stability"); ax.grid(True, alpha=0.3)

# Convergence
ax = axes[2]
for r in sweep_results:
    ax.plot(r["drift_z0_deg"], r.get("convergence_angle_deg", 0), "o", ms=8, alpha=0.7)
ax.set_xlabel("Z₀ drift (°)"); ax.set_ylabel("Convergence angle (°)")
ax.set_title("Convergence angle stability"); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'pareto_gauge_regularization.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  pareto_gauge_regularization.png OK")

print("\nAll figures regenerated with serif fonts.")
