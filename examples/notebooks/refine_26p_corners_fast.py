#!/usr/bin/env python3
"""Corner BA refinement — loads pre-computed state from .npz, no pipeline rerun."""

import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

# ── Load state ──
print("Loading intermediate state...", flush=True)
data = np.load(OUT / "intermediate_state.npz")
left_pixels = data["left_pixels"]     # (n_frames, 165, 2)
right_pixels = data["right_pixels"]
obj_pts = data["obj_pts"]             # (165, 3)
opt_R = data["opt_R"]                 # (n_frames, 3, 3)
opt_t = data["opt_t"]                 # (n_frames, 3)
x_rf = data["x_26p"]                  # (26,)
n_frames = int(data["n_frames"])
IMG_SIZE = tuple(data["image_size"])
FX = float(data["FX"])
print(f"  {n_frames} frames, {obj_pts.shape[0]} corners, 26p model loaded", flush=True)

# ── Helpers ──
def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

def build_full_model(x):
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(x[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
    rv_L, t_L = x[14:17], x[17:20]; rv_R, t_R = x[20:23], x[23:26]
    return m_tel, rv_L, t_L, rv_R, t_R

def compute_px_rms(x_model, poses_rv, poses_t):
    m_tel, rv_L, t_L, rv_R, t_R = build_full_model(x_model)
    epx = []
    for pi in range(n_frames):
        Rm = Rotation.from_rotvec(poses_rv[pi]).as_matrix(); t = poses_t[pi]
        Xw = (Rm @ obj_pts.T).T + t[None, :]; n_plane = Rm[:, 2]
        for k in range(obj_pts.shape[0]):
            for uv_px, ch in [(left_pixels[pi,k], "left"), (right_pixels[pi,k], "right")]:
                O_tel, d_tel = m_tel.ray(np.array([uv_px[0]]), np.array([uv_px[1]]), ch)
                rv, t_arm = (rv_L, t_L) if ch == "left" else (rv_R, t_R)
                O_u, d_u = apply_se3(O_tel.reshape(1,3), d_tel.reshape(1,3), rv, t_arm)
                O_u, d_u = O_u[0], d_u[0]
                dn = float(np.dot(d_u, n_plane))
                if abs(dn) > 1e-10:
                    tL_val = float(np.dot(t-O_u, n_plane))/dn
                    e = float(np.linalg.norm((O_u + tL_val*d_u)-Xw[k]))
                    epx.append(e/max(abs(tL_val),1.0)*FX)
    epx_arr = np.array(epx)
    return float(np.sqrt(np.mean(epx_arr**2))), float(np.percentile(epx_arr,50)), float(np.percentile(epx_arr,95))

def corner_residuals(model_x, pose_x):
    m_tel, rv_L, t_L, rv_R, t_R = build_full_model(model_x)
    total = np.zeros(n_frames * 165 * 2, dtype=np.float64)
    idx = 0
    for pi in range(n_frames):
        rv_pose = pose_x[6*pi:6*pi+3]; t_pose = pose_x[6*pi+3:6*pi+6]
        R_pose = Rotation.from_rotvec(rv_pose).as_matrix()
        Xw = (R_pose @ obj_pts.T).T + t_pose[None, :]; n_plane = R_pose[:, 2]
        for k in range(obj_pts.shape[0]):
            for uv_px, ch in [(left_pixels[pi,k], "left"), (right_pixels[pi,k], "right")]:
                O_tel, d_tel = m_tel.ray(np.array([uv_px[0]]), np.array([uv_px[1]]), ch)
                rv, t_arm = (rv_L, t_L) if ch == "left" else (rv_R, t_R)
                O_u, d_u = apply_se3(O_tel.reshape(1,3), d_tel.reshape(1,3), rv, t_arm)
                O_u, d_u = O_u[0], d_u[0]
                dn = float(np.dot(d_u, n_plane))
                if abs(dn) > 1e-10:
                    tL_val = float(np.dot(t_pose-O_u, n_plane))/dn
                    total[idx] = float(np.linalg.norm((O_u + tL_val*d_u)-Xw[k]))
                else:
                    total[idx] = 1e3
                idx += 1
    return total

# ── Initial poses from Zernike ──
poses_rv_init = np.array([Rotation.from_matrix(opt_R[pi]).as_rotvec() for pi in range(n_frames)])
poses_t_init = opt_t.copy()

# Before
px_b, p50_b, p95_b = compute_px_rms(x_rf, poses_rv_init, poses_t_init)
print(f"\n{'='*60}")
print(f"Before corner BA:  {px_b:.2f} px  P50={p50_b:.2f} px  P95={p95_b:.2f} px", flush=True)

# ── BA: model fixed, poses only ──
print("\nStep 1: optimize poses only (model fixed)...", flush=True)
pose_lo = np.empty(n_frames * 6, dtype=np.float64)
pose_hi = np.empty(n_frames * 6, dtype=np.float64)
for pi in range(n_frames):
    pose_lo[6*pi:6*pi+3] = -np.pi; pose_lo[6*pi+3:6*pi+6] = -np.inf
    pose_hi[6*pi:6*pi+3] = np.pi;  pose_hi[6*pi+3:6*pi+6] = np.inf
# Interleave: [rv0(3), t0(3), rv1(3), t1(3), ...]
x_poses = np.empty(n_frames * 6, dtype=np.float64)
for pi in range(n_frames):
    x_poses[6*pi:6*pi+3] = poses_rv_init[pi]
    x_poses[6*pi+3:6*pi+6] = poses_t_init[pi]

t0 = time.time()
_iter_count = [0]
def _res_p_with_progress(p):
    _iter_count[0] += 1
    if _iter_count[0] % 20 == 0:
        r = corner_residuals(x_rf, p)
        print(f"    iter {_iter_count[0]}: RMS={np.sqrt(np.mean(r**2)):.4f} mm", flush=True)
        return r
    return corner_residuals(x_rf, p)

sol_p = least_squares(_res_p_with_progress, x0=x_poses,
                       bounds=(pose_lo, pose_hi),
                       loss="soft_l1", f_scale=0.001, max_nfev=200, xtol=1e-8)
print(f"  nfev={sol_p.nfev}, time={time.time()-t0:.0f}s", flush=True)

poses_rv_p = np.array([sol_p.x[6*pi:6*pi+3] for pi in range(n_frames)])
poses_t_p = np.array([sol_p.x[6*pi+3:6*pi+6] for pi in range(n_frames)])
px_p, p50_p, p95_p = compute_px_rms(x_rf, poses_rv_p, poses_t_p)
print(f"  Pixel RMS = {px_p:.2f} px  P50={p50_p:.2f} px  P95={p95_p:.2f} px", flush=True)

# ── BA: joint model + poses ──
print("\nStep 2: joint model + poses...", flush=True)
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
rot_lo = np.full(3, -0.08); rot_hi = np.full(3, 0.08); trans_lo = np.full(3, -3.0); trans_hi = np.full(3, 3.0)
arm_lo = np.concatenate([rot_lo, trans_lo, rot_lo, trans_lo])
arm_hi = np.concatenate([rot_hi, trans_hi, rot_hi, trans_hi])
model_lo = np.concatenate([lo_tel, arm_lo]); model_hi = np.concatenate([hi_tel, arm_hi])

x0_joint = np.concatenate([x_rf, sol_p.x])
lo_joint = np.concatenate([model_lo, pose_lo]); hi_joint = np.concatenate([model_hi, pose_hi])

t0 = time.time()
sol_j = least_squares(lambda x: corner_residuals(x[:26], x[26:]), x0=x0_joint,
                       bounds=(lo_joint, hi_joint),
                       loss="soft_l1", f_scale=0.001, max_nfev=300, xtol=1e-8)
print(f"  nfev={sol_j.nfev}, time={time.time()-t0:.0f}s", flush=True)

x_final = sol_j.x
poses_rv_f = np.array([x_final[26+6*pi:26+6*pi+3] for pi in range(n_frames)])
poses_t_f = np.array([x_final[26+6*pi+3:26+6*pi+6] for pi in range(n_frames)])
px_f, p50_f, p95_f = compute_px_rms(x_final[:26], poses_rv_f, poses_t_f)

# ── Results ──
print(f"\n{'='*60}")
print(f"{'Stage':>30s}  {'Px RMS':>8s}  {'P50':>8s}  {'P95':>8s}")
print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}")
print(f"  {'26p rayfield fit':>30s}  {px_b:8.2f}  {p50_b:8.2f}  {p95_b:8.2f}")
print(f"  {'+ pose-only BA':>30s}  {px_p:8.2f}  {p50_p:8.2f}  {p95_p:8.2f}")
print(f"  {'+ joint model+pose BA':>30s}  {px_f:8.2f}  {p50_f:8.2f}  {p95_f:8.2f}")

improvement = (px_b - px_f) / px_b * 100
print(f"\nImprovement: {improvement:+.1f}%")

# Parameter shifts
print(f"\nModel parameter shifts:")
for i, name in enumerate(["f_obj","WD","b","cx","cy","f_ang","theta","d_y","sx_L","sy_L","sx_R","sy_R","rho_x","rho_y"]):
    delta = x_final[i] - x_rf[i]
    print(f"  {name:8s}: {x_rf[i]:10.4f} → {x_final[i]:10.4f}  Δ={delta:+.4f}")

# Save
artifact = {
    "description": "26p model refined directly on corner reprojection",
    "before_rayfield": {"px_rms": px_b, "px_p50": p50_b, "px_p95": p95_b},
    "after_pose_only": {"px_rms": px_p, "px_p50": p50_p, "px_p95": p95_p},
    "after_joint_ba": {"px_rms": px_f, "px_p50": p50_f, "px_p95": p95_f},
    "improvement_pct": improvement,
}
with open(OUT / "corner_ba_refinement.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {OUT / 'corner_ba_refinement.json'}")
print("Done!")
