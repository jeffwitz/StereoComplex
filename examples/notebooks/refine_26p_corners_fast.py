#!/usr/bin/env python3
"""Corner BA — 400+ iterations with parameter snapshots every 10 iter."""

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
left_pixels = data["left_pixels"]
right_pixels = data["right_pixels"]
obj_pts = data["obj_pts"]
opt_R = data["opt_R"]
opt_t = data["opt_t"]
x_rf = data["x_26p"]
n_frames = int(data["n_frames"])
IMG_SIZE = tuple(data["image_size"])
FX = float(data["FX"])
print(f"  {n_frames} frames, {obj_pts.shape[0]} corners, 26p model loaded", flush=True)

PARAM_NAMES = ["f_obj","WD","b","cx","cy","f_ang","theta","d_y",
               "sx_L","sy_L","sx_R","sy_R","rho_x","rho_y",
               "rv_Lx","rv_Ly","rv_Lz","t_Lx","t_Ly","t_Lz",
               "rv_Rx","rv_Ry","rv_Rz","t_Rx","t_Ry","t_Rz"]

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

# ── BA: joint model + poses ──
poses_rv_init = np.array([Rotation.from_matrix(opt_R[pi]).as_rotvec() for pi in range(n_frames)])
poses_t_init = opt_t.copy()

# Before
px_b, p50_b, p95_b = compute_px_rms(x_rf, poses_rv_init, poses_t_init)
print(f"\n{'='*60}")
print(f"Before corner BA:  {px_b:.2f} px  P50={p50_b:.2f} px  P95={p95_b:.2f} px", flush=True)

# Bounds
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
rot_lo = np.full(3, -0.08); rot_hi = np.full(3, 0.08)
trans_lo = np.full(3, -3.0); trans_hi = np.full(3, 3.0)
arm_lo = np.concatenate([rot_lo, trans_lo, rot_lo, trans_lo])
arm_hi = np.concatenate([rot_hi, trans_hi, rot_hi, trans_hi])
model_lo = np.concatenate([lo_tel, arm_lo]); model_hi = np.concatenate([hi_tel, arm_hi])

pose_lo = np.empty(n_frames * 6); pose_hi = np.empty(n_frames * 6)
for pi in range(n_frames):
    pose_lo[6*pi:6*pi+3] = -np.pi; pose_lo[6*pi+3:6*pi+6] = -np.inf
    pose_hi[6*pi:6*pi+3] = np.pi;  pose_hi[6*pi+3:6*pi+6] = np.inf

x_poses_init = np.empty(n_frames * 6)
for pi in range(n_frames):
    x_poses_init[6*pi:6*pi+3] = poses_rv_init[pi]
    x_poses_init[6*pi+3:6*pi+6] = poses_t_init[pi]

x0_joint = np.concatenate([x_rf, x_poses_init])
lo_joint = np.concatenate([model_lo, pose_lo]); hi_joint = np.concatenate([model_hi, pose_hi])

# ── Residual function with snapshots ──
_snapshots = []
_iter_count = [0]

def corner_residuals(x):
    model_x = x[:26]; pose_x = x[26:]
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

def res_with_progress(x):
    _iter_count[0] += 1
    n = _iter_count[0]
    r = corner_residuals(x)
    rms = np.sqrt(np.mean(r**2))
    if n % 10 == 0:
        # Snapshot every 10 iterations
        snap = {"iter": n, "rms_mm": float(rms)}
        for ii, name in enumerate(PARAM_NAMES):
            snap[name] = float(x[ii])
        _snapshots.append(snap)
        # Save partial results every snapshot
        with open(OUT / "corner_ba_refinement.json", "w") as f:
            json.dump({"snapshots": _snapshots, "in_progress": True}, f)
        print(f"    iter {n}: RMS={rms:.4f} mm  f_obj={x[0]:.1f}  WD={x[1]:.1f}  b={x[2]:.2f}", flush=True)
    return r

# ── Run joint BA ──
print(f"\nStep: joint model+pose BA (max_nfev=200)...", flush=True)
t0 = time.time()
sol = least_squares(res_with_progress, x0=x0_joint,
                    bounds=(lo_joint, hi_joint),
                    loss="soft_l1", f_scale=0.001, max_nfev=200, xtol=1e-8)
print(f"  nfev={sol.nfev}, time={time.time()-t0:.0f}s", flush=True)

x_final = sol.x
x_final = sol.x
poses_rv_f = np.array([x_final[26+6*pi:26+6*pi+3] for pi in range(n_frames)])
poses_t_f = np.array([x_final[26+6*pi+3:26+6*pi+6] for pi in range(n_frames)])
px_f, p50_f, p95_f = compute_px_rms(x_final[:26], poses_rv_f, poses_t_f)

print(f"\n{'='*60}")
print(f"{'Stage':>30s}  {'Px RMS':>8s}  {'P50':>8s}  {'P95':>8s}")
print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}")
print(f"  {'26p rayfield fit':>30s}  {px_b:8.2f}  {p50_b:8.2f}  {p95_b:8.2f}")
print(f"  {'+ joint model+pose BA':>30s}  {px_f:8.2f}  {p50_f:8.2f}  {p95_f:8.2f}")

improvement = (px_b - px_f) / px_b * 100
print(f"\nImprovement: {improvement:+.1f}%")
print(f"Snapshots saved: {len(_snapshots)}")

# Parameter shifts
print(f"\nModel parameter shifts:")
for i, name in enumerate(PARAM_NAMES):
    delta = x_final[i] - x_rf[i]
    print(f"  {name:8s}: {x_rf[i]:10.4f} → {x_final[i]:10.4f}  Δ={delta:+.4f}")

# Save with snapshots
artifact = {
    "description": "26p model refined directly on corner reprojection (joint BA, 400 nfev)",
    "before_rayfield": {"px_rms": px_b, "px_p50": p50_b, "px_p95": p95_b},
    "after_joint_ba": {"px_rms": px_f, "px_p50": p50_f, "px_p95": p95_f},
    "improvement_pct": improvement,
    "final_params": {name: float(x_final[i]) for i, name in enumerate(PARAM_NAMES)},
    "snapshots_every_10_iter": _snapshots,
}
with open(OUT / "corner_ba_refinement.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {OUT / 'corner_ba_refinement.json'}")
print("Done!")
