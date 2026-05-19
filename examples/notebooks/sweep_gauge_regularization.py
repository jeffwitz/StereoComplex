#!/usr/bin/env python3
"""Gauge-regularized full-pose Zernike fit — sweep regularization strengths.

Uses raw OpenCV ChArUco detection (same as notebook 09).
Generates: docs/assets/pycaso_real_data/zernike_gauge_regularization_sweep.json
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"

from stereocomplex.core.model_compact.zernike import eval_real_zernike, zernike_modes
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients,
)
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import (
    estimate_initial_poses_from_central_pinhole, ZernikeFitDiagnostics,
)

# ── Load constrained solution as prior ───────────────────────────────
with open(OUT / "zernike_pose_variants.json") as f:
    pv = json.load(f)

IMG_SIZE = tuple(pv["dataset"]["image_size"])
W, H = IMG_SIZE
FX = 25600.0
K_arr = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

def _arr(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 3)

cc = pv["zernike_constrained"]
cf_full = pv["zernike_full_poses"]
config2 = ZernikeOriginFieldConfig(image_size=IMG_SIZE, max_order=2)
all_modes = config2.modes()
n_modes = len(all_modes)

def make_field(origin_list, dir_list, K):
    return ZernikeRayField(K=K, config=config2,
        coefficients=ZernikeRayFieldCoefficients(
            origin_coeffs=_arr(origin_list), direction_coeffs=_arr(dir_list)))

lf_constrained = make_field(cc["left_origin_coeffs"], cc["left_direction_coeffs"], K_arr)
rf_constrained = make_field(cc["right_origin_coeffs"], cc["right_direction_coeffs"], K_arr)
prior_L_d = lf_constrained.direction_coeffs
prior_R_d = rf_constrained.direction_coeffs

print("=== Gauge-regularized sweep ===")
print(f"Prior Z0 d_L = {prior_L_d[0]}")
print(f"Prior Z1_cos  = {prior_L_d[1]}")
print(f"Prior Z1_sin  = {prior_L_d[2]}")

# ── Load observations (same raw OpenCV as notebook) ──────────────────
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"

SQR = 0.3; NCX, NCY = 16, 12; DICT_NAME = "DICT_6X6_250"

from cv2 import aruco
dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
ocv_board = aruco.CharucoBoard((NCX, NCY), SQR, SQR / 2, dictionary)
ocv_board.setLegacyPattern(True)
chess3 = ocv_board.getChessboardCorners()

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 75
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate = 0.005
params.maxMarkerPerimeterRate = 0.20
params.polygonalApproxAccuracyRate = 0.08
params.minCornerDistanceRate = 0.02
params.minDistanceToBorder = 1

detector = aruco.ArucoDetector(dictionary, params)
charuco_detector = aruco.CharucoDetector(ocv_board)

lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float)
print(f"\n{len(paired_z)} stereo pairs")

all_corners_L, all_corners_R, all_ids_L, all_ids_R = [], [], [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    for img, corners_list, ids_list in [(lg, all_corners_L, all_ids_L), (rg, all_corners_R, all_ids_R)]:
        cc_c, ci, mc, mi = charuco_detector.detectBoard(img)
        if cc_c is not None and ci is not None:
            corners_list.append(np.asarray(cc_c, dtype=np.float64).reshape(-1, 2))
            ids_list.append(np.asarray(ci, dtype=int).ravel())
        else:
            corners_list.append(np.zeros((0, 2)))
            ids_list.append(np.zeros(0, dtype=int))

counts = [(len(l), len(r)) for l, r in zip(all_corners_L, all_corners_R)]
print(f"Corners: mean L={np.mean([c[0] for c in counts]):.0f}, R={np.mean([c[1] for c in counts]):.0f}")
print(f"         min  L={np.min([c[0] for c in counts])}, R={np.min([c[1] for c in counts])}")

# Filter frames with >= 30 corners in both channels
# Use ChArUco ID intersection so left/right corners map to same 3D points
valid_L, valid_R, valid_ids_L, valid_ids_R = [], [], [], []
valid_rvecs, valid_tvecs = [], []
for i in range(len(paired_z)):
    if len(all_corners_L[i]) < 30 or len(all_corners_R[i]) < 30:
        continue
    # Intersect detected IDs in both channels
    ids_L = all_ids_L[i]; ids_R = all_ids_R[i]
    common_ids = np.intersect1d(ids_L, ids_R)
    if len(common_ids) < 30:
        continue
    mask_L = np.isin(ids_L, common_ids)
    mask_R = np.isin(ids_R, common_ids)
    valid_L.append(all_corners_L[i][mask_L])
    valid_R.append(all_corners_R[i][mask_R])
    valid_ids_L.append(ids_L[mask_L])
    valid_ids_R.append(ids_R[mask_R])
    # Initial pose from left
    lp = all_corners_L[i][mask_L]
    s, rv, tv = cv2.solvePnP(
        chess3[ids_L[mask_L]].astype(np.float32), lp.astype(np.float32),
        K_arr.astype(np.float32), np.zeros(5, dtype=np.float32),
    )
    valid_rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    valid_tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))

print(f"After >=30 common IDs filter: {len(valid_L)} frames")
print(f"Corners per frame: {[len(l) for l in valid_L]}")

# Use ChArUco IDs as point indices (direct mapping to object_points_mm)
point_indices = [np.asarray(ids, dtype=int) for ids in valid_ids_L]

obs = CharucoObservationSet(
    object_points_mm=chess3,
    pose_rvecs=np.array(valid_rvecs),
    pose_tvecs=np.array(valid_tvecs),
    left_pixels=valid_L,
    right_pixels=valid_R,
    point_indices=point_indices,
    noise_std_px=0.0,
    image_size=IMG_SIZE,
)

# ── Regularized fitting function ─────────────────────────────────────
def fit_gauge_regularized_zernike(
    obs, lf_prior, rf_prior,
    sigma_z0_deg=0.5, sigma_z1_deg=2.0, sigma_z20_deg=100.0,
    max_nfev=500, origin_reg_weight=1e-3,
):
    W_loc, H_loc = IMG_SIZE
    K_L = K_arr; K_R = K_arr.copy()
    config = config2
    modes = all_modes
    n_modes_loc = len(modes)
    n_zernike = n_modes_loc * 6

    R_est, t_est = estimate_initial_poses_from_central_pinhole(obs, K_L)
    n_poses = len(R_est)

    x0_poses = []
    for i in range(n_poses):
        rv = Rotation.from_matrix(R_est[i]).as_rotvec()
        x0_poses.append(rv)
        x0_poses.append(np.asarray(t_est[i], dtype=np.float64).reshape(3))
    x0_poses_arr = np.concatenate(x0_poses)

    uL_all, vL_all, idxL_all, poseL_all = [], [], [], []
    uR_all, vR_all, idxR_all, poseR_all = [], [], [], []
    for pi in range(len(obs.left_pixels)):
        lp = obs.left_pixels[pi]; rp = obs.right_pixels[pi]
        if lp.size == 0 and rp.size == 0: continue
        idxL = obs.point_indices[pi]
        nL = lp.shape[0]; nR = rp.shape[0]
        if nL > 0:
            uL_all.append(lp[:, 0]); vL_all.append(lp[:, 1]); idxL_all.append(idxL[:nL])
            poseL_all.append(np.full(nL, pi, dtype=int))
        if nR > 0:
            uR_all.append(rp[:, 0]); vR_all.append(rp[:, 1]); idxR_all.append(idxL[:nR])
            poseR_all.append(np.full(nR, pi, dtype=int))

    uL = np.concatenate(uL_all); vL = np.concatenate(vL_all)
    idxL = np.concatenate(idxL_all); poseL = np.concatenate(poseL_all)
    uR = np.concatenate(uR_all); vR = np.concatenate(vR_all)
    idxR = np.concatenate(idxR_all); poseR = np.concatenate(poseR_all)
    obj_pts = obs.object_points_mm

    def _precompute(u_arr, v_arr, K):
        xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(W_loc - 1) - 1.0
        zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(H_loc - 1) - 1.0
        rho = np.sqrt(xi*xi + zeta*zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, n_modes_loc), dtype=np.float64)
        for j, mode in enumerate(modes):
            A[:, j] = eval_real_zernike(mode, rho, theta)
        Kk = np.asarray(K, dtype=np.float64).reshape(3, 3)
        fx_inv = 1.0/Kk[0,0]; fy_inv = 1.0/Kk[1,1]
        cx, cy = Kk[0,2], Kk[1,2]
        dx = (u_arr-cx)*fx_inv; dy = (v_arr-cy)*fy_inv
        dz = np.ones_like(dx)
        inv = 1.0/np.sqrt(dx*dx + dy*dy + dz*dz)
        d0 = np.column_stack([dx*inv, dy*inv, dz*inv])
        return A, d0

    class _G:
        __slots__ = ("pose_idx", "A", "d0", "X_local")
    groups_L, groups_R = [], []
    for pi in range(n_poses):
        mask_L = poseL == pi; mask_R = poseR == pi
        if mask_L.any():
            g = _G(); g.pose_idx = pi
            g.A, g.d0 = _precompute(uL[mask_L], vL[mask_L], K_L)
            g.X_local = obj_pts[idxL[mask_L]]; groups_L.append(g)
        if mask_R.any():
            g = _G(); g.pose_idx = pi
            g.A, g.d0 = _precompute(uR[mask_R], vR[mask_R], K_R)
            g.X_local = obj_pts[idxR[mask_R]]; groups_R.append(g)

    def _chan_residuals(origin_c, dir_c, pose_params, groups):
        blocks = []
        for g in groups:
            pi = g.pose_idx
            rv = pose_params[6*pi:6*pi+3]; tv = pose_params[6*pi+3:6*pi+6]
            R_mat = Rotation.from_rotvec(rv).as_matrix()
            t = np.asarray(tv, dtype=np.float64).reshape(3)
            X_world = (R_mat @ g.X_local.T).T + t[None, :]
            d_delta_raw = g.A @ dir_c
            d_delta = d_delta_raw - np.sum(d_delta_raw*g.d0, axis=1, keepdims=True)*g.d0
            d = (g.d0 + d_delta)
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            O_raw = g.A @ origin_c
            O = O_raw - np.sum(O_raw*d, axis=1, keepdims=True)*d
            delta = X_world - O
            proj = np.sum(delta*d, axis=1, keepdims=True)*d
            blocks.append((delta - proj).reshape(-1))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    prior_L_d = lf_prior.direction_coeffs
    prior_R_d = rf_prior.direction_coeffs

    deg_to_rad = np.pi / 180.0
    sigma_z0 = max(sigma_z0_deg, 1e-6) * deg_to_rad
    sigma_z1 = max(sigma_z1_deg, 1e-6) * deg_to_rad
    sigma_z20 = max(sigma_z20_deg, 1e-6) * deg_to_rad

    prior_mask = np.zeros(n_modes_loc, dtype=bool)
    prior_sigmas = np.zeros(n_modes_loc, dtype=np.float64)
    prior_mask[0] = True; prior_sigmas[0] = sigma_z0
    prior_mask[1] = True; prior_sigmas[1] = sigma_z1
    prior_mask[2] = True; prior_sigmas[2] = sigma_z1
    if sigma_z20_deg < 50:
        prior_mask[3] = True; prior_sigmas[3] = sigma_z20

    def residuals_reg(x):
        cL = x[:n_zernike]; cR = x[n_zernike:2*n_zernike]
        origin_L = cL[:n_zernike//2].reshape(n_modes_loc, 3)
        dir_L = cL[n_zernike//2:].reshape(n_modes_loc, 3)
        origin_R = cR[:n_zernike//2].reshape(n_modes_loc, 3)
        dir_R = cR[n_zernike//2:].reshape(n_modes_loc, 3)
        pose_params = x[2*n_zernike:]

        rL = _chan_residuals(origin_L, dir_L, pose_params, groups_L)
        rR = _chan_residuals(origin_R, dir_R, pose_params, groups_R)
        reg_blocks = [rL, rR]

        if origin_reg_weight > 0:
            reg_blocks.append(np.sqrt(origin_reg_weight) * origin_L[:, 2])
            reg_blocks.append(np.sqrt(origin_reg_weight) * origin_R[:, 2])

        for m in range(n_modes_loc):
            if not prior_mask[m]: continue
            s = prior_sigmas[m]
            if s < 1e-10: continue
            for comp in range(3):
                reg_blocks.append(np.array([(dir_L[m, comp] - prior_L_d[m, comp]) / s]))
                reg_blocks.append(np.array([(dir_R[m, comp] - prior_R_d[m, comp]) / s]))
        return np.concatenate(reg_blocks)

    n_half = n_zernike // 2
    origin_bounds_lo = np.full(n_half, -np.inf)
    origin_bounds_hi = np.full(n_half, np.inf)
    for j in range(2, n_half, 3):
        origin_bounds_lo[j] = -20.0; origin_bounds_hi[j] = 20.0
    dir_bounds_lo = np.full(n_half, -0.5)
    dir_bounds_hi = np.full(n_half, 0.5)
    coeff_lo = np.concatenate([origin_bounds_lo, dir_bounds_lo])
    coeff_hi = np.concatenate([origin_bounds_hi, dir_bounds_hi])
    bounds = (
        np.concatenate([coeff_lo, coeff_lo, x0_poses_arr - 0.3]),
        np.concatenate([coeff_hi, coeff_hi, x0_poses_arr + 0.3]),
    )

    x0 = np.concatenate([
        np.zeros(n_zernike, dtype=np.float64),
        np.zeros(n_zernike, dtype=np.float64),
        x0_poses_arr,
    ])

    sol = least_squares(
        residuals_reg, x0=x0, bounds=bounds, method="trf",
        loss="linear", max_nfev=int(max_nfev),
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    def _build_field(coeffs_flat, K):
        arr = np.asarray(coeffs_flat, dtype=np.float64).reshape(-1)
        return ZernikeRayField(K=K, config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=arr[:n_modes_loc*3].reshape(n_modes_loc, 3),
                direction_coeffs=arr[n_modes_loc*3:].reshape(n_modes_loc, 3)))

    lf_out = _build_field(sol.x[:n_zernike], K_L)
    rf_out = _build_field(sol.x[n_zernike:2*n_zernike], K_R)

    def residuals_geo(x):
        cL = x[:n_zernike]; cR = x[n_zernike:2*n_zernike]
        origin_L = cL[:n_zernike//2].reshape(n_modes_loc, 3)
        dir_L = cL[n_zernike//2:].reshape(n_modes_loc, 3)
        origin_R = cR[:n_zernike//2].reshape(n_modes_loc, 3)
        dir_R = cR[n_zernike//2:].reshape(n_modes_loc, 3)
        pose_params = x[2*n_zernike:]
        rL = _chan_residuals(origin_L, dir_L, pose_params, groups_L)
        rR = _chan_residuals(origin_R, dir_R, pose_params, groups_R)
        return np.concatenate([rL, rR])

    r_geo = residuals_geo(sol.x)
    rms_mm = float(np.sqrt(np.mean(r_geo**2)))

    diag = ZernikeFitDiagnostics(
        max_order=2, n_zernike_coeffs=n_zernike, n_poses=n_poses,
        n_observations=uL.size + uR.size, ray_rms_mm=rms_mm,
        converged=bool(sol.success), nfev=int(sol.nfev),
    )
    return lf_out, rf_out, diag, sol

# ── Physical indicators ──────────────────────────────────────────────
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
u_grid_f = u_grid.ravel(); v_grid_f = v_grid.ravel()

def extract_indicators(field_L, field_R):
    O_L, d_L = field_L.ray(u_grid_f, v_grid_f)
    O_R, d_R = field_R.ray(u_grid_f, v_grid_f)
    u_c = np.array([1024.0]); v_c = np.array([1024.0])
    _, dL_c = field_L.ray(u_c, v_c)
    _, dR_c = field_R.ray(u_c, v_c)
    return {
        "baseline_mm": float(np.linalg.norm(np.mean(O_R, axis=0) - np.mean(O_L, axis=0))),
        "convergence_angle_deg": float(np.degrees(np.arccos(np.clip(
            np.dot(dL_c[0], dR_c[0]), -1.0, 1.0)))),
        "dy_range_L": float(np.max(d_L[:, 1]) - np.min(d_L[:, 1])),
        "dy_range_R": float(np.max(d_R[:, 1]) - np.min(d_R[:, 1])),
        "subpupil_depth_mm": float((abs(np.mean(O_L[:, 2])) + abs(np.mean(O_R[:, 2]))) / 2),
        "dy_asymmetry": float(np.mean(d_L[:, 1]) - np.mean(d_R[:, 1])),
        "dx_antisymmetry": float(dL_c[0, 0] + dR_c[0, 0]),
    }

ind_constrained = extract_indicators(lf_constrained, rf_constrained)
print(f"\nConstrained indicators: b={ind_constrained['baseline_mm']:.1f}mm, "
      f"θ={ind_constrained['convergence_angle_deg']:.1f}°, "
      f"dy_range_L={ind_constrained['dy_range_L']:.4f}")

# ── Sweep ────────────────────────────────────────────────────────────
runs = [
    ("full_pose_baseline", 100.0, 100.0, 100.0),
    ("z0_0.05deg", 0.05, 100.0, 100.0),
    ("z0_0.1deg", 0.1, 100.0, 100.0),
    ("z0_0.2deg", 0.2, 100.0, 100.0),
    ("z0_0.5deg", 0.5, 100.0, 100.0),
    ("z0_1.0deg", 1.0, 100.0, 100.0),
    ("z0_2.0deg", 2.0, 100.0, 100.0),
    ("z0z1_0.1_0.5", 0.1, 0.5, 100.0),
    ("z0z1_0.2_0.5", 0.2, 0.5, 100.0),
    ("z0z1_0.2_1.0", 0.2, 1.0, 100.0),
    ("z0z1_0.5_1.0", 0.5, 1.0, 100.0),
    ("z0z1_0.5_2.0", 0.5, 2.0, 100.0),
]

print(f"\n{'Run':>25s}  {'RMSmm':>9s}  {'Z0△°':>7s}  {'Z1△°':>7s}  "
      f"{'b_mm':>7s}  {'θ°':>6s}  {'dy_ranL':>8s}  {'NFEV':>5s}  {'s':>4s}")
print("-" * 100)

sweep_results = []
for label, sz0, sz1, sz20 in runs:
    t0 = time.time()
    try:
        lf_reg, rf_reg, diag, sol = fit_gauge_regularized_zernike(
            obs, lf_constrained, rf_constrained,
            sigma_z0_deg=sz0, sigma_z1_deg=sz1, sigma_z20_deg=sz20,
            max_nfev=400, origin_reg_weight=1e-3,
        )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  {label:25s}  FAILED: {e}")
        import traceback; traceback.print_exc()
        continue

    dir_L_reg = lf_reg.direction_coeffs
    dir_R_reg = rf_reg.direction_coeffs
    drift_z0_L = np.linalg.norm(dir_L_reg[0] - prior_L_d[0])
    drift_z0_R = np.linalg.norm(dir_R_reg[0] - prior_R_d[0])
    drift_z1_cos_L = np.linalg.norm(dir_L_reg[1] - prior_L_d[1])
    drift_z1_sin_L = np.linalg.norm(dir_L_reg[2] - prior_L_d[2])
    drift_z1_cos_R = np.linalg.norm(dir_R_reg[1] - prior_R_d[1])
    drift_z1_sin_R = np.linalg.norm(dir_R_reg[2] - prior_R_d[2])
    drift_z0_deg = float(np.degrees(0.5*(drift_z0_L + drift_z0_R)))
    drift_z1_deg = float(np.degrees(0.25*(drift_z1_cos_L + drift_z1_sin_L
                                          + drift_z1_cos_R + drift_z1_sin_R)))

    ind = extract_indicators(lf_reg, rf_reg)

    result = {
        "label": label,
        "sigma_z0_deg": sz0, "sigma_z1_deg": sz1, "sigma_z20_deg": sz20,
        "ray_rms_mm": diag.ray_rms_mm,
        "converged": diag.converged, "nfev": diag.nfev,
        "drift_z0_deg": drift_z0_deg, "drift_z1_deg": drift_z1_deg,
        "drift_z0_L": float(drift_z0_L), "drift_z0_R": float(drift_z0_R),
        "drift_z1_cos_L": float(drift_z1_cos_L), "drift_z1_sin_L": float(drift_z1_sin_L),
        "drift_z1_cos_R": float(drift_z1_cos_R), "drift_z1_sin_R": float(drift_z1_sin_R),
        **{f"ind_{k}": (float(v) if isinstance(v, (np.floating, float)) else v)
           for k, v in ind.items()},
        "time_s": elapsed,
    }
    sweep_results.append(result)

    print(f"  {label:25s}  {diag.ray_rms_mm:9.6f}  {drift_z0_deg:7.3f}  {drift_z1_deg:7.3f}  "
          f"{ind['baseline_mm']:7.1f}  {ind['convergence_angle_deg']:6.1f}  "
          f"{ind['dy_range_L']:8.4f}  {diag.nfev:5d}  {elapsed:4.0f}")

constrained_rms = cc["ray_rms_mm"]
full_rms = cf_full["ray_rms_mm"]
print(f"\n  {'constrained (ref)':25s}  {constrained_rms:9.6f}  {0.0:7.3f}  {0.0:7.3f}  "
      f"{ind_constrained['baseline_mm']:7.1f}  {ind_constrained['convergence_angle_deg']:6.1f}  "
      f"{ind_constrained['dy_range_L']:8.4f}")
print(f"  {'full_pose (ref)':25s}  {full_rms:9.6f}")

# ── Pareto ───────────────────────────────────────────────────────────
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

print(f"\nPareto-optimal ({len(pareto)}):")
for r in sorted(pareto, key=lambda x: x["ray_rms_mm"]):
    print(f"  {r['label']:25s}  RMS={r['ray_rms_mm']:.6f}mm  "
          f"Z0△={r['drift_z0_deg']:.3f}°  "
          f"b={r['ind_baseline_mm']:.1f}mm  θ={r['ind_convergence_angle_deg']:.1f}°")

sweet = None
for r in sorted(sweep_results, key=lambda x: x["ray_rms_mm"]):
    if r["drift_z0_deg"] < 0.5:
        sweet = r
if sweet:
    print(f"\nSweet spot (Z0 drift < 0.5°, lowest RMS):")
    print(f"  {sweet['label']}: RMS={sweet['ray_rms_mm']:.6f}mm  "
          f"Z0△={sweet['drift_z0_deg']:.3f}°  "
          f"b={sweet['ind_baseline_mm']:.1f}mm  θ={sweet['ind_convergence_angle_deg']:.1f}°  "
          f"dy_range={sweet['ind_dy_range_L']:.4f}")

# ── Save ─────────────────────────────────────────────────────────────
artifact = {
    "description": "Gauge-regularized full-pose Zernike sweep — Z0/Z1 direction anchor",
    "constrained_reference": {
        "ray_rms_mm": constrained_rms,
        "indicators": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in ind_constrained.items()},
    },
    "full_pose_reference": {"ray_rms_mm": full_rms},
    "sweep": sweep_results,
    "pareto_optimal": [r["label"] for r in pareto],
    "sweet_spot": sweet["label"] if sweet else None,
}

fname = OUT / "zernike_gauge_regularization_sweep.json"
with open(fname, "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {fname}")
print("Done!")
