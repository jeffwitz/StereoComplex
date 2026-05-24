#!/usr/bin/env python3
"""Run gauge regularization sweep with proper Hessian+TPS completed data.

Equivalent to notebook 09 sections 1–4 + 10.3 with RUN_SWEEP=True.
Generates: pareto_gauge_regularization.png + zernike_gauge_regularization_sweep.json
"""

from __future__ import annotations
import json, sys, time, math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from cv2 import aruco

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"
OUT.mkdir(parents=True, exist_ok=True)

from stereocomplex.core.model_compact.zernike import eval_real_zernike, zernike_modes
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients,
)
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_constrained_zernike_rayfield,
    estimate_initial_poses_from_central_pinhole,
)

# ═══════════════════════════════════════════════════════════════
# Parameters (same as notebook 09)
# ═══════════════════════════════════════════════════════════════
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
NCX, NCY, SQR = 16, 12, 0.3
IMG_SIZE = (2048, 2048)
W, H = IMG_SIZE
DICT_NAME = "DICT_6X6_250"
FX = 25600.0
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════
# 1 — Detection + Hessian completion + TPS denoising
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1 — ChArUco detection + Hessian completion + TPS")
print("=" * 60)

dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
ocv_board = aruco.CharucoBoard((NCX, NCY), SQR, SQR / 2, dictionary)
ocv_board.setLegacyPattern(True)
chess3 = ocv_board.getChessboardCorners()

board_ids = ocv_board.getIds().ravel()
board_obj = ocv_board.getObjPoints()
id_to_obj = {int(board_ids[i]): np.asarray(board_obj[i], dtype=np.float64)[:, :2]
             for i in range(len(board_ids))}

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

aruco_det = aruco.ArucoDetector(dictionary, params)
charuco_det = aruco.CharucoDetector(ocv_board)

# --- Hessian helpers (same as notebook) ---
def abs_det_hessian(gray, sigma=9.0):
    f = gray.astype(np.float32)
    if f.max() > 2: f /= 255.0
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    Ixx = cv2.Sobel(f, cv2.CV_64F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(f, cv2.CV_64F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(f, cv2.CV_64F, 1, 1, ksize=3)
    return np.abs(Ixx * Iyy - Ixy * Ixy)

def otsu_mask(response):
    r8 = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask

def blob_barycentre(mask, xi, yi, d, prefer_largest=True):
    h, w = mask.shape
    x0 = max(0, int(xi)-d); x1 = min(w, int(xi)+d)
    y0 = max(0, int(yi)-d); y1 = min(h, int(yi)+d)
    if x1 <= x0+2 or y1 <= y0+2: return math.nan, math.nan, math.nan
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if nl <= 1: return math.nan, math.nan, math.nan
    if prefer_largest:
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
    else:
        cx_c = xi - x0; cy_c = yi - y0
        best_k, best_d2 = 1, float("inf")
        for k in range(1, nl):
            sx = stats[k, cv2.CC_STAT_LEFT]; sy = stats[k, cv2.CC_STAT_TOP]
            sw = stats[k, cv2.CC_STAT_WIDTH]; sh = stats[k, cv2.CC_STAT_HEIGHT]
            d2 = (sx+sw/2-cx_c)**2 + (sy+sh/2-cy_c)**2
            if d2 < best_d2: best_d2 = d2; best_k = k
        k = best_k
    m = cv2.moments((labels == k).astype(np.uint8), binaryImage=True)
    if m["m00"] < 1e-10: return math.nan, math.nan, math.nan
    return (m["m10"]/m["m00"] + x0, m["m01"]/m["m00"] + y0, float(m["m00"]))

def win_spot_2pass(mask, l_step, d_init, xi, yi, prefer_largest):
    for d_search in [d_init, max(d_init, int(l_step*0.5))]:
        xd, yd, area = blob_barycentre(mask, xi, yi, d_search, prefer_largest)
        if not math.isnan(xd): return xd, yd, area
    return math.nan, math.nan, math.nan

def ids_to_grid(ids, ncx=16):
    ids = np.asarray(ids, dtype=np.float32).reshape(-1)
    nx = ncx - 1
    return np.column_stack([ids % nx, ids // nx]).astype(np.float32)

def fit_affine(img_pts, ids_arr, ncx=16):
    img = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    grid = ids_to_grid(np.asarray(ids_arr, dtype=np.int32).reshape(-1), ncx)
    A, _ = cv2.estimateAffine2D(grid, img, method=cv2.LMEDS)
    if A is None:
        X = np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])
        At, *_ = np.linalg.lstsq(X, img, rcond=None)
        A = At.T.astype(np.float32)
    return A

def project_affine(A, ids, ncx=16):
    grid = ids_to_grid(np.asarray(ids, dtype=np.int32).reshape(-1), ncx)
    return (np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])) @ A.T

def complete_corners_hessian(gray, charuco_corners, charuco_ids, ncx, ncy,
                              marker_corners=None, marker_ids=None,
                              id_to_obj=None, chess3_obj=None):
    nx, ny = ncx-1, ncy-1
    n_corners = nx * ny
    R = abs_det_hessian(gray)
    mask = otsu_mask(R)
    detected = {}
    if charuco_ids is not None and len(charuco_ids) > 0:
        ids_arr = np.asarray(charuco_ids).ravel()
        corners_arr = np.asarray(charuco_corners).reshape(-1, 2)
        for i in range(len(ids_arr)):
            detected[int(ids_arr[i])] = corners_arr[i].astype(np.float64)
    cids = sorted(detected.keys())
    pred_xy = None
    if marker_corners is not None and marker_ids is not None and id_to_obj is not None and chess3_obj is not None:
        obj_xy_list, img_uv_list = [], []
        for i in range(len(marker_ids)):
            mid = int(marker_ids[i].ravel()[0])
            o = id_to_obj.get(mid)
            if o is None: continue
            mc = np.asarray(marker_corners[i], dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] == 4:
                obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            try:
                pred_xy = predict_points_rayfield_tps_robust(
                    np.concatenate(obj_xy_list, axis=0),
                    np.concatenate(img_uv_list, axis=0),
                    chess3_obj[:, :2].astype(np.float64),
                    lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
            except Exception:
                pred_xy = None
    if pred_xy is None and len(cids) >= 3:
        A = fit_affine(np.array([detected[i] for i in cids]), np.array(cids), ncx)
        pred_xy = project_affine(A, np.arange(n_corners), ncx)
    l_step = 50.0
    if len(cids) >= 2:
        dp = float(np.linalg.norm(detected[cids[-1]] - detected[cids[0]]))
        g0 = ids_to_grid(np.array([cids[0]]), ncx)[0]
        g1 = ids_to_grid(np.array([cids[-1]]), ncx)[0]
        dg = float(np.linalg.norm(g1 - g0))
        if dg > 1e-8: l_step = dp / dg
    d_init = max(3, int(l_step * 0.3))
    if len(cids) > 0:
        xA, yA = float(detected[cids[0]][0]), float(detected[cids[0]][1])
        _, _, a_test = win_spot_2pass(mask, l_step, int(l_step*2/3), xA, yA, True)
        if not math.isnan(a_test) and float(a_test) > 0:
            d_init = max(3, int(math.sqrt(float(a_test))))
    result = np.full((n_corners, 2), np.nan)
    for idx in range(n_corners):
        if idx in detected:
            result[idx] = detected[idx]
        else:
            xi, yi = float(pred_xy[idx, 0]), float(pred_xy[idx, 1])
            xd, yd, _ = win_spot_2pass(mask, l_step, d_init, xi, yi, False)
            if not math.isnan(xd): result[idx] = [float(xd), float(yd)]
    for idx in range(n_corners):
        if np.isnan(result[idx, 0]):
            result[idx] = [float(pred_xy[idx, 0]), float(pred_xy[idx, 1])]
    return result

# --- Process frames ---
lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float)
print(f"{len(paired_z)} stereo pairs")

denoised_L, denoised_R = [], []
det_counts_L, det_counts_R = [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L)
    nR = 0 if ids_R is None else len(ids_R)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]
    mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]

    comp_L = complete_corners_hessian(lg, cc_L, ids_L, NCX, NCY,
        marker_corners=mk_c_L, marker_ids=mk_ids_L, id_to_obj=id_to_obj, chess3_obj=chess3)
    comp_R = complete_corners_hessian(rg, cc_R, ids_R, NCX, NCY,
        marker_corners=mk_c_R, marker_ids=mk_ids_R, id_to_obj=id_to_obj, chess3_obj=chess3)

    for mk_c, mk_ids, comp, out in [
        (mk_c_L, mk_ids_L, comp_L, denoised_L),
        (mk_c_R, mk_ids_R, comp_R, denoised_R),
    ]:
        obj_xy_list, img_uv_list = [], []
        if mk_ids is not None:
            for i in range(len(mk_ids)):
                mid = int(mk_ids[i].ravel()[0])
                o = id_to_obj.get(mid)
                if o is None: continue
                mc = np.asarray(mk_c[i], dtype=np.float64).reshape(-1, 2)
                if mc.shape[0] == 4:
                    obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            obj_xy = np.concatenate(obj_xy_list, axis=0)
            img_uv = np.concatenate(img_uv_list, axis=0)
            pred = predict_points_rayfield_tps_robust(
                obj_xy, img_uv, chess3[:, :2].astype(np.float64),
                lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
        else:
            pred = comp
        # TPS re-denoising on completed 165 corners
        re_denoised = predict_points_rayfield_tps_robust(
            chess3[:, :2].astype(np.float64), pred.astype(np.float64),
            chess3[:, :2].astype(np.float64),
            lam=3.0, huber_c=1.5, iters=2, ransac_reproj_px=2.0)
        out.append(re_denoised)

    print(f"  {z_str}: L {nL}→165  R {nR}→165")
    det_counts_L.append(nL); det_counts_R.append(nR)

print(f"Completed: {len(denoised_L)} frames, all 165 corners")

# ═══════════════════════════════════════════════════════════════
# 2 — Constrained Zernike fit O(0)+d(2)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2 — Constrained Zernike O(0)+d(2)")
print("=" * 60)

obj_pts = chess3.astype(np.float64)
left_pixels = [dn.astype(np.float64) for dn in denoised_L]
right_pixels = [dn.astype(np.float64) for dn in denoised_R]
point_indices = [np.arange(165, dtype=int) for _ in range(len(paired_z))]

rvecs, tvecs = [], []
for lp in left_pixels:
    s, rv, tv = cv2.solvePnP(
        obj_pts.astype(np.float32), lp.astype(np.float32),
        K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))

obs = CharucoObservationSet(
    object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
    left_pixels=left_pixels, right_pixels=right_pixels,
    point_indices=point_indices, noise_std_px=0.0, image_size=IMG_SIZE)

t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(
    obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(),
    max_order_d=2, max_nfev=500, origin_reg_weight=0.0)
print(f"  RMS={zd.ray_rms_mm:.6f} mm  nfev={zd.nfev}  time={time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# 3 — Gauge-regularized full-pose sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3 — Gauge-regularized sweep")
print("=" * 60)

prior_L_d = lf.direction_coeffs.copy()
prior_R_d = rf.direction_coeffs.copy()
print(f"Prior Z0 d_L = {prior_L_d[0]}")

config2 = ZernikeOriginFieldConfig(image_size=IMG_SIZE, max_order=2)
modes2 = config2.modes()
n_modes2 = len(modes2)
n_zernike = n_modes2 * 6

# Group observations
uL_all, vL_all, idxL_all, poseL_all = [], [], [], []
uR_all, vR_all, idxR_all, poseR_all = [], [], [], []
for pi in range(len(obs.left_pixels)):
    lp = obs.left_pixels[pi]; rp = obs.right_pixels[pi]
    idx = obs.point_indices[pi]
    nL = lp.shape[0]; nR = rp.shape[0]
    if nL > 0:
        uL_all.append(lp[:, 0]); vL_all.append(lp[:, 1]); idxL_all.append(idx[:nL])
        poseL_all.append(np.full(nL, pi, dtype=int))
    if nR > 0:
        uR_all.append(rp[:, 0]); vR_all.append(rp[:, 1]); idxR_all.append(idx[:nR])
        poseR_all.append(np.full(nR, pi, dtype=int))

uL = np.concatenate(uL_all); vL = np.concatenate(vL_all)
idxL_arr = np.concatenate(idxL_all); poseL = np.concatenate(poseL_all)
uR = np.concatenate(uR_all); vR = np.concatenate(vR_all)
idxR_arr = np.concatenate(idxR_all); poseR = np.concatenate(poseR_all)
obj_pts_arr = obs.object_points_mm
n_poses = len(obs.left_pixels)

def _precompute(u_arr, v_arr, K_):
    xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(W-1) - 1.0
    zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(H-1) - 1.0
    rho = np.sqrt(xi*xi + zeta*zeta) / np.sqrt(2.0)
    theta = np.arctan2(zeta, xi)
    A = np.empty((rho.size, n_modes2), dtype=np.float64)
    for j, mode in enumerate(modes2):
        A[:, j] = eval_real_zernike(mode, rho, theta)
    Kk = np.asarray(K_, dtype=np.float64).reshape(3, 3)
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
    mL = poseL == pi; mR = poseR == pi
    if mL.any():
        g = _G(); g.pose_idx = pi
        g.A, g.d0 = _precompute(uL[mL], vL[mL], K)
        g.X_local = obj_pts_arr[idxL_arr[mL]]; groups_L.append(g)
    if mR.any():
        g = _G(); g.pose_idx = pi
        g.A, g.d0 = _precompute(uR[mR], vR[mR], K.copy())
        g.X_local = obj_pts_arr[idxR_arr[mR]]; groups_R.append(g)

R_est = [opt_R[pi] for pi in range(n_poses)]
t_est = [opt_t[pi] for pi in range(n_poses)]
x0_poses = []
for pi in range(n_poses):
    rv = Rotation.from_matrix(R_est[pi]).as_rotvec()
    x0_poses.append(rv)
    x0_poses.append(np.asarray(t_est[pi], dtype=np.float64).reshape(3))
x0_poses_arr = np.concatenate(x0_poses)

def _chan_residuals(origin_c, dir_c, pose_params, groups_):
    blocks = []
    for g in groups_:
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

def fit_one(sigma_z0_deg, sigma_z1_deg, max_nfev=400):
    deg_to_rad = np.pi / 180.0
    sz0 = max(sigma_z0_deg, 1e-6) * deg_to_rad
    sz1 = max(sigma_z1_deg, 1e-6) * deg_to_rad

    prior_mask = np.zeros(n_modes2, dtype=bool)
    prior_sigmas = np.zeros(n_modes2, dtype=np.float64)
    prior_mask[0] = True; prior_sigmas[0] = sz0
    prior_mask[1] = True; prior_sigmas[1] = sz1
    prior_mask[2] = True; prior_sigmas[2] = sz1

    def residuals_reg(x):
        cL = x[:n_zernike]; cR = x[n_zernike:2*n_zernike]
        oL = cL[:n_zernike//2].reshape(n_modes2, 3)
        dL = cL[n_zernike//2:].reshape(n_modes2, 3)
        oR = cR[:n_zernike//2].reshape(n_modes2, 3)
        dR = cR[n_zernike//2:].reshape(n_modes2, 3)
        pp = x[2*n_zernike:]
        rL = _chan_residuals(oL, dL, pp, groups_L)
        rR = _chan_residuals(oR, dR, pp, groups_R)
        reg = [rL, rR, np.sqrt(1e-3)*oL[:,2], np.sqrt(1e-3)*oR[:,2]]
        for m in range(n_modes2):
            if not prior_mask[m]: continue
            s = prior_sigmas[m]
            for c in range(3):
                reg.append(np.array([(dL[m, c] - prior_L_d[m, c]) / s]))
                reg.append(np.array([(dR[m, c] - prior_R_d[m, c]) / s]))
        return np.concatenate(reg)

    n_half = n_zernike // 2
    o_lo = np.full(n_half, -np.inf); o_hi = np.full(n_half, np.inf)
    for j in range(2, n_half, 3): o_lo[j] = -20.0; o_hi[j] = 20.0
    d_lo = np.full(n_half, -0.5); d_hi = np.full(n_half, 0.5)
    c_lo = np.concatenate([o_lo, d_lo]); c_hi = np.concatenate([o_hi, d_hi])
    bounds = (np.concatenate([c_lo, c_lo, x0_poses_arr - 0.3]),
              np.concatenate([c_hi, c_hi, x0_poses_arr + 0.3]))
    x0 = np.concatenate([np.zeros(n_zernike), np.zeros(n_zernike), x0_poses_arr])
    sol = least_squares(residuals_reg, x0=x0, bounds=bounds, method="trf",
                        loss="linear", max_nfev=int(max_nfev),
                        xtol=1e-8, ftol=1e-8, gtol=1e-8)
    def _field(cfs, Kk):
        a = np.asarray(cfs).reshape(-1)
        return ZernikeRayField(K=Kk, config=config2,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=a[:n_modes2*3].reshape(n_modes2, 3),
                direction_coeffs=a[n_modes2*3:].reshape(n_modes2, 3)))
    return _field(sol.x[:n_zernike], K), _field(sol.x[n_zernike:2*n_zernike], K.copy()), sol

# ── Sweep ──
sweep_runs = [
    ("full_pose_baseline", 100.0, 100.0),
    ("z0_0.05deg", 0.05, 100.0),
    ("z0_0.1deg", 0.1, 100.0),
    ("z0_0.2deg", 0.2, 100.0),
    ("z0_0.5deg", 0.5, 100.0),
    ("z0_1.0deg", 1.0, 100.0),
    ("z0_2.0deg", 2.0, 100.0),
    ("z0z1_0.1_0.5", 0.1, 0.5),
    ("z0z1_0.2_0.5", 0.2, 0.5),
    ("z0z1_0.2_1.0", 0.2, 1.0),
    ("z0z1_0.5_1.0", 0.5, 1.0),
    ("z0z1_0.5_2.0", 0.5, 2.0),
]

print(f"\n{'Run':>22s}  {'RMSmm':>9s}  {'Z0△°':>7s}  {'Z1△°':>7s}  "
      f"{'b_mm':>7s}  {'θ°':>6s}  {'dy_ranL':>8s}  {'NFEV':>5s}  {'s':>4s}")
print("-" * 100)

sweep_results = []
u_g, v_g = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
u_gf, v_gf = u_g.ravel(), v_g.ravel()

for label, sz0, sz1 in sweep_runs:
    t0 = time.time()
    lf_s, rf_s, sol = fit_one(sz0, sz1, max_nfev=400)
    elapsed = time.time() - t0

    dL_s = lf_s.direction_coeffs; dR_s = rf_s.direction_coeffs
    dz0_L = np.linalg.norm(dL_s[0] - prior_L_d[0])
    dz0_R = np.linalg.norm(dR_s[0] - prior_R_d[0])
    dz1_L = 0.5*(np.linalg.norm(dL_s[1]-prior_L_d[1]) + np.linalg.norm(dL_s[2]-prior_L_d[2]))
    dz1_R = 0.5*(np.linalg.norm(dR_s[1]-prior_R_d[1]) + np.linalg.norm(dR_s[2]-prior_R_d[2]))
    drift_z0 = float(np.degrees(0.5*(dz0_L + dz0_R)))
    drift_z1 = float(np.degrees(0.5*(dz1_L + dz1_R)))

    O_L, d_L = lf_s.ray(u_gf, v_gf)
    O_R, d_R = rf_s.ray(u_gf, v_gf)
    uc, vc = np.array([1024.]), np.array([1024.])
    _, dL_c = lf_s.ray(uc, vc); _, dR_c = rf_s.ray(uc, vc)
    b_val = float(np.linalg.norm(np.mean(O_R, axis=0) - np.mean(O_L, axis=0)))
    theta_val = float(np.degrees(np.arccos(np.clip(np.dot(dL_c[0], dR_c[0]), -1.0, 1.0))))
    dy_range_L = float(np.max(d_L[:, 1]) - np.min(d_L[:, 1]))

    def geo_rms(x):
        cL = x[:n_zernike]; cR = x[n_zernike:2*n_zernike]
        oL = cL[:n_zernike//2].reshape(n_modes2, 3)
        dLc = cL[n_zernike//2:].reshape(n_modes2, 3)
        oR = cR[:n_zernike//2].reshape(n_modes2, 3)
        dRc = cR[n_zernike//2:].reshape(n_modes2, 3)
        pp = x[2*n_zernike:]
        rL = _chan_residuals(oL, dLc, pp, groups_L)
        rR = _chan_residuals(oR, dRc, pp, groups_R)
        return np.concatenate([rL, rR])
    r_geo = geo_rms(sol.x)
    rms_val = float(np.sqrt(np.mean(r_geo**2)))

    result = {
        "label": label, "sigma_z0_deg": sz0, "sigma_z1_deg": sz1,
        "ray_rms_mm": rms_val, "converged": bool(sol.success), "nfev": int(sol.nfev),
        "drift_z0_deg": drift_z0, "drift_z1_deg": drift_z1,
        "baseline_mm": b_val, "convergence_angle_deg": theta_val,
        "dy_range_L": dy_range_L, "time_s": elapsed,
    }
    sweep_results.append(result)
    print(f"  {label:22s}  {rms_val:9.6f}  {drift_z0:7.3f}  {drift_z1:7.3f}  "
          f"{b_val:7.1f}  {theta_val:6.1f}  {dy_range_L:8.4f}  {sol.nfev:5d}  {elapsed:4.0f}")

print(f"\n  {'constrained (ref)':22s}  {zd.ray_rms_mm:9.6f}  {0.0:7.3f}  {0.0:7.3f}")

# ── Pareto ──
pareto = []
for i, r in enumerate(sweep_results):
    dominated = any(
        r2["ray_rms_mm"] <= r["ray_rms_mm"] and r2["drift_z0_deg"] <= r["drift_z0_deg"]
        and (r2["ray_rms_mm"] < r["ray_rms_mm"] or r2["drift_z0_deg"] < r["drift_z0_deg"])
        for j, r2 in enumerate(sweep_results) if i != j)
    if not dominated:
        pareto.append(r)

print(f"\nPareto-optimal ({len(pareto)}):")
for r in sorted(pareto, key=lambda x: x["ray_rms_mm"]):
    print(f"  {r['label']:22s}  RMS={r['ray_rms_mm']:.6f}mm  "
          f"Z0△={r['drift_z0_deg']:.3f}°  "
          f"b={r['baseline_mm']:.1f}mm  θ={r['convergence_angle_deg']:.1f}°")

# ═══════════════════════════════════════════════════════════════
# Save + Plot
# ═══════════════════════════════════════════════════════════════
with open(OUT / "zernike_gauge_regularization_sweep.json", "w") as f:
    json.dump({
        "description": "Gauge-regularized full-pose Zernike sweep — Z0/Z1 direction anchor",
        "constrained_rms_mm": zd.ray_rms_mm,
        "sweep": sweep_results,
        "pareto_optimal": [r["label"] for r in pareto],
    }, f, indent=2)
print(f"\nSaved: {OUT / 'zernike_gauge_regularization_sweep.json'}")

# ── Pareto plot ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1) RMS vs Z0 drift
ax = axes[0]
all_rms = [r["ray_rms_mm"] for r in sweep_results]
all_z0 = [r["drift_z0_deg"] for r in sweep_results]
pareto_rms = [r["ray_rms_mm"] for r in pareto]
pareto_z0 = [r["drift_z0_deg"] for r in pareto]

ax.scatter(all_rms, all_z0, c="steelblue", s=60, zorder=3, label="all runs")
ax.scatter(pareto_rms, pareto_z0, c="darkorange", s=120, zorder=4, edgecolors="black", linewidth=0.5, label="Pareto-optimal")
# Connect Pareto points
po_sorted = sorted(pareto, key=lambda r_: r_["ray_rms_mm"])
if len(po_sorted) >= 2:
    px = [r_["ray_rms_mm"] for r_ in po_sorted]
    py = [r_["drift_z0_deg"] for r_ in po_sorted]
    ax.plot(px, py, "darkorange", lw=1.5, alpha=0.6, zorder=2)
for r in pareto:
    ax.annotate(r["label"].replace("_", "\n"), (r["ray_rms_mm"], r["drift_z0_deg"]),
                textcoords="offset points", xytext=(8, 6), fontsize=6, alpha=0.8)
ax.axvline(x=zd.ray_rms_mm, color="gray", ls="--", alpha=0.5, label="constrained ref")
ax.set_xlabel("Ray RMS (mm)")
ax.set_ylabel("Z₀ drift (°)")
ax.set_title("Pareto frontier: RMS vs gauge drift")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2) Baseline stability
ax = axes[1]
for r in sweep_results:
    sigma_label = f"σ₀={r['sigma_z0_deg']:.2f}" if r["sigma_z1_deg"] > 50 else f"σ₀={r['sigma_z0_deg']:.2f},σ₁={r['sigma_z1_deg']:.1f}"
    ax.plot(r["drift_z0_deg"], r["baseline_mm"], "o", ms=8, alpha=0.7)
# Reference line from constrained fit
O_cL, d_cL = lf.ray(u_gf, v_gf); O_cR, d_cR = rf.ray(u_gf, v_gf)
b_constrained = float(np.linalg.norm(np.mean(O_cR, axis=0) - np.mean(O_cL, axis=0)))
_, dL_cc = lf.ray(np.array([1024.]), np.array([1024.]))
_, dR_cc = rf.ray(np.array([1024.]), np.array([1024.]))
theta_constrained = float(np.degrees(np.arccos(np.clip(np.dot(dL_cc[0], dR_cc[0]), -1.0, 1.0))))
ax.axhline(y=b_constrained, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Z₀ drift (°)")
ax.set_ylabel("Baseline (mm)")
ax.set_title("Baseline stability")
ax.grid(True, alpha=0.3)

# 3) Convergence angle stability
ax = axes[2]
for r in sweep_results:
    ax.plot(r["drift_z0_deg"], r["convergence_angle_deg"], "o", ms=8, alpha=0.7)
ax.set_xlabel("Z₀ drift (°)")
ax.set_ylabel("Convergence angle (°)")
ax.set_title("Convergence angle stability")
ax.axhline(y=theta_constrained, color="gray", ls="--", alpha=0.5)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fname = OUT / "pareto_gauge_regularization.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
print(f"Saved: {fname}")
plt.close()
print("Done!")
