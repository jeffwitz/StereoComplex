#!/usr/bin/env python3
"""Test global Plücker arm alignment for the CMO telecentric model.

Fits per-channel SE(3) transforms on top of the telecentric rayfield
against the stable Zernike O(0)+d(2) oracle.

Hypothesis: the Z0-dominated residual is a global line-bundle misalignment,
not a spatial field distortion.
"""

from __future__ import annotations
import json, sys, time, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"
OUT.mkdir(parents=True, exist_ok=True)

import cv2
from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _ray_rms, _normalize
from stereocomplex.physics.model_selection import rayfield_two_plane_residuals, _grid_pixels

# ═══════════════════════════════════════════════════════════════
# 1 — Pipeline (same as before)
# ═══════════════════════════════════════════════════════════════
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
NCX, NCY, SQR = 16, 12, 0.3
IMG_SIZE = (2048, 2048); W, H = IMG_SIZE
FX = 25600.0
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

dictionary = aruco.getPredefinedDictionary(getattr(aruco, "DICT_6X6_250"))
ocv_board = aruco.CharucoBoard((NCX, NCY), SQR, SQR / 2, dictionary)
ocv_board.setLegacyPattern(True)
chess3 = ocv_board.getChessboardCorners()
board_ids = ocv_board.getIds().ravel()
board_obj = ocv_board.getObjPoints()
id_to_obj = {int(board_ids[i]): np.asarray(board_obj[i], dtype=np.float64)[:, :2] for i in range(len(board_ids))}

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax = 3, 75
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate, params.maxMarkerPerimeterRate = 0.005, 0.20
params.polygonalApproxAccuracyRate, params.minCornerDistanceRate = 0.08, 0.02
params.minDistanceToBorder = 1
aruco_det = aruco.ArucoDetector(dictionary, params)
charuco_det = aruco.CharucoDetector(ocv_board)

def abs_det_hessian(gray, sigma=9.0):
    f = gray.astype(np.float32)
    if f.max() > 2: f /= 255.0
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return np.abs(cv2.Sobel(f, cv2.CV_64F, 2, 0, ksize=3) * cv2.Sobel(f, cv2.CV_64F, 0, 2, ksize=3)
                  - cv2.Sobel(f, cv2.CV_64F, 1, 1, ksize=3)**2)

def otsu_mask(response):
    r8 = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask

def blob_barycentre(mask, xi, yi, d, prefer_largest=True):
    h, w = mask.shape
    x0, x1 = max(0, int(xi)-d), min(w, int(xi)+d)
    y0, y1 = max(0, int(yi)-d), min(h, int(yi)+d)
    if x1 <= x0+2 or y1 <= y0+2: return math.nan, math.nan, math.nan
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if nl <= 1: return math.nan, math.nan, math.nan
    if prefer_largest: k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        cx_c, cy_c = xi-x0, yi-y0; best_k, best_d2 = 1, float("inf")
        for k in range(1, nl):
            sx, sy = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            sw, sh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            d2 = (sx+sw/2-cx_c)**2 + (sy+sh/2-cy_c)**2
            if d2 < best_d2: best_d2, best_k = d2, k
        k = best_k
    m = cv2.moments((labels == k).astype(np.uint8), binaryImage=True)
    if m["m00"] < 1e-10: return math.nan, math.nan, math.nan
    return (m["m10"]/m["m00"]+x0, m["m01"]/m["m00"]+y0, float(m["m00"]))

def win_spot_2pass(mask, l_step, d_init, xi, yi, prefer_largest):
    for d_search in [d_init, max(d_init, int(l_step*0.5))]:
        xd, yd, area = blob_barycentre(mask, xi, yi, d_search, prefer_largest)
        if not math.isnan(xd): return xd, yd, area
    return math.nan, math.nan, math.nan

def ids_to_grid(ids, ncx=16):
    ids_arr = np.asarray(ids, dtype=np.float32).reshape(-1); nx = ncx - 1
    return np.column_stack([ids_arr % nx, ids_arr // nx]).astype(np.float32)

def fit_affine(img_pts, ids_arr, ncx=16):
    img = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    grid = ids_to_grid(np.asarray(ids_arr, dtype=np.int32).reshape(-1), ncx)
    A, _ = cv2.estimateAffine2D(grid, img, method=cv2.LMEDS)
    if A is None:
        X = np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])
        A = np.linalg.lstsq(X, img, rcond=None)[0].T.astype(np.float32)
    return A

def project_affine(A, ids, ncx=16):
    grid = ids_to_grid(np.asarray(ids, dtype=np.int32).reshape(-1), ncx)
    return (np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])) @ A.T

def complete_corners_hessian(gray, cc, ids, ncx, ncy, marker_corners=None, marker_ids=None):
    nx, ny = ncx-1, ncy-1; n_corners = nx * ny
    R = abs_det_hessian(gray); mask = otsu_mask(R)
    detected = {}
    if ids is not None and len(ids) > 0:
        for i in range(len(np.asarray(ids).ravel())):
            detected[int(np.asarray(ids).ravel()[i])] = np.asarray(cc).reshape(-1, 2)[i].astype(np.float64)
    cids = sorted(detected.keys())
    pred_xy = None
    if marker_corners is not None and marker_ids is not None:
        obj_xy_list, img_uv_list = [], []
        for i in range(len(marker_ids)):
            mid = int(marker_ids[i].ravel()[0]); o = id_to_obj.get(mid)
            if o is None: continue
            mc = np.asarray(marker_corners[i], dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] == 4: obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            try:
                pred_xy = predict_points_rayfield_tps_robust(
                    np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0),
                    chess3[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
            except Exception: pred_xy = None
    if pred_xy is None and len(cids) >= 3:
        A = fit_affine(np.array([detected[i] for i in cids]), np.array(cids), ncx)
        pred_xy = project_affine(A, np.arange(n_corners), ncx)
    l_step = 50.0
    if len(cids) >= 2:
        dp = float(np.linalg.norm(detected[cids[-1]] - detected[cids[0]]))
        g0, g1 = ids_to_grid(np.array([cids[0]]), ncx)[0], ids_to_grid(np.array([cids[-1]]), ncx)[0]
        dg = float(np.linalg.norm(g1 - g0))
        if dg > 1e-8: l_step = dp / dg
    d_init = max(3, int(l_step * 0.3))
    if len(cids) > 0:
        xA, yA = float(detected[cids[0]][0]), float(detected[cids[0]][1])
        _, _, a_test = win_spot_2pass(mask, l_step, int(l_step*2/3), xA, yA, True)
        if not math.isnan(a_test) and float(a_test) > 0: d_init = max(3, int(math.sqrt(float(a_test))))
    result = np.full((n_corners, 2), np.nan)
    for idx in range(n_corners):
        if idx in detected: result[idx] = detected[idx]
        else:
            xi, yi = float(pred_xy[idx, 0]), float(pred_xy[idx, 1])
            xd, yd, _ = win_spot_2pass(mask, l_step, d_init, xi, yi, False)
            if not math.isnan(xd): result[idx] = [float(xd), float(yd)]
    for idx in range(n_corners):
        if np.isnan(result[idx, 0]): result[idx] = [float(pred_xy[idx, 0]), float(pred_xy[idx, 1])]
    return result

lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float)

denoised_L, denoised_R = [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]
    mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]
    comp_L = complete_corners_hessian(lg, cc_L, ids_L, NCX, NCY, mk_c_L, mk_ids_L)
    comp_R = complete_corners_hessian(rg, cc_R, ids_R, NCX, NCY, mk_c_R, mk_ids_R)
    for mk_c, mk_ids, comp, out in [(mk_c_L, mk_ids_L, comp_L, denoised_L), (mk_c_R, mk_ids_R, comp_R, denoised_R)]:
        obj_xy_list, img_uv_list = [], []
        if mk_ids is not None:
            for i in range(len(mk_ids)):
                mid = int(mk_ids[i].ravel()[0]); o = id_to_obj.get(mid)
                if o is None: continue
                mc = np.asarray(mk_c[i], dtype=np.float64).reshape(-1, 2)
                if mc.shape[0] == 4: obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            pred = predict_points_rayfield_tps_robust(
                np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0),
                chess3[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
        else: pred = comp
        re_denoised = predict_points_rayfield_tps_robust(
            chess3[:, :2].astype(np.float64), pred.astype(np.float64),
            chess3[:, :2].astype(np.float64), lam=3.0, huber_c=1.5, iters=2, ransac_reproj_px=2.0)
        out.append(re_denoised)
    nL = 0 if ids_L is None else len(ids_L); nR = 0 if ids_R is None else len(ids_R)
    print(f"  {z_str}: L {nL}→165  R {nR}→165")

obj_pts = chess3.astype(np.float64)
left_pixels = [dn.astype(np.float64) for dn in denoised_L]
right_pixels = [dn.astype(np.float64) for dn in denoised_R]
point_indices = [np.arange(165, dtype=int) for _ in range(len(paired_z))]
rvecs, tvecs = [], []
for lp in left_pixels:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))
obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
                             left_pixels=left_pixels, right_pixels=right_pixels, point_indices=point_indices,
                             noise_std_px=0.0, image_size=IMG_SIZE)

# Zernike
t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(),
                                                              max_order_d=2, max_nfev=500, origin_reg_weight=0.0)
print(f"\nZernike RMS={zd.ray_rms_mm:.6f} mm  time={time.time()-t0:.0f}s")

# Telecentric baseline
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(len(opt_t))]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
f_obj_est = WD_est - float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))

x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dcL[0,1],
                   0., 0., 0., 0., 0., 0.], dtype=np.float64)
lo_tel = np.array([1., 1., 0., 0., 0., 20., 0., -0.3, -10., -10., -10., -10., -10., -10.], dtype=np.float64)
hi_tel = np.array([500., 1000., 200., 2048., 2048., 200., 0.5, 0.3, 10., 10., 10., 10., 10., 10.], dtype=np.float64)
support = _grid_pixels(IMG_SIZE, (12, 9))

def build_tel(x):
    return CMOTelecentricStereoModel.from_parameter_vector(x, pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
def res_tel(x):
    m = build_tel(x); l = m.channel("left"); r = m.channel("right")
    return np.concatenate([rayfield_two_plane_residuals(lf, l, support, z_planes=(50., 80.)),
                           rayfield_two_plane_residuals(rf, r, support, z_planes=(50., 80.))])
sol_tel = least_squares(res_tel, x0=x0_tel, bounds=(lo_tel, hi_tel), loss="linear", max_nfev=500,
                         xtol=1e-10, ftol=1e-10, gtol=1e-10)
m_tel = build_tel(sol_tel.x)
print(f"Telecentric baseline: Ray RMS={float(np.sqrt(0.5*(_ray_rms(rayfield_two_plane_residuals(lf, m_tel.channel('left'), support, z_planes=(50.,80.)))**2+_ray_rms(rayfield_two_plane_residuals(rf, m_tel.channel('right'), support, z_planes=(50.,80.)))**2))):.4f} mm")

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — Global Plücker arm alignment
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1: Global Plücker arm alignment")
print("=" * 60)

# Reference: Zernike rayfield on a grid
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf); OzR, dzR = rf.ray(uf, vf)
mzL = np.cross(OzL, dzL); mzR = np.cross(OzR, dzR)

# Telecentric rays on the same grid
OmL, dmL = m_tel.ray(uf, vf, "left"); OmR, dmR = m_tel.ray(uf, vf, "right")
mmL = np.cross(OmL, dmL); mmR = np.cross(OmR, dmR)

# Baselines (no alignment)
dot_L = np.clip(np.sum(dzL*dmL, axis=1), -1, 1); dot_R = np.clip(np.sum(dzR*dmR, axis=1), -1, 1)
dir_rms_L = float(np.sqrt(np.mean(np.degrees(np.arccos(dot_L))**2)))
dir_rms_R = float(np.sqrt(np.mean(np.degrees(np.arccos(dot_R))**2)))
mom_rms_L = float(np.sqrt(np.mean(np.linalg.norm(mzL-mmL, axis=1)**2)))
mom_rms_R = float(np.sqrt(np.mean(np.linalg.norm(mzR-mmR, axis=1)**2)))
print(f"Baseline — Dir RMS: L={dir_rms_L:.4f}°  R={dir_rms_R:.4f}°  Mom RMS: L={mom_rms_L:.4f} mm  R={mom_rms_R:.4f} mm")

def two_plane_residual_plucker(d_model, m_model, d_ref, O_ref, d_ref2, z_planes=(50., 80.)):
    """Two-plane residual from Plücker coordinates vs reference rays."""
    # Reference rays: (O_ref, d_ref2) — d_ref2 is same as d_ref (just for clarity)
    blocks = []
    for z in z_planes:
        t_ref = (z - O_ref[:, 2]) / d_ref[:, 2]
        P_ref = O_ref + t_ref[:, None] * d_ref
        # Model ray intersection: need O from (d_model, m_model)
        # m = O × d => O = (m × d) / |d|²  (perpendicular component)
        # Actually any O satisfying O×d = m works. Use O = d × m (since d·O=0)
        O_model = np.cross(d_model, m_model)  # O ⟂ d
        t_model = (z - O_model[:, 2]) / d_model[:, 2]
        P_model = O_model + t_model[:, None] * d_model
        blocks.append((P_ref - P_model).reshape(-1))
    return np.concatenate(blocks)

def apply_se3(d, m, rotvec, t):
    """Apply SE(3) transform to Plücker lines (d, m)."""
    R = Rotation.from_rotvec(rotvec).as_matrix()
    d_new = (R @ d.T).T  # (N, 3)
    m_new = (R @ m.T).T + np.cross(t[None, :], d_new)  # (N, 3)
    return d_new, m_new

# ── Test variants ──
results_se3 = []

# 1. Rotation only per channel
def fit_rotation(channel):
    d_ref = dzL if channel == "left" else dzR
    O_ref = OzL if channel == "left" else OzR
    m_ref = mzL if channel == "left" else mzR
    d_base = dmL if channel == "left" else dmR
    m_base = mmL if channel == "left" else mmR

    def res(x):
        d_new, m_new = apply_se3(d_base, m_base, x[:3], np.zeros(3))
        return two_plane_residual_plucker(d_new, m_new, d_ref, O_ref, d_ref)

    x0 = np.zeros(3); lo = np.full(3, -0.05); hi = np.full(3, 0.05)  # ±3°
    sol = least_squares(res, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=200, xtol=1e-10)
    rv = sol.x; angle = float(np.linalg.norm(rv))
    d_new, m_new = apply_se3(d_base, m_base, rv, np.zeros(3))
    return d_new, m_new, angle, rv

d_rotL, m_rotL, angL, rvL = fit_rotation("left")
d_rotR, m_rotR, angR, rvR = fit_rotation("right")
res_rot = np.concatenate([two_plane_residual_plucker(d_rotL, m_rotL, dzL, OzL, dzL),
                           two_plane_residual_plucker(d_rotR, m_rotR, dzR, OzR, dzR)])
rms_rot = float(np.sqrt(np.mean(_ray_rms(res_rot)**2)))
# Direction and moment after rotation
dot_rotL = np.clip(np.sum(dzL*d_rotL, axis=1), -1, 1)
dot_rotR = np.clip(np.sum(dzR*d_rotR, axis=1), -1, 1)
results_se3.append({
    "name": "rotation only (3+3)", "rms_mm": rms_rot,
    "dir_L": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_rotL))**2))),
    "dir_R": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_rotR))**2))),
    "mom_L": float(np.sqrt(np.mean(np.linalg.norm(mzL-np.cross(np.cross(d_rotL,m_rotL),d_rotL), axis=1)**2))),
    "mom_R": float(np.sqrt(np.mean(np.linalg.norm(mzR-np.cross(np.cross(d_rotR,m_rotR),d_rotR), axis=1)**2))),
    "rot_L_deg": float(np.degrees(angL)), "rot_R_deg": float(np.degrees(angR)),
})
print(f"\n  Rotation only: RMS={rms_rot:.4f} mm  L={np.degrees(angL):.3f}°  R={np.degrees(angR):.3f}°")

# 2. Translation only per channel
def fit_translation(channel):
    d_ref = dzL if channel == "left" else dzR
    O_ref = OzL if channel == "left" else OzR
    d_base = dmL if channel == "left" else dmR
    m_base = mmL if channel == "left" else mmR

    def res(x):
        d_new, m_new = apply_se3(d_base, m_base, np.zeros(3), x[:3])
        return two_plane_residual_plucker(d_new, m_new, d_ref, O_ref, d_ref)

    x0 = np.zeros(3); lo = np.full(3, -2.0); hi = np.full(3, 2.0)
    sol = least_squares(res, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=200, xtol=1e-10)
    t = sol.x
    d_new, m_new = apply_se3(d_base, m_base, np.zeros(3), t)
    return d_new, m_new, t

d_trL, m_trL, tL = fit_translation("left")
d_trR, m_trR, tR = fit_translation("right")
res_tr = np.concatenate([two_plane_residual_plucker(d_trL, m_trL, dzL, OzL, dzL),
                          two_plane_residual_plucker(d_trR, m_trR, dzR, OzR, dzR)])
rms_tr = float(np.sqrt(np.mean(_ray_rms(res_tr)**2)))
dot_trL = np.clip(np.sum(dzL*d_trL, axis=1), -1, 1)
dot_trR = np.clip(np.sum(dzR*d_trR, axis=1), -1, 1)
results_se3.append({
    "name": "translation only (3+3)", "rms_mm": rms_tr,
    "dir_L": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_trL))**2))),
    "dir_R": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_trR))**2))),
    "t_L_mm": [float(tL[0]), float(tL[1]), float(tL[2])],
    "t_R_mm": [float(tR[0]), float(tR[1]), float(tR[2])],
})
print(f"  Translation only: RMS={rms_tr:.4f} mm  tL={tL}  tR={tR}")

# 3. Full SE(3) per channel
def fit_se3(channel):
    d_ref = dzL if channel == "left" else dzR
    O_ref = OzL if channel == "left" else OzR
    d_base = dmL if channel == "left" else dmR
    m_base = mmL if channel == "left" else mmR

    def res(x):
        d_new, m_new = apply_se3(d_base, m_base, x[:3], x[3:6])
        return two_plane_residual_plucker(d_new, m_new, d_ref, O_ref, d_ref)

    x0 = np.zeros(6)
    lo = np.array([-0.05, -0.05, -0.05, -2.0, -2.0, -2.0])
    hi = np.array([0.05, 0.05, 0.05, 2.0, 2.0, 2.0])
    sol = least_squares(res, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=300, xtol=1e-10)
    rv, t = sol.x[:3], sol.x[3:6]
    d_new, m_new = apply_se3(d_base, m_base, rv, t)
    return d_new, m_new, float(np.linalg.norm(rv)), rv, t

d_se3L, m_se3L, ang_se3L, rv_se3L, t_se3L = fit_se3("left")
d_se3R, m_se3R, ang_se3R, rv_se3R, t_se3R = fit_se3("right")
res_se3 = np.concatenate([two_plane_residual_plucker(d_se3L, m_se3L, dzL, OzL, dzL),
                           two_plane_residual_plucker(d_se3R, m_se3R, dzR, OzR, dzR)])
rms_se3 = float(np.sqrt(np.mean(_ray_rms(res_se3)**2)))
dot_se3L = np.clip(np.sum(dzL*d_se3L, axis=1), -1, 1)
dot_se3R = np.clip(np.sum(dzR*d_se3R, axis=1), -1, 1)
results_se3.append({
    "name": "full SE(3) (6+6)", "rms_mm": rms_se3,
    "dir_L": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_se3L))**2))),
    "dir_R": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_se3R))**2))),
    "mom_L": float(np.sqrt(np.mean(np.linalg.norm(mzL-np.cross(np.cross(d_se3L,m_se3L),d_se3L), axis=1)**2))),
    "mom_R": float(np.sqrt(np.mean(np.linalg.norm(mzR-np.cross(np.cross(d_se3R,m_se3R),d_se3R), axis=1)**2))),
    "rot_L_deg": float(np.degrees(ang_se3L)), "rot_R_deg": float(np.degrees(ang_se3R)),
    "t_L_mm": [float(t_se3L[0]), float(t_se3L[1]), float(t_se3L[2])],
    "t_R_mm": [float(t_se3R[0]), float(t_se3R[1]), float(t_se3R[2])],
})
print(f"  Full SE(3): RMS={rms_se3:.4f} mm  L rot={np.degrees(ang_se3L):.3f}° t={t_se3L}  R rot={np.degrees(ang_se3R):.3f}° t={t_se3R}")

# 4. Symmetric SE(3): R_L = R_mean @ R_stereo, R_R = R_mean @ R_stereo^{-1}, t_L = -t_R
# (Preserves CMO structure)
def fit_se3_symmetric():
    d_refs = [dzL, dzR]; O_refs = [OzL, OzR]
    d_bases = [dmL, dmR]; m_bases = [mmL, mmR]

    def res(x):
        # x = [rv_mean(3), rv_stereo(3), t_common(3)]
        rv_mean = x[:3]; rv_stereo = x[3:6]; t_common = x[6:9]
        R_mean = Rotation.from_rotvec(rv_mean).as_matrix()
        R_stereo = Rotation.from_rotvec(rv_stereo).as_matrix()
        R_stereo_inv = Rotation.from_matrix(R_stereo).inv().as_matrix()
        blocks = []
        for ch_idx, (d_base, m_base, d_ref, O_ref) in enumerate([
            (d_bases[0], m_bases[0], d_refs[0], O_refs[0]),
            (d_bases[1], m_bases[1], d_refs[1], O_refs[1])]):
            R_ch = R_mean @ (R_stereo if ch_idx == 0 else R_stereo_inv)
            t_ch = t_common if ch_idx == 0 else -t_common
            d_new, m_new = apply_se3(d_base, m_base, Rotation.from_matrix(R_ch).as_rotvec(), t_ch)
            blocks.append(two_plane_residual_plucker(d_new, m_new, d_ref, O_ref, d_ref))
        return np.concatenate(blocks)

    x0 = np.zeros(9)
    lo = np.array([-0.05]*6 + [-2.0]*3); hi = np.array([0.05]*6 + [2.0]*3)
    sol = least_squares(res, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=300, xtol=1e-10)
    rv_mean = sol.x[:3]; rv_stereo = sol.x[3:6]; t_common = sol.x[6:9]
    R_mean = Rotation.from_rotvec(rv_mean).as_matrix()
    R_stereo = Rotation.from_rotvec(rv_stereo).as_matrix()
    R_stereo_inv = Rotation.from_matrix(R_stereo).inv().as_matrix()
    d_sym = []
    for ch_idx, (d_base, m_base) in enumerate([(dmL, mmL), (dmR, mmR)]):
        R_ch = R_mean @ (R_stereo if ch_idx == 0 else R_stereo_inv)
        t_ch = t_common if ch_idx == 0 else -t_common
        d_new, m_new = apply_se3(d_base, m_base, Rotation.from_matrix(R_ch).as_rotvec(), t_ch)
        d_sym.append((d_new, m_new))
    return d_sym[0][0], d_sym[0][1], d_sym[1][0], d_sym[1][1], rv_mean, rv_stereo, t_common

d_symL, m_symL, d_symR, m_symR, rv_mean, rv_stereo, t_sym = fit_se3_symmetric()
res_sym = np.concatenate([two_plane_residual_plucker(d_symL, m_symL, dzL, OzL, dzL),
                           two_plane_residual_plucker(d_symR, m_symR, dzR, OzR, dzR)])
rms_sym = float(np.sqrt(np.mean(_ray_rms(res_sym)**2)))
dot_symL = np.clip(np.sum(dzL*d_symL, axis=1), -1, 1)
dot_symR = np.clip(np.sum(dzR*d_symR, axis=1), -1, 1)
results_se3.append({
    "name": "symmetric SE(3) (3+3+3)", "rms_mm": rms_sym,
    "dir_L": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_symL))**2))),
    "dir_R": float(np.sqrt(np.mean(np.degrees(np.arccos(dot_symR))**2))),
    "rot_mean_deg": float(np.degrees(np.linalg.norm(rv_mean))),
    "rot_stereo_deg": float(np.degrees(np.linalg.norm(rv_stereo))),
    "t_common_mm": [float(t) for t in t_sym],
})
print(f"  Symmetric SE(3): RMS={rms_sym:.4f} mm  mean rot={np.degrees(np.linalg.norm(rv_mean)):.3f}°  stereo rot={np.degrees(np.linalg.norm(rv_stereo)):.3f}°  t={t_sym}")

# Summary table
baseline_rms = float(np.sqrt(np.mean(_ray_rms(np.concatenate([
    two_plane_residual_plucker(dmL, mmL, dzL, OzL, dzL),
    two_plane_residual_plucker(dmR, mmR, dzR, OzR, dzR)]))**2)))
print(f"\n{'─'*70}")
print(f"{'Model':>25s}  {'Params':>6s}  {'Ray RMS':>9s}  {'Dir L':>7s}  {'Dir R':>7s}  {'Mom L':>7s}  {'Mom R':>7s}")
print(f"  {'─'*25}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
print(f"  {'telecentric baseline':>25s}  {14:>6d}  {baseline_rms:9.4f}  {dir_rms_L:7.4f}  {dir_rms_R:7.4f}  {mom_rms_L:7.4f}  {mom_rms_R:7.4f}")
for r in results_se3:
    print(f"  {r['name']:>25s}  {'?':>6s}  {r['rms_mm']:9.4f}  {r.get('dir_L',0):7.4f}  {r.get('dir_R',0):7.4f}  {r.get('mom_L',0):7.4f}  {r.get('mom_R',0):7.4f}")
zernike_rms = float(np.sqrt(np.mean(_ray_rms(np.concatenate([
    two_plane_residual_plucker(dzL, mzL, dzL, OzL, dzL),
    two_plane_residual_plucker(dzR, mzR, dzR, OzR, dzR)]))**2)))
print(f"  {'Zernike ref (self)':>25s}  {57:>6d}  {zernike_rms:9.6f}")

# ── Pixel reprojection for SE(3) aligned rays ──
# We can't easily create a CMO model with SE(3). Instead, approximate:
# For each corner pixel, evaluate the telecentric ray, then apply SE(3).
def pixel_reproj_plucker(d_aligned, m_aligned, channel_label):
    """Reprojection using aligned Plücker rays."""
    epx = []
    O_aligned = [np.cross(d_aligned[i], m_aligned[i]) for i in range(len(d_aligned))]
    grid_shape = (41, 41)
    for pi in range(len(paired_z)):
        Rm, t = opt_R[pi], opt_t[pi]
        Xw = (Rm @ obj_pts.T).T + t[None, :]; n_plane = Rm[:, 2]
        for k in range(obj_pts.shape[0]):
            uv = left_pixels[pi][k] if channel_label == "left" else right_pixels[pi][k]
            # Find nearest grid point
            u_idx = np.argmin(np.abs(u_grid[0,:] - uv[0]))
            v_idx = np.argmin(np.abs(v_grid[:,0] - uv[1]))
            idx = v_idx * 41 + u_idx
            if idx < len(d_aligned):
                O_test = O_aligned[idx]
                d_test = d_aligned[idx]
                dn = float(np.dot(d_test, n_plane))
                if abs(dn) > 1e-10:
                    tL = float(np.dot(t - O_test, n_plane)) / dn
                    e = float(np.linalg.norm((O_test + tL * d_test) - Xw[k]))
                    epx.append(e / max(abs(tL), 1.0) * FX)
    epx_arr = np.array(epx)
    return float(np.sqrt(np.mean(epx_arr**2))), float(np.percentile(epx_arr, 50))

# Telecentric baseline pixel RMS
tel_px_rms = 14.55  # from previous runs
print(f"\nPixel RMS (from previous run): telecentric = {tel_px_rms:.2f} px")

# Save
artifact = {
    "description": "Global Plücker SE(3) arm alignment for CMO telecentric model",
    "baseline": {"rms_mm": baseline_rms, "dir_L_deg": dir_rms_L, "dir_R_deg": dir_rms_R,
                 "mom_L_mm": mom_rms_L, "mom_R_mm": mom_rms_R},
    "variants": results_se3,
}
with open(OUT / "arm_alignment_diagnostic.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {OUT / 'arm_alignment_diagnostic.json'}")
print("Done!")
