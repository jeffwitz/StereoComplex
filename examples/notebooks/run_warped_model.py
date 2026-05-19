#!/usr/bin/env python3
"""Run section 9.6: warped CMO model fitting.

Reuses the detection + Hessian + TPS + Zernike pipeline from run_sweep_complete.py.
Fits telecentric L0, then warped L1/L2, reports pixel RMS.
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

# ═══════════════════════════════════════════════════════════════
# 1 — Detection + Hessian + TPS (same as run_sweep_complete.py)
# ═══════════════════════════════════════════════════════════════
import cv2
from cv2 import aruco
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield

PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
NCX, NCY, SQR = 16, 12, 0.3
IMG_SIZE = (2048, 2048); W, H = IMG_SIZE
DICT_NAME = "DICT_6X6_250"
FX = 25600.0
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

print("=" * 60)
print("1 — ChArUco + Hessian + double TPS")
print("=" * 60)

dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
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

# Hessian helpers (copied from notebook)
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
    x0, x1 = max(0, int(xi)-d), min(w, int(xi)+d)
    y0, y1 = max(0, int(yi)-d), min(h, int(yi)+d)
    if x1 <= x0+2 or y1 <= y0+2: return math.nan, math.nan, math.nan
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if nl <= 1: return math.nan, math.nan, math.nan
    if prefer_largest:
        areas = stats[1:, cv2.CC_STAT_AREA]; k = 1 + int(np.argmax(areas))
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
    ids_arr = np.asarray(ids, dtype=np.float32).reshape(-1)
    nx = ncx - 1
    return np.column_stack([ids_arr % nx, ids_arr // nx]).astype(np.float32)

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
                              id_to_obj_=None, chess3_obj=None):
    nx, ny = ncx-1, ncy-1; n_corners = nx * ny
    R = abs_det_hessian(gray); mask = otsu_mask(R)
    detected = {}
    if charuco_ids is not None and len(charuco_ids) > 0:
        ids_arr = np.asarray(charuco_ids).ravel()
        corners_arr = np.asarray(charuco_corners).reshape(-1, 2)
        for i in range(len(ids_arr)):
            detected[int(ids_arr[i])] = corners_arr[i].astype(np.float64)
    cids = sorted(detected.keys())
    pred_xy = None
    if marker_corners is not None and marker_ids is not None and id_to_obj_ is not None and chess3_obj is not None:
        obj_xy_list, img_uv_list = [], []
        for i in range(len(marker_ids)):
            mid = int(marker_ids[i].ravel()[0])
            o = id_to_obj_.get(mid)
            if o is None: continue
            mc = np.asarray(marker_corners[i], dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] == 4: obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            try:
                pred_xy = predict_points_rayfield_tps_robust(
                    np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0),
                    chess3_obj[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
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
print(f"{len(paired_z)} stereo pairs")

denoised_L, denoised_R = [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]
    mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]
    comp_L = complete_corners_hessian(lg, cc_L, ids_L, NCX, NCY, marker_corners=mk_c_L, marker_ids=mk_ids_L, id_to_obj_=id_to_obj, chess3_obj=chess3)
    comp_R = complete_corners_hessian(rg, cc_R, ids_R, NCX, NCY, marker_corners=mk_c_R, marker_ids=mk_ids_R, id_to_obj_=id_to_obj, chess3_obj=chess3)
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

# ═══════════════════════════════════════════════════════════════
# 2 — Constrained Zernike O(0)+d(2)
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
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))

obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
                             left_pixels=left_pixels, right_pixels=right_pixels, point_indices=point_indices,
                             noise_std_px=0.0, image_size=IMG_SIZE)

t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(),
                                                              max_order_d=2, max_nfev=500, origin_reg_weight=0.0)
print(f"  Zernike RMS={zd.ray_rms_mm:.6f} mm  nfev={zd.nfev}  time={time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# 3 — Telecentric model fit (section 9)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3 — Telecentric CMO + pupil shear (14 params)")
print("=" * 60)

from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _ray_rms
from stereocomplex.physics.model_selection import rayfield_two_plane_residuals, _grid_pixels
from scipy.optimize import least_squares

support = _grid_pixels(IMG_SIZE, (12, 9))

# Read CMO descriptors from Zernike
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
Ol_c, dl_c = OcL, dcL; Or_c, dr_c = OcR, dcR
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(len(opt_t))]))
b_est = float(np.linalg.norm(Or_c[0] - Ol_c[0]))
f_obj_est = WD_est - float((abs(Ol_c[0,2]) + abs(Or_c[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))
print(f"  WD={WD_est:.1f}mm  b={b_est:.1f}mm  f_obj={f_obj_est:.1f}mm  theta={np.degrees(theta_fixed):.1f}°")

# Build parameter vector directly (14-param: per-channel slopes + shared shear)
# Layout: f_obj, WD, b, cx, cy, f_ang, theta, d_y, sx_L, sy_L, sx_R, sy_R, rho_x, rho_y
x0_14 = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dl_c[0,1],
                  0., 0., 0., 0., 0., 0.], dtype=np.float64)
lo14 = np.array([1., 1., 0., 0., 0., 20., 0., -0.3, -10., -10., -10., -10., -10., -10.], dtype=np.float64)
hi14 = np.array([500., 1000., 200., 2048., 2048., 200., 0.5, 0.3, 10., 10., 10., 10., 10., 10.], dtype=np.float64)

def _build_ps(x):
    return CMOTelecentricStereoModel.from_parameter_vector(x, pixel_pitch_mm=0.0055, image_size=IMG_SIZE)

def _res_ps(x):
    m = _build_ps(x); l = m.channel("left"); r = m.channel("right")
    return np.concatenate([rayfield_two_plane_residuals(lf, l, support, z_planes=(50., 80.)),
                           rayfield_two_plane_residuals(rf, r, support, z_planes=(50., 80.))])

sol_ps = least_squares(_res_ps, x0=x0_14, bounds=(lo14, hi14), loss="linear", max_nfev=500,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
m_ps = _build_ps(sol_ps.x)
lr = rayfield_two_plane_residuals(lf, m_ps.channel("left"), support, z_planes=(50., 80.))
rr = rayfield_two_plane_residuals(rf, m_ps.channel("right"), support, z_planes=(50., 80.))
rms_ps = float(np.sqrt(0.5 * (_ray_rms(lr)**2 + _ray_rms(rr)**2)))
fp = m_ps.parameter_dict()["free"]
print(f"  Ray RMS={rms_ps:.4f} mm")
print(f"  theta={fp['theta_convergence_half_deg']:.1f}°  d_y={fp['d_y_common']:.4f}  "
      f"sx=({fp['s_x_L']:.3f},{fp['s_x_R']:.3f})  sy=({fp['s_y_L']:.3f},{fp['s_y_R']:.3f})  "
      f"rho=({fp.get('rho_x',fp.get('rho_x_L','?')):.3f},{fp.get('rho_y',fp.get('rho_y_L','?')):.3f})")

# Pixel reprojection for telecentric
class _W:
    def __init__(s, m, c): s.m = m; s.c = c
    def ray(s, u, v): return s.m.channel(s.c).ray(u, v)

epx_ps = []
for pi in range(len(paired_z)):
    Rm, t = opt_R[pi], opt_t[pi]
    Xw = (Rm @ obj_pts.T).T + t[None, :]; n_plane = Rm[:, 2]
    for k in range(obj_pts.shape[0]):
        for uv, f in [(left_pixels[pi][k], _W(m_ps, "left")), (right_pixels[pi][k], _W(m_ps, "right"))]:
            O, d = f.ray(np.array([uv[0]]), np.array([uv[1]]))
            dn = float(np.dot(d[0], n_plane))
            if abs(dn) > 1e-10:
                tL = float(np.dot(t - O[0], n_plane)) / dn
                e = float(np.linalg.norm((O[0] + tL * d[0]) - Xw[k]))
                epx_ps.append(e / max(abs(tL), 1.0) * FX)
epx_ps_arr = np.array(epx_ps)
print(f"  Pixel RMS={np.sqrt(np.mean(epx_ps_arr**2)):.2f} px  P50={np.percentile(epx_ps_arr, 50):.2f} px")

# ═══════════════════════════════════════════════════════════════
# 4 — Warped CMO models (section 9.6)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4 — Warped CMO models L1 (affine) and L2 (quadratic)")
print("=" * 60)

from stereocomplex.physics.cmo_physical import (
    CMOWarpedStereoModel, fit_cmo_warped_model_to_rayfields,
    compute_cmo_zernike_residuals, _n_warp_coeff_per_axis,
)

tele_x = m_ps.parameter_vector()
print(f"Telecentric base vector: {tele_x.size} elements")

warp_results = []
for level in [1, 2]:
    per_axis = _n_warp_coeff_per_axis(level)
    # Identity warp init
    xi_init = [0.0, 1.0, 0.0] + [0.0] * max(0, per_axis - 3)
    eta_init = [0.0, 0.0, 1.0] + [0.0] * max(0, per_axis - 3)
    x0_warp = np.concatenate([tele_x, xi_init, eta_init])
    print(f"\nLevel {level}: {len(x0_warp)} params, fitting...")

    t0 = time.time()
    result = fit_cmo_warped_model_to_rayfields(
        lf, rf, IMG_SIZE, x0_warp,
        pixel_pitch_mm=0.0055, z_planes=(50., 80.), grid_shape=(12, 9),
        warp_level=level, shared_warp=True, max_nfev=500,
    )
    elapsed = time.time() - t0
    m_fitted = result.model

    # Pixel reprojection
    epx = []
    for pi in range(len(paired_z)):
        Rm, t = opt_R[pi], opt_t[pi]
        Xw = (Rm @ obj_pts.T).T + t[None, :]; n_plane = Rm[:, 2]
        for k in range(obj_pts.shape[0]):
            for uv, f in [(left_pixels[pi][k], _W(m_fitted, "left")),
                          (right_pixels[pi][k], _W(m_fitted, "right"))]:
                O, d = f.ray(np.array([uv[0]]), np.array([uv[1]]))
                dn = float(np.dot(d[0], n_plane))
                if abs(dn) > 1e-10:
                    tL = float(np.dot(t - O[0], n_plane)) / dn
                    e = float(np.linalg.norm((O[0] + tL * d[0]) - Xw[k]))
                    epx.append(e / max(abs(tL), 1.0) * FX)
    epx_arr = np.array(epx)
    px_rms = float(np.sqrt(np.mean(epx_arr**2)))
    px_p50 = float(np.percentile(epx_arr, 50))
    px_p95 = float(np.percentile(epx_arr, 95))

    warp_results.append({
        "level": level, "n_params": result.n_parameters,
        "ray_rms_mm": result.rms_mm, "px_rms": px_rms, "px_p50": px_p50, "px_p95": px_p95,
        "model": m_fitted, "time_s": elapsed,
    })
    print(f"  Ray RMS={result.rms_mm:.4f} mm, pixel RMS={px_rms:.2f} px, P50={px_p50:.2f} px, P95={px_p95:.2f} px, time={elapsed:.0f}s")

    # Warp coefficients
    d = m_fitted.parameter_dict()
    for ch in ["L", "R"]:
        coeffs = [f"{d.get(f'warp_xi_{ch}_u{pu}v{pv}', 0):+.4f}" for pu, pv in [(0,0),(1,0),(0,1)] + ([(2,0),(1,1),(0,2)] if level>=2 else [])]
        print(f"    warp_xi_{ch}: {coeffs}")
        coeffs = [f"{d.get(f'warp_eta_{ch}_u{pu}v{pv}', 0):+.4f}" for pu, pv in [(0,0),(1,0),(0,1)] + ([(2,0),(1,1),(0,2)] if level>=2 else [])]
        print(f"    warp_eta_{ch}: {coeffs}")

# Residual analysis on best model
print(f"\n{'─'*60}")
print("Residual analysis on best warped model vs Zernike")
best_model = warp_results[-1]["model"]
res = compute_cmo_zernike_residuals(best_model, lf, rf, grid_shape=(17, 13), image_size=IMG_SIZE, zernike_order=4)
print(f"  Direction RMS: L={res['dir_rms_deg_L']:.4f}°  R={res['dir_rms_deg_R']:.4f}°  total={res['dir_rms_deg_total']:.4f}°")
print(f"  Moment RMS:    L={res['mom_rms_mm_L']:.4f} mm  R={res['mom_rms_mm_R']:.4f} mm  total={res['mom_rms_mm_total']:.4f} mm")
print(f"  Top residual Zernike modes:")
for m in res["top_direction_modes"][:8]:
    if abs(m["frac_var_L"]) + abs(m["frac_var_R"]) < 0.005: continue
    print(f"    {m['mode']:18s}  L={m['frac_var_L']*100:5.1f}%  R={m['frac_var_R']*100:5.1f}%")

# Comparison table
print(f"\n{'─'*60}")
print(f"{'Model':>25s}  {'Params':>6s}  {'Ray RMS':>8s}  {'Px RMS':>7s}  {'Px P50':>7s}  {'Px P95':>7s}")
print(f"  {'─'*25}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
print(f"  {'telecentric (L0) + shear':>25s}  {12:>6d}  {rms_ps:8.4f}  {np.sqrt(np.mean(epx_ps_arr**2)):7.2f}  {np.percentile(epx_ps_arr, 50):7.2f}  {np.percentile(epx_ps_arr, 95):7.2f}")
for wr in warp_results:
    label = f"warped L{wr['level']}"
    print(f"  {label:>25s}  {wr['n_params']:>6d}  {wr['ray_rms_mm']:8.4f}  {wr['px_rms']:7.2f}  {wr['px_p50']:7.2f}  {wr['px_p95']:7.2f}")
print(f"  {'Zernike O(0)+d(2)':>25s}  {57:>6d}  {0.0007:8.4f}  {0.47:7.2f}  {'─':>7s}  {'─':>7s}")

# Save results
saved = {
    "telecentric_L0": {"n_params": 12, "ray_rms_mm": rms_ps, "px_rms": float(np.sqrt(np.mean(epx_ps_arr**2))),
                       "px_p50": float(np.percentile(epx_ps_arr, 50)), "px_p95": float(np.percentile(epx_ps_arr, 95))},
    "warped": [{"level": wr["level"], "n_params": wr["n_params"], "ray_rms_mm": wr["ray_rms_mm"],
                "px_rms": wr["px_rms"], "px_p50": wr["px_p50"], "px_p95": wr["px_p95"]} for wr in warp_results],
    "residual_analysis": {k: res[k] for k in ["dir_rms_deg_L", "dir_rms_deg_R", "dir_rms_deg_total",
                                                "mom_rms_mm_L", "mom_rms_mm_R", "mom_rms_mm_total",
                                                "top_direction_modes"]},
}
with open(OUT / "warped_model_comparison.json", "w") as f:
    json.dump(saved, f, indent=2)
print(f"\nSaved: {OUT / 'warped_model_comparison.json'}")
print("Done!")
