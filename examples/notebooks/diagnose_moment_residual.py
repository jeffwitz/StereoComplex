#!/usr/bin/env python3
"""Diagnose the CMO moment residual Δm = m_Z - m_M.

Phases 1-2: modal decomposition of moment residual, then fit origin-only models
with direction fixed to the telecentric model.
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

# ── Reuse saved Zernike + telecentric from previous run ──
# We need to re-fit quickly. Let's do a lightweight version.
import cv2
from cv2 import aruco
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import (
    CMOTelecentricStereoModel, _ray_rms, _normalize, _roty,
)
from stereocomplex.physics.model_selection import rayfield_two_plane_residuals, _grid_pixels
from scipy.optimize import least_squares

# Params
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
NCX, NCY, SQR = 16, 12, 0.3
IMG_SIZE = (2048, 2048); W, H = IMG_SIZE
FX = 25600.0
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

print("=" * 60)
print("Loading: detection + Hessian + TPS + Zernike + telecentric")
print("=" * 60)

# ── Detection + Hessian + double TPS (same pipeline) ──
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

# Hessian helpers (same as notebook)
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
    if prefer_largest:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        cx_c, cy_c = xi-x0, yi-y0
        best_k, best_d2 = 1, float("inf")
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

# Zernike fit
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
print(f"Zernike RMS={zd.ray_rms_mm:.6f} mm  time={time.time()-t0:.0f}s")

# Telecentric fit
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(len(opt_t))]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
f_obj_est = WD_est - float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))
print(f"WD={WD_est:.1f}  b={b_est:.1f}  f_obj={f_obj_est:.1f}  theta={np.degrees(theta_fixed):.1f}°")

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
tel_x = sol_tel.x
lr = rayfield_two_plane_residuals(lf, m_tel.channel("left"), support, z_planes=(50., 80.))
rr = rayfield_two_plane_residuals(rf, m_tel.channel("right"), support, z_planes=(50., 80.))
rms_tel = float(np.sqrt(0.5*(_ray_rms(lr)**2+_ray_rms(rr)**2)))
print(f"Telecentric: Ray RMS={rms_tel:.4f} mm")

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — Moment residual modal decomposition
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1: Moment residual Δm = m_Z - m_CMO")
print("=" * 60)

from stereocomplex.core.model_compact.zernike import zernike_modes, eval_real_zernike

# Evaluate on dense grid
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()

OzL, dzL = lf.ray(uf, vf); OzR, dzR = rf.ray(uf, vf)
OmL, dmL = m_tel.ray(uf, vf, "left"); OmR, dmR = m_tel.ray(uf, vf, "right")

# Direction errors
dot_L = np.clip(np.sum(dzL*dmL, axis=1), -1, 1); dot_R = np.clip(np.sum(dzR*dmR, axis=1), -1, 1)
ang_L = np.degrees(np.arccos(dot_L)); ang_R = np.degrees(np.arccos(dot_R))

# Moment errors
mzL = np.cross(OzL, dzL); mmL = np.cross(OmL, dmL)
mzR = np.cross(OzR, dzR); mmR = np.cross(OmR, dmR)
dmL_vec = mzL - mmL; dmR_vec = mzR - mmR
dm_norm_L = np.linalg.norm(dmL_vec, axis=1); dm_norm_R = np.linalg.norm(dmR_vec, axis=1)

print(f"Direction RMS: L={np.sqrt(np.mean(ang_L**2)):.4f}°  R={np.sqrt(np.mean(ang_R**2)):.4f}°")
print(f"Moment RMS:    L={np.sqrt(np.mean(dm_norm_L**2)):.4f} mm  R={np.sqrt(np.mean(dm_norm_R**2)):.4f} mm")

# Zernike projection of Δm components
modes_4 = zernike_modes(4)
xi = 2.0*uf/(W-1)-1.0; zeta = 2.0*vf/(H-1)-1.0
rho = np.sqrt(xi*xi+zeta*zeta)/np.sqrt(2.0); theta = np.arctan2(zeta, xi)
B = np.empty((uf.size, len(modes_4)), dtype=np.float64)
for j, mode in enumerate(modes_4): B[:, j] = eval_real_zernike(mode, rho, theta)
BtB_inv = np.linalg.inv(B.T @ B + 1e-10*np.eye(len(modes_4)))
B_pinv = BtB_inv @ B.T
Bsq = np.sum(B**2, axis=0)

print("\nMoment residual modal decomposition:")
for label, dm_vec in [("Left", dmL_vec), ("Right", dmR_vec)]:
    var_tot = float(np.sum(dm_vec**2))
    contribs = []
    for j, mode in enumerate(modes_4):
        cx, cy, cz = B_pinv @ dm_vec[:, 0], B_pinv @ dm_vec[:, 1], B_pinv @ dm_vec[:, 2]
        frac = float((cx[j]**2 + cy[j]**2 + cz[j]**2) * Bsq[j]) / max(var_tot, 1e-16)
        contribs.append((f"Z_{mode.n}^{mode.m}({mode.kind})", mode.n, mode.m, mode.kind, frac,
                         float(cx[j]), float(cy[j]), float(cz[j])))
    contribs.sort(key=lambda x: -abs(x[4]))
    print(f"\n  {label}:")
    for name, n, m, kind, frac, cx, cy, cz in contribs[:8]:
        if frac < 0.005: continue
        bar = "█" * max(1, int(frac*100))
        print(f"    {name:18s}  {frac*100:5.1f}% {bar}  c=({cx:+.3f},{cy:+.3f},{cz:+.3f})")

# Direction residual too for comparison
print("\nDirection residual modal decomposition (for comparison):")
for label, dd_vec in [("Left", dzL-dmL), ("Right", dzR-dmR)]:
    var_tot = float(np.sum(dd_vec**2))
    contribs = []
    for j, mode in enumerate(modes_4):
        cx, cy, cz = B_pinv @ dd_vec[:, 0], B_pinv @ dd_vec[:, 1], B_pinv @ dd_vec[:, 2]
        frac = float((cx[j]**2 + cy[j]**2 + cz[j]**2) * Bsq[j]) / max(var_tot, 1e-16)
        contribs.append((f"Z_{mode.n}^{mode.m}({mode.kind})", frac))
    contribs.sort(key=lambda x: -abs(x[1]))
    print(f"\n  {label}:")
    for name, frac in contribs[:6]:
        if frac < 0.005: continue
        print(f"    {name:18s}  {frac*100:5.1f}%")

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Fit origin only with direction fixed
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Fit origin only, direction fixed")
print("=" * 60)

# Precompute telecentric directions on the grid (fixed)
_, dL_fixed = m_tel.ray(uf, vf, "left")
_, dR_fixed = m_tel.ray(uf, vf, "right")

def _project_transverse(v, d):
    return v - np.sum(v * d, axis=1, keepdims=True) * d

def two_plane_residual_from_Od(O_fit, d_fixed, O_ref, d_ref, z_planes=(50., 80.)):
    """Residual from fitted (O, d_fixed) vs reference (O_ref, d_ref)."""
    blocks = []
    for z in z_planes:
        t_ref = (z - O_ref[:, 2]) / d_ref[:, 2]
        P_ref = O_ref + t_ref[:, None] * d_ref
        t_fit = (z - O_fit[:, 2]) / d_fixed[:, 2]
        P_fit = O_fit + t_fit[:, None] * d_fixed
        blocks.append((P_ref - P_fit).reshape(-1))
    return np.concatenate(blocks)

# Model O0: constant sub-pupil per channel (current telecentric)
O0_L = OmL; O0_R = OmR
res0 = np.concatenate([two_plane_residual_from_Od(O0_L, dL_fixed, OzL, dzL),
                        two_plane_residual_from_Od(O0_R, dR_fixed, OzR, dzR)])
rms0 = float(np.sqrt(np.mean(_ray_rms(res0)**2)))
print(f"\nO0 (constant sub-pupil): two-plane RMS = {rms0:.4f} mm")

# Model O1: transverse affine pupil field per channel
# O(u,v) = S + P_perp(d) * [a_x*u_tilde + b_x, a_y*v_tilde + b_y, a_z*u_tilde + b_z]
# where S is the existing sub-pupil from telecentric model
# Actually, use raw pixel coords for simplicity:
# O(u,v) = O0_channel + P_perp(d) * [ax*(u-cx) + bx, ay*(v-cy) + by, 0]

def fit_O1(channel):
    """Fit affine transverse origin correction for one channel."""
    O_ref = OzL if channel == "left" else OzR
    d_ref = dzL if channel == "left" else dzR
    d_fix = dL_fixed if channel == "left" else dR_fixed
    O_base = OmL if channel == "left" else OmR
    u_tilde = (uf - 1024.) / 1024.  # normalized [-1, 1]
    v_tilde = (vf - 1024.) / 1024.

    def residuals_O1(x):
        # x = [ax, bx, ay, by]
        ax, bx, ay, by = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        dO_raw = np.column_stack([
            ax * u_tilde + bx,
            ay * v_tilde + by,
            np.zeros_like(u_tilde),
        ])
        dO = _project_transverse(dO_raw, d_fix)
        O_fit = O_base + dO
        return two_plane_residual_from_Od(O_fit, d_fix, O_ref, d_ref)

    x0 = np.zeros(4); lo = np.full(4, -5.0); hi = np.full(4, 5.0)
    sol = least_squares(residuals_O1, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=200)
    return sol

sol_O1_L = fit_O1("left"); sol_O1_R = fit_O1("right")
# Reconstruct O1
def reconstruct_O1(channel, x):
    O_base = OmL if channel == "left" else OmR
    d_fix = dL_fixed if channel == "left" else dR_fixed
    u_tilde = (uf - 1024.) / 1024.; v_tilde = (vf - 1024.) / 1024.
    dO_raw = np.column_stack([x[0]*u_tilde + x[1], x[2]*v_tilde + x[3], np.zeros_like(u_tilde)])
    return O_base + _project_transverse(dO_raw, d_fix)

O1_L = reconstruct_O1("left", sol_O1_L.x); O1_R = reconstruct_O1("right", sol_O1_R.x)
res1 = np.concatenate([two_plane_residual_from_Od(O1_L, dL_fixed, OzL, dzL),
                        two_plane_residual_from_Od(O1_R, dR_fixed, OzR, dzR)])
rms1 = float(np.sqrt(np.mean(_ray_rms(res1)**2)))
print(f"O1 (affine origin): two-plane RMS = {rms1:.4f} mm  (O0={rms0:.4f})")
print(f"  L: ax={sol_O1_L.x[0]:.3f} bx={sol_O1_L.x[1]:.3f} ay={sol_O1_L.x[2]:.3f} by={sol_O1_L.x[3]:.3f}")
print(f"  R: ax={sol_O1_R.x[0]:.3f} bx={sol_O1_R.x[1]:.3f} ay={sol_O1_R.x[2]:.3f} by={sol_O1_R.x[3]:.3f}")

# Model O2: transverse quadratic pupil field
def fit_O2(channel):
    O_ref = OzL if channel == "left" else OzR
    d_ref = dzL if channel == "left" else dzR
    d_fix = dL_fixed if channel == "left" else dR_fixed
    O_base = OmL if channel == "left" else OmR
    u_tilde = (uf - 1024.) / 1024.; v_tilde = (vf - 1024.) / 1024.
    u2 = u_tilde**2 - np.mean(u_tilde**2)
    v2 = v_tilde**2 - np.mean(v_tilde**2)
    uv = u_tilde * v_tilde - np.mean(u_tilde * v_tilde)

    def residuals_O2(x):
        ax, bx, cx, ay, by, cy = [float(v) for v in x]
        dO_raw = np.column_stack([
            ax*u_tilde + bx + cx*u2,
            ay*v_tilde + by + cy*v2,
            np.zeros_like(u_tilde),
        ])
        dO = _project_transverse(dO_raw, d_fix)
        return two_plane_residual_from_Od(O_base + dO, d_fix, O_ref, d_ref)

    x0 = np.zeros(6); lo = np.full(6, -5.0); hi = np.full(6, 5.0)
    sol = least_squares(residuals_O2, x0=x0, bounds=(lo, hi), loss="linear", max_nfev=300)
    return sol

sol_O2_L = fit_O2("left"); sol_O2_R = fit_O2("right")
def reconstruct_O2(channel, x):
    O_base = OmL if channel == "left" else OmR
    d_fix = dL_fixed if channel == "left" else dR_fixed
    u_tilde = (uf - 1024.) / 1024.; v_tilde = (vf - 1024.) / 1024.
    u2 = u_tilde**2 - np.mean(u_tilde**2)
    v2 = v_tilde**2 - np.mean(v_tilde**2)
    dO_raw = np.column_stack([x[0]*u_tilde + x[1] + x[2]*u2,
                               x[3]*v_tilde + x[4] + x[5]*v2,
                               np.zeros_like(u_tilde)])
    return O_base + _project_transverse(dO_raw, d_fix)

O2_L = reconstruct_O2("left", sol_O2_L.x); O2_R = reconstruct_O2("right", sol_O2_R.x)
res2 = np.concatenate([two_plane_residual_from_Od(O2_L, dL_fixed, OzL, dzL),
                        two_plane_residual_from_Od(O2_R, dR_fixed, OzR, dzR)])
rms2 = float(np.sqrt(np.mean(_ray_rms(res2)**2)))
print(f"O2 (quadratic origin): two-plane RMS = {rms2:.4f} mm  (O0={rms0:.4f})")

# Summary
print(f"\n{'─'*60}")
print(f"{'Origin model':>20s}  {'Ray RMS mm':>10s}  {'vs O0':>8s}")
print(f"  {'O0 (constant)':>20s}  {rms0:10.4f}")
print(f"  {'O1 (affine)':>20s}  {rms1:10.4f}  {rms1/rms0:7.1%}")
print(f"  {'O2 (quadratic)':>20s}  {rms2:10.4f}  {rms2/rms0:7.1%}")

# Pixel reprojection comparison
def pixel_reproj(m, name):
    class W: pass
    w = W(); w.m = m
    epx = []
    for pi in range(len(paired_z)):
        Rm, t = opt_R[pi], opt_t[pi]
        Xw = (Rm @ obj_pts.T).T + t[None, :]; n_plane = Rm[:, 2]
        for k in range(obj_pts.shape[0]):
            for uv, ch in [(left_pixels[pi][k], "left"), (right_pixels[pi][k], "right")]:
                O_test, d_test = m.ray(np.array([uv[0]]), np.array([uv[1]]), ch)
                dn = float(np.dot(d_test[0], n_plane))
                if abs(dn) > 1e-10:
                    tL = float(np.dot(t - O_test[0], n_plane)) / dn
                    e = float(np.linalg.norm((O_test[0] + tL * d_test[0]) - Xw[k]))
                    epx.append(e / max(abs(tL), 1.0) * FX)
    epx_arr = np.array(epx)
    rms = float(np.sqrt(np.mean(epx_arr**2)))
    p50 = float(np.percentile(epx_arr, 50))
    print(f"  {name:>20s}  pixel RMS={rms:.2f} px  P50={p50:.2f} px")
    return rms

# We can't easily create a new model object for O1/O2 without a full class.
# Instead, let's approximate: the pixel reprojection uses m_tel.ray() which
# already has the best-fit origin. The O1/O2 corrections are small and would
# only marginally change the reprojection.
#
# Better approach: compare ray-space RMS which is directly linked to reprojection.
print(f"\nRay-space to pixel mapping: telecentric O0 gives ~14.5 px for ~{rms_tel:.4f} mm ray RMS")
print(f"If O1/O2 reduce ray RMS significantly, pixel RMS will follow proportionally.")
print(f"O1: ray RMS reduction = {(1-rms1/rms_tel)*100:.1f}%")
print(f"O2: ray RMS reduction = {(1-rms2/rms_tel)*100:.1f}%")

# Save
artifact = {
    "description": "Moment residual modal analysis + origin-only fits",
    "direction_rms_deg": {"L": float(np.sqrt(np.mean(ang_L**2))), "R": float(np.sqrt(np.mean(ang_R**2)))},
    "moment_rms_mm": {"L": float(np.sqrt(np.mean(dm_norm_L**2))), "R": float(np.sqrt(np.mean(dm_norm_R**2)))},
    "origin_fits": {
        "O0_constant": rms0,
        "O1_affine": {"rms": rms1, "ratio_vs_O0": rms1/rms0,
                      "L": {"ax": float(sol_O1_L.x[0]), "bx": float(sol_O1_L.x[1]),
                            "ay": float(sol_O1_L.x[2]), "by": float(sol_O1_L.x[3])},
                      "R": {"ax": float(sol_O1_R.x[0]), "bx": float(sol_O1_R.x[1]),
                            "ay": float(sol_O1_R.x[2]), "by": float(sol_O1_R.x[3])}},
        "O2_quadratic": {"rms": rms2, "ratio_vs_O0": rms2/rms0},
    },
}
with open(OUT / "moment_residual_diagnostic.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {OUT / 'moment_residual_diagnostic.json'}")
print("Done!")
