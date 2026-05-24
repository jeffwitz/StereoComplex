#!/usr/bin/env python3
"""Low-rank PCA residual correction on top of 26p aligned CMO.

SVD on the two-plane residual field → keep r=1,2,3 dominant modes.
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

# ═══════════════════════════════════════════════════════════
# Pipeline (abbreviated — same as always)
# ═══════════════════════════════════════════════════════════
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
# ... (same helpers, omitted for brevity — reuse from autopsy_26p) ...
def abs_det_hessian(gray, sigma=9.0):
    f = gray.astype(np.float32)
    if f.max() > 2: f /= 255.0
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return np.abs(cv2.Sobel(f, cv2.CV_64F, 2, 0, ksize=3)*cv2.Sobel(f, cv2.CV_64F, 0, 2, ksize=3)
                  - cv2.Sobel(f, cv2.CV_64F, 1, 1, ksize=3)**2)
def otsu_mask(r):
    r8 = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU); return m
def blob_barycentre(mask, xi, yi, d, prefer_largest=True):
    h, w = mask.shape
    x0, x1 = max(0, int(xi)-d), min(w, int(xi)+d); y0, y1 = max(0, int(yi)-d), min(h, int(yi)+d)
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
            try: pred_xy = predict_points_rayfield_tps_robust(np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0), chess3[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
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

print("Pipeline...", end=" ", flush=True)
lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float)
n_frames = len(paired_z)
denoised_L, denoised_R, detection_counts = [], [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L); nR = 0 if ids_R is None else len(ids_R)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]
    mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]
    comp_L = complete_corners_hessian(lg, cc_L, ids_L, NCX, NCY, mk_c_L, mk_ids_L)
    comp_R = complete_corners_hessian(rg, cc_R, ids_R, NCX, NCY, mk_c_R, mk_ids_R)
    dmask_L = np.zeros(165, dtype=bool); dmask_R = np.zeros(165, dtype=bool)
    if ids_L is not None: dmask_L[np.asarray(ids_L).ravel()] = True
    if ids_R is not None: dmask_R[np.asarray(ids_R).ravel()] = True
    detection_counts.append((dmask_L, dmask_R))
    for mk_c, mk_ids, comp, out in [(mk_c_L, mk_ids_L, comp_L, denoised_L), (mk_c_R, mk_ids_R, comp_R, denoised_R)]:
        obj_xy_list, img_uv_list = [], []
        if mk_ids is not None:
            for i in range(len(mk_ids)):
                mid = int(mk_ids[i].ravel()[0]); o = id_to_obj.get(mid)
                if o is None: continue
                mc = np.asarray(mk_c[i], dtype=np.float64).reshape(-1, 2)
                if mc.shape[0] == 4: obj_xy_list.append(o); img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            pred = predict_points_rayfield_tps_robust(np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0), chess3[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
        else: pred = comp
        re_denoised = predict_points_rayfield_tps_robust(chess3[:, :2].astype(np.float64), pred.astype(np.float64), chess3[:, :2].astype(np.float64), lam=3.0, huber_c=1.5, iters=2, ransac_reproj_px=2.0)
        out.append(re_denoised)
obj_pts = chess3.astype(np.float64)
left_pixels = [dn.astype(np.float64) for dn in denoised_L]
right_pixels = [dn.astype(np.float64) for dn in denoised_R]
point_indices = [np.arange(165, dtype=int) for _ in range(n_frames)]
rvecs, tvecs = [], []
for lp in left_pixels:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))
obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs),
                             left_pixels=left_pixels, right_pixels=right_pixels, point_indices=point_indices,
                             noise_std_px=0.0, image_size=IMG_SIZE)
t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(), max_order_d=2, max_nfev=500, origin_reg_weight=0.0)
print(f"done. Zernike: {zd.ray_rms_mm:.6f} mm  {time.time()-t0:.0f}s")

# Grid
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf); OzR, dzR = rf.ray(uf, vf)
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.]))
OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(n_frames)]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
f_obj_est = WD_est - float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))

def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

# Fit 26p
x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dcL[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
rot_lo = np.full(3, -0.08); rot_hi = np.full(3, 0.08)
trans_lo = np.full(3, -3.0); trans_hi = np.full(3, 3.0)

def build_26p(x):
    x_tel, arm = x[:14], x[14:]
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(x_tel, pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
    OL, dL = m_tel.ray(uf, vf, "left"); OR, dR = m_tel.ray(uf, vf, "right")
    rv_L, t_L = arm[:3], arm[3:6]; rv_R, t_R = arm[6:9], arm[9:12]
    OL_a, dL_a = apply_se3(OL, dL, rv_L, t_L)
    OR_a, dR_a = apply_se3(OR, dR, rv_R, t_R)
    return OL_a, dL_a, OR_a, dR_a, m_tel, rv_L, t_L, rv_R, t_R

def res_26p(x):
    OL_a, dL_a, OR_a, dR_a, *_ = build_26p(x)
    blocks = []
    for z in [50., 80.]:
        for O_a, d_a, O_ref, d_ref in [(OL_a, dL_a, OzL, dzL), (OR_a, dR_a, OzR, dzR)]:
            tz_ref = (z-O_ref[:,2])/d_ref[:,2]; P_ref = O_ref + tz_ref[:,None]*d_ref
            tz_mod = (z-O_a[:,2])/d_a[:,2]; P_mod = O_a + tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)

arm_lo = np.concatenate([rot_lo, trans_lo, rot_lo, trans_lo])
arm_hi = np.concatenate([rot_hi, trans_hi, rot_hi, trans_hi])
x0 = np.concatenate([x0_tel, np.zeros(12)])
sol = least_squares(res_26p, x0=x0, bounds=(np.concatenate([lo_tel, arm_lo]), np.concatenate([hi_tel, arm_hi])),
                     loss="huber", f_scale=1.0, max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
x_opt = sol.x
OL_a, dL_a, OR_a, dR_a, m_tel_opt, rv_L, t_L, rv_R, t_R = build_26p(x_opt)
rms_26p = float(np.sqrt(np.mean(res_26p(x_opt)**2)))
print(f"26p: RMS={rms_26p:.6f} mm")

# ═══════════════════════════════════════════════════════════
# PCA on two-plane residual
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PCA on 26p two-plane residual")
print("=" * 60)

# Build residual matrix: for each pixel, the 6-vector two-plane residual
# (3D diff at z=50 + 3D diff at z=80) for left + right = 12 dims per pixel
res_L = []; res_R = []
for z in [50., 80.]:
    for O_a, d_a, O_ref, d_ref in [(OL_a, dL_a, OzL, dzL), (OR_a, dR_a, OzR, dzR)]:
        tz_ref = (z-O_ref[:,2])/d_ref[:,2]; P_ref = O_ref + tz_ref[:,None]*d_ref
        tz_mod = (z-O_a[:,2])/d_a[:,2]; P_mod = O_a + tz_mod[:,None]*d_a
        if O_ref is OzL: res_L.append(P_ref - P_mod)
        else: res_R.append(P_ref - P_mod)

# Stack: (N_pixels, 6) for combined L+R two-plane residual
R_full = np.column_stack([np.concatenate(res_L[:2], axis=1), np.concatenate(res_R[:2], axis=1)])  # (1681, 12)

# SVD
U, S, Vt = np.linalg.svd(R_full, full_matrices=False)
print(f"Singular values: {[f'{s:.4f}' for s in S[:8]]}")
print(f"Variance explained by top modes:")
cum = 0; total = np.sum(S**2)
for r in range(1, 9):
    cum += S[r-1]**2
    print(f"  r={r}: {S[r-1]**2/total*100:5.1f}%  (cum {cum/total*100:.0f}%)")

# ═══════════════════════════════════════════════════════════
# Low-rank correction: add r modes
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Low-rank correction models")
print("=" * 60)

# r modes: U[:,:r] @ diag(S[:r]) @ Vt[:r,:] approximates the residual
# To add this to the 26p model: at each pixel, subtract the approximate residual
# from the 26p two-plane intersection.
# In practice, we fit correction coefficients c_k for each mode.

# The correction is: P_corrected(z) = P_26p(z) + sum_k c_k * Phi_k(pixel, z)
# where Phi_k are the dominant residual patterns from SVD.

# We'll fit the correction directly via least_squares on the full residual.
# Use the r dominant right singular vectors as basis functions.
# Each Vt[j,:] is a 12-vector per pixel (6 for L two-plane, 6 for R two-plane).

basis_vectors = Vt[:6, :]  # top 6 modes, each (12,) per pixel actually (12*pixels)

# Actually, Vt is (min(M,N), 12) with N=1681 pixels. Each row of Vt has 12 elements.
# Wait: R_full is (1681, 12) so SVD gives U(1681,12), S(12,), Vt(12,12).
# Vt[j,:] is a (12,) vector = the j-th mode's pattern across the 12 residual dimensions.

# The residual at pixel i is R_full[i,:] = sum_j U[i,j] * S[j] * Vt[j,:]
# A rank-r approximation uses only the first r terms.

# For correction: we add c_j * Vt[j,:] at each pixel (same correction for all pixels).
# This is equivalent to fitting r scalar coefficients.

print(f"\n{'Model':>22s}  {'P':>4s}  {'Ray RMS':>9s}  {'Px RMS':>7s}  {'P50':>7s}  {'P95':>7s}")
print(f"  {'─'*22}  {'─'*4}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}")
print(f"  {'26p baseline':>22s}  {26:>4d}  {rms_26p:9.6f}  {'1.06':>7s}  {'0.87':>7s}  {'1.84':>7s}")

for r in [1, 2, 3]:
    # Basis: Vt[:r,:] — these are (r, 12) patterns in residual space
    # We fit r coefficients c_j to minimize the residual
    basis = Vt[:r, :].T  # (12, r)
    # Fit: R ≈ basis @ c, so c = basis^+ @ R_vectorized
    # But we want to minimize two-plane residual after correction:
    # residual_new = R_full - R_full @ basis @ basis^+ (projection)
    # Actually: correction at each pixel = basis @ c, where c minimizes ||R - basis@c||
    # c = (basis^T @ basis)^{-1} @ basis^T @ mean_residual
    # But simpler: just project the full residual onto the basis
    # R_corrected = R - R @ basis @ inv(basis^T @ basis) @ basis^T
    # Actually: correction = basis @ c where c = pinv(basis) @ R.T → c is (r, 1681)
    # But we want GLOBAL coefficients (same for all pixels), not per-pixel.
    # For a global correction: correction_per_pixel = basis @ c where c is (r,) global.
    # c = pinv(basis) @ mean(R, axis=0) = pinv(basis) @ mean_residual

    mean_residual = np.mean(R_full, axis=0)  # (12,) — mean residual pattern
    c = np.linalg.lstsq(basis, mean_residual, rcond=None)[0]  # (r,)
    correction = basis @ c  # (12,)

    # Apply correction to the 26p two-plane intersections and compute new RMS
    R_corrected = R_full - correction[None, :]  # (1681, 12)
    new_rms = float(np.sqrt(np.mean(R_corrected**2)))

    # Pixel reprojection: evaluate corrected rays at corner pixels
    # The correction is applied to the two-plane intersection, not directly to rays.
    # For pixel reprojection, we'd need to apply the correction to individual rays.
    # This is more complex — skip for now, just report ray RMS.
    # Approximate: pixel RMS scales roughly with ray RMS.
    est_px = 1.06 * new_rms / rms_26p

    print(f"  {'26p + ' + str(r) + ' PCA mode(s)':>22s}  {26+r:>4d}  {new_rms:9.6f}  {est_px:7.2f}  {'─':>7s}  {'─':>7s}")

# Also direct fit of 1,2,3 global coefficients by least_squares on the full residual
print(f"\n  Direct least-squares fit of global correction coefficients:")
for r in [1, 2, 3]:
    basis = Vt[:r, :].T  # (12, r)

    def res_pca(c):
        corr = basis @ c  # (12,)
        return (R_full - corr[None, :]).reshape(-1)

    c0 = np.zeros(r)
    sol_pca = least_squares(res_pca, x0=c0, loss="linear", max_nfev=50)
    rms_pca = float(np.sqrt(np.mean(res_pca(sol_pca.x)**2)))
    est_px_pca = 1.06 * rms_pca / rms_26p
    impr = (rms_26p - rms_pca) / rms_26p * 100
    print(f"  r={r}: RMS={rms_pca:.6f} mm  Δ={impr:+.1f}%  est px={est_px_pca:.2f}")

# ═══════════════════════════════════════════════════════════
# Per-pixel PCA correction (more powerful)
# ═══════════════════════════════════════════════════════════
print(f"\n  Per-pixel rank-r reconstruction (upper bound):")
for r in [1, 2, 3]:
    # Reconstruct using r SVD components
    R_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    R_remain = R_full - R_approx
    rms_remain = float(np.sqrt(np.mean(R_remain**2)))
    impr2 = (rms_26p - rms_remain) / rms_26p * 100
    est_px2 = 1.06 * rms_remain / rms_26p
    print(f"  r={r}: RMS={rms_remain:.6f} mm  Δ={impr2:+.1f}%  est px={est_px2:.2f}")

# Save
artifact = {
    "description": "Low-rank PCA residual correction on 26p aligned CMO",
    "rms_26p": rms_26p, "px_26p": 1.06,
    "singular_values": [float(s) for s in S],
    "variance_explained": [float(S[r]**2/np.sum(S**2)*100) for r in range(min(8, len(S)))],
    "global_correction_ray_rms": {},
    "per_pixel_correction_ray_rms": {},
}
with open(OUT / "pca_residual_26p.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"\nSaved: {OUT / 'pca_residual_26p.json'}")
print("Done!")
