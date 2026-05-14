#!/usr/bin/env python3
"""Run pipeline once, save all intermediate state to .npz for fast reuse."""
import json, sys, time, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"

import cv2; from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

# ── Params ──
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
NCX, NCY, SQR = 16, 12, 0.3; IMG_SIZE = (2048, 2048); W, H = IMG_SIZE
FX = 25600.0; K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

# ── Detection + Hessian + TPS ── (abbreviated helpers — same as always)
dictionary = aruco.getPredefinedDictionary(getattr(aruco, "DICT_6X6_250"))
ocv_board = aruco.CharucoBoard((NCX, NCY), SQR, SQR / 2, dictionary); ocv_board.setLegacyPattern(True)
chess3 = ocv_board.getChessboardCorners()
board_ids = ocv_board.getIds().ravel(); board_obj = ocv_board.getObjPoints()
id_to_obj = {int(board_ids[i]): np.asarray(board_obj[i], dtype=np.float64)[:, :2] for i in range(len(board_ids))}
params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax = 3, 75
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate, params.maxMarkerPerimeterRate = 0.005, 0.20
params.polygonalApproxAccuracyRate, params.minCornerDistanceRate = 0.08, 0.02
params.minDistanceToBorder = 1
aruco_det = aruco.ArucoDetector(dictionary, params); charuco_det = aruco.CharucoDetector(ocv_board)

def abs_det_hessian(gray, sigma=9.0):
    f = gray.astype(np.float32)
    if f.max() > 2: f /= 255.0
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return np.abs(cv2.Sobel(f, cv2.CV_64F, 2, 0, ksize=3)*cv2.Sobel(f, cv2.CV_64F, 0, 2, ksize=3) - cv2.Sobel(f, cv2.CV_64F, 1, 1, ksize=3)**2)
def otsu_mask(r):
    r8 = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU); return m
def blob_barycentre(mask, xi, yi, d, prefer_largest=True):
    h, w = mask.shape; x0, x1 = max(0, int(xi)-d), min(w, int(xi)+d); y0, y1 = max(0, int(yi)-d), min(h, int(yi)+d)
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
        for i in range(len(np.asarray(ids).ravel())): detected[int(np.asarray(ids).ravel()[i])] = np.asarray(cc).reshape(-1, 2)[i].astype(np.float64)
    cids = sorted(detected.keys()); pred_xy = None
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
        A = fit_affine(np.array([detected[i] for i in cids]), np.array(cids), ncx); pred_xy = project_affine(A, np.arange(n_corners), ncx)
    l_step = 50.0
    if len(cids) >= 2:
        dp = float(np.linalg.norm(detected[cids[-1]] - detected[cids[0]]))
        g0, g1 = ids_to_grid(np.array([cids[0]]), ncx)[0], ids_to_grid(np.array([cids[-1]]), ncx)[0]; dg = float(np.linalg.norm(g1 - g0))
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

print("Pipeline...", flush=True)
lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float); n_frames = len(paired_z)
denoised_L, denoised_R = [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0); rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg); cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L); nR = 0 if ids_R is None else len(ids_R)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]; mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]
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
        if len(obj_xy_list) >= 4: pred = predict_points_rayfield_tps_robust(np.concatenate(obj_xy_list, axis=0), np.concatenate(img_uv_list, axis=0), chess3[:, :2].astype(np.float64), lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0)
        else: pred = comp
        re_denoised = predict_points_rayfield_tps_robust(chess3[:, :2].astype(np.float64), pred.astype(np.float64), chess3[:, :2].astype(np.float64), lam=3.0, huber_c=1.5, iters=2, ransac_reproj_px=2.0)
        out.append(re_denoised)
    print(f"  {z_str}: L {nL}→165  R {nR}→165", flush=True)

obj_pts = chess3.astype(np.float64)
left_pixels = [dn.astype(np.float64) for dn in denoised_L]; right_pixels = [dn.astype(np.float64) for dn in denoised_R]
point_indices = [np.arange(165, dtype=int) for _ in range(n_frames)]
rvecs, tvecs = [], []
for lp in left_pixels:
    s, rv, tv = cv2.solvePnP(obj_pts.astype(np.float32), lp.astype(np.float32), K.astype(np.float32), np.zeros(5, dtype=np.float32))
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3)); tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0.,0.,65.]))
obs = CharucoObservationSet(object_points_mm=obj_pts, pose_rvecs=np.array(rvecs), pose_tvecs=np.array(tvecs), left_pixels=left_pixels, right_pixels=right_pixels, point_indices=point_indices, noise_std_px=0.0, image_size=IMG_SIZE)

t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(), max_order_d=2, max_nfev=500, origin_reg_weight=0.0)
print(f"Zernike: {zd.ray_rms_mm:.6f} mm  {time.time()-t0:.0f}s", flush=True)

# 26p rayfield fit
print("26p rayfield fit...", flush=True)
u_grid, v_grid = np.meshgrid(np.linspace(0, W-1, 41), np.linspace(0, H-1, 41))
uf, vf = u_grid.ravel(), v_grid.ravel()
OzL, dzL = lf.ray(uf, vf); OzR, dzR = rf.ray(uf, vf)
OcL, dcL = lf.ray(np.array([1024.]), np.array([1024.])); OcR, dcR = rf.ray(np.array([1024.]), np.array([1024.]))
WD_est = float(np.mean([float(opt_t[i][2]) for i in range(n_frames)]))
b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
f_obj_est = WD_est - float((abs(OcL[0,2]) + abs(OcR[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))

def apply_se3(O, d, rv, t):
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

x0_tel = np.array([f_obj_est, WD_est, b_est, 1024., 1024., f_obj_est, theta_fixed, dcL[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo_tel = np.array([1.,1.,0.,0.,0.,20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200.,2048.,2048.,200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
rot_lo = np.full(3, -0.08); rot_hi = np.full(3, 0.08); trans_lo = np.full(3, -3.0); trans_hi = np.full(3, 3.0)

def res_rf(x):
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(x[:14], pixel_pitch_mm=0.0055, image_size=IMG_SIZE)
    rv_L, t_L = x[14:17], x[17:20]; rv_R, t_R = x[20:23], x[23:26]
    OL, dL = m_tel.ray(uf, vf, "left"); OR, dR = m_tel.ray(uf, vf, "right")
    OL_a, dL_a = apply_se3(OL, dL, rv_L, t_L); OR_a, dR_a = apply_se3(OR, dR, rv_R, t_R)
    blocks = []
    for z in [50., 80.]:
        for O_a, d_a, O_ref, d_ref in [(OL_a, dL_a, OzL, dzL), (OR_a, dR_a, OzR, dzR)]:
            tz_ref = (z-O_ref[:,2])/d_ref[:,2]; P_ref = O_ref + tz_ref[:,None]*d_ref
            tz_mod = (z-O_a[:,2])/d_a[:,2]; P_mod = O_a + tz_mod[:,None]*d_a
            blocks.append((P_ref-P_mod).reshape(-1))
    return np.concatenate(blocks)

arm_lo = np.concatenate([rot_lo, trans_lo, rot_lo, trans_lo]); arm_hi = np.concatenate([rot_hi, trans_hi, rot_hi, trans_hi])
sol_rf = least_squares(res_rf, x0=np.concatenate([x0_tel, np.zeros(12)]),
                        bounds=(np.concatenate([lo_tel, arm_lo]), np.concatenate([hi_tel, arm_hi])),
                        loss="huber", f_scale=1.0, max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
x_rf = sol_rf.x
print(f"26p RMS = {float(np.sqrt(np.mean(res_rf(x_rf)**2))):.6f} mm", flush=True)

# ── Save ──
opt_R_arr = np.array([R for R in opt_R])  # (n_frames, 3, 3)
opt_t_arr = np.array(opt_t)  # (n_frames, 3)
# Stack corner arrays
left_px = np.stack([lp for lp in left_pixels], axis=0)  # (n_frames, 165, 2)
right_px = np.stack([rp for rp in right_pixels], axis=0)

fname = OUT / "intermediate_state.npz"
np.savez_compressed(fname,
    left_pixels=left_px, right_pixels=right_px,
    obj_pts=obj_pts,  # (165, 3)
    opt_R=opt_R_arr, opt_t=opt_t_arr,
    x_26p=x_rf,
    n_frames=n_frames, image_size=np.array(IMG_SIZE),
    FX=FX,
)
print(f"\nSaved: {fname}")
print(f"  left_pixels:  {left_px.shape}")
print(f"  right_pixels: {right_px.shape}")
print(f"  obj_pts:      {obj_pts.shape}")
print(f"  opt_R:        {opt_R_arr.shape}")
print(f"  opt_t:        {opt_t_arr.shape}")
print(f"  x_26p:        {x_rf.shape} ({x_rf.size} params)")
print("Done!")
