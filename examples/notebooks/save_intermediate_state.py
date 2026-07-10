#!/usr/bin/env python3
"""Run pipeline once, save all intermediate state to .npz for fast reuse."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
OUT = ROOT / "docs" / "assets" / "pycaso_real_data"

import cv2; from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
from stereocomplex.benchmarks.rayfield_from_observations import fit_constrained_zernike_rayfield
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize
from cmo_corner_preprocessing import (
    complete_corners_hessian,
    marker_tps_corners,
    second_tps_pass,
)

# ── Params ──
PYCASO = ROOT / "examples" / "pycaso_data"
LEFT_DIR = PYCASO / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO / "Exemple" / "Images_example" / "right_calibration11"
# The paper calibration stack deliberately excludes the auxiliary 3.014-mm
# view. Keep this explicit so raw-image rebuilds reproduce the versioned 10-pair
# state rather than silently using every filename present in Pycaso.
CALIBRATION_Z_MM = (2.65, 2.72, 2.79, 2.86, 2.93, 3.00, 3.07, 3.21, 3.28, 3.35)
NCX, NCY, SQR = 16, 12, 0.3; IMG_SIZE = (2048, 2048); W, H = IMG_SIZE
FX = 25600.0; K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)

# ── Marker-TPS preprocessing with Hessian fallback ──
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

print("Pipeline...", flush=True)
left_stems = {float(f.stem): f.stem for f in LEFT_DIR.glob("*.png")}
right_stems = {float(f.stem): f.stem for f in RIGHT_DIR.glob("*.png")}
missing = [z for z in CALIBRATION_Z_MM if z not in left_stems or z not in right_stems]
if missing:
    raise FileNotFoundError(f"Missing canonical calibration pairs at z={missing}")
paired_z = [(z, left_stems[z], right_stems[z]) for z in CALIBRATION_Z_MM]
n_frames = len(paired_z)
denoised_L, denoised_R = [], []
for z_mm, left_stem, right_stem in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{left_stem}.png"), 0); rg = cv2.imread(str(RIGHT_DIR / f"{right_stem}.png"), 0)
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg); cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L); nR = 0 if ids_R is None else len(ids_R)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]; mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]
    comp_L = complete_corners_hessian(
        lg, cc_L, ids_L, NCX, NCY, marker_object_xy=id_to_obj,
        chessboard_xy=chess3[:, :2], marker_corners=mk_c_L, marker_ids=mk_ids_L,
    )
    comp_R = complete_corners_hessian(
        rg, cc_R, ids_R, NCX, NCY, marker_object_xy=id_to_obj,
        chessboard_xy=chess3[:, :2], marker_corners=mk_c_R, marker_ids=mk_ids_R,
    )
    for mk_c, mk_ids, comp, out in [(mk_c_L, mk_ids_L, comp_L, denoised_L), (mk_c_R, mk_ids_R, comp_R, denoised_R)]:
        pred = marker_tps_corners(mk_c, mk_ids, id_to_obj, chess3[:, :2])
        if pred is None:
            pred = comp
        re_denoised = second_tps_pass(chess3[:, :2], pred)
        out.append(re_denoised)
    print(f"  {z_mm:g}: L {nL}→165  R {nR}→165", flush=True)

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

# Principal point left FREE across the full image. cx, cy are non-identifiable
# weak modes (Schur score < 0.1, "driven by gauge") and the rayfield-identified
# optimum places them at the image corner. This is the pipeline that produced
# the manuscript's 1.06 px result; constraining the PP to the central region
# (commit 435f26f) costs ~0.19 px (1.06 -> 1.25) and breaks reproduction of the
# published numbers. The world-frame Y-axis sign that the constraint targeted is
# handled downstream (pycaso_schur_regularized_ba.py), as in the paper.
cx0, cy0 = 0.5 * W, 0.5 * H
x0_tel = np.array([f_obj_est, WD_est, b_est, cx0, cy0,
                   f_obj_est, theta_fixed, dcL[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo_tel = np.array([1.,1.,0., 0., 0.,
                   20.,0.,-0.3,-10.,-10.,-10.,-10.,-10.,-10.], dtype=np.float64)
hi_tel = np.array([500.,1000.,200., float(W), float(H),
                   200.,0.5,0.3,10.,10.,10.,10.,10.,10.], dtype=np.float64)
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
    paired_z_mm=np.asarray(CALIBRATION_Z_MM, dtype=np.float64),
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
