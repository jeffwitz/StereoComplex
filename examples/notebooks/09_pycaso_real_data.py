# %% [markdown]
# # 09 — StereoComplex on real CMO microscope data (Pycaso)
#
# This notebook runs the full StereoComplex pipeline on real calibration
# images from a **Pycaso CMO stereo microscope**.
#
# **Pipeline:**
# 1. ChArUco detection (OpenCV `CharucoDetector`, legacy pattern)
# 2. Hessian-based corner completion ($|\det H|$ + Otsu + barycentre)
# 3. Double Ray2D TPS denoising (ArUco markers → 165 corners → TPS smoothing)
# 4. Constrained Zernike rayfield fit O(0)+d(2)
# 5. Pixel reprojection errors
# 6. Zernike/pose identifiability diagnostic (gauge mode analysis)
# 7. Gauge-regularized full-pose sweep with Pareto frontier
#
# **Key result — Ray3D as a diagnostic instrument for corner quality:**
# The Zernike rayfield reveals that single-pass TPS leaves residual noise
# that creates a gauge ambiguity (Z₀ drift = 8.5°).  Double TPS eliminates
# it (Z₀ drift = 0.023°).  This feedback loop — Ray2D → Ray3D → diagnose →
# fix Ray2D → verify with Ray3D — is a general strategy for any stereo
# calibration pipeline.  The 2‑D reprojection error is blind to this gauge;
# only the rayfield can see it.

# %%
from __future__ import annotations

import math, sys, time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
)
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_constrained_zernike_rayfield,
)
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
PYCASO_CLONE = Path("examples/pycaso_data")
LEFT_DIR = PYCASO_CLONE / "Exemple" / "Images_example" / "left_calibration11"
RIGHT_DIR = PYCASO_CLONE / "Exemple" / "Images_example" / "right_calibration11"

# ══════════════════════════════════════════════════════════════════════
# Board parameters (Pycaso: 16×12 squares, 0.3 mm, DICT_6X6_250)
# ══════════════════════════════════════════════════════════════════════
NCX, NCY, SQR = 16, 12, 0.3
IMG_SIZE = (2048, 2048)
DICT_NAME = "DICT_6X6_250"

# %% [markdown]
# ## 1 — Corner detection + Hessian completion + ray2D TPS
#
# Pycaso images use an **old-format ChArUco board** requiring
# ``setLegacyPattern(True)`` and ``DICT_6X6_250``.
# Missing corners (at extreme Z) are completed via the Hessian determinant
# $|\det(H)| = |I_{xx}I_{yy} - I_{xy}^2|$, Otsu threshold, and sub-pixel
# barycentre via ``cv2.moments``.
# All 165 corners are then denoised with a thin-plate-spline (TPS) ray2D
# field fitted to the ArUco marker corners via robust IRLS.

# %%


def abs_det_hessian(gray: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    """$R = |I_{xx}I_{yy} - I_{xy}^2|$ via Sobel on Gaussian-blurred image."""
    f = gray.astype(np.float32)
    if f.max() > 2:
        f /= 255.0
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma,
                         borderType=cv2.BORDER_REPLICATE)
    Ixx = cv2.Sobel(f, cv2.CV_64F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(f, cv2.CV_64F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(f, cv2.CV_64F, 1, 1, ksize=3)
    return np.abs(Ixx * Iyy - Ixy * Ixy)


def otsu_mask(response: np.ndarray) -> np.ndarray:
    """Normalise to uint8 and apply Otsu threshold."""
    r8 = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def blob_barycentre(mask: np.ndarray, xi: float, yi: float, d: int,
                    prefer_largest: bool = True) -> tuple[float, float, float]:
    """Sub-pixel centroid of the central/largest blob in a window via ``cv2.moments``."""
    h, w = mask.shape
    x0 = max(0, int(xi) - d); x1 = min(w, int(xi) + d)
    y0 = max(0, int(yi) - d); y1 = min(h, int(yi) + d)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return math.nan, math.nan, math.nan
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if nl <= 1:
        return math.nan, math.nan, math.nan
    cx0, cy0 = xi - x0, yi - y0
    best_lab, best_sc = None, None
    for lab in range(1, nl):
        a = int(stats[lab, cv2.CC_STAT_AREA])
        if a <= 0:
            continue
        sx, sy = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP]
        sw, sh = stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT]
        contains = (sx <= cx0 < sx + sw) and (sy <= cy0 < sy + sh)
        comp = (labels == lab).astype(np.uint8)
        M = cv2.moments(comp, binaryImage=True)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        sc = -float(a) if prefer_largest else float((cx - cx0) ** 2 + (cy - cy0) ** 2) - (
            1e6 if contains else 0.0
        )
        if best_sc is None or sc < best_sc:
            best_sc, best_lab = sc, lab
    if best_lab is None:
        return math.nan, math.nan, math.nan
    comp = (labels == best_lab).astype(np.uint8)
    M = cv2.moments(comp, binaryImage=True)
    return float(x0 + M["m10"] / M["m00"]), float(y0 + M["m01"] / M["m00"]), float(M["m00"])


def win_spot_2pass(mask: np.ndarray, l_step: float, d: int, xi: float, yi: float,
                   prefer_largest: bool = False) -> tuple[float, float, float]:
    """Two-pass blob search: predicted position, then refined centre."""
    d_eff = int(d)
    while d_eff < int(l_step):
        x1, y1, a1 = blob_barycentre(mask, xi, yi, d_eff, prefer_largest)
        if not math.isnan(x1):
            break
        d_eff += max(1, int(l_step) // 8)
    if math.isnan(x1):
        return math.nan, math.nan, math.nan
    x2, y2, a2 = blob_barycentre(mask, x1, y1, int(d), prefer_largest)
    if math.isnan(x2):
        return x1, y1, a1
    return x2, y2, a2


def ids_to_grid(ids: np.ndarray, ncx: int = 16) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.float32).reshape(-1)
    nx = ncx - 1
    return np.column_stack([ids % nx, ids // nx]).astype(np.float32)


def fit_affine(img_pts: np.ndarray, ids_arr: np.ndarray, ncx: int = 16) -> np.ndarray:
    img = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    grid = ids_to_grid(np.asarray(ids_arr, dtype=np.int32).reshape(-1), ncx)
    A, _ = cv2.estimateAffine2D(grid, img, method=cv2.LMEDS)
    if A is None:
        X = np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])
        At, *_ = np.linalg.lstsq(X, img, rcond=None)
        A = At.T.astype(np.float32)
    return A


def project_affine(A: np.ndarray, ids: np.ndarray, ncx: int = 16) -> np.ndarray:
    grid = ids_to_grid(np.asarray(ids, dtype=np.int32).reshape(-1), ncx)
    return (np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])) @ A.T


def complete_corners_hessian(
    gray: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
    ncx: int = 16,
    ncy: int = 12,
    *,
    marker_corners: list | None = None,
    marker_ids: np.ndarray | None = None,
    id_to_obj: dict | None = None,
    chess3_obj: np.ndarray | None = None,
) -> np.ndarray:
    """Fill missing ChArUco corners via Ray2D TPS (or affine) + Hessian barycentre.

    Prediction uses Ray2D TPS on ArUco marker corners when available (handles
    lens distortion), falling back to affine projection.  Missing corners are
    then refined via the Hessian determinant blob barycentre method.

    Returns ``(N, 2)`` array for all ``(ncx-1)*(ncy-1)`` inner corners.
    OpenCV-detected corners are kept (sub-pixel); missing ones are completed.
    """
    nx, ny = ncx - 1, ncy - 1
    n_corners = nx * ny

    R = abs_det_hessian(gray, sigma=9.0)
    mask = otsu_mask(R)

    detected: dict[int, np.ndarray] = {}
    if charuco_ids is not None and len(charuco_ids) > 0:
        ids_arr = np.asarray(charuco_ids).ravel()
        corners_arr = np.asarray(charuco_corners).reshape(-1, 2)
        for i in range(len(ids_arr)):
            detected[int(ids_arr[i])] = corners_arr[i].astype(np.float64)

    cids = sorted(detected.keys())

    # ── Prediction: Ray2D TPS first, affine fallback ──
    pred_xy = None
    if marker_corners is not None and marker_ids is not None and id_to_obj is not None and chess3_obj is not None:
        obj_xy_list, img_uv_list = [], []
        for i in range(len(marker_ids)):
            mid = int(marker_ids[i].ravel()[0])
            o = id_to_obj.get(mid)
            if o is None:
                continue
            mc = np.asarray(marker_corners[i], dtype=np.float64).reshape(-1, 2)
            if mc.shape[0] == 4:
                obj_xy_list.append(o)
                img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            try:
                pred_xy = predict_points_rayfield_tps_robust(
                    np.concatenate(obj_xy_list, axis=0),
                    np.concatenate(img_uv_list, axis=0),
                    chess3_obj[:, :2].astype(np.float64),
                    lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0,
                )
            except Exception:
                pred_xy = None

    if pred_xy is None:
        # Affine fallback
        A = fit_affine(charuco_corners, charuco_ids, ncx)
        pred_xy = project_affine(A, np.arange(n_corners, dtype=np.int32), ncx)

    # ── Grid step + blob size ──
    l_step = 50.0
    if len(cids) >= 2:
        dp = float(np.linalg.norm(detected[cids[-1]] - detected[cids[0]]))
        g0 = ids_to_grid(np.array([cids[0]]), ncx)[0]
        g1 = ids_to_grid(np.array([cids[-1]]), ncx)[0]
        dg = float(np.linalg.norm(g1 - g0))
        if dg > 1e-8:
            l_step = dp / dg

    d_init = max(3, int(l_step * 0.3))
    if len(cids) > 0:
        xA, yA = float(detected[cids[0]][0]), float(detected[cids[0]][1])
        _, _, a_test = win_spot_2pass(mask, l_step, int(l_step * 2 / 3), xA, yA, True)
        if not math.isnan(a_test) and float(a_test) > 0:
            d_init = max(3, int(math.sqrt(float(a_test))))

    # ── Fill: keep detected, complete missing via Hessian ──
    result = np.full((n_corners, 2), np.nan)
    for idx in range(n_corners):
        if idx in detected:
            result[idx] = detected[idx]
        else:
            xi, yi = float(pred_xy[idx, 0]), float(pred_xy[idx, 1])
            xd, yd, _ = win_spot_2pass(mask, l_step, d_init, xi, yi, False)
            if not math.isnan(xd):
                result[idx] = [float(xd), float(yd)]

    for idx in range(n_corners):
        if np.isnan(result[idx, 0]):
            result[idx] = [float(pred_xy[idx, 0]), float(pred_xy[idx, 1])]
    return result

# %%
# ── Build ChArUco runtime ──────────────────────────────────
dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
ocv_board = aruco.CharucoBoard((NCX, NCY), SQR, SQR / 2, dictionary)
ocv_board.setLegacyPattern(True)
chess3 = ocv_board.getChessboardCorners()

# Marker → object mapping (for ray2D TPS)
board_ids = ocv_board.getIds().ravel()
board_obj = ocv_board.getObjPoints()
id_to_obj = {
    int(board_ids[i]): np.asarray(board_obj[i], dtype=np.float64)[:, :2]
    for i in range(len(board_ids))
}

# Detector params (tuned for Pycaso small markers)
params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 75
params.adaptiveThreshWinSizeStep = 4
params.minMarkerPerimeterRate = 0.005
params.maxMarkerPerimeterRate = 0.20
params.polygonalApproxAccuracyRate = 0.03
params.minCornerDistanceRate = 0.02
params.minDistanceToBorder = 1
params.errorCorrectionRate = 0.6

charuco_det = aruco.CharucoDetector(
    ocv_board, aruco.CharucoParameters(), params, aruco.RefineParameters(),
)
aruco_det = aruco.ArucoDetector(dictionary, params)

# ── Paired frames ──────────────────────────────────────────
lz = sorted([f.stem for f in LEFT_DIR.iterdir() if f.suffix == ".png"], key=float)
rz = sorted([f.stem for f in RIGHT_DIR.iterdir() if f.suffix == ".png"], key=float)
paired_z = sorted(set(lz) & set(rz), key=float)
print(f"Paired stereo frames: {len(paired_z)}")

# ── Detect + complete + TPS-denoise ────────────────────────
print("\nDetect → Hessian complete → ray2D TPS denoise:")
denoised_L, denoised_R = [], []
det_counts_L, det_counts_R = [], []
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)

    # ChArUco detection
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L)
    nR = 0 if ids_R is None else len(ids_R)

    # ArUco marker detection (for Ray2D prediction in completion + TPS denoising)
    mk_c_L, mk_ids_L = aruco_det.detectMarkers(lg)[:2]
    mk_c_R, mk_ids_R = aruco_det.detectMarkers(rg)[:2]

    # Hessian completion (uses Ray2D TPS for missing-corner prediction)
    comp_L = complete_corners_hessian(
        lg, cc_L, ids_L, NCX, NCY,
        marker_corners=mk_c_L, marker_ids=mk_ids_L,
        id_to_obj=id_to_obj, chess3_obj=chess3,
    )
    comp_R = complete_corners_hessian(
        rg, cc_R, ids_R, NCX, NCY,
        marker_corners=mk_c_R, marker_ids=mk_ids_R,
        id_to_obj=id_to_obj, chess3_obj=chess3,
    )

    # ray2D TPS denoising (fit on marker corners, predict all 165)
    for mk_c, mk_ids, comp, out in [
        (mk_c_L, mk_ids_L, comp_L, denoised_L),
        (mk_c_R, mk_ids_R, comp_R, denoised_R),
    ]:
        obj_xy_list, img_uv_list = [], []
        if mk_ids is not None:
            for i in range(len(mk_ids)):
                mid = int(mk_ids[i].ravel()[0])
                o = id_to_obj.get(mid)
                if o is None:
                    continue
                mc = np.asarray(mk_c[i], dtype=np.float64).reshape(-1, 2)
                if mc.shape[0] == 4:
                    obj_xy_list.append(o)
                    img_uv_list.append(mc)
        if len(obj_xy_list) >= 4:
            obj_xy = np.concatenate(obj_xy_list, axis=0)
            img_uv = np.concatenate(img_uv_list, axis=0)
            pred = predict_points_rayfield_tps_robust(
                obj_xy, img_uv, chess3[:, :2].astype(np.float64),
                lam=10.0, huber_c=3.0, iters=3, ransac_reproj_px=3.0,
            )
        else:
            pred = comp
        # ── TPS re-denoising on the completed 165 corners ──
        # The Hessian completion gives 165 corners; TPS smoothing on the
        # full set removes residual detection noise while preserving the
        # grid structure (homography base + thin-plate spline residuals).
        re_denoised = predict_points_rayfield_tps_robust(
            chess3[:, :2].astype(np.float64),  # known object positions
            pred.astype(np.float64),            # completed image positions
            chess3[:, :2].astype(np.float64),  # same query = denoising
            lam=3.0, huber_c=1.5, iters=2, ransac_reproj_px=2.0,
        )
        out.append(re_denoised)

    print(f"  {z_str}: L {nL}→165  R {nR}→165")
    det_counts_L.append(nL)
    det_counts_R.append(nR)

# Save detection summary
import json
DET_DIR = Path("docs/assets/pycaso_real_data")
DET_DIR.mkdir(parents=True, exist_ok=True)
detection_summary = {
    "n_pairs": len(paired_z),
    "n_corners_expected": 165,
    "n_corners_completed": 165,
    "left_detected_per_frame": det_counts_L,
    "right_detected_per_frame": det_counts_R,
    "left_mean_detected": float(np.mean(det_counts_L)),
    "right_mean_detected": float(np.mean(det_counts_R)),
    "left_min_detected": int(np.min(det_counts_L)),
    "right_min_detected": int(np.min(det_counts_R)),
    "left_max_detected": int(np.max(det_counts_L)),
    "right_max_detected": int(np.max(det_counts_R)),
}
with open(DET_DIR / "detection_summary.json", "w") as f:
    json.dump(detection_summary, f, indent=2)

# %% [markdown]
# ## 2 — Zernike rayfield fit
#
# Model: **origin order 0** (rigid sub-pupil per channel) +
# **direction order 2** (spatially varying direction correction).
# Poses constrained: shared rotation + X,Y translation, per-pose Z
# (the board is mounted on a Z-only translation stage).

# %%
obj_pts = chess3.astype(np.float64)  # (165, 3), ChArUco ID = index
left_pixels = [dn.astype(np.float64) for dn in denoised_L]
right_pixels = [dn.astype(np.float64) for dn in denoised_R]
point_indices = [np.arange(165, dtype=int) for _ in range(len(paired_z))]

# Initial poses via solvePnP (left camera, fronto-parallel guess)
FX = 25600  # from Z-stack span ratio
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)
rvecs, tvecs = [], []
for lp in left_pixels:
    s, rv, tv = cv2.solvePnP(
        obj_pts.astype(np.float32), lp.astype(np.float32),
        K.astype(np.float32), np.zeros(5, dtype=np.float32),
    )
    rvecs.append(rv.ravel().astype(np.float64) if s else np.zeros(3))
    tvecs.append(tv.ravel().astype(np.float64) if s else np.array([0., 0., 65.]))

obs = CharucoObservationSet(
    object_points_mm=obj_pts,
    pose_rvecs=np.array(rvecs),
    pose_tvecs=np.array(tvecs),
    left_pixels=left_pixels,
    right_pixels=right_pixels,
    point_indices=point_indices,
    noise_std_px=0.0,
    image_size=IMG_SIZE,
)

print(f"Observations: {len(paired_z)} frames × 165 corners = {len(paired_z)*165*2} rays")
print(f"Model: O(0) + d(2), shared R+XY, per-pose Z → 57 params")

t0 = time.time()
lf, rf, zd, opt_R, opt_t = fit_constrained_zernike_rayfield(
    obs,
    image_size=IMG_SIZE,
    K_left=K,
    K_right=K.copy(),
    max_order_d=2,
    max_nfev=500,
    origin_reg_weight=0.0,
)
fit_time = time.time() - t0

print(f"\nFit: {'converged' if zd.converged else 'max NFEV'}, "
      f"{zd.nfev} NFEV, {fit_time:.0f} s")
print(f"Ray RMS: {zd.ray_rms_mm:.4f} mm")

# %% [markdown]
# ## 3 — Reprojection errors
#
# For each of the 3300 rays, the ray at the observed pixel is intersected
# with the board plane.  The 3-D distance to the true board point is
# converted to a local pixel-equivalent residual ``e_mm / (t / fx)``.
# This is a local first-order approximation, not an OpenCV image
# reprojection residual.

# %%
all_err_px_L, all_err_px_R = [], []
all_err_mm_L, all_err_mm_R = [], []

for pi in range(len(paired_z)):
    R_mat = opt_R[pi]
    t_vec = opt_t[pi]
    X_world = (R_mat @ obj_pts.T).T + t_vec[None, :]
    n_plane = R_mat[:, 2]

    for k in range(obj_pts.shape[0]):
        uv_L = left_pixels[pi][k]
        uv_R = right_pixels[pi][k]
        Xk = X_world[k]

        # Left channel
        O, d = lf.ray(np.array([uv_L[0]]), np.array([uv_L[1]]))
        denom = float(np.dot(d[0], n_plane))
        if abs(denom) > 1e-10:
            tL = float(np.dot(t_vec - O[0], n_plane)) / denom
            err = float(np.linalg.norm((O[0] + tL * d[0]) - Xk))
            all_err_mm_L.append(err)
            all_err_px_L.append(err / max(abs(tL), 1.0) * FX)

        # Right channel
        O, d = rf.ray(np.array([uv_R[0]]), np.array([uv_R[1]]))
        denom = float(np.dot(d[0], n_plane))
        if abs(denom) > 1e-10:
            tR = float(np.dot(t_vec - O[0], n_plane)) / denom
            err = float(np.linalg.norm((O[0] + tR * d[0]) - Xk))
            all_err_mm_R.append(err)
            all_err_px_R.append(err / max(abs(tR), 1.0) * FX)

all_err_px_L = np.array(all_err_px_L)
all_err_px_R = np.array(all_err_px_R)
all_err_mm_L = np.array(all_err_mm_L)
all_err_mm_R = np.array(all_err_mm_R)


def _stats(a: np.ndarray, unit: str) -> str:
    return (
        f"RMS={np.sqrt(np.mean(a ** 2)):.2f} {unit}  "
        f"P50={np.percentile(a, 50):.2f} {unit}  "
        f"P95={np.percentile(a, 95):.2f} {unit}  "
        f"Max={np.max(a):.2f} {unit}"
    )


print(f"Local pixel-equivalent reprojection errors (ray→plane intersection):")
print(f"  Left:   {_stats(all_err_px_L, 'px')}")
print(f"  Right:  {_stats(all_err_px_R, 'px')}")
combined = np.concatenate([all_err_px_L, all_err_px_R])
print(f"  Both:   RMS={np.sqrt(np.mean(combined ** 2)):.2f} px  "
      f"P95={np.percentile(combined, 95):.2f} px")

# %% [markdown]
# ## 4 — Rayfield geometry

# %%
Ol, dl = lf.ray(np.array([1024.0]), np.array([1024.0]))
Or, dr = rf.ray(np.array([1024.0]), np.array([1024.0]))
Ol_c, dl_c = Ol[0], dl[0]
Or_c, dr_c = Or[0], dr[0]

print(f"Centre pixel (1024, 1024):")
print(f"  Left:  O = ({Ol_c[0]:.1f}, {Ol_c[1]:.1f}, {Ol_c[2]:.1f}) mm")
print(f"         d = ({dl_c[0]:.4f}, {dl_c[1]:.4f}, {dl_c[2]:.4f})")
print(f"  Right: O = ({Or_c[0]:.1f}, {Or_c[1]:.1f}, {Or_c[2]:.1f}) mm")
print(f"         d = ({dr_c[0]:.4f}, {dr_c[1]:.4f}, {dr_c[2]:.4f})")
print(f"  Baseline: {np.linalg.norm(Or_c - Ol_c):.1f} mm")

# Chief-ray convergence angle
angle = float(np.degrees(np.arccos(np.clip(np.dot(dl_c, dr_c), -1, 1))))
print(f"  Convergence angle: {angle:.1f}°")

print(f"\nPer-frame Z (shared R + XY, per-pose Z):")
for pi, z_str in enumerate(paired_z):
    print(f"  {z_str}: Z = {opt_t[pi][2]:.2f} mm")

# %% [markdown]
# ## 5 — Physical interpretation: CMO-like descriptors from the rayfield
#
# The Zernike rayfield $\mathcal{R}(u,v) = (O(u,v), d(u,v))$ is a **measured
# geometric quantity**.  By postulating a CMO physical model, we can read
# geometric descriptors directly from the rayfield at the centre pixel —
# **without any numerical optimisation**.  These are CMO-consistent
# readouts, not fitted CMO parameters (the Zernike origin has gauge freedom).
#
# ### 5.1 — Sub-pupil positions → baseline
#
# In a CMO stereo microscope, the two channels share a common main objective
# but look through **off-axis sub-pupils** of the objective's aperture.
# Each channel's ray origin $O(u,v)$ is the 3‑D position of that sub-pupil
# in the camera coordinate frame.
#
# From the Zernike rayfield at the centre pixel:
# $$
# O_L = (-12.7,\,-0.1,\,2.7)\;\text{mm},
# \qquad
# O_R = (12.1,\,-0.1,\,2.3)\;\text{mm}
# $$
#
# The **stereo baseline** is the Euclidean distance between the two sub-pupils:
# $$
# b = \|O_R - O_L\| \approx 24.9\;\text{mm}
# $$
#
# The near-antisymmetry $O_L \approx -O_R$ confirms a well-balanced stereo
# geometry.  Under a CMO interpretation these points correspond to effective
# off-axis sub-pupils of the shared objective.
#
# ### 5.2 — Sub-pupil depth → objective focal length
#
# In the CMO model, a sub-pupil is located at $z_{\text{pupil}} = WD - f_{\text{obj}}$,
# where $WD$ is the working distance (objective → object plane) and
# $f_{\text{obj}}$ is the objective's effective focal length.
#
# From the Zernike rayfield:
# $$
# z_{\text{pupil}} = \frac{|O_{L,z}| + |O_{R,z}|}{2} \approx 2.5\;\text{mm}
# $$
#
# The board's Z position is given by the optimised poses; averaging over
# all frames gives $WD \approx 64.7\;\text{mm}$.  Hence:
# $$
# f_{\text{obj}} = WD - z_{\text{pupil}} \approx 64.7 - 2.5 = 62.2\;\text{mm}
# $$
#
# ### 5.3 — Chief-ray directions → convergence angle
#
# At the centre pixel, the left and right chief-ray directions are:
# $$
# d_L = (0.204,\,0.059,\,0.977),
# \qquad
# d_R = (-0.187,\,0.060,\,0.980)
# $$
#
# The **convergence angle** (the angle between the two chief rays) is:
# $$
# \theta = \arccos(d_L \cdot d_R) \approx 22.6^\circ
# $$
#
# This is a strong stereo angle — consistent with a microscope designed for
# 3‑D depth perception at short working distance.
#
# ### 5.4 — Summary: CMO-consistent descriptors read from the rayfield
#
# | Parameter | Symbol | Value | Source |
# |---|---|---|---|
# | Baseline | $b$ | 24.9 mm | $\|O_R - O_L\|$ |
# | Sub-pupil depth | $z_p$ | 2.5 mm | $(|O_{L,z}|+|O_{R,z}|)/2$ |
# | Working distance | $WD$ | 64.7 mm | Mean board Z from poses |
# | Objective focal length | $f_{\text{obj}}$ | 62.2 mm | $WD - z_p$ |
# | Convergence angle | $\theta$ | 22.6° | $\arccos(d_L \cdot d_R)$ |
#
# **No numerical optimisation was used.**  These descriptors are a direct
# geometric reading of the measured rayfield (under Zernike gauge).

# %% [markdown]
# ## 6 — Model comparison: Zernike vs CMO across the field of view
#
# With these CMO-consistent descriptors, we can construct a CMO physical
# model and compare its rays to the Zernike rayfield **across the entire
# sensor**, not just at the centre pixel.  This reveals what a simple CMO
# model captures — and what a perspective CMO model misses.
#
# ### 6.1 — Building the CMO model from derived parameters
#
# The CMO model computes rays from physical parameters.  With $f_{\text{obj}},
# WD, b$ fixed to the values read from the rayfield, and assuming zero
# distortion and principal point at the image centre:
#
# ```python
# cmo = CMOPhysicalStereoModel(
#     f_obj_mm=62.2, working_distance_mm=64.7, b_mm=24.9,
#     f_tube_mm=50.0,
#     cx_principal_px=1024, cy_principal_px=1024,
#     pixel_pitch_mm=0.0055,
#     distortion_left=(0,0,0,0,0),
#     distortion_right=(0,0,0,0,0),
# )
# ```

# %%
from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel

Ol_c, dl_c = lf.ray(np.array([1024.0]), np.array([1024.0]))
Or_c, dr_c = rf.ray(np.array([1024.0]), np.array([1024.0]))
b_est = float(np.linalg.norm(Or_c[0] - Ol_c[0]))
WD_est = float(np.mean([opt_t[i][2] for i in range(len(opt_t))]))
z_pupil = float((abs(Ol_c[0, 2]) + abs(Or_c[0, 2])) / 2)
f_obj_est = WD_est - z_pupil

cmo_params = np.array([
    f_obj_est, WD_est, b_est, 50.0, 1024.0, 1024.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)
cmo = CMOPhysicalStereoModel.from_parameter_vector(
    cmo_params, image_size=IMG_SIZE, pixel_pitch_mm=0.0055,
)

# Evaluate both models on a 11×11 grid spanning the full sensor
u_grid = np.linspace(0, 2047, 11)
v_grid = np.linspace(0, 2047, 11)
uu, vv = np.meshgrid(u_grid, v_grid)
u_flat = uu.ravel()
v_flat = vv.ravel()

Ol_z, dl_z = lf.ray(u_flat, v_flat)
Or_z, dr_z = rf.ray(u_flat, v_flat)
Ol_cmo, dl_cmo = cmo.ray(u_flat, v_flat, "left")
Or_cmo, dr_cmo = cmo.ray(u_flat, v_flat, "right")

ang_L = np.degrees(np.arccos(np.clip(np.sum(dl_z * dl_cmo, axis=1), -1, 1)))
ang_R = np.degrees(np.arccos(np.clip(np.sum(dr_z * dr_cmo, axis=1), -1, 1)))

# %% [markdown]
# ### 6.2 — Direction differences across the field

# %%
print("Direction difference (Zernike − CMO) in degrees")
print(f"{'':>6s}", end="")
for u in u_grid:
    print(f"{u:>8.0f}", end="")
print()
for i, v in enumerate(v_grid):
    print(f"{v:>6.0f}", end="")
    for j in range(len(u_grid)):
        idx = i * len(u_grid) + j
        print(f"{ang_L[idx]:>8.3f}", end="")
    print()
print(f"\nMean angular error: L={ang_L.mean():.2f}°  R={ang_R.mean():.2f}°")

# %% [markdown]
# ### 6.3 — The Y-component reveals telecentricity
#
# The direction vector's Y-component $d_y$ tells us how much the chief ray
# tilts vertically as we move across the sensor.  A **perspective** model
# (like the CMO) predicts a strong linear gradient: $d_y \propto (v - c_y)$.
# A **telecentric** system has $d_y \approx \text{constant}$ across the field.

# %%
print("d_y (Zernike, Left) — nearly constant → telecentric")
for i, v in enumerate(v_grid):
    print(f"  v={v:4.0f}: ", end="")
    for j in range(len(u_grid)):
        idx = i * len(u_grid) + j
        print(f"{dl_z[idx,1]:7.4f}", end="")
    print()

print(f"\n  Zernike d_y range: {dl_z[:,1].max()-dl_z[:,1].min():.4f}")

print("\nd_y (CMO, Left) — linear gradient → perspective")
for i, v in enumerate(v_grid):
    print(f"  v={v:4.0f}: ", end="")
    for j in range(len(u_grid)):
        idx = i * len(u_grid) + j
        print(f"{dl_cmo[idx,1]:7.4f}", end="")
    print()

print(f"\n  CMO d_y range: {dl_cmo[:,1].max()-dl_cmo[:,1].min():.4f}")

# %% [markdown]
# ### 6.4 — Interpretation
#
# The Zernike rayfield shows $d_y \approx 0.059 \pm 0.04$ across the entire
# sensor — a nearly **constant** vertical tilt of ~3.4°.  The CMO model
# predicts $d_y$ varying from −0.116 (top) to +0.116 (bottom) — a **linear
# perspective gradient** with range 0.23, about **3× larger** than the real
# rayfield.
#
# This is the signature of **telecentricity**: the real microscope's chief
# rays are nearly parallel (constant direction) across the field of view,
# whereas the simple CMO model assumes they all converge to a single sub-pupil
# point.  The tube lens and objective together create an approximately
# **object-space telecentric** condition in the Y direction.
#
# The CMO model captures the **first-order geometry** (sub-pupil positions,
# baseline, working distance, convergence angle) correctly, but cannot
# reproduce the **detailed ray-direction structure** of the real optics.
# This structural mismatch explains why fitting the CMO model to the Zernike
# rayfield produces a ray-space RMS of ~3.7 mm and a pixel reprojection RMS
# of ~600 px — the optimiser pushes the principal point to extreme values and
# saturates the distortion coefficients at their bounds, trying to compensate
# for a perspective-to-telecentric gap that no combination of its 18
# parameters can bridge.
#
# The Zernike rayfield, with 42 parameters (origin order 0 + direction
# order 2, per channel), has enough flexibility to capture the real ray
# geometry and achieves **0.47 px RMS** stereo reprojection.

# %% [markdown]
# ## 7 — Conclusions
#
# 1. **StereoComplex calibrates a real CMO microscope where OpenCV fails**
#    (OpenCV stereo RMS > 300 px vs. StereoComplex 0.47 px).
#
# 2. **The Zernike rayfield is an observable**: from it, we read CMO-consistent
#    geometric descriptors
#    $b \approx 24.9\;\text{mm}$, $f_{\text{obj}} \approx 62\;\text{mm}$,
#    $WD \approx 65\;\text{mm}$, and a convergence angle of $22.6^\circ$ —
#    all without running a physical model fit.
#
# 3. **Model comparison in ray space is a diagnostic**: the Zernike-vs-CMO
#    comparison across the FOV reveals that the real optics are more
#    telecentric than the perspective CMO model, explaining why the CMO
#    fit cannot achieve better than ~600 px reprojection.
#
# 4. **The workflow generalises**: the same rayfield → physical reading →
#    model comparison sequence can be applied to any stereo microscope to
#    identify its optical architecture and quantify deviations from ideal
#    models.

# %% [markdown]
# ## 8 — Zernike order sweep: how many parameters are needed?
#
# The baseline model uses O(0)+d(2): rigid sub-pupil (3 params per channel)
# + spatially-varying direction correction up to radial order 2 (18 params
# per channel).  Can higher orders reduce the reprojection error?
#
# We sweep O order 0–2 and d order 2–4.  Each model is fitted with the same
# constrained poses (shared R+XY, per-pose Z).

# %%
zernike_mode_count = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15}
orders_to_test = [(0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (1, 4), (2, 4)]

print(f"{'Model':>12s}  {'Params':>6s}  {'RMS(px)':>9s}  {'P95(px)':>9s}  {'NFEV':>5s}  {'Time':>5s}")
print("-" * 55)
results = []
for o_order, d_order in orders_to_test:
    nO = zernike_mode_count[o_order] * 3
    nd = zernike_mode_count[d_order] * 3
    n_params = (nO + nd) * 2 + 15  # 2×(O+d) + (3 rot + 2 XY + 10 Z) poses

    t0 = time.time()
    _lf, _rf, _zd, _oR, _ot = fit_constrained_zernike_rayfield(
        obs, image_size=IMG_SIZE, K_left=K, K_right=K.copy(),
        max_order_o=o_order, max_order_d=d_order,
        max_nfev=500, origin_reg_weight=0.0,
    )
    elapsed = time.time() - t0

    # Compute reprojection
    _eL, _eR = [], []
    for pi in range(len(paired_z)):
        Rm, tv = _oR[pi], _ot[pi]
        Xw = (Rm @ obj_pts.T).T + tv[None, :]
        n_plane = Rm[:, 2]
        for k in range(obj_pts.shape[0]):
            for uv, Xk, field, el in [
                (left_pixels[pi][k], Xw[k], _lf, _eL),
                (right_pixels[pi][k], Xw[k], _rf, _eR),
            ]:
                O, d = field.ray(np.array([uv[0]]), np.array([uv[1]]))
                dn = float(np.dot(d[0], n_plane))
                if abs(dn) > 1e-10:
                    tL = float(np.dot(tv - O[0], n_plane)) / dn
                    el.append(
                        float(np.linalg.norm((O[0] + tL * d[0]) - Xk))
                        / max(abs(tL), 1.0) * FX
                    )
    _ee = np.concatenate([np.array(_eL), np.array(_eR)])

    # Physical parameters from centre pixel
    _Ol, _dl = _lf.ray(np.array([1024.0]), np.array([1024.0]))
    _Or, _dr = _rf.ray(np.array([1024.0]), np.array([1024.0]))
    _b = float(np.linalg.norm(_Or[0] - _Ol[0]))
    _zp = float((abs(_Ol[0, 2]) + abs(_Or[0, 2])) / 2)
    _WD = float(np.mean([_ot[i][2] for i in range(len(_ot))]))
    _f_obj = _WD - _zp
    _angle = float(np.degrees(np.arccos(np.clip(np.dot(_dl[0], _dr[0]), -1, 1))))

    results.append({
        "O": o_order, "d": d_order, "p": n_params,
        "rms": np.sqrt(np.mean(_ee ** 2)), "p95": np.percentile(_ee, 95),
        "nfev": _zd.nfev, "time": elapsed,
        "b": _b, "f_obj": _f_obj, "WD": _WD, "angle": _angle,
    })
    marker = " ← baseline" if (o_order, d_order) == (0, 2) else ""
    print(
        f"O({o_order})+d({d_order})  {n_params:>6d}  "
        f"{results[-1]['rms']:>8.3f}  {results[-1]['p95']:>8.3f}  "
        f"{results[-1]['nfev']:>5d}  {elapsed:>4.0f}s{marker}"
    )

best = min(results, key=lambda r: r["rms"])
baseline = results[0]
improvement = (baseline["rms"] - best["rms"]) / baseline["rms"] * 100

# Save sweep results
import json
SWEEP_DIR = Path("docs/assets/pycaso_real_data")
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
with open(SWEEP_DIR / "zernike_order_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSweep saved to {SWEEP_DIR / 'zernike_order_sweep.json'}")

# Save comprehensive summary.json
summary = {
    "dataset": {
        "n_pairs": len(paired_z),
        "image_size": list(IMG_SIZE),
        "board": f"{NCX-1}×{NCY-1} ChArUco, {SQR} mm, {DICT_NAME}",
        "z_range_mm": [float(paired_z[0]), float(paired_z[-1])],
    },
    "detection": detection_summary,
    "zernike_fit": {
        "model": "O(0)+d(2), constrained poses (shared R+XY, per-pose Z)",
        "n_params": 57,
        "converged": bool(zd.converged),
        "nfev": int(zd.nfev),
        "ray_rms_mm": float(zd.ray_rms_mm),
    },
    "reprojection": {
        "metric": "local pixel-equivalent (ray-plane intersection)",
        "left_rms_px": float(np.sqrt(np.mean(all_err_px_L ** 2))),
        "left_p95_px": float(np.percentile(all_err_px_L, 95)),
        "right_rms_px": float(np.sqrt(np.mean(all_err_px_R ** 2))),
        "right_p95_px": float(np.percentile(all_err_px_R, 95)),
        "both_rms_px": float(np.sqrt(np.mean(np.concatenate([all_err_px_L, all_err_px_R]) ** 2))),
    },
    "cmo_descriptors": {
        "baseline_mm": float(np.linalg.norm(Or_c[0] - Ol_c[0])),
        "subpupil_depth_mm": float((abs(float(Ol_c[0,2])) + abs(float(Or_c[0,2]))) / 2),
        "working_distance_mm": float(np.mean([float(opt_t[i][2]) for i in range(len(opt_t))])),
        "convergence_angle_deg": float(np.degrees(np.arccos(np.clip(float(np.dot(dl_c[0], dr_c[0])), -1.0, 1.0)))),
        "f_obj_mm": 0.0,
    },
    "order_sweep_best": {
        "model": f"O({best['O']})+d({best['d']})",
        "n_params": best["p"],
        "rms_px": best["rms"],
        "p95_px": best["p95"],
    },
}
s = summary["cmo_descriptors"]
s["f_obj_mm"] = s["working_distance_mm"] - s["subpupil_depth_mm"]
with open(SWEEP_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {SWEEP_DIR / 'summary.json'}")

# %% [markdown]
# ### 8.1 — Interpretation
#
# The baseline O(0)+d(2) already achieves 0.47 px RMS.  Adding more
# parameters reduces this further to 0.41 px — a **13 %
# improvement** — before plateauing at O(2)+d(3).  Beyond this point,
# the fit starts modelling detection noise rather than optical structure.
#
# **Physical parameter stability:** WD is rock-solid (spread < 0.5 mm).
# f_obj varies by ~1.5 mm (2 %).  The baseline b is the most sensitive:
# O(0) gives b ≈ 25 mm (rigid sub-pupil, most physical interpretation),
# while O(≥1) allows the origin to vary spatially and "absorbs" ~5 mm
# of baseline into per-pixel variations — a known gauge freedom when the
# origin field has degrees of freedom beyond piston.

# %%
print("Physical parameter stability across Zernike orders:")
print(f"{'Model':>12s}  {'b(mm)':>7s}  {'f_obj':>7s}  {'WD':>7s}  {'θ(°)':>7s}")
for r in results:
    print(
        f"O({r['O']})+d({r['d']})  "
        f"{r['b']:>7.2f}  {r['f_obj']:>7.2f}  {r['WD']:>7.2f}  {r['angle']:>7.2f}"
    )

b_vals = [r["b"] for r in results]
f_vals = [r["f_obj"] for r in results]
wd_vals = [r["WD"] for r in results]
print(f"\nSpread (max−min): b={max(b_vals)-min(b_vals):.2f}mm  "
      f"f_obj={max(f_vals)-min(f_vals):.2f}mm  WD={max(wd_vals)-min(wd_vals):.2f}mm")

# %% [markdown]
# ## 9 — Compact physical model: telecentric CMO with pupil shear
#
# The Zernike rayfield gives us a measured $(O, d)$.  From it we diagnosed
# that the perspective CMO model fails because $d_y$ is nearly constant
# (telecentric), not linearly varying (perspective).  We built
# `CMOTelecentricStereoModel`: rigid sub-pupils + affine direction field.
# This achieved 0.16 mm ray-space RMS (22× better than perspective).
#
# The Plucker diagnostic revealed the remaining error is in the **line
# moment** $O \times d$, not the direction.  We now add **pupil shear** —
# a small affine variation of the origin, transverse to the direction:
#
# $$O_c(u,v) = S_c + \rho_{x,c} \tilde{u}\, e_x + \rho_{y,c} \tilde{v}\, e_y$$
#
# with $\Delta O_c \leftarrow (I - d_c d_c^T) \Delta O_c$ (transverse gauge).

# %%
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _ray_rms
from stereocomplex.physics.model_selection import rayfield_two_plane_residuals, _grid_pixels
from scipy.optimize import least_squares

support = _grid_pixels(IMG_SIZE, (12, 9))

# Per-channel slopes + shared pupil shear — 14 params
f_obj_est = WD_est - float((abs(Ol_c[0,2]) + abs(Or_c[0,2])) / 2)
theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))
base14 = np.array([f_obj_est, WD_est, b_est, 0,0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
x0_14 = np.array([1024., 1024., f_obj_est, theta_fixed, dl_c[0,1], 0.,0.,0.,0.,0.,0.], dtype=np.float64)
lo14 = np.array([0.,0.,20.,0.,-0.3, -10.,-10.,-10.,-10., -10.,-10.], dtype=np.float64)
hi14 = np.array([2048.,2048.,200.,0.5,0.3, 10.,10.,10.,10., 10.,10.], dtype=np.float64)

def _build_ps(x, b):
    p = b.copy()
    for i in range(len(x)):
        p[3 + i] = x[i]
    return CMOTelecentricStereoModel.from_parameter_vector(p, pixel_pitch_mm=0.0055, image_size=IMG_SIZE)

def _res_ps(x):
    m = _build_ps(x, base14)
    l, r = m.channel("left"), m.channel("right")
    return np.concatenate([
        rayfield_two_plane_residuals(lf, l, support, z_planes=(50., 80.)),
        rayfield_two_plane_residuals(rf, r, support, z_planes=(50., 80.)),
    ])

sol_ps = least_squares(_res_ps, x0=x0_14, bounds=(lo14, hi14), loss="linear",
                        max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
m_ps = _build_ps(sol_ps.x, base14)
l_ps = rayfield_two_plane_residuals(lf, m_ps.channel("left"), support, z_planes=(50., 80.))
r_ps = rayfield_two_plane_residuals(rf, m_ps.channel("right"), support, z_planes=(50., 80.))
rms_ps = float(np.sqrt(0.5 * (_ray_rms(l_ps)**2 + _ray_rms(r_ps)**2)))
fp = m_ps.parameter_dict()["free"]

print(f"Telecentric + pupil shear (14 params):")
print(f"  Ray-space RMS = {rms_ps:.4f} mm  (Zernike ref: {zd.ray_rms_mm:.4f} mm)")
print(f"  Slopes: L(sx={fp['s_x_L']:.3f}, sy={fp['s_y_L']:.3f})  R(sx={fp['s_x_R']:.3f}, sy={fp['s_y_R']:.3f})")
print(f"  Pupil shear: rho=({fp['rho_x_L']:.3f}, {fp['rho_y_L']:.3f}) mm")
print(f"  theta={fp['theta_convergence_half_deg']:.1f}  d_y={fp['d_y_common']:.4f}  f_ang={fp['f_angular_mm']:.1f} mm")
print(f"  PP=({fp['cx_principal_px']:.0f}, {fp['cy_principal_px']:.0f})")

# Plucker re-check after shear
u_test = np.linspace(0, 2047, 11); v_test = np.linspace(0, 2047, 11)
uu, vv = np.meshgrid(u_test, v_test); uf, vf = uu.ravel(), vv.ravel()
O_z, d_z = lf.ray(uf, vf)
O_mL, d_mL = m_ps.ray(uf, vf, "left")
O_mR, d_mR = m_ps.ray(uf, vf, "right")
O_zR, d_zR = rf.ray(uf, vf)
m_zL = np.cross(O_z, d_z); m_mL = np.cross(O_mL, d_mL)
m_zR = np.cross(O_zR, d_zR); m_mR = np.cross(O_mR, d_mR)

dir_err = np.degrees(np.arccos(np.clip(np.sum(d_z * d_mL, axis=1), -1, 1)))
mom_err_L = np.linalg.norm(m_zL - m_mL, axis=1)
mom_err_R = np.linalg.norm(m_zR - m_mR, axis=1)
print(f"\n  Plucker after shear:")
print(f"    Direction RMS = {np.sqrt(np.mean(dir_err**2)):.2f} deg")
print(f"    Moment RMS L  = {np.sqrt(np.mean(mom_err_L**2)):.3f} mm")
print(f"    Moment RMS R  = {np.sqrt(np.mean(mom_err_R**2)):.3f} mm")
print(f"    (before shear: direction ~2.0 deg, moment ~0.5 mm)")

# Pixel-equivalent reprojection errors
class _W:
    def __init__(s, m, c): s.m = m; s.c = c
    def ray(s, u, v): return s.m.channel(s.c).ray(u, v)
epx_ps = []
for pi in range(len(paired_z)):
    Rm, t = opt_R[pi], opt_t[pi]
    Xw = (Rm @ obj_pts.T).T + t[None, :]
    n_plane = Rm[:, 2]
    for k in range(obj_pts.shape[0]):
        for uv, f in [(left_pixels[pi][k], _W(m_ps, "left")), (right_pixels[pi][k], _W(m_ps, "right"))]:
            O, d = f.ray(np.array([uv[0]]), np.array([uv[1]]))
            dn = float(np.dot(d[0], n_plane))
            if abs(dn) > 1e-10:
                tL = float(np.dot(t - O[0], n_plane)) / dn
                e = float(np.linalg.norm((O[0] + tL * d[0]) - Xw[k]))
                epx_ps.append(e / max(abs(tL), 1.0) * FX)
epx_ps = np.array(epx_ps)
print(f"\n  Pixel-equivalent reprojection:")
print(f"    RMS = {np.sqrt(np.mean(epx_ps**2)):.2f} px  P50 = {np.percentile(epx_ps, 50):.2f} px  P95 = {np.percentile(epx_ps, 95):.2f} px")
print(f"    (Zernike: 0.47 px, telecentric no-shear: ~28 px, perspective: ~86 px)")

# %% [markdown]
# ## 10 — Zernike/pose identifiability: conditioning diagnostic
#
# The difference between constrained-pose Zernike (0.47 px) and full-pose
# Zernike (0.17 px) hides an identifiability problem.  Freeing poses
# changes the rayfield dramatically: Δd ≈ 8.5°, Δm ≈ 9.7 mm, and baseline
# jumps from 17 mm to 28 mm.  We diagnose whether this is caused by poorly
# constrained Zernike modes trading off with pose parameters.
#
# ### 10.1 — Design matrix conditioning
#
# The Zernike basis on the square sensor is well-conditioned
# (cond(B₂) = 4.8, cond(B₄) = 14.5).  However, Z₀⁰ and Z₂⁰ are
# not orthogonal on the square (off-diagonal correlation 0.56), and on
# sparse ChArUco-like sampling the conditioning degrades to cond(B₄) = 71.
# Z₂²(cos) loads almost entirely onto the last singular vector — it is
# the least observable mode.

# %%
import json
from pathlib import Path

SWEEP_DIR = Path("docs/assets/pycaso_real_data")

# Load the diagnostic
with open(SWEEP_DIR / "zernike_conditioning_diagnostic.json") as f:
    diag = json.load(f)

# Phase 1: design matrix
p1 = diag["phase1_design_matrix"]
print("Design matrix conditioning:")
for key in ["regular_grid_41x41", "sparse_central_85pct"]:
    d = p1[key]
    print(f"  {key}: n={d['n_pixels']}, cond(B₂)={d['condition_number_order2']:.1f}, "
          f"cond(B₄)={d['condition_number_order4']:.1f}, "
          f"max_corr={d['max_off_diagonal_correlation']:.3f}")

# Phase 2: modal decomposition
p2 = diag["phase2_modal_decomposition"]
print(f"\nModal decomposition of Δd = d_full − d_constrained:")
for ch in ["left", "right"]:
    dch = p2[ch]
    print(f"  {ch}: RMS={dch['delta_direction_rms_deg']:.2f}°, "
          f"P50={dch['delta_direction_p50_deg']:.2f}°, "
          f"Δm RMS={dch['delta_moment_rms_mm']:.2f} mm")
    for m in dch["top_direction_modes"][:4]:
        if m["frac_var_d"] < 0.005:
            continue
        print(f"    {m['mode']:18s}  {m['frac_var_d']*100:5.1f}%  "
              f"({m['rms_d_deg']:.2f}°)  "
              f"cd=({m['c_d'][0]:+.4f}, {m['c_d'][1]:+.4f}, {m['c_d'][2]:+.4f})")

# Phase 3: sensitivity
p3 = diag["phase3_sensitivity"]
sens = p3["mode_sensitivities"]
print(f"\nMode → physical indicator sensitivity:")
print(f"  {'Mode':18s}  {'Top O→':>25s}  {'Top d→':>25s}")
for s in sens:
    o_label = f"{s['top_O_key']}={s['top_O_val']:.2f}"
    d_label = f"{s['top_d_key']}={s['top_d_val']:.2f}"
    flag = " ★" if s["top_d_val"] > 1.0 else ""
    print(f"  {s['mode']:18s}  {o_label:>25s}  {d_label:>25s}{flag}")

# Phase 4: stability
p4 = diag["phase4_stability"]
stab = p4["coefficient_variation"]
print(f"\nStability: {p4['n_stable']} stable, {p4['n_moderate']} moderate, "
      f"{p4['n_unstable']} unstable out of {len(stab)} modes")
print(f"  {'Mode':18s}  {'ΔO_L':>8s}  {'ΔO_R':>8s}  {'Δd_L':>8s}  {'Δd_R':>8s}  {'Stability':>10s}")
for s in stab:
    print(f"  {s['mode']:18s}  {s['delta_O_L_mm']:8.3f}  {s['delta_O_R_mm']:8.3f}  "
          f"{s['delta_d_L']:8.4f}  {s['delta_d_R']:8.4f}  {s['stability']:>10s}")

# Physical indicator shifts
ind_deltas = p3["indicator_deltas"]
print(f"\nPhysical indicator shifts (full − constrained):")
print(f"  baseline:          {ind_deltas['baseline_mm']:+.2f} mm")
print(f"  convergence angle: {ind_deltas['convergence_angle_deg']:+.2f}°")
print(f"  dy_range_L:        {ind_deltas['dy_range_L']:+.4f}")
print(f"  dy_range_R:        {ind_deltas['dy_range_R']:+.4f}")
print(f"  subpupil_depth:    {ind_deltas['subpupil_depth_mm']:+.2f} mm")

# %% [markdown]
# ### 10.2 — Interpretation: the rayfield as a diagnostic instrument
#
# **Key finding: 90 % of Δd is Z₀⁰ (global direction piston).**
# This is a gauge freedom — equivalent to changing the effective focal
# length — that poses can absorb.  The 2‑D reprojection error is
# **blind** to this: a full-pose fit absorbs corner noise into rayfield
# distortions without increasing pixel RMS.
#
# **This turns the gauge into a quality test for 2‑D preprocessing.**
# If constrained and full-pose rayfields differ substantially, the
# corners carry residual noise that the optimizer exploits.  The fix
# is to improve Ray2D, not to constrain poses:
#
# ```text
# Ray2D → Ray3D → ΔZ₀ < 0.1° ? ──no──→ improve Ray2D → repeat
#                        │
#                       yes
#                        │
#               rayfield stable
#               → physical interpretation reliable
# ```
#
# **The double TPS pass resolves the gauge.**  Using the completed 165
# corners as TPS control points (λ=3, Huber c=1.5) eliminates the noise
# that was exciting the gauge mode.  The second TPS pass is a denoising
# regularizer whose validity is confirmed not by the 2‑D residual but by
# the disappearance of gauge drift in Ray3D.
#
# **Stability criterion:**
#
# | Criterion | Threshold | Status |
# |---|---|---|
# | ΔZ₀ (direction piston) | < 0.1° | depends on preprocessing |
# | \|Δb\| (baseline) | < 0.5 mm | depends on preprocessing |
# | \|Δθ\| (convergence) | < 0.2° | depends on preprocessing |
#
# When these pass, the rayfield is **stable** — the choice of pose model
# no longer affects physical interpretation.
#
# This test is automated in section 10.3 (set ``RUN_SWEEP = True``).

# %% [markdown]
# ### 10.3 — Verification: gauge-regularized sweep
#
# As a verification (not a fix), we run a sweep of gauge-regularized
# full-pose fits anchoring Z₀ and Z₁ to the constrained solution with
# angular tolerance σ.  If preprocessing is sufficient, even the
# unregularized fit (σ→∞) should show negligible drift.
#
# Set `RUN_SWEEP = True` to execute (takes ~10–15 min on 10 frames).

# %%
RUN_SWEEP = False  # Set to True to run the regularization sweep

if RUN_SWEEP:
    import time
    from scipy.optimize import least_squares
    from stereocomplex.benchmarks.rayfield_from_observations import (
        estimate_initial_poses_from_central_pinhole,
        ZernikeFitDiagnostics,
    )
    from stereocomplex.rayfields.zernike_origin_field import (
        ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients,
    )
    from stereocomplex.core.model_compact.zernike import eval_real_zernike, zernike_modes

    # ── Prior: constrained-fit direction coefficients ──
    prior_L_d = lf.direction_coeffs.copy()  # (6, 3)
    prior_R_d = rf.direction_coeffs.copy()

    # ── Build full-pose fit with gauge priors ──
    W_img, H_img = IMG_SIZE
    config2 = ZernikeOriginFieldConfig(image_size=IMG_SIZE, max_order=2)
    modes2 = config2.modes()
    n_modes2 = len(modes2)
    n_zernike = n_modes2 * 6  # 3 origin + 3 direction per mode

    # Pre-group observations (reuse the existing obs, denoised_L/R from earlier cells)
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
    obj_pts = obs.object_points_mm
    n_poses = len(obs.left_pixels)

    def _precompute(u_arr, v_arr, K):
        xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(W_img - 1) - 1.0
        zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(H_img - 1) - 1.0
        rho = np.sqrt(xi*xi + zeta*zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, n_modes2), dtype=np.float64)
        for j, mode in enumerate(modes2):
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
        mL = poseL == pi; mR = poseR == pi
        if mL.any():
            g = _G(); g.pose_idx = pi
            g.A, g.d0 = _precompute(uL[mL], vL[mL], K)
            g.X_local = obj_pts[idxL_arr[mL]]; groups_L.append(g)
        if mR.any():
            g = _G(); g.pose_idx = pi
            g.A, g.d0 = _precompute(uR[mR], vR[mR], K.copy())
            g.X_local = obj_pts[idxR_arr[mR]]; groups_R.append(g)

    # Initial poses from constrained solution
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
            reg = [rL, rR]
            # Origin Z regularization
            reg.append(np.sqrt(1e-3) * oL[:, 2])
            reg.append(np.sqrt(1e-3) * oR[:, 2])
            # Gauge prior
            for m in range(n_modes2):
                if not prior_mask[m]: continue
                s = prior_sigmas[m]
                for c in range(3):
                    reg.append(np.array([(dL[m, c] - prior_L_d[m, c]) / s]))
                    reg.append(np.array([(dR[m, c] - prior_R_d[m, c]) / s]))
            return np.concatenate(reg)

        n_half = n_zernike // 2
        o_lo = np.full(n_half, -np.inf); o_hi = np.full(n_half, np.inf)
        for j in range(2, n_half, 3):
            o_lo[j] = -20.0; o_hi[j] = 20.0
        d_lo = np.full(n_half, -0.5); d_hi = np.full(n_half, 0.5)
        c_lo = np.concatenate([o_lo, d_lo]); c_hi = np.concatenate([o_hi, d_hi])
        bounds = (
            np.concatenate([c_lo, c_lo, x0_poses_arr - 0.3]),
            np.concatenate([c_hi, c_hi, x0_poses_arr + 0.3]),
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
        # Build fields
        def _field(cfs, Kk):
            a = np.asarray(cfs, dtype=np.float64).reshape(-1)
            return ZernikeRayField(K=Kk, config=config2,
                coefficients=ZernikeRayFieldCoefficients(
                    origin_coeffs=a[:n_modes2*3].reshape(n_modes2, 3),
                    direction_coeffs=a[n_modes2*3:].reshape(n_modes2, 3)))
        lf_out = _field(sol.x[:n_zernike], K)
        rf_out = _field(sol.x[n_zernike:2*n_zernike], K.copy())
        return lf_out, rf_out, sol

    # ── Sweep ──
    sweep_runs = [
        ("full_pose_baseline", 100.0, 100.0),
        ("z0_0.05deg",         0.05, 100.0),
        ("z0_0.1deg",          0.1,  100.0),
        ("z0_0.2deg",          0.2,  100.0),
        ("z0_0.5deg",          0.5,  100.0),
        ("z0_1.0deg",          1.0,  100.0),
        ("z0z1_0.1_0.5",       0.1,  0.5),
        ("z0z1_0.2_0.5",       0.2,  0.5),
        ("z0z1_0.2_1.0",       0.2,  1.0),
        ("z0z1_0.5_1.0",       0.5,  1.0),
    ]

    print(f"\n{'Run':>22s}  {'RMSmm':>9s}  {'Z0△°':>7s}  {'Z1△°':>7s}  "
          f"{'b_mm':>7s}  {'θ°':>6s}  {'dy_ranL':>8s}  {'NFEV':>5s}  {'s':>4s}")
    print("-" * 100)

    sweep_results = []
    for label, sz0, sz1 in sweep_runs:
        t0 = time.time()
        lf_s, rf_s, sol = fit_one(sz0, sz1, max_nfev=400)
        elapsed = time.time() - t0

        dL_s = lf_s.direction_coeffs; dR_s = rf_s.direction_coeffs
        dz0_L = np.linalg.norm(dL_s[0] - prior_L_d[0])
        dz0_R = np.linalg.norm(dR_s[0] - prior_R_d[0])
        dz1_L = 0.5*(np.linalg.norm(dL_s[1] - prior_L_d[1]) + np.linalg.norm(dL_s[2] - prior_L_d[2]))
        dz1_R = 0.5*(np.linalg.norm(dR_s[1] - prior_R_d[1]) + np.linalg.norm(dR_s[2] - prior_R_d[2]))
        drift_z0 = float(np.degrees(0.5*(dz0_L + dz0_R)))
        drift_z1 = float(np.degrees(0.5*(dz1_L + dz1_R)))

        # Physical indicators on 41×41 grid
        u_g, v_g = np.meshgrid(np.linspace(0, W_img-1, 41), np.linspace(0, H_img-1, 41))
        O_L, d_L = lf_s.ray(u_g.ravel(), v_g.ravel())
        O_R, d_R = rf_s.ray(u_g.ravel(), v_g.ravel())
        uc, vc = np.array([1024.]), np.array([1024.])
        _, dL_c = lf_s.ray(uc, vc); _, dR_c = rf_s.ray(uc, vc)
        b_val = float(np.linalg.norm(np.mean(O_R, axis=0) - np.mean(O_L, axis=0)))
        theta_val = float(np.degrees(np.arccos(np.clip(np.dot(dL_c[0], dR_c[0]), -1.0, 1.0))))
        dy_range_L = float(np.max(d_L[:, 1]) - np.min(d_L[:, 1]))

        # RMS from final residuals
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

    # Pareto frontier
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

    # Save
    import json
    with open(SWEEP_DIR / "zernike_gauge_regularization_sweep.json", "w") as f:
        json.dump({
            "description": "Gauge-regularized full-pose Zernike sweep — Z0/Z1 direction anchor",
            "constrained_rms_mm": zd.ray_rms_mm,
            "sweep": sweep_results,
            "pareto_optimal": [r["label"] for r in pareto],
        }, f, indent=2)
    print(f"\nSaved to {SWEEP_DIR / 'zernike_gauge_regularization_sweep.json'}")

    # ── Stability test ──
    bl = sweep_results[0]  # full_pose_baseline
    dz0_pass = bl["drift_z0_deg"] < 0.1
    db_pass = abs(bl["baseline_mm"] - 24.9) < 0.5
    dtheta_pass = abs(bl["convergence_angle_deg"] - 22.3) < 0.2
    all_pass = dz0_pass and db_pass and dtheta_pass
    print(f"\n=== Rayfield stability test ===")
    print(f"  ΔZ₀  = {bl['drift_z0_deg']:.4f}°  (< 0.1°)  {'PASS' if dz0_pass else 'FAIL'}")
    print(f"  |Δb| = {abs(bl['baseline_mm']-24.9):.2f} mm  (< 0.5mm) {'PASS' if db_pass else 'FAIL'}")
    print(f"  |Δθ| = {abs(bl['convergence_angle_deg']-22.3):.2f}°  (< 0.2°)  {'PASS' if dtheta_pass else 'FAIL'}")
    print(f"  {'→ RAYFIELD STABLE' if all_pass else '→ IMPROVE Ray2D PREPROCESSING'}")

    # ── Pareto plot ──
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1) RMS vs Z0 drift (Pareto frontier)
    ax = axes[0]
    all_rms = [r["ray_rms_mm"] for r in sweep_results]
    all_z0 = [r["drift_z0_deg"] for r in sweep_results]
    pareto_rms = [r["ray_rms_mm"] for r in pareto]
    pareto_z0 = [r["drift_z0_deg"] for r in pareto]
    ax.scatter(all_rms, all_z0, c="steelblue", s=60, zorder=3, label="all runs")
    ax.scatter(pareto_rms, pareto_z0, c="darkorange", s=100, zorder=4, label="Pareto-optimal")
    # Annotate Pareto points
    for r in pareto:
        ax.annotate(r["label"], (r["ray_rms_mm"], r["drift_z0_deg"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=7, alpha=0.8)
    ax.axvline(x=zd.ray_rms_mm, color="gray", ls="--", alpha=0.5, label="constrained ref")
    ax.set_xlabel("Ray RMS (mm)")
    ax.set_ylabel("Z₀ drift (°)")
    ax.set_title("Pareto: RMS vs gauge drift")
    ax.legend(fontsize=7)

    # 2) Baseline stability
    ax = axes[1]
    for r in sweep_results:
        sigma_label = f"σ₀={r['sigma_z0_deg']:.2f}" if r["sigma_z1_deg"] > 50 else f"σ₀={r['sigma_z0_deg']:.2f},σ₁={r['sigma_z1_deg']:.1f}"
        ax.plot(r["drift_z0_deg"], r["baseline_mm"], "o", ms=8, alpha=0.7)
    ax.set_xlabel("Z₀ drift (°)")
    ax.set_ylabel("Baseline (mm)")
    ax.set_title("Baseline vs Z₀ drift")

    # 3) Convergence angle stability
    ax = axes[2]
    for r in sweep_results:
        ax.plot(r["drift_z0_deg"], r["convergence_angle_deg"], "o", ms=8, alpha=0.7)
    ax.set_xlabel("Z₀ drift (°)")
    ax.set_ylabel("Convergence angle (°)")
    ax.set_title("Convergence angle vs Z₀ drift")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SWEEP_DIR / "pareto_gauge_regularization.png", dpi=150,
                bbox_inches="tight")
    print(f"Pareto plot saved to {SWEEP_DIR / 'pareto_gauge_regularization.png'}")
    plt.show()

else:
    print("RUN_SWEEP = False — skipping regularization sweep. "
          "Set to True to run (takes ~10-15 min).")

# %% [markdown]
# ## 11 — Conclusions
#
# 1. **StereoComplex calibrates a real CMO microscope where OpenCV fails**
#    (OpenCV stereo RMS > 300 px vs. StereoComplex 0.47 px baseline,
#    0.41 px with O(2)+d(3)).
#
# 2. **The Zernike rayfield is an observable**: from it, we read CMO-consistent
#    geometric descriptors
#    $b \approx 24.9\;\text{mm}$, $f_{\text{obj}} \approx 62\;\text{mm}$,
#    $WD \approx 65\;\text{mm}$, and a convergence angle of $22.6^\circ$ —
#    all without running a physical model fit.
#
# 3. **Physical parameters are largely stable across Zernike orders.**
#    WD varies < 0.5 mm, f_obj ~1.5 mm.  The baseline b shows the most
#    sensitivity (20–25 mm) because higher O-orders can absorb spatial
#    baseline variations.  O(0) gives the most physically interpretable
#    rigid-sub-pupil baseline of 24.9 mm.
#
# 4. **Model comparison in ray space is a diagnostic**: the Zernike-vs-CMO
#    comparison across the FOV reveals that the real optics are more
#    telecentric than the perspective CMO model, explaining why the CMO
#    fit cannot achieve better than ~600 px reprojection.
#
# 5. **The rayfield is a diagnostic instrument for 2‑D corner quality.**
#    Comparing constrained vs. full-pose Zernike fits reveals a gauge
#    drift (Z₀, 90 % of Δd) that is invisible to 2‑D reprojection error.
#    The stability criterion ΔZ₀ < 0.1°, |Δb| < 0.5 mm, |Δθ| < 0.2°
#    tells you whether your corners are clean enough for physically
#    interpretable calibration.  If it fails, the fix is to improve
#    Ray2D preprocessing — not to constrain poses.
#
# 6. **Double TPS eliminates the gauge ambiguity.**  The second TPS pass
#    on the completed 165‑corner set reduces Z₀ drift from 8.5° to
#    0.023°, making constrained and full-pose rayfields nearly identical.
#    The sweep (section 10.3) confirms that regularization adds nothing
#    when corners are clean — the Pareto curve is vertical.
#
# 7. **The Ray2D → Ray3D feedback loop is a general strategy.**  Any
#    stereo calibration pipeline can use it: measure the rayfield,
#    compare pose models, diagnose corner quality, fix preprocessing,
#    verify.  This turns an apparent weakness (gauge instability) into a
#    systematic quality-control tool.
#
# 8. **The workflow generalises**: the same rayfield → physical reading →
#    model comparison sequence can be applied to any stereo microscope to
#    identify its optical architecture and quantify deviations from ideal
#    models.
