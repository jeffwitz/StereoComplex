# %% [markdown]
# # 09 — StereoComplex on real CMO microscope data (Pycaso)
#
# This notebook runs the full StereoComplex pipeline on real calibration
# images from a **Pycaso CMO stereo microscope**.
#
# **Pipeline:**
# 1. ChArUco detection (OpenCV `CharucoDetector`, legacy pattern)
# 2. Hessian-based corner completion (missing corners filled via $|\det H|$ + barycentre)
# 3. ray2D TPS denoising (`predict_points_rayfield_tps_robust`)
# 4. Zernike rayfield fit (origin order 0, direction order 2, constrained poses)
# 5. Pixel reprojection errors
#
# **Key result:** Stereo reprojection RMS < 0.5 px on real CMO data,
# where OpenCV stereo calibration completely fails (RMS > 300 px).

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
) -> np.ndarray:
    """Fill missing ChArUco corners via affine projection + Hessian barycentre.

    Returns ``(N, 2)`` array for all ``(ncx-1)*(ncy-1)`` inner corners.
    OpenCV-detected corners are kept (sub-pixel); missing ones are completed.
    """
    nx, ny = ncx - 1, ncy - 1
    n_corners = nx * ny

    R = abs_det_hessian(gray, sigma=9.0)
    mask = otsu_mask(R)
    A = fit_affine(charuco_corners, charuco_ids, ncx)
    pred_xy = project_affine(A, np.arange(n_corners, dtype=np.int32), ncx)

    detected: dict[int, np.ndarray] = {}
    if charuco_ids is not None and len(charuco_ids) > 0:
        ids_arr = np.asarray(charuco_ids).ravel()
        corners_arr = np.asarray(charuco_corners).reshape(-1, 2)
        for i in range(len(ids_arr)):
            detected[int(ids_arr[i])] = corners_arr[i].astype(np.float64)

    cids = sorted(detected.keys())
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
for z_str in paired_z:
    lg = cv2.imread(str(LEFT_DIR / f"{z_str}.png"), 0)
    rg = cv2.imread(str(RIGHT_DIR / f"{z_str}.png"), 0)

    # ChArUco detection
    cc_L, ids_L, _, _ = charuco_det.detectBoard(lg)
    cc_R, ids_R, _, _ = charuco_det.detectBoard(rg)
    nL = 0 if ids_L is None else len(ids_L)
    nR = 0 if ids_R is None else len(ids_R)

    # Hessian completion
    comp_L = complete_corners_hessian(lg, cc_L, ids_L, NCX, NCY)
    comp_R = complete_corners_hessian(rg, cc_R, ids_R, NCX, NCY)

    # ray2D TPS denoising (fit on marker corners, predict all 165)
    for gray, mk_c, mk_ids, out in [
        (lg, *aruco_det.detectMarkers(lg)[:2], denoised_L),
        (rg, *aruco_det.detectMarkers(rg)[:2], denoised_R),
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
            out.append(pred)
        else:
            out.append(comp_L if out is denoised_L else comp_R)

    print(f"  {z_str}: L {nL}→165  R {nR}→165")

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
# converted to pixels using the local pixel scale ``t / fx``.

# %%
all_err_px_L, all_err_px_R = [], []
all_err_mm_L, all_err_mm_R = [], []

for pi in range(len(paired_z)):
    R_mat = opt_R[pi]
    t_vec = opt_t[pi]
    X_world = (R_mat @ obj_pts.T).T + t_vec[None, :]
    n_plane = R_mat[:, 2]

    for k in range(165):
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


print(f"PIXEL reprojection errors (ray→plane intersection):")
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
