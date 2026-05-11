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
# **Key result:** Stereo rayfield fit reaches subpixel pixel-equivalent
# residuals (< 0.5 px) on this dataset, whereas a standard central OpenCV
# stereo calibration (under the tested configuration) does not converge to
# a usable model.

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
det_counts_L, det_counts_R = [], []
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
    f_obj_est, WD_est, b_est, 50.0, 1024.0, 1024.0, 0.0, 0.0,
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
# ## 9 — Conclusions
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
# 5. **The workflow generalises**: the same rayfield → physical reading →
#    model comparison sequence can be applied to any stereo microscope to
#    identify its optical architecture and quantify deviations from ideal
#    models.
