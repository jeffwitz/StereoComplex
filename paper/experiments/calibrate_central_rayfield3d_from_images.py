from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stereocomplex.core.geometry import triangulate_midpoint
from stereocomplex.core.pinhole_fit import (
    BrownPinholeParams,
    distortion_displacement_metrics,
    fit_brown_pinhole_from_camera_points,
    project_brown_pinhole,
    project_brown_pinhole_with_rvec,
)
from stereocomplex.ray3d.central_ba import FrameObservations
from stereocomplex.ray3d.central_stereo_ba import StereoFrameObservations, fit_central_stereo_rayfield_ba


Side = Literal["left", "right"]


@dataclass(frozen=True)
class ViewDetections:
    marker_ids: np.ndarray  # (M,)
    marker_corners: list[np.ndarray]  # list of (4,2)
    charuco_ids: np.ndarray  # (K,)
    charuco_xy: np.ndarray  # (K,2) in dataset pixel-center convention


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"rms": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "rms": _rms(x),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def _rel_percent_abs(est: float, gt: float, *, eps: float = 1e-12) -> float:
    if not np.isfinite(gt) or abs(gt) <= eps:
        return float("nan")
    return float(100.0 * abs(est - gt) / abs(gt))


def _percent_vs_gt_K(K_est: np.ndarray, K_gt: np.ndarray) -> dict[str, float]:
    K_est = np.asarray(K_est, dtype=np.float64).reshape(3, 3)
    K_gt = np.asarray(K_gt, dtype=np.float64).reshape(3, 3)
    return {
        "fx": _rel_percent_abs(float(K_est[0, 0]), float(K_gt[0, 0])),
        "fy": _rel_percent_abs(float(K_est[1, 1]), float(K_gt[1, 1])),
        "cx": _rel_percent_abs(float(K_est[0, 2]), float(K_gt[0, 2])),
        "cy": _rel_percent_abs(float(K_est[1, 2]), float(K_gt[1, 2])),
    }


def _baseline_direction_metrics(C_R_in_L_mm: np.ndarray) -> dict[str, float]:
    """
    Direction-only metrics vs an x-axis baseline (GT convention).
    """
    C = np.asarray(C_R_in_L_mm, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(C))
    if not np.isfinite(n) or n < 1e-12:
        return {"angle_deg": float("nan"), "offaxis_mm": float("nan")}
    offaxis = float(np.linalg.norm(C[1:]))
    cosang = float(np.clip(C[0] / n, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    return {"angle_deg": angle_deg, "offaxis_mm": offaxis}


def _umeyama_similarity(X_src: np.ndarray, X_dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Similarity transform aligning X_src to X_dst (Umeyama, least squares).
    Returns (s, R, t) such that s*R*X_src + t approximates X_dst.
    """
    X_src = np.asarray(X_src, dtype=np.float64).reshape(-1, 3)
    X_dst = np.asarray(X_dst, dtype=np.float64).reshape(-1, 3)
    if X_src.shape != X_dst.shape or X_src.shape[0] < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)
    mu_x = np.mean(X_src, axis=0)
    mu_y = np.mean(X_dst, axis=0)
    Xc = X_src - mu_x
    Yc = X_dst - mu_y
    cov = (Yc.T @ Xc) / float(X_src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    var_x = float(np.mean(np.sum(Xc * Xc, axis=1)))
    s = float(np.sum(S) / (var_x + 1e-12))
    t = mu_y - s * (Rm @ mu_x)
    return s, Rm.astype(np.float64), t.astype(np.float64)


def _similarity_fixed_origin(X_src: np.ndarray, X_dst: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Similarity alignment with translation fixed to 0 (camera center at origin).
    Returns (s, R) such that s*R*X_src approximates X_dst.
    """
    X_src = np.asarray(X_src, dtype=np.float64).reshape(-1, 3)
    X_dst = np.asarray(X_dst, dtype=np.float64).reshape(-1, 3)
    if X_src.shape != X_dst.shape or X_src.shape[0] < 3:
        return 1.0, np.eye(3, dtype=np.float64)
    cov = (X_dst.T @ X_src) / float(X_src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    var_x = float(np.mean(np.sum(X_src * X_src, axis=1)))
    s = float(np.sum(S) / (var_x + 1e-12))
    return s, Rm.astype(np.float64)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def build_charuco_from_meta(meta: dict[str, Any]):
    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    dict_name = str(board_meta["aruco_dictionary"])
    dictionary = getattr(aruco, dict_name)
    dictionary = aruco.getPredefinedDictionary(dictionary)

    squares_x = int(board_meta["squares_x"])
    squares_y = int(board_meta["squares_y"])
    square_size = float(board_meta["square_size_mm"])
    marker_size = float(board_meta["marker_size_mm"])

    board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    detector_params = aruco.DetectorParameters()

    aruco_detector = None
    charuco_detector = None
    if hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(board, aruco.CharucoParameters(), detector_params)
    return cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector


def detect_view(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img: np.ndarray,
) -> ViewDetections | None:
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return None
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
            return None
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
        return ViewDetections(
            marker_ids=np.asarray(marker_ids, dtype=np.int32).reshape(-1),
            marker_corners=[np.asarray(c, dtype=np.float64).reshape(4, 2) for c in marker_corners],
            charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
            charuco_xy=charuco_xy,
        )

    if aruco_detector is not None:
        corners, ids, _rejected = aruco_detector.detectMarkers(img)
    else:  # pragma: no cover
        corners, ids, _rejected = aruco.detectMarkers(img, dictionary, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return None
    ret = aruco.interpolateCornersCharuco(corners, ids, img, board)
    if ret is None:
        return None
    charuco_corners, charuco_ids, _ = ret
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None
    charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
    return ViewDetections(
        marker_ids=np.asarray(ids, dtype=np.int32).reshape(-1),
        marker_corners=[np.asarray(c, dtype=np.float64).reshape(4, 2) for c in corners],
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return {int(i): xy[k].astype(np.float64) for k, i in enumerate(ids.tolist())}


def _project_brown(
    *,
    view_meta: dict[str, Any],
    f_um: float,
    brown: dict[str, Any],
    XYZ_cam_mm: np.ndarray,
) -> np.ndarray:
    from stereocomplex.core.distortion import brown_from_dict  # noqa: PLC0415
    from stereocomplex.core.geometry import sensor_um_to_pixel  # noqa: PLC0415
    from stereocomplex.meta import parse_view_meta  # noqa: PLC0415

    view = parse_view_meta(view_meta)
    XYZ_cam_mm = np.asarray(XYZ_cam_mm, dtype=np.float64)
    X = XYZ_cam_mm[:, 0]
    Y = XYZ_cam_mm[:, 1]
    Z = XYZ_cam_mm[:, 2]

    uv = np.full((XYZ_cam_mm.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(Z) & (np.abs(Z) > 1e-12)
    if not np.any(good):
        return uv

    x = X[good] / Z[good]
    y = Y[good] / Z[good]
    dist = brown_from_dict(brown)
    xd, yd = dist.distort(x, y)
    x_um = xd * float(f_um)
    y_um = yd * float(f_um)
    u_px, v_px = sensor_um_to_pixel(view, x_um, y_um)
    uv[good, 0] = u_px
    uv[good, 1] = v_px
    return uv


def _pinhole_rays_from_pixels(
    *,
    view_meta: dict[str, Any],
    f_um: float,
    brown: dict[str, Any],
    uv_px: np.ndarray,
) -> np.ndarray:
    from stereocomplex.core.distortion import brown_from_dict  # noqa: PLC0415
    from stereocomplex.core.geometry import PinholeCamera, pixel_to_sensor_um  # noqa: PLC0415
    from stereocomplex.meta import parse_view_meta  # noqa: PLC0415

    view = parse_view_meta(view_meta)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    x_um, y_um = pixel_to_sensor_um(view, uv_px[:, 0], uv_px[:, 1])

    xd = x_um / float(f_um)
    yd = y_um / float(f_um)

    dist = brown_from_dict(brown)
    x, y = dist.undistort(xd, yd, iterations=12)
    return PinholeCamera(f_um=float(f_um)).ray_directions_cam_from_norm(x, y)


def _rig_from_poses(
    common_fids: list[int],
    rvecs_L: dict[int, np.ndarray],
    tvecs_L: dict[int, np.ndarray],
    rvecs_R: dict[int, np.ndarray],
    tvecs_R: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a single rig transform (R_RL, t_RL) from per-frame board poses.

    Convention: P_R = R_RL P_L + t_RL (t_RL in right coordinates).
    Returns (R_RL, t_RL, C_R_in_L) where C_R_in_L is right camera center in left frame.
    """
    from scipy.spatial.transform import Rotation as R  # type: ignore

    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for fid in common_fids:
        R_L = R.from_rotvec(rvecs_L[fid]).as_matrix()
        R_R = R.from_rotvec(rvecs_R[fid]).as_matrix()
        t_L = tvecs_L[fid].reshape(3)
        t_R = tvecs_R[fid].reshape(3)

        R_RL = R_R @ R_L.T
        t_RL = t_R - R_RL @ t_L
        Rs.append(R_RL)
        ts.append(t_RL)

    rot_mean = R.from_matrix(np.stack(Rs, axis=0)).mean().as_matrix()
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)

    C_R_in_L = -rot_mean.T @ t_mean
    return rot_mean, t_mean, C_R_in_L


def _opencv_calibrate_pinhole(
    cv2,
    *,
    image_size: tuple[int, int],
    obj_by_frame: dict[int, np.ndarray],
    uv_by_frame: dict[int, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    obj_pts_list: list[np.ndarray] = []
    img_pts_list: list[np.ndarray] = []
    for fid in sorted(obj_by_frame):
        obj = np.asarray(obj_by_frame[fid], dtype=np.float32).reshape(-1, 1, 3)
        uv = np.asarray(uv_by_frame[fid], dtype=np.float32).reshape(-1, 1, 2)
        if obj.shape[0] < 6:
            continue
        obj_pts_list.append(obj)
        img_pts_list.append(uv)
    if not obj_pts_list:
        raise RuntimeError("no calibration frames")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts_list,
        img_pts_list,
        image_size,
        None,
        None,
        flags=0,
        criteria=criteria,
    )
    return float(rms), np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1)


def _opencv_stereo_calibrate_fix_intrinsics(
    cv2,
    *,
    image_size: tuple[int, int],
    obj_by_frame: dict[int, np.ndarray],
    uvL_by_frame: dict[int, np.ndarray],
    uvR_by_frame: dict[int, np.ndarray],
    K1: np.ndarray,
    d1: np.ndarray,
    K2: np.ndarray,
    d2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    obj_pts_list: list[np.ndarray] = []
    imgL_list: list[np.ndarray] = []
    imgR_list: list[np.ndarray] = []
    for fid in sorted(obj_by_frame):
        obj = np.asarray(obj_by_frame[fid], dtype=np.float32).reshape(-1, 1, 3)
        uvL = np.asarray(uvL_by_frame[fid], dtype=np.float32).reshape(-1, 1, 2)
        uvR = np.asarray(uvR_by_frame[fid], dtype=np.float32).reshape(-1, 1, 2)
        if obj.shape[0] < 6:
            continue
        if uvL.shape[0] != obj.shape[0] or uvR.shape[0] != obj.shape[0]:
            continue
        obj_pts_list.append(obj)
        imgL_list.append(uvL)
        imgR_list.append(uvR)
    if not obj_pts_list:
        raise RuntimeError("no stereo frames")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _K1, _d1, _K2, _d2, R, T, _E, _F = cv2.stereoCalibrate(
        obj_pts_list,
        imgL_list,
        imgR_list,
        K1,
        d1,
        K2,
        d2,
        image_size,
        criteria=criteria,
        flags=flags,
    )
    return float(rms), np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64).reshape(3)


def _undistort_norm_points(cv2, uv_px: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 1, 2)
    pts = cv2.undistortPoints(uv_px, K, d)  # (N,1,2), normalized
    return pts.reshape(-1, 2).astype(np.float64)


def _init_pose_from_homography(
    cv2,
    *,
    obj_xy_mm: np.ndarray,
    uv_px: np.ndarray,
    K0: np.ndarray,
    ransac_thresh_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Initialize (rvec,tvec) from a planar homography using an estimated gauge K0.

    This is *not* solvePnP: we only use the plane-induced homography H and a K0
    obtained from homography self-calibration (Zhang-style). K0 is used as a gauge
    to map pixels to normalized coordinates for the decomposition.
    """
    obj_xy_mm = np.asarray(obj_xy_mm, dtype=np.float64).reshape(-1, 2)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    if obj_xy_mm.shape[0] < 6:
        return None

    H, _mask = cv2.findHomography(obj_xy_mm, uv_px, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_thresh_px))
    if H is None:
        return None

    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K0)):
        return None
    if abs(float(K0[2, 2]) - 1.0) > 1e-6:
        K0 = K0 / float(K0[2, 2])
    if float(K0[2, 2]) == 0.0:
        return None

    # Rotation/translation from the classic decomposition (K0 only as gauge).
    Hn = np.linalg.inv(K0) @ H
    h1 = Hn[:, 0]
    h2 = Hn[:, 1]
    h3 = Hn[:, 2]
    s1 = np.linalg.norm(h1)
    s2 = np.linalg.norm(h2)
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 < 1e-12 or s2 < 1e-12:
        return None
    s = 1.0 / (0.5 * (s1 + s2))
    r1 = s * h1
    r2 = s * h2
    r3 = np.cross(r1, r2)
    R0 = np.stack([r1, r2, r3], axis=1)
    # Orthonormalize.
    U, _S, Vt = np.linalg.svd(R0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt

    t = (s * h3).reshape(3).astype(np.float64)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
        # Sign ambiguity: flip if needed.
        t = (-t).reshape(3)
        if t[2] <= 0:
            return None

    rvec, _ = cv2.Rodrigues(Rm)
    return rvec.reshape(3).astype(np.float64), t.reshape(3).astype(np.float64)


def _init_coeffs_pinhole_prior(
    *,
    uv_all: np.ndarray,
    nmax: int,
    image_size: tuple[int, int],
    f0_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize Zernike coefficients to approximate a simple pinhole prior:
      x ~= (u-cx)/f0,  y ~= (v-cy)/f0
    """
    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    u0, v0, radius = default_disk(w, h)

    uv_all = np.asarray(uv_all, dtype=np.float64).reshape(-1, 2)
    A, mask, _modes = zernike_design_matrix(
        uv_all[:, 0], uv_all[:, 1], nmax=int(nmax), u0_px=float(u0), v0_px=float(v0), radius_px=float(radius)
    )
    if not np.all(mask):
        A = A[mask]
        uv_all = uv_all[mask]

    x_t = (uv_all[:, 0] - cx) / float(f0_px)
    y_t = (uv_all[:, 1] - cy) / float(f0_px)

    lam = 1e-9
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x_t)
    ay = np.linalg.solve(ATA, A.T @ y_t)
    return ax.astype(np.float64), ay.astype(np.float64)


def _estimate_K0_from_homographies(
    *,
    homographies: list[np.ndarray],
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Zhang-style self-calibration from plane homographies.

    Returns a 3x3 K0 in pixels. If the estimate is unstable, falls back to a generic K0.
    """
    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    K_fallback = np.array([[1.5 * float(max(w, h)), 0.0, cx], [0.0, 1.5 * float(max(w, h)), cy], [0.0, 0.0, 1.0]])

    Hs: list[np.ndarray] = []
    for H in homographies:
        H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(H)):
            continue
        if abs(float(H[2, 2])) < 1e-12:
            continue
        Hs.append(H / float(H[2, 2]))
    if len(Hs) < 3:
        return K_fallback.astype(np.float64)

    # Prefer a constrained estimate (square pixels, principal point at center, zero skew).
    # This is much more stable when H are only approximate (distortion, noise).
    f_min = 0.5 * float(max(w, h))
    f_max = 3.0 * float(max(w, h))
    fs = np.logspace(np.log10(f_min), np.log10(f_max), num=80, dtype=np.float64)

    def cost_for_f(f: float) -> float:
        Kinv = np.array([[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]], dtype=np.float64)
        c = 0.0
        for H in Hs:
            Hn = Kinv @ H
            h1 = Hn[:, 0]
            h2 = Hn[:, 1]
            n1 = float(np.linalg.norm(h1))
            n2 = float(np.linalg.norm(h2))
            if not np.isfinite(n1) or not np.isfinite(n2) or n1 < 1e-12 or n2 < 1e-12:
                continue
            dot = float(np.dot(h1, h2)) / (n1 * n2)
            ratio = (n1 / n2) - 1.0
            c += dot * dot + ratio * ratio
        return c

    costs = np.array([cost_for_f(float(f)) for f in fs], dtype=np.float64)
    if np.all(np.isfinite(costs)):
        f0 = float(fs[int(np.argmin(costs))])
        if np.isfinite(f0) and f0 > 1.0:
            return np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Fallback to full Zhang self-calibration (less constrained).
    def v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
        h_i = H[:, i].reshape(3)
        h_j = H[:, j].reshape(3)
        return np.array(
            [
                h_i[0] * h_j[0],
                h_i[0] * h_j[1] + h_i[1] * h_j[0],
                h_i[1] * h_j[1],
                h_i[2] * h_j[0] + h_i[0] * h_j[2],
                h_i[2] * h_j[1] + h_i[1] * h_j[2],
                h_i[2] * h_j[2],
            ],
            dtype=np.float64,
        )

    V_rows: list[np.ndarray] = []
    for H in Hs:
        V_rows.append(v_ij(H, 0, 1))
        V_rows.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    V = np.stack(V_rows, axis=0)
    # Solve V b = 0 via SVD.
    _U, _S, Vt = np.linalg.svd(V)
    b = Vt[-1, :].reshape(6)
    if not np.all(np.isfinite(b)):
        return K_fallback.astype(np.float64)

    b11, b12, b22, b13, b23, b33 = (float(x) for x in b.tolist())
    denom = b11 * b22 - b12 * b12
    if not np.isfinite(denom) or abs(denom) < 1e-18:
        return K_fallback.astype(np.float64)

    v0 = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11
    if not np.isfinite(lam) or lam <= 0:
        return K_fallback.astype(np.float64)

    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / denom)
    gamma = -b12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha * alpha / lam

    if not all(np.isfinite(x) for x in [alpha, beta, gamma, u0, v0]):
        return K_fallback.astype(np.float64)
    if alpha <= 1e-6 or beta <= 1e-6:
        return K_fallback.astype(np.float64)

    K0 = np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Soft clamp principal point near image center to avoid extreme solutions with few frames.
    # (Keep it mild: we only want an initialization.)
    K0[0, 2] = float(np.clip(K0[0, 2], cx - 0.5 * w, cx + 0.5 * w))
    K0[1, 2] = float(np.clip(K0[1, 2], cy - 0.5 * h, cy + 0.5 * h))

    return K0


def _init_coeffs_from_pose_guess(
    *,
    frames: dict[int, FrameObservations],
    rvecs0: dict[int, np.ndarray],
    tvecs0: dict[int, np.ndarray],
    nmax: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize Zernike coefficients by bootstrapping (x,y) = (X/Z, Y/Z) from pose guesses.
    """
    import cv2  # type: ignore

    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    u0, v0, radius = default_disk(w, h)

    uv_all: list[np.ndarray] = []
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for fid, fr in frames.items():
        rvec = np.asarray(rvecs0[int(fid)], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs0[int(fid)], dtype=np.float64).reshape(3)
        rot, _ = cv2.Rodrigues(rvec)
        P_cam = (rot @ fr.P_board_mm.T).T + tvec.reshape(1, 3)
        Z = P_cam[:, 2]
        good = np.isfinite(Z) & (np.abs(Z) > 1e-9)
        if not np.any(good):
            continue
        uv_all.append(np.asarray(fr.uv_px, dtype=np.float64)[good])
        x_all.append((P_cam[good, 0] / Z[good]).astype(np.float64))
        y_all.append((P_cam[good, 1] / Z[good]).astype(np.float64))

    if not uv_all:
        # Fallback to a generic pinhole prior around image center.
        f0_px = 1.5 * float(max(w, h))
        return _init_coeffs_pinhole_prior(uv_all=np.zeros((1, 2)), nmax=int(nmax), image_size=image_size, f0_px=f0_px)

    uv = np.concatenate(uv_all, axis=0)
    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    A, mask, _modes = zernike_design_matrix(uv[:, 0], uv[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)
    if not np.all(mask):
        A = A[mask]
        x = x[mask]
        y = y[mask]

    lam = 1e-6
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x)
    ay = np.linalg.solve(ATA, A.T @ y)
    return ax.astype(np.float64), ay.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate a central ray-field (Zernike) from images (multi-poses) using point-to-ray BA."
    )
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all).")
    ap.add_argument("--method2d", default="rayfield_tps_robust", choices=["raw", "rayfield_tps_robust"])
    ap.add_argument(
        "--max-points-per-frame",
        type=int,
        default=0,
        help="Optional cap on the number of ChArUco corners per frame (0=all).",
    )
    ap.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="Seed used for per-frame subsampling when --max-points-per-frame > 0.",
    )

    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)

    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--lam-coeff", type=float, default=1e-3)
    ap.add_argument("--outer-iters", type=int, default=6)
    ap.add_argument("--fscale-mm", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("paper/tables/rayfield3d_ba_from_images.json"))
    ap.add_argument(
        "--export-model",
        type=Path,
        default=None,
        help="Optional directory to export a reusable stereo model (model.json + weights.npz).",
    )
    args = ap.parse_args()

    scene_dir = Path(args.dataset_root) / str(args.split) / str(args.scene)
    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(scene_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)

    # 2D ray-field predictor.
    from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust  # noqa: PLC0415

    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}
    chess3 = np.asarray(board.getChessboardCorners(), dtype=np.float64)  # (Nc,3)
    chess2 = chess3[:, :2]

    sim = meta.get("sim_params", {})
    baseline_gt = float(sim.get("baseline_mm", float("nan")))
    f_um = float(sim.get("f_um", float("nan")))
    dist_model = str(sim.get("distortion_model", "none"))
    dist_L = sim.get("distortion_left", {}) if dist_model == "brown" else {}
    dist_R = sim.get("distortion_right", {}) if dist_model == "brown" else {}

    # GT for evaluation.
    gt = np.load(str(scene_dir / "gt_charuco_corners.npz"))
    gt_frame_id = gt["frame_id"].astype(np.int32).reshape(-1)
    gt_corner_id = gt["corner_id"].astype(np.int32).reshape(-1)
    gt_xyz_L = gt["XYZ_world_mm"].astype(np.float64).reshape(-1, 3)
    gt_uv_L = gt["uv_left_px"].astype(np.float64).reshape(-1, 2)
    gt_uv_R = gt["uv_right_px"].astype(np.float64).reshape(-1, 2)
    gt_by_frame: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for fid in np.unique(gt_frame_id).tolist():
        mask = gt_frame_id == int(fid)
        ids = gt_corner_id[mask].tolist()
        xyz = gt_xyz_L[mask]
        uvL = gt_uv_L[mask]
        uvR = gt_uv_R[mask]
        gt_by_frame[int(fid)] = {
            int(i): {"XYZ_L": xyz[k], "uvL": uvL[k], "uvR": uvR[k]} for k, i in enumerate(ids)
        }

    def rayfield2d_predict(marker_ids: np.ndarray, marker_corners: list[np.ndarray], target_ids: np.ndarray):
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None or mc.shape != (4, 2) or o.shape != (4, 2):
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return None
        obj_xy = np.concatenate(obj_pts, axis=0)
        img_uv = np.concatenate(img_pts, axis=0)

        target_ids = np.asarray(target_ids, dtype=np.int32).reshape(-1)
        if target_ids.size == 0:
            return {}
        target_xy = chess2[target_ids]
        pred = predict_points_rayfield_tps_robust(
            obj_xy,
            img_uv,
            target_xy,
            lam=float(args.tps_lam),
            huber_c=float(args.tps_huber),
            iters=int(args.tps_iters),
        )
        return _dict_from_ids_xy(target_ids, pred)

    # Collect per-frame observations for each side.
    obs_by_side: dict[Side, dict[int, FrameObservations]] = {"left": {}, "right": {}}
    ids_by_frame: dict[int, list[int]] = {}

    for fr in frames:
        fid = int(fr["frame_id"])
        gt_frame = gt_by_frame.get(fid)
        if gt_frame is None:
            continue

        det_by_side: dict[Side, dict[int, np.ndarray]] = {}
        for side in ("left", "right"):
            img_path = scene_dir / side / str(fr[side])
            from stereocomplex.core.image_io import load_gray_u8  # noqa: PLC0415

            img = load_gray_u8(img_path)
            det = detect_view(cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
            if det is None:
                continue

            if args.method2d == "raw":
                det_by_side[side] = _dict_from_ids_xy(det.charuco_ids, det.charuco_xy)
            else:
                pred = rayfield2d_predict(det.marker_ids, det.marker_corners, det.charuco_ids)
                if pred is None:
                    continue
                det_by_side[side] = pred

        if "left" not in det_by_side or "right" not in det_by_side:
            continue

        ids_common = sorted(set(det_by_side["left"]).intersection(det_by_side["right"]).intersection(gt_frame))
        if len(ids_common) < 10:
            continue

        if args.max_points_per_frame and int(args.max_points_per_frame) > 0 and len(ids_common) > int(args.max_points_per_frame):
            rng = np.random.default_rng(int(args.subsample_seed) + int(fid))
            take = rng.choice(len(ids_common), size=int(args.max_points_per_frame), replace=False)
            ids_common = [ids_common[int(k)] for k in np.sort(take).tolist()]

        ids_by_frame[fid] = ids_common

        for side in ("left", "right"):
            uv = np.stack([det_by_side[side][int(i)] for i in ids_common], axis=0).astype(np.float64)
            P = chess3[np.asarray(ids_common, dtype=np.int32)].astype(np.float64)
            obs_by_side[side][fid] = FrameObservations(uv_px=uv, P_board_mm=P)

    if not obs_by_side["left"] or not obs_by_side["right"]:
        raise RuntimeError("no usable frames (detections failed)")

    # Pose initialization without solvePnP and without known intrinsics:
    # use plane homographies + Zhang-style self-calibration to obtain a gauge K0,
    # then decompose each homography into (R,t).
    image_size = (int(meta["stereo"]["left"]["image"]["width_px"]), int(meta["stereo"]["left"]["image"]["height_px"]))

    def init_poses(side: Side) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        homographies: list[np.ndarray] = []
        for fr_obs in obs_by_side[side].values():
            obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
            uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
            if obj_xy.shape[0] < 6:
                continue
            H, _mask = cv2.findHomography(obj_xy, uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is None:
                continue
            homographies.append(np.asarray(H, dtype=np.float64))
        K0 = _estimate_K0_from_homographies(homographies=homographies, image_size=image_size)

        rvecs0: dict[int, np.ndarray] = {}
        tvecs0: dict[int, np.ndarray] = {}
        for fid, fr_obs in obs_by_side[side].items():
            obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
            uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
            init = _init_pose_from_homography(cv2, obj_xy_mm=obj_xy, uv_px=uv, K0=K0)
            if init is None:
                continue
            rvecs0[fid], tvecs0[fid] = init
        return rvecs0, tvecs0

    rvecs0_L, tvecs0_L = init_poses("left")
    rvecs0_R, tvecs0_R = init_poses("right")

    # Keep only frames with initialized poses.
    common = sorted(set(rvecs0_L).intersection(rvecs0_R).intersection(obs_by_side["left"]).intersection(obs_by_side["right"]))
    obs_L = {fid: obs_by_side["left"][fid] for fid in common}
    obs_R = {fid: obs_by_side["right"][fid] for fid in common}
    rvecs0_L = {fid: rvecs0_L[fid] for fid in common}
    tvecs0_L = {fid: tvecs0_L[fid] for fid in common}
    rvecs0_R = {fid: rvecs0_R[fid] for fid in common}
    tvecs0_R = {fid: tvecs0_R[fid] for fid in common}

    if len(common) < 2:
        raise RuntimeError("not enough frames with initialized poses")

    w = int(meta["stereo"]["left"]["image"]["width_px"])
    h = int(meta["stereo"]["left"]["image"]["height_px"])

    # Ray-field coefficient initialization: bootstrap (x,y) from pose guesses (no pinhole K needed).
    coeffs0_x_L, coeffs0_y_L = _init_coeffs_from_pose_guess(
        frames=obs_L,
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        nmax=int(args.nmax),
        image_size=image_size,
    )
    coeffs0_x_R, coeffs0_y_R = _init_coeffs_from_pose_guess(
        frames=obs_R,
        rvecs0=rvecs0_R,
        tvecs0=tvecs0_R,
        nmax=int(args.nmax),
        image_size=image_size,
    )

    # Joint stereo BA (shared rig, board poses in left camera coordinates).
    stereo_frames = {
        fid: StereoFrameObservations(
            uv_left_px=obs_L[fid].uv_px,
            uv_right_px=obs_R[fid].uv_px,
            P_board_mm=obs_L[fid].P_board_mm,
        )
        for fid in common
    }
    # Initial rig from homography-decomposed poses.
    R_RL0, t_RL0, _C_R_in_L0 = _rig_from_poses(common, rvecs0_L, tvecs0_L, rvecs0_R, tvecs0_R)
    rig_rvec0, _ = cv2.Rodrigues(R_RL0)
    rig_tvec0 = t_RL0.reshape(3)

    res = fit_central_stereo_rayfield_ba(
        frames=stereo_frames,
        image_width_px=w,
        image_height_px=h,
        nmax=int(args.nmax),
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        rig_rvec0=rig_rvec0.reshape(3),
        rig_tvec0=rig_tvec0.reshape(3),
        coeffs0_left_x=coeffs0_x_L,
        coeffs0_left_y=coeffs0_y_L,
        coeffs0_right_x=coeffs0_x_R,
        coeffs0_right_y=coeffs0_y_R,
        lam_coeff=float(args.lam_coeff),
        lam_center=1e-1,
        lam_jacobian=10.0,
        loss="huber",
        f_scale_mm=float(args.fscale_mm),
        max_nfev=max(800, 200 * int(args.outer_iters)),
    )

    from scipy.spatial.transform import Rotation as R  # type: ignore

    R_RL = R.from_rotvec(res.rig_rvec.reshape(3)).as_matrix()
    t_RL = res.rig_tvec.reshape(3)
    C_R_in_L = -R_RL.T @ t_RL
    baseline_est_mm = float(np.linalg.norm(C_R_in_L))
    baseline_dir = _baseline_direction_metrics(C_R_in_L)

    # OpenCV pinhole calibration from the same detected points (non-GT).
    obj_by_frame = {fid: obs_L[fid].P_board_mm for fid in common}
    uvL_by_frame = {fid: obs_L[fid].uv_px for fid in common}
    uvR_by_frame = {fid: obs_R[fid].uv_px for fid in common}
    mono_rms_L, K_L_est, dL_est = _opencv_calibrate_pinhole(
        cv2, image_size=image_size, obj_by_frame=obj_by_frame, uv_by_frame=uvL_by_frame
    )
    mono_rms_R, K_R_est, dR_est = _opencv_calibrate_pinhole(
        cv2, image_size=image_size, obj_by_frame=obj_by_frame, uv_by_frame=uvR_by_frame
    )
    stereo_rms, R_lr_est, T_lr_est = _opencv_stereo_calibrate_fix_intrinsics(
        cv2,
        image_size=image_size,
        obj_by_frame=obj_by_frame,
        uvL_by_frame=uvL_by_frame,
        uvR_by_frame=uvR_by_frame,
        K1=K_L_est,
        d1=dL_est,
        K2=K_R_est,
        d2=dR_est,
    )
    # OpenCV stereoCalibrate returns X_R = R X_L + T, so C_R(in L) = -R^T T.
    C_R_in_L_pinhole = -R_lr_est.T @ T_lr_est.reshape(3)
    baseline_est_pinhole_mm = float(np.linalg.norm(C_R_in_L_pinhole))
    baseline_dir_pinhole = _baseline_direction_metrics(C_R_in_L_pinhole)

    # Evaluate triangulation on per-frame correspondences (GT ids for that frame).
    err3d_pin: list[float] = []
    err3d_rf: list[float] = []
    skew_pin: list[float] = []
    skew_rf: list[float] = []
    repr_L_pin: list[float] = []
    repr_R_pin: list[float] = []
    repr_L_rf: list[float] = []
    repr_R_rf: list[float] = []
    err3d_pinhole_cal: list[float] = []
    skew_pinhole_cal: list[float] = []
    repr_L_pinhole_cal: list[float] = []
    repr_R_pinhole_cal: list[float] = []
    XYZ_hat_rf_all: list[np.ndarray] = []
    uv_obs_L_all: list[np.ndarray] = []
    uv_obs_R_all: list[np.ndarray] = []
    XYZ_gt_all: list[np.ndarray] = []
    uv_gt_L_all: list[np.ndarray] = []
    uv_gt_R_all: list[np.ndarray] = []

    C_L = np.zeros((3,), dtype=np.float64)
    dR_rot_to_L = R_RL.T  # maps right vectors to left coordinates
    C_R_in_L_gt = np.array([baseline_gt, 0.0, 0.0], dtype=np.float64) if np.isfinite(baseline_gt) else None
    dR_cal_rot_to_L = R_lr_est.T
    T_L_to_R_gt = np.array([-baseline_gt, 0.0, 0.0], dtype=np.float64).reshape(1, 3) if np.isfinite(baseline_gt) else None

    for fid in common:
        frL = obs_L[fid]
        frR = obs_R[fid]
        ids_common = ids_by_frame[fid]
        if len(ids_common) != frL.uv_px.shape[0] or len(ids_common) != frR.uv_px.shape[0]:
            raise RuntimeError("internal error: ids/observations mismatch")

        # Pinhole oracle rays from observed pixels.
        dL_pin = _pinhole_rays_from_pixels(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, uv_px=frL.uv_px)
        dR_pin = _pinhole_rays_from_pixels(view_meta=meta["stereo"]["right"], f_um=f_um, brown=dist_R, uv_px=frR.uv_px)
        if C_R_in_L_gt is None:
            raise RuntimeError("baseline_gt is not finite; cannot evaluate pinhole oracle")
        XYZ_hat_pin, skew_p = triangulate_midpoint(C_L, dL_pin, C_R_in_L_gt, dR_pin)

        # Ray-field 3D rays from learned coefficients.
        from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike  # noqa: PLC0415

        rfL = CentralRayFieldZernike(
            nmax=res.nmax,
            u0_px=res.u0_px,
            v0_px=res.v0_px,
            radius_px=res.radius_px,
            coeffs_x=res.coeffs_left_x,
            coeffs_y=res.coeffs_left_y,
            modes=tuple(),  # unused by ray_directions_cam
            C_mm=C_L,
        )
        rfR = CentralRayFieldZernike(
            nmax=res.nmax,
            u0_px=res.u0_px,
            v0_px=res.v0_px,
            radius_px=res.radius_px,
            coeffs_x=res.coeffs_right_x,
            coeffs_y=res.coeffs_right_y,
            modes=tuple(),
            C_mm=C_L,
        )
        dL_rf = rfL.ray_directions_cam(frL.uv_px[:, 0], frL.uv_px[:, 1])
        dR_rf = rfR.ray_directions_cam(frR.uv_px[:, 0], frR.uv_px[:, 1])
        dR_rf_L = (dR_rot_to_L @ dR_rf.T).T
        XYZ_hat_rf, skew_r = triangulate_midpoint(C_L, dL_rf, C_R_in_L, dR_rf_L)

        # Compare to GT points for this frame.
        gt_frame = gt_by_frame.get(fid, {})
        XYZ_gt = np.stack([gt_frame[i]["XYZ_L"] for i in ids_common], axis=0).astype(np.float64)
        uv_gt_L = np.stack([gt_frame[i]["uvL"] for i in ids_common], axis=0).astype(np.float64)
        uv_gt_R = np.stack([gt_frame[i]["uvR"] for i in ids_common], axis=0).astype(np.float64)

        err3d_pin.extend(np.linalg.norm(XYZ_hat_pin - XYZ_gt, axis=-1).tolist())
        err3d_rf.extend(np.linalg.norm(XYZ_hat_rf - XYZ_gt, axis=-1).tolist())
        XYZ_hat_rf_all.append(np.asarray(XYZ_hat_rf, dtype=np.float64))
        uv_obs_L_all.append(np.asarray(frL.uv_px, dtype=np.float64))
        uv_obs_R_all.append(np.asarray(frR.uv_px, dtype=np.float64))
        XYZ_gt_all.append(np.asarray(XYZ_gt, dtype=np.float64))
        uv_gt_L_all.append(np.asarray(uv_gt_L, dtype=np.float64))
        uv_gt_R_all.append(np.asarray(uv_gt_R, dtype=np.float64))
        skew_pin.extend(np.asarray(skew_p, dtype=np.float64).tolist())
        skew_rf.extend(np.asarray(skew_r, dtype=np.float64).tolist())

        uv_hat_L_pin = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=XYZ_hat_pin)
        uv_hat_R_pin = _project_brown(
            view_meta=meta["stereo"]["right"],
            f_um=f_um,
            brown=dist_R,
            XYZ_cam_mm=XYZ_hat_pin + (T_L_to_R_gt if T_L_to_R_gt is not None else 0.0),
        )
        uv_hat_L_rf = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=XYZ_hat_rf)
        uv_hat_R_rf = _project_brown(
            view_meta=meta["stereo"]["right"],
            f_um=f_um,
            brown=dist_R,
            XYZ_cam_mm=XYZ_hat_rf + (T_L_to_R_gt if T_L_to_R_gt is not None else 0.0),
        )
        repr_L_pin.extend(np.linalg.norm(uv_hat_L_pin - uv_gt_L, axis=-1).tolist())
        repr_R_pin.extend(np.linalg.norm(uv_hat_R_pin - uv_gt_R, axis=-1).tolist())
        repr_L_rf.extend(np.linalg.norm(uv_hat_L_rf - uv_gt_L, axis=-1).tolist())
        repr_R_rf.extend(np.linalg.norm(uv_hat_R_rf - uv_gt_R, axis=-1).tolist())

        # Pinhole calibrated from the same observed pixels.
        xL = _undistort_norm_points(cv2, frL.uv_px, K_L_est, dL_est)
        xR = _undistort_norm_points(cv2, frR.uv_px, K_R_est, dR_est)
        dL = np.concatenate([xL, np.ones((xL.shape[0], 1), dtype=np.float64)], axis=1)
        dR = np.concatenate([xR, np.ones((xR.shape[0], 1), dtype=np.float64)], axis=1)
        dL /= np.linalg.norm(dL, axis=1, keepdims=True)
        dR /= np.linalg.norm(dR, axis=1, keepdims=True)
        dR_L = (dR_cal_rot_to_L @ dR.T).T
        XYZ_hat_cal, skew_cal = triangulate_midpoint(C_L, dL, C_R_in_L_pinhole, dR_L)
        err3d_pinhole_cal.extend(np.linalg.norm(XYZ_hat_cal - XYZ_gt, axis=-1).tolist())
        skew_pinhole_cal.extend(np.asarray(skew_cal, dtype=np.float64).tolist())

        uv_hat_L_cal = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=XYZ_hat_cal)
        uv_hat_R_cal = _project_brown(
            view_meta=meta["stereo"]["right"],
            f_um=f_um,
            brown=dist_R,
            XYZ_cam_mm=XYZ_hat_cal + (T_L_to_R_gt if T_L_to_R_gt is not None else 0.0),
        )
        repr_L_pinhole_cal.extend(np.linalg.norm(uv_hat_L_cal - uv_gt_L, axis=-1).tolist())
        repr_R_pinhole_cal.extend(np.linalg.norm(uv_hat_R_cal - uv_gt_R, axis=-1).tolist())

    Z_all = gt_xyz_L[:, 2]
    Z_mean = float(np.mean(Z_all))

    # Optional interpretability metrics: similarity-aligned triangulation errors (mm).
    if XYZ_hat_rf_all and XYZ_gt_all:
        Xh = np.concatenate(XYZ_hat_rf_all, axis=0)
        Xg = np.concatenate(XYZ_gt_all, axis=0)
        s_sim, R_sim, t_sim = _umeyama_similarity(Xh, Xg)
        Xh_aligned = (s_sim * (R_sim @ Xh.T)).T + t_sim.reshape(1, 3)
        err3d_rf_aligned = np.linalg.norm(Xh_aligned - Xg, axis=-1)
        sim_info = {"scale": float(s_sim), "R": R_sim.tolist(), "t": t_sim.tolist()}

        s0, R0 = _similarity_fixed_origin(Xh, Xg)
        Xh_aligned0 = (s0 * (R0 @ Xh.T)).T
        err3d_rf_aligned0 = np.linalg.norm(Xh_aligned0 - Xg, axis=-1)
        sim0_info = {"scale": float(s0), "R": R0.tolist()}

        # Reprojection errors after origin-fixed similarity alignment, measured against GT pixels.
        uv_gt_L_cat = np.concatenate(uv_gt_L_all, axis=0)
        uv_gt_R_cat = np.concatenate(uv_gt_R_all, axis=0)
        uv_hat_L_rf0 = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=Xh_aligned0)
        uv_hat_R_rf0 = _project_brown(
            view_meta=meta["stereo"]["right"],
            f_um=f_um,
            brown=dist_R,
            XYZ_cam_mm=Xh_aligned0 + (T_L_to_R_gt if T_L_to_R_gt is not None else 0.0),
        )
        repr_L_rf_aligned0 = np.linalg.norm(uv_hat_L_rf0 - uv_gt_L_cat, axis=-1)
        repr_R_rf_aligned0 = np.linalg.norm(uv_hat_R_rf0 - uv_gt_R_cat, axis=-1)
    else:
        err3d_rf_aligned = np.asarray([], dtype=np.float64)
        sim_info = {"scale": float("nan"), "R": (np.eye(3)).tolist(), "t": [float("nan"), float("nan"), float("nan")]}
        err3d_rf_aligned0 = np.asarray([], dtype=np.float64)
        sim0_info = {"scale": float("nan"), "R": (np.eye(3)).tolist()}
        repr_L_rf_aligned0 = np.asarray([], dtype=np.float64)
        repr_R_rf_aligned0 = np.asarray([], dtype=np.float64)
    # Convert mm baseline error to a disparity-equivalent px scale at mean depth using GT fx.
    from stereocomplex.eval.charuco_detection import _camera_params_from_meta  # noqa: PLC0415

    K_L_gt, _dist_L_gt = _camera_params_from_meta(meta["stereo"]["left"], f_um=f_um, brown=dist_L)
    fx_px = float(K_L_gt[0, 0])
    baseline_abs_error_px = float(abs(baseline_est_mm - baseline_gt) * fx_px / (Z_mean + 1e-12)) if np.isfinite(baseline_gt) else float("nan")
    baseline_abs_error_px_pinhole = (
        float(abs(baseline_est_pinhole_mm - baseline_gt) * float(K_L_est[0, 0]) / (Z_mean + 1e-12))
        if np.isfinite(baseline_gt)
        else float("nan")
    )

    # Fit a Brown pinhole model from the ray-field 3D reconstruction (post-hoc identification).
    # The reconstructed 3D points are in left camera coordinates. For the right camera we map
    # them to right coordinates using the estimated rig: P_R = R_RL P_L + t_RL.
    XYZ_rf_cat = np.concatenate(XYZ_hat_rf_all, axis=0) if XYZ_hat_rf_all else np.zeros((0, 3), dtype=np.float64)
    uvL_cat = np.concatenate(uv_obs_L_all, axis=0) if uv_obs_L_all else np.zeros((0, 2), dtype=np.float64)
    uvR_cat = np.concatenate(uv_obs_R_all, axis=0) if uv_obs_R_all else np.zeros((0, 2), dtype=np.float64)
    if XYZ_rf_cat.shape[0] >= 12:
        init_L = BrownPinholeParams(
            fx=float(K_L_est[0, 0]),
            fy=float(K_L_est[1, 1]),
            cx=float(K_L_est[0, 2]),
            cy=float(K_L_est[1, 2]),
            k1=float(dL_est[0]),
            k2=float(dL_est[1]),
            p1=float(dL_est[2]),
            p2=float(dL_est[3]),
            k3=float(dL_est[4]) if dL_est.size >= 5 else 0.0,
        )
        pinL, pinL_diag = fit_brown_pinhole_from_camera_points(
            XYZ_cam=XYZ_rf_cat,
            uv_px=uvL_cat,
            image_size=image_size,
            init=init_L,
            fit_rotation=True,
            loss="huber",
            f_scale_px=2.0,
            max_nfev=4000,
        )
        XYZ_rf_R = (R_RL @ XYZ_rf_cat.T).T + t_RL.reshape(1, 3)
        init_R = BrownPinholeParams(
            fx=float(K_R_est[0, 0]),
            fy=float(K_R_est[1, 1]),
            cx=float(K_R_est[0, 2]),
            cy=float(K_R_est[1, 2]),
            k1=float(dR_est[0]),
            k2=float(dR_est[1]),
            p1=float(dR_est[2]),
            p2=float(dR_est[3]),
            k3=float(dR_est[4]) if dR_est.size >= 5 else 0.0,
        )
        pinR, pinR_diag = fit_brown_pinhole_from_camera_points(
            XYZ_cam=XYZ_rf_R,
            uv_px=uvR_cat,
            image_size=image_size,
            init=init_R,
            fit_rotation=True,
            loss="huber",
            f_scale_px=2.0,
            max_nfev=4000,
        )
        K_L_from3d = pinL.K()
        dL_from3d = pinL.dist()
        K_R_from3d = pinR.K()
        dR_from3d = pinR.dist()
        rvecL = np.asarray(pinL_diag.get("rvec", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        rvecR = np.asarray(pinR_diag.get("rvec", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        reproj_L_from3d = np.linalg.norm((project_brown_pinhole_with_rvec(pinL, XYZ_rf_cat, rvecL) - uvL_cat), axis=1)
        reproj_R_from3d = np.linalg.norm((project_brown_pinhole_with_rvec(pinR, XYZ_rf_R, rvecR) - uvR_cat), axis=1)
    else:  # pragma: no cover
        K_L_from3d = np.full((3, 3), np.nan, dtype=np.float64)
        dL_from3d = np.full((5,), np.nan, dtype=np.float64)
        K_R_from3d = np.full((3, 3), np.nan, dtype=np.float64)
        dR_from3d = np.full((5,), np.nan, dtype=np.float64)
        pinL_diag = {"opt_cost": float("nan"), "opt_nfev": float("nan"), "opt_success": float("nan")}
        pinR_diag = {"opt_cost": float("nan"), "opt_nfev": float("nan"), "opt_success": float("nan")}
        reproj_L_from3d = np.asarray([], dtype=np.float64)
        reproj_R_from3d = np.asarray([], dtype=np.float64)

    out = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "scene": str(args.scene),
        "method2d": str(args.method2d),
        "tps": {"lam": float(args.tps_lam), "huber_c": float(args.tps_huber), "iters": int(args.tps_iters)},
        "rig": {
            "baseline_gt_mm": baseline_gt,
            "baseline_est_mm": baseline_est_mm,
            "baseline_abs_error_mm": float(abs(baseline_est_mm - baseline_gt)) if np.isfinite(baseline_gt) else float("nan"),
            "baseline_abs_error_px_at_mean_depth": baseline_abs_error_px,
            "C_R_in_L_mm": C_R_in_L.tolist(),
            "baseline_direction": baseline_dir,
        },
        "opencv_pinhole_calib": {
            "mono_rms_left_px": mono_rms_L,
            "mono_rms_right_px": mono_rms_R,
            "stereo_rms_px": stereo_rms,
            "left": {"K": K_L_est.tolist(), "dist": dL_est.tolist()},
            "right": {"K": K_R_est.tolist(), "dist": dR_est.tolist()},
            "rig": {
                "baseline_est_mm": baseline_est_pinhole_mm,
                "baseline_abs_error_mm": float(abs(baseline_est_pinhole_mm - baseline_gt)) if np.isfinite(baseline_gt) else float("nan"),
                "baseline_abs_error_px_at_mean_depth": baseline_abs_error_px_pinhole,
                "C_R_in_L_mm": C_R_in_L_pinhole.tolist(),
                "baseline_direction": baseline_dir_pinhole,
                "R_lr": R_lr_est.tolist(),
                "T_lr": T_lr_est.tolist(),
            },
            "triangulation_error_mm": _stats(np.asarray(err3d_pinhole_cal)),
            "triangulation_error_rel_depth_percent": _stats(100.0 * np.asarray(err3d_pinhole_cal) / (Z_mean + 1e-12)),
            "ray_skew_mm": _stats(np.asarray(skew_pinhole_cal)),
            "reprojection_error_left_px": _stats(np.asarray(repr_L_pinhole_cal)),
            "reprojection_error_right_px": _stats(np.asarray(repr_R_pinhole_cal)),
        },
        "pinhole_from_rayfield3d": {
            "left": {"K": K_L_from3d.tolist(), "dist": dL_from3d.tolist()},
            "right": {"K": K_R_from3d.tolist(), "dist": dR_from3d.tolist()},
            "reprojection_error_left_px": _stats(np.asarray(reproj_L_from3d)),
            "reprojection_error_right_px": _stats(np.asarray(reproj_R_from3d)),
            "opt_diagnostics_left": pinL_diag,
            "opt_diagnostics_right": pinR_diag,
        },
        "depth_mm": {"mean": Z_mean},
        "pinhole_oracle": {
            "triangulation_error_mm": _stats(np.asarray(err3d_pin)),
            "triangulation_error_rel_depth_percent": _stats(100.0 * np.asarray(err3d_pin) / (Z_mean + 1e-12)),
            "ray_skew_mm": _stats(np.asarray(skew_pin)),
            "reprojection_error_left_px": _stats(np.asarray(repr_L_pin)),
            "reprojection_error_right_px": _stats(np.asarray(repr_R_pin)),
        },
        "rayfield3d_ba": {
            "settings": {
                "nmax": int(args.nmax),
                "lam_coeff": float(args.lam_coeff),
                "outer_iters_hint": int(args.outer_iters),
                "f_scale_mm": float(args.fscale_mm),
            },
            "model": {
                "image_size_px": [int(w), int(h)],
                "nmax": int(res.nmax),
                "disk": [float(res.u0_px), float(res.v0_px), float(res.radius_px)],
                "R_RL": R_RL.tolist(),
                "t_RL": t_RL.reshape(3).tolist(),
            },
            "triangulation_error_mm": _stats(np.asarray(err3d_rf)),
            "triangulation_error_mm_aligned_similarity": _stats(np.asarray(err3d_rf_aligned)),
            "triangulation_alignment_similarity": sim_info,
            "triangulation_error_mm_aligned_origin_similarity": _stats(np.asarray(err3d_rf_aligned0)),
            "triangulation_alignment_origin_similarity": sim0_info,
            "triangulation_error_rel_depth_percent": _stats(100.0 * np.asarray(err3d_rf) / (Z_mean + 1e-12)),
            "ray_skew_mm": _stats(np.asarray(skew_rf)),
            "reprojection_error_left_px": _stats(np.asarray(repr_L_rf)),
            "reprojection_error_right_px": _stats(np.asarray(repr_R_rf)),
            "reprojection_error_left_px_aligned_origin_similarity": _stats(np.asarray(repr_L_rf_aligned0)),
            "reprojection_error_right_px_aligned_origin_similarity": _stats(np.asarray(repr_R_rf_aligned0)),
            "diagnostics": res.diagnostics,
        },
    }

    # Parameter-level comparison vs GT (synthetic datasets only).
    if np.isfinite(f_um):
        K_L_gt, dist_L_gt = _camera_params_from_meta(meta["stereo"]["left"], f_um=f_um, brown=dist_L)
        K_R_gt, dist_R_gt = _camera_params_from_meta(meta["stereo"]["right"], f_um=f_um, brown=dist_R)
        out["pinhole_vs_gt"] = {
            "opencv_pinhole_calib": {
                "left": {"K_percent": _percent_vs_gt_K(K_L_est, K_L_gt)},
                "right": {"K_percent": _percent_vs_gt_K(K_R_est, K_R_gt)},
                "distortion_displacement_vs_gt": {
                    "left": distortion_displacement_metrics(
                        K_gt=K_L_gt, dist_gt=dist_L_gt, dist_est=dL_est, image_size=image_size
                    ),
                    "right": distortion_displacement_metrics(
                        K_gt=K_R_gt, dist_gt=dist_R_gt, dist_est=dR_est, image_size=image_size
                    ),
                },
            },
            "pinhole_from_rayfield3d": {
                "left": {"K_percent": _percent_vs_gt_K(K_L_from3d, K_L_gt)},
                "right": {"K_percent": _percent_vs_gt_K(K_R_from3d, K_R_gt)},
                "distortion_displacement_vs_gt": {
                    "left": distortion_displacement_metrics(
                        K_gt=K_L_gt, dist_gt=dist_L_gt, dist_est=dL_from3d, image_size=image_size
                    ),
                    "right": distortion_displacement_metrics(
                        K_gt=K_R_gt, dist_gt=dist_R_gt, dist_est=dR_from3d, image_size=image_size
                    ),
                },
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(out, indent=2))

    if args.export_model is not None:
        from stereocomplex.api.model_io import save_stereo_central_rayfield  # noqa: PLC0415
        from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel  # noqa: PLC0415

        model = StereoCentralRayFieldModel.from_coeffs(
            image_width_px=int(w),
            image_height_px=int(h),
            nmax=int(res.nmax),
            u0_px=float(res.u0_px),
            v0_px=float(res.v0_px),
            radius_px=float(res.radius_px),
            coeffs_left_x=np.asarray(res.coeffs_left_x, dtype=np.float64),
            coeffs_left_y=np.asarray(res.coeffs_left_y, dtype=np.float64),
            coeffs_right_x=np.asarray(res.coeffs_right_x, dtype=np.float64),
            coeffs_right_y=np.asarray(res.coeffs_right_y, dtype=np.float64),
            R_RL=np.asarray(R_RL, dtype=np.float64),
            t_RL=np.asarray(t_RL, dtype=np.float64).reshape(3),
        )
        model_json = save_stereo_central_rayfield(Path(args.export_model), model)
        out["rayfield3d_ba"]["exported_model"] = str(model_json)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {model_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
