from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np

from stereocomplex.api._calibration_charuco import (
    _CharucoRuntime,
    _build_charuco_runtime,
    _detect_refined_stereo_pair,
    build_charuco_board as build_charuco_board,
    detect_charuco_corners as detect_charuco_corners,
)
from stereocomplex.api._calibration_images import (
    _ensure_gray_u8,
    _image_pairs_from_dataset,
    _image_pairs_from_dirs,
    _normalize_image_pairs,
)
from stereocomplex.api._calibration_types import (
    CameraSetup,
    CharucoBoardSpec,
    NCameraCalibrationResult,
    StereoCentralRayFieldFitReport,
    StereoCentralRayFieldFitResult,
    StereoImagePair,
    StereoOpenCVCalibrationReport,
    StereoOpenCVCalibrationResult,
    _OriginFieldDatasetSeed,
    _OriginFieldImageObservations,
    _OriginFieldPinholeSeed,
)
from stereocomplex.api.corner_refinement import RefineMethod
from stereocomplex.api.model_io import save_stereo_central_rayfield
from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel
from stereocomplex.ray3d.central_ba import FrameObservations
from stereocomplex.ray3d.central_stereo_ba import (
    StereoFrameObservations,
    fit_central_stereo_rayfield_ba,
)


def _estimate_K0_from_homographies(
    *, homographies: list[np.ndarray], image_size: tuple[int, int]
) -> np.ndarray:
    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    K_fallback = np.array(
        [[1.5 * float(max(w, h)), 0.0, cx], [0.0, 1.5 * float(max(w, h)), cy], [0.0, 0.0, 1.0]]
    )

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

    f_min = 0.5 * float(max(w, h))
    f_max = 3.0 * float(max(w, h))
    fs = np.logspace(np.log10(f_min), np.log10(f_max), num=80, dtype=np.float64)

    def cost_for_f(f: float) -> float:
        Kinv = np.array(
            [[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]], dtype=np.float64
        )
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
    K0[0, 2] = float(np.clip(K0[0, 2], cx - 0.5 * w, cx + 0.5 * w))
    K0[1, 2] = float(np.clip(K0[1, 2], cy - 0.5 * h, cy + 0.5 * h))
    return K0


def _init_pose_from_homography(
    cv2: Any,
    *,
    obj_xy_mm: np.ndarray,
    uv_px: np.ndarray,
    K0: np.ndarray,
    ransac_thresh_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    obj_xy_mm = np.asarray(obj_xy_mm, dtype=np.float64).reshape(-1, 2)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    if obj_xy_mm.shape[0] < 6:
        return None

    H, _mask = cv2.findHomography(
        obj_xy_mm,
        uv_px,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh_px),
    )
    if H is None:
        return None

    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K0)):
        return None
    if abs(float(K0[2, 2]) - 1.0) > 1e-6:
        K0 = K0 / float(K0[2, 2])
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
    U, _S, Vt = np.linalg.svd(R0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    t = (s * h3).reshape(3).astype(np.float64)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
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
    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    u0, v0, radius = default_disk(w, h)

    uv_all = np.asarray(uv_all, dtype=np.float64).reshape(-1, 2)
    A, mask, _modes = zernike_design_matrix(
        uv_all[:, 0],
        uv_all[:, 1],
        nmax=int(nmax),
        u0_px=float(u0),
        v0_px=float(v0),
        radius_px=float(radius),
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


def _init_coeffs_from_pose_guess(
    *,
    frames: dict[int, FrameObservations],
    rvecs0: dict[int, np.ndarray],
    tvecs0: dict[int, np.ndarray],
    nmax: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
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
        f0_px = 1.5 * float(max(w, h))
        return _init_coeffs_pinhole_prior(
            uv_all=np.zeros((1, 2)), nmax=int(nmax), image_size=image_size, f0_px=f0_px
        )

    uv = np.concatenate(uv_all, axis=0)
    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    A, mask, _modes = zernike_design_matrix(
        uv[:, 0], uv[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius
    )
    if not np.all(mask):
        A = A[mask]
        x = x[mask]
        y = y[mask]

    lam = 1e-6
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x)
    ay = np.linalg.solve(ATA, A.T @ y)
    return ax.astype(np.float64), ay.astype(np.float64)


def _rig_from_poses(
    common_fids: list[int],
    rvecs_L: dict[int, np.ndarray],
    tvecs_L: dict[int, np.ndarray],
    rvecs_R: dict[int, np.ndarray],
    tvecs_R: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _init_pose_guesses_from_observations(
    *,
    runtime: _CharucoRuntime,
    observations: dict[int, FrameObservations],
    image_size: tuple[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    homographies: list[np.ndarray] = []
    for fr_obs in observations.values():
        obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
        uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
        if obj_xy.shape[0] < 6:
            continue
        H, _mask = runtime.cv2.findHomography(
            obj_xy, uv, method=runtime.cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if H is None:
            continue
        homographies.append(np.asarray(H, dtype=np.float64))
    K0 = _estimate_K0_from_homographies(homographies=homographies, image_size=image_size)

    rvecs0: dict[int, np.ndarray] = {}
    tvecs0: dict[int, np.ndarray] = {}
    for fid, fr_obs in observations.items():
        obj_xy = np.asarray(fr_obs.P_board_mm, dtype=np.float64)[:, :2]
        uv = np.asarray(fr_obs.uv_px, dtype=np.float64)
        init = _init_pose_from_homography(runtime.cv2, obj_xy_mm=obj_xy, uv_px=uv, K0=K0)
        if init is None:
            continue
        rvecs0[fid], tvecs0[fid] = init
    return rvecs0, tvecs0


def _ray_fit_health_metrics(
    *,
    model: StereoCentralRayFieldModel,
    frames: dict[int, StereoFrameObservations],
    rvecs: dict[int, np.ndarray],
    tvecs: dict[int, np.ndarray],
) -> dict[str, float]:
    from scipy.spatial.transform import Rotation as R  # type: ignore

    skew_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    n_points_by_frame: list[int] = []

    for fid in sorted(frames):
        fr = frames[int(fid)]
        uv_left = np.asarray(fr.uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right = np.asarray(fr.uv_right_px, dtype=np.float64).reshape(-1, 2)
        points_board = np.asarray(fr.P_board_mm, dtype=np.float64).reshape(-1, 3)
        if uv_left.size == 0:
            continue

        rot = R.from_rotvec(np.asarray(rvecs[int(fid)], dtype=np.float64).reshape(3)).as_matrix()
        tvec = np.asarray(tvecs[int(fid)], dtype=np.float64).reshape(3)
        points_left = (rot @ points_board.T).T + tvec.reshape(1, 3)
        points_right = (model.R_RL @ points_left.T).T + model.t_RL.reshape(1, 3)

        dirs_left = model.left.ray_directions_cam(uv_left[:, 0], uv_left[:, 1])
        dirs_right = model.right.ray_directions_cam(uv_right[:, 0], uv_right[:, 1])
        proj_left = np.sum(points_left * dirs_left, axis=-1, keepdims=True) * dirs_left
        proj_right = np.sum(points_right * dirs_right, axis=-1, keepdims=True) * dirs_right
        residual_values.append(np.linalg.norm(points_left - proj_left, axis=-1))
        residual_values.append(np.linalg.norm(points_right - proj_right, axis=-1))

        _xyz, skew = model.triangulate(uv_left, uv_right)
        skew_values.append(np.asarray(skew, dtype=np.float64).reshape(-1))
        n_points_by_frame.append(int(uv_left.shape[0]))

    if not n_points_by_frame:
        raise RuntimeError("cannot compute fit health metrics without training observations")

    skew_all = np.concatenate(skew_values, axis=0)
    residual_all = np.concatenate(residual_values, axis=0)
    return {
        "train_skew_rms_mm": float(np.sqrt(np.mean(skew_all**2))),
        "train_skew_p95_mm": float(np.quantile(skew_all, 0.95)),
        "train_point_to_ray_rms_mm": float(np.sqrt(np.mean(residual_all**2))),
        "train_point_to_ray_p95_mm": float(np.quantile(residual_all, 0.95)),
        "n_points_total": float(sum(n_points_by_frame)),
        "mean_common_corners_per_frame": float(np.mean(n_points_by_frame)),
    }


def _collect_origin_field_image_observations(
    *,
    pairs: Sequence[StereoImagePair],
    runtime: _CharucoRuntime,
    chess3: np.ndarray,
    method2d: RefineMethod,
    min_common_corners: int,
    tps_lam: float,
    huber_c: float,
    iters: int,
) -> _OriginFieldImageObservations:
    image_size: tuple[int, int] | None = None
    frame_maps_left: list[dict[int, np.ndarray]] = []
    frame_maps_right: list[dict[int, np.ndarray]] = []
    frame_common_ids: list[list[int]] = []
    obj_left: list[np.ndarray] = []
    img_left_cv: list[np.ndarray] = []
    obj_right: list[np.ndarray] = []
    img_right_cv: list[np.ndarray] = []

    for pair in pairs:
        img_l = _ensure_gray_u8(pair.left_path)
        img_r = _ensure_gray_u8(pair.right_path)
        if image_size is None:
            image_size = (int(img_l.shape[1]), int(img_l.shape[0]))
        if img_l.shape != img_r.shape or (int(img_l.shape[1]), int(img_l.shape[0])) != image_size:
            continue

        refined = _detect_refined_stereo_pair(
            runtime=runtime,
            img_left=img_l,
            img_right=img_r,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        if refined is None:
            continue

        ids_l = np.asarray(refined.det_left.charuco_ids, dtype=np.int32).reshape(-1)
        ids_r = np.asarray(refined.det_right.charuco_ids, dtype=np.int32).reshape(-1)
        common_ids = sorted(set(refined.map_left).intersection(refined.map_right))
        if len(common_ids) < int(min_common_corners):
            continue

        frame_maps_left.append(refined.map_left)
        frame_maps_right.append(refined.map_right)
        frame_common_ids.append(common_ids)

        if ids_l.size >= 4:
            obj_left.append(chess3[ids_l].astype(np.float32).reshape(-1, 3))
            img_left_cv.append(refined.xy_left.astype(np.float32).reshape(-1, 2))
        if ids_r.size >= 4:
            obj_right.append(chess3[ids_r].astype(np.float32).reshape(-1, 3))
            img_right_cv.append(refined.xy_right.astype(np.float32).reshape(-1, 2))

    if not frame_common_ids:
        raise RuntimeError("no usable stereo frames after ChArUco detection")
    if image_size is None:
        raise RuntimeError("no image pairs were available for ChArUco detection")

    return _OriginFieldImageObservations(
        image_size=image_size,
        frame_maps_left=frame_maps_left,
        frame_maps_right=frame_maps_right,
        frame_common_ids=frame_common_ids,
        obj_left=obj_left,
        img_left_cv=img_left_cv,
        obj_right=obj_right,
        img_right_cv=img_right_cv,
    )


def _calibrate_origin_field_pinhole_seed(
    *,
    cv2_module: Any,
    observations: _OriginFieldImageObservations,
    chess3: np.ndarray,
) -> _OriginFieldPinholeSeed:
    if len(observations.obj_left) < 2 or len(observations.obj_right) < 2:
        raise RuntimeError("not enough frames for mono calibration (need ≥ 2 per side)")

    criteria = (
        cv2_module.TERM_CRITERIA_EPS + cv2_module.TERM_CRITERIA_MAX_ITER,
        200,
        1e-9,
    )
    _, K_left_cv, dist_left_cv, _, _ = cv2_module.calibrateCamera(
        observations.obj_left,
        observations.img_left_cv,
        observations.image_size,
        None,
        None,
        flags=0,
        criteria=criteria,
    )
    _, K_right_cv, dist_right_cv, _, _ = cv2_module.calibrateCamera(
        observations.obj_right,
        observations.img_right_cv,
        observations.image_size,
        None,
        None,
        flags=0,
        criteria=criteria,
    )

    obj_stereo = [
        chess3[np.asarray(cids, dtype=np.int32)].astype(np.float32).reshape(-1, 3)
        for cids in observations.frame_common_ids
    ]
    img_stereo_l = [
        np.stack(
            [observations.frame_maps_left[i][c] for c in observations.frame_common_ids[i]], axis=0
        ).astype(np.float32)
        for i in range(len(observations.frame_common_ids))
    ]
    img_stereo_r = [
        np.stack(
            [observations.frame_maps_right[i][c] for c in observations.frame_common_ids[i]], axis=0
        ).astype(np.float32)
        for i in range(len(observations.frame_common_ids))
    ]

    _, K_left_cv, dist_left_cv, K_right_cv, _dist_right_cv, R_rl, t_rl, _, _ = (
        cv2_module.stereoCalibrate(
            obj_stereo,
            img_stereo_l,
            img_stereo_r,
            K_left_cv,
            dist_left_cv,
            K_right_cv,
            dist_right_cv,
            observations.image_size,
            criteria=criteria,
            flags=cv2_module.CALIB_FIX_INTRINSIC,
        )
    )

    R_rl_arr = np.asarray(R_rl, dtype=np.float64)
    t_rl_arr = np.asarray(t_rl, dtype=np.float64).reshape(3)
    T_right_left = np.eye(4, dtype=np.float64)
    T_right_left[:3, :3] = R_rl_arr
    T_right_left[:3, 3] = t_rl_arr
    return _OriginFieldPinholeSeed(
        K_left=np.asarray(K_left_cv, dtype=np.float64),
        K_right=np.asarray(K_right_cv, dtype=np.float64),
        # Distortion coefficients seed solvePnP but are intentionally not forwarded to
        # the Zernike BA: the origin field O(u,v) absorbs non-pinhole behaviour directly.
        dist_left=np.asarray(dist_left_cv, dtype=np.float64).reshape(-1),
        T_right_left=T_right_left,
    )


def _build_origin_field_dataset_seed(
    *,
    cv2_module: Any,
    dataset_cls: Any,
    observations: _OriginFieldImageObservations,
    pinhole_seed: _OriginFieldPinholeSeed,
    chess3: np.ndarray,
    min_common_corners: int,
) -> _OriginFieldDatasetSeed:
    # SyntheticStereoDataset requires a single shared object_points array, so we
    # restrict to corners visible in EVERY frame. Frames with partial board
    # coverage are restricted to this global intersection.
    global_common: set[int] = set(observations.frame_common_ids[0])
    for cids in observations.frame_common_ids[1:]:
        global_common = global_common.intersection(cids)
    global_ids = sorted(global_common)
    if len(global_ids) < int(min_common_corners):
        raise RuntimeError(
            f"only {len(global_ids)} corners visible in every frame; need {min_common_corners}. "
            "Try reducing min_common_corners or adding more frames with full board coverage."
        )

    object_points_3d = chess3[np.asarray(global_ids, dtype=np.int32)].astype(np.float64)
    board_poses: list[np.ndarray] = []
    left_pixels: list[np.ndarray] = []
    right_pixels: list[np.ndarray] = []

    for i in range(len(observations.frame_common_ids)):
        uv_l = np.stack([observations.frame_maps_left[i][c] for c in global_ids], axis=0).astype(
            np.float64
        )
        uv_r = np.stack([observations.frame_maps_right[i][c] for c in global_ids], axis=0).astype(
            np.float64
        )
        success, rvec, tvec = cv2_module.solvePnP(
            object_points_3d.astype(np.float32).reshape(-1, 3),
            uv_l.astype(np.float32).reshape(-1, 2),
            pinhole_seed.K_left.astype(np.float32),
            pinhole_seed.dist_left.astype(np.float32),
        )
        if not success:
            continue
        R_board, _ = cv2_module.Rodrigues(rvec)
        T_board = np.eye(4, dtype=np.float64)
        T_board[:3, :3] = np.asarray(R_board, dtype=np.float64)
        T_board[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        board_poses.append(T_board)
        left_pixels.append(uv_l)
        right_pixels.append(uv_r)

    if not board_poses:
        raise RuntimeError("solvePnP failed for all frames")

    dataset = dataset_cls(
        object_points=object_points_3d,
        board_poses=board_poses,
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        K_left=pinhole_seed.K_left,
        K_right=pinhole_seed.K_right,
        T_left_world=np.eye(4, dtype=np.float64),
        T_right_world=pinhole_seed.T_right_left.copy(),
        image_size=observations.image_size,
        oracle_left_params=None,
        oracle_right_params=None,
    )
    return _OriginFieldDatasetSeed(dataset=dataset, board_poses=board_poses)


def fit_opencv_stereo_from_image_pairs(
    *,
    image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]],
    board: CharucoBoardSpec,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    """
    Calibrate a standard OpenCV stereo pinhole model from image pairs.

    This is the onboarding-friendly companion to the 3D ray-field API: it uses the
    same public ChArUco detection/refinement path, then runs plain OpenCV mono and
    stereo calibration. Use `method2d="raw"` for baseline OpenCV and
    `method2d="rayfield_tps_robust"` for OpenCV fed with refined corners.
    """
    import cv2  # type: ignore

    pairs = _normalize_image_pairs(image_pairs)
    if not pairs:
        raise ValueError("image_pairs is empty")

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    image_size: tuple[int, int] | None = None
    detected_pairs = 0
    used_frame_ids: list[int] = []

    obj_left: list[np.ndarray] = []
    img_left: list[np.ndarray] = []
    obj_right: list[np.ndarray] = []
    img_right: list[np.ndarray] = []
    obj_stereo: list[np.ndarray] = []
    img_stereo_left: list[np.ndarray] = []
    img_stereo_right: list[np.ndarray] = []

    for pair in pairs:
        img_left_gray = _ensure_gray_u8(pair.left_path)
        img_right_gray = _ensure_gray_u8(pair.right_path)
        current_size = (int(img_left_gray.shape[1]), int(img_left_gray.shape[0]))
        if image_size is None:
            image_size = current_size
        if current_size != image_size or img_right_gray.shape[:2] != img_left_gray.shape[:2]:
            raise ValueError("all left/right images must share the same size")

        refined = _detect_refined_stereo_pair(
            runtime=runtime,
            img_left=img_left_gray,
            img_right=img_right_gray,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        if refined is None:
            continue
        detected_pairs += 1

        ids_left = np.asarray(refined.det_left.charuco_ids, dtype=np.int32).reshape(-1)
        ids_right = np.asarray(refined.det_right.charuco_ids, dtype=np.int32).reshape(-1)
        if ids_left.size >= 4:
            obj_left.append(np.asarray(chess3[ids_left], dtype=np.float32).reshape(-1, 3))
            img_left.append(np.asarray(refined.xy_left, dtype=np.float32).reshape(-1, 2))
        if ids_right.size >= 4:
            obj_right.append(np.asarray(chess3[ids_right], dtype=np.float32).reshape(-1, 3))
            img_right.append(np.asarray(refined.xy_right, dtype=np.float32).reshape(-1, 2))

        common_ids = sorted(set(refined.map_left).intersection(refined.map_right))
        if len(common_ids) < int(min_common_corners):
            continue

        obj_stereo.append(
            np.asarray(chess3[np.asarray(common_ids, dtype=np.int32)], dtype=np.float32).reshape(
                -1, 3
            )
        )
        img_stereo_left.append(
            np.asarray([refined.map_left[i] for i in common_ids], dtype=np.float32).reshape(-1, 2)
        )
        img_stereo_right.append(
            np.asarray([refined.map_right[i] for i in common_ids], dtype=np.float32).reshape(-1, 2)
        )
        used_frame_ids.append(
            int(pair.frame_id if pair.frame_id is not None else len(used_frame_ids))
        )

    if image_size is None:
        raise RuntimeError("no readable image pair found")
    if len(obj_left) < 2 or len(obj_right) < 2:
        raise RuntimeError("not enough mono calibration frames after ChArUco detection/refinement")
    if len(obj_stereo) < 2:
        raise RuntimeError(
            "not enough stereo calibration frames after ChArUco detection/refinement"
        )

    mono_left_rms, K_left, dist_left, _rvecs_left, _tvecs_left = runtime.cv2.calibrateCamera(
        obj_left,
        img_left,
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    mono_right_rms, K_right, dist_right, _rvecs_right, _tvecs_right = runtime.cv2.calibrateCamera(
        obj_right,
        img_right,
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    stereo_rms, K_left, dist_left, K_right, dist_right, R_lr, t_lr, E, F = (
        runtime.cv2.stereoCalibrate(
            obj_stereo,
            img_stereo_left,
            img_stereo_right,
            K_left,
            dist_left,
            K_right,
            dist_right,
            image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    )

    report = StereoOpenCVCalibrationReport(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        n_input_pairs=len(pairs),
        n_detected_pairs=int(detected_pairs),
        n_mono_frames_left=len(obj_left),
        n_mono_frames_right=len(obj_right),
        n_stereo_frames=len(obj_stereo),
        used_frame_ids=tuple(int(fid) for fid in used_frame_ids),
        method2d=method2d,
        min_common_corners=int(min_common_corners),
        mono_left_rms_px=float(mono_left_rms),
        mono_right_rms_px=float(mono_right_rms),
        stereo_rms_px=float(stereo_rms),
        baseline_mm=float(np.linalg.norm(np.asarray(t_lr, dtype=np.float64).reshape(3))),
    )
    return StereoOpenCVCalibrationResult(
        K_left=np.asarray(K_left, dtype=np.float64),
        dist_left=np.asarray(dist_left, dtype=np.float64).reshape(-1),
        K_right=np.asarray(K_right, dtype=np.float64),
        dist_right=np.asarray(dist_right, dtype=np.float64).reshape(-1),
        R_right_from_left=np.asarray(R_lr, dtype=np.float64),
        t_right_from_left_mm=np.asarray(t_lr, dtype=np.float64).reshape(3),
        essential_matrix=np.asarray(E, dtype=np.float64),
        fundamental_matrix=np.asarray(F, dtype=np.float64),
        report=report,
    )


def compare_opencv_stereo_calibration(
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    *,
    max_pairs: int | None = None,
    method2d: str = "rayfield_tps_robust",
    **kwargs,
) -> dict:
    """Run OpenCV stereo calibration with raw AND refined corners.

    This is the recommended first step for an OpenCV user: it runs
    ``fit_opencv_stereo_from_image_dirs`` twice (once with
    ``method2d="raw"``, once with *method2d*) and returns a comparison
    dictionary.

    Parameters are forwarded to ``fit_opencv_stereo_from_image_dirs``.
    """
    raw = fit_opencv_stereo_from_image_dirs(
        left_dir=Path(left_dir),
        right_dir=Path(right_dir),
        board=board,
        method2d="raw",
        max_pairs=max_pairs,
        **kwargs,
    )
    refined = fit_opencv_stereo_from_image_dirs(
        left_dir=Path(left_dir),
        right_dir=Path(right_dir),
        board=board,
        method2d=method2d,
        max_pairs=max_pairs,
        **kwargs,
    )
    return {
        "raw": raw.to_dict(),
        "refined": refined.to_dict(),
        "raw_result": raw,
        "refined_result": refined,
        "improvement_px": float(raw.report.stereo_rms_px - refined.report.stereo_rms_px),
    }


def fit_opencv_stereo_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_pairs: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    pairs = _image_pairs_from_dirs(left_dir, right_dir, max_pairs=max_pairs)
    return fit_opencv_stereo_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )


def fit_opencv_stereo_from_dataset(
    *,
    dataset_root: str | Path,
    split: str = "train",
    scene: str = "scene_0000",
    max_frames: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> StereoOpenCVCalibrationResult:
    board, pairs = _image_pairs_from_dataset(
        dataset_root=dataset_root, split=split, scene=scene, max_frames=max_frames
    )
    return fit_opencv_stereo_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )


def fit_stereo_central_rayfield_from_image_pairs(
    *,
    image_pairs: Sequence[StereoImagePair | tuple[str | Path, str | Path]],
    board: CharucoBoardSpec,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    """
    Calibrate a public stereo central ray-field model directly from image pairs.

    This is the stable high-level entry point for users who already have left/right
    image pairs and a known ChArUco board.
    """
    pairs = _normalize_image_pairs(image_pairs)
    if not pairs:
        raise ValueError("image_pairs is empty")

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    obs_by_side: dict[str, dict[int, FrameObservations]] = {"left": {}, "right": {}}
    detected_pairs = 0
    image_size: tuple[int, int] | None = None

    for pair in pairs:
        img_left = _ensure_gray_u8(pair.left_path)
        img_right = _ensure_gray_u8(pair.right_path)
        if image_size is None:
            image_size = (int(img_left.shape[1]), int(img_left.shape[0]))
        if img_left.shape != img_right.shape:
            raise ValueError(f"left/right image size mismatch for frame {pair.frame_id}")
        if image_size != (int(img_left.shape[1]), int(img_left.shape[0])):
            raise ValueError("all images must have the same size")

        refined = _detect_refined_stereo_pair(
            runtime=runtime,
            img_left=img_left,
            img_right=img_right,
            method2d=method2d,
            tps_lam=tps_lam,
            huber_c=huber_c,
            iters=iters,
        )
        if refined is None:
            continue
        detected_pairs += 1

        common_ids = sorted(set(refined.map_left).intersection(refined.map_right))
        if len(common_ids) < int(min_common_corners):
            continue

        uv_left = np.stack([refined.map_left[i] for i in common_ids], axis=0).astype(np.float64)
        uv_right = np.stack([refined.map_right[i] for i in common_ids], axis=0).astype(np.float64)
        obj = chess3[np.asarray(common_ids, dtype=np.int32)].astype(np.float64)
        fid = int(pair.frame_id if pair.frame_id is not None else len(obs_by_side["left"]))
        obs_by_side["left"][fid] = FrameObservations(uv_px=uv_left, P_board_mm=obj)
        obs_by_side["right"][fid] = FrameObservations(uv_px=uv_right, P_board_mm=obj)

    if not obs_by_side["left"] or not obs_by_side["right"]:
        raise RuntimeError("no usable stereo frames after ChArUco detection/refinement")
    assert image_size is not None

    rvecs0_L, tvecs0_L = _init_pose_guesses_from_observations(
        runtime=runtime, observations=obs_by_side["left"], image_size=image_size
    )
    rvecs0_R, tvecs0_R = _init_pose_guesses_from_observations(
        runtime=runtime, observations=obs_by_side["right"], image_size=image_size
    )
    common = sorted(
        set(rvecs0_L)
        .intersection(rvecs0_R)
        .intersection(obs_by_side["left"])
        .intersection(obs_by_side["right"])
    )
    if len(common) < 2:
        raise RuntimeError("not enough frames with initialized poses")

    obs_left = {fid: obs_by_side["left"][fid] for fid in common}
    obs_right = {fid: obs_by_side["right"][fid] for fid in common}
    rvecs0_L = {fid: rvecs0_L[fid] for fid in common}
    tvecs0_L = {fid: tvecs0_L[fid] for fid in common}
    rvecs0_R = {fid: rvecs0_R[fid] for fid in common}
    tvecs0_R = {fid: tvecs0_R[fid] for fid in common}

    coeffs0_x_L, coeffs0_y_L = _init_coeffs_from_pose_guess(
        frames=obs_left,
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        nmax=int(nmax),
        image_size=image_size,
    )
    coeffs0_x_R, coeffs0_y_R = _init_coeffs_from_pose_guess(
        frames=obs_right,
        rvecs0=rvecs0_R,
        tvecs0=tvecs0_R,
        nmax=int(nmax),
        image_size=image_size,
    )

    stereo_frames = {
        fid: StereoFrameObservations(
            uv_left_px=obs_left[fid].uv_px,
            uv_right_px=obs_right[fid].uv_px,
            P_board_mm=obs_left[fid].P_board_mm,
        )
        for fid in common
    }
    R_RL0, t_RL0, _C_R_in_L0 = _rig_from_poses(common, rvecs0_L, tvecs0_L, rvecs0_R, tvecs0_R)
    rig_rvec0, _ = runtime.cv2.Rodrigues(R_RL0)
    res = fit_central_stereo_rayfield_ba(
        frames=stereo_frames,
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        nmax=int(nmax),
        rvecs0=rvecs0_L,
        tvecs0=tvecs0_L,
        rig_rvec0=rig_rvec0.reshape(3),
        rig_tvec0=t_RL0.reshape(3),
        coeffs0_left_x=coeffs0_x_L,
        coeffs0_left_y=coeffs0_y_L,
        coeffs0_right_x=coeffs0_x_R,
        coeffs0_right_y=coeffs0_y_R,
        lam_coeff=float(lam_coeff),
        lam_center=1e-1,
        lam_jacobian=10.0,
        loss="huber",
        f_scale_mm=1.0,
        max_nfev=int(max_nfev),
    )

    from scipy.spatial.transform import Rotation as R  # type: ignore

    R_RL = R.from_rotvec(res.rig_rvec.reshape(3)).as_matrix()
    t_RL = res.rig_tvec.reshape(3)
    model = StereoCentralRayFieldModel.from_coeffs(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
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
    fit_health = _ray_fit_health_metrics(
        model=model,
        frames=stereo_frames,
        rvecs=res.rvecs,
        tvecs=res.tvecs,
    )

    exported_model_json = None
    if export_model_dir is not None:
        exported_model_json = str(save_stereo_central_rayfield(Path(export_model_dir), model))

    report = StereoCentralRayFieldFitReport(
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        n_input_pairs=len(pairs),
        n_detected_pairs=int(detected_pairs),
        n_observation_frames=len(obs_by_side["left"]),
        n_initialized_frames=len(common),
        used_frame_ids=tuple(int(fid) for fid in common),
        method2d=method2d,
        min_common_corners=int(min_common_corners),
        nmax=int(nmax),
        train_skew_rms_mm=float(fit_health["train_skew_rms_mm"]),
        train_skew_p95_mm=float(fit_health["train_skew_p95_mm"]),
        train_point_to_ray_rms_mm=float(fit_health["train_point_to_ray_rms_mm"]),
        train_point_to_ray_p95_mm=float(fit_health["train_point_to_ray_p95_mm"]),
        n_points_total=int(fit_health["n_points_total"]),
        mean_common_corners_per_frame=float(fit_health["mean_common_corners_per_frame"]),
        diagnostics={**dict(res.diagnostics), **fit_health},
        exported_model_json=exported_model_json,
    )
    return StereoCentralRayFieldFitResult(model=model, report=report)


def fit_stereo_central_rayfield_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_pairs: int = 0,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    pairs = _image_pairs_from_dirs(left_dir, right_dir, max_pairs=max_pairs)
    return fit_stereo_central_rayfield_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        nmax=nmax,
        lam_coeff=lam_coeff,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
        max_nfev=max_nfev,
        export_model_dir=export_model_dir,
    )


def calibrate(
    *,
    cameras: Sequence[CameraSetup],
    board: CharucoBoardSpec,
    max_pairs: int = 0,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> NCameraCalibrationResult:
    """Calibrate a named camera set.

    Phase 1 exposes an N-camera API surface while preserving the existing
    stereo implementation for the supported left/right case.
    """
    if not cameras:
        raise ValueError("at least one camera is required")
    camera_by_name = {camera.name: camera for camera in cameras}
    if len(camera_by_name) != len(cameras):
        raise ValueError("camera names must be unique")
    channel_names = tuple(camera.name for camera in cameras)
    if channel_names != ("left", "right"):
        raise NotImplementedError("calibrate currently routes only ('left', 'right') camera setups")

    stereo_result = fit_stereo_central_rayfield_from_image_dirs(
        left_dir=Path(camera_by_name["left"].image_dir),
        right_dir=Path(camera_by_name["right"].image_dir),
        board=board,
        max_pairs=max_pairs,
        method2d=method2d,
        min_common_corners=min_common_corners,
        nmax=nmax,
        lam_coeff=lam_coeff,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
        max_nfev=max_nfev,
        export_model_dir=export_model_dir,
    )
    return NCameraCalibrationResult(channel_names=channel_names, stereo_result=stereo_result)


def fit_stereo_zernike_origin_field_from_image_dirs(
    *,
    left_dir: str | Path,
    right_dir: str | Path,
    board: CharucoBoardSpec,
    max_order: int = 4,
    max_pairs: int = 0,
    method2d: RefineMethod = "raw",
    min_common_corners: int = 10,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    regularization: float = 1e-6,
    max_nfev: int = 200,
):
    """
    Calibrate a non-central stereo Zernike origin-field model from image directories.

    Detects ChArUco corners, runs OpenCV mono+stereo calibration to obtain K and
    the stereo rig transform, solves per-frame board poses, then fits a Zernike
    origin field O(u,v) on top of the pinhole directions.

    Parameters
    ----------
    left_dir, right_dir:
        Directories containing sorted left/right calibration images (same count).
    board:
        ChArUco board specification.
    max_order:
        Maximum Zernike radial order for the origin field.
    max_pairs:
        Cap on the number of image pairs used (0 = use all).
    method2d:
        Corner refinement method passed to refine_charuco_corners.
    min_common_corners:
        Minimum corners common to left and right per frame.
    regularization:
        L2 regularization weight on the Zernike origin coefficients.
    max_nfev:
        Maximum function evaluations for the Zernike BA.

    Returns
    -------
    StereoZernikeOriginFieldFitResult

    Notes
    -----
    All frames are restricted to the intersection of corners visible in EVERY
    frame, because `SyntheticStereoDataset` requires a single shared
    `object_points` array.  Corners not detected in all frames are dropped.
    If this intersection is too small, add more frames with full board coverage
    or reduce `min_common_corners`.
    """
    import cv2  # type: ignore

    from stereocomplex.calibration.fit_zernike_origin_field import fit_stereo_zernike_origin_field  # noqa: PLC0415
    from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig  # noqa: PLC0415
    from stereocomplex.synthetic.parallel_plate import SyntheticStereoDataset  # noqa: PLC0415

    pairs = _image_pairs_from_dirs(left_dir, right_dir, max_pairs=max_pairs)

    runtime = _build_charuco_runtime(board)
    chess3 = np.asarray(runtime.board.getChessboardCorners(), dtype=np.float64)
    observations = _collect_origin_field_image_observations(
        pairs=pairs,
        runtime=runtime,
        chess3=chess3,
        method2d=method2d,
        min_common_corners=min_common_corners,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )
    pinhole_seed = _calibrate_origin_field_pinhole_seed(
        cv2_module=cv2,
        observations=observations,
        chess3=chess3,
    )
    dataset_seed = _build_origin_field_dataset_seed(
        cv2_module=cv2,
        dataset_cls=SyntheticStereoDataset,
        observations=observations,
        pinhole_seed=pinhole_seed,
        chess3=chess3,
        min_common_corners=min_common_corners,
    )

    config = ZernikeOriginFieldConfig(image_size=observations.image_size, max_order=int(max_order))
    return fit_stereo_zernike_origin_field(
        observations=dataset_seed.dataset,
        K_left=pinhole_seed.K_left,
        K_right=pinhole_seed.K_right,
        T_right_left_initial=pinhole_seed.T_right_left,
        board_poses_initial=dataset_seed.board_poses,
        config_left=config,
        config_right=config,
        regularization=float(regularization),
        max_nfev=int(max_nfev),
    )


def fit_stereo_central_rayfield_from_dataset(
    *,
    dataset_root: str | Path,
    split: str = "train",
    scene: str = "scene_0000",
    max_frames: int = 0,
    method2d: RefineMethod = "rayfield_tps_robust",
    min_common_corners: int = 10,
    nmax: int = 10,
    lam_coeff: float = 1e-3,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
    max_nfev: int = 800,
    export_model_dir: str | Path | None = None,
) -> StereoCentralRayFieldFitResult:
    board, pairs = _image_pairs_from_dataset(
        dataset_root=dataset_root, split=split, scene=scene, max_frames=max_frames
    )
    return fit_stereo_central_rayfield_from_image_pairs(
        image_pairs=pairs,
        board=board,
        method2d=method2d,
        min_common_corners=min_common_corners,
        nmax=nmax,
        lam_coeff=lam_coeff,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
        max_nfev=max_nfev,
        export_model_dir=export_model_dir,
    )
