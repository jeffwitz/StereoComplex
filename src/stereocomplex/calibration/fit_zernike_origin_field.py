from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import numpy as np

from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginField,
    ZernikeOriginFieldCoefficients,
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
    _project_transverse,
)
from stereocomplex.synthetic.parallel_plate import SyntheticStereoDataset, transform_points


@dataclass(frozen=True)
class StereoZernikeOriginFieldFitResult:
    left_field: ZernikeOriginField
    right_field: ZernikeOriginField
    stereo_transform: np.ndarray
    board_poses: list[np.ndarray]
    residual_rms: float
    residual_median: float
    residual_p95: float
    n_observations: int
    success: bool
    message: str


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _mode_weights(field: ZernikeOriginField) -> np.ndarray:
    return np.array([1.0 + float(mode.n) ** 2 for mode in field.modes], dtype=np.float64)


def _point_line_residual(P: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.cross(P - origin, direction)


def _left_board_poses(
    board_poses: Sequence[np.ndarray], T_left_world: np.ndarray,
) -> list[np.ndarray]:
    T_left_world_arr = np.asarray(T_left_world, dtype=np.float64).reshape(4, 4)
    return [
        T_left_world_arr @ np.asarray(T_board_world, dtype=np.float64).reshape(4, 4)
        for T_board_world in board_poses
    ]


def _frame_points_from_left_board_poses(
    object_points: np.ndarray, left_board_poses: Sequence[np.ndarray],
) -> list[np.ndarray]:
    return [transform_points(T_left_board, object_points) for T_left_board in left_board_poses]


def _pose_params_from_transforms(left_board_poses: Sequence[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # type: ignore

    params = []
    for T in left_board_poses:
        T_arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
        params.append(Rotation.from_matrix(T_arr[:3, :3]).as_rotvec())
        params.append(T_arr[:3, 3])
    return np.concatenate(params, axis=0)


def _se3_params_from_transform(T: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # type: ignore

    T_arr = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([Rotation.from_matrix(T_arr[:3, :3]).as_rotvec(), T_arr[:3, 3]], axis=0)


def _se3_params_to_transform(params: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation  # type: ignore

    p = np.asarray(params, dtype=np.float64).reshape(6)
    return _make_transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])


def _pose_params_to_transforms(params: np.ndarray, n_frames: int) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation  # type: ignore

    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size != 6 * int(n_frames):
        raise ValueError(f"expected {6 * int(n_frames)} pose parameters, got {p.size}")
    poses = []
    for i in range(int(n_frames)):
        rv = p[6 * i : 6 * i + 3]
        t = p[6 * i + 3 : 6 * i + 6]
        poses.append(_make_transform(Rotation.from_rotvec(rv).as_matrix(), t))
    return poses


def _world_board_poses(
    left_board_poses: Sequence[np.ndarray], T_left_world: np.ndarray,
) -> list[np.ndarray]:
    T_world_left = np.linalg.inv(np.asarray(T_left_world, dtype=np.float64).reshape(4, 4))
    return [T_world_left @ np.asarray(T, dtype=np.float64).reshape(4, 4) for T in left_board_poses]


def _residual_stats(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.sqrt(np.mean(vals**2))),
        float(np.median(vals)),
        float(np.percentile(vals, 95)),
    )


def fit_stereo_zernike_origin_field(
    observations: SyntheticStereoDataset,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_right_left_initial: np.ndarray,
    board_poses_initial: Sequence[np.ndarray],
    config_left: ZernikeOriginFieldConfig,
    config_right: ZernikeOriginFieldConfig,
    optimize_board_poses: bool = False,
    optimize_stereo_extrinsics: bool = False,
    optimize_directions: bool = False,
    robust_loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"] = "huber",
    regularization: float = 1e-6,
    direction_regularization: float | None = None,
    pose_regularization: float = 0.0,
    rig_regularization: float = 0.0,
    max_nfev: int = 200,
) -> StereoZernikeOriginFieldFitResult:
    """
    Fit a generic Zernike origin field from point-to-ray residuals.

    By default this keeps board poses, stereo extrinsics and directions fixed to
    isolate the identification of `O(u,v)`. Experimental flags can also optimize
    a direction field `d(u,v)` and board poses. The fit never accesses the
    physical parallel-plate oracle parameters.
    """
    if len(board_poses_initial) != len(observations.left_pixels):
        raise ValueError("board_poses_initial must match the number of observed frames")
    left0 = ZernikeOriginField(K_left, config_left)
    right0 = ZernikeOriginField(K_right, config_right)
    if len(left0.modes) != len(right0.modes):
        raise ValueError("left and right fields must use the same number of Zernike terms")
    n_terms = len(left0.modes)

    T_RL_initial = np.asarray(T_right_left_initial, dtype=np.float64).reshape(4, 4)
    left_board_initial = _left_board_poses(board_poses_initial, observations.T_left_world)
    all_obj_per_frame = (
        observations.per_frame_object_points
        if observations.per_frame_object_points is not None
        else [observations.object_points] * len(observations.left_pixels)
    )
    P_left_frames = [
        transform_points(T, obj)
        for T, obj in zip(left_board_initial, all_obj_per_frame, strict=True)
    ]

    frame_data: list[tuple[np.ndarray, np.ndarray]] = []
    for uvL, uvR, P_L in zip(
        observations.left_pixels, observations.right_pixels, P_left_frames, strict=True,
    ):
        uvL_arr = np.asarray(uvL, dtype=np.float64).reshape(-1, 2)
        uvR_arr = np.asarray(uvR, dtype=np.float64).reshape(-1, 2)
        P_L_arr = np.asarray(P_L, dtype=np.float64).reshape(-1, 3)
        is_left_only = uvR_arr.shape[0] == 0
        if not is_left_only and uvL_arr.shape != uvR_arr.shape:
            raise ValueError("inconsistent observation shapes")
        if uvL_arr.shape[0] != P_L_arr.shape[0]:
            raise ValueError("inconsistent observation shapes")
        frame_data.append((uvL_arr, uvR_arr))

    # Pixel coordinates never change during optimisation; precompute Zernike design
    # matrices A(u,v) and pinhole directions d0(u,v) once for each frame.
    A_left_per_frame: list[np.ndarray] = []
    A_right_per_frame: list[np.ndarray] = []
    d0_left_per_frame: list[np.ndarray] = []
    d0_right_per_frame: list[np.ndarray] = []
    for uvL_arr, uvR_arr in frame_data:
        A_left_per_frame.append(left0.basis(uvL_arr[:, 0], uvL_arr[:, 1]))
        d0_left_per_frame.append(left0.direction(uvL_arr[:, 0], uvL_arr[:, 1]))
        if uvR_arr.shape[0] > 0:
            A_right_per_frame.append(right0.basis(uvR_arr[:, 0], uvR_arr[:, 1]))
            d0_right_per_frame.append(right0.direction(uvR_arr[:, 0], uvR_arr[:, 1]))
        else:
            A_right_per_frame.append(np.zeros((0, n_terms), dtype=np.float64))
            d0_right_per_frame.append(np.zeros((0, 3), dtype=np.float64))
    _gauge_left = config_left.enforce_transverse_gauge
    _gauge_right = config_right.enforce_transverse_gauge

    weights = _mode_weights(left0)
    sqrt_reg = np.sqrt(float(max(regularization, 0.0)))
    direction_reg = (
        regularization if direction_regularization is None else float(direction_regularization)
    )
    sqrt_dir_reg = np.sqrt(float(max(direction_reg, 0.0)))
    sqrt_pose_reg = np.sqrt(float(max(pose_regularization, 0.0)))
    sqrt_rig_reg = np.sqrt(float(max(rig_regularization, 0.0)))
    n_coeff = n_terms * 3
    pose0 = _pose_params_from_transforms(left_board_initial)
    rig0 = _se3_params_from_transform(T_RL_initial)

    p0_parts = [np.zeros((2 * n_coeff,), dtype=np.float64)]
    if optimize_directions:
        p0_parts.append(np.zeros((2 * n_coeff,), dtype=np.float64))
    if optimize_board_poses:
        p0_parts.append(pose0.copy())
    if optimize_stereo_extrinsics:
        p0_parts.append(rig0.copy())
    p0 = np.concatenate(p0_parts, axis=0)

    def unpack(p: np.ndarray):
        """Unpack the parameter vector into origin, direction, poses, and rig components."""
        arr = np.asarray(p, dtype=np.float64).reshape(-1)
        cursor = 0
        left_origin = arr[cursor : cursor + n_coeff].reshape(n_terms, 3)
        cursor += n_coeff
        right_origin = arr[cursor : cursor + n_coeff].reshape(n_terms, 3)
        cursor += n_coeff
        if optimize_directions:
            left_direction = arr[cursor : cursor + n_coeff].reshape(n_terms, 3)
            cursor += n_coeff
            right_direction = arr[cursor : cursor + n_coeff].reshape(n_terms, 3)
            cursor += n_coeff
        else:
            left_direction = np.zeros_like(left_origin)
            right_direction = np.zeros_like(right_origin)
        if optimize_board_poses:
            n_pose_params = 6 * len(frame_data)
            pose_params = arr[cursor : cursor + n_pose_params]
            cursor += n_pose_params
            left_board_poses = _pose_params_to_transforms(pose_params, len(frame_data))
        else:
            pose_params = pose0
            left_board_poses = left_board_initial
        if optimize_stereo_extrinsics:
            rig_params = arr[cursor : cursor + 6]
            cursor += 6
        else:
            rig_params = rig0
        if cursor != arr.size:
            raise ValueError(f"unused optimization parameters: {arr.size - cursor}")
        T_RL_current = _se3_params_to_transform(rig_params)
        return (
            left_origin,
            right_origin,
            left_direction,
            right_direction,
            pose_params,
            left_board_poses,
            rig_params,
            T_RL_current,
        )

    def make_fields(
        left_origin: np.ndarray,
        right_origin: np.ndarray,
        left_direction: np.ndarray,
        right_direction: np.ndarray,
    ):
        """Build left and right ZernikeRayField from coefficient arrays."""
        if optimize_directions:
            left = ZernikeRayField(
                K_left,
                config_left,
                ZernikeRayFieldCoefficients(left_origin, left_direction),
            )
            right = ZernikeRayField(
                K_right,
                config_right,
                ZernikeRayFieldCoefficients(right_origin, right_direction),
            )
        else:
            left = ZernikeOriginField(
                K_left, config_left, ZernikeOriginFieldCoefficients(left_origin),
            )
            right = ZernikeOriginField(
                K_right, config_right, ZernikeOriginFieldCoefficients(right_origin),
            )
        return left, right

    def _ray_cached(
        A: np.ndarray,
        d0: np.ndarray,
        origin_coeffs: np.ndarray,
        direction_coeffs: np.ndarray,
        enforce_gauge: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (O, d) using precomputed design matrix A and pinhole directions d0.

        Mirrors ZernikeOriginField.origin() / ZernikeRayField.direction() using
        _project_transverse so both code paths stay in sync automatically.
        """
        O_raw = A @ origin_coeffs   # (N, 3)
        if optimize_directions:
            d = d0 + _project_transverse(A @ direction_coeffs, d0)
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
        else:
            d = d0
        origin = _project_transverse(O_raw, d) if enforce_gauge else O_raw
        return origin, d

    def residuals(p: np.ndarray) -> np.ndarray:
        """Compute ray-space residuals for the current BA state."""
        (
            left_origin,
            right_origin,
            left_direction,
            right_direction,
            pose_params,
            left_board_poses,
            rig_params,
            T_RL_current,
        ) = unpack(p)
        P_left_current = [
            transform_points(T, obj)
            for T, obj in zip(left_board_poses, all_obj_per_frame, strict=True)
        ]
        R_RL = T_RL_current[:3, :3]
        t_RL = T_RL_current[:3, 3]
        parts: list[np.ndarray] = []
        for i, ((_uvL_arr, uvR_arr), P_L_arr) in enumerate(
            zip(frame_data, P_left_current, strict=True),
        ):
            O_L, d_L = _ray_cached(
                A_left_per_frame[i], d0_left_per_frame[i], left_origin, left_direction,
                _gauge_left,
            )
            parts.append(_point_line_residual(P_L_arr, O_L, d_L).reshape(-1))
            if uvR_arr.shape[0] > 0:
                P_R_arr = (R_RL @ P_L_arr.T).T + t_RL.reshape(1, 3)
                O_R, d_R = _ray_cached(
                    A_right_per_frame[i], d0_right_per_frame[i], right_origin,
                    right_direction, _gauge_right,
                )
                parts.append(_point_line_residual(P_R_arr, O_R, d_R).reshape(-1))
        if sqrt_reg > 0:
            reg_left = (np.sqrt(weights)[:, None] * left_origin).reshape(-1)
            reg_right = (np.sqrt(weights)[:, None] * right_origin).reshape(-1)
            parts.append(sqrt_reg * reg_left)
            parts.append(sqrt_reg * reg_right)
        if optimize_directions and sqrt_dir_reg > 0:
            reg_left_dir = (np.sqrt(weights)[:, None] * left_direction).reshape(-1)
            reg_right_dir = (np.sqrt(weights)[:, None] * right_direction).reshape(-1)
            parts.append(sqrt_dir_reg * reg_left_dir)
            parts.append(sqrt_dir_reg * reg_right_dir)
        if optimize_board_poses and sqrt_pose_reg > 0:
            parts.append(sqrt_pose_reg * (pose_params - pose0))
        if optimize_stereo_extrinsics and sqrt_rig_reg > 0:
            parts.append(sqrt_rig_reg * (rig_params - rig0))
        return np.concatenate(parts, axis=0)

    from scipy.optimize import least_squares  # type: ignore

    sol = least_squares(
        residuals, p0, method="trf", loss=robust_loss, f_scale=1.0, max_nfev=int(max_nfev),
    )
    (
        left_origin,
        right_origin,
        left_direction,
        right_direction,
        _pose_params,
        left_board_poses,
        _rig_params,
        T_RL_final,
    ) = unpack(sol.x)
    left_field, right_field = make_fields(
        left_origin, right_origin, left_direction, right_direction,
    )
    P_left_final = [
        transform_points(T, obj)
        for T, obj in zip(left_board_poses, all_obj_per_frame, strict=True)
    ]
    R_RL = T_RL_final[:3, :3]
    t_RL = T_RL_final[:3, 3]

    residual_norms: list[np.ndarray] = []
    for (uvL_arr, uvR_arr), P_L_arr in zip(frame_data, P_left_final, strict=True):
        O_L, d_L = left_field.ray(uvL_arr[:, 0], uvL_arr[:, 1])
        residual_norms.append(np.linalg.norm(_point_line_residual(P_L_arr, O_L, d_L), axis=1))
        if uvR_arr.shape[0] > 0:
            P_R_arr = (R_RL @ P_L_arr.T).T + t_RL.reshape(1, 3)
            O_R, d_R = right_field.ray(uvR_arr[:, 0], uvR_arr[:, 1])
            residual_norms.append(np.linalg.norm(_point_line_residual(P_R_arr, O_R, d_R), axis=1))
    all_norms = np.concatenate(residual_norms, axis=0)
    rms, median, p95 = _residual_stats(all_norms)
    return StereoZernikeOriginFieldFitResult(
        left_field=left_field,
        right_field=right_field,
        stereo_transform=T_RL_final,
        board_poses=_world_board_poses(left_board_poses, observations.T_left_world),
        residual_rms=rms,
        residual_median=median,
        residual_p95=p95,
        n_observations=int(sum(uvL.shape[0] for uvL, _uvR in frame_data)),
        success=bool(sol.success),
        message=str(sol.message),
    )
