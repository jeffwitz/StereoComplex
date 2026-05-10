"""Fit a Zernike rayfield directly from ChArUco corner observations.

This closes the loop in pipeline B: instead of using the oracle rayfield
as a shortcut, we estimate a generic pixel-to-line map from the same 2-D
corner data that pipeline A uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares  # type: ignore
from scipy.spatial.transform import Rotation

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
)
from stereocomplex.benchmarks.direct_inversion import (
    estimate_initial_poses_from_central_pinhole,
)
from stereocomplex.benchmarks.rayfield_projection import point_ray_residual
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)

Array = np.ndarray


@dataclass(frozen=True)
class ZernikeFitDiagnostics:
    """Diagnostics from a Zernike rayfield fit to ChArUco observations."""

    max_order: int
    n_zernike_coeffs: int
    n_poses: int
    n_observations: int
    pixel_rms_px: float
    ray_rms_mm: float | None
    converged: bool
    nfev: int


def fit_zernike_rayfield_from_charuco_observations(
    obs: CharucoObservationSet,
    image_size: tuple[int, int],
    K_left: np.ndarray,
    K_right: np.ndarray,
    max_order: int = 4,
    initial_poses_R: list[np.ndarray] | None = None,
    initial_poses_t: list[np.ndarray] | None = None,
    *,
    max_nfev: int = 300,
) -> tuple[ZernikeRayField, ZernikeRayField, ZernikeFitDiagnostics]:
    """Fit a Zernike rayfield (origin + direction fields) per channel.

    Jointly optimises Zernike coefficients and board poses by minimising
    the perpendicular distance from each 3-D board point to the ray
    at the observed pixel.  This is the BA stage that absorbs poses into
    the rayfield measurement.

    Parameters
    ----------
    obs : CharucoObservationSet
    image_size : (W, H)
    K_left, K_right : (3,3) ndarray
    max_order : int  — Zernike order (default 4 → 15 modes per field).
    initial_poses_R, initial_poses_t : lists of ndarray | None
        Initial pose estimates.  If None, estimated via central pinhole.
    max_nfev : int

    Returns
    -------
    left_field, right_field : ZernikeRayField
    diag : ZernikeFitDiagnostics
    """
    W, H = int(image_size[0]), int(image_size[1])
    K_L = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    K_R = np.asarray(K_right, dtype=np.float64).reshape(3, 3)
    config = ZernikeOriginFieldConfig(image_size=image_size, max_order=max_order)
    n_modes = len(config.modes())
    n_zernike = n_modes * 6  # 3 origin + 3 direction per mode, per channel

    # --- pose initialisation ---
    if initial_poses_R is None or initial_poses_t is None:
        R_est, t_est = estimate_initial_poses_from_central_pinhole(obs, K_L)
    else:
        R_est, t_est = list(initial_poses_R), list(initial_poses_t)

    n_poses = len(R_est)
    if n_poses == 0:
        raise ValueError("no poses available for Zernike fit")

    # Pose vector: for each pose, 3 rotvec + 3 tvec = 6
    x0_poses = []
    for i in range(n_poses):
        rv = Rotation.from_matrix(R_est[i]).as_rotvec()
        x0_poses.append(rv)
        x0_poses.append(np.asarray(t_est[i], dtype=np.float64).reshape(3))
    x0_poses_arr = np.concatenate(x0_poses)
    n_pose_params = x0_poses_arr.size

    # --- observation vectors ---
    # We flatten all observations: each measurement = (pixel_u, pixel_v, X_3d_local)
    uL_all, vL_all, idxL_all, poseL_all = [], [], [], []
    uR_all, vR_all, idxR_all, poseR_all = [], [], [], []
    for pi in range(len(obs.left_pixels)):
        lp = obs.left_pixels[pi]
        rp = obs.right_pixels[pi]
        if lp.size == 0:
            continue
        idx = obs.point_indices[pi]
        n = lp.shape[0]
        uL_all.append(lp[:, 0])
        vL_all.append(lp[:, 1])
        idxL_all.append(idx)
        poseL_all.append(np.full(n, pi, dtype=int))
        uR_all.append(rp[:, 0])
        vR_all.append(rp[:, 1])
        idxR_all.append(idx)
        poseR_all.append(np.full(n, pi, dtype=int))

    if not uL_all:
        raise ValueError("no observations available for Zernike fit")

    uL = np.concatenate(uL_all)
    vL = np.concatenate(vL_all)
    idxL = np.concatenate(idxL_all)
    poseL = np.concatenate(poseL_all)
    uR = np.concatenate(uR_all)
    vR = np.concatenate(vR_all)
    idxR = np.concatenate(idxR_all)
    poseR = np.concatenate(poseR_all)

    obj_pts = obs.object_points_mm
    n_obs = uL.size + uR.size

    # --- optimisation ---
    def make_field(flat_coeffs: Array, K: Array) -> ZernikeRayField:
        arr = np.asarray(flat_coeffs, dtype=np.float64).reshape(-1)
        return ZernikeRayField(
            K=K, config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=arr[:n_modes*3].reshape(n_modes, 3),
                direction_coeffs=arr[n_modes*3:n_modes*6].reshape(n_modes, 3),
            ),
        )

    def residuals(x: Array) -> Array:
        coeffs_L = x[:n_zernike]
        coeffs_R = x[n_zernike : 2 * n_zernike]
        pose_params = x[2 * n_zernike :]

        field_L = make_field(coeffs_L, K_L)
        field_R = make_field(coeffs_R, K_R)

        r_blocks = []
        # Left channel
        for k in range(uL.size):
            pi = poseL[k]
            rv = pose_params[6 * pi : 6 * pi + 3]
            tv = pose_params[6 * pi + 3 : 6 * pi + 6]
            R = Rotation.from_rotvec(rv).as_matrix()
            t = np.asarray(tv, dtype=np.float64).reshape(3)
            X_local = obj_pts[idxL[k]]
            X_world = R @ X_local + t
            uv = np.array([uL[k], vL[k]], dtype=np.float64)
            r = point_ray_residual(uv, field_L, X_world)
            r_blocks.append(r)
        # Right channel
        for k in range(uR.size):
            pi = poseR[k]
            rv = pose_params[6 * pi : 6 * pi + 3]
            tv = pose_params[6 * pi + 3 : 6 * pi + 6]
            R = Rotation.from_rotvec(rv).as_matrix()
            t = np.asarray(tv, dtype=np.float64).reshape(3)
            X_local = obj_pts[idxR[k]]
            X_world = R @ X_local + t
            uv = np.array([uR[k], vR[k]], dtype=np.float64)
            r = point_ray_residual(uv, field_R, X_world)
            r_blocks.append(r)
        return np.concatenate(r_blocks) if r_blocks else np.zeros(0)

    x0 = np.concatenate([
        np.zeros(n_zernike, dtype=np.float64),  # left coeffs
        np.zeros(n_zernike, dtype=np.float64),  # right coeffs
        x0_poses_arr,
    ])
    # Bound poses loosely around initial estimate
    lo_pose = x0_poses_arr - 0.3
    hi_pose = x0_poses_arr + 0.3
    bounds = (
        np.concatenate([np.full(2 * n_zernike, -np.inf), lo_pose]),
        np.concatenate([np.full(2 * n_zernike, np.inf), hi_pose]),
    )

    sol = least_squares(
        residuals, x0=x0, bounds=bounds, method="trf",
        loss="huber", f_scale=1.0, max_nfev=int(max_nfev),
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    left_field = make_field(sol.x[:n_zernike], K_L)
    right_field = make_field(sol.x[n_zernike : 2 * n_zernike], K_R)
    r_final = residuals(sol.x)
    rms = float(np.sqrt(np.mean(r_final ** 2)))

    diag = ZernikeFitDiagnostics(
        max_order=max_order,
        n_zernike_coeffs=n_zernike,
        n_poses=n_poses,
        n_observations=n_obs,
        pixel_rms_px=rms,  # this is in mm (perpendicular distance)
        ray_rms_mm=rms,
        converged=bool(sol.success),
        nfev=int(sol.nfev),
    )
    return left_field, right_field, diag
