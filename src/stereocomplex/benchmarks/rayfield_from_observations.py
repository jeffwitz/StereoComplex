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
from stereocomplex.core.model_compact.zernike import eval_real_zernike
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
    ray_rms_mm: float
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
    origin_reg_weight: float = 1e-3,
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

    # Pre-group observations by pose and pre-compute the Zernike design
    # matrices and pinhole directions.  These depend only on pixel
    # coordinates and K — not on the Zernike coefficients — so we compute
    # them once and reuse at every iteration.
    width, height = image_size
    diag_norm = float(np.sqrt(2.0) * max(width - 1, height - 1)) if max(width - 1, height - 1) > 0 else 1.0

    def _precompute(u_arr: np.ndarray, v_arr: np.ndarray, K: np.ndarray):
        """Return (A, d0) for a batch of pixel coordinates."""
        xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(width - 1) - 1.0
        zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(height - 1) - 1.0
        rho = np.sqrt(xi * xi + zeta * zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, n_modes), dtype=np.float64)
        for j, mode in enumerate(config.modes()):
            A[:, j] = eval_real_zernike(mode, rho, theta)
        # pinhole direction from pixel
        K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
        fx_inv = 1.0 / K_arr[0, 0]
        fy_inv = 1.0 / K_arr[1, 1]
        cx = K_arr[0, 2]
        cy = K_arr[1, 2]
        dx = (u_arr - cx) * fx_inv
        dy = (v_arr - cy) * fy_inv
        dz = np.ones_like(dx)
        inv_norm = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
        d0 = np.column_stack([dx * inv_norm, dy * inv_norm, dz * inv_norm])
        return A, d0

    class _CachedGroup:
        __slots__ = ("pose_idx", "A", "d0", "X_local")
        pose_idx: int
        A: np.ndarray     # (N, n_modes)
        d0: np.ndarray    # (N, 3)
        X_local: np.ndarray  # (N, 3)

    groups_L: list[_CachedGroup] = []
    groups_R: list[_CachedGroup] = []
    for pi in range(n_poses):
        mask_L = poseL == pi
        mask_R = poseR == pi
        if mask_L.any():
            g = _CachedGroup()
            g.pose_idx = pi
            g.A, g.d0 = _precompute(uL[mask_L], vL[mask_L], K_L)
            g.X_local = obj_pts[idxL[mask_L]]
            groups_L.append(g)
        if mask_R.any():
            g = _CachedGroup()
            g.pose_idx = pi
            g.A, g.d0 = _precompute(uR[mask_R], vR[mask_R], K_R)
            g.X_local = obj_pts[idxR[mask_R]]
            groups_R.append(g)

    # --- optimisation ---
    # Pre-compute nothing else — X_world depends on pose parameters which
    # change at each iteration.  But the basis A and pinhole directions d0
    # are fixed, so we inline the field evaluation with A and d0.

    @staticmethod
    def _channel_residuals(
        origin_c: np.ndarray,    # (n_modes, 3)
        dir_c: np.ndarray,       # (n_modes, 3)
        pose_params: np.ndarray,
        groups: list[_CachedGroup],
        enforce_gauge: bool,
    ) -> np.ndarray:
        """Vectorized ray-to-point residuals with pre-computed basis."""
        blocks: list[np.ndarray] = []
        for g in groups:
            pi = g.pose_idx
            rv = pose_params[6 * pi : 6 * pi + 3]
            tv = pose_params[6 * pi + 3 : 6 * pi + 6]
            R = Rotation.from_rotvec(rv).as_matrix()
            t = np.asarray(tv, dtype=np.float64).reshape(3)
            X_world = (R @ g.X_local.T).T + t[None, :]  # (N, 3)

            # Direction: d = normalize(d0 + proj_perp(A @ dir_c, d0))
            d_delta_raw = g.A @ dir_c  # (N, 3)
            d_delta = d_delta_raw - np.sum(d_delta_raw * g.d0, axis=1, keepdims=True) * g.d0
            d_raw = g.d0 + d_delta
            d = d_raw / np.linalg.norm(d_raw, axis=1, keepdims=True)

            # Origin: O = proj_perp(A @ origin_c, d) if gauge
            O_raw = g.A @ origin_c  # (N, 3)
            if enforce_gauge:
                O = O_raw - np.sum(O_raw * d, axis=1, keepdims=True) * d
            else:
                O = O_raw

            delta = X_world - O
            proj = np.sum(delta * d, axis=1, keepdims=True) * d
            blocks.append((delta - proj).reshape(-1))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    enforce_gauge = config.enforce_transverse_gauge

    def residuals(x: Array) -> Array:
        cL = x[:n_zernike]
        cR = x[n_zernike : 2 * n_zernike]
        # origin: first half, direction: second half, each (n_modes, 3)
        origin_L = cL[:n_zernike//2].reshape(n_modes, 3)
        dir_L = cL[n_zernike//2:].reshape(n_modes, 3)
        origin_R = cR[:n_zernike//2].reshape(n_modes, 3)
        dir_R = cR[n_zernike//2:].reshape(n_modes, 3)
        pose_params = x[2 * n_zernike :]

        rL = _channel_residuals(origin_L, dir_L, pose_params, groups_L, enforce_gauge)
        rR = _channel_residuals(origin_R, dir_R, pose_params, groups_R, enforce_gauge)
        return np.concatenate([rL, rR])

    x0 = np.concatenate([
        np.zeros(n_zernike, dtype=np.float64),  # left coeffs
        np.zeros(n_zernike, dtype=np.float64),  # right coeffs
        x0_poses_arr,
    ])
    # Bound Zernike coefficients to prevent origin-from-object-plane degeneracy.
    # Origin coefficients: ±20 mm in Z (sub-pupil is < 10 mm from axis).
    # Direction coefficients: ±0.5 dimensionless (small correction to pinhole).
    n_half = n_zernike // 2  # = n_modes * 3
    origin_bounds_lo = np.full(n_half, -np.inf)
    origin_bounds_hi = np.full(n_half, np.inf)
    # Pin Z components (indices 2, 5, 8, ... within each half) to ±20 mm
    for j in range(2, n_half, 3):
        origin_bounds_lo[j] = -20.0
        origin_bounds_hi[j] = 20.0
    dir_bounds_lo = np.full(n_half, -0.5)
    dir_bounds_hi = np.full(n_half, 0.5)
    coeff_lo = np.concatenate([origin_bounds_lo, dir_bounds_lo])
    coeff_hi = np.concatenate([origin_bounds_hi, dir_bounds_hi])
    # Pose bounds
    lo_pose = x0_poses_arr - 0.3
    hi_pose = x0_poses_arr + 0.3
    bounds = (
        np.concatenate([coeff_lo, coeff_lo, lo_pose]),
        np.concatenate([coeff_hi, coeff_hi, hi_pose]),
    )

    # Regularization: penalize origin Z deviation from pinhole model (Oz≈0).
    # Without this, the BA can place origins at the object plane for some
    # pixels, exploiting a gauge degeneracy with near-flat poses.

    def residuals_reg(x: Array) -> Array:
        r_geo = residuals(x)
        # Penalize all origin coefficients (both channels) to stay near zero
        cL = x[:n_zernike]
        cR = x[n_zernike : 2 * n_zernike]
        origin_L = cL[:n_zernike//2].reshape(n_modes, 3)
        origin_R = cR[:n_zernike//2].reshape(n_modes, 3)
        # Only penalize the Z component of origin coefficients
        reg_L = np.sqrt(origin_reg_weight) * origin_L[:, 2]
        reg_R = np.sqrt(origin_reg_weight) * origin_R[:, 2]
        return np.concatenate([r_geo, reg_L, reg_R])

    sol = least_squares(
        residuals_reg, x0=x0, bounds=bounds, method="trf",
        loss="linear", max_nfev=int(max_nfev),
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    # --- rebuild final fields ---
    def _build_field(coeffs_flat: np.ndarray, K: np.ndarray) -> ZernikeRayField:
        arr = np.asarray(coeffs_flat, dtype=np.float64).reshape(-1)
        return ZernikeRayField(
            K=K, config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=arr[:n_modes*3].reshape(n_modes, 3),
                direction_coeffs=arr[n_modes*3:].reshape(n_modes, 3),
            ),
        )

    left_field = _build_field(sol.x[:n_zernike], K_L)
    right_field = _build_field(sol.x[n_zernike : 2 * n_zernike], K_R)
    r_final = residuals(sol.x)
    rms = float(np.sqrt(np.mean(r_final ** 2)))

    diag = ZernikeFitDiagnostics(
        max_order=max_order,
        n_zernike_coeffs=n_zernike,
        n_poses=n_poses,
        n_observations=n_obs,
        ray_rms_mm=rms,
        converged=bool(sol.success),
        nfev=int(sol.nfev),
    )
    return left_field, right_field, diag


def fit_constrained_zernike_rayfield(
    obs: CharucoObservationSet,
    image_size: tuple[int, int],
    K_left: np.ndarray,
    K_right: np.ndarray,
    max_order_o: int = 0,
    max_order_d: int = 2,
    *,
    max_nfev: int = 300,
    origin_reg_weight: float = 0.0,
) -> tuple[ZernikeRayField, ZernikeRayField, ZernikeFitDiagnostics, list[np.ndarray], list[np.ndarray]]:
    """Fit Zernike rayfield with constrained poses: shared rotation + XY, per-pose Z.

    O: order *max_order_o* (origin field; 0 = rigid sub-pupil).
    d: order *max_order_d* (direction correction).

    Poses are constrained: the board is assumed perfectly vertical, mounted
    on a Z-only translation stage.  All poses share the same rotation
    (3 rotvec params) and X,Y translation (2 params); only Z varies
    per pose (1 param each).

    Returns
    -------
    left_field, right_field : ZernikeRayField
    diag : ZernikeFitDiagnostics
    opt_R : list of (3,3) ndarray — optimized rotation matrix (same for all)
    opt_t : list of (3,) ndarray — optimized translation per pose
    """
    from stereocomplex.core.model_compact.zernike import eval_real_zernike, zernike_modes  # noqa: PLC0415

    W, H = int(image_size[0]), int(image_size[1])
    K_L = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    K_R = np.asarray(K_right, dtype=np.float64).reshape(3, 3)

    # Zernike configs
    modes_O = tuple(zernike_modes(int(max_order_o)))   # 1 mode for order 0, 3 for order 1, 6 for order 2
    modes_d = tuple(zernike_modes(int(max_order_d)))
    n_modes_O = len(modes_O)
    n_modes_d = len(modes_d)

    # --- precompute basis matrices ---
    # We need: A_O for O (order 0), A_d for d (order 2), and d0 (pinhole direction)
    # Flatten observations
    uL_all, vL_all, idxL_all, poseL_all = [], [], [], []
    uR_all, vR_all, idxR_all, poseR_all = [], [], [], []
    for pi in range(len(obs.left_pixels)):
        lp = obs.left_pixels[pi]; rp = obs.right_pixels[pi]
        if lp.size == 0: continue
        idx = obs.point_indices[pi]; n = lp.shape[0]
        uL_all.append(lp[:, 0]); vL_all.append(lp[:, 1]); idxL_all.append(idx)
        poseL_all.append(np.full(n, pi, dtype=int))
        uR_all.append(rp[:, 0]); vR_all.append(rp[:, 1]); idxR_all.append(idx)
        poseR_all.append(np.full(n, pi, dtype=int))

    if not uL_all:
        raise ValueError("no observations")
    uL = np.concatenate(uL_all); vL = np.concatenate(vL_all)
    idxL = np.concatenate(idxL_all); poseL = np.concatenate(poseL_all)
    uR = np.concatenate(uR_all); vR = np.concatenate(vR_all)
    idxR = np.concatenate(idxR_all); poseR = np.concatenate(poseR_all)
    obj_pts = obs.object_points_mm
    n_obs = uL.size + uR.size
    n_poses = len(obs.left_pixels)

    def _basis(u_arr, v_arr, modes):
        xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(W - 1) - 1.0
        zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(H - 1) - 1.0
        rho = np.sqrt(xi*xi + zeta*zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, len(modes)), dtype=np.float64)
        for j, mode in enumerate(modes):
            A[:, j] = eval_real_zernike(mode, rho, theta)
        return A

    def _pinhole_dir(u_arr, v_arr, K_arr):
        fx_inv = 1.0 / K_arr[0, 0]; fy_inv = 1.0 / K_arr[1, 1]
        cx, cy = K_arr[0, 2], K_arr[1, 2]
        dx = (u_arr - cx) * fx_inv; dy = (v_arr - cy) * fy_inv
        dz = np.ones_like(dx)
        inv = 1.0 / np.sqrt(dx*dx + dy*dy + dz*dz)
        return np.column_stack([dx*inv, dy*inv, dz*inv])

    class _Group:
        __slots__ = ("pose_idx", "A_O", "A_d", "d0", "X_local")
        pose_idx: int
        A_O: np.ndarray; A_d: np.ndarray; d0: np.ndarray; X_local: np.ndarray

    groups_L: list[_Group] = []
    groups_R: list[_Group] = []
    for pi in range(n_poses):
        for side, (u_arr, v_arr, idx_arr, K_side, groups) in enumerate([
            (uL, vL, idxL, K_L, groups_L),
            (uR, vR, idxR, K_R, groups_R),
        ]):
            mask = (poseL if side == 0 else poseR) == pi
            if not mask.any(): continue
            g = _Group(); g.pose_idx = pi
            g.A_O = _basis(u_arr[mask], v_arr[mask], modes_O)
            g.A_d = _basis(u_arr[mask], v_arr[mask], modes_d)
            g.d0 = _pinhole_dir(u_arr[mask], v_arr[mask], K_side)
            g.X_local = obj_pts[idx_arr[mask]]
            groups.append(g)

    # --- initial poses (un-flipped, consistent) ---
    from stereocomplex.benchmarks.direct_inversion import estimate_initial_poses_from_central_pinhole  # noqa: PLC0415
    R_est, t_est = estimate_initial_poses_from_central_pinhole(obs, K_L)
    if not R_est:
        raise ValueError("no initial poses")

    # Un-flip: solvePnP can return poses with > 90° rotation (planar ambiguity).
    # Ensure all rotations have angle < 90° for a consistent normal direction.
    rotvecs_unflipped = []
    tvecs_unflipped = []
    for i in range(len(R_est)):
        rv = Rotation.from_matrix(R_est[i]).as_rotvec()
        angle = np.linalg.norm(rv)
        if angle > np.pi / 2:
            # Flip 180° around board X axis
            R_mat = R_est[i] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            rv = Rotation.from_matrix(R_mat).as_rotvec()
        rotvecs_unflipped.append(rv)
        tvecs_unflipped.append(np.asarray(t_est[i], dtype=np.float64))

    # Use median rotation and XY as shared initial values
    all_rv = np.array(rotvecs_unflipped)  # (n_poses, 3)
    shared_rotvec = np.median(all_rv, axis=0)
    all_xy = np.array([t[:2] for t in tvecs_unflipped])  # (n_poses, 2)
    shared_xy = np.median(all_xy, axis=0)

    # Z: use tz from estimates as initial guess (they include stage progression)
    z_per_pose = np.array([tvecs_unflipped[i][2] for i in range(n_poses)], dtype=np.float64)

    # --- parameter layout ---
    # O coeffs: n_modes_O * 3 per channel (left then right)
    n_O = n_modes_O * 3  # = 3 for order 0
    # d coeffs: n_modes_d * 3 per channel
    n_d = n_modes_d * 3  # = 18 for order 2
    n_field_per_ch = n_O + n_d  # = 21
    # Poses: shared rotvec(3) + shared XY(2) + per-pose Z(n_poses)
    n_pose_params = 3 + 2 + n_poses

    x0 = np.concatenate([
        np.zeros(n_field_per_ch, dtype=np.float64),  # left
        np.zeros(n_field_per_ch, dtype=np.float64),  # right
        shared_rotvec,
        shared_xy,
        z_per_pose,
    ])

    # Bounds: origin Z ±20mm, direction coeffs ±5.0, poses loose
    O_lo = np.full(n_O, -np.inf); O_hi = np.full(n_O, np.inf)
    for j in range(2, n_O, 3):  # pin Z components of all O modes
        O_lo[j] = -20.0; O_hi[j] = 20.0
    d_lo = np.full(n_d, -5.0); d_hi = np.full(n_d, 5.0)
    field_lo = np.concatenate([O_lo, d_lo])
    field_hi = np.concatenate([O_hi, d_hi])
    pose_lo = np.concatenate([
        shared_rotvec - 0.05,      # tight rotation bound
        shared_xy - 1.0,           # loose XY
        z_per_pose - 1.0,          # loose Z
    ])
    pose_hi = np.concatenate([
        shared_rotvec + 0.05,
        shared_xy + 1.0,
        z_per_pose + 1.0,
    ])
    bounds = (
        np.concatenate([field_lo, field_lo, pose_lo]),
        np.concatenate([field_hi, field_hi, pose_hi]),
    )

    # --- residual ---
    def _chan_residuals(
        O_c: np.ndarray,   # (n_modes_O, 3)  origin coeffs
        d_c: np.ndarray,   # (n_modes_d, 3)  direction coeffs
        rotvec: np.ndarray,
        xy: np.ndarray,
        z_vals: np.ndarray,
        groups: list[_Group],
    ) -> np.ndarray:
        R = Rotation.from_rotvec(rotvec).as_matrix()
        blocks = []
        for g in groups:
            pi = g.pose_idx
            t = np.array([xy[0], xy[1], z_vals[pi]], dtype=np.float64)
            X_world = (R @ g.X_local.T).T + t[None, :]

            # Direction: d = normalize(d0 + proj_perp(A_d @ d_c, d0))
            d_delta_raw = g.A_d @ d_c
            d_delta = d_delta_raw - np.sum(d_delta_raw * g.d0, axis=1, keepdims=True) * g.d0
            d = g.d0 + d_delta
            d = d / np.linalg.norm(d, axis=1, keepdims=True)

            # Origin: O = proj_perp(A_O @ O_c, d)  (gauge)
            O_raw = g.A_O @ O_c  # (N, 3)
            O = O_raw - np.sum(O_raw * d, axis=1, keepdims=True) * d

            delta = X_world - O
            proj = np.sum(delta * d, axis=1, keepdims=True) * d
            blocks.append((delta - proj).reshape(-1))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def residuals(x: np.ndarray) -> np.ndarray:
        cL = x[:n_field_per_ch]; cR = x[n_field_per_ch:2*n_field_per_ch]
        pose_p = x[2*n_field_per_ch:]
        rotvec = pose_p[:3]; xy = pose_p[3:5]; z_vals = pose_p[5:]

        O_L = cL[:n_O].reshape(n_modes_O, 3); d_L = cL[n_O:].reshape(n_modes_d, 3)
        O_R = cR[:n_O].reshape(n_modes_O, 3); d_R = cR[n_O:].reshape(n_modes_d, 3)

        rL = _chan_residuals(O_L, d_L, rotvec, xy, z_vals, groups_L)
        rR = _chan_residuals(O_R, d_R, rotvec, xy, z_vals, groups_R)

        # Regularization on origin Z
        if origin_reg_weight > 0:
            reg = np.sqrt(origin_reg_weight) * np.array([O_L[0,2], O_R[0,2]])
            return np.concatenate([rL, rR, reg])
        return np.concatenate([rL, rR])

    # --- solve ---
    sol = least_squares(
        residuals, x0=x0, bounds=bounds, method="trf",
        loss="linear", max_nfev=int(max_nfev),
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    # --- extract solution ---
    cL_opt = sol.x[:n_field_per_ch]; cR_opt = sol.x[n_field_per_ch:2*n_field_per_ch]
    pose_opt = sol.x[2*n_field_per_ch:]
    opt_rotvec = pose_opt[:3]; opt_xy = pose_opt[3:5]; opt_z = pose_opt[5:]

    # Build the full ZernikeRayField for each channel
    # For O: use modes_O (order 0), for d: use modes_d (order 2)
    # But ZernikeRayField expects the SAME set of modes for both O and d.
    # Workaround: create config with the larger mode set (order 2), and
    # pad the O coefficients with zeros for the extra modes.
    max_order_full = max(int(max_order_o), int(max_order_d))
    config_full = ZernikeOriginFieldConfig(image_size=image_size, max_order=max_order_full)
    all_modes = config_full.modes()
    n_all = len(all_modes)

    def _pad_coeffs(c_O: np.ndarray, c_d: np.ndarray) -> ZernikeRayFieldCoefficients:
        """Pad O and d coeffs to match the unified mode count."""
        O_padded = np.zeros((n_all, 3), dtype=np.float64)
        O_padded[:n_modes_O, :] = c_O
        d_padded = np.zeros((n_all, 3), dtype=np.float64)
        d_padded[:n_modes_d, :] = c_d
        return ZernikeRayFieldCoefficients(origin_coeffs=O_padded, direction_coeffs=d_padded)

    left_field = ZernikeRayField(K=K_L, config=config_full,
                                  coefficients=_pad_coeffs(
                                      cL_opt[:n_O].reshape(n_modes_O, 3),
                                      cL_opt[n_O:].reshape(n_modes_d, 3)))
    right_field = ZernikeRayField(K=K_R, config=config_full,
                                   coefficients=_pad_coeffs(
                                       cR_opt[:n_O].reshape(n_modes_O, 3),
                                       cR_opt[n_O:].reshape(n_modes_d, 3)))

    r_final = residuals(sol.x)
    # Trim regularization terms for RMS
    n_geo = n_obs * 3
    r_geo = r_final[:n_geo] if origin_reg_weight > 0 else r_final
    rms = float(np.sqrt(np.mean(r_geo ** 2)))

    opt_R = [Rotation.from_rotvec(opt_rotvec).as_matrix() for _ in range(n_poses)]
    opt_t = [np.array([opt_xy[0], opt_xy[1], opt_z[i]], dtype=np.float64) for i in range(n_poses)]

    diag = ZernikeFitDiagnostics(
        max_order=max_order_full,
        n_zernike_coeffs=n_field_per_ch,
        n_poses=n_poses,
        n_observations=n_obs,
        ray_rms_mm=rms,
        converged=bool(sol.success),
        nfev=int(sol.nfev),
    )
    return left_field, right_field, diag, opt_R, opt_t
