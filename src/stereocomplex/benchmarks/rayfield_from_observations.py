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
    MultiCameraCharucoObservationSet,
)
from stereocomplex.benchmarks.direct_inversion import (
    estimate_initial_poses_from_central_pinhole,
)
from stereocomplex.core.model_compact.zernike import eval_real_zernike
from stereocomplex.rayfields.zernike_origin_field import (
    MultiCameraZernikeRayField,
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)

Array = np.ndarray


@dataclass(frozen=True)
class ZernikeFitDiagnostics:
    """Diagnostics from a Zernike rayfield fit to ChArUco observations.

    Attributes
    ----------
    max_order : int
        Largest Zernike radial order represented by the returned rayfield.
    n_zernike_coeffs : int
        Number of fitted rayfield coefficients per camera channel.
    n_poses : int
        Number of calibration-board frames included in the fit.
    n_observations : int
        Total number of observed rays across all channels.
    ray_rms_mm : float
        RMS component of the geometric point-to-ray residual, millimetres;
        prior and regularisation rows are excluded.
    converged : bool
        Whether SciPy reported successful least-squares termination.
    nfev : int
        Number of nonlinear residual evaluations.
    channel_names : tuple[str, ...]
        Camera-channel order used by the coefficient vector and observations.
    stage_scale : float or None
        Fitted common dimensionless stage scale when a
        :class:`StagePosePrior` is active.
    stage_jitter_rms_mm : float or None
        RMS of fitted frame-specific axial offsets, millimetres.
    stage_axis : tuple[float, float, float] or None
        Fitted unit translation-stage axis in the rayfield camera frame.
    """

    max_order: int
    n_zernike_coeffs: int
    n_poses: int
    n_observations: int
    ray_rms_mm: float
    converged: bool
    nfev: int
    channel_names: tuple[str, ...] = ("left", "right")
    stage_scale: float | None = None
    stage_jitter_rms_mm: float | None = None
    stage_axis: tuple[float, float, float] | None = None

    @property
    def n_channels(self) -> int:
        """Number of channels in the observation dataset."""
        return len(self.channel_names)


@dataclass(frozen=True)
class StagePosePrior:
    """Hierarchical axial-stage prior for the constrained Zernike BA.

    The translation of frame ``i`` is parameterised as
    ``t_i = t0 + a * (s * (z_nominal_i - mean(z_nominal)) + epsilon_i)``.
    The shared scale ``s`` carries the systematic stage-calibration
    uncertainty, while ``epsilon_i`` captures frame-specific positioning
    jitter.

    Parameters
    ----------
    nominal_positions_mm : ndarray, shape (n_frames,)
        Nominal physical stage positions in frame order, millimetres. Their
        sign must follow the camera-frame direction of stage travel.
    scale_sigma : float
        One-standard-deviation uncertainty of the dimensionless common scale
        around one.
    jitter_sigma_mm : float
        One-standard-deviation prior for each frame-specific axial jitter,
        millimetres.
    ray_sigma_mm : float
        Expected one-component point-to-ray residual, millimetres. This
        converts the dimensionless Gaussian prior residuals to the same scale
        as the geometric residuals.
    estimate_axis : bool
        If True, estimate two slopes for the unit stage axis
        ``normalize([a_x, a_y, 1])``. If False, use the camera Z axis.
    """

    nominal_positions_mm: Array
    scale_sigma: float = 0.1
    jitter_sigma_mm: float = 0.1
    ray_sigma_mm: float = 1e-3
    estimate_axis: bool = False


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
        Maximum number of least-squares function evaluations.
    origin_reg_weight : float
        L2 regularisation weight applied to origin-Z coefficients for each
        channel.

    Returns
    -------
    left_field, right_field : ZernikeRayField
    diag : ZernikeFitDiagnostics
    """
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
        __slots__ = ("A", "X_local", "d0", "pose_idx")
        pose_idx: int
        A: np.ndarray  # (N, n_modes)
        d0: np.ndarray  # (N, 3)
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
        origin_c: np.ndarray,  # (n_modes, 3)
        dir_c: np.ndarray,  # (n_modes, 3)
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
                origin = O_raw - np.sum(O_raw * d, axis=1, keepdims=True) * d
            else:
                origin = O_raw

            delta = X_world - origin
            proj = np.sum(delta * d, axis=1, keepdims=True) * d
            blocks.append((delta - proj).reshape(-1))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    enforce_gauge = config.enforce_transverse_gauge

    def residuals(x: Array) -> Array:
        """Compute ray-space residuals for the current BA state."""
        cL = x[:n_zernike]
        cR = x[n_zernike : 2 * n_zernike]
        # origin: first half, direction: second half, each (n_modes, 3)
        origin_L = cL[: n_zernike // 2].reshape(n_modes, 3)
        dir_L = cL[n_zernike // 2 :].reshape(n_modes, 3)
        origin_R = cR[: n_zernike // 2].reshape(n_modes, 3)
        dir_R = cR[n_zernike // 2 :].reshape(n_modes, 3)
        pose_params = x[2 * n_zernike :]

        rL = _channel_residuals(origin_L, dir_L, pose_params, groups_L, enforce_gauge)
        rR = _channel_residuals(origin_R, dir_R, pose_params, groups_R, enforce_gauge)
        return np.concatenate([rL, rR])

    x0 = np.concatenate(
        [
            np.zeros(n_zernike, dtype=np.float64),  # left coeffs
            np.zeros(n_zernike, dtype=np.float64),  # right coeffs
            x0_poses_arr,
        ]
    )
    # Bound Zernike coefficients to prevent origin-from-object-plane degeneracy.
    # Origin coefficients: ±20 mm in Z (sub-pupil is < 10 mm from axis).
    # Direction coefficients: ±0.5 dimensionless (small correction to pinhole).
    n_half = n_zernike // 2  # = n_modes * 3
    origin_bounds_lo = np.full(n_half, -np.inf)
    origin_bounds_hi = np.full(n_half, np.inf)
    # Pin Z components (indices 2, 5, 8, ... within each half) to ±20 mm
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
        """Compute regularised ray-space residuals."""
        r_geo = residuals(x)
        # Penalize all origin coefficients (both channels) to stay near zero
        cL = x[:n_zernike]
        cR = x[n_zernike : 2 * n_zernike]
        origin_L = cL[: n_zernike // 2].reshape(n_modes, 3)
        origin_R = cR[: n_zernike // 2].reshape(n_modes, 3)
        # Only penalize the Z component of origin coefficients
        reg_L = np.sqrt(origin_reg_weight) * origin_L[:, 2]
        reg_R = np.sqrt(origin_reg_weight) * origin_R[:, 2]
        return np.concatenate([r_geo, reg_L, reg_R])

    sol = least_squares(
        residuals_reg,
        x0=x0,
        bounds=bounds,
        method="trf",
        loss="linear",
        max_nfev=int(max_nfev),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    # --- rebuild final fields ---
    def _build_field(coeffs_flat: np.ndarray, K: np.ndarray) -> ZernikeRayField:
        arr = np.asarray(coeffs_flat, dtype=np.float64).reshape(-1)
        return ZernikeRayField(
            K=K,
            config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=arr[: n_modes * 3].reshape(n_modes, 3),
                direction_coeffs=arr[n_modes * 3 :].reshape(n_modes, 3),
            ),
        )

    left_field = _build_field(sol.x[:n_zernike], K_L)
    right_field = _build_field(sol.x[n_zernike : 2 * n_zernike], K_R)
    r_final = residuals(sol.x)
    rms = float(np.sqrt(np.mean(r_final**2)))

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


def fit_zernike_rayfields_from_multi_camera_observations(
    obs: MultiCameraCharucoObservationSet,
    image_size: tuple[int, int],
    intrinsics_by_channel: dict[str, np.ndarray],
    max_order: int = 4,
    initial_poses_R: list[np.ndarray] | None = None,
    initial_poses_t: list[np.ndarray] | None = None,
    *,
    max_nfev: int = 300,
    origin_reg_weight: float = 1e-3,
) -> tuple[MultiCameraZernikeRayField, ZernikeFitDiagnostics]:
    """Fit Zernike rayfields from the multi-camera observation container.

    The current optimizer remains stereo internally.  This entry point gives
    Phase 1 callers a channel-indexed contract while keeping the existing
    left/right solver path intact.
    """
    if obs.channel_names != ("left", "right"):
        raise NotImplementedError(
            "multi-camera Zernike fitting currently supports left/right observations"
        )
    missing = [name for name in obs.channel_names if name not in intrinsics_by_channel]
    if missing:
        raise ValueError(f"missing intrinsics for channels: {missing}")

    stereo_obs = CharucoObservationSet(
        object_points_mm=obs.object_points_mm,
        pose_rvecs=obs.pose_rvecs,
        pose_tvecs=obs.pose_tvecs,
        left_pixels=obs.pixels("left"),
        right_pixels=obs.pixels("right"),
        point_indices=obs.point_indices,
        noise_std_px=obs.noise_std_px,
        image_size=obs.image_size,
        diagnostics=obs.diagnostics,
    )
    left, right, diag = fit_zernike_rayfield_from_charuco_observations(
        stereo_obs,
        image_size,
        intrinsics_by_channel["left"],
        intrinsics_by_channel["right"],
        max_order=max_order,
        initial_poses_R=initial_poses_R,
        initial_poses_t=initial_poses_t,
        max_nfev=max_nfev,
        origin_reg_weight=origin_reg_weight,
    )
    diag = ZernikeFitDiagnostics(
        max_order=diag.max_order,
        n_zernike_coeffs=diag.n_zernike_coeffs,
        n_poses=diag.n_poses,
        n_observations=diag.n_observations,
        ray_rms_mm=diag.ray_rms_mm,
        converged=diag.converged,
        nfev=diag.nfev,
        channel_names=obs.channel_names,
    )
    return MultiCameraZernikeRayField.from_fields({"left": left, "right": right}), diag


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
    stage_prior: StagePosePrior | None = None,
) -> tuple[
    ZernikeRayField, ZernikeRayField, ZernikeFitDiagnostics, list[np.ndarray], list[np.ndarray]
]:
    """Fit a Zernike rayfield with a constrained translation-stage pose model.

    O: order *max_order_o* (origin field; 0 = rigid sub-pupil).
    d: order *max_order_d* (direction correction).

    Without ``stage_prior``, all frames share one rotation and X,Y translation,
    while Z is free per frame. With ``stage_prior``, translations follow a
    nominal stage ladder with one common scale and weak frame-specific jitter.
    The stage axis can optionally be estimated with two slope parameters.

    Parameters
    ----------
    obs : CharucoObservationSet
        Stereo ChArUco observations, one entry per stage frame.
    image_size : tuple[int, int]
        Sensor width and height, pixels.
    K_left, K_right : ndarray, shape (3, 3)
        Central intrinsics supplying the base ray directions.
    max_order_o : int
        Maximum Zernike radial order for ray origins.
    max_order_d : int
        Maximum Zernike radial order for direction perturbations.
    max_nfev : int
        Maximum number of nonlinear least-squares evaluations.
    origin_reg_weight : float
        L2 weight on the piston Z component of each channel origin.
    stage_prior : StagePosePrior or None
        Optional hierarchical stage-metrology prior. ``None`` preserves the
        historical free-per-frame-Z parameterisation.

    Returns
    -------
    left_field, right_field : ZernikeRayField
    diag : ZernikeFitDiagnostics
    opt_R : list of (3,3) ndarray — optimized rotation matrix (same for all)
    opt_t : list of (3,) ndarray — optimized translation per pose
    """
    from stereocomplex.core.model_compact.zernike import (
        eval_real_zernike,
        zernike_modes,
    )

    W, H = int(image_size[0]), int(image_size[1])
    K_L = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    K_R = np.asarray(K_right, dtype=np.float64).reshape(3, 3)

    # Zernike configs
    modes_O = tuple(
        zernike_modes(int(max_order_o))
    )  # 1 mode for order 0, 3 for order 1, 6 for order 2
    modes_d = tuple(zernike_modes(int(max_order_d)))
    n_modes_O = len(modes_O)
    n_modes_d = len(modes_d)

    # --- precompute basis matrices ---
    # We need: A_O for O (order 0), A_d for d (order 2), and d0 (pinhole direction)
    # Flatten observations
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
        raise ValueError("no observations")
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
    n_poses = len(obs.left_pixels)
    nominal_offsets: np.ndarray | None = None
    if stage_prior is not None:
        nominal_positions = np.asarray(
            stage_prior.nominal_positions_mm, dtype=np.float64
        ).reshape(-1)
        if nominal_positions.size != n_poses:
            raise ValueError(
                "stage_prior.nominal_positions_mm must match the number of poses"
            )
        if not np.all(np.isfinite(nominal_positions)):
            raise ValueError("stage nominal positions must be finite")
        if float(np.ptp(nominal_positions)) <= 0.0:
            raise ValueError("stage nominal positions must span a non-zero range")
        if stage_prior.scale_sigma <= 0.0:
            raise ValueError("stage_prior.scale_sigma must be positive")
        if stage_prior.jitter_sigma_mm <= 0.0:
            raise ValueError("stage_prior.jitter_sigma_mm must be positive")
        if stage_prior.ray_sigma_mm <= 0.0:
            raise ValueError("stage_prior.ray_sigma_mm must be positive")
        nominal_offsets = nominal_positions - float(np.mean(nominal_positions))

    def _basis(u_arr, v_arr, modes):
        xi = 2.0 * np.asarray(u_arr, dtype=np.float64) / float(W - 1) - 1.0
        zeta = 2.0 * np.asarray(v_arr, dtype=np.float64) / float(H - 1) - 1.0
        rho = np.sqrt(xi * xi + zeta * zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, len(modes)), dtype=np.float64)
        for j, mode in enumerate(modes):
            A[:, j] = eval_real_zernike(mode, rho, theta)
        return A

    def _pinhole_dir(u_arr, v_arr, K_arr):
        fx_inv = 1.0 / K_arr[0, 0]
        fy_inv = 1.0 / K_arr[1, 1]
        cx, cy = K_arr[0, 2], K_arr[1, 2]
        dx = (u_arr - cx) * fx_inv
        dy = (v_arr - cy) * fy_inv
        dz = np.ones_like(dx)
        inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.column_stack([dx * inv, dy * inv, dz * inv])

    class _Group:
        __slots__ = ("A_O", "A_d", "X_local", "d0", "pose_idx")
        pose_idx: int
        A_O: np.ndarray
        A_d: np.ndarray
        d0: np.ndarray
        X_local: np.ndarray

    groups_L: list[_Group] = []
    groups_R: list[_Group] = []
    for pi in range(n_poses):
        for side, (u_arr, v_arr, idx_arr, K_side, groups) in enumerate(
            [
                (uL, vL, idxL, K_L, groups_L),
                (uR, vR, idxR, K_R, groups_R),
            ]
        ):
            mask = (poseL if side == 0 else poseR) == pi
            if not mask.any():
                continue
            g = _Group()
            g.pose_idx = pi
            g.A_O = _basis(u_arr[mask], v_arr[mask], modes_O)
            g.A_d = _basis(u_arr[mask], v_arr[mask], modes_d)
            g.d0 = _pinhole_dir(u_arr[mask], v_arr[mask], K_side)
            g.X_local = obj_pts[idx_arr[mask]]
            groups.append(g)

    # --- initial poses (un-flipped, consistent) ---
    from stereocomplex.benchmarks.direct_inversion import (
        estimate_initial_poses_from_central_pinhole,
    )

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
    field_x0 = [
        np.zeros(n_field_per_ch, dtype=np.float64),  # left
        np.zeros(n_field_per_ch, dtype=np.float64),  # right
    ]
    if stage_prior is None:
        pose_x0 = np.concatenate([shared_rotvec, shared_xy, z_per_pose])
    else:
        assert nominal_offsets is not None
        t0_initial = np.array(
            [shared_xy[0], shared_xy[1], float(np.mean(z_per_pose))],
            dtype=np.float64,
        )
        epsilon_initial = z_per_pose - t0_initial[2] - nominal_offsets
        stage_parts = [shared_rotvec, t0_initial]
        if stage_prior.estimate_axis:
            stage_parts.append(np.zeros(2, dtype=np.float64))
        stage_parts.extend([np.ones(1, dtype=np.float64), epsilon_initial])
        pose_x0 = np.concatenate(stage_parts)
    x0 = np.concatenate([*field_x0, pose_x0])

    # Bounds: origin Z ±20mm, direction coeffs ±5.0, poses loose
    O_lo = np.full(n_O, -np.inf)
    O_hi = np.full(n_O, np.inf)
    for j in range(2, n_O, 3):  # pin Z components of all O modes
        O_lo[j] = -20.0
        O_hi[j] = 20.0
    d_lo = np.full(n_d, -5.0)
    d_hi = np.full(n_d, 5.0)
    field_lo = np.concatenate([O_lo, d_lo])
    field_hi = np.concatenate([O_hi, d_hi])
    if stage_prior is None:
        pose_lo = np.concatenate(
            [
                shared_rotvec - 0.05,  # tight rotation bound
                shared_xy - 1.0,  # loose XY
                z_per_pose - 1.0,  # loose Z
            ]
        )
        pose_hi = np.concatenate(
            [
                shared_rotvec + 0.05,
                shared_xy + 1.0,
                z_per_pose + 1.0,
            ]
        )
    else:
        t0_initial = pose_x0[3:6]
        epsilon_initial = pose_x0[-n_poses:]
        pose_lo_parts = [shared_rotvec - 0.05, t0_initial - 1.0]
        pose_hi_parts = [shared_rotvec + 0.05, t0_initial + 1.0]
        if stage_prior.estimate_axis:
            pose_lo_parts.append(np.full(2, -0.25, dtype=np.float64))
            pose_hi_parts.append(np.full(2, 0.25, dtype=np.float64))
        pose_lo_parts.extend(
            [np.array([0.1], dtype=np.float64), epsilon_initial - 1.0]
        )
        pose_hi_parts.extend(
            [np.array([10.0], dtype=np.float64), epsilon_initial + 1.0]
        )
        pose_lo = np.concatenate(pose_lo_parts)
        pose_hi = np.concatenate(pose_hi_parts)
    bounds = (
        np.concatenate([field_lo, field_lo, pose_lo]),
        np.concatenate([field_hi, field_hi, pose_hi]),
    )

    # --- residual ---
    def _chan_residuals(
        O_c: np.ndarray,  # (n_modes_O, 3)  origin coeffs
        d_c: np.ndarray,  # (n_modes_d, 3)  direction coeffs
        rotvec: np.ndarray,
        translations: np.ndarray,
        groups: list[_Group],
    ) -> np.ndarray:
        R = Rotation.from_rotvec(rotvec).as_matrix()
        blocks = []
        for g in groups:
            pi = g.pose_idx
            t = translations[pi]
            X_world = (R @ g.X_local.T).T + t[None, :]

            # Direction: d = normalize(d0 + proj_perp(A_d @ d_c, d0))
            d_delta_raw = g.A_d @ d_c
            d_delta = d_delta_raw - np.sum(d_delta_raw * g.d0, axis=1, keepdims=True) * g.d0
            d = g.d0 + d_delta
            d = d / np.linalg.norm(d, axis=1, keepdims=True)

            # Origin: O = proj_perp(A_O @ O_c, d)  (gauge)
            O_raw = g.A_O @ O_c  # (N, 3)
            origin = O_raw - np.sum(O_raw * d, axis=1, keepdims=True) * d

            delta = X_world - origin
            proj = np.sum(delta * d, axis=1, keepdims=True) * d
            blocks.append((delta - proj).reshape(-1))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def _unpack_poses(
        pose_p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray | None, np.ndarray | None]:
        """Map constrained pose parameters to one translation per frame."""
        rotvec = pose_p[:3]
        if stage_prior is None:
            xy = pose_p[3:5]
            z_vals = pose_p[5:]
            translations = np.column_stack(
                [
                    np.full(n_poses, xy[0], dtype=np.float64),
                    np.full(n_poses, xy[1], dtype=np.float64),
                    z_vals,
                ]
            )
            return rotvec, translations, None, None, None

        assert nominal_offsets is not None
        cursor = 3
        t0 = pose_p[cursor : cursor + 3]
        cursor += 3
        if stage_prior.estimate_axis:
            slopes = pose_p[cursor : cursor + 2]
            cursor += 2
            axis = np.array([slopes[0], slopes[1], 1.0], dtype=np.float64)
            axis /= np.linalg.norm(axis)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        scale = float(pose_p[cursor])
        cursor += 1
        epsilon = pose_p[cursor : cursor + n_poses]
        cursor += n_poses
        if cursor != pose_p.size:
            raise ValueError(
                f"unused constrained-pose parameters: {pose_p.size - cursor}"
            )
        axial_offsets = scale * nominal_offsets + epsilon
        translations = t0[None, :] + axial_offsets[:, None] * axis[None, :]
        return rotvec, translations, scale, epsilon, axis

    def _geometric_residuals(x: np.ndarray) -> np.ndarray:
        """Compute only point-to-ray residuals, excluding all priors."""
        cL = x[:n_field_per_ch]
        cR = x[n_field_per_ch : 2 * n_field_per_ch]
        pose_p = x[2 * n_field_per_ch :]
        rotvec, translations, _scale, _epsilon, _axis = _unpack_poses(pose_p)

        O_L = cL[:n_O].reshape(n_modes_O, 3)
        d_L = cL[n_O:].reshape(n_modes_d, 3)
        O_R = cR[:n_O].reshape(n_modes_O, 3)
        d_R = cR[n_O:].reshape(n_modes_d, 3)

        rL = _chan_residuals(O_L, d_L, rotvec, translations, groups_L)
        rR = _chan_residuals(O_R, d_R, rotvec, translations, groups_R)
        return np.concatenate([rL, rR])

    def residuals(x: np.ndarray) -> np.ndarray:
        """Compute ray-space residuals and optional metrology priors."""
        parts = [_geometric_residuals(x)]
        if origin_reg_weight > 0:
            cL = x[:n_field_per_ch]
            cR = x[n_field_per_ch : 2 * n_field_per_ch]
            O_L = cL[:n_O].reshape(n_modes_O, 3)
            O_R = cR[:n_O].reshape(n_modes_O, 3)
            reg = np.sqrt(origin_reg_weight) * np.array([O_L[0, 2], O_R[0, 2]])
            parts.append(reg)
        if stage_prior is not None:
            pose_p = x[2 * n_field_per_ch :]
            _rotvec, _translations, scale, epsilon, _axis = _unpack_poses(pose_p)
            assert scale is not None
            assert epsilon is not None
            parts.append(
                np.array(
                    [
                        stage_prior.ray_sigma_mm
                        * (scale - 1.0)
                        / stage_prior.scale_sigma
                    ],
                    dtype=np.float64,
                )
            )
            parts.append(
                stage_prior.ray_sigma_mm * epsilon / stage_prior.jitter_sigma_mm
            )
        return np.concatenate(parts)

    # --- solve ---
    sol = least_squares(
        residuals,
        x0=x0,
        bounds=bounds,
        method="trf",
        loss="linear",
        max_nfev=int(max_nfev),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    # --- extract solution ---
    cL_opt = sol.x[:n_field_per_ch]
    cR_opt = sol.x[n_field_per_ch : 2 * n_field_per_ch]
    pose_opt = sol.x[2 * n_field_per_ch :]
    opt_rotvec, opt_translations, opt_scale, opt_epsilon, opt_axis = _unpack_poses(
        pose_opt
    )

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

    left_field = ZernikeRayField(
        K=K_L,
        config=config_full,
        coefficients=_pad_coeffs(
            cL_opt[:n_O].reshape(n_modes_O, 3), cL_opt[n_O:].reshape(n_modes_d, 3)
        ),
    )
    right_field = ZernikeRayField(
        K=K_R,
        config=config_full,
        coefficients=_pad_coeffs(
            cR_opt[:n_O].reshape(n_modes_O, 3), cR_opt[n_O:].reshape(n_modes_d, 3)
        ),
    )

    r_geo = _geometric_residuals(sol.x)
    rms = float(np.sqrt(np.mean(r_geo**2)))

    opt_R = [Rotation.from_rotvec(opt_rotvec).as_matrix() for _ in range(n_poses)]
    opt_t = [opt_translations[i].copy() for i in range(n_poses)]

    diag = ZernikeFitDiagnostics(
        max_order=max_order_full,
        n_zernike_coeffs=n_field_per_ch,
        n_poses=n_poses,
        n_observations=n_obs,
        ray_rms_mm=rms,
        converged=bool(sol.success),
        nfev=int(sol.nfev),
        stage_scale=opt_scale,
        stage_jitter_rms_mm=(
            None
            if opt_epsilon is None
            else float(np.sqrt(np.mean(np.asarray(opt_epsilon) ** 2)))
        ),
        stage_axis=(
            None if opt_axis is None else tuple(float(value) for value in opt_axis)
        ),
    )
    return left_field, right_field, diag, opt_R, opt_t
