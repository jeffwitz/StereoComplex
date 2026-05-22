"""Direct model inversion from ChArUco corner observations.

Pipeline A of the direct-vs-rayfield study: jointly optimise optical
parameters and board poses by minimising 2-D reprojection error.
"""

from __future__ import annotations

from dataclasses import dataclass
import time as _time

import numpy as np
from scipy.optimize import least_squares  # type: ignore
from scipy.spatial.transform import Rotation

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
)
from stereocomplex.benchmarks.rayfield_projection import (
    project_point_by_rayfield_inverse,
)
from stereocomplex.physics.model_selection import PhysicalModelSpec

Array = np.ndarray

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class DirectFitResult:
    """Result of a direct optical-model fit to ChArUco observations.

    Attributes
    ----------
    model_name : str
    converged : bool  — ``least_squares`` success flag.
    message : str
    n_parameters_optics : int
    n_parameters_poses : int
    n_parameters_total : int
    n_observations : int
    rss_px2 : float
    rms_px : float
    aic : float
    bic : float
    parameter_vector : ndarray  — full vector ``[theta | eta_1 … eta_n]``.
    parameter_dict : dict  — named optical parameters.
    n_iterations : int
    elapsed_s : float
    """

    model_name: str
    converged: bool
    message: str
    n_parameters_optics: int
    n_parameters_poses: int
    n_parameters_total: int
    n_observations: int
    rss_px2: float
    rms_px: float
    aic: float
    bic: float
    parameter_vector: np.ndarray
    parameter_dict: dict
    n_iterations: int
    elapsed_s: float


def estimate_initial_poses_from_central_pinhole(
    obs: CharucoObservationSet,
    K: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Estimate per-frame (R, t) using OpenCV solvePnP on the LEFT channel.

    Uses a central pinhole model with the supplied ``K``.  This is
    deliberately the wrong optical model for non-central oracles, but
    it provides a reasonable starting point (within ~5 % of truth) that
    the joint nonlinear fit can refine.

    Parameters
    ----------
    obs : CharucoObservationSet
    K : (3,3) ndarray — intrinsic matrix for the left channel.

    Returns
    -------
    R_list : list of (3,3) ndarray
    t_list : list of (3,) ndarray
        One ``(R, t)`` pair per accepted pose in *obs*.
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required for pose estimation")
    R_list, t_list = [], []
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    for i in range(len(obs.left_pixels)):
        lp = obs.left_pixels[i]
        if lp.shape[0] < 4:
            continue
        idx = obs.point_indices[i]
        obj_pts = obs.object_points_mm[idx].astype(np.float64)
        img_pts = lp.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K_arr, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            R_list.append(np.asarray(R, dtype=np.float64))
            t_list.append(np.asarray(tvec, dtype=np.float64).reshape(3))
    return R_list, t_list


def _vector_to_poses(pose_params: Array) -> list[tuple[Array, Array]]:
    """Convert a flat pose-parameter array to a list of (rvec, tvec) pairs."""
    arr = np.asarray(pose_params, dtype=np.float64).reshape(-1)
    n_poses = arr.size // 6
    poses = []
    for i in range(n_poses):
        rv = arr[6 * i : 6 * i + 3]
        tv = arr[6 * i + 3 : 6 * i + 6]
        poses.append((rv, tv))
    return poses


def _poses_to_vector(poses: list[tuple[Array, Array]]) -> Array:
    """Inverse of :func:`_vector_to_poses`."""
    parts = []
    for rv, tv in poses:
        parts.append(np.asarray(rv, dtype=np.float64).reshape(3))
        parts.append(np.asarray(tv, dtype=np.float64).reshape(3))
    return np.concatenate(parts)


def fit_direct_model_from_observations(
    observations: CharucoObservationSet,
    model_spec: PhysicalModelSpec,
    initial_optical_parameters: Array | None = None,
    initial_poses_R: list[np.ndarray] | None = None,
    initial_poses_t: list[np.ndarray] | None = None,
    K_left: Array | None = None,
    K_right: Array | None = None,
    image_size: tuple[int, int] | None = None,
    *,
    bounds_optics: tuple[Array, Array] | None = None,
    robust_loss: str = "huber",
    max_nfev: int = 500,
) -> DirectFitResult:
    """Fit an optical model directly to ChArUco 2-D corner observations.

    Jointly optimises the optical parameters *θ* and the board poses *η*
    by minimising the pixel reprojection error.

    Parameters
    ----------
    observations : CharucoObservationSet
        Simulated corner observations (poses and 2-D pixels).
    model_spec : PhysicalModelSpec
        The optical candidate to fit.
    initial_optical_parameters : (p,) ndarray | None
        Initial guess for the optical parameters.  Defaults to
        ``model_spec.initial_parameters``.
    initial_pose_parameters : (6·P,) ndarray | None
        Initial guess for the pose parameters (rvec, tvec per pose).
        Defaults to the ground-truth poses stored in *observations*.
    K_left, K_right : (3,3) ndarray | None
        Intrinsic matrices.  Defaults to ``observations`` image_size with
        fx=fy=200, cx=W/2, cy=H/2 in the absence of a better estimate.
    image_size : (W, H) | None
        Override for the sensor dimensions.
    bounds_optics : (lo, hi) | None
        Bounds for the optical parameters.
    robust_loss : str
        Passed to ``scipy.optimize.least_squares``.
    max_nfev : int

    Returns
    -------
    DirectFitResult
    """
    img = image_size or observations.image_size
    W, H = int(img[0]), int(img[1])

    # --- intrinsics ---
    if K_left is None:
        K_left = np.array([[200.0, 0.0, (W - 1) / 2],
                           [0.0, 200.0, (H - 1) / 2],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
    if K_right is None:
        K_right = K_left.copy()
    K_L = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
    K_R = np.asarray(K_right, dtype=np.float64).reshape(3, 3)

    # --- model setup ---
    model_class = model_spec.model_class
    model_kwargs = dict(model_spec.model_kwargs or {})

    if initial_optical_parameters is None:
        x0_optics = np.asarray(model_spec.initial_parameters, dtype=np.float64).reshape(-1)
    else:
        x0_optics = np.asarray(initial_optical_parameters, dtype=np.float64).reshape(-1)

    p_optics = int(x0_optics.size)

    # --- pose initialisation ---
    n_poses = len(observations.left_pixels)

    if initial_poses_R is not None and initial_poses_t is not None:
        # Use provided pose estimates
        pose_pairs = []
        for R, t in zip(initial_poses_R, initial_poses_t, strict=True):
            rv = Rotation.from_matrix(R).as_rotvec()
            pose_pairs.append((rv, np.asarray(t, dtype=np.float64).reshape(3)))
        x0_poses = _poses_to_vector(pose_pairs)
    elif cv2 is not None:
        # Auto-estimate via central pinhole
        try:
            R_est, t_est = estimate_initial_poses_from_central_pinhole(
                observations, K_L,
            )
            pose_pairs = []
            for R, t in zip(R_est, t_est, strict=True):
                rv = Rotation.from_matrix(R).as_rotvec()
                pose_pairs.append((rv, np.asarray(t, dtype=np.float64).reshape(3)))
            x0_poses = _poses_to_vector(pose_pairs) if pose_pairs else np.zeros(
                6 * n_poses, dtype=np.float64
            )
        except Exception:
            x0_poses = np.zeros(6 * n_poses, dtype=np.float64)
    else:
        # Fallback: zeros (not great, but functional for well-posed problems)
        x0_poses = np.zeros(6 * n_poses, dtype=np.float64)

    if x0_poses.size != 6 * n_poses:
        raise ValueError(
            f"pose vector length {x0_poses.size} does not match {n_poses} poses"
        )
    p_poses = int(x0_poses.size)

    # --- observation vectors (flattened for the residual) ---
    # Build arrays: obs_u, obs_v, point_3d, per measurement
    obs_u_L: list[Array] = []
    obs_v_L: list[Array] = []
    obs_u_R: list[Array] = []
    obs_v_R: list[Array] = []
    obs_pts: list[Array] = []  # 3-D points in local frame for each measurement

    for pose_idx in range(n_poses):
        lp = observations.left_pixels[pose_idx]
        rp = observations.right_pixels[pose_idx]
        idx = observations.point_indices[pose_idx]
        if lp.size == 0:
            continue
        obj_pts_local = observations.object_points_mm[idx]  # (N_k, 3)
        obs_u_L.append(lp[:, 0])
        obs_v_L.append(lp[:, 1])
        obs_u_R.append(rp[:, 0])
        obs_v_R.append(rp[:, 1])
        obs_pts.append(obj_pts_local)

    if not obs_u_L:
        raise ValueError("no visible corners — cannot fit")

    uL_all = np.concatenate(obs_u_L)
    vL_all = np.concatenate(obs_v_L)
    uR_all = np.concatenate(obs_u_R)
    vR_all = np.concatenate(obs_v_R)
    pts_all = np.concatenate(obs_pts, axis=0)

    # Per-measurement offsets for pose lookup
    meas_per_pose = [arr.shape[0] for arr in obs_u_L]
    pose_start = np.cumsum([0] + meas_per_pose)

    n_meas = uL_all.size
    n_scalar = 2 * n_meas * 2  # 2 channels × 2 coordinates per measurement
    n_obs = n_meas  # independent pixel observations per channel

    # --- model factory ---
    _is_shared = bool(getattr(model_class, "is_stereo_shared", False))

    def _channel_field(model, channel: str):
        """Extract a per-channel rayfield from a (possibly shared) model."""
        if hasattr(model, "channel"):
            return model.channel(channel)
        return model

    def make_optical_model(x_optics: Array) -> object:
        """Instantiate the optical model and return a per-channel rayfield."""
        if hasattr(model_class, "from_parameter_vector"):
            m = model_class.from_parameter_vector(x_optics, K=K_L, **model_kwargs)
        else:
            m = model_class(K=K_L, **model_kwargs)
        return _channel_field(m, "left")

    def make_optical_model_right(x_optics: Array) -> object:
        if hasattr(model_class, "from_parameter_vector"):
            m = model_class.from_parameter_vector(x_optics, K=K_R, **model_kwargs)
        else:
            m = model_class(K=K_R, **model_kwargs)
        return _channel_field(m, "right")

    # --- residual function ---
    def residuals(x_all: Array) -> Array:
        x_optics = x_all[:p_optics]
        x_poses = x_all[p_optics:]

        model_L = make_optical_model(x_optics)
        model_R = make_optical_model_right(x_optics)
        poses = _vector_to_poses(x_poses)

        r_blocks = []
        for pi, (rv, tv) in enumerate(poses):
            i0, i1 = pose_start[pi], pose_start[pi + 1]
            if i0 == i1:
                continue
            R = Rotation.from_rotvec(rv).as_matrix()
            t = np.asarray(tv, dtype=np.float64).reshape(3)
            pts_local = pts_all[i0:i1]
            # Transform to world frame
            pts_world = (R @ pts_local.T).T + t[None, :]

            uL_pred = np.empty(i1 - i0, dtype=np.float64)
            vL_pred = np.empty(i1 - i0, dtype=np.float64)
            uR_pred = np.empty(i1 - i0, dtype=np.float64)
            vR_pred = np.empty(i1 - i0, dtype=np.float64)

            for k in range(i1 - i0):
                X = pts_world[k]
                # Use analytic projection if the model supports it
                if hasattr(model_L, "project_point"):
                    uvL, okL = model_L.project_point(X)
                else:
                    uvL, okL, _ = project_point_by_rayfield_inverse(
                        model_L, X, (W, H), max_nfev=30,
                    )
                    okL = bool(okL)
                if hasattr(model_R, "project_point"):
                    uvR, okR = model_R.project_point(X)
                else:
                    uvR, okR, _ = project_point_by_rayfield_inverse(
                        model_R, X, (W, H), max_nfev=30,
                    )
                    okR = bool(okR)
                uL_pred[k], vL_pred[k] = uvL[0], uvL[1]
                uR_pred[k], vR_pred[k] = uvR[0], uvR[1]

            # Pixel residuals
            rL_u = uL_pred - uL_all[i0:i1]
            rL_v = vL_pred - vL_all[i0:i1]
            rR_u = uR_pred - uR_all[i0:i1]
            rR_v = vR_pred - vR_all[i0:i1]
            r_blocks.append(np.column_stack([rL_u, rL_v, rR_u, rR_v]).reshape(-1))

        return np.concatenate(r_blocks) if r_blocks else np.zeros(0, dtype=np.float64)

    # --- bounds ---
    x0 = np.concatenate([x0_optics, x0_poses])
    if bounds_optics is not None:
        lo_opt, hi_opt = (np.asarray(v, dtype=np.float64).reshape(-1) for v in bounds_optics)
    else:
        lo_opt = np.full(p_optics, -np.inf)
        hi_opt = np.full(p_optics, np.inf)
    lo_pose = x0_poses - 0.5  # allow ±0.5 rad / ±0.5 mm around truth
    hi_pose = x0_poses + 0.5
    bounds = (
        np.concatenate([lo_opt, lo_pose]),
        np.concatenate([hi_opt, hi_pose]),
    )

    # --- optimise ---
    t0 = _time.time()
    sol = least_squares(
        residuals, x0=x0, bounds=bounds, method="trf",
        loss=robust_loss, f_scale=1.0, max_nfev=int(max_nfev),
        xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )
    elapsed = _time.time() - t0

    # --- result ---
    r_final = residuals(sol.x)
    rss = float(np.sum(r_final ** 2))
    rms = float(np.sqrt(rss / max(n_scalar, 1)))
    p_total = p_optics + p_poses
    aic_val = float(2.0 * p_total + n_scalar * np.log(max(rss / n_scalar, 1e-30)))
    bic_val = float(p_total * np.log(max(n_obs, 2)) + n_scalar * np.log(max(rss / n_scalar, 1e-30)))

    fitted_model = make_optical_model(sol.x[:p_optics])
    param_dict = fitted_model.parameter_dict() if hasattr(fitted_model, "parameter_dict") else {}

    return DirectFitResult(
        model_name=model_spec.name,
        converged=bool(sol.success),
        message=str(sol.message),
        n_parameters_optics=p_optics,
        n_parameters_poses=p_poses,
        n_parameters_total=p_total,
        n_observations=n_obs,
        rss_px2=rss,
        rms_px=rms,
        aic=aic_val,
        bic=bic_val,
        parameter_vector=np.asarray(sol.x, dtype=np.float64).copy(),
        parameter_dict=param_dict,
        n_iterations=int(sol.nfev),
        elapsed_s=elapsed,
    )
