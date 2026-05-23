"""Direct (unregularised) optical bundle adjustment for the Pycaso CMO case.

Step 2 of ``CdC_BA_optique_Schur_CMO_Pycaso.md``: a single SciPy
``least_squares`` wrapper that jointly refines the 26-parameter CMO + SE(3)
optical vector and the per-frame board poses against the point-to-ray
residual of :mod:`stereocomplex.optical_ba.residuals`.

This file deliberately stops short of any regularisation: it is the
baseline against which the Schur-based prior (Steps 3 and 4) will be
compared. The wrapper also exports the before/after Schur diagnostic, the
drift of ``theta`` projected onto the weak / strong eigenspaces, and a
small set of physical descriptors (baseline, working distance, convergence
angle) so an experimentalist can read the result without re-deriving them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from stereocomplex.optical_ba.fisher import FisherBlocks, build_fisher_blocks
from stereocomplex.optical_ba.residuals import (
    N_POSE_PER_FRAME,
    N_THETA,
    PycasoCMOObservations,
    default_parameter_scales,
    point_to_ray_residuals_cmo_se3,
)
from stereocomplex.optical_ba.priors import SchurPrior, schur_prior_residuals
from stereocomplex.optical_ba.schur import diagnose_schur_modes


@dataclass(frozen=True)
class OpticalBAResult:
    """Outcome of one direct optical BA run on the Pycaso checkpoint.

    Attributes
    ----------
    theta : ndarray, shape (26,)
        Optimised 26-parameter CMO + per-arm SE(3) optical vector.
    pose_vec : ndarray, shape (6 * n_frames,)
        Optimised per-frame ``(rotvec, tvec)`` vector, in the layout of
        :func:`point_to_ray_residuals_cmo_se3`.
    success : bool
        Whether ``scipy.optimize.least_squares`` reported convergence.
    message : str
        Optimiser status message.
    nfev : int
        Number of residual evaluations performed.
    rms_mm : float
        Initial / final RMS of the transverse point-to-ray distance, in mm.
    rms_px_equivalent : float
        Pixel-equivalent of ``rms_mm`` using ``fx_ref / mean(|t_z|)``.
    p50_px_equivalent, p95_px_equivalent : float
        Median and 95th percentile of the per-observation pixel-equivalent
        distances after optimisation.
    schur_coupling_before, schur_coupling_after : float
        Pose / optics coupling norm at ``theta0`` and at ``theta_final``.
    theta_drift_norm : float
        ``||D_theta^{-1} (theta_final - theta0)||``.
    weak_mode_drift_norm : float
        Norm of the drift restricted to the weak-mode subspace of the
        Schur complement at ``theta0``.
    strong_mode_drift_norm : float
        Norm of the drift restricted to the strong-mode subspace.
    descriptors : dict[str, float]
        Physical descriptors at ``theta_final`` (baseline, working
        distance, focal-objective, convergence angle).
    diagnostics : dict[str, float]
        Free-form key/value extras (initial residual, runtime, etc.).
    """

    theta: np.ndarray
    pose_vec: np.ndarray
    success: bool
    message: str
    nfev: int
    rms_mm: float
    rms_px_equivalent: float
    p50_px_equivalent: float
    p95_px_equivalent: float
    schur_coupling_before: float
    schur_coupling_after: float
    theta_drift_norm: float
    weak_mode_drift_norm: float
    strong_mode_drift_norm: float
    descriptors: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)


def _tel_bounds(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` bounds for the 14-param telecentric CMO vector.

    The principal-point entries (indices 3 and 4) are constrained to the
    central half of the sensor so the affine model cannot push the
    principal point to a corner and compensate with sign flips in the
    slope / shear parameters — that degenerate solution inverts the
    world-frame Y axis.
    """
    cx_center = 0.5 * image_size[0]
    cy_center = 0.5 * image_size[1]
    margin = 0.25 * min(image_size)
    lo = np.array(
        [1.0, 1.0, 0.0, cx_center - margin, cy_center - margin,
         20.0, 0.0, -0.3, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
    )
    hi = np.array(
        [500.0, 1000.0, 200.0, cx_center + margin, cy_center + margin,
         200.0, 0.5, 0.3, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    return lo, hi


_ARM_BOUNDS_LO = np.concatenate([np.full(3, -0.08), np.full(3, -3.0),
                                  np.full(3, -0.08), np.full(3, -3.0)])
_ARM_BOUNDS_HI = np.concatenate([np.full(3, 0.08), np.full(3, 3.0),
                                  np.full(3, 0.08), np.full(3, 3.0)])

# Per-frame pose bounds.  The rotation vector is left unbounded — the
# Rodrigues parameterisation is not unique near 180° rotations, which the
# Pycaso boards (facing the camera, ~175° around the X axis) sit close to.
# Constraining the rotvec there would clip the initial guess catastrophically.
_POSE_BOUNDS_LO = np.full(6, -np.inf)
_POSE_BOUNDS_HI = np.full(6, +np.inf)


def default_bounds(
    n_frames: int, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Default ``(lo, hi)`` bounds for the full ``[theta, pose]`` vector."""
    tel_lo, tel_hi = _tel_bounds(image_size)
    theta_lo = np.concatenate([tel_lo, _ARM_BOUNDS_LO])
    theta_hi = np.concatenate([tel_hi, _ARM_BOUNDS_HI])
    pose_lo = np.tile(_POSE_BOUNDS_LO, n_frames)
    pose_hi = np.tile(_POSE_BOUNDS_HI, n_frames)
    lo = np.concatenate([theta_lo, pose_lo])
    hi = np.concatenate([theta_hi, pose_hi])
    return lo, hi


def _residual_rms_mm(residuals: np.ndarray) -> float:
    r3 = residuals.reshape(-1, 3)
    d_mm = np.linalg.norm(r3, axis=1)
    return float(np.sqrt(np.mean(d_mm**2)))


def _residual_px_stats(
    residuals: np.ndarray, fx_ref: float, mean_z_mm: float
) -> tuple[float, float, float]:
    """Return ``(rms, p50, p95)`` of the pixel-equivalent distances."""
    r3 = residuals.reshape(-1, 3)
    d_mm = np.linalg.norm(r3, axis=1)
    if mean_z_mm <= 0:
        nan = float("nan")
        return nan, nan, nan
    d_px = d_mm * (fx_ref / mean_z_mm)
    return (
        float(np.sqrt(np.mean(d_px**2))),
        float(np.percentile(d_px, 50)),
        float(np.percentile(d_px, 95)),
    )


def _descriptors_from_theta(theta: np.ndarray) -> dict[str, float]:
    """Best-effort physical descriptors derived from the 26-param vector.

    Reads names from ``CMOTelecentricStereoModel.parameter_vector``
    (``shared_slopes=False, shared_shear=True``) without rebuilding the
    model, so the dict is cheap and always available.
    """
    return {
        "f_obj_mm": float(theta[0]),
        "working_distance_mm": float(theta[1]),
        "b_mm": float(theta[2]),
        "f_angular_mm": float(theta[5]),
        "theta_convergence_half_rad": float(theta[6]),
        "rv_L_norm_rad": float(np.linalg.norm(theta[14:17])),
        "t_L_norm_mm": float(np.linalg.norm(theta[17:20])),
        "rv_R_norm_rad": float(np.linalg.norm(theta[20:23])),
        "t_R_norm_mm": float(np.linalg.norm(theta[23:26])),
    }


def _weak_strong_projectors(
    fisher_blocks,
    *,
    damping_pose: float,
    weak_threshold: float,
    theta_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(P_strong, P_weak, coupling)`` from a Schur diagnostic.

    Each projector is a ``(26, 26)`` orthogonal projector onto the
    eigenspace of the Schur complement above (resp. below) the weak
    threshold. The scaled drift ``D_theta^{-1} (theta - theta0)`` projected
    by ``P_weak`` measures movement in the directions the BA is allowed to
    explore "for free" — exactly what the Schur prior would want to
    pin down.
    """
    diag = diagnose_schur_modes(
        fisher_blocks.I_tt,
        fisher_blocks.I_tp,
        fisher_blocks.I_pp,
        weak_threshold=weak_threshold,
        damping_pose=damping_pose,
    )
    V_strong = np.delete(diag.eigvecs, diag.weak_mode_indices, axis=1)
    V_weak = diag.eigvecs[:, diag.weak_mode_indices]
    P_strong = V_strong @ V_strong.T
    P_weak = V_weak @ V_weak.T
    # Sanity: the two projectors should sum to identity (modulo round-off).
    return P_strong, P_weak, diag.coupling_norm


def run_optical_ba(
    *,
    theta0: np.ndarray,
    pose0: np.ndarray,
    observations: PycasoCMOObservations,
    fx_ref_px: float,
    loss: str = "soft_l1",
    f_scale_mm: float = 0.005,
    max_nfev: int = 200,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    weak_threshold: float = 1e-3,
    damping_pose: float = 1e-8,
    fd_method: str = "central",
    fd_rel_step: float = 1e-6,
) -> OpticalBAResult:
    """Run a direct (unregularised) optical BA on the Pycaso checkpoint.

    Minimises the stack of point-to-ray transverse residuals
    (:func:`point_to_ray_residuals_cmo_se3`) over the joint vector
    ``[theta, pose]`` using SciPy's trust-region least-squares solver with
    a robust loss. Before and after the fit, a scaled Schur diagnostic is
    rebuilt so the drift of ``theta`` can be split into weak / strong
    contributions (CDC §7.2).

    Parameters
    ----------
    theta0 : ndarray, shape (26,)
        Initial optical vector (typically the rayfield-identified
        CMO + SE(3) solution).
    pose0 : ndarray, shape (6 * n_frames,)
        Initial per-frame ``(rotvec, tvec)`` vector.
    observations : PycasoCMOObservations
        ChArUco observations and CMO sensor metadata.
    fx_ref_px : float
        Reference pixel focal length used to translate millimetre
        residuals into pixel-equivalent ones (CDC §2.4).
    loss : str
        Robust loss name accepted by ``scipy.optimize.least_squares``.
    f_scale_mm : float
        Robust-loss transition scale, in millimetres. The default
        ``0.005 mm`` is ~2 pixel-equivalents on the Pycaso geometry —
        well above the typical RMS, so outliers are gently softened
        without distorting the noise floor.
    max_nfev : int
        Maximum residual-function evaluations.
    bounds : tuple of (ndarray, ndarray) or None
        ``(lo, hi)`` bounds on the full vector. If ``None``, falls back to
        :func:`default_bounds`.
    weak_threshold : float
        Threshold passed through to :func:`diagnose_schur_modes`.
    damping_pose : float
        Tikhonov damping on the pose block of the Fisher.
    fd_method, fd_rel_step
        Finite-difference settings for the Fisher computation.

    Returns
    -------
    OpticalBAResult
        Optimised parameters, residual statistics, before/after Schur
        diagnostic, drift broken down by Schur subspace, and physical
        descriptors.
    """
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(-1)
    if theta0.size != N_THETA:
        raise ValueError(f"theta0 must have {N_THETA} entries, got {theta0.size}")
    if pose0.size != N_POSE_PER_FRAME * observations.n_frames:
        raise ValueError("pose0 size inconsistent with observations.n_frames")

    n_frames = observations.n_frames
    theta_scales, pose_scales = default_parameter_scales(n_frames)
    if bounds is None:
        bounds = default_bounds(n_frames, observations.image_size)

    def residual_fun(x: np.ndarray) -> np.ndarray:
        return point_to_ray_residuals_cmo_se3(x, observations)

    # Build the "before" Schur diagnostic at theta0.
    fisher_before = build_fisher_blocks(
        residual_fun=residual_fun,
        theta0=theta0,
        pose0=pose0,
        theta_scales=theta_scales,
        pose_scales=pose_scales,
        rel_step=fd_rel_step,
        method=fd_method,
    )
    P_strong, P_weak, coupling_before = _weak_strong_projectors(
        fisher_before,
        damping_pose=damping_pose,
        weak_threshold=weak_threshold,
        theta_scales=theta_scales,
    )

    x0 = np.concatenate([theta0, pose0])
    # Clip the initial guess strictly inside the bounds — the rayfield-derived
    # pose rotvecs can graze the rotation limit (~0.5 rad). Use a small eps
    # so the first least_squares step does not immediately hit a wall.
    lo, hi = bounds
    eps = 1e-6
    x0 = np.clip(x0, np.where(np.isfinite(lo), lo + eps, lo),
                 np.where(np.isfinite(hi), hi - eps, hi))
    sol = least_squares(
        residual_fun,
        x0=x0,
        bounds=bounds,
        loss=loss,
        f_scale=float(f_scale_mm),
        max_nfev=int(max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    theta_final = sol.x[:N_THETA]
    pose_final = sol.x[N_THETA:]
    r_final = residual_fun(sol.x)

    mean_z = float(np.mean(np.abs(pose_final.reshape(n_frames, 6)[:, 5])))
    rms_mm = _residual_rms_mm(r_final)
    rms_px, p50_px, p95_px = _residual_px_stats(r_final, fx_ref_px, mean_z)

    # Build the "after" Schur diagnostic so we can quote the coupling that
    # remains once the optimiser has explored the manifold.
    fisher_after = build_fisher_blocks(
        residual_fun=residual_fun,
        theta0=theta_final,
        pose0=pose_final,
        theta_scales=theta_scales,
        pose_scales=pose_scales,
        rel_step=fd_rel_step,
        method=fd_method,
    )
    _, _, coupling_after = _weak_strong_projectors(
        fisher_after,
        damping_pose=damping_pose,
        weak_threshold=weak_threshold,
        theta_scales=theta_scales,
    )

    # Drift in scaled space, projected on the eigenspaces of the *initial*
    # Schur complement: that is the basis the regulariser would have used.
    delta_scaled = (theta_final - theta0) / theta_scales
    drift_total = float(np.linalg.norm(delta_scaled))
    drift_weak = float(np.linalg.norm(P_weak @ delta_scaled))
    drift_strong = float(np.linalg.norm(P_strong @ delta_scaled))

    descriptors = _descriptors_from_theta(theta_final)

    diagnostics = {
        "rms_mm_initial": _residual_rms_mm(point_to_ray_residuals_cmo_se3(x0, observations)),
        "cost_final": float(sol.cost),
        "f_scale_mm": float(f_scale_mm),
        "loss": str(loss),
        "weak_threshold": float(weak_threshold),
        "damping_pose": float(damping_pose),
        "mean_abs_t_z_mm": mean_z,
    }

    return OpticalBAResult(
        theta=theta_final,
        pose_vec=pose_final,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
        rms_mm=rms_mm,
        rms_px_equivalent=rms_px,
        p50_px_equivalent=p50_px,
        p95_px_equivalent=p95_px,
        schur_coupling_before=coupling_before,
        schur_coupling_after=coupling_after,
        theta_drift_norm=drift_total,
        weak_mode_drift_norm=drift_weak,
        strong_mode_drift_norm=drift_strong,
        descriptors=descriptors,
        diagnostics=diagnostics,
    )


def run_schur_regularized_optical_ba(
    *,
    theta0: np.ndarray,
    pose0: np.ndarray,
    observations: PycasoCMOObservations,
    fx_ref_px: float,
    prior,
    loss: str = "soft_l1",
    f_scale_mm: float = 0.005,
    max_nfev: int = 200,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    weak_threshold: float = 1e-3,
    damping_pose: float = 1e-8,
    fisher_before: FisherBlocks | None = None,
    compute_fisher_after: bool = True,
    fd_method: str = "central",
    fd_rel_step: float = 1e-6,
) -> OpticalBAResult:
    """Run an optical BA with Schur-based or isotropic regularisation.

    Identical to :func:`run_optical_ba` except that the residual vector
    is augmented with prior residuals obtained from *prior* (a
    :class:`~stereocomplex.optical_ba.priors.SchurPrior` or any object
    with a compatible ``schur_prior_residuals``-style interface).

    The *prior* is treated as a black-box callable: the function calls
    ``prior(theta)`` and expects an array of penalty residuals back.
    This works for both :class:`SchurPrior` (via
    :func:`~stereocomplex.optical_ba.priors.schur_prior_residuals`) and
    a simple isotropic lambda.

    Parameters
    ----------
    prior : callable
        ``prior(theta) -> ndarray`` returning the regularisation
        residuals to append to the data residual.
    All other parameters are identical to :func:`run_optical_ba`.
    """
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(-1)
    if theta0.size != N_THETA:
        raise ValueError(f"theta0 must have {N_THETA} entries, got {theta0.size}")
    if pose0.size != N_POSE_PER_FRAME * observations.n_frames:
        raise ValueError("pose0 size inconsistent with observations.n_frames")

    n_frames = observations.n_frames
    theta_scales, pose_scales = default_parameter_scales(n_frames)
    if bounds is None:
        bounds = default_bounds(n_frames, observations.image_size)

    if isinstance(prior, SchurPrior):
        prior_fn = lambda th: schur_prior_residuals(th, prior)  # noqa: E731
    else:
        prior_fn = prior  # assume callable

    def residual_fun(x: np.ndarray) -> np.ndarray:
        data_res = point_to_ray_residuals_cmo_se3(x, observations)
        prior_res = prior_fn(x[:N_THETA])
        return np.concatenate([data_res, prior_res])

    # "Before" Schur diagnostic at theta0 (data residuals only, no prior).
    # Accept a pre-computed FisherBlocks from the caller (e.g. a sweep that
    # already built the diagnostic to construct the SchurPrior) to avoid
    # redundant finite-difference work.
    data_only_fun = lambda x: point_to_ray_residuals_cmo_se3(x, observations)  # noqa: E731
    if fisher_before is None:
        fisher_before = build_fisher_blocks(
            residual_fun=data_only_fun,
            theta0=theta0,
            pose0=pose0,
            theta_scales=theta_scales,
            pose_scales=pose_scales,
            rel_step=fd_rel_step,
            method=fd_method,
        )
    P_strong, P_weak, coupling_before = _weak_strong_projectors(
        fisher_before,
        damping_pose=damping_pose,
        weak_threshold=weak_threshold,
        theta_scales=theta_scales,
    )

    x0 = np.concatenate([theta0, pose0])
    lo, hi = bounds
    eps = 1e-6
    x0 = np.clip(x0, np.where(np.isfinite(lo), lo + eps, lo),
                 np.where(np.isfinite(hi), hi - eps, hi))
    sol = least_squares(
        residual_fun,
        x0=x0,
        bounds=bounds,
        loss=loss,
        f_scale=float(f_scale_mm),
        max_nfev=int(max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    theta_final = sol.x[:N_THETA]
    pose_final = sol.x[N_THETA:]
    r_data = point_to_ray_residuals_cmo_se3(sol.x, observations)

    mean_z = float(np.mean(np.abs(pose_final.reshape(n_frames, 6)[:, 5])))
    rms_mm = _residual_rms_mm(r_data)
    rms_px, p50_px, p95_px = _residual_px_stats(r_data, fx_ref_px, mean_z)

    if compute_fisher_after:
        fisher_after = build_fisher_blocks(
            residual_fun=data_only_fun,
            theta0=theta_final,
            pose0=pose_final,
            theta_scales=theta_scales,
            pose_scales=pose_scales,
            rel_step=fd_rel_step,
            method=fd_method,
        )
        _, _, coupling_after = _weak_strong_projectors(
            fisher_after,
            damping_pose=damping_pose,
            weak_threshold=weak_threshold,
            theta_scales=theta_scales,
        )
    else:
        coupling_after = float("nan")

    delta_scaled = (theta_final - theta0) / theta_scales
    drift_total = float(np.linalg.norm(delta_scaled))
    drift_weak = float(np.linalg.norm(P_weak @ delta_scaled))
    drift_strong = float(np.linalg.norm(P_strong @ delta_scaled))

    descriptors = _descriptors_from_theta(theta_final)

    diagnostics = {
        "rms_mm_initial": _residual_rms_mm(point_to_ray_residuals_cmo_se3(x0, observations)),
        "cost_final": float(sol.cost),
        "f_scale_mm": float(f_scale_mm),
        "loss": str(loss),
        "weak_threshold": float(weak_threshold),
        "damping_pose": float(damping_pose),
        "mean_abs_t_z_mm": mean_z,
    }
    if isinstance(prior, SchurPrior):
        diagnostics["prior_alpha"] = float(prior.alpha)
        diagnostics["prior_power"] = float(prior.power)
        diagnostics["prior_epsilon"] = float(prior.epsilon)
        diagnostics["prior_weak_only"] = bool(prior.weak_only)

    return OpticalBAResult(
        theta=theta_final,
        pose_vec=pose_final,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
        rms_mm=rms_mm,
        rms_px_equivalent=rms_px,
        p50_px_equivalent=p50_px,
        p95_px_equivalent=p95_px,
        schur_coupling_before=coupling_before,
        schur_coupling_after=coupling_after,
        theta_drift_norm=drift_total,
        weak_mode_drift_norm=drift_weak,
        strong_mode_drift_norm=drift_strong,
        descriptors=descriptors,
        diagnostics=diagnostics,
    )
