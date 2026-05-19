"""Inverse-problem conditioning diagnostics.

Computes finite-difference Jacobians, Schur-complement information
matrices, and condition numbers for direct (pipeline A) and rayfield-
mediated (pipeline B) model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class InverseProblemDiagnostics:
    """Conditioning analysis for an optical inverse problem.

    Attributes
    ----------
    singular_values_full : (p_total,) ndarray
        Singular values of the full Jacobian ``[J_optics | J_poses]``.
    singular_values_optics : (p_optics,) ndarray
        Singular values of the optics-only Jacobian block.
    singular_values_schur : (p_optics,) ndarray
        Singular values of :math:`J_θ - J_η (J_ηᵀ J_η)⁻¹ J_ηᵀ J_θ`.
    condition_full : float
        Condition number of the full Jacobian.
    condition_optics : float
        Condition number of the optics-only block.
    condition_schur : float
        Condition number after eliminating poses (Schur complement).
    rank_full : int
        Numerical rank of the full Jacobian (tol = 1e-10).
    rank_schur : int
        Numerical rank of the Schur complement.
    optical_pose_coupling_norm : float
        Frobenius norm of ``J_opticsᵀ J_poses``, normalised.
    max_parameter_correlation : float
        Maximum absolute correlation between any optical parameter and
        any pose parameter.
    correlation_matrix : (p_optics, p_poses) ndarray
        Pairwise correlation coefficients.
    """

    singular_values_full: np.ndarray
    singular_values_optics: np.ndarray
    singular_values_schur: np.ndarray
    condition_full: float
    condition_optics: float
    condition_schur: float
    rank_full: int
    rank_schur: int
    optical_pose_coupling_norm: float
    max_parameter_correlation: float
    correlation_matrix: np.ndarray


def _condition_number(s: Array) -> float:
    s_pos = np.maximum(s, 1e-30)
    return float(s_pos[0] / s_pos[-1])


def _numerical_rank(s: Array, tol: float = 1e-10) -> int:
    return int(np.sum(s > tol * s[0]))


def finite_difference_jacobian(
    residual_fun,
    x0: Array,
    *,
    step: float = 1e-6,
) -> Array:
    """Compute the Jacobian of *residual_fun* at *x0* by central differences.

    Parameters
    ----------
    residual_fun : callable
        ``r = residual_fun(x)`` returning a 1-D residual array.
    x0 : (p,) ndarray
        Point at which to evaluate the Jacobian.
    step : float
        Finite-difference step size.

    Returns
    -------
    J : (n_residuals, p) ndarray
    """
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    p = x.size
    r0 = np.asarray(residual_fun(x), dtype=np.float64).reshape(-1)
    n = r0.size
    J = np.empty((n, p), dtype=np.float64)
    h = float(step)
    for j in range(p):
        xp = x.copy()
        xp[j] += h
        xm = x.copy()
        xm[j] -= h
        rp = np.asarray(residual_fun(xp), dtype=np.float64).reshape(-1)
        rm = np.asarray(residual_fun(xm), dtype=np.float64).reshape(-1)
        J[:, j] = (rp - rm) / (2.0 * h)
    return J


def compute_schur_information(
    J_optics: Array,
    J_poses: Array,
) -> Array:
    """Compute the Schur-complement information matrix for optics.

    .. math::
        I_{θ|η} = J_θᵀ J_θ - J_θᵀ J_η (J_ηᵀ J_η)⁻¹ J_ηᵀ J_θ.

    This is the information available about the optical parameters after
    the nuisance pose parameters have been profiled out.

    Parameters
    ----------
    J_optics : (n, p_optics) ndarray
    J_poses : (n, p_poses) ndarray

    Returns
    -------
    I_schur : (p_optics, p_optics) ndarray
    """
    Jo = np.asarray(J_optics, dtype=np.float64)
    Jp = np.asarray(J_poses, dtype=np.float64)
    JpJp = Jp.T @ Jp
    # Regularise if near-singular
    try:
        JpJp_inv = np.linalg.inv(JpJp)
    except np.linalg.LinAlgError:
        JpJp_inv = np.linalg.pinv(JpJp, hermitian=True)
    return Jo.T @ Jo - Jo.T @ Jp @ JpJp_inv @ Jp.T @ Jo


def compute_inverse_problem_diagnostics(
    residual_function,
    theta_opt: Array,
    eta_opt: Array,
    *,
    step: float = 1e-6,
) -> InverseProblemDiagnostics:
    """Full conditioning analysis for a direct-inversion problem.

    Parameters
    ----------
    residual_function : callable
        ``r = residual_function(theta, eta)`` where *theta* are optical
        parameters and *eta* are pose parameters.  Returns a 1-D residual.
    theta_opt : (p_optics,) ndarray
        Optimal optical parameters.
    eta_opt : (p_poses,) ndarray
        Optimal pose parameters.
    step : float
        Finite-difference step.

    Returns
    -------
    InverseProblemDiagnostics
    """
    theta = np.asarray(theta_opt, dtype=np.float64).reshape(-1)
    eta = np.asarray(eta_opt, dtype=np.float64).reshape(-1)
    p_opt = theta.size
    p_pose = eta.size
    x_full = np.concatenate([theta, eta])

    def full_residual(x):
        return residual_function(x[:p_opt], x[p_opt:])

    J_full = finite_difference_jacobian(full_residual, x_full, step=step)
    J_opt = J_full[:, :p_opt]
    J_pose = J_full[:, p_opt:]
    s_full = np.linalg.svd(J_full, compute_uv=False)
    s_opt = np.linalg.svd(J_opt, compute_uv=False) if p_opt > 0 else np.zeros(0)

    if p_pose > 0 and p_opt > 0:
        I_schur = compute_schur_information(J_opt, J_pose)
        eigvals, _ = np.linalg.eigh(I_schur)
        s_schur = np.sqrt(np.maximum(eigvals, 0.0))
        s_schur = np.sort(s_schur)[::-1]
    elif p_opt > 0:
        I_schur = J_opt.T @ J_opt
        s_schur = np.linalg.svd(J_opt, compute_uv=False)
    else:
        s_schur = np.zeros(0)

    # Correlation between optical and pose parameters
    corr = np.zeros((p_opt, p_pose), dtype=np.float64) if p_opt > 0 and p_pose > 0 else np.zeros((p_opt, p_pose))
    max_corr = 0.0
    if p_opt > 0 and p_pose > 0:
        for i in range(p_opt):
            for j in range(p_pose):
                c = np.corrcoef(J_opt[:, i], J_pose[:, j])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        max_corr = float(np.max(np.abs(corr))) if corr.size > 0 else 0.0

    coupling = float(np.linalg.norm(J_opt.T @ J_pose, "fro")) if p_opt > 0 and p_pose > 0 else 0.0
    norm_opt = float(np.linalg.norm(J_opt, "fro")) if p_opt > 0 else 1.0
    norm_pose = float(np.linalg.norm(J_pose, "fro")) if p_pose > 0 else 1.0
    coupling_norm = coupling / max(norm_opt * norm_pose, 1e-30)

    return InverseProblemDiagnostics(
        singular_values_full=s_full,
        singular_values_optics=s_opt,
        singular_values_schur=s_schur,
        condition_full=_condition_number(s_full) if s_full.size > 0 else 0.0,
        condition_optics=_condition_number(s_opt) if s_opt.size > 0 else 0.0,
        condition_schur=_condition_number(s_schur) if s_schur.size > 0 else 0.0,
        rank_full=_numerical_rank(s_full) if s_full.size > 0 else 0,
        rank_schur=_numerical_rank(s_schur) if s_schur.size > 0 else 0,
        optical_pose_coupling_norm=coupling_norm,
        max_parameter_correlation=max_corr,
        correlation_matrix=corr,
    )


def compute_pipeline_condition_number(
    residual_function,
    theta: Array,
    eta: Array | None = None,
    *,
    step: float = 1e-6,
) -> dict[str, float]:
    """Compute condition numbers for both pipeline types from a residual.

    Parameters
    ----------
    residual_function : callable
        ``r = f(theta, eta=None)``.  If *eta* is None, the residual
        has no pose parameters (pipeline B / rayfield).
    theta : (p_optics,) ndarray
        Optical parameters at the optimum.
    eta : (p_poses,) ndarray | None
        Pose parameters.  If None, only the optics-only condition is computed.

    Returns
    -------
    dict with keys ``condition_optics_only`` and, if *eta* is provided,
    ``condition_schur`` and ``coupling_norm``.
    """
    th = np.asarray(theta, dtype=np.float64).reshape(-1)
    p_opt = th.size

    if eta is not None:
        et = np.asarray(eta, dtype=np.float64).reshape(-1)
        p_pose = et.size

        def full_res(x):
            return residual_function(x[:p_opt], x[p_opt:])

        J_full = finite_difference_jacobian(full_res, np.concatenate([th, et]), step=step)
        J_opt = J_full[:, :p_opt]
        J_pose = J_full[:, p_opt:]
        s_full = np.linalg.svd(J_full, compute_uv=False)
        I_schur = compute_schur_information(J_opt, J_pose)
        eigvals, _ = np.linalg.eigh(I_schur)
        s_schur = np.sqrt(np.maximum(eigvals, 0.0))
        s_schur = np.sort(s_schur)[::-1]

        coupling = float(np.linalg.norm(J_opt.T @ J_pose, "fro"))
        norm_opt = float(np.linalg.norm(J_opt, "fro"))
        norm_pose = float(np.linalg.norm(J_pose, "fro"))
        coupling_norm = coupling / max(norm_opt * norm_pose, 1e-30)

        return {
            "condition_optics_only": _condition_number(s_full[:p_opt]) if p_opt > 0 else 0.0,
            "condition_full": _condition_number(s_full) if s_full.size > 0 else 0.0,
            "condition_schur": _condition_number(s_schur) if s_schur.size > 0 else 0.0,
            "coupling_norm": coupling_norm,
            "rank_full": _numerical_rank(s_full) if s_full.size > 0 else 0,
            "p_optics": p_opt,
            "p_poses": p_pose,
        }
    else:
        # Pipeline B: no pose parameters
        def opt_res(x):
            return residual_function(x, None)

        J_opt = finite_difference_jacobian(opt_res, th, step=step)
        s_opt = np.linalg.svd(J_opt, compute_uv=False) if p_opt > 0 else np.zeros(0)
        return {
            "condition_optics_only": _condition_number(s_opt) if s_opt.size > 0 else 0.0,
            "condition_full": _condition_number(s_opt) if s_opt.size > 0 else 0.0,
            "condition_schur": _condition_number(s_opt) if s_opt.size > 0 else 0.0,
            "coupling_norm": 0.0,
            "rank_full": _numerical_rank(s_opt) if s_opt.size > 0 else 0,
            "p_optics": p_opt,
            "p_poses": 0,
        }
