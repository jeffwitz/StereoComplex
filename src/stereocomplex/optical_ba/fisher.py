"""Scaled, block-partitioned Fisher information for the BA problem.

Implements the Gauss-Newton approximation of the BA Hessian used by the
Schur-based diagnostic of ``CdC_BA_optique_Schur_CMO_Pycaso.md``, §3.

Given a residual function ``r(x)`` and a parameter vector ``x = [theta, eta]``
split into optical and pose blocks, the Fisher matrix at ``x_0`` is

.. math::

    \\mathcal{I} = J^{T} W J,
    \\qquad
    J = \\frac{\\partial r}{\\partial \\tilde x}\\Big|_{x_0},

where :math:`\\tilde x_i = x_i / s_i` is the parameter in normalised units —
the scaling ``s`` is essential because the Schur complement and its
condition number depend strongly on it (CDC §11.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FisherBlocks:
    """Output of :func:`build_fisher_blocks`.

    All Fisher blocks are expressed in the normalised parameter space — i.e.
    the columns of ``J`` correspond to ``s_i * dr/dx_i``. To translate a
    direction back to physical units, multiply by ``parameter_scales_*``.

    Attributes
    ----------
    I_tt : ndarray, shape (n_theta, n_theta)
        Optical / optical block.
    I_tp : ndarray, shape (n_theta, n_eta)
        Optical / pose block.
    I_pt : ndarray, shape (n_eta, n_theta)
        Pose / optical block (``I_tp.T``); kept for convenience.
    I_pp : ndarray, shape (n_eta, n_eta)
        Pose / pose block.
    J : ndarray, shape (n_residuals, n_theta + n_eta)
        Full scaled Jacobian; first ``n_theta`` columns are optical.
    residuals : ndarray, shape (n_residuals,)
        Residual vector evaluated at the linearisation point.
    parameter_scales_theta : ndarray, shape (n_theta,)
        Scale ``s`` applied to the optical parameters.
    parameter_scales_pose : ndarray, shape (n_eta,)
        Scale ``s`` applied to the pose parameters.
    """

    I_tt: np.ndarray
    I_tp: np.ndarray
    I_pt: np.ndarray
    I_pp: np.ndarray
    J: np.ndarray
    residuals: np.ndarray
    parameter_scales_theta: np.ndarray
    parameter_scales_pose: np.ndarray


def finite_difference_jacobian_scaled(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    scales: np.ndarray,
    *,
    rel_step: float = 1e-6,
    method: str = "central",
) -> np.ndarray:
    """Finite-difference Jacobian of ``fun`` in scaled parameter space.

    Computes columns ``J[:, i] = s_i * df/dx_i`` using either forward or
    central differences with step ``s_i * rel_step``. Working in scaled
    space makes the resulting Jacobian dimensionless along each column,
    which is exactly what the Schur diagnostic needs to compare modes on a
    common footing.

    Parameters
    ----------
    fun : callable
        Residual function ``x -> r(x)``; must accept a 1-D ndarray and
        return a 1-D ndarray. Called ``1 + n`` times for ``method="forward"``
        and ``2 * n`` times for ``method="central"`` (plus the baseline).
    x0 : ndarray, shape (n,)
        Linearisation point in raw (un-scaled) units.
    scales : ndarray, shape (n,)
        Per-parameter scale; must be strictly positive.
    rel_step : float
        Relative step size in the scaled space; the raw step on parameter
        ``i`` is ``rel_step * scales[i]``.
    method : {"central", "forward"}
        Difference scheme.

    Returns
    -------
    ndarray, shape (n_residuals, n)
        Scaled Jacobian.

    Raises
    ------
    ValueError
        If ``scales`` is non-positive or has the wrong shape, or if
        ``method`` is not recognised.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    if scales.shape != x0.shape:
        raise ValueError("scales must have the same shape as x0")
    if np.any(scales <= 0.0):
        raise ValueError("scales must be strictly positive")
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    n = x0.size
    r0 = np.asarray(fun(x0), dtype=np.float64).reshape(-1)
    n_res = r0.size
    J = np.empty((n_res, n), dtype=np.float64)

    for i in range(n):
        h = float(rel_step) * float(scales[i])
        if method == "central":
            x_plus = x0.copy()
            x_plus[i] += h
            x_minus = x0.copy()
            x_minus[i] -= h
            r_plus = np.asarray(fun(x_plus), dtype=np.float64).reshape(-1)
            r_minus = np.asarray(fun(x_minus), dtype=np.float64).reshape(-1)
            # column i = s_i * dr/dx_i = s_i * (r(+) - r(-)) / (2 h)
            #         = s_i * (r(+) - r(-)) / (2 * rel_step * s_i)
            #         = (r(+) - r(-)) / (2 * rel_step)
            J[:, i] = (r_plus - r_minus) / (2.0 * float(rel_step))
        else:
            x_plus = x0.copy()
            x_plus[i] += h
            r_plus = np.asarray(fun(x_plus), dtype=np.float64).reshape(-1)
            J[:, i] = (r_plus - r0) / float(rel_step)

    return J


def build_fisher_blocks(
    *,
    residual_fun: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    pose0: np.ndarray,
    theta_scales: np.ndarray,
    pose_scales: np.ndarray,
    robust_weights: np.ndarray | None = None,
    rel_step: float = 1e-6,
    method: str = "central",
) -> FisherBlocks:
    """Build a scaled, block-partitioned Fisher matrix at ``(theta0, pose0)``.

    Concatenates ``[theta0, pose0]`` into a single parameter vector, computes
    the scaled finite-difference Jacobian of ``residual_fun``, optionally
    re-weights each residual by ``sqrt(robust_weights)``, then returns
    :math:`\\mathcal{I} = J^{T} W J` partitioned into optical / pose blocks.

    Parameters
    ----------
    residual_fun : callable
        Maps the concatenated parameter vector ``x`` (of length
        ``n_theta + n_eta``) to the residual vector ``r(x)``. The
        ``n_theta`` optical components must come first.
    theta0 : ndarray, shape (n_theta,)
        Optical parameters at the linearisation point.
    pose0 : ndarray, shape (n_eta,)
        Pose parameters at the linearisation point.
    theta_scales : ndarray, shape (n_theta,)
        Per-parameter optical scales (CDC §4.3 / §11.1). All strictly
        positive.
    pose_scales : ndarray, shape (n_eta,)
        Per-parameter pose scales. All strictly positive.
    robust_weights : ndarray, shape (n_residuals,) or None
        Optional non-negative weights, one per residual; if given,
        :math:`\\mathcal{I} = J^{T} \\operatorname{diag}(w) J`. Pass the
        Huber/soft-L1 weights of the underlying robust loss to obtain the
        Fisher of the *robustified* problem.
    rel_step : float
        Relative step in the scaled space, as in
        :func:`finite_difference_jacobian_scaled`.
    method : {"central", "forward"}
        Finite-difference scheme.

    Returns
    -------
    FisherBlocks
        Optical / pose Fisher partition plus the Jacobian, residuals, and
        scales used to build them.
    """
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(-1)
    theta_scales = np.asarray(theta_scales, dtype=np.float64).reshape(-1)
    pose_scales = np.asarray(pose_scales, dtype=np.float64).reshape(-1)
    if theta_scales.shape != theta0.shape:
        raise ValueError("theta_scales must have the same shape as theta0")
    if pose_scales.shape != pose0.shape:
        raise ValueError("pose_scales must have the same shape as pose0")

    x0 = np.concatenate([theta0, pose0])
    scales = np.concatenate([theta_scales, pose_scales])

    J = finite_difference_jacobian_scaled(
        residual_fun, x0, scales, rel_step=rel_step, method=method
    )
    r0 = np.asarray(residual_fun(x0), dtype=np.float64).reshape(-1)

    if robust_weights is not None:
        w = np.asarray(robust_weights, dtype=np.float64).reshape(-1)
        if w.shape != (J.shape[0],):
            raise ValueError("robust_weights must have one entry per residual")
        if np.any(w < 0.0):
            raise ValueError("robust_weights must be non-negative")
        Jw = J * np.sqrt(w)[:, None]
        I_full = Jw.T @ Jw
    else:
        I_full = J.T @ J

    n_theta = theta0.size
    I_tt = I_full[:n_theta, :n_theta]
    I_tp = I_full[:n_theta, n_theta:]
    I_pp = I_full[n_theta:, n_theta:]
    return FisherBlocks(
        I_tt=I_tt,
        I_tp=I_tp,
        I_pt=I_tp.T,
        I_pp=I_pp,
        J=J,
        residuals=r0,
        parameter_scales_theta=theta_scales,
        parameter_scales_pose=pose_scales,
    )
