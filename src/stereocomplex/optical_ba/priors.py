"""Schur-based and isotropic regularisation priors for optical BA.

Implements CDC §4 : prior residuals that penalise displacement of the
optical parameter vector *theta* away from the rayfield-identified
initial value, weighted by the eigen-spectrum of the Schur complement
so that poorly observable directions are penalised more heavily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SchurPrior:
    """Regularisation prior derived from a Schur diagnostic at *theta0*.

    The prior penalises the normalised displacement
    ``D_theta^{-1} (theta - theta0)`` projected onto each eigenmode of
    the Schur complement ``S_theta``.  Mode *i* receives weight
    ``w_i = (lambda_max / (lambda_i + epsilon * lambda_max))^power``
    (CDC §4.3, continuous variant).

    Attributes
    ----------
    theta0 : ndarray, shape (n_theta,)
        Rayfield-identified optical parameter vector used as the
        regularisation centre.
    eigvals : ndarray, shape (n_theta,)
        Eigenvalues of ``S_theta``, sorted descending.
    eigvecs : ndarray, shape (n_theta, n_theta)
        Eigenvectors of ``S_theta`` (columns).
    theta_scales : ndarray, shape (n_theta,)
        Per-parameter scales for the normalised displacement
        ``delta = D_theta^{-1} (theta - theta0)``.
    alpha : float
        Overall prior strength.
    epsilon : float
        Regularisation floor for the eigenvalue weight
        (default 1e-6).
    power : float
        Exponent for the eigenvalue weight (1.0 = moderate,
        2.0 = aggressive).
    weak_only : bool
        If ``True``, only penalise modes whose relative eigenvalue
        ``lambda_i / lambda_max < weak_threshold``.
    weak_threshold : float
        Relative eigenvalue threshold for ``weak_only`` mode.
    """

    theta0: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    theta_scales: np.ndarray
    alpha: float
    epsilon: float = 1e-6
    power: float = 1.0
    weak_only: bool = False
    weak_threshold: float = 1e-3


def schur_prior_residuals(
    theta: np.ndarray,
    prior: SchurPrior,
) -> np.ndarray:
    """Extra residual rows encoding the Schur-based regularisation.

    The residual for mode *i* is ::

        r_i = sqrt(alpha * w_i) * (v_i^T @ delta_theta)

    where ``delta_theta = D_theta^{-1} (theta - theta0)`` is the scaled
    optical displacement and ``w_i`` is the eigenvalue-derived weight.

    Parameters
    ----------
    theta : ndarray, shape (n_theta,)
        Current optical parameter vector.
    prior : SchurPrior
        Prior specification built from the "before" Schur diagnostic.

    Returns
    -------
    ndarray, shape (n_theta,)
        Prior residuals to append to the data residual vector.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size != prior.theta0.size:
        raise ValueError(
            f"theta has {theta.size} entries, expected {prior.theta0.size}"
        )

    delta = (theta - prior.theta0) / prior.theta_scales
    lam = np.asarray(prior.eigvals, dtype=np.float64)
    lam_max = float(lam[0]) if lam.size else 1.0

    # Eigenvalue-driven weights (CDC §4.3, continuous variant)
    w = (lam_max / (lam + prior.epsilon * lam_max)) ** prior.power

    if prior.weak_only:
        w[lam / max(lam_max, 1e-30) >= prior.weak_threshold] = 0.0

    # Project delta onto each eigenmode
    proj = prior.eigvecs.T @ delta  # shape (n_theta,)

    return np.sqrt(float(prior.alpha) * w) * proj


def isotropic_prior_residuals(
    theta: np.ndarray,
    theta0: np.ndarray,
    theta_scales: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Isotropic (Tikhonov) prior residuals for baseline comparison.

    ``r = sqrt(alpha) * D_theta^{-1} (theta - theta0)``

    Parameters
    ----------
    theta : ndarray, shape (n_theta,)
        Current optical parameter vector.
    theta0 : ndarray, shape (n_theta,)
        Regularisation centre (typically the rayfield solution).
    theta_scales : ndarray, shape (n_theta,)
        Per-parameter scales.
    alpha : float
        Regularisation strength.

    Returns
    -------
    ndarray, shape (n_theta,)
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    theta0 = np.asarray(theta0, dtype=np.float64).reshape(-1)
    scales = np.asarray(theta_scales, dtype=np.float64).reshape(-1)
    delta = (theta - theta0) / scales
    return np.sqrt(float(alpha)) * delta
