"""Schur complement of the BA Fisher with respect to the pose block.

Implements the observability-aware diagnostic of
``CdC_BA_optique_Schur_CMO_Pycaso.md``, §3 and §4.

Notation. The full Fisher matrix of the bundle-adjustment problem is split
block-wise into optical (``theta``) and pose (``eta``) parameters,

.. math::

    \\mathcal{I} =
    \\begin{bmatrix}
    \\mathcal{I}_{\\theta\\theta} & \\mathcal{I}_{\\theta\\eta} \\\\
    \\mathcal{I}_{\\eta\\theta}    & \\mathcal{I}_{\\eta\\eta}
    \\end{bmatrix},

and the Schur complement on the optical block,

.. math::

    S_\\theta
    = \\mathcal{I}_{\\theta\\theta}
    - \\mathcal{I}_{\\theta\\eta}\\,\\mathcal{I}_{\\eta\\eta}^{-1}\\,
      \\mathcal{I}_{\\eta\\theta},

is the information remaining on ``theta`` once the poses ``eta`` are
marginalised out. Eigen-directions of :math:`S_\\theta` with vanishing
eigenvalue are optical directions that cannot be told apart from a pose
adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SchurDiagnostic:
    """Result of :func:`diagnose_schur_modes`.

    Attributes
    ----------
    S_theta : ndarray, shape (n_theta, n_theta)
        Schur complement of the Fisher matrix on the optical block.
    eigvals : ndarray, shape (n_theta,)
        Eigenvalues of ``S_theta``, sorted in descending order.
    eigvecs : ndarray, shape (n_theta, n_theta)
        Eigenvectors of ``S_theta``, columns aligned with ``eigvals``.
    coupling_norm : float
        Normalised pose / optics coupling indicator
        :math:`\\|\\mathcal{I}_{\\theta\\eta}\\mathcal{I}_{\\eta\\eta}^{-1}\\mathcal{I}_{\\eta\\theta}\\|_F /
        \\|\\mathcal{I}_{\\theta\\theta}\\|_F`. Scale-dependent: only meaningful
        when compared at fixed ``D_theta`` parameter scales.
    weak_mode_indices : ndarray of int
        Indices into ``eigvals`` of the weak modes (eigenvalues smaller than
        ``weak_threshold * eigvals[0]``).
    condition_number : float
        ``eigvals[0] / max(eigvals[-1], tiny)``; high values flag mass in the
        null space of the marginal information.
    rank_effective : int
        Number of eigenvalues above the weak threshold.
    """

    S_theta: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    coupling_norm: float
    weak_mode_indices: np.ndarray
    condition_number: float
    rank_effective: int


def _solve_pose(
    I_pp: np.ndarray,
    rhs: np.ndarray,
    *,
    damping_pose: float,
    pinv_rcond: float,
) -> np.ndarray:
    """Solve ``(I_pp + damping_pose * I) @ X = rhs`` robustly.

    Uses the Cholesky factor when the damped pose block is positive definite,
    and falls back to a truncated SVD pseudo-inverse otherwise. The damping
    handles a near-rank-deficient pose block (gauge directions, very few
    observations on a frame), the pseudo-inverse handles outright
    indefiniteness from numerical noise on a near-singular Hessian.
    """
    n = I_pp.shape[0]
    I_pp_d = I_pp + float(damping_pose) * np.eye(n, dtype=np.float64)
    try:
        return np.linalg.solve(I_pp_d, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(I_pp_d, rcond=float(pinv_rcond)) @ rhs


def schur_complement_theta(
    I_tt: np.ndarray,
    I_tp: np.ndarray,
    I_pp: np.ndarray,
    *,
    damping_pose: float = 1e-8,
    pinv_rcond: float = 1e-10,
) -> np.ndarray:
    """Return the Schur complement of the Fisher matrix on the optical block.

    Computes :math:`S_\\theta = \\mathcal{I}_{\\theta\\theta}
    - \\mathcal{I}_{\\theta\\eta}\\,(\\mathcal{I}_{\\eta\\eta} +
    \\lambda I)^{-1}\\,\\mathcal{I}_{\\eta\\theta}`, with a small Tikhonov
    damping ``lambda = damping_pose`` on the pose block to absorb gauge
    directions / near-singularities. The result is symmetrised explicitly to
    guard against round-off.

    Parameters
    ----------
    I_tt : ndarray, shape (n_theta, n_theta)
        Optical-block Fisher information.
    I_tp : ndarray, shape (n_theta, n_eta)
        Cross-block (theta-eta) Fisher information.
    I_pp : ndarray, shape (n_eta, n_eta)
        Pose-block Fisher information.
    damping_pose : float
        Tikhonov damping added to the pose block before inversion. Set to
        zero only when ``I_pp`` is known to be well-conditioned.
    pinv_rcond : float
        Relative cut-off used by the SVD pseudo-inverse fallback when the
        damped pose block still cannot be Cholesky-factored.

    Returns
    -------
    ndarray, shape (n_theta, n_theta)
        The Schur complement :math:`S_\\theta`, symmetric.
    """
    I_tt = np.asarray(I_tt, dtype=np.float64)
    I_tp = np.asarray(I_tp, dtype=np.float64)
    I_pp = np.asarray(I_pp, dtype=np.float64)
    if I_tt.shape[0] != I_tt.shape[1]:
        raise ValueError("I_tt must be square")
    if I_pp.shape[0] != I_pp.shape[1]:
        raise ValueError("I_pp must be square")
    if I_tp.shape != (I_tt.shape[0], I_pp.shape[0]):
        raise ValueError("I_tp must have shape (n_theta, n_eta)")

    X = _solve_pose(I_pp, I_tp.T, damping_pose=damping_pose, pinv_rcond=pinv_rcond)
    S = I_tt - I_tp @ X
    return 0.5 * (S + S.T)


def coupling_norm_schur(
    I_tt: np.ndarray,
    I_tp: np.ndarray,
    I_pp: np.ndarray,
    *,
    damping_pose: float = 1e-8,
    pinv_rcond: float = 1e-10,
) -> float:
    """Normalised pose / optics coupling indicator (CDC §3.4).

    Returns

    .. math::

        c = \\frac{
        \\bigl\\|\\mathcal{I}_{\\theta\\eta}\\,\\mathcal{I}_{\\eta\\eta}^{-1}\\,
        \\mathcal{I}_{\\eta\\theta}\\bigr\\|_F
        }{
        \\bigl\\|\\mathcal{I}_{\\theta\\theta}\\bigr\\|_F
        }.

    Interpretation: ``c`` close to zero means the optical block carries most
    of its apparent information independently of the poses; ``c`` close to
    one (or larger, numerically) means most of that information is washed
    out once the poses are marginalised.

    **Scale dependence.** ``c`` depends on the parameter scales used for
    ``theta`` and ``eta``. It is meaningful as a diagnostic comparison
    between two runs that use the *same* scale matrix, not as an absolute
    physical quantity.

    Parameters
    ----------
    I_tt, I_tp, I_pp : ndarray
        Same conventions as :func:`schur_complement_theta`.
    damping_pose, pinv_rcond : float
        Same conventions as :func:`schur_complement_theta`.

    Returns
    -------
    float
        The Frobenius-norm coupling ratio. Returns ``nan`` when
        ``I_tt`` is zero (no optical information at all).
    """
    I_tt = np.asarray(I_tt, dtype=np.float64)
    I_tp = np.asarray(I_tp, dtype=np.float64)
    I_pp = np.asarray(I_pp, dtype=np.float64)

    X = _solve_pose(I_pp, I_tp.T, damping_pose=damping_pose, pinv_rcond=pinv_rcond)
    cross = I_tp @ X
    denom = float(np.linalg.norm(I_tt, ord="fro"))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(cross, ord="fro") / denom)


def diagnose_schur_modes(
    I_tt: np.ndarray,
    I_tp: np.ndarray,
    I_pp: np.ndarray,
    *,
    weak_threshold: float = 1e-3,
    damping_pose: float = 1e-8,
    pinv_rcond: float = 1e-10,
) -> SchurDiagnostic:
    """Eigen-decompose the optical Schur complement and flag its weak modes.

    Computes :math:`S_\\theta` via :func:`schur_complement_theta`,
    diagonalises it with the Hermitian eigendecomposition, sorts eigenvalues
    in descending order, and marks as "weak" any eigenvalue smaller than
    ``weak_threshold * eigvals[0]``. Weak directions are the optical
    directions that survive the BA pose marginalisation only thanks to the
    Tikhonov damping — physically, they are unidentifiable from the data
    alone.

    Parameters
    ----------
    I_tt, I_tp, I_pp : ndarray
        Fisher blocks; see :func:`schur_complement_theta`.
    weak_threshold : float
        A mode is "weak" when its eigenvalue, divided by the largest
        eigenvalue, falls below this threshold.
    damping_pose, pinv_rcond : float
        Passed through to :func:`schur_complement_theta`.

    Returns
    -------
    SchurDiagnostic
        Schur complement, eigenpair, coupling norm, weak-mode indices,
        condition number and effective rank.
    """
    S = schur_complement_theta(
        I_tt, I_tp, I_pp, damping_pose=damping_pose, pinv_rcond=pinv_rcond
    )
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lambda_max = float(eigvals[0]) if eigvals.size else 0.0
    if lambda_max <= 0.0:
        # No identifiable direction at all — flag every mode as weak.
        weak_idx = np.arange(eigvals.size, dtype=np.int64)
        rank_eff = 0
        cond = float("inf")
    else:
        weak_idx = np.where(eigvals < weak_threshold * lambda_max)[0].astype(np.int64)
        rank_eff = int(np.sum(eigvals >= weak_threshold * lambda_max))
        lambda_min_eff = max(float(eigvals[-1]), np.finfo(np.float64).tiny)
        cond = lambda_max / lambda_min_eff

    coupling = coupling_norm_schur(
        I_tt, I_tp, I_pp, damping_pose=damping_pose, pinv_rcond=pinv_rcond
    )

    return SchurDiagnostic(
        S_theta=S,
        eigvals=eigvals,
        eigvecs=eigvecs,
        coupling_norm=coupling,
        weak_mode_indices=weak_idx,
        condition_number=cond,
        rank_effective=rank_eff,
    )
