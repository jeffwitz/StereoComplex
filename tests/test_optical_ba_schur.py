"""Tests for the Schur-based observability diagnostic.

Mirrors the unit-test list of CdC_BA_optique_Schur_CMO_Pycaso.md §9.1.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from stereocomplex.optical_ba.fisher import (
    build_fisher_blocks,
    finite_difference_jacobian_scaled,
)
from stereocomplex.optical_ba.schur import (
    coupling_norm_schur,
    diagnose_schur_modes,
    schur_complement_theta,
)

# --- Schur complement ----------------------------------------------------


def test_schur_complement_is_symmetric() -> None:
    """CDC §9.1 (1): symmetric input → symmetric Schur complement."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    I_tt = A @ A.T + 0.1 * np.eye(5)
    B = rng.standard_normal((5, 3))
    I_tp = B
    C = rng.standard_normal((3, 3))
    I_pp = C @ C.T + 0.5 * np.eye(3)

    S = schur_complement_theta(I_tt, I_tp, I_pp, damping_pose=0.0)

    assert S.shape == (5, 5)
    assert np.allclose(S, S.T, atol=1e-10)


def test_schur_zero_coupling_recovers_optical_block() -> None:
    """CDC §9.1 (2): I_tp = 0 ⇒ S_theta = I_tt exactly."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    I_tt = A @ A.T + 0.5 * np.eye(4)
    I_tp = np.zeros((4, 6))
    C = rng.standard_normal((6, 6))
    I_pp = C @ C.T + 0.5 * np.eye(6)

    S = schur_complement_theta(I_tt, I_tp, I_pp, damping_pose=0.0)

    # Schur reduces to the optical block; symmetrisation does not change it.
    assert np.allclose(S, I_tt, atol=1e-12)


def test_strong_coupling_lowers_weak_schur_eigenvalues() -> None:
    """CDC §9.1 (3): scaling up the cross block shrinks weak modes of S.

    Constructed example: ``I_tt = diag(1, 1)``, ``I_pp = diag(1, 1)``,
    ``I_tp = alpha * I``. Schur becomes ``diag(1 - alpha^2, 1 - alpha^2)``,
    so eigenvalues monotonically decrease with ``|alpha|``.
    """
    I_tt = np.eye(2)
    I_pp = np.eye(2)

    eigvals_by_alpha = []
    for alpha in (0.0, 0.3, 0.6, 0.9):
        I_tp = alpha * np.eye(2)
        S = schur_complement_theta(I_tt, I_tp, I_pp, damping_pose=0.0)
        eigvals = np.sort(np.linalg.eigvalsh(S))
        eigvals_by_alpha.append(eigvals)

    # Smallest eigenvalue strictly decreases as alpha grows.
    smallest = [v[0] for v in eigvals_by_alpha]
    assert smallest[0] > smallest[1] > smallest[2] > smallest[3]
    # Numerical sanity vs. analytical 1 - alpha^2.
    for alpha, eigvals in zip((0.0, 0.3, 0.6, 0.9), eigvals_by_alpha, strict=True):
        assert np.allclose(eigvals, 1.0 - alpha**2, atol=1e-10)


def test_schur_complement_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        schur_complement_theta(np.zeros((3, 4)), np.zeros((3, 2)), np.eye(2))
    with pytest.raises(ValueError, match="square"):
        schur_complement_theta(np.eye(3), np.zeros((3, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        schur_complement_theta(np.eye(3), np.zeros((4, 2)), np.eye(2))


# --- Coupling norm -------------------------------------------------------


def test_coupling_norm_zero_when_blocks_decoupled() -> None:
    I_tt = np.eye(4)
    I_tp = np.zeros((4, 3))
    I_pp = np.eye(3)
    assert coupling_norm_schur(I_tt, I_tp, I_pp, damping_pose=0.0) == 0.0


def test_coupling_norm_increases_with_cross_block_norm() -> None:
    """A larger cross block ⇒ a larger coupling norm at fixed I_tt, I_pp."""
    I_tt = np.eye(3)
    I_pp = np.eye(4)
    values = [
        coupling_norm_schur(I_tt, alpha * np.ones((3, 4)), I_pp, damping_pose=0.0)
        for alpha in (0.0, 0.1, 0.5, 1.0)
    ]
    # Strictly increasing.
    for a, b in itertools.pairwise(values):
        assert a < b


# --- Diagnose modes ------------------------------------------------------


def test_diagnose_marks_weak_modes_below_threshold() -> None:
    # Construct a Schur complement directly by choosing I_tt diagonal,
    # I_tp = 0 (so S = I_tt). Eigenvalues are then exactly the diagonal.
    I_tt = np.diag([10.0, 5.0, 1e-2, 1e-4])
    I_tp = np.zeros((4, 3))
    I_pp = np.eye(3)

    diag = diagnose_schur_modes(I_tt, I_tp, I_pp, weak_threshold=1e-3, damping_pose=0.0)

    # Sorted descending: 10, 5, 1e-2, 1e-4. Threshold 1e-3 vs lambda_max=10
    # → weak iff < 1e-2. Only 1e-4 qualifies.
    assert np.allclose(diag.eigvals, [10.0, 5.0, 1e-2, 1e-4])
    assert diag.weak_mode_indices.tolist() == [3]
    assert diag.rank_effective == 3


def test_diagnose_flags_all_weak_when_optical_block_is_zero() -> None:
    n = 3
    diag = diagnose_schur_modes(
        np.zeros((n, n)), np.zeros((n, 2)), np.eye(2), weak_threshold=1e-3
    )
    assert diag.weak_mode_indices.tolist() == list(range(n))
    assert diag.rank_effective == 0


# --- Fisher / scaled Jacobian -------------------------------------------


def test_fd_jacobian_matches_analytic_on_quadratic() -> None:
    """``r(x) = A x`` has constant Jacobian ``A``; scaled FD recovers it."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 4))

    def fun(x: np.ndarray) -> np.ndarray:
        return A @ x

    x0 = rng.standard_normal(4)
    scales = np.array([1.0, 2.0, 5.0, 0.1])

    J = finite_difference_jacobian_scaled(fun, x0, scales, rel_step=1e-6)

    # column i should be s_i * A[:, i]
    expected = A * scales[None, :]
    assert np.allclose(J, expected, atol=1e-6)


def test_fd_jacobian_rejects_nonpositive_scales() -> None:
    def fun(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="strictly positive"):
        finite_difference_jacobian_scaled(fun, np.zeros(3), np.array([1.0, 0.0, 1.0]))


def test_build_fisher_blocks_shapes_and_symmetry() -> None:
    rng = np.random.default_rng(3)
    n_theta, n_eta, n_res = 4, 5, 12
    A = rng.standard_normal((n_res, n_theta + n_eta))

    def fun(x: np.ndarray) -> np.ndarray:
        return A @ x

    fisher = build_fisher_blocks(
        residual_fun=fun,
        theta0=np.zeros(n_theta),
        pose0=np.zeros(n_eta),
        theta_scales=np.ones(n_theta),
        pose_scales=np.ones(n_eta),
    )

    assert fisher.I_tt.shape == (n_theta, n_theta)
    assert fisher.I_tp.shape == (n_theta, n_eta)
    assert fisher.I_pp.shape == (n_eta, n_eta)
    assert fisher.J.shape == (n_res, n_theta + n_eta)
    # J^T J is symmetric (modulo round-off).
    assert np.allclose(fisher.I_tt, fisher.I_tt.T, atol=1e-10)
    assert np.allclose(fisher.I_pp, fisher.I_pp.T, atol=1e-10)
    assert np.allclose(fisher.I_pt, fisher.I_tp.T, atol=1e-12)


def test_build_fisher_blocks_honours_robust_weights() -> None:
    rng = np.random.default_rng(4)
    n_theta, n_eta, n_res = 3, 4, 10
    A = rng.standard_normal((n_res, n_theta + n_eta))

    def fun(x: np.ndarray) -> np.ndarray:
        return A @ x

    w = rng.uniform(0.1, 1.0, size=n_res)
    fisher = build_fisher_blocks(
        residual_fun=fun,
        theta0=np.zeros(n_theta),
        pose0=np.zeros(n_eta),
        theta_scales=np.ones(n_theta),
        pose_scales=np.ones(n_eta),
        robust_weights=w,
    )

    expected = A.T @ np.diag(w) @ A
    full = np.block(
        [
            [fisher.I_tt, fisher.I_tp],
            [fisher.I_pt, fisher.I_pp],
        ]
    )
    assert np.allclose(full, expected, atol=1e-8)
