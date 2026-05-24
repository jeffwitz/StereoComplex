"""Smoke tests for inverse-problem diagnostics."""

from __future__ import annotations

import numpy as np

from stereocomplex.benchmarks.inverse_problem_diagnostics import (
    InverseProblemDiagnostics,
    compute_inverse_problem_diagnostics,
    compute_schur_information,
    finite_difference_jacobian,
)


def _simple_residual(x: np.ndarray) -> np.ndarray:
    """r(x) = [x[0] + x[1], x[0] - x[1]]."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.array([x_arr[0] + x_arr[1], x_arr[0] - x_arr[1]], dtype=np.float64)


def test_finite_difference_jacobian_exact_for_linear():
    J = finite_difference_jacobian(_simple_residual, np.zeros(2), step=1e-6)
    expected = np.array([[1.0, 1.0], [1.0, -1.0]])
    assert np.allclose(J, expected, atol=1e-5)


def test_schur_complement_is_psd():
    """The Schur information matrix must be positive semi-definite."""
    J_opt = np.random.default_rng(42).normal(size=(50, 3))
    J_pose = np.random.default_rng(43).normal(size=(50, 6))
    I_schur = compute_schur_information(J_opt, J_pose)
    eigvals = np.linalg.eigvalsh(I_schur)
    assert np.all(eigvals >= -1e-10)


def test_diagnostics_on_simple_residual():
    """Run the full diagnostics on a toy residual with known structure."""

    # Residual: r(theta, eta) = [sin(theta + eta), cos(theta - eta)]
    def residual(theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
        t = float(theta[0])
        e = float(eta[0])
        return np.array([np.sin(t + e), np.cos(t - e)], dtype=np.float64)

    diag = compute_inverse_problem_diagnostics(
        residual, np.array([0.1]), np.array([0.2]), step=1e-6,
    )
    assert isinstance(diag, InverseProblemDiagnostics)
    assert diag.singular_values_full.size > 0
    assert diag.condition_full > 0
    assert 0.0 <= diag.optical_pose_coupling_norm <= 1.0


def test_pipeline_diagnostics_pose_params_zero_for_rayfield():
    """Pipeline B (rayfield) should report p_poses=0."""
    from stereocomplex.benchmarks.inverse_problem_diagnostics import (
        compute_pipeline_condition_number,
    )

    def rayfield_residual(theta, eta):
        return theta  # trivial

    result = compute_pipeline_condition_number(
        rayfield_residual, np.array([1.0, 2.0]), eta=None,
    )
    assert result["p_poses"] == 0
    assert result["coupling_norm"] == 0.0


def test_pipeline_A_has_higher_condition_number_than_B():
    """On a coupled residual, pipeline A should have higher condition number."""
    from stereocomplex.benchmarks.inverse_problem_diagnostics import (
        compute_pipeline_condition_number,
    )

    # Residual with strong optics-pose coupling
    def coupled_residual(theta, eta=None):
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if eta is not None:
            eta = np.asarray(eta, dtype=np.float64).reshape(-1)
            return np.array([theta[0] + eta[0], theta[0] - eta[0]*0.99])
        return np.array([theta[0], theta[0]])

    result_A = compute_pipeline_condition_number(
        coupled_residual, np.array([1.0]), eta=np.array([1.0]),
    )
    result_B = compute_pipeline_condition_number(
        coupled_residual, np.array([1.0]), eta=None,
    )
    # Pipeline A with poses should have higher (worse) condition number
    # due to optics-pose coupling
    assert result_A["condition_schur"] >= result_B["condition_optics_only"]
    assert result_A["coupling_norm"] > 0.0
