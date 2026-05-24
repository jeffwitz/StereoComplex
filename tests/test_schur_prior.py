"""Tests for Schur-based and isotropic regularisation priors."""

from __future__ import annotations

import numpy as np
import pytest

from stereocomplex.optical_ba.priors import (
    SchurPrior,
    isotropic_prior_residuals,
    schur_prior_residuals,
)


def _make_toy_prior(**overrides) -> SchurPrior:
    n = 4
    theta0 = np.array([1.0, 2.0, 3.0, 4.0])
    eigvals = np.array([100.0, 10.0, 1.0, 0.01])
    eigvecs = np.eye(n)
    scales = np.array([1.0, 1.0, 1.0, 1.0])
    kw = dict(
        theta0=theta0, eigvals=eigvals, eigvecs=eigvecs,
        theta_scales=scales, alpha=1.0,
    )
    kw.update(overrides)
    return SchurPrior(**kw)


# ── SchurPrior dataclass ────────────────────────────────────


def test_schur_prior_defaults():
    p = _make_toy_prior()
    assert p.epsilon == 1e-6
    assert p.power == 1.0
    assert not p.weak_only
    assert p.weak_threshold == 1e-3


# ── schur_prior_residuals ───────────────────────────────────


def test_schur_prior_residuals_zero_at_theta0():
    p = _make_toy_prior()
    r = schur_prior_residuals(p.theta0, p)
    np.testing.assert_allclose(r, 0.0, atol=1e-14)


def test_schur_prior_residuals_penalises_displacement():
    p = _make_toy_prior()
    # Displace in a weak mode (eigvecs[3] = [0,0,0,1], weak eigenvalue)
    theta = p.theta0 + np.array([0.0, 0.0, 0.0, 1.0])
    r = schur_prior_residuals(theta, p)
    # The last mode (weakest) should have the largest weighted projection
    assert abs(r[3]) > abs(r[0])
    assert abs(r[3]) > abs(r[1])
    assert abs(r[3]) > abs(r[2])


def test_schur_prior_residuals_strong_mode_penalised_less():
    p = _make_toy_prior()
    # Displace equally in all modes, but strong modes should be penalised less
    theta = p.theta0 + np.ones(4)
    r = schur_prior_residuals(theta, p)
    # Weakest mode (index 3) gets largest weight → largest residual
    assert abs(r[3]) > abs(r[0])


def test_schur_prior_residuals_alpha_scales():
    p1 = _make_toy_prior(alpha=1.0)
    p100 = _make_toy_prior(alpha=100.0)
    theta = p1.theta0 + np.array([0.0, 0.0, 0.0, 1.0])
    r1 = schur_prior_residuals(theta, p1)
    r100 = schur_prior_residuals(theta, p100)
    np.testing.assert_allclose(r100, 10.0 * r1, atol=1e-12)


def test_schur_prior_residuals_power_sharpens_weights():
    p1 = _make_toy_prior(power=1.0)
    p2 = _make_toy_prior(power=2.0)
    theta = p1.theta0 + np.array([0.0, 0.0, 0.0, 1.0])
    r1 = schur_prior_residuals(theta, p1)
    r2 = schur_prior_residuals(theta, p2)
    # power=2 penalises the weakest mode more aggressively
    ratio_1 = abs(r1[3]) / max(abs(r1[0]), 1e-30)
    ratio_2 = abs(r2[3]) / max(abs(r2[0]), 1e-30)
    assert ratio_2 > ratio_1


def test_schur_prior_residuals_weak_only_zeros_strong_modes():
    p = _make_toy_prior(weak_only=True, weak_threshold=0.1)
    theta = p.theta0 + np.ones(4)
    r = schur_prior_residuals(theta, p)
    # Modes 0 and 1 (eigval 100, 10) are above threshold 0.1 → zero weight
    assert abs(r[0]) < 1e-14
    assert abs(r[1]) < 1e-14
    # Modes 2 and 3 (eigval 1, 0.01) are below → penalised
    assert abs(r[2]) > 0
    assert abs(r[3]) > 0


def test_schur_prior_residuals_shape_mismatch_raises():
    p = _make_toy_prior()
    with pytest.raises(ValueError, match="entries"):
        schur_prior_residuals(np.zeros(5), p)


# ── isotropic_prior_residuals ───────────────────────────────


def test_isotropic_prior_residuals_zero_at_theta0():
    r = isotropic_prior_residuals(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]),
        np.array([1.0, 1.0]), alpha=1.0,
    )
    np.testing.assert_allclose(r, 0.0, atol=1e-14)


def test_isotropic_prior_residuals_uniform_penalty():
    r = isotropic_prior_residuals(
        np.array([2.0, 2.0]), np.array([1.0, 1.0]),
        np.array([1.0, 1.0]), alpha=1.0,
    )
    np.testing.assert_allclose(r, [1.0, 1.0])


def test_isotropic_prior_residuals_scales_affect_penalty():
    # A parameter with a larger scale should be penalised less
    r = isotropic_prior_residuals(
        np.array([2.0, 2.0]), np.array([1.0, 1.0]),
        np.array([1.0, 10.0]), alpha=1.0,
    )
    assert abs(r[0]) > abs(r[1])
