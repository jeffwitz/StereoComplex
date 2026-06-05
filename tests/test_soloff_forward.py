"""Tests for the true Soloff forward model (forward fit + LM inversion).

The :class:`SoloffForwardModel` is a dependency-free reimplementation of the
Soloff calibration method (Soloff et al. 1997): fit the forward projection
``pixels = A . M(x, y, z)`` then recover ``(x, y, z)`` by batched
Levenberg--Marquardt inversion. These tests use a fully synthetic pinhole
stereo rig (no external data) so the geometry is known exactly.
"""

from __future__ import annotations

import numpy as np

from stereocomplex.eval.soloff_poly import (
    SoloffForwardModel,
    _soloff_monomial_powers,
)


def _pinhole(xyz: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3-D points through a pinhole camera to (u, v) pixels."""
    cam = xyz @ R.T + t[None, :]
    proj = cam @ K.T
    return proj[:, :2] / proj[:, 2:3]


def _synthetic_stereo(n: int, rng: np.random.Generator):
    """Random 3-D cloud + its left/right pinhole projections."""
    K = np.array([[1500.0, 0.0, 512.0], [0.0, 1500.0, 512.0], [0.0, 0.0, 1.0]])
    R_l = np.eye(3)
    t_l = np.array([0.0, 0.0, 60.0])
    ang = np.radians(8.0)
    R_r = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    t_r = np.array([12.0, 0.0, 60.0])
    xyz = np.column_stack([
        rng.uniform(-5, 5, n),
        rng.uniform(-4, 4, n),
        rng.uniform(-1.5, 1.5, n),
    ])
    return xyz, _pinhole(xyz, K, R_l, t_l), _pinhole(xyz, K, R_r, t_r)


def test_monomial_counts():
    """Form codes expand to the Pycaso-documented monomial counts."""
    assert len(_soloff_monomial_powers(111)) == 4
    assert len(_soloff_monomial_powers(222)) == 10
    assert len(_soloff_monomial_powers(332)) == 19
    assert len(_soloff_monomial_powers(333)) == 20


def test_forward_fit_reprojects():
    """A degree-3 Soloff fits a smooth pinhole projection to sub-pixel error."""
    rng = np.random.default_rng(0)
    xyz, uvl, uvr = _synthetic_stereo(2000, rng)
    model = SoloffForwardModel.fit(uvl, uvr, xyz, form=332)
    pred = model.project(xyz)
    rms = np.sqrt(np.mean(np.sum((pred - np.column_stack([uvl, uvr])) ** 2, axis=1)))
    assert rms < 0.05  # px


def test_lm_inversion_recovers_points():
    """fit -> identify round-trips held-out points to micron precision."""
    rng = np.random.default_rng(1)
    xyz, uvl, uvr = _synthetic_stereo(3000, rng)
    model = SoloffForwardModel.fit(uvl, uvr, xyz, form=332)

    # Held-out points well inside the calibrated volume (unique inverse).
    xt = np.column_stack([
        rng.uniform(-4, 4, 1500),
        rng.uniform(-3, 3, 1500),
        rng.uniform(-1.0, 1.0, 1500),
    ])
    obs = model.project(xt)
    rec = model.identify(obs[:, :2], obs[:, 2:])
    err = np.linalg.norm(rec - xt, axis=1)
    assert np.median(err) < 1e-3  # mm
    assert np.percentile(err, 95) < 1e-2  # mm
