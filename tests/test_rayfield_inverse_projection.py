"""Smoke tests for rayfield inverse projection."""

from __future__ import annotations

import numpy as np

from stereocomplex.physics import CentralPinholeModel
from stereocomplex.benchmarks.rayfield_projection import (
    point_ray_residual,
    project_point_by_rayfield_inverse,
    project_points_by_rayfield_inverse,
)


def _pinhole_field():
    K = np.array([[200.0, 0.0, 79.5], [0.0, 200.0, 59.5], [0.0, 0.0, 1.0]],
                 dtype=np.float64)
    return CentralPinholeModel(K=K)


def test_point_ray_residual_zero_when_point_on_ray():
    """The residual is zero for a point that lies exactly on the ray."""
    field = _pinhole_field()
    u, v = 79.5, 59.5
    origins, d = field.ray(np.array([u]), np.array([v]))
    origin_vec, d_vec = origins.reshape(3), d.reshape(3)
    # Point at distance 100 along the ray
    X = origin_vec + 100.0 * d_vec
    r = point_ray_residual(np.array([u, v]), field, X)
    assert np.linalg.norm(r) < 1e-12


def test_point_ray_residual_nonzero_when_point_off_ray():
    """The residual measures perpendicular distance correctly."""
    field = _pinhole_field()
    u, v = 79.5, 59.5
    origins, d = field.ray(np.array([u]), np.array([v]))
    origin_vec = origins.reshape(3)
    # Point offset by 5 mm perpendicular to the ray
    perp = np.array([1.0, 0.0, 0.0])  # approximately perpendicular for central pixel
    X = origin_vec + 100.0 * d.reshape(3) + 5.0 * perp
    r = point_ray_residual(np.array([u, v]), field, X)
    assert 4.5 < np.linalg.norm(r) < 5.5


def test_inverse_projection_pinhole_round_trip():
    """Project a point on the ray, then recover the exact pixel."""
    field = _pinhole_field()
    u_true, v_true = 50.0, 40.0
    origins, d = field.ray(np.array([u_true]), np.array([v_true]))
    X = origins.reshape(3) + 80.0 * d.reshape(3)

    uv, success, dist = project_point_by_rayfield_inverse(
        field, X, (160, 120), initial_uv=np.array([79.5, 59.5]),
    )
    assert success
    assert dist < 1e-6
    assert abs(uv[0] - u_true) < 1e-4
    assert abs(uv[1] - v_true) < 1e-4


def test_inverse_projection_returns_false_for_out_of_bounds():
    """A point not visible by any pixel should fail."""
    field = _pinhole_field()
    # Point far to the side (not visible from this camera)
    X = np.array([1e6, 0.0, 80.0], dtype=np.float64)
    _uv, success, _dist = project_point_by_rayfield_inverse(
        field, X, (160, 120),
    )
    # Should either fail or return a large distance
    # (the optimiser might converge to an edge pixel)
    assert not success or _dist > 1000.0


def test_batch_projection_returns_correct_shapes():
    """Batch version returns arrays of the expected size."""
    field = _pinhole_field()
    points = np.array([
        [0.0, 0.0, 80.0],
        [10.0, 5.0, 90.0],
        [-5.0, -2.0, 70.0],
    ], dtype=np.float64)
    uv, success, dist = project_points_by_rayfield_inverse(
        field, points, (160, 120),
    )
    assert uv.shape == (3, 2)
    assert success.shape == (3,)
    assert dist.shape == (3,)
    assert np.all(np.isfinite(uv))
    assert np.all(np.isfinite(dist))
