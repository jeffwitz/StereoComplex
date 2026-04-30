from __future__ import annotations

import numpy as np

from stereocomplex.metrics.rayfield_metrics import intersect_rays_with_z_plane
from stereocomplex.synthetic.parallel_plate import (
    ParallelPlateSyntheticParams,
    normal_from_tilts,
    parallel_plate_ray_from_pixel,
    pinhole_ray_from_pixel,
    project_point_with_parallel_plate,
)


def test_normal_from_tilts_is_unit_length():
    q = normal_from_tilts(12.0, 5.0)
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_parallel_plate_keeps_pinhole_direction():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    params = ParallelPlateSyntheticParams(thickness=8.0, alpha_deg=12.0, beta_deg=5.0)
    _O_true, d_true = parallel_plate_ray_from_pixel(180.0, 120.0, K, params)
    d_pinhole = pinhole_ray_from_pixel(180.0, 120.0, K)
    assert np.allclose(d_true, d_pinhole, atol=1e-12)


def test_zero_thickness_is_same_ray_as_central_model_on_planes():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    params = ParallelPlateSyntheticParams(thickness=0.0, alpha_deg=17.0, beta_deg=-4.0)
    O_plate, d_plate = parallel_plate_ray_from_pixel(260.0, 210.0, K, params)
    O_central = np.zeros(3)
    d_central = pinhole_ray_from_pixel(260.0, 210.0, K)
    for z in (100.0, 1000.0):
        P_plate = intersect_rays_with_z_plane(O_plate.reshape(1, 3), d_plate.reshape(1, 3), z)
        P_central = intersect_rays_with_z_plane(O_central.reshape(1, 3), d_central.reshape(1, 3), z)
        assert np.allclose(P_plate, P_central, atol=1e-10)


def test_parallel_plate_projection_roundtrip_point_to_ray_distance():
    K = np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]])
    params = ParallelPlateSyntheticParams(thickness=12.0, alpha_deg=11.0, beta_deg=6.0)
    P = np.array([25.0, -18.0, 700.0])
    u, v = project_point_with_parallel_plate(P, K, params, image_size=(640, 480))
    origin, direction = parallel_plate_ray_from_pixel(u, v, K, params)
    dist = np.linalg.norm(np.cross(P - origin, direction))
    assert dist < 1e-7
