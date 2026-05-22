from __future__ import annotations

import numpy as np

from stereocomplex.rayfields.zernike_origin_field import (
    MultiCameraZernikeRayField,
    ZernikeOriginField,
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)


def test_zero_coefficients_reduce_to_central_origin():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    field = ZernikeOriginField(K, ZernikeOriginFieldConfig(image_size=(640, 480), max_order=3))
    u = np.array([10.0, 320.0, 630.0])
    v = np.array([20.0, 240.0, 460.0])
    assert np.allclose(field.origin(u, v), 0.0)


def test_origin_field_enforces_transverse_gauge():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    field = ZernikeOriginField(K, ZernikeOriginFieldConfig(image_size=(640, 480), max_order=2))
    coeffs = field.coeffs.copy()
    coeffs[:, :] = np.array([2.0, -3.0, 5.0])
    field = ZernikeOriginField(
        K,
        ZernikeOriginFieldConfig(image_size=(640, 480), max_order=2),
        coefficients=type(field.coefficients)(coeffs),
    )
    u = np.array([10.0, 320.0, 630.0])
    v = np.array([20.0, 240.0, 460.0])
    origin, direction = field.ray(u, v)
    assert np.allclose(np.sum(origin * direction, axis=1), 0.0, atol=1e-12)


def test_origin_field_directions_are_normalized():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    field = ZernikeOriginField(K, ZernikeOriginFieldConfig(image_size=(640, 480), max_order=1))
    _origin, direction = field.ray(np.array([0.0, 120.0, 639.0]), np.array([0.0, 330.0, 479.0]))
    assert np.allclose(np.linalg.norm(direction, axis=1), 1.0)


def test_ray_field_identifies_direction_perturbations():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    config = ZernikeOriginFieldConfig(image_size=(640, 480), max_order=1)
    base = ZernikeOriginField(K, config)
    coeffs = ZernikeRayFieldCoefficients(
        origin_coeffs=np.zeros_like(base.coeffs),
        direction_coeffs=np.full_like(base.coeffs, 1e-3),
    )
    field = ZernikeRayField(K, config, coeffs)
    u = np.array([50.0, 320.0, 590.0])
    v = np.array([40.0, 240.0, 430.0])
    _origin, direction = field.ray(u, v)
    assert np.allclose(np.linalg.norm(direction, axis=1), 1.0)
    assert not np.allclose(direction, base.direction(u, v))


def test_multi_camera_zernike_rayfield_dispatches_by_name():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    config = ZernikeOriginFieldConfig(image_size=(640, 480), max_order=1)
    left = ZernikeRayField(K, config)
    right = ZernikeRayField(K, config)
    rig = MultiCameraZernikeRayField.from_fields({"left": left, "right": right})

    u = np.array([50.0, 320.0])
    v = np.array([40.0, 240.0])
    origin, direction = rig.ray("right", u, v)

    assert rig.names == ("left", "right")
    assert rig.n_channels == 2
    assert np.allclose(origin, right.origin(u, v))
    assert np.allclose(direction, right.direction(u, v))


def test_multi_camera_zernike_rayfield_builds_per_camera_configs():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    rig = MultiCameraZernikeRayField.from_camera_configs(
        {"left": K, "context": K},
        {
            "left": ZernikeOriginFieldConfig(image_size=(640, 480), max_order=1),
            "context": ZernikeOriginFieldConfig(image_size=(320, 240), max_order=2),
        },
    )

    assert rig.names == ("left", "context")
    assert rig.channel("left").config.image_size == (640, 480)
    assert rig.channel("context").config.max_order == 2
