"""Tests for explicit coordinate-frame conventions."""

from __future__ import annotations

import numpy as np
import pytest

from stereocomplex.core.conventions import (
    check_frame_convention,
    image_to_phys_xy,
    phys_to_image_xy,
    pixel_to_normalized_opencv,
    pixel_to_normalized_physical_y_up,
    transform_points_cv_to_phys,
    transform_points_phys_to_cv,
    transform_rays_cv_to_phys,
    transform_rays_phys_to_cv,
    transform_vectors_cv_to_phys,
    transform_vectors_phys_to_cv,
)


class _DummyModel:
    def __init__(self, convention: str | None = None):
        if convention is not None:
            self.frame_convention = convention


# ── pixel → normalised ──────────────────────────────────────


def test_pixel_to_normalized_opencv_signs():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    x, y = pixel_to_normalized_opencv(np.array([501, 500]), np.array([500, 501]), K)
    assert x[0] > 0  # u → +X
    assert x[1] == 0  # u = cx → x = 0
    assert y[0] == 0  # v = cy → y = 0
    assert y[1] > 0  # v → +Y


def test_pixel_to_normalized_physical_y_up_signs():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    _, y = pixel_to_normalized_physical_y_up(np.array([500]), np.array([501]), K)
    assert y[0] < 0  # Y-up: v ↑ → Y ↓


# ── point / vector / ray transforms ─────────────────────────


def test_points_cv_to_phys_flips_y():
    X_cv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X_phys = transform_points_cv_to_phys(X_cv)
    np.testing.assert_allclose(X_phys[:, 0], [1.0, 4.0])
    np.testing.assert_allclose(X_phys[:, 1], [-2.0, -5.0])
    np.testing.assert_allclose(X_phys[:, 2], [3.0, 6.0])


def test_vectors_cv_to_phys_flips_y():
    V_cv = np.array([[0.1, 0.2, 1.0]])
    V_phys = transform_vectors_cv_to_phys(V_cv)
    assert V_phys[0, 0] == 0.1
    assert V_phys[0, 1] == -0.2
    assert V_phys[0, 2] == 1.0


def test_rays_cv_to_phys_flips_y():
    O_cv = np.array([[1.0, 2.0, 3.0]])
    d_cv = np.array([[0.1, 0.2, 1.0]])
    O_phys, d_phys = transform_rays_cv_to_phys(O_cv, d_cv)
    assert O_phys[0, 1] == -2.0
    assert d_phys[0, 1] == -0.2


# ── round-trip ──────────────────────────────────────────────


def test_points_roundtrip_cv_phys():
    X = np.random.randn(100, 3)
    X2 = transform_points_phys_to_cv(transform_points_cv_to_phys(X))
    np.testing.assert_allclose(X, X2, atol=1e-14)


def test_vectors_roundtrip_cv_phys():
    V = np.random.randn(100, 3)
    V2 = transform_vectors_phys_to_cv(transform_vectors_cv_to_phys(V))
    np.testing.assert_allclose(V, V2, atol=1e-14)


def test_rays_roundtrip_cv_phys():
    origins = np.random.randn(10, 3)
    d = np.random.randn(10, 3)
    origins2, d2 = transform_rays_phys_to_cv(*transform_rays_cv_to_phys(origins, d))
    np.testing.assert_allclose(origins, origins2, atol=1e-14)
    np.testing.assert_allclose(d, d2, atol=1e-14)


# ── convention check ────────────────────────────────────────


def test_check_frame_convention_ok():
    a = _DummyModel("opencv_y_down")
    b = _DummyModel("opencv_y_down")
    check_frame_convention(a, b)  # should not raise


def test_check_frame_convention_mismatch_raises():
    a = _DummyModel("opencv_y_down")
    b = _DummyModel("physical_y_up")
    with pytest.raises(ValueError, match="frame_convention"):
        check_frame_convention(a, b, label="test")


def test_check_frame_convention_missing_attribute_is_ok():
    a = _DummyModel("opencv_y_down")
    b = _DummyModel()  # no frame_convention attribute
    check_frame_convention(a, b)  # should not raise (attribute missing is not a mismatch)


# ── image ↔ physical coordinate conversions ──────────────────


def test_phys_to_image_xy_converts_y_sign():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    Z = np.array([1.0])
    u, v = phys_to_image_xy(np.array([0.0]), np.array([0.1]), Z, K)
    # Y_up = +0.1 → Y_cv = -0.1 → v = cy + fy*(-0.1)/Z = 500 - 100 = 400
    assert abs(v[0] - 400.0) < 0.01


def test_image_to_phys_xy_converts_y_sign():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    Z = np.array([1.0])
    X, Y = image_to_phys_xy(np.array([600]), np.array([400]), Z, K)
    # v=400 → Y_cv = (400-500)/1000 = -0.1 → Y_phys = +0.1
    assert abs(Y[0] - 0.1) < 0.001


def test_phys_image_roundtrip():
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=float)
    Z = np.array([2.5])
    X0, Y0 = np.array([0.1]), np.array([-0.05])
    u, v = phys_to_image_xy(X0, Y0, Z, K)
    X1, Y1 = image_to_phys_xy(u, v, Z, K)
    np.testing.assert_allclose(X0, X1, atol=1e-12)
    np.testing.assert_allclose(Y0, Y1, atol=1e-12)


# ── key models declare the internal convention ──────────────


def test_cmo_telecentric_model_declares_opencv_convention():
    from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel

    m = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024.0, cy_principal_px=1024.0,
        pixel_pitch_mm=0.0055,
    )
    assert m.frame_convention == "opencv_y_down"


def test_zernike_rayfield_declares_opencv_convention():
    from stereocomplex.rayfields.zernike_origin_field import (
        ZernikeOriginFieldConfig,
        ZernikeRayField,
    )

    K = np.array([[25600, 0, 1024], [0, 25600, 1024], [0, 0, 1]], dtype=float)
    config = ZernikeOriginFieldConfig(image_size=(2048, 2048), max_order=2)
    rf = ZernikeRayField(K=K, config=config)
    assert rf.frame_convention == "opencv_y_down"
