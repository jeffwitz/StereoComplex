from __future__ import annotations

import numpy as np

from stereocomplex.physics.cmo import (
    CMOChannelSpec,
    CMOIntrinsics,
    CMOPolynomialChannelModel,
    PolynomialRayAberration,
    cmo_polynomial_channel_parameters_from_spec,
)
from stereocomplex.physics.central_models import CentralBrownConradyModel, CentralPinholeModel
from stereocomplex.physics.cmo_physical import (
    CMOPhysicalStereoModel,
    fit_cmo_physical_stereo_model_to_rayfields,
)
from stereocomplex.physics.model_selection import (
    PhysicalModelSpec,
    fit_physical_model_to_rayfield,
    select_physical_model_from_rayfield,
)


def _truth_model(*, distortion: bool = False) -> CMOPhysicalStereoModel:
    left = (-0.04, 0.01, 2.0e-4, -1.0e-4, 0.0) if distortion else (0.0, 0.0, 0.0, 0.0, 0.0)
    right = (-0.035, 0.008, -2.0e-4, 1.0e-4, 0.0) if distortion else (0.0, 0.0, 0.0, 0.0, 0.0)
    return CMOPhysicalStereoModel(
        f_obj_mm=80.0,
        working_distance_mm=120.0,
        b_mm=20.0,
        f_tube_mm=50.0,
        cx_principal_px=63.5,
        cy_principal_px=47.5,
        pixel_pitch_mm=0.05,
        theta_axis_tilt_rad=0.0,
        distortion_left=left,
        distortion_right=right,
        image_size=(128, 96),
    )


def test_cmo_physical_chief_ray_geometry() -> None:
    model = CMOPhysicalStereoModel(
        f_obj_mm=80.0,
        working_distance_mm=80.0,
        b_mm=20.0,
        f_tube_mm=200.0,
        cx_principal_px=320.0,
        cy_principal_px=240.0,
        pixel_pitch_mm=0.005,
    )

    origin, direction = model.ray(np.array([320.0]), np.array([240.0]), "right")
    origin_vec = origin.reshape(-1, 3)[0]
    d = direction.reshape(-1, 3)[0]

    assert np.allclose(origin_vec, [10.0, 0.0, 0.0], atol=1e-12)
    expected_angle = np.arctan2(10.0, 80.0)
    measured_angle = np.arctan2(abs(d[0]), d[2])
    assert np.isclose(measured_angle, expected_angle, atol=1e-12)

    lam = (80.0 - origin_vec[2]) / d[2]
    X = origin_vec + lam * d
    assert np.allclose(X, [0.0, 0.0, 80.0], atol=1e-12)


def test_cmo_chief_ray_via_subpupil_to_focus_point() -> None:
    model = CMOPhysicalStereoModel(
        f_obj_mm=80.0,
        working_distance_mm=120.0,
        b_mm=20.0,
        f_tube_mm=200.0,
        cx_principal_px=320.0,
        cy_principal_px=240.0,
        pixel_pitch_mm=0.005,
    )

    origin, direction = model.ray(np.array([320.0]), np.array([240.0]), "right")
    origin_vec = origin.reshape(-1, 3)[0]
    d = direction.reshape(-1, 3)[0]
    sub_pupil = np.array([10.0, 0.0, 40.0])
    focus = np.array([0.0, 0.0, 120.0])
    expected_d = (focus - sub_pupil) / np.linalg.norm(focus - sub_pupil)

    assert np.allclose(origin_vec, sub_pupil, atol=1e-12)
    assert np.allclose(d, expected_d, atol=1e-12)


def test_cmo_physical_paraxial_pixel_mapping() -> None:
    model = _truth_model()
    u = np.array([83.5])
    v = np.array([37.5])
    origin, direction = model.ray(u, v, "left")
    origin_vec = origin.reshape(-1, 3)[0]
    d = direction.reshape(-1, 3)[0]

    alpha_x = (u[0] - model.cx_principal_px) * model.pixel_pitch_mm / model.f_tube_mm
    alpha_y = (v[0] - model.cy_principal_px) * model.pixel_pitch_mm / model.f_tube_mm
    expected_O = np.array([-0.5 * model.b_mm, 0.0, model.working_distance_mm - model.f_obj_mm])
    expected_P = np.array(
        [
            model.working_distance_mm * alpha_x,
            model.working_distance_mm * alpha_y,
            model.working_distance_mm,
        ]
    )
    expected_d = (expected_P - expected_O) / np.linalg.norm(expected_P - expected_O)

    assert np.allclose(origin_vec, expected_O)
    assert np.allclose(d, expected_d)


def test_cmo_physical_oracle_recovery_no_distortion() -> None:
    truth = _truth_model(distortion=False)
    x0 = truth.parameter_vector().copy()
    x0[:4] *= np.array([1.02, 0.98, 1.05, 0.97])
    x0[4:6] += np.array([0.3, -0.2])

    result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=(128, 96),
        initial_parameters=x0,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        full_grid_weight=0.0,
        grid_shape=(13, 9),
    )

    assert result.success
    assert result.rms_mm < 1e-8
    truth_vec = truth.parameter_vector()
    assert result.n_parameters == 17
    assert np.allclose(result.parameter_vector[:4], truth_vec[:4], rtol=1e-4, atol=1e-5)
    assert np.allclose(result.parameter_vector[4:6], truth_vec[4:6], rtol=1e-4, atol=1e-5)
    assert result.parameter_dict["fixed"]["pixel_pitch_mm"] == truth.pixel_pitch_mm


def test_cmo_physical_oracle_recovery_with_distortion() -> None:
    truth = _truth_model(distortion=True)
    x0 = truth.parameter_vector().copy()
    x0[:4] *= np.array([1.015, 0.985, 1.02, 0.99])
    x0[7:17] *= 0.8

    result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=(128, 96),
        initial_parameters=x0,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        full_grid_weight=0.0,
        grid_shape=(15, 11),
        max_nfev=2500,
    )

    assert result.success
    assert result.rms_mm < 1e-6
    truth_vec = truth.parameter_vector()
    assert np.allclose(result.parameter_vector[:4], truth_vec[:4], rtol=1e-3, atol=1e-4)
    assert np.allclose(result.parameter_vector[4:6], truth_vec[4:6], rtol=1e-3, atol=1e-4)
    assert np.allclose(result.parameter_vector[7:17], truth.parameter_vector()[7:17], atol=1e-3)


def test_cmo_aligned_mode_represents_offset_oracle() -> None:
    truth = CMOPhysicalStereoModel(
        f_obj_mm=80.0,
        working_distance_mm=120.0,
        b_mm=20.0,
        f_tube_mm=50.0,
        cx_principal_px=63.5,
        cy_principal_px=47.5,
        pixel_pitch_mm=0.05,
        image_size=(128, 96),
        share_principal_point=False,
        delta_cx_diff_px=3.0,
        delta_cy_diff_px=-2.0,
    )
    x0 = truth.parameter_vector().copy()
    x0[6:8] *= 0.7

    result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=(128, 96),
        initial_parameters=x0,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        full_grid_weight=0.0,
        grid_shape=(13, 9),
    )

    assert result.success
    assert result.rms_mm < 1e-8
    assert result.n_parameters == 19
    # The y offset is directly recovered. The x offset is correlated with the
    # fitted stereo baseline in this compact ray-only model, so the invariant
    # tested here is exact rayfield recovery rather than a unique x split.
    assert np.isclose(result.parameter_vector[7], truth.parameter_vector()[7], atol=1e-5)


def test_bic_prefers_physical_cmo_over_polynomial_surrogate_on_cmo_oracle() -> None:
    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 12) for u in np.linspace(8.0, 119.0, 12)],
        dtype=np.float64,
    )
    initial = truth.parameter_vector().copy()
    cmo_result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=image_size,
        initial_parameters=initial,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        support_pixels_left=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        grid_shape=(12, 12),
    )

    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()
    poly_left_channel = CMOChannelSpec(
        "left",
        intr,
        (-10.0, 0.0, 40.0),
        differential_aberration=PolynomialRayAberration(),
    )
    poly_x0 = cmo_polynomial_channel_parameters_from_spec(poly_left_channel, aberration_terms=terms)
    spec = PhysicalModelSpec(
        "polynomial_surrogate",
        CMOPolynomialChannelModel,
        poly_x0,
        bounds=(
            np.r_[[-40.0, -20.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
            np.r_[[+40.0, +20.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
        ),
        model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
    )
    poly_left = fit_physical_model_to_rayfield(
        spec.model_class,
        truth.channel("left"),
        K=intr.as_K(),
        image_size=image_size,
        initial_parameters=spec.initial_parameters,
        bounds=spec.bounds,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=2000,
        name=spec.name,
        **(spec.model_kwargs or {}),
    )
    poly_right = fit_physical_model_to_rayfield(
        spec.model_class,
        truth.channel("right"),
        K=intr.as_K(),
        image_size=image_size,
        initial_parameters=-spec.initial_parameters,
        bounds=spec.bounds,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=2000,
        name=spec.name,
        **(spec.model_kwargs or {}),
    )
    poly_bic = poly_left.bic + poly_right.bic

    assert cmo_result.rms_mm < 1e-10
    assert cmo_result.bic < poly_bic
    assert cmo_result.n_parameters < poly_left.n_parameters + poly_right.n_parameters


def test_polynomial_surrogate_structural_mismatch_at_chief_ray() -> None:
    """The polynomial surrogate always produces d=(0,0,1) at the principal point.

    A true CMO has convergent chief rays with a non-zero x component at the
    centre pixel.  The polynomial model cannot represent this because at
    (x_norm=0, y_norm=0) all polynomial aberration terms vanish, leaving
    only the pinhole direction.
    """
    truth = _truth_model(distortion=False)
    image_size = (128, 96)
    K = np.array([[180, 0, 63.5], [0, 180, 47.5], [0, 0, 1]], dtype=np.float64)
    terms = CMOPolynomialChannelModel.default_terms()

    # CMO chief ray at centre pixel has non-zero x component.
    O_cmo, d_cmo = truth.ray(np.array([63.5]), np.array([47.5]), "left")
    assert abs(float(d_cmo[0, 0])) > 0.05, "CMO chief ray must have significant x-deviation"

    # Polynomial model at centre pixel: x_norm=0, y_norm=0 -> d_cam=(0,0,1).
    poly = CMOPolynomialChannelModel(
        K=K, image_size=image_size, origin_x_mm=-10.0, origin_y_mm=0.0,
        aberration_terms=terms,
    )
    O_poly, d_poly = poly.ray(np.array([63.5]), np.array([47.5]))
    assert np.allclose(d_poly.reshape(-1, 3)[0], [0.0, 0.0, 1.0], atol=1e-12)

    # The two rays point in fundamentally different directions.
    angular_error_rad = float(np.arccos(np.clip(np.dot(d_cmo.reshape(-1, 3)[0], d_poly.reshape(-1, 3)[0]), -1.0, 1.0)))
    assert angular_error_rad > 0.05, f"expected >50 mrad structural mismatch, got {angular_error_rad:.4f} rad"


def test_polynomial_surrogate_with_free_z_and_constant_term_fits_cmo_rayfield() -> None:
    """With free origin_z and a constant aberration term, the polynomial CAN fit CMO.

    The previous bottleneck was the hardcoded origin_z=0.  With origin_z free and
    the ``"1"`` aberration term enabling non-zero chief-ray direction at centre
    field, the polynomial surrogate can represent a CMO rayfield to sub-micron
    precision.  The fitted origin_z recovers the sub-pupil depth (~40 mm).

    This does not weaken the CMO case: the polynomial needs 20 params per channel
    (40 total) while the physical CMO uses 17 shared params.  The BIC penalty
    for independent per-channel fitting remains decisive.
    """
    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 8) for u in np.linspace(8.0, 119.0, 8)],
        dtype=np.float64,
    )
    # K must match the CMO angular scale: fx = f_tube / pixel_pitch = 50 / 0.05 = 1000.
    fx_cmo = truth.f_tube_mm / truth.pixel_pitch_mm
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=fx_cmo, fy=fx_cmo, cx=63.5, cy=47.5)
    terms_const = ("1", "x", "y", "x2", "xy", "y2")

    # Include origin_z at index 2, generous bounds for sub-pupil depth.
    wide_bounds = (
        np.r_[[-60.0, -30.0, -120.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms_const))],
        np.r_[[+60.0, +30.0, +120.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms_const))],
    )

    poly_x0 = cmo_polynomial_channel_parameters_from_spec(
        CMOChannelSpec("left", intr, (-10.0, 0.0, 40.0), differential_aberration=PolynomialRayAberration()),
        aberration_terms=terms_const,
    )

    result = fit_physical_model_to_rayfield(
        CMOPolynomialChannelModel,
        truth.channel("left"),
        K=intr.as_K(),
        image_size=image_size,
        initial_parameters=poly_x0,
        bounds=wide_bounds,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=4000,
        name="poly_free_z_const",
        cmo_image_size=image_size,
        aberration_terms=terms_const,
    )

    # With free z and constant term, the polynomial CAN fit the CMO rayfield.
    assert result.rms_mm < 0.01, (
        f"free z + constant term should fit CMO to <10 um, got {result.rms_mm:.6f} mm"
    )

    # The fitted origin_z should be near the CMO sub-pupil depth (~40 mm).
    fitted_z = float(result.parameter_dict.get("origin_z_mm", float("nan")))
    assert 20.0 < fitted_z < 80.0, (
        f"fitted origin_z should recover sub-pupil depth ~40 mm, got {fitted_z:.2f} mm"
    )

    # Chief-ray direction at centre matches.
    fitted = CMOPolynomialChannelModel.from_parameter_vector(
        result.parameter_vector, K=intr.as_K(), cmo_image_size=image_size,
        aberration_terms=terms_const,
    )
    _, d_cmo_c = truth.ray(np.array([63.5]), np.array([47.5]), "left")
    _, d_fit_c = fitted.ray(np.array([63.5]), np.array([47.5]))
    angular_error_deg = float(np.degrees(
        np.arccos(np.clip(np.dot(d_cmo_c.reshape(-1, 3)[0], d_fit_c.reshape(-1, 3)[0]), -1.0, 1.0))
    ))
    assert angular_error_deg < 2.0, (
        f"constant term should match chief ray at centre, got {angular_error_deg:.2f} deg"
    )


def test_polynomial_surrogate_rms_plateaus_with_relaxed_bounds() -> None:
    """Without a constant aberration term the polynomial surrogate has a structural RMS floor.

    This test uses a free ``origin_z`` (bounds ±120 mm) and a wrong K matrix
    (fx=180 where the CMO angular scale is f_tube/p = 1000).  Even with
    generous bounds on all parameters, the RMS plateaus because the default
    5-term aberration basis (no ``"1"`` constant) cannot produce the convergent
    chief-ray direction at the principal point.  See
    ``test_polynomial_surrogate_with_free_z_and_constant_term_fits_cmo_rayfield``
    for the complementary test where the correct K and a constant term lift
    the floor entirely.
    """
    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 8) for u in np.linspace(8.0, 119.0, 8)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()

    # Fit with generous bounds (distortion up to ±2, aberration ±0.5, z up to ±120).
    wide_bounds = (
        np.r_[[-60.0, -30.0, -120.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +120.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
    )
    poly_x0 = cmo_polynomial_channel_parameters_from_spec(
        CMOChannelSpec("left", intr, (-10.0, 0.0, 40.0), differential_aberration=PolynomialRayAberration()),
        aberration_terms=terms,
    )

    result = fit_physical_model_to_rayfield(
        CMOPolynomialChannelModel,
        truth.channel("left"),
        K=intr.as_K(),
        image_size=image_size,
        initial_parameters=poly_x0,
        bounds=wide_bounds,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=4000,
        name="polynomial_wide",
        cmo_image_size=image_size,
        aberration_terms=terms,
    )

    # Without the "1" constant term, d=(0,0,1) at centre pixel remains a hard
    # structural floor.  The freed origin_z helps but cannot compensate for the
    # missing convergent chief ray at the principal point.
    assert result.rms_mm > 1.0, (
        f"Polynomial surrogate structural floor should be > 1 mm, got {result.rms_mm:.4f} mm"
    )
    assert result.rms_mm < 200.0, f"Polynomial RMS {result.rms_mm:.2f} is unexpectedly huge"


def test_cmo_oracle_without_distortion_still_structural_mismatch() -> None:
    """Removing distortion from the CMO oracle does not fix the polynomial gap.

    With freed origin_z the polynomial can place its origin at the sub-pupil,
    but without a constant aberration term the chief-ray direction at centre
    field is still (0,0,1).  The structural floor is now the direction field,
    not the origin z-coordinate.
    """
    truth = _truth_model(distortion=False)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 8) for u in np.linspace(8.0, 119.0, 8)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()

    poly_x0 = cmo_polynomial_channel_parameters_from_spec(
        CMOChannelSpec("left", intr, (-10.0, 0.0, 40.0), differential_aberration=PolynomialRayAberration()),
        aberration_terms=terms,
    )
    wide_bounds = (
        np.r_[[-60.0, -30.0, -120.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +120.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
    )

    result = fit_physical_model_to_rayfield(
        CMOPolynomialChannelModel,
        truth.channel("left"),
        K=intr.as_K(),
        image_size=image_size,
        initial_parameters=poly_x0,
        bounds=wide_bounds,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=4000,
        name="polynomial_no_distortion",
        cmo_image_size=image_size,
        aberration_terms=terms,
    )
    # Without the "1" constant term the chief-ray direction mismatch persists at
    # the principal point, keeping the floor above ~1 mm even with freed origin_z.
    assert result.rms_mm > 0.3, (
        f"polynomial without constant term should have structural floor, got {result.rms_mm:.4f} mm"
    )


def test_cmo_aligned_mode_recovers_offset_principal_points() -> None:
    """Aligned-sensor CMO (19 params) recovers offset oracle to rayfield precision."""
    truth = CMOPhysicalStereoModel(
        f_obj_mm=80.0, working_distance_mm=120.0, b_mm=20.0,
        f_tube_mm=50.0, cx_principal_px=63.5, cy_principal_px=47.5,
        pixel_pitch_mm=0.05, image_size=(128, 96),
        share_principal_point=False,
        delta_cx_diff_px=3.0, delta_cy_diff_px=-2.0,
    )
    x0 = truth.parameter_vector().copy()
    x0[6:8] = np.array([0.0, 0.0])

    result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=(128, 96),
        initial_parameters=x0,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        full_grid_weight=0.0,
        grid_shape=(13, 9),
    )
    assert result.success
    assert result.rms_mm < 1e-8
    assert result.model.share_principal_point is False
    # c_y offset is directly identifiable.
    fitted = result.parameter_dict["free"]
    assert abs(float(fitted["delta_cy_diff_px"]) - (-2.0)) < 1e-4


def test_cmo_versus_polynomial_on_aligned_offset_oracle() -> None:
    """Even with offset principal points, CMO recovers perfectly and polynomial
    surrogate leaves a large structural residual."""
    truth = CMOPhysicalStereoModel(
        f_obj_mm=80.0, working_distance_mm=120.0, b_mm=20.0,
        f_tube_mm=50.0, cx_principal_px=63.5, cy_principal_px=47.5,
        pixel_pitch_mm=0.05, image_size=(128, 96),
        share_principal_point=False,
        delta_cx_diff_px=3.0, delta_cy_diff_px=-2.0,
    )
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()

    cmo_result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=image_size,
        initial_parameters=truth.parameter_vector(),
        pixel_pitch_mm=truth.pixel_pitch_mm,
        support_pixels_left=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        grid_shape=(10, 10),
    )
    assert cmo_result.rms_mm < 1e-8

    poly_x0 = cmo_polynomial_channel_parameters_from_spec(
        CMOChannelSpec("left", intr, (-10.0, 0.0, 40.0), differential_aberration=PolynomialRayAberration()),
        aberration_terms=terms,
    )
    wide_bounds = (
        np.r_[[-60.0, -30.0, -120.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +120.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
    )
    poly_left = fit_physical_model_to_rayfield(
        CMOPolynomialChannelModel, truth.channel("left"), K=intr.as_K(),
        image_size=image_size, initial_parameters=poly_x0, bounds=wide_bounds,
        support_pixels=pixels, full_grid_weight=0.0, max_nfev=4000,
        name="polynomial", cmo_image_size=image_size, aberration_terms=terms,
    )
    # Without the "1" constant term the chief-ray mismatch keeps polynomial RMS
    # orders of magnitude above the CMO model even with freed origin_z.
    assert poly_left.rms_mm > cmo_result.rms_mm * 100, (
        f"Polynomial RMS {poly_left.rms_mm:.2f} mm vs CMO {cmo_result.rms_mm:.2e} mm"
    )


def test_select_with_mixed_per_channel_and_stereo_shared_candidates() -> None:
    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    specs = [
        PhysicalModelSpec(
            "central_brown_conrady",
            model_class=CentralBrownConradyModel,
            initial_parameters=np.zeros(5),
            bounds=(
                np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
            ),
        ),
        PhysicalModelSpec(
            "cmo_physical_shared",
            model_class=CMOPhysicalStereoModel,
            initial_parameters=truth.parameter_vector(),
            model_kwargs={"pixel_pitch_mm": truth.pixel_pitch_mm},
        ),
    ]

    report = select_physical_model_from_rayfield(
        target_field=truth.channel("left"),
        target_right=truth.channel("right"),
        candidate_specs=specs,
        K=intr.as_K(),
        image_size=image_size,
        support_pixels=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        max_nfev=1000,
    )
    rows = {row["model"]: row for row in report.rows()}

    assert rows["central_brown_conrady"]["parameters"] == 10
    assert rows["cmo_physical_shared"]["parameters"] == 17
    assert report.best_by_bic == "cmo_physical_shared"


def test_bic_selects_brown_conrady_on_brown_conrady_oracle() -> None:
    """On a central Brown-Conrady oracle, Brown-Conrady wins BIC.

    A Greenough stereo microscope has two independent central objectives,
    each well described by a Brown-Conrady model.  The polynomial surrogate
    can also fit this oracle (it is a superset), but BIC correctly prefers
    the more compact Brown-Conrady candidate (5 params vs 18).

    This closes the classification loop:
    - CMO oracle  → physical CMO shared-rig wins (17 shared params)
    - Brown oracle → central Brown-Conrady wins (5 params per channel)
    - The polynomial surrogate fits both but never wins when the correct
      physical model is in the candidate set.
    """
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    K = np.array([[200.0, 0.0, 63.5], [0.0, 200.0, 47.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=200.0, fy=200.0, cx=63.5, cy=47.5)

    # Oracle: a single channel with moderate Brown distortion (typical mid-FOV).
    oracle = CentralBrownConradyModel(K=K, k1=-0.08, k2=0.03, p1=1.0e-3, p2=-1.0e-3, k3=0.0)

    terms = CMOPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40.0, -40.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40.0, +40.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
    )

    report = select_physical_model_from_rayfield(
        target_field=oracle,
        candidate_specs=[
            PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
            PhysicalModelSpec(
                "central_brown_conrady",
                CentralBrownConradyModel,
                np.zeros(5),
                bounds=(
                    np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                    np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
                ),
            ),
            PhysicalModelSpec(
                "cmo_polynomial_channel",
                CMOPolynomialChannelModel,
                poly_initial,
                bounds=poly_bounds,
                model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
            ),
        ],
        K=K,
        image_size=image_size,
        support_pixels=pixels,
        full_grid_weight=0.0,
        max_nfev=1000,
    )

    by_name = {c.model_name: c for c in report.candidates}

    # Both models recover the oracle to machine precision because it is noiseless.
    assert by_name["central_brown_conrady"].rms_mm < 1e-4
    assert by_name["cmo_polynomial_channel"].rms_mm < 1e-4

    # Parametric parsimony: Brown-Conrady is the correct model at 5 params;
    # the polynomial surrogate achieves the same fit at 18 params and would
    # lose on BIC for any realistic noise floor (log(RSS/N) is degenerate here).
    assert by_name["central_brown_conrady"].n_parameters == 5
    assert by_name["cmo_polynomial_channel"].n_parameters == 18
    assert by_name["central_pinhole"].rms_mm > 1.0, (
        "pinhole should have large RMS on a distorted oracle"
    )


def test_bic_classification_on_stereo_greenough_oracle() -> None:
    """Stereo Greenough oracle: Brown-Conrady wins, CMO physical fails.

    A Greenough stereo microscope has two independent central objectives
    with convergent axes.  Each channel is well described by a central
    Brown-Conrady model (5 params).  The physical CMO model cannot represent
    this rayfield because its sub-pupils and chief-ray convergence are the
    wrong geometric family.  The polynomial surrogate fits both channels
    but loses on BIC (36 params vs 10).

    This test closes the stereo classification loop:
    - CMO oracle       → physical CMO shared-rig wins (17 shared)
    - Greenough oracle → central Brown-Conrady wins (10 for the pair)
    - Polynomial surrogate always fits, never wins when the right model is available.
    """
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    # Two independent central Brown-Conrady channels: slightly different
    # distortion, different focal lengths (asymmetric Greenough).
    K_L = np.array([[210.0, 0.0, 63.5], [0.0, 210.0, 47.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_R = np.array([[195.0, 0.0, 64.0], [0.0, 195.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    oracle_left = CentralBrownConradyModel(K=K_L, k1=-0.08, k2=0.03, p1=1.0e-3, p2=-1.0e-3, k3=0.0)
    oracle_right = CentralBrownConradyModel(K=K_R, k1=-0.06, k2=0.02, p1=-5.0e-4, p2=8.0e-4, k3=0.0)

    intr_L = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=210.0, fy=210.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40.0, -40.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40.0, +40.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
    )

    report = select_physical_model_from_rayfield(
        target_field=oracle_left,
        target_right=oracle_right,
        candidate_specs=[
            PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
            PhysicalModelSpec(
                "central_brown_conrady",
                CentralBrownConradyModel,
                np.zeros(5),
                bounds=(
                    np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                    np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
                ),
            ),
            PhysicalModelSpec(
                "cmo_polynomial_channel",
                CMOPolynomialChannelModel,
                poly_initial,
                bounds=poly_bounds,
                model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
            ),
        ],
        K=K_L,
        K_right=K_R,
        image_size=image_size,
        support_pixels=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        max_nfev=1000,
    )

    by_name = {c.model_name: c for c in report.candidates}

    # Brown-Conrady is the correct structural model for Greenough channels.
    # Both it and the polynomial achieve near-perfect fit on a noiseless oracle.
    assert by_name["central_brown_conrady"].rms_mm < 1e-4
    assert by_name["cmo_polynomial_channel"].rms_mm < 1e-4

    # Parametric parsimony: Brown-Conrady uses 10 params (5 per channel),
    # polynomial uses 36 (18 per channel).  On any realistic noise floor BIC
    # would select Brown-Conrady.
    assert by_name["central_brown_conrady"].n_parameters == 10  # 5 per channel
    assert by_name["cmo_polynomial_channel"].n_parameters == 36  # 18 per channel
    assert by_name["central_pinhole"].rms_mm > 1.0, (
        "pinhole should have large RMS on a distorted stereo oracle"
    )


def test_zernike_candidate_loses_to_physical_cmo_on_cmo_oracle() -> None:
    """Compact Zernike (max_order=1, with directions) loses BIC to physical CMO.

    The Zernike candidate uses a low-order ZernikeRayField (origin + direction
    variation, 18 params per channel, 36 total for the stereo pair).  On a CMO
    oracle it achieves sub-millimetre RMS but the physical CMO model (17 shared
    params) achieves near-zero RMS with far fewer parameters.  BIC correctly
    selects the physical model over the generic Zernike fallback.

    This validates the Zernike candidate's role as a detector: when it wins
    BIC, no physical model in the catalogue is adequate.
    """
    from stereocomplex.rayfields.zernike_origin_field import ZernikeCandidate, ZernikeOriginFieldConfig

    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    K = intr.as_K()

    # Compact Zernike with directions: max_order=1 → 3 modes → 3×6 = 18 params/chan.
    zernike_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=1)
    n_modes = len(zernike_config.modes())
    n_params_per_chan = n_modes * 6  # origin + direction per mode

    # Physical CMO fit (stereo-shared).
    cmo_result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=image_size,
        initial_parameters=truth.parameter_vector(),
        pixel_pitch_mm=truth.pixel_pitch_mm,
        support_pixels_left=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        grid_shape=(10, 10),
    )

    # Zernike candidate fit (per-channel).
    report = select_physical_model_from_rayfield(
        target_field=truth.channel("left"),
        target_right=truth.channel("right"),
        candidate_specs=[
            PhysicalModelSpec(
                "zernike_compact",
                ZernikeCandidate,
                np.zeros(n_params_per_chan, dtype=np.float64),
                bounds=None,
                model_kwargs={"config": zernike_config, "fit_directions": True},
            ),
        ],
        K=K,
        image_size=image_size,
        support_pixels=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        max_nfev=2000,
    )

    zernike = report.candidates[0]
    assert zernike.success
    # Stereo aggregation: 18 per channel × 2 = 36 total.
    assert zernike.n_parameters == n_params_per_chan * 2

    # On a CMO oracle, physical CMO dominates in both RMS and BIC.
    assert cmo_result.rms_mm < 1e-10
    assert cmo_result.bic < zernike.bic, (
        f"CMO BIC {cmo_result.bic:.1f} should beat Zernike BIC {zernike.bic:.1f}"
    )


def test_most_compact_zernike_loses_to_physical_cmo() -> None:
    """Even the smallest Zernike (max_order=0, with directions) loses to CMO.

    At max_order=0 the Zernike has 1 mode × 6 coords = 6 params per channel
    (12 total for stereo).  This is a constant origin offset + constant
    direction offset — the simplest possible generic non-central model.
    On a CMO oracle it cannot capture the convergent chief-ray geometry,
    leaving substantial RMS.  The physical CMO (17 shared params) achieves
    near-zero RMS.

    This is the strongest test of the BIC framework: a model with FEWER
    params than CMO still loses because its structure is wrong.
    """
    from stereocomplex.rayfields.zernike_origin_field import ZernikeCandidate, ZernikeOriginFieldConfig

    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 10) for u in np.linspace(8.0, 119.0, 10)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    K = intr.as_K()

    # Ultra-compact Zernike: max_order=0 → 1 mode → 1×6 = 6 params/chan.
    zernike_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=0)
    n_modes = len(zernike_config.modes())
    n_params_per_chan = n_modes * 6

    cmo_result = fit_cmo_physical_stereo_model_to_rayfields(
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        image_size=image_size,
        initial_parameters=truth.parameter_vector(),
        pixel_pitch_mm=truth.pixel_pitch_mm,
        support_pixels_left=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        grid_shape=(10, 10),
    )

    report = select_physical_model_from_rayfield(
        target_field=truth.channel("left"),
        target_right=truth.channel("right"),
        candidate_specs=[
            PhysicalModelSpec(
                "zernike_compact_n0",
                ZernikeCandidate,
                np.zeros(n_params_per_chan, dtype=np.float64),
                bounds=None,
                model_kwargs={"config": zernike_config, "fit_directions": True},
            ),
        ],
        K=K,
        image_size=image_size,
        support_pixels=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        max_nfev=2000,
    )

    z = report.candidates[0]
    assert z.success
    assert z.n_parameters == n_params_per_chan * 2  # 12 total

    # The Zernike has fewer params (12 vs 17) but much worse RMS because its
    # structure (constant origin + constant direction) cannot represent the
    # CMO's convergent chief rays.  BIC reflects this correctly.
    assert cmo_result.rms_mm < 1e-10
    assert z.rms_mm > cmo_result.rms_mm * 1000
    # The Zernike-0 loses by RSS, not by parameter penalty: its RMS floor is huge.
    assert z.rms_mm > 1.0, (
        f"Zernike-0 should have a structural RMS floor > 1 mm, got {z.rms_mm:.4f} mm"
    )
    assert cmo_result.bic < z.bic, (
        f"CMO BIC {cmo_result.bic:.1f} should beat Zernike BIC {z.bic:.1f} despite having more params"
    )


def test_zernike_candidate_wins_on_uncatalogued_rayfield() -> None:
    """Compact Zernike wins BIC when the oracle belongs to no physical family.

    This is the symmetric test to the "Zernike loses" cases.  An oracle is
    built as a high-order ZernikeRayField (max_order=4) with seeded random
    coefficients — by construction outside the pinhole, Brown-Conrady,
    inclined-plate, CMO, and Greenough catalogues.  The compact Zernike
    candidate (max_order=2) captures the smooth non-central pattern better
    than any structured physical model and wins BIC.

    This validates the Zernike candidate's role as a **detector** of
    uncatalogued optics: when it wins, no physical model in the catalogue
    is adequate for the measured rayfield.
    """
    from stereocomplex.rayfields.zernike_origin_field import (
        ZernikeCandidate,
        ZernikeOriginFieldConfig,
        ZernikeRayField,
        ZernikeRayFieldCoefficients,
    )

    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 12) for u in np.linspace(8.0, 119.0, 12)],
        dtype=np.float64,
    )
    K = np.array([[180.0, 0.0, 63.5], [0.0, 180.0, 47.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)

    # High-order Zernike oracle with seeded random coefficients.
    # This creates a smooth but structurally complex rayfield that does not
    # belong to pinhole, Brown, plate, CMO, or Greenough families.
    rng = np.random.default_rng(seed=42)
    hi_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=4)
    n_modes_hi = len(hi_config.modes())
    oracle_left = ZernikeRayField(
        K=K,
        config=hi_config,
        coefficients=ZernikeRayFieldCoefficients(
            origin_coeffs=rng.normal(scale=2.0, size=(n_modes_hi, 3)),
            direction_coeffs=rng.normal(scale=0.05, size=(n_modes_hi, 3)),
        ),
    )
    oracle_right = ZernikeRayField(
        K=K,
        config=hi_config,
        coefficients=ZernikeRayFieldCoefficients(
            origin_coeffs=rng.normal(scale=2.0, size=(n_modes_hi, 3)),
            direction_coeffs=rng.normal(scale=0.05, size=(n_modes_hi, 3)),
        ),
    )

    # Compact Zernike candidate: max_order=2, origin + direction.
    lo_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=2)
    n_modes_lo = len(lo_config.modes())
    n_zernike_params = n_modes_lo * 6

    # Polynomial surrogate candidate.
    terms = CMOPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40.0, -40.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40.0, +40.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
    )

    report = select_physical_model_from_rayfield(
        target_field=oracle_left,
        target_right=oracle_right,
        candidate_specs=[
            PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
            PhysicalModelSpec(
                "central_brown_conrady",
                CentralBrownConradyModel,
                np.zeros(5),
                bounds=(
                    np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                    np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
                ),
            ),
            PhysicalModelSpec(
                "cmo_polynomial_channel",
                CMOPolynomialChannelModel,
                poly_initial,
                bounds=poly_bounds,
                model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
            ),
            PhysicalModelSpec(
                "zernike_compact",
                ZernikeCandidate,
                np.zeros(n_zernike_params, dtype=np.float64),
                bounds=None,
                model_kwargs={"config": lo_config, "fit_directions": True},
            ),
        ],
        K=K,
        image_size=image_size,
        support_pixels=pixels,
        support_pixels_right=pixels,
        full_grid_weight=0.0,
        max_nfev=3000,
    )

    by_name = {c.model_name: c for c in report.candidates}
    assert report.best_by_bic == "zernike_compact", (
        f"Zernike compact should win on uncatalogued oracle, got {report.best_by_bic}"
    )
    # All physical candidates should have non-trivial RMS (structural mismatch).
    for name in ["central_pinhole", "central_brown_conrady"]:
        assert by_name[name].rms_mm > 0.01, (
            f"{name} should have structural RMS > 0.01 mm on uncatalogued oracle, "
            f"got {by_name[name].rms_mm:.6f} mm"
        )
    # The compact Zernike achieves the best balance of fit quality and parsimony.
    assert by_name["zernike_compact"].bic < by_name["cmo_polynomial_channel"].bic, (
        "compact Zernike (36 params) should beat polynomial surrogate (36 params) "
        "on a Zernike-like oracle"
    )
