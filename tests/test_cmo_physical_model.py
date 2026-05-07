from __future__ import annotations

import numpy as np

from stereocomplex.physics.cmo import (
    CMOChannelSpec,
    CMOIntrinsics,
    CMOPolynomialChannelModel,
    PolynomialRayAberration,
    cmo_polynomial_channel_parameters_from_spec,
)
from stereocomplex.physics.central_models import CentralBrownConradyModel
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
            np.r_[[-40.0, -20.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
            np.r_[[+40.0, +20.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
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

    # Polynomial model at centre pixel: x_norm=0, y_norm=0 → d_cam=(0,0,1).
    poly = CMOPolynomialChannelModel(
        K=K, image_size=image_size, origin_x_mm=-10.0, origin_y_mm=0.0,
        aberration_terms=terms,
    )
    O_poly, d_poly = poly.ray(np.array([63.5]), np.array([47.5]))
    assert np.allclose(d_poly.reshape(-1, 3)[0], [0.0, 0.0, 1.0], atol=1e-12)

    # The two rays point in fundamentally different directions.
    angular_error_rad = float(np.arccos(np.clip(np.dot(d_cmo.reshape(-1, 3)[0], d_poly.reshape(-1, 3)[0]), -1.0, 1.0)))
    assert angular_error_rad > 0.05, f"expected >50 mrad structural mismatch, got {angular_error_rad:.4f} rad"


def test_polynomial_surrogate_rms_plateaus_with_relaxed_bounds() -> None:
    """Even with wide bounds the polynomial surrogate cannot match a CMO rayfield.

    The structural mismatch (all rays through one z=0 origin vs all rays
    through a sub-pupil at z=40) means the residual plateaus far above zero
    regardless of parameter freedom.
    """
    truth = _truth_model(distortion=True)
    image_size = (128, 96)
    pixels = np.array(
        [[u, v] for v in np.linspace(8.0, 119.0, 8) for u in np.linspace(8.0, 119.0, 8)],
        dtype=np.float64,
    )
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1], fx=180.0, fy=180.0, cx=63.5, cy=47.5)
    terms = CMOPolynomialChannelModel.default_terms()

    # Fit with generous bounds (distortion up to ±2, aberration ±0.5).
    wide_bounds = (
        np.r_[[-60.0, -30.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
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

    # The polynomial RMS should plateau above ~50 mm, not reach zero.
    assert result.rms_mm > 10.0, (
        f"Polynomial surrogate structural floor should be >> 0 mm, got {result.rms_mm:.2f} mm"
    )
    assert result.rms_mm < 200.0, f"Polynomial RMS {result.rms_mm:.2f} is unexpectedly huge"


def test_cmo_oracle_without_distortion_still_structural_mismatch() -> None:
    """Removing distortion from the CMO oracle does not fix the polynomial gap.

    The RMS gap comes from the geometric structure (sub-pupil at z=40 vs
    polynomial origin at z=0), not from the distortion coefficients.
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
        np.r_[[-60.0, -30.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
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
    assert result.rms_mm > 0.1, (
        f"Even without distortion, polynomial cannot collapse CMO rayfield (got {result.rms_mm:.2f} mm)"
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
        np.r_[[-60.0, -30.0, -2.0, -2.0, -1.0, -1.0, -2.0], -0.5 * np.ones(2 * len(terms))],
        np.r_[[+60.0, +30.0, +2.0, +2.0, +1.0, +1.0, +2.0], +0.5 * np.ones(2 * len(terms))],
    )
    poly_left = fit_physical_model_to_rayfield(
        CMOPolynomialChannelModel, truth.channel("left"), K=intr.as_K(),
        image_size=image_size, initial_parameters=poly_x0, bounds=wide_bounds,
        support_pixels=pixels, full_grid_weight=0.0, max_nfev=4000,
        name="polynomial", cmo_image_size=image_size, aberration_terms=terms,
    )
    # Polynomial RMS should be orders of magnitude larger than CMO RMS.
    assert poly_left.rms_mm > cmo_result.rms_mm * 1000, (
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
