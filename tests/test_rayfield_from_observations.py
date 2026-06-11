"""Tests for image-based Zernike rayfield fitting from ChArUco observations."""

from __future__ import annotations

import numpy as np
import pytest

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_oracle
from stereocomplex.benchmarks.rayfield_from_observations import (
    StagePosePrior,
    ZernikeFitDiagnostics,
    fit_constrained_zernike_rayfield,
    fit_zernike_rayfield_from_charuco_observations,
    fit_zernike_rayfields_from_multi_camera_observations,
)
from stereocomplex.rayfields.zernike_origin_field import MultiCameraZernikeRayField


def _make_axial_stage_observations() -> tuple[
    CharucoObservationSet, np.ndarray, np.ndarray
]:
    image_size = (160, 120)
    focal_px = 220.0
    K = np.array(
        [
            [focal_px, 0.0, image_size[0] / 2],
            [0.0, focal_px, image_size[1] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    xx, yy = np.meshgrid(np.arange(4) * 3.0, np.arange(3) * 3.0)
    object_points = np.column_stack(
        [xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)]
    )
    nominal_positions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    jitter_mm = np.array([0.0, 0.02, -0.01, 0.01, -0.02])
    z_per_frame = 100.0 + nominal_positions + jitter_mm
    baseline_mm = 10.0

    left_pixels = []
    right_pixels = []
    for z in z_per_frame:
        points = object_points + np.array([-4.5, -3.0, z])
        left_pixels.append(
            np.column_stack(
                [
                    focal_px * points[:, 0] / points[:, 2] + image_size[0] / 2,
                    focal_px * points[:, 1] / points[:, 2] + image_size[1] / 2,
                ]
            )
        )
        right_pixels.append(
            np.column_stack(
                [
                    focal_px * (points[:, 0] - baseline_mm) / points[:, 2]
                    + image_size[0] / 2,
                    focal_px * points[:, 1] / points[:, 2] + image_size[1] / 2,
                ]
            )
        )

    observations = CharucoObservationSet(
        object_points_mm=object_points,
        pose_rvecs=np.zeros((len(z_per_frame), 3)),
        pose_tvecs=np.column_stack(
            [
                np.full(len(z_per_frame), -4.5),
                np.full(len(z_per_frame), -3.0),
                z_per_frame,
            ]
        ),
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        point_indices=[
            np.arange(len(object_points), dtype=int) for _ in z_per_frame
        ],
        noise_std_px=0.0,
        image_size=image_size,
    )
    return observations, K, nominal_positions


def test_zernike_fit_diagnostics_report_residual_and_pose_count():
    """Fit on a pinhole oracle should produce diagnostics with expected fields."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    )
    _left, _right, diag = fit_zernike_rayfield_from_charuco_observations(
        obs,
        (160, 120),
        oracle.K_left,
        oracle.K_right,
        max_order=2,
        max_nfev=50,
    )
    assert isinstance(diag, ZernikeFitDiagnostics)
    assert diag.n_poses > 0
    assert diag.n_observations > 0
    assert diag.ray_rms_mm is not None
    assert np.isfinite(diag.ray_rms_mm)
    assert diag.channel_names == ("left", "right")
    assert diag.n_channels == 2


def test_multi_camera_zernike_fit_accepts_stereo_observation_container():
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    ).to_multi_camera()
    fields, diag = fit_zernike_rayfields_from_multi_camera_observations(
        obs,
        (160, 120),
        {"left": oracle.K_left, "right": oracle.K_right},
        max_order=2,
        max_nfev=50,
    )

    assert isinstance(fields, MultiCameraZernikeRayField)
    assert fields.names == ("left", "right")
    assert diag.channel_names == ("left", "right")
    assert diag.n_channels == 2


def test_zernike_fit_matches_oracle_on_pinhole_within_tolerance():
    """On a pinhole oracle, the fitted Zernike should have near-zero RMS."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    )
    _left, _right, diag = fit_zernike_rayfield_from_charuco_observations(
        obs,
        (160, 120),
        oracle.K_left,
        oracle.K_right,
        max_order=2,
        max_nfev=100,
    )
    assert diag.converged
    assert diag.ray_rms_mm < 1e-6, (
        f"pinhole oracle should have near-zero Zernike residual, got {diag.ray_rms_mm:.2e} mm"
    )


def test_constrained_fit_preserves_legacy_pose_parameterisation_by_default():
    observations, K, _nominal_positions = _make_axial_stage_observations()
    _left, _right, diag, _rotations, translations = fit_constrained_zernike_rayfield(
        observations,
        observations.image_size,
        K,
        K,
        max_order_o=0,
        max_order_d=0,
        max_nfev=30,
    )

    assert diag.converged
    assert diag.stage_scale is None
    assert diag.stage_jitter_rms_mm is None
    assert diag.stage_axis is None
    assert np.ptp(np.array(translations)[:, 2]) > 1.9


def test_constrained_fit_separates_stage_scale_and_frame_jitter():
    observations, K, nominal_positions = _make_axial_stage_observations()
    prior = StagePosePrior(
        nominal_positions_mm=nominal_positions,
        scale_sigma=1e-3,
        jitter_sigma_mm=5e-3,
        ray_sigma_mm=1e-5,
    )
    _left, _right, diag, _rotations, translations = fit_constrained_zernike_rayfield(
        observations,
        observations.image_size,
        K,
        K,
        max_order_o=0,
        max_order_d=0,
        max_nfev=30,
        stage_prior=prior,
    )

    assert diag.converged
    assert diag.stage_scale == pytest.approx(1.0, abs=2e-3)
    assert diag.stage_jitter_rms_mm == pytest.approx(0.014, abs=2e-3)
    assert diag.stage_axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert np.array(translations)[:, 2] == pytest.approx(
        observations.pose_tvecs[:, 2], abs=5e-5
    )


def test_stage_prior_rejects_wrong_frame_count():
    observations, K, nominal_positions = _make_axial_stage_observations()
    prior = StagePosePrior(nominal_positions_mm=nominal_positions[:-1])

    with pytest.raises(ValueError, match="must match the number of poses"):
        fit_constrained_zernike_rayfield(
            observations,
            observations.image_size,
            K,
            K,
            max_order_o=0,
            max_order_d=0,
            stage_prior=prior,
        )


@pytest.mark.slow
def test_zernike_fit_matches_oracle_on_cmo_oracle():
    """On a CMO oracle, the Zernike fit should achieve reasonable RMS."""
    from stereocomplex.benchmarks.model_selection_oracles import build_cmo_oracle

    oracle = build_cmo_oracle(image_size=(160, 120))
    z_dist = oracle.ground_truth_parameters["working_distance_mm"]
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        z_distance_mm=z_dist,
        squares_x=9,
        squares_y=7,
        square_size_mm=3.0,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=3,
    )
    _left, _right, diag = fit_zernike_rayfield_from_charuco_observations(
        obs,
        (160, 120),
        oracle.K_left,
        oracle.K_right,
        max_order=2,
        max_nfev=100,
    )
    assert diag.converged
    assert diag.ray_rms_mm < 0.01, f"CMO Zernike fit should be sub-mm, got {diag.ray_rms_mm:.4f} mm"
