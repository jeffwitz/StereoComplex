"""Tests for image-based Zernike rayfield fitting from ChArUco observations."""

from __future__ import annotations

import numpy as np
import pytest

from stereocomplex.benchmarks.charuco_observation_simulator import (
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_oracle
from stereocomplex.benchmarks.rayfield_from_observations import (
    ZernikeFitDiagnostics,
    fit_zernike_rayfield_from_charuco_observations,
    fit_zernike_rayfields_from_multi_camera_observations,
)
from stereocomplex.rayfields.zernike_origin_field import MultiCameraZernikeRayField


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
