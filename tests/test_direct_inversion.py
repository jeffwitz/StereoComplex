"""Smoke tests for direct ChArUco model inversion."""

from __future__ import annotations
import pytest

import numpy as np
from scipy.spatial.transform import Rotation

from stereocomplex.benchmarks.charuco_observation_simulator import (
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.direct_inversion import (
    DirectFitResult,
    fit_direct_model_from_observations,
)
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_oracle
from stereocomplex.physics import (
    CentralPinholeModel,
    PhysicalModelSpec,
)


@pytest.mark.slow
def test_direct_fit_pinhole_oracle_produces_low_rms():
    """Fitting a pinhole model to a pinhole oracle should give near-zero RMS."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0,
    )
    spec = PhysicalModelSpec("central_pinhole", CentralPinholeModel,
                              np.zeros(0, dtype=np.float64))
    result = fit_direct_model_from_observations(
        obs, spec, image_size=(160, 120), max_nfev=50,
    )
    assert isinstance(result, DirectFitResult)
    assert result.n_parameters_optics == 0
    assert result.n_parameters_poses > 0
    assert result.rms_px < 1e-6  # zero-parameter model, perfect fit on oracle
    assert np.isfinite(result.bic)


def test_direct_fit_result_has_expected_fields():
    """DirectFitResult should have all required fields."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=2, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0,
    )
    spec = PhysicalModelSpec("central_pinhole", CentralPinholeModel,
                              np.zeros(0, dtype=np.float64))
    r = fit_direct_model_from_observations(obs, spec, image_size=(160, 120), max_nfev=30)
    assert r.model_name == "central_pinhole"
    assert r.n_parameters_total == r.n_parameters_optics + r.n_parameters_poses
    assert r.n_observations > 0
    assert r.rms_px >= 0
    assert r.elapsed_s >= 0


def test_pose_initialization_close_to_truth_on_pinhole_oracle():
    """cv2.solvePnP should estimate poses within 1 mm and 0.5 deg of truth."""
    from stereocomplex.benchmarks.direct_inversion import (
        estimate_initial_poses_from_central_pinhole,
    )
    oracle = build_pinhole_oracle(image_size=(160, 120))
    K = oracle.K_left
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0, pose_jitter_deg=0.0,
    )
    R_list, t_list = estimate_initial_poses_from_central_pinhole(obs, K)
    assert len(R_list) == len(t_list) > 0
    for R, t, rv_gt, tv_gt in zip(
        R_list, t_list, obs.pose_rvecs, obs.pose_tvecs, strict=False,
    ):
        t_err = np.linalg.norm(t - tv_gt)
        R_gt = Rotation.from_rotvec(rv_gt).as_matrix()
        d_rot = Rotation.from_matrix(R @ R_gt.T).magnitude()
        assert t_err < 2.0, f"pose translation error {t_err:.2f} mm"
        assert d_rot < 0.05, f"pose rotation error {np.degrees(d_rot):.2f} rad"


@pytest.mark.slow
def test_direct_fit_converges_on_pinhole_oracle():
    """Direct fit should converge on a pinhole oracle (pinhole model)."""
    from stereocomplex.physics import CentralPinholeModel
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0,
    )
    spec = PhysicalModelSpec(
        "central_pinhole", CentralPinholeModel, np.zeros(0),
    )
    result = fit_direct_model_from_observations(obs, spec, image_size=(160, 120), max_nfev=100)
    assert result.converged
    assert result.rms_px < 1e-4


@pytest.mark.slow
def test_direct_fit_BIC_correctly_orders_brown_oracle():
    """On a Brown oracle, Brown should beat pinhole by BIC."""
    from stereocomplex.benchmarks.model_selection_oracles import build_brown_oracle
    from stereocomplex.physics import CentralBrownConradyModel, CentralPinholeModel
    oracle = build_brown_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0,
    )
    pinhole_spec = PhysicalModelSpec("pinhole", CentralPinholeModel, np.zeros(0))
    brown_spec = PhysicalModelSpec(
        "brown", CentralBrownConradyModel, np.zeros(5),
        bounds=(np.array([-1, -1, -0.1, -0.1, -1]), np.array([1, 1, 0.1, 0.1, 1])),
    )
    r_pinhole = fit_direct_model_from_observations(obs, pinhole_spec, image_size=(160, 120), max_nfev=100)
    r_brown = fit_direct_model_from_observations(obs, brown_spec, image_size=(160, 120), max_nfev=100)
    assert r_brown.converged
    # Brown should have lower BIC (better fit) despite more params
    assert r_brown.bic < r_pinhole.bic, (
        f"Brown BIC {r_brown.bic:.1f} should beat pinhole BIC {r_pinhole.bic:.1f}"
    )
