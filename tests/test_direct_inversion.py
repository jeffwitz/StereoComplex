"""Smoke tests for direct ChArUco model inversion."""

from __future__ import annotations
import pytest

import numpy as np

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
