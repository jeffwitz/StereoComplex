from __future__ import annotations
import pytest

import numpy as np

from stereocomplex.advanced import fit_physical_model_to_rayfield
from stereocomplex.physics import (
    CentralBrownConradyModel,
    CentralPinholeModel,
    PinholeParallelPlateFitParams,
    PinholeParallelPlateModel,
    PhysicalModelSpec,
    select_physical_model_from_rayfield,
)
from stereocomplex.physics.central_models import brown_conrady_distort_normalized, undistort_brown_normalized
from stereocomplex.physics.parallel_plate_fit import rayfield_two_plane_residuals
from stereocomplex.synthetic.parallel_plate import ParallelPlateSyntheticParams, parallel_plate_ray_from_pixel


class _OracleField:
    def __init__(self, K: np.ndarray, params: ParallelPlateSyntheticParams):
        self.K = K
        self.params = params

    def ray(self, u, v):
        return parallel_plate_ray_from_pixel(u, v, self.K, self.params)


def _camera_matrix() -> np.ndarray:
    return np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_physical_candidates_share_ray_interface():
    K = _camera_matrix()
    models = [
        CentralPinholeModel(K),
        CentralBrownConradyModel(K, k1=0.01, k2=-0.005, p1=0.001, p2=-0.001, k3=0.0),
        PinholeParallelPlateModel(
            K,
            PinholeParallelPlateFitParams(alpha_deg=12.0, beta_deg=5.0, thickness_mm=8.0),
        ),
    ]
    u = np.array([10.0, 319.5, 620.0], dtype=np.float64)
    v = np.array([20.0, 239.5, 450.0], dtype=np.float64)
    for model in models:
        origins, d = model.ray(u, v)
        assert origins.shape == d.shape == (3, 3)
        assert np.all(np.isfinite(origins))
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_brown_zero_coefficients_matches_pinhole_by_ray_planes():
    K = _camera_matrix()
    pinhole = CentralPinholeModel(K)
    brown = CentralBrownConradyModel(K)
    pixels = np.array([[0.0, 0.0], [319.5, 239.5], [639.0, 479.0]], dtype=np.float64)

    residuals = rayfield_two_plane_residuals(pinhole, brown, pixels, z_planes=(100.0, 1000.0))
    assert np.sqrt(np.mean(residuals**2)) < 1e-12


@pytest.mark.slow
def test_ray_space_selection_prefers_plate_over_central_brown_on_plate_oracle():
    K = _camera_matrix()
    truth = ParallelPlateSyntheticParams(eta=1.5, thickness=8.0, alpha_deg=12.0, beta_deg=5.0, d1=80.0)
    oracle = _OracleField(K, truth)

    report = select_physical_model_from_rayfield(
        target_field=oracle,
        candidate_specs=None,
        K=K,
        image_size=(640, 480),
        grid_shape=(11, 9),
        full_grid_weight=0.0,
    )
    by_name = {candidate.model_name: candidate for candidate in report.candidates}
    plate = by_name["pinhole_parallel_plate"]
    brown = by_name["central_brown_conrady"]
    central = by_name["central_pinhole"]

    assert report.best_by_bic == "pinhole_parallel_plate"
    assert report.best_by_rms == "pinhole_parallel_plate"
    assert plate.rms_mm < brown.rms_mm
    assert plate.bic < brown.bic
    assert plate.rms_mm < central.rms_mm
    assert brown.rms_mm > 0.1
    assert abs(plate.parameter_dict["alpha_deg"] - truth.alpha_deg) < 0.2
    assert abs(plate.parameter_dict["beta_deg"] - truth.beta_deg) < 0.2
    assert abs(plate.parameter_dict["thickness_mm"] - truth.thickness) < 0.2


def test_brown_distort_undistort_roundtrip():
    """undistort_brown_normalized must invert brown_conrady_distort_normalized."""
    rng = np.random.default_rng(42)
    # Normalized coordinates within radius 0.5 (typical mid-FOV range).
    angles = rng.uniform(0, 2 * np.pi, 200)
    radii = rng.uniform(0, 0.5, 200)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    k1, k2, p1, p2, k3 = -0.1, 0.05, 0.001, -0.001, 0.0
    xd, yd = brown_conrady_distort_normalized(x, y, k1, k2, p1, p2, k3)
    x_rec, y_rec = undistort_brown_normalized(xd, yd, k1, k2, p1, p2, k3, n_iter=10)
    assert np.max(np.abs(x_rec - x)) < 1e-9
    assert np.max(np.abs(y_rec - y)) < 1e-9


@pytest.mark.slow
def test_brown_with_real_coefficients_recovers_oracle():
    """fit_physical_model_to_rayfield must recover Brown coefficients from a Brown oracle."""
    K = _camera_matrix()
    truth_coeffs = dict(k1=-0.1, k2=0.05, p1=0.001, p2=-0.001, k3=0.0)
    oracle = CentralBrownConradyModel(K, **truth_coeffs)

    result = fit_physical_model_to_rayfield(
        model_class=CentralBrownConradyModel,
        target_field=oracle,
        K=K,
        image_size=(640, 480),
        initial_parameters=np.zeros(5),
        grid_shape=(15, 11),
        full_grid_weight=0.0,
        name="central_brown_conrady",
    )

    assert result.rms_mm < 1e-6, f"residual too large: {result.rms_mm:.2e} mm"
    for param, expected in truth_coeffs.items():
        assert abs(result.parameter_dict[param] - expected) < 1e-4, (
            f"{param}: got {result.parameter_dict[param]:.6f}, expected {expected:.6f}"
        )


@pytest.mark.slow
def test_selection_prefers_brown_on_brown_oracle():
    """Model selection must choose Brown by BIC when the oracle is a Brown rayfield.

    This is the discrimination test: it proves the framework is not biased toward
    the plate model and can correctly identify a radially-symmetric distortion pattern.
    """
    K = _camera_matrix()
    # Pure radial distortion (no tangential) — structurally different from a plate.
    oracle = CentralBrownConradyModel(K, k1=-0.1, k2=0.0, p1=0.0, p2=0.0, k3=0.0)

    report = select_physical_model_from_rayfield(
        target_field=oracle,
        candidate_specs=None,
        K=K,
        image_size=(640, 480),
        grid_shape=(11, 9),
        full_grid_weight=0.0,
    )

    by_name = {c.model_name: c for c in report.candidates}
    assert report.best_by_bic == "central_brown_conrady", (
        f"Expected brown to win by BIC, got {report.best_by_bic}. "
        f"BICs: {[(n, f'{c.bic:.1f}') for n, c in by_name.items()]}"
    )
