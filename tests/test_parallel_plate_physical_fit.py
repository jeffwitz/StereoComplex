from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stereocomplex.advanced import (
    compare_3d_reconstruction_with_without_origin_field,
    fit_parallel_plate_to_zernike_rayfield,
)
from stereocomplex.benchmarks.parallel_plate_origin_field import (
    make_default_parallel_plate_dataset,
    run_parallel_plate_origin_field_benchmark,
)
from stereocomplex.physics.parallel_plate_fit import (
    PinholeParallelPlateFitParams,
    PinholeParallelPlateRayField,
    rayfield_two_plane_residuals,
)
from stereocomplex.synthetic.parallel_plate import ParallelPlateSyntheticParams, parallel_plate_ray_from_pixel


def _camera_matrix() -> np.ndarray:
    return np.array([[620.0, 0.0, 319.5], [0.0, 620.0, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)


class _OracleField:
    def __init__(self, K: np.ndarray, params: ParallelPlateSyntheticParams):
        self.K = K
        self.params = params

    def ray(self, u, v):
        return parallel_plate_ray_from_pixel(u, v, self.K, self.params)


def test_parallel_plate_fit_recovers_oracle_parameters_without_noise():
    K = _camera_matrix()
    truth = ParallelPlateSyntheticParams(eta=1.5, thickness=8.0, alpha_deg=12.0, beta_deg=5.0, d1=80.0)
    result = fit_parallel_plate_to_zernike_rayfield(
        _OracleField(K, truth),
        K=K,
        image_size=(640, 480),
        eta=truth.eta,
        grid_shape=(15, 11),
        full_grid_weight=0.0,
        oracle_params=truth,
    )

    assert result.success
    assert abs(result.params.alpha_deg - truth.alpha_deg) < 0.2
    assert abs(result.params.beta_deg - truth.beta_deg) < 0.2
    assert abs(result.params.thickness_mm - truth.thickness) < 0.2
    assert result.rayfield_rms_full_mm < 1e-9


def test_parallel_plate_ray_lines_are_invariant_to_d1():
    K = _camera_matrix()
    params_a = PinholeParallelPlateFitParams(alpha_deg=12.0, beta_deg=5.0, thickness_mm=8.0, eta=1.5, d1_mm=20.0)
    params_b = PinholeParallelPlateFitParams(alpha_deg=12.0, beta_deg=5.0, thickness_mm=8.0, eta=1.5, d1_mm=140.0)
    field_a = PinholeParallelPlateRayField(K, params_a)
    field_b = PinholeParallelPlateRayField(K, params_b)
    pixels = np.array([[20.0, 30.0], [319.5, 239.5], [610.0, 450.0]], dtype=np.float64)

    residuals = rayfield_two_plane_residuals(field_a, field_b, pixels, z_planes=(100.0, 1000.0))
    assert np.sqrt(np.mean(residuals**2)) < 1e-9


def test_fitted_plate_from_zernike_rayfield_improves_reconstruction():
    dataset = make_default_parallel_plate_dataset(noise_std_px=0.0)
    report = run_parallel_plate_origin_field_benchmark(max_order=4, noise_std_px=0.0)
    support_left = np.concatenate(dataset.left_pixels, axis=0)
    support_right = np.concatenate(dataset.right_pixels, axis=0)

    plate_left = fit_parallel_plate_to_zernike_rayfield(
        report.fit_result.left_field,
        K=dataset.K_left,
        image_size=dataset.image_size,
        support_pixels=support_left,
        grid_shape=(15, 11),
        oracle_params=dataset.oracle_left_params,
    )
    plate_right = fit_parallel_plate_to_zernike_rayfield(
        report.fit_result.right_field,
        K=dataset.K_right,
        image_size=dataset.image_size,
        support_pixels=support_right,
        grid_shape=(15, 11),
        oracle_params=dataset.oracle_right_params,
    )

    plate_model = SimpleNamespace(
        left_field=PinholeParallelPlateRayField(dataset.K_left, plate_left.params),
        right_field=PinholeParallelPlateRayField(dataset.K_right, plate_right.params),
        stereo_transform=dataset.T_right_left,
    )
    plate_comparison = compare_3d_reconstruction_with_without_origin_field(dataset, None, plate_model)
    zernike_comparison = report.reconstruction_comparison

    assert plate_left.rayfield_rms_support_mm < 0.25
    assert plate_right.rayfield_rms_support_mm < 0.25
    assert plate_comparison.with_origin_field.rms_3d < 0.5
    assert plate_comparison.with_origin_field.rms_3d < plate_comparison.central.rms_3d
    assert zernike_comparison.with_origin_field.rms_3d < plate_comparison.with_origin_field.rms_3d
