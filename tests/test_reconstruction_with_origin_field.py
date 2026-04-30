from __future__ import annotations

import numpy as np

from stereocomplex.benchmarks.parallel_plate_origin_field import (
    make_default_parallel_plate_dataset,
    run_parallel_plate_rendered_image_benchmark,
)
from stereocomplex.calibration.fit_zernike_origin_field import fit_stereo_zernike_origin_field
from stereocomplex.metrics.reconstruction_metrics import (
    compare_3d_reconstruction_with_without_origin_field,
    oracle_reconstruction_floor_report,
    triangulate_two_rays,
)
from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig
from stereocomplex.rayfields.zernike_origin_field import ZernikeRayField
from stereocomplex.synthetic.parallel_plate import transform_points


def test_triangulate_two_perfect_rays():
    P = np.array([20.0, -10.0, 500.0])
    O_left = np.array([0.0, 0.0, 0.0])
    O_right = np.array([80.0, 0.0, 0.0])
    d_left = P - O_left
    d_left /= np.linalg.norm(d_left)
    d_right = P - O_right
    d_right /= np.linalg.norm(d_right)
    P_hat, gap = triangulate_two_rays(O_left, d_left, O_right, d_right)
    assert np.linalg.norm(P_hat - P) < 1e-9
    assert gap < 1e-9


def test_oracle_parallel_plate_rays_reconstruct_dataset_points():
    dataset = make_default_parallel_plate_dataset(noise_std_px=0.0)
    T_RL = dataset.T_right_left
    R_RL = T_RL[:3, :3]
    t_RL = T_RL[:3, 3]
    left_oracle = dataset.oracle_left_ray_function
    right_oracle = dataset.oracle_right_ray_function
    assert left_oracle is not None
    assert right_oracle is not None
    errors = []
    gaps = []
    for T_board_world, uvL, uvR in zip(dataset.board_poses, dataset.left_pixels, dataset.right_pixels, strict=True):
        truth = transform_points(dataset.T_left_world, transform_points(T_board_world, dataset.object_points))
        OL, dL = left_oracle(uvL[:, 0], uvL[:, 1])
        OR_R, dR_R = right_oracle(uvR[:, 0], uvR[:, 1])
        OR_L = (R_RL.T @ (OR_R - t_RL.reshape(1, 3)).T).T
        dR_L = (R_RL.T @ dR_R.T).T
        for i in range(uvL.shape[0]):
            P_hat, gap = triangulate_two_rays(OL[i], dL[i], OR_L[i], dR_L[i])
            errors.append(np.linalg.norm(P_hat - truth[i]))
            gaps.append(gap)
    assert float(np.sqrt(np.mean(np.asarray(errors) ** 2))) < 1e-6
    assert float(np.sqrt(np.mean(np.asarray(gaps) ** 2))) < 1e-6


def test_oracle_noisy_pixel_floor_is_reported():
    dataset = make_default_parallel_plate_dataset(noise_std_px=0.05)
    clean_dataset = make_default_parallel_plate_dataset(noise_std_px=0.0)
    floor = oracle_reconstruction_floor_report(dataset_observed=dataset, dataset_clean=clean_dataset)

    assert floor.oracle_clean_pixels.rms_3d < 1e-6
    assert 0.5 < floor.oracle_observed_pixels.rms_3d < 1.2
    assert 0.04 < floor.oracle_observed_pixels.ray_gap_rms < 0.15
    assert floor.oracle_observed_pixels.n_points == floor.oracle_clean_pixels.n_points


def test_identified_origin_field_improves_reconstruction_against_central_model():
    dataset = make_default_parallel_plate_dataset(noise_std_px=0.0)
    config = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=4)
    fit_result = fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=dataset.K_left,
        K_right=dataset.K_right,
        T_right_left_initial=dataset.T_right_left,
        board_poses_initial=dataset.board_poses,
        config_left=config,
        config_right=config,
        regularization=1e-6,
    )
    report = compare_3d_reconstruction_with_without_origin_field(dataset, None, fit_result)
    ratio = report.with_origin_field.rms_3d / report.central.rms_3d
    assert fit_result.success
    assert ratio < 0.5


def test_full_rayfield_pose_and_rig_ba_reaches_clean_reconstruction_scale():
    dataset = make_default_parallel_plate_dataset(noise_std_px=0.0)
    config = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=3)
    fit_result = fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=dataset.K_left,
        K_right=dataset.K_right,
        T_right_left_initial=dataset.T_right_left,
        board_poses_initial=dataset.board_poses,
        config_left=config,
        config_right=config,
        optimize_directions=True,
        optimize_board_poses=True,
        optimize_stereo_extrinsics=True,
        regularization=1e-5,
        direction_regularization=1e-2,
        pose_regularization=10.0,
        rig_regularization=100.0,
        max_nfev=80,
    )
    report = compare_3d_reconstruction_with_without_origin_field(dataset, None, fit_result)
    assert fit_result.success
    assert isinstance(fit_result.left_field, ZernikeRayField)
    assert np.linalg.norm(fit_result.stereo_transform[:3, 3] - dataset.T_right_left[:3, 3]) < 1e-3
    assert report.with_origin_field.rms_3d < 0.1


def test_rendered_image_detection_pipeline_improves_over_central_model(tmp_path):
    report = run_parallel_plate_rendered_image_benchmark(tmp_path / "rendered", max_order=3)

    assert report.fit_result.success
    assert report.n_frames >= 10
    assert report.n_common_corners >= 50
    assert report.n_points_total >= 500
    assert report.rendered.left_images[0].exists()
    assert report.rendered.right_images[0].exists()
    assert report.reconstruction_comparison.with_origin_field.rms_3d < report.reconstruction_comparison.central.rms_3d
    assert report.reconstruction_comparison.with_origin_field.rms_3d <= 1.1 * report.oracle_detected.rms_3d
