"""Smoke tests for the ChArUco observation simulator."""

from __future__ import annotations

import numpy as np

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
    MultiCameraCharucoObservationSet,
    _make_board_points,
    _make_pose_sweep,
    simulate_charuco_observations_from_camera_fields,
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_n_camera_oracle
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_oracle


def test_make_board_points():
    pts = _make_board_points(5, 4, 20.0)
    # 4×3 = 12 inner corners for a 5×4 board
    assert pts.shape == (12, 3)
    assert pts[0, 2] == 0.0  # z = 0 (board plane)


def test_make_pose_sweep():
    rvecs, tvecs = _make_pose_sweep(8, 100.0, seed=42)
    assert len(rvecs) == 8
    assert len(tvecs) == 8
    assert all(rv.shape == (3,) for rv in rvecs)
    assert all(tv.shape == (3,) for tv in tvecs)


def test_simulate_zero_noise_observations():
    """With zero noise, the simulator should produce observations."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,  # disable rejection for backward compat
    )
    assert isinstance(obs, CharucoObservationSet)
    assert obs.noise_std_px == 0.0
    assert obs.image_size == (160, 120)
    assert obs.object_points_mm.shape[1] == 3
    assert len(obs.left_pixels) == 4
    assert len(obs.right_pixels) == 4
    total_corners = sum(p.shape[0] for p in obs.left_pixels)
    assert total_corners > 0, "at least some corners should be visible"


def test_stereo_observation_set_converts_to_multi_camera():
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=3,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    )
    multi = obs.to_multi_camera()

    assert isinstance(multi, MultiCameraCharucoObservationSet)
    assert multi.channel_names == ("left", "right")
    assert multi.n_channels == 2
    assert multi.n_poses == len(obs.point_indices)
    assert multi.pixels("left") is obs.left_pixels
    assert multi.pixels("right") is obs.right_pixels


def test_simulate_multi_camera_observations_from_four_pinhole_fields():
    oracle = build_pinhole_n_camera_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_camera_fields(
        oracle.fields_by_channel,
        image_size=oracle.image_size,
        n_poses=3,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    )

    assert isinstance(obs, MultiCameraCharucoObservationSet)
    assert obs.channel_names == oracle.channel_names
    assert obs.n_channels == 4
    assert obs.n_poses == 3
    assert all(len(obs.pixels(name)) == 3 for name in obs.channel_names)


def test_noise_adds_perturbation():
    """With noise_std_px > 0, pixels are perturbed from the oracle."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs_clean = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
    )
    obs_noisy = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.5,
        seed=42,
        min_corners_per_frame=0,
    )
    # At least some pixels should differ
    diffs = []
    for Lc, Ln in zip(obs_clean.left_pixels, obs_noisy.left_pixels, strict=True):
        if Lc.shape[0] > 0 and Ln.shape[0] > 0:
            diffs.append(np.max(np.abs(Lc[: min(len(Lc), len(Ln))] - Ln[: min(len(Lc), len(Ln))])))
    assert len(diffs) > 0
    assert max(diffs) > 0.01


def test_simulator_meets_min_corners_per_frame_on_cmo_oracle():
    """With min_corners_per_frame=20, all accepted poses have enough corners."""
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
        min_corners_per_frame=5,
        seed=42,
    )
    assert obs.diagnostics is not None
    diag = obs.diagnostics
    assert diag.n_poses_accepted >= 2, (
        f"CMO narrow FOV: expected ≥2 accepted poses, got {diag.n_poses_accepted}"
    )
    assert diag.n_attempts_used <= 200


def test_simulator_diagnostics_report_correct_counts():
    """SamplingDiagnostics must match the actual observation set."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=3,
        min_corners_per_frame=0,
        seed=42,
    )
    diag = obs.diagnostics
    assert diag is not None
    assert diag.n_poses_accepted == len(obs.left_pixels)
    actual_corners = [p.shape[0] for p in obs.left_pixels]
    if actual_corners:
        assert diag.min_corners == min(actual_corners)
        assert diag.max_corners == max(actual_corners)


def test_simulator_returns_zero_poses_when_min_unsatisfiable():
    """With impossible min_corners, simulator returns empty set, not loop."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        min_corners_per_frame=10000,
        max_pose_attempts=20,
        seed=42,
    )
    assert obs.diagnostics is not None
    assert obs.diagnostics.n_poses_accepted == 0
    assert len(obs.left_pixels) == 0


def test_pose_jitter_changes_corner_positions():
    """pose_jitter_deg > 0 rotates the board around its normal, moving corners."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    # With zero jitter, results are reproducible across runs with the same seed
    obs_ref = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
        pose_jitter_deg=0.0,
    )
    obs_same = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
        pose_jitter_deg=0.0,
    )
    for Lr, Ls in zip(obs_ref.left_pixels, obs_same.left_pixels, strict=True):
        assert np.allclose(Lr, Ls), "zero jitter should be deterministic"

    # With large jitter, corners shift noticeably
    obs_jit = simulate_charuco_observations_from_rayfield(
        oracle.left_field,
        oracle.right_field,
        image_size=(160, 120),
        n_poses=4,
        noise_std_px=0.0,
        seed=42,
        min_corners_per_frame=0,
        pose_jitter_deg=30.0,
    )
    max_shift = 0.0
    for Lr, Lj in zip(obs_ref.left_pixels, obs_jit.left_pixels, strict=True):
        min_len = min(Lr.shape[0], Lj.shape[0])
        if min_len > 0:
            shift = np.max(np.abs(Lr[:min_len] - Lj[:min_len]))
            max_shift = max(max_shift, shift)
    assert max_shift > 0.5, (
        f"pose_jitter_deg=30 expected to shift corners by >0.5 px, got {max_shift:.3f}"
    )
