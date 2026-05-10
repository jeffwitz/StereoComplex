"""Smoke tests for the ChArUco observation simulator."""

from __future__ import annotations

import numpy as np

from stereocomplex.benchmarks.charuco_observation_simulator import (
    _make_board_points,
    _make_pose_sweep,
    CharucoObservationSet,
    simulate_charuco_observations_from_rayfield,
)
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
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
    )
    assert isinstance(obs, CharucoObservationSet)
    assert obs.noise_std_px == 0.0
    assert obs.image_size == (160, 120)
    assert obs.object_points_mm.shape[1] == 3
    assert len(obs.left_pixels) == 4
    assert len(obs.right_pixels) == 4
    total_corners = sum(p.shape[0] for p in obs.left_pixels)
    assert total_corners > 0, "at least some corners should be visible"


def test_noise_adds_perturbation():
    """With noise_std_px > 0, pixels are perturbed from the oracle."""
    oracle = build_pinhole_oracle(image_size=(160, 120))
    obs_clean = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.0, seed=42,
    )
    obs_noisy = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=(160, 120), n_poses=4, noise_std_px=0.5, seed=42,
    )
    # At least some pixels should differ
    diffs = []
    for Lc, Ln in zip(obs_clean.left_pixels, obs_noisy.left_pixels, strict=True):
        if Lc.shape[0] > 0 and Ln.shape[0] > 0:
            diffs.append(np.max(np.abs(Lc[: min(len(Lc), len(Ln))] - Ln[: min(len(Lc), len(Ln))])))
    assert len(diffs) > 0
    assert max(diffs) > 0.01
