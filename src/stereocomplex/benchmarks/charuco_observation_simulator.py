"""Simulate ChArUco corner observations from stereo rayfield oracles.

Given a stereo rayfield pair (left + right), a ChArUco board specification,
and a list of calibration poses, this module projects every 3-D board point
through each channel's rayfield to produce synthetic 2-D corner coordinates.
Gaussian pixel noise and random dropout can be added to mimic realistic
detection conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.benchmarks.rayfield_projection import (
    project_point_by_rayfield_inverse,
)

Array = np.ndarray


@dataclass(frozen=True)
class CharucoObservationSet:
    """Synthetic ChArUco observations for one stereo pair.

    Attributes
    ----------
    object_points_mm : (M, 3) ndarray
        3-D coordinates of all ChArUco inner corners on the board
        (in the board's local frame, z = 0).
    pose_rvecs : (P, 3) ndarray
        Rodrigues vectors for each board pose.
    pose_tvecs : (P, 3) ndarray
        Translation vectors for each board pose.
    left_pixels : list of (N_p, 2) ndarray
        Detected pixel coordinates in the left image, one array per pose.
    right_pixels : list of (N_p, 2) ndarray
        Same for the right image.
    point_indices : list of (N_p,) ndarray
        Indices into *object_points_mm* of the visible corners per pose.
    noise_std_px : float
        Standard deviation of the Gaussian noise added to pixel coordinates.
    image_size : (int, int)
        ``(width, height)`` in pixels.
    """

    object_points_mm: np.ndarray
    pose_rvecs: np.ndarray
    pose_tvecs: np.ndarray
    left_pixels: list[np.ndarray]
    right_pixels: list[np.ndarray]
    point_indices: list[np.ndarray]
    noise_std_px: float
    image_size: tuple[int, int]


def _make_board_points(squares_x: int, squares_y: int,
                       square_size_mm: float) -> np.ndarray:
    """Return (M, 3) array of ChArUco inner corner coordinates (z = 0)."""
    pts = []
    for iy in range(squares_y - 1):
        for ix in range(squares_x - 1):
            pts.append([float(ix) * square_size_mm, float(iy) * square_size_mm, 0.0])
    return np.array(pts, dtype=np.float64)


def _make_pose_sweep(
    n_poses: int,
    z_distance_mm: float,
    *,
    tilt_max_rad: float = 0.25,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate a sweep of board poses roughly facing the cameras.

    Returns lists of (rvec, tvec) pairs.
    """
    rng = np.random.default_rng(seed)
    rvecs, tvecs = [], []
    for _ in range(n_poses):
        # Small random rotation around x and y
        rx = rng.uniform(-tilt_max_rad, tilt_max_rad)
        ry = rng.uniform(-tilt_max_rad, tilt_max_rad)
        rz = rng.uniform(0, 2 * np.pi)
        # Build rotation matrix from Euler angles (rz * ry * rx)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R = Rz @ Ry @ Rx
        # Convert to Rodrigues
        from scipy.spatial.transform import Rotation
        rvec = Rotation.from_matrix(R).as_rotvec()
        # Translation: board placed at ~z_distance_mm along the camera z axis,
        # with small lateral shifts.
        tx = rng.uniform(-15.0, 15.0)
        ty = rng.uniform(-10.0, 10.0)
        tz = z_distance_mm + rng.uniform(-10.0, 10.0)
        rvecs.append(np.asarray(rvec, dtype=np.float64))
        tvecs.append(np.array([tx, ty, tz], dtype=np.float64))
    return rvecs, tvecs


def simulate_charuco_observations_from_rayfield(
    left_field,
    right_field,
    squares_x: int = 5,
    squares_y: int = 4,
    square_size_mm: float = 20.0,
    n_poses: int = 8,
    z_distance_mm: float = 100.0,
    image_size: tuple[int, int] = (160, 120),
    noise_std_px: float = 0.0,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> CharucoObservationSet:
    """Simulate ChArUco corner observations from a stereo rayfield oracle.

    Parameters
    ----------
    left_field, right_field :
        Rayfields with ``.ray(u, v)`` methods.
    squares_x, squares_y : int
        Number of ChArUco squares on the board.
    square_size_mm : float
        Size of one square in millimetres.
    n_poses : int
        Number of calibration poses to simulate.
    z_distance_mm : float
        Approximate distance from the cameras to the board.
    image_size : (W, H)
        Sensor dimensions in pixels.
    noise_std_px : float
        Gaussian pixel noise standard deviation.
    dropout_rate : float
        Fraction of corners to randomly drop (0.0 = none, 1.0 = all).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CharucoObservationSet
    """
    rng = np.random.default_rng(seed)

    # Board points in local frame
    obj_pts = _make_board_points(squares_x, squares_y, square_size_mm)
    M = obj_pts.shape[0]

    # Generate poses
    rvecs, tvecs = _make_pose_sweep(n_poses, z_distance_mm, seed=seed)

    left_pixels: list[np.ndarray] = []
    right_pixels: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []

    from scipy.spatial.transform import Rotation

    for rv, tv in zip(rvecs, tvecs, strict=True):
        R = Rotation.from_rotvec(rv).as_matrix()
        t = np.asarray(tv, dtype=np.float64).reshape(3)
        # Transform board points to world frame
        world_pts = (R @ obj_pts.T).T + t[None, :]

        uv_left_list, uv_right_list, idx_list = [], [], []
        for k in range(M):
            X = world_pts[k]
            # Project through left channel
            uvL, okL, distL = project_point_by_rayfield_inverse(
                left_field, X, image_size, max_nfev=80,
            )
            uvR, okR, distR = project_point_by_rayfield_inverse(
                right_field, X, image_size, max_nfev=80,
            )
            if okL and okR and distL < 1e-2 and distR < 1e-2:
                uv_left_list.append(uvL)
                uv_right_list.append(uvR)
                idx_list.append(k)

        if not uv_left_list:
            # No visible corners in this pose — add empty arrays
            left_pixels.append(np.empty((0, 2), dtype=np.float64))
            right_pixels.append(np.empty((0, 2), dtype=np.float64))
            point_indices.append(np.empty(0, dtype=int))
            continue

        uvL_arr = np.array(uv_left_list, dtype=np.float64)
        uvR_arr = np.array(uv_right_list, dtype=np.float64)
        idx_arr = np.array(idx_list, dtype=int)

        # Apply dropout
        if dropout_rate > 0:
            n_vis = uvL_arr.shape[0]
            keep = rng.random(n_vis) > dropout_rate
            uvL_arr = uvL_arr[keep]
            uvR_arr = uvR_arr[keep]
            idx_arr = idx_arr[keep]

        left_pixels.append(uvL_arr)
        right_pixels.append(uvR_arr)
        point_indices.append(idx_arr)

    # Apply pixel noise (only to non-empty arrays)
    if noise_std_px > 0:
        for i in range(len(left_pixels)):
            if left_pixels[i].size > 0:
                left_pixels[i] = left_pixels[i] + rng.normal(
                    scale=noise_std_px, size=left_pixels[i].shape,
                ).astype(np.float64)
                right_pixels[i] = right_pixels[i] + rng.normal(
                    scale=noise_std_px, size=right_pixels[i].shape,
                ).astype(np.float64)

    return CharucoObservationSet(
        object_points_mm=obj_pts,
        pose_rvecs=np.array(rvecs, dtype=np.float64),
        pose_tvecs=np.array(tvecs, dtype=np.float64),
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        point_indices=point_indices,
        noise_std_px=float(noise_std_px),
        image_size=image_size,
    )
