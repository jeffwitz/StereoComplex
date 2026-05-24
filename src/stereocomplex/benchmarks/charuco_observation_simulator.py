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
class SamplingDiagnostics:
    """Rejection-sampling statistics for the observation simulator."""

    n_poses_requested: int
    n_poses_accepted: int
    n_attempts_used: int
    mean_corners_per_frame: float
    min_corners: int
    max_corners: int


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
    diagnostics: SamplingDiagnostics | None = None

    def to_multi_camera(self) -> MultiCameraCharucoObservationSet:
        """Convert stereo observations to multi-camera format."""
        return MultiCameraCharucoObservationSet(
            object_points_mm=self.object_points_mm,
            pose_rvecs=self.pose_rvecs,
            pose_tvecs=self.pose_tvecs,
            pixels_by_channel={"left": self.left_pixels, "right": self.right_pixels},
            point_indices=self.point_indices,
            noise_std_px=self.noise_std_px,
            image_size=self.image_size,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True)
class MultiCameraCharucoObservationSet:
    """Synthetic ChArUco observations for an ordered set of camera channels."""

    object_points_mm: np.ndarray
    pose_rvecs: np.ndarray
    pose_tvecs: np.ndarray
    pixels_by_channel: dict[str, list[np.ndarray]]
    point_indices: list[np.ndarray]
    noise_std_px: float
    image_size: tuple[int, int]
    diagnostics: SamplingDiagnostics | None = None

    def __post_init__(self) -> None:
        if not self.pixels_by_channel:
            raise ValueError("at least one camera channel is required")
        expected = len(self.point_indices)
        for name, pixels in self.pixels_by_channel.items():
            if not name:
                raise ValueError("camera channel names must be non-empty")
            if len(pixels) != expected:
                raise ValueError(f"channel {name!r} has {len(pixels)} frames, expected {expected}")

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Channel names in insertion order."""
        return tuple(self.pixels_by_channel)

    @property
    def n_channels(self) -> int:
        """Total number of channels in the observations."""
        return len(self.pixels_by_channel)

    @property
    def n_poses(self) -> int:
        """Number of board poses captured."""
        return len(self.point_indices)

    def pixels(self, channel: str) -> list[np.ndarray]:
        """Pixel coordinates for a given channel and pose."""
        return self.pixels_by_channel[channel]


def _make_board_points(squares_x: int, squares_y: int, square_size_mm: float) -> np.ndarray:
    """Return (M, 3) array of ChArUco inner corner coordinates (z = 0)."""
    pts = [[float(ix) * square_size_mm, float(iy) * square_size_mm, 0.0]
           for iy in range(squares_y - 1) for ix in range(squares_x - 1)]
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
        tx = rng.uniform(-5.0, 5.0)
        ty = rng.uniform(-5.0, 5.0)
        tz = z_distance_mm + rng.uniform(-5.0, 5.0)
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
    min_corners_per_frame: int = 30,
    max_pose_attempts: int = 200,
    pose_jitter_deg: float = 5.0,
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
    min_corners_per_frame : int
        Minimum visible corners required to accept a pose.
    max_pose_attempts : int
        Maximum number of pose samples before giving up.
    pose_jitter_deg : float
        Max random rotation (degrees) of the board around its own normal axis.

    Returns
    -------
    CharucoObservationSet
    """
    rng = np.random.default_rng(seed)
    from scipy.spatial.transform import Rotation

    # Board points in local frame
    obj_pts = _make_board_points(squares_x, squares_y, square_size_mm)

    # Rejection sampling of poses
    accepted_rvecs: list[np.ndarray] = []
    accepted_tvecs: list[np.ndarray] = []
    left_pixels: list[np.ndarray] = []
    right_pixels: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    n_attempts = 0
    corner_counts: list[int] = []

    pose_seed = seed
    while len(accepted_rvecs) < n_poses and n_attempts < max_pose_attempts:
        n_attempts += 1
        pose_seed += 1
        # Sample one pose centered in the FOV
        rvecs_one, tvecs_one = _make_pose_sweep(1, z_distance_mm, seed=pose_seed)
        rv_single, tv_single = rvecs_one[0], tvecs_one[0]

        R = Rotation.from_rotvec(rv_single).as_matrix()
        if pose_jitter_deg > 0:
            jitter_rad = np.deg2rad(rng.uniform(-pose_jitter_deg, pose_jitter_deg))
            c, s = np.cos(jitter_rad), np.sin(jitter_rad)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R = R @ Rz
        t = np.asarray(tv_single, dtype=np.float64).reshape(3)
        world_pts = (R @ obj_pts.T).T + t[None, :]

        uv_left_list, uv_right_list, idx_list = [], [], []
        for k in range(obj_pts.shape[0]):
            X = world_pts[k]
            uvL, okL, distL = project_point_by_rayfield_inverse(
                left_field,
                X,
                image_size,
                max_nfev=80,
            )
            uvR, okR, distR = project_point_by_rayfield_inverse(
                right_field,
                X,
                image_size,
                max_nfev=80,
            )
            if okL and okR and distL < 1e-2 and distR < 1e-2:
                uv_left_list.append(uvL)
                uv_right_list.append(uvR)
                idx_list.append(k)

        n_vis = len(uv_left_list)
        corner_counts.append(n_vis)
        if n_vis >= min_corners_per_frame:
            accepted_rvecs.append(np.asarray(rv_single, dtype=np.float64))
            accepted_tvecs.append(np.asarray(tv_single, dtype=np.float64))
            left_pixels.append(np.array(uv_left_list, dtype=np.float64))
            right_pixels.append(np.array(uv_right_list, dtype=np.float64))
            point_indices.append(np.array(idx_list, dtype=int))

    # Build diagnostics
    if corner_counts:
        diag = SamplingDiagnostics(
            n_poses_requested=n_poses,
            n_poses_accepted=len(accepted_rvecs),
            n_attempts_used=n_attempts,
            mean_corners_per_frame=float(np.mean(corner_counts)),
            min_corners=int(np.min(corner_counts)),
            max_corners=int(np.max(corner_counts)),
        )
    else:
        diag = SamplingDiagnostics(
            n_poses_requested=n_poses,
            n_poses_accepted=0,
            n_attempts_used=n_attempts,
            mean_corners_per_frame=0.0,
            min_corners=0,
            max_corners=0,
        )

    rvecs = accepted_rvecs  # already np arrays
    tvecs = accepted_tvecs  # already np arrays

    # Apply dropout (on accepted poses)
    if dropout_rate > 0:
        for i in range(len(left_pixels)):
            n_vis = left_pixels[i].shape[0]
            keep = rng.random(n_vis) > dropout_rate
            left_pixels[i] = left_pixels[i][keep]
            right_pixels[i] = right_pixels[i][keep]
            point_indices[i] = point_indices[i][keep]

    # Apply pixel noise (only to non-empty arrays)
    if noise_std_px > 0:
        for i in range(len(left_pixels)):
            if left_pixels[i].size > 0:
                left_pixels[i] = left_pixels[i] + rng.normal(
                    scale=noise_std_px,
                    size=left_pixels[i].shape,
                ).astype(np.float64)
                right_pixels[i] = right_pixels[i] + rng.normal(
                    scale=noise_std_px,
                    size=right_pixels[i].shape,
                ).astype(np.float64)

    return CharucoObservationSet(
        object_points_mm=obj_pts,
        pose_rvecs=np.array(rvecs, dtype=np.float64)
        if rvecs
        else np.empty((0, 3), dtype=np.float64),
        pose_tvecs=np.array(tvecs, dtype=np.float64)
        if tvecs
        else np.empty((0, 3), dtype=np.float64),
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        point_indices=point_indices,
        noise_std_px=float(noise_std_px),
        image_size=image_size,
        diagnostics=diag,
    )


def simulate_charuco_observations_from_camera_fields(
    fields_by_channel: dict[str, object],
    squares_x: int = 5,
    squares_y: int = 4,
    square_size_mm: float = 20.0,
    n_poses: int = 8,
    z_distance_mm: float = 100.0,
    image_size: tuple[int, int] = (160, 120),
    noise_std_px: float = 0.0,
    dropout_rate: float = 0.0,
    seed: int = 42,
    min_corners_per_frame: int = 30,
    max_pose_attempts: int = 200,
    pose_jitter_deg: float = 5.0,
) -> MultiCameraCharucoObservationSet:
    """Simulate ChArUco observations for an arbitrary named camera set."""
    if not fields_by_channel:
        raise ValueError("at least one camera field is required")

    rng = np.random.default_rng(seed)
    from scipy.spatial.transform import Rotation

    obj_pts = _make_board_points(squares_x, squares_y, square_size_mm)
    accepted_rvecs: list[np.ndarray] = []
    accepted_tvecs: list[np.ndarray] = []
    pixels_by_channel: dict[str, list[np.ndarray]] = {name: [] for name in fields_by_channel}
    point_indices: list[np.ndarray] = []
    n_attempts = 0
    corner_counts: list[int] = []

    pose_seed = seed
    while len(accepted_rvecs) < n_poses and n_attempts < max_pose_attempts:
        n_attempts += 1
        pose_seed += 1
        rvecs_one, tvecs_one = _make_pose_sweep(1, z_distance_mm, seed=pose_seed)
        rv_single, tv_single = rvecs_one[0], tvecs_one[0]

        R = Rotation.from_rotvec(rv_single).as_matrix()
        if pose_jitter_deg > 0:
            jitter_rad = np.deg2rad(rng.uniform(-pose_jitter_deg, pose_jitter_deg))
            c, s = np.cos(jitter_rad), np.sin(jitter_rad)
            R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        t = np.asarray(tv_single, dtype=np.float64).reshape(3)
        world_pts = (R @ obj_pts.T).T + t[None, :]

        uv_lists: dict[str, list[np.ndarray]] = {name: [] for name in fields_by_channel}
        idx_list: list[int] = []
        for k, point in enumerate(world_pts):
            projected: dict[str, np.ndarray] = {}
            visible = True
            for name, field in fields_by_channel.items():
                uv, ok, dist = project_point_by_rayfield_inverse(
                    field, point, image_size, max_nfev=80
                )
                if not ok or dist >= 1e-2:
                    visible = False
                    break
                projected[name] = uv
            if visible:
                for name, uv in projected.items():
                    uv_lists[name].append(uv)
                idx_list.append(k)

        n_vis = len(idx_list)
        corner_counts.append(n_vis)
        if n_vis >= min_corners_per_frame:
            accepted_rvecs.append(np.asarray(rv_single, dtype=np.float64))
            accepted_tvecs.append(np.asarray(tv_single, dtype=np.float64))
            for name in fields_by_channel:
                pixels_by_channel[name].append(np.array(uv_lists[name], dtype=np.float64))
            point_indices.append(np.array(idx_list, dtype=int))

    if dropout_rate > 0:
        for i in range(len(point_indices)):
            n_vis = point_indices[i].shape[0]
            keep = rng.random(n_vis) > dropout_rate
            point_indices[i] = point_indices[i][keep]
            for name in fields_by_channel:
                pixels_by_channel[name][i] = pixels_by_channel[name][i][keep]

    if noise_std_px > 0:
        for i in range(len(point_indices)):
            for name in fields_by_channel:
                if pixels_by_channel[name][i].size > 0:
                    pixels_by_channel[name][i] = pixels_by_channel[name][i] + rng.normal(
                        scale=noise_std_px,
                        size=pixels_by_channel[name][i].shape,
                    ).astype(np.float64)

    diag = SamplingDiagnostics(
        n_poses_requested=n_poses,
        n_poses_accepted=len(accepted_rvecs),
        n_attempts_used=n_attempts,
        mean_corners_per_frame=float(np.mean(corner_counts)) if corner_counts else 0.0,
        min_corners=int(np.min(corner_counts)) if corner_counts else 0,
        max_corners=int(np.max(corner_counts)) if corner_counts else 0,
    )
    return MultiCameraCharucoObservationSet(
        object_points_mm=obj_pts,
        pose_rvecs=np.array(accepted_rvecs, dtype=np.float64)
        if accepted_rvecs
        else np.empty((0, 3), dtype=np.float64),
        pose_tvecs=np.array(accepted_tvecs, dtype=np.float64)
        if accepted_tvecs
        else np.empty((0, 3), dtype=np.float64),
        pixels_by_channel=pixels_by_channel,
        point_indices=point_indices,
        noise_std_px=float(noise_std_px),
        image_size=image_size,
        diagnostics=diag,
    )
