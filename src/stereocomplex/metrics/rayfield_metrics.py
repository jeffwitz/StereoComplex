from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np


@dataclass(frozen=True)
class RayfieldComparisonReport:
    """Summary of rayfield comparison between two models.

    Attributes
    ----------
    plane_intersection_rms : float
        RMS of 3D intersection point distances at the z-plane, in mm.
    plane_intersection_median : float
        Median intersection point distance at the z-plane, in mm.
    plane_intersection_p95 : float
        95th percentile intersection distance at the z-plane, in mm.
    direction_angle_rms_deg : float
        RMS angular difference between ray directions, in degrees.
    n_samples : int
        Number of rays compared.
    """
    plane_intersection_rms: float
    plane_intersection_median: float
    plane_intersection_p95: float
    direction_angle_rms_deg: float
    n_samples: int


def intersect_rays_with_z_plane(
    origin: np.ndarray, direction: np.ndarray, z_plane: float
) -> np.ndarray:
    """Intersect rays with a horizontal z-plane.

    Parameters
    ----------
    origin : np.ndarray
        Ray origins, shape (N, 3).
    direction : np.ndarray
        Ray directions (unit vectors), shape (N, 3).
    z_plane : float
        Z-coordinate of the target plane.

    Returns
    -------
    np.ndarray
        Intersection points, shape (N, 3).

    Raises
    ------
    ValueError
        If any ray is parallel to the z-plane (z-component < 1e-12).
    """
    origin_arr = np.asarray(origin, dtype=np.float64).reshape(-1, 3)
    direction_arr = np.asarray(direction, dtype=np.float64).reshape(-1, 3)
    denom = direction_arr[:, 2]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("ray is parallel to z plane")
    lam = (float(z_plane) - origin_arr[:, 2]) / denom
    return origin_arr + lam[:, None] * direction_arr


def compare_rayfields_on_planes(
    fitted_field,
    oracle_ray_function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    z_planes: tuple[float, float],
    grid_shape: tuple[int, int] = (25, 19),
) -> RayfieldComparisonReport:
    """
    Compare two ray fields by intersecting each ray with two reference z planes.

    This avoids comparing raw origins that may be expressed in different gauges.
    """
    width, height = image_size
    nx, ny = grid_shape
    u = np.linspace(0.0, float(width - 1), int(nx))
    v = np.linspace(0.0, float(height - 1), int(ny))
    uu, vv = np.meshgrid(u, v)
    uf = uu.reshape(-1)
    vf = vv.reshape(-1)

    O_fit, d_fit = fitted_field.ray(uf, vf)
    O_true, d_true = oracle_ray_function(uf, vf)
    O_true = np.asarray(O_true, dtype=np.float64).reshape(-1, 3)
    d_true = np.asarray(d_true, dtype=np.float64).reshape(-1, 3)

    errors: list[np.ndarray] = []
    for z in z_planes:
        A_fit = intersect_rays_with_z_plane(O_fit, d_fit, z)
        A_true = intersect_rays_with_z_plane(O_true, d_true, z)
        errors.append(np.linalg.norm(A_fit - A_true, axis=1))
    plane_err = np.sqrt(errors[0] ** 2 + errors[1] ** 2)

    dots = np.sum(d_fit * d_true, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(dots))
    return RayfieldComparisonReport(
        plane_intersection_rms=float(np.sqrt(np.mean(plane_err**2))),
        plane_intersection_median=float(np.median(plane_err)),
        plane_intersection_p95=float(np.percentile(plane_err, 95)),
        direction_angle_rms_deg=float(np.sqrt(np.mean(angles_deg**2))),
        n_samples=int(uf.size),
    )
