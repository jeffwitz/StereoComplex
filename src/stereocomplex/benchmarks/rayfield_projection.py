"""Inverse-project 3D points through generic rayfields.

Given a rayfield :math:`\\mathcal{R}(u,v) = (O(u,v), d(u,v))` and a 3-D
point :math:`X`, find the pixel :math:`(u,v)` whose ray passes closest to
:math:`X`.  This is the inverse of the usual ray evaluation and is needed
for direct ChArUco model fitting (pipeline A in the direct-vs-rayfield
study).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares  # type: ignore

Array = np.ndarray


def point_ray_residual(uv: Array, field, X: Array) -> Array:
    """Perpendicular distance vector from point *X* to the ray at pixel *uv*.

    Parameters
    ----------
    uv : (2,) ndarray  — pixel coordinates ``(u, v)``.
    field :
        Object with a ``.ray(u, v)`` method returning ``(origins, directions)``.
    X : (3,) ndarray  — 3-D point in the same coordinate frame as the ray.

    Returns
    -------
    r : (3,) ndarray  — ``(I - d·dᵀ)(X - O)``, the vector from the ray to X.
    """
    u_arr = np.asarray([float(uv[0])], dtype=np.float64)
    v_arr = np.asarray([float(uv[1])], dtype=np.float64)
    O, d = field.ray(u_arr, v_arr)  # noqa: E741
    O_flat = np.asarray(O, dtype=np.float64).reshape(-1, 3)
    d_flat = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    O_vec = O_flat[0]
    d_vec = d_flat[0]
    delta = np.asarray(X, dtype=np.float64).reshape(3) - O_vec
    proj = np.dot(delta, d_vec) * d_vec
    return delta - proj


def project_point_by_rayfield_inverse(
    field,
    X: Array,
    image_size: tuple[int, int],
    *,
    initial_uv: Array | None = None,
    max_nfev: int = 100,
) -> tuple[Array, bool, float]:
    """Find the pixel whose ray passes closest to a 3-D point.

    Parameters
    ----------
    field :
        Rayfield with ``.ray(u, v)``.
    X : (3,) ndarray  — 3-D point in the rayfield's coordinate frame.
    image_size : (W, H)  — sensor dimensions in pixels.
    initial_uv : (2,) ndarray | None  — starting guess.  Defaults to the
        image centre ``(cx, cy)``.
    max_nfev : int  — max function evaluations for the optimiser.

    Returns
    -------
    uv : (2,) ndarray  — ``(u, v)`` of the best-fit pixel.
    success : bool  — whether the optimiser converged and the result is
        within the image bounds.
    distance_mm : float  — final point-to-line distance in millimetres.
    """
    W, H = int(image_size[0]), int(image_size[1])
    x0 = np.asarray(initial_uv, dtype=np.float64).reshape(2) if initial_uv is not None else np.array([(W - 1) / 2, (H - 1) / 2], dtype=np.float64)

    def fun(uv):
        return point_ray_residual(uv, field, X)

    bounds = (np.array([0.0, 0.0]), np.array([float(W - 1), float(H - 1)]))
    sol = least_squares(
        fun, x0=x0, bounds=bounds, method="trf",
        max_nfev=int(max_nfev), xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )
    uv_opt = np.asarray(sol.x, dtype=np.float64).reshape(2)
    dist = float(np.linalg.norm(sol.fun))
    u_ok = 0.0 <= uv_opt[0] <= float(W - 1)
    v_ok = 0.0 <= uv_opt[1] <= float(H - 1)
    # Allow a tiny margin for the bounds
    success = bool(sol.success) and u_ok and v_ok
    return uv_opt, success, dist


def project_points_by_rayfield_inverse(
    field,
    points: Array,
    image_size: tuple[int, int],
    *,
    initial_uvs: Array | None = None,
    max_nfev: int = 100,
) -> tuple[Array, Array, Array]:
    """Batch version of :func:`project_point_by_rayfield_inverse`.

    Parameters
    ----------
    field :
        Rayfield with ``.ray(u, v)``.
    points : (N, 3) ndarray  — 3-D points.
    image_size : (W, H)
    initial_uvs : (N, 2) ndarray | None  — initial guesses per point.
    max_nfev : int

    Returns
    -------
    uv : (N, 2) ndarray  — fitted pixel coordinates.
    success : (N,) bool ndarray
    distances_mm : (N,) ndarray
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    N = pts.shape[0]
    if initial_uvs is None:
        W, H = int(image_size[0]), int(image_size[1])
        initial_uvs = np.full((N, 2), [(W - 1) / 2, (H - 1) / 2], dtype=np.float64)
    else:
        initial_uvs = np.asarray(initial_uvs, dtype=np.float64).reshape(-1, 2)

    uv_out = np.empty((N, 2), dtype=np.float64)
    success_out = np.empty(N, dtype=bool)
    dist_out = np.empty(N, dtype=np.float64)

    for k in range(N):
        uv_out[k], success_out[k], dist_out[k] = project_point_by_rayfield_inverse(
            field, pts[k], image_size,
            initial_uv=initial_uvs[k], max_nfev=max_nfev,
        )

    return uv_out, success_out, dist_out
