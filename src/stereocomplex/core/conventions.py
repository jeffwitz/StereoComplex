"""Explicit coordinate-frame conventions for StereoComplex.

All internal ray computations use the **OpenCV / image-aligned** frame::

    u → +X   (right)
    v → +Y   (down, following the image sensor readout direction)
    Z  forward (right-handed: X × Y = Z)

This is the convention used by OpenCV, ChArUco, ``pinhole_ray_from_pixel``,
Zernike rayfields, Brown-Conrady, and the CMO physical models.

A second convention — **physical / mechanical Y-up** — is available for
interpreting CMO geometry in a mechanically natural frame (X right,
Y up, Z forward).  The conversion between the two is the signed
scaling matrix ``S = diag(1, -1, 1)`` applied to Y.  It is NOT a
rotation (det S = -1) and must never be injected as if it were an
SE(3) rigid transform.

**Rule:** every ``.ray(u, v)`` method returns rays in the internal
OpenCV frame.  Comparison functions (e.g. rayfield residuals) should
verify that both operands declare the same ``frame_convention``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

FrameConvention = Literal["opencv_y_down", "physical_y_up"]

# Y-flip matrices: X_right, Y_up = diag(1, -1, 1) @ (X_right, Y_down, Z).
S_CV_TO_PHYS = np.diag([1.0, -1.0, 1.0])
S_PHYS_TO_CV = S_CV_TO_PHYS  # self-inverse


# ── pixel → normalized coordinates ──────────────────────────


def pixel_to_normalized_opencv(
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates to normalised image-plane coordinates (OpenCV frame).

    Parameters
    ----------
    u, v : ndarray
        Pixel coordinates.
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix (OpenCV convention).

    Returns
    -------
    x, y : ndarray
        Normalised coordinates: ``x = (u - cx) / fx``,
        ``y = (v - cy) / fy``.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]


def pixel_to_normalized_physical_y_up(
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel to normalised coordinates in the physical Y-up frame.

    Same as :func:`pixel_to_normalized_opencv` but negates the Y coordinate.
    """
    x, y_cv = pixel_to_normalized_opencv(u, v, K)
    return x, -y_cv


# ── point / vector / ray transforms ─────────────────────────


def transform_points_cv_to_phys(X: np.ndarray) -> np.ndarray:
    """Apply ``diag(1, -1, 1)`` to an array of 3-D points.

    Parameters
    ----------
    X : ndarray, shape (..., 3)
        Points in OpenCV Y-down frame.

    Returns
    -------
    ndarray, shape (..., 3)
        Points in physical Y-up frame.
    """
    X = np.asarray(X, dtype=np.float64)
    return X @ S_CV_TO_PHYS.T


def transform_vectors_cv_to_phys(V: np.ndarray) -> np.ndarray:
    """Apply ``diag(1, -1, 1)`` to an array of 3-D vectors.

    Parameters
    ----------
    V : ndarray, shape (..., 3)
        Vectors in OpenCV Y-down frame.

    Returns
    -------
    ndarray, shape (..., 3)
        Vectors in physical Y-up frame.
    """
    V = np.asarray(V, dtype=np.float64)
    return V @ S_CV_TO_PHYS.T


def transform_rays_cv_to_phys(
    O: np.ndarray,  # noqa: E741  — canonical optical-origin symbol
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rays from OpenCV frame to physical Y-up frame.

    Parameters
    ----------
    O : ndarray, shape (..., 3)
        Ray origins in OpenCV Y-down frame.
    d : ndarray, shape (..., 3)
        Unit ray directions in OpenCV Y-down frame.

    Returns
    -------
    O_phys : ndarray, shape (..., 3)
    d_phys : ndarray, shape (..., 3)
    """
    return transform_points_cv_to_phys(O), transform_vectors_cv_to_phys(d)


def transform_points_phys_to_cv(X: np.ndarray) -> np.ndarray:
    """Apply ``diag(1, -1, 1)`` to an array of 3-D points (inverse)."""
    return transform_points_cv_to_phys(X)  # self-inverse


def transform_vectors_phys_to_cv(V: np.ndarray) -> np.ndarray:
    """Apply ``diag(1, -1, 1)`` to an array of 3-D vectors (inverse)."""
    return transform_vectors_cv_to_phys(V)  # self-inverse


def transform_rays_phys_to_cv(
    O: np.ndarray,  # noqa: E741  — canonical optical-origin symbol
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rays from physical Y-up frame to OpenCV frame (inverse)."""
    return transform_rays_cv_to_phys(O, d)  # self-inverse


# ── convention assertion ────────────────────────────────────


# ── image ↔ physical coordinate conversions ──────────────────


def phys_to_image_xy(
    X_phys: np.ndarray,
    Y_phys: np.ndarray,
    Z: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D physical points to image pixel coordinates.

    Uses the standard pinhole projection with the given intrinsic matrix
    *K*.  This is the explicit conversion boundary between the physical
    world frame (X right, Y up, Z forward) and the image frame (u right,
    v down).

    Parameters
    ----------
    X_phys, Y_phys, Z : ndarray
        Point coordinates in the physical Y-up frame (mm).
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    u, v : ndarray
        Pixel coordinates (image frame, origin at top-left).
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    X_cv = np.asarray(X_phys, dtype=np.float64)
    Y_cv = -np.asarray(Y_phys, dtype=np.float64)  # Y-up → Y-down
    Z_arr = np.asarray(Z, dtype=np.float64)
    u = K[0, 0] * X_cv / Z_arr + K[0, 2]
    v = K[1, 1] * Y_cv / Z_arr + K[1, 2]
    return u, v


def image_to_phys_xy(
    u: np.ndarray,
    v: np.ndarray,
    Z: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image pixel coordinates to physical 3-D points.

    Parameters
    ----------
    u, v : ndarray
        Pixel coordinates.
    Z : ndarray
        Depth at each point (mm).
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    X_phys, Y_phys : ndarray
        Point coordinates in the physical Y-up frame (mm).
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)
    X_cv = (u_arr - K[0, 2]) * Z_arr / K[0, 0]
    Y_cv = (v_arr - K[1, 2]) * Z_arr / K[1, 1]
    return X_cv, -Y_cv  # Y-down → Y-up


# ── convention assertion ────────────────────────────────────


def check_frame_convention(
    *objects,
    expected: FrameConvention = "opencv_y_down",
    label: str = "",
) -> None:
    """Raise ``ValueError`` if any object declares a different frame convention.

    Each *object* must have a ``frame_convention`` attribute (a
    :data:`FrameConvention` string).
    """
    for i, obj in enumerate(objects):
        actual = getattr(obj, "frame_convention", None)
        if actual is not None and actual != expected:
            prefix = f"{label}: " if label else ""
            raise ValueError(
                f"{prefix}object {i} ({type(obj).__name__}) declares "
                f"frame_convention={actual!r}, expected {expected!r}"
            )
