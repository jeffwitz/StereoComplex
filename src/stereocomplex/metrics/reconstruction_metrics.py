from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginField
from stereocomplex.synthetic.parallel_plate import (
    SyntheticStereoDataset,
    parallel_plate_ray_from_pixel,
    pinhole_ray_from_pixel,
    transform_points,
)


@dataclass(frozen=True)
class ReconstructionResult:
    """Triangulated 3-D points and ray-gap validity diagnostics."""

    points_3d: np.ndarray
    ray_gap: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ReconstructionErrorReport:
    """Aggregate 3-D reconstruction errors in millimetres."""

    rms_3d: float
    median_3d: float
    p95_3d: float
    max_3d: float
    rms_x: float
    rms_y: float
    rms_z: float
    ray_gap_rms: float
    ray_gap_median: float
    ray_gap_p95: float
    n_points: int


@dataclass(frozen=True)
class ReconstructionComparisonReport:
    """Central-vs-origin-field reconstruction comparison summary."""

    central: ReconstructionErrorReport
    with_origin_field: ReconstructionErrorReport
    improvement_rms_factor: float
    improvement_median_factor: float
    improvement_p95_factor: float


@dataclass(frozen=True)
class OracleReconstructionFloorReport:
    """Noise-free and observed-pixel oracle reconstruction floors."""

    oracle_clean_pixels: ReconstructionErrorReport
    oracle_observed_pixels: ReconstructionErrorReport


def triangulate_two_rays(
    O_left: np.ndarray,
    d_left: np.ndarray,
    O_right: np.ndarray,
    d_right: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Midpoint triangulation of two 3-D rays, returning the 3-D point of closest
    approach and the ray gap (minimum distance between the two skew lines).

    The triangulated point is the midpoint of the shortest segment connecting
    the two rays.  No gauge constraint is applied to the ray origins — they are
    used as given.  If the rays are parallel or nearly parallel (denominator
    < 1e-12), the midpoint of the origins is returned and the ray gap is
    computed as the orthogonal distance from one origin to the other ray.

    Parameters
    ----------
    O_left : ndarray, shape (3,)
        Left ray origin in world coordinates, in millimetres.
    d_left : ndarray, shape (3,)
        Left ray direction (unit vector, or will be normalised).
    O_right : ndarray, shape (3,)
        Right ray origin in world coordinates, in millimetres.
    d_right : ndarray, shape (3,)
        Right ray direction (unit vector, or will be normalised).

    Returns
    -------
    P : ndarray, shape (3,)
        Midpoint of the closest-approach segment, in millimetres.
    gap : float
        Minimum distance between the two rays, in millimetres.  Zero when the
        rays intersect exactly.
    """
    O1 = np.asarray(O_left, dtype=np.float64).reshape(3)
    O2 = np.asarray(O_right, dtype=np.float64).reshape(3)
    d1 = np.asarray(d_left, dtype=np.float64).reshape(3)
    d2 = np.asarray(d_right, dtype=np.float64).reshape(3)
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    w0 = O1 - O2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    dd = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return np.full(3, np.nan), float("nan")
    lam = (b * e - c * dd) / denom
    mu = (a * e - b * dd) / denom
    P1 = O1 + lam * d1
    P2 = O2 + mu * d2
    return 0.5 * (P1 + P2), float(np.linalg.norm(P1 - P2))


def _triangulate_many(
    OL: np.ndarray, dL: np.ndarray, OR: np.ndarray, dR: np.ndarray,
) -> ReconstructionResult:
    # Normalize (mirrors triangulate_two_rays per-row behaviour)
    d1 = dL / np.linalg.norm(dL, axis=1, keepdims=True)
    d2 = dR / np.linalg.norm(dR, axis=1, keepdims=True)
    w0 = OL - OR                                      # (N, 3)
    b  = np.einsum("ij,ij->i", d1, d2)                # d1·d2  (N,)
    dd = np.einsum("ij,ij->i", d1, w0)                # d1·w0  (N,)
    e  = np.einsum("ij,ij->i", d2, w0)                # d2·w0  (N,)
    # a = c = 1 after normalization, so denom = a*c - b*b = 1 - b*b
    denom = 1.0 - b * b
    bad = np.abs(denom) < 1e-12
    safe_denom = np.where(bad, 1.0, denom)            # avoid division by zero; NaN injected below
    lam = np.where(bad, np.nan, (b * e - dd) / safe_denom)
    mu  = np.where(bad, np.nan, (e - b * dd) / safe_denom)
    P1 = OL + lam[:, None] * d1
    P2 = OR + mu[:, None] * d2
    pts  = 0.5 * (P1 + P2)
    gaps = np.linalg.norm(P1 - P2, axis=1)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(gaps)
    return ReconstructionResult(points_3d=pts, ray_gap=gaps, valid_mask=valid)


def reconstruct_points_central_stereo(
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_right_left: np.ndarray,
) -> ReconstructionResult:
    """Reconstruct 3-D points from stereo pixel correspondences using central pinhole rays.

    Back-projects each pixel pair through the camera matrices, transforms the
    right ray into the left camera frame via T_right_left, then triangulates
    by midpoint of closest approach.

    Parameters
    ----------
    left_pixels : ndarray, shape (N, 2)
        Left pixel coordinates (u, v) in pixels.
    right_pixels : ndarray, shape (N, 2)
        Right pixel coordinates (u, v) in pixels.
    K_left, K_right : ndarray, shape (3, 3)
        Camera matrices for each channel.
    T_right_left : ndarray, shape (4, 4)
        Rigid transform from left to right camera frame.

    Returns
    -------
    ReconstructionResult
        Dataclass result object with ``points_3d`` (N, 3), ``ray_gap`` (N,),
        and ``valid_mask`` (N, bool).
    """
    uvL = np.asarray(left_pixels, dtype=np.float64).reshape(-1, 2)
    uvR = np.asarray(right_pixels, dtype=np.float64).reshape(-1, 2)
    if uvL.shape != uvR.shape:
        raise ValueError("left_pixels and right_pixels must have the same shape")
    T = np.asarray(T_right_left, dtype=np.float64).reshape(4, 4)
    R_RL = T[:3, :3]
    t_RL = T[:3, 3]
    dL = pinhole_ray_from_pixel(uvL[:, 0], uvL[:, 1], K_left)
    dR_R = pinhole_ray_from_pixel(uvR[:, 0], uvR[:, 1], K_right)
    dR_L = (R_RL.T @ dR_R.T).T
    OL = np.zeros_like(dL)
    OR_L = np.repeat((-R_RL.T @ t_RL).reshape(1, 3), uvL.shape[0], axis=0)
    return _triangulate_many(OL, dL, OR_L, dR_L)


def reconstruct_points_with_origin_fields(
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    left_field: ZernikeOriginField,
    right_field: ZernikeOriginField,
    T_right_left: np.ndarray,
) -> ReconstructionResult:
    """Reconstruct 3-D points using per-pixel origin fields (non-central rays).

    Unlike the central model where all rays share one camera centre, each
    pixel gets its own ray origin O(u,v) and direction d(u,v) from the
    Zernike origin field.  Triangulation uses midpoint of closest approach
    between the two skew rays.

    Parameters
    ----------
    left_pixels : ndarray, shape (N, 2)
        Left pixel coordinates (u, v) in pixels.
    right_pixels : ndarray, shape (N, 2)
        Right pixel coordinates (u, v) in pixels.
    left_field : ZernikeOriginField
        Fitted left-channel non-central rayfield.
    right_field : ZernikeOriginField
        Fitted right-channel non-central rayfield.
    T_right_left : ndarray, shape (4, 4)
        Rigid transform from left to right camera frame.

    Returns
    -------
    ReconstructionResult
        Dataclass result object with ``points_3d`` (N, 3), ``ray_gap`` (N,),
        and ``valid_mask`` (N, bool).
    """
    uvL = np.asarray(left_pixels, dtype=np.float64).reshape(-1, 2)
    uvR = np.asarray(right_pixels, dtype=np.float64).reshape(-1, 2)
    if uvL.shape != uvR.shape:
        raise ValueError("left_pixels and right_pixels must have the same shape")
    T = np.asarray(T_right_left, dtype=np.float64).reshape(4, 4)
    R_RL = T[:3, :3]
    t_RL = T[:3, 3]
    OL, dL = left_field.ray(uvL[:, 0], uvL[:, 1])
    OR_R, dR_R = right_field.ray(uvR[:, 0], uvR[:, 1])
    OR_L = (R_RL.T @ (OR_R - t_RL.reshape(1, 3)).T).T
    dR_L = (R_RL.T @ dR_R.T).T
    return _triangulate_many(OL, dL, OR_L, dR_L)


def reconstruct_points_with_parallel_plate_oracle(
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    dataset: SyntheticStereoDataset,
) -> ReconstructionResult:
    """
    Reconstruct correspondences with the physical oracle rayfields.

    The oracle keeps the physical exit point I2. This function is for evaluation
    only and must not be used by the generic Zernike fit.
    """
    if dataset.oracle_left_params is None or dataset.oracle_right_params is None:
        raise ValueError("dataset does not contain oracle parallel-plate parameters")
    uvL = np.asarray(left_pixels, dtype=np.float64).reshape(-1, 2)
    uvR = np.asarray(right_pixels, dtype=np.float64).reshape(-1, 2)
    if uvL.shape != uvR.shape:
        raise ValueError("left_pixels and right_pixels must have the same shape")
    T = np.asarray(dataset.T_right_left, dtype=np.float64).reshape(4, 4)
    R_RL = T[:3, :3]
    t_RL = T[:3, 3]
    OL, dL = parallel_plate_ray_from_pixel(
        uvL[:, 0], uvL[:, 1], dataset.K_left, dataset.oracle_left_params
    )
    OR_R, dR_R = parallel_plate_ray_from_pixel(
        uvR[:, 0], uvR[:, 1], dataset.K_right, dataset.oracle_right_params
    )
    OR_L = (R_RL.T @ (OR_R - t_RL.reshape(1, 3)).T).T
    dR_L = (R_RL.T @ dR_R.T).T
    return _triangulate_many(OL, dL, OR_L, dR_L)


def reconstruction_error_report(
    result: ReconstructionResult, true_points: np.ndarray,
) -> ReconstructionErrorReport:
    """Compute 3D reconstruction error statistics against ground truth.

    Parameters
    ----------
    result : ReconstructionResult
        Reconstructed points, ray gaps, and validity mask.
    true_points : np.ndarray
        Ground truth 3D points, shape ``(N, 3)``.

    Returns
    -------
    ReconstructionErrorReport
        RMS, median, P95, and max 3D errors plus per-axis RMS and ray-gap stats.
    """
    true = np.asarray(true_points, dtype=np.float64).reshape(-1, 3)
    pred = np.asarray(result.points_3d, dtype=np.float64).reshape(-1, 3)
    if true.shape != pred.shape:
        raise ValueError("true_points and result.points_3d must have the same shape")
    mask = np.asarray(result.valid_mask, dtype=bool).reshape(-1)
    if not np.any(mask):
        raise ValueError("no valid reconstructed points")
    err = pred[mask] - true[mask]
    norm = np.linalg.norm(err, axis=1)
    gaps = np.asarray(result.ray_gap, dtype=np.float64).reshape(-1)[mask]
    return ReconstructionErrorReport(
        rms_3d=float(np.sqrt(np.mean(norm**2))),
        median_3d=float(np.median(norm)),
        p95_3d=float(np.percentile(norm, 95)),
        max_3d=float(np.max(norm)),
        rms_x=float(np.sqrt(np.mean(err[:, 0] ** 2))),
        rms_y=float(np.sqrt(np.mean(err[:, 1] ** 2))),
        rms_z=float(np.sqrt(np.mean(err[:, 2] ** 2))),
        ray_gap_rms=float(np.sqrt(np.mean(gaps**2))),
        ray_gap_median=float(np.median(gaps)),
        ray_gap_p95=float(np.percentile(gaps, 95)),
        n_points=int(mask.sum()),
    )


def _dataset_left_camera_points(dataset: SyntheticStereoDataset) -> np.ndarray:
    pts: list[np.ndarray] = []
    for i, T_board_world in enumerate(dataset.board_poses):
        obj = (
            dataset.per_frame_object_points[i]
            if dataset.per_frame_object_points is not None
            else dataset.object_points
        )
        P_world = transform_points(T_board_world, obj)
        pts.append(transform_points(dataset.T_left_world, P_world))
    return np.concatenate(pts, axis=0)


def oracle_reconstruction_floor_report(
    dataset_observed: SyntheticStereoDataset,
    dataset_clean: SyntheticStereoDataset | None = None,
) -> OracleReconstructionFloorReport:
    """
    Compute the oracle floor for clean and observed/noisy pixels.

    `dataset_clean` should share the same geometry and oracle parameters as
    `dataset_observed`, but with noise-free pixels. If omitted, the clean case is
    evaluated with `dataset_observed` pixels.
    """
    if dataset_clean is None:
        dataset_clean = dataset_observed
    truth = _dataset_left_camera_points(dataset_observed)
    clean = reconstruct_points_with_parallel_plate_oracle(
        np.concatenate(dataset_clean.left_pixels, axis=0),
        np.concatenate(dataset_clean.right_pixels, axis=0),
        dataset_observed,
    )
    observed = reconstruct_points_with_parallel_plate_oracle(
        np.concatenate(dataset_observed.left_pixels, axis=0),
        np.concatenate(dataset_observed.right_pixels, axis=0),
        dataset_observed,
    )
    return OracleReconstructionFloorReport(
        oracle_clean_pixels=reconstruction_error_report(clean, truth),
        oracle_observed_pixels=reconstruction_error_report(observed, truth),
    )


def compare_3d_reconstruction_with_without_origin_field(
    dataset: SyntheticStereoDataset,
    central_model_result,
    origin_field_result,
) -> ReconstructionComparisonReport:
    """Compare central stereo against non-central reconstruction with origin fields.

    Triangulates the same point correspondences using a central model (all rays
    from a single camera centre) and a non-central model (per-pixel origins).
    Reports RMS, median, and P95 reconstruction error in 3-D.

    Parameters
    ----------
    dataset : SyntheticStereoDataset
        Stereo observations with ground-truth 3-D points.
    central_model_result : StereoCentralRayFieldFitResult
        Result from central rayfield calibration.
    origin_field_result : StereoZernikeOriginFieldFitResult
        Result from non-central Zernike origin field calibration.

    Returns
    -------
    ReconstructionComparisonReport
        Dataclass result object with ``central`` and ``with_origin_field``
        error statistics.
    """
    del central_model_result
    uvL = np.concatenate(dataset.left_pixels, axis=0)
    uvR = np.concatenate(dataset.right_pixels, axis=0)
    truth = _dataset_left_camera_points(dataset)
    central = reconstruct_points_central_stereo(
        uvL, uvR, dataset.K_left, dataset.K_right, dataset.T_right_left
    )
    with_origin = reconstruct_points_with_origin_fields(
        uvL,
        uvR,
        origin_field_result.left_field,
        origin_field_result.right_field,
        origin_field_result.stereo_transform,
    )
    central_report = reconstruction_error_report(central, truth)
    origin_report = reconstruction_error_report(with_origin, truth)
    return ReconstructionComparisonReport(
        central=central_report,
        with_origin_field=origin_report,
        improvement_rms_factor=float(central_report.rms_3d / origin_report.rms_3d),
        improvement_median_factor=float(central_report.median_3d / origin_report.median_3d),
        improvement_p95_factor=float(central_report.p95_3d / origin_report.p95_3d),
    )
