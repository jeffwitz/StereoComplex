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
    points_3d: np.ndarray
    ray_gap: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class ReconstructionErrorReport:
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
    central: ReconstructionErrorReport
    with_origin_field: ReconstructionErrorReport
    improvement_rms_factor: float
    improvement_median_factor: float
    improvement_p95_factor: float


@dataclass(frozen=True)
class OracleReconstructionFloorReport:
    oracle_clean_pixels: ReconstructionErrorReport
    oracle_observed_pixels: ReconstructionErrorReport


def triangulate_two_rays(
    O_left: np.ndarray,
    d_left: np.ndarray,
    O_right: np.ndarray,
    d_right: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the midpoint triangulation of two 3D rays and their minimum distance."""
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


def _triangulate_many(OL: np.ndarray, dL: np.ndarray, OR: np.ndarray, dR: np.ndarray) -> ReconstructionResult:
    pts = np.empty_like(OL, dtype=np.float64)
    gaps = np.empty((OL.shape[0],), dtype=np.float64)
    valid = np.ones((OL.shape[0],), dtype=bool)
    for i in range(OL.shape[0]):
        pts[i], gaps[i] = triangulate_two_rays(OL[i], dL[i], OR[i], dR[i])
        valid[i] = bool(np.all(np.isfinite(pts[i])) and np.isfinite(gaps[i]))
    return ReconstructionResult(points_3d=pts, ray_gap=gaps, valid_mask=valid)


def reconstruct_points_central_stereo(
    left_pixels: np.ndarray,
    right_pixels: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    T_right_left: np.ndarray,
) -> ReconstructionResult:
    """Reconstruct correspondences with central pinhole rays."""
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
    """Reconstruct correspondences using identified origin fields."""
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
    OL, dL = parallel_plate_ray_from_pixel(uvL[:, 0], uvL[:, 1], dataset.K_left, dataset.oracle_left_params)
    OR_R, dR_R = parallel_plate_ray_from_pixel(uvR[:, 0], uvR[:, 1], dataset.K_right, dataset.oracle_right_params)
    OR_L = (R_RL.T @ (OR_R - t_RL.reshape(1, 3)).T).T
    dR_L = (R_RL.T @ dR_R.T).T
    return _triangulate_many(OL, dL, OR_L, dR_L)


def reconstruction_error_report(result: ReconstructionResult, true_points: np.ndarray) -> ReconstructionErrorReport:
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
    for T_board_world in dataset.board_poses:
        P_world = transform_points(T_board_world, dataset.object_points)
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
    """Compare central stereo against reconstruction with identified origin fields."""
    del central_model_result
    uvL = np.concatenate(dataset.left_pixels, axis=0)
    uvR = np.concatenate(dataset.right_pixels, axis=0)
    truth = _dataset_left_camera_points(dataset)
    central = reconstruct_points_central_stereo(uvL, uvR, dataset.K_left, dataset.K_right, dataset.T_right_left)
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
