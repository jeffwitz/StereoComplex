"""Corner preprocessing used by the Pycaso CMO calibration pipeline."""

from __future__ import annotations

import math

import cv2
import numpy as np

from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust


def abs_det_hessian(gray: np.ndarray, sigma: float = 9.0) -> np.ndarray:
    image = gray.astype(np.float32)
    if image.max() > 2:
        image /= 255.0
    image = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    hxx = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=3)
    hyy = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=3)
    hxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    return np.abs(hxx * hyy - hxy**2)


def otsu_mask(response: np.ndarray) -> np.ndarray:
    response_u8 = cv2.normalize(
        response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, mask = cv2.threshold(
        response_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return mask


def blob_barycentre(
    mask: np.ndarray,
    xi: float,
    yi: float,
    radius: int,
    *,
    prefer_largest: bool = True,
) -> tuple[float, float, float]:
    height, width = mask.shape
    x0, x1 = max(0, int(xi) - radius), min(width, int(xi) + radius)
    y0, y1 = max(0, int(yi) - radius), min(height, int(yi) + radius)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return math.nan, math.nan, math.nan
    roi = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi, connectivity=8
    )
    if n_labels <= 1:
        return math.nan, math.nan, math.nan
    if prefer_largest:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        cx, cy = xi - x0, yi - y0
        selected = min(
            range(1, n_labels),
            key=lambda k: (
                stats[k, cv2.CC_STAT_LEFT]
                + stats[k, cv2.CC_STAT_WIDTH] / 2
                - cx
            ) ** 2
            + (
                stats[k, cv2.CC_STAT_TOP]
                + stats[k, cv2.CC_STAT_HEIGHT] / 2
                - cy
            ) ** 2,
        )
    moments = cv2.moments((labels == selected).astype(np.uint8), binaryImage=True)
    if moments["m00"] < 1e-10:
        return math.nan, math.nan, math.nan
    return (
        moments["m10"] / moments["m00"] + x0,
        moments["m01"] / moments["m00"] + y0,
        float(moments["m00"]),
    )


def win_spot_2pass(
    mask: np.ndarray,
    grid_step: float,
    initial_radius: int,
    xi: float,
    yi: float,
    *,
    prefer_largest: bool,
) -> tuple[float, float, float]:
    for radius in (initial_radius, max(initial_radius, int(grid_step * 0.5))):
        xd, yd, area = blob_barycentre(
            mask, xi, yi, radius, prefer_largest=prefer_largest
        )
        if not math.isnan(xd):
            return xd, yd, area
    return math.nan, math.nan, math.nan


def ids_to_grid(ids: np.ndarray, ncx: int = 16) -> np.ndarray:
    ids_arr = np.asarray(ids, dtype=np.float32).reshape(-1)
    nx = ncx - 1
    return np.column_stack([ids_arr % nx, ids_arr // nx]).astype(np.float32)


def fit_affine(img_pts: np.ndarray, ids: np.ndarray, ncx: int = 16) -> np.ndarray:
    image = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    grid = ids_to_grid(ids, ncx)
    affine, _ = cv2.estimateAffine2D(grid, image, method=cv2.LMEDS)
    if affine is None:
        design = np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])
        affine = np.linalg.lstsq(design, image, rcond=None)[0].T.astype(np.float32)
    return affine


def project_affine(affine: np.ndarray, ids: np.ndarray, ncx: int = 16) -> np.ndarray:
    grid = ids_to_grid(ids, ncx)
    design = np.column_stack([grid, np.ones(len(grid), dtype=np.float32)])
    return design @ affine.T


def marker_tps_corners(
    marker_corners: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    marker_ids: np.ndarray | None,
    marker_object_xy: dict[int, np.ndarray],
    chessboard_xy: np.ndarray,
) -> np.ndarray | None:
    """Predict every chessboard corner from detected ArUco marker corners."""
    if marker_corners is None or marker_ids is None:
        return None
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for marker_id, corners in zip(
        np.asarray(marker_ids).reshape(-1), marker_corners, strict=True
    ):
        obj = marker_object_xy.get(int(marker_id))
        image = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if obj is not None and image.shape == (4, 2):
            object_points.append(obj)
            image_points.append(image)
    if len(object_points) < 4:
        return None
    return predict_points_rayfield_tps_robust(
        np.concatenate(object_points, axis=0),
        np.concatenate(image_points, axis=0),
        np.asarray(chessboard_xy, dtype=np.float64),
        lam=10.0,
        huber_c=3.0,
        iters=3,
        ransac_reproj_px=3.0,
    )


def second_tps_pass(chessboard_xy: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply the production smoothing pass to a complete corner grid."""
    board_xy = np.asarray(chessboard_xy, dtype=np.float64)
    return predict_points_rayfield_tps_robust(
        board_xy,
        np.asarray(corners, dtype=np.float64),
        board_xy,
        lam=3.0,
        huber_c=1.5,
        iters=2,
        ransac_reproj_px=2.0,
    )


def complete_corners_hessian(
    gray: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
    ncx: int,
    ncy: int,
    *,
    marker_object_xy: dict[int, np.ndarray],
    chessboard_xy: np.ndarray,
    marker_corners: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    marker_ids: np.ndarray | None = None,
    hessian_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Complete a ChArUco grid by Hessian blobs near a marker-TPS prediction."""
    nx, ny = ncx - 1, ncy - 1
    n_corners = nx * ny
    mask = (
        otsu_mask(abs_det_hessian(gray))
        if hessian_mask is None
        else np.asarray(hessian_mask, dtype=np.uint8)
    )
    detected: dict[int, np.ndarray] = {}
    if charuco_ids is not None and len(charuco_ids) > 0:
        ids_flat = np.asarray(charuco_ids).reshape(-1)
        corners = np.asarray(charuco_corners).reshape(-1, 2)
        detected = {
            int(corner_id): corners[i].astype(np.float64)
            for i, corner_id in enumerate(ids_flat)
        }

    detected_ids = sorted(detected)
    prediction = marker_tps_corners(
        marker_corners, marker_ids, marker_object_xy, chessboard_xy
    )
    if prediction is None and len(detected_ids) >= 3:
        affine = fit_affine(
            np.array([detected[i] for i in detected_ids]),
            np.array(detected_ids),
            ncx,
        )
        prediction = project_affine(affine, np.arange(n_corners), ncx)
    if prediction is None:
        raise ValueError("at least three ChArUco corners or four markers are required")

    grid_step = 50.0
    if len(detected_ids) >= 2:
        image_span = float(
            np.linalg.norm(detected[detected_ids[-1]] - detected[detected_ids[0]])
        )
        grid_ends = ids_to_grid(
            np.array([detected_ids[0], detected_ids[-1]]), ncx
        )
        grid_span = float(np.linalg.norm(grid_ends[1] - grid_ends[0]))
        if grid_span > 1e-8:
            grid_step = image_span / grid_span
    initial_radius = max(3, int(grid_step * 0.3))
    if detected_ids:
        x0, y0 = detected[detected_ids[0]]
        _, _, area = win_spot_2pass(
            mask,
            grid_step,
            int(grid_step * 2 / 3),
            float(x0),
            float(y0),
            prefer_largest=True,
        )
        if not math.isnan(area) and area > 0:
            initial_radius = max(3, int(math.sqrt(area)))

    result = np.full((n_corners, 2), np.nan)
    for corner_id in range(n_corners):
        if corner_id in detected:
            result[corner_id] = detected[corner_id]
            continue
        xi, yi = prediction[corner_id]
        xd, yd, _ = win_spot_2pass(
            mask,
            grid_step,
            initial_radius,
            float(xi),
            float(yi),
            prefer_largest=False,
        )
        if not math.isnan(xd):
            result[corner_id] = [xd, yd]
    missing = np.isnan(result[:, 0])
    result[missing] = prediction[missing]
    return result
