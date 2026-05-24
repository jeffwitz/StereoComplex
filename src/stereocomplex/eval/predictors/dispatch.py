"""Dispatch ChArUco corner prediction methods."""

from __future__ import annotations

import numpy as np

from stereocomplex.eval.detectors.charuco import ImageFeatures
from stereocomplex.eval.predictors.warps import (
    build_marker_correspondences,
    predict_hybrid,
    predict_points_affine_field,
    predict_points_homography,
    predict_points_mls_affine,
    predict_points_mls_homography,
    predict_points_piecewise_affine,
    predict_points_rayfield,
    predict_points_rayfield_tps,
    predict_points_rayfield_tps_robust,
    predict_points_tps,
)


def marker_correspondences(charuco_board, marker_ids, marker_corners, *, ndim: int = 2):
    """Pair detected ArUco marker corners with board object points."""
    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        return None
    return build_marker_correspondences(charuco_board, marker_ids, marker_corners, ndim=ndim)


def predict_marker_warp(
    method: str, obj_pts: np.ndarray, img_pts: np.ndarray, chess: np.ndarray
) -> np.ndarray:
    """Predict board-plane coordinates from a marker warp model."""
    predictors = {
        "kfield": predict_points_affine_field,
        "rayfield": predict_points_rayfield,
        "rayfield_tps": predict_points_rayfield_tps,
        "rayfield_tps_robust": predict_points_rayfield_tps_robust,
        "mls_affine": predict_points_mls_affine,
        "mls_h": predict_points_mls_homography,
        "pw_affine": predict_points_piecewise_affine,
        "tps": predict_points_tps,
    }
    return predictors[method](obj_pts, img_pts, chess)


def predict_charuco_points(
    cv2,
    charuco_board,
    features: ImageFeatures,
    method: str,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Predict ChArUco corner positions from marker correspondences."""
    if method == "mls":
        method = "mls_affine"

    if method == "hybrid":
        return predict_hybrid(
            charuco_board,
            features.charuco_ids,
            features.charuco_corners,
            features.marker_ids,
            features.marker_corners,
        )

    if method in (
        "kfield",
        "rayfield",
        "rayfield_tps",
        "rayfield_tps_robust",
        "mls_affine",
        "mls_h",
        "pw_affine",
        "tps",
    ):
        corr = marker_correspondences(
            charuco_board, features.marker_ids, features.marker_corners, ndim=2
        )
        if corr is None:
            return None
        obj_pts, img_pts = corr
        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        charuco_xy = predict_marker_warp(method, obj_pts, img_pts, chess)
        return charuco_ids, charuco_xy

    if method == "pnp":
        corr = marker_correspondences(
            charuco_board, features.marker_ids, features.marker_corners, ndim=3
        )
        if corr is None:
            return None
        obj_pts, img_pts = corr

        if camera_matrix is None or dist_coeffs is None:
            raise RuntimeError("pnp method requires camera_matrix and dist_coeffs from meta.json")

        ok, rvec, tvec, _inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            iterationsCount=200,
            confidence=0.999,
        )
        if not ok:
            return None

        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)
        charuco_ids = np.arange(chess.shape[0], dtype=np.int32)
        proj, _jac = cv2.projectPoints(chess, rvec, tvec, camera_matrix, dist_coeffs)
        return charuco_ids, proj.reshape(-1, 2).astype(np.float64)

    if method == "homography":
        corr = marker_correspondences(
            charuco_board, features.marker_ids, features.marker_corners, ndim=2
        )
        if corr is None:
            return None
        obj_pts, img_pts = corr
        chess = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)[:, :2]
        charuco_xy = predict_points_homography(obj_pts, img_pts, chess)
        if charuco_xy is None:
            return None
        return np.arange(chess.shape[0], dtype=np.int32), charuco_xy

    if method == "charuco":
        if features.charuco_ids is None or features.charuco_corners is None:
            return None
        charuco_ids = features.charuco_ids.reshape(-1).astype(np.int32)
        charuco_xy = features.charuco_corners.reshape(-1, 2).astype(np.float64) - 0.5
        return charuco_ids, charuco_xy

    raise ValueError(
        "method must be charuco|homography|pnp|mls|mls_h|pw_affine|tps|hybrid"
        "|kfield|rayfield|rayfield_tps|rayfield_tps_robust"
    )
