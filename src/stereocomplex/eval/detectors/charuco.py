"""OpenCV ChArUco/Aruco feature detection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageFeatures:
    charuco_corners: object | None
    charuco_ids: object | None
    marker_corners: object | None
    marker_ids: object | None


def method_requires_markers(method: str) -> bool:
    """Check whether a detection method needs ArUco marker data."""
    return method in {
        "homography",
        "pnp",
        "mls",
        "mls_affine",
        "mls_h",
        "pw_affine",
        "tps",
        "kfield",
        "rayfield",
        "rayfield_tps",
        "rayfield_tps_robust",
    }


def detect_image_features(
    aruco,
    dictionary,
    charuco_board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img: np.ndarray,
    method: str,
) -> tuple[ImageFeatures | None, int]:
    """Detect ArUco markers and ChArUco corners in an image."""
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        if method_requires_markers(method):
            if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
                return None, 0
        elif charuco_ids is None or charuco_corners is None:
            return None, 0
        return ImageFeatures(charuco_corners, charuco_ids, marker_corners, marker_ids), 0

    if aruco_detector is not None:
        corners, ids, _rejected = aruco_detector.detectMarkers(img)
    else:  # pragma: no cover
        corners, ids, _rejected = aruco.detectMarkers(img, dictionary, parameters=detector_params)

    if ids is None or len(ids) == 0:
        return None, 0

    charuco_corners, charuco_ids = None, None
    if method in ("charuco", "hybrid"):
        if not hasattr(aruco, "interpolateCornersCharuco"):
            raise RuntimeError("OpenCV build has no CharucoDetector or interpolateCornersCharuco.")

        ret = aruco.interpolateCornersCharuco(corners, ids, img, charuco_board)
        if ret is None:
            return None, len(ids)
        charuco_corners, charuco_ids, _ = ret
        if charuco_ids is None or charuco_corners is None:
            return None, len(ids)

    return ImageFeatures(charuco_corners, charuco_ids, corners, ids), 0
