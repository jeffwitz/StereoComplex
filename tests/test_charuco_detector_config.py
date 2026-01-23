from __future__ import annotations

import numpy as np
import pytest


def _make_even_parity_charuco_like_board(*, cv2, aruco, squares_x: int, squares_y: int, square_px: int, marker_px: int) -> np.ndarray:
    """
    Generate a Pycaso-like ChArUco board raster:
    - markers are placed on even-parity squares ((r+c)%2==0),
    - other squares are solid black.

    This deliberately differs from OpenCV's CharucoBoard default (odd parity),
    so CharucoDetector must be configured with checkMarkers=False to interpolate
    ChArUco corners.
    """
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    img = np.zeros((squares_y * square_px, squares_x * square_px), dtype=np.uint8)

    marker_id = 0
    for r in range(squares_y):
        for c in range(squares_x):
            if ((r + c) & 1) == 0:
                marker = aruco.generateImageMarker(dictionary, marker_id, marker_px)
                marker_id += 1
                sq = np.full((square_px, square_px), 255, dtype=np.uint8)
                off = (square_px - marker_px) // 2
                sq[off : off + marker_px, off : off + marker_px] = marker
            else:
                sq = np.zeros((square_px, square_px), dtype=np.uint8)
            img[r * square_px : (r + 1) * square_px, c * square_px : (c + 1) * square_px] = sq

    return img


def test_charuco_detector_checkmarkers_config_enables_even_parity_board() -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco not available (need opencv-contrib-python)")
    import cv2.aruco as aruco  # type: ignore

    from stereocomplex.cli.refine_corners import build_charuco_from_meta, detect_view

    img = _make_even_parity_charuco_like_board(cv2=cv2, aruco=aruco, squares_x=6, squares_y=4, square_px=80, marker_px=40)

    base_meta = {
        "board": {
            "type": "charuco",
            "aruco_dictionary": "DICT_6X6_250",
            "squares_x": 6,
            "squares_y": 4,
            "square_size_mm": 1.0,
            "marker_size_mm": 0.5,
        }
    }

    # Default OpenCV CharucoDetector has checkMarkers=True and fails to interpolate corners.
    cv2_, aruco_, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(base_meta)
    det0 = detect_view(cv2_, aruco_, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
    assert det0 is not None
    assert det0.charuco_ids.size == 0

    # With checkMarkers=False, the same CharucoBoard can interpolate corners on an even-parity layout.
    cfg_meta = {
        **base_meta,
        "opencv": {
            "charuco_detector": {"checkMarkers": False},
        },
    }
    cv2_, aruco_, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(cfg_meta)
    det1 = detect_view(cv2_, aruco_, dictionary, board, detector_params, aruco_detector, charuco_detector, img)
    assert det1 is not None
    assert det1.charuco_ids.size > 0

