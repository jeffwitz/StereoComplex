from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stereocomplex.api._calibration_images import _ensure_gray_u8
from stereocomplex.api._calibration_types import CharucoBoardSpec, _RefinedStereoDetections
from stereocomplex.api.corner_refinement import (
    CharucoDetections,
    RefineMethod,
    refine_charuco_corners,
)


@dataclass(frozen=True)
class _CharucoRuntime:
    cv2: Any
    aruco: Any
    board: Any
    dictionary: Any
    detector_params: Any
    aruco_detector: Any
    charuco_detector: Any


def _build_charuco_runtime(board: CharucoBoardSpec) -> _CharucoRuntime:
    import cv2  # type: ignore
    from cv2 import aruco  # type: ignore

    dict_id = getattr(aruco, str(board.aruco_dictionary), None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {board.aruco_dictionary}")
    dictionary = aruco.getPredefinedDictionary(dict_id)

    if hasattr(aruco, "CharucoBoard"):
        charuco_board = aruco.CharucoBoard(
            (int(board.squares_x), int(board.squares_y)),
            float(board.square_size_mm),
            float(board.marker_size_mm),
            dictionary,
        )
        if board.legacy_pattern and hasattr(charuco_board, "setLegacyPattern"):
            charuco_board.setLegacyPattern(True)
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        charuco_board = aruco.CharucoBoard_create(
            int(board.squares_x),
            int(board.squares_y),
            float(board.square_size_mm),
            float(board.marker_size_mm),
            dictionary,
        )
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = int(board.corner_refinement_win_size)
    detector_params.cornerRefinementMaxIterations = int(board.corner_refinement_max_iterations)
    detector_params.cornerRefinementMinAccuracy = float(board.corner_refinement_min_accuracy)
    if board.adaptive_thresh_win_size_max is not None:
        detector_params.adaptiveThreshWinSizeMax = int(board.adaptive_thresh_win_size_max)
    if hasattr(detector_params, "minMarkerPerimeterRate"):
        detector_params.minMarkerPerimeterRate = 0.005
    if hasattr(detector_params, "maxMarkerPerimeterRate"):
        detector_params.maxMarkerPerimeterRate = 0.20
    if hasattr(detector_params, "polygonalApproxAccuracyRate"):
        detector_params.polygonalApproxAccuracyRate = 0.03

    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(charuco_board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)
        if hasattr(charuco_detector, "getCharucoParameters") and hasattr(
            charuco_detector, "setCharucoParameters"
        ):
            cp = charuco_detector.getCharucoParameters()
            if board.check_markers is not None:
                cp.checkMarkers = bool(board.check_markers)
            if board.min_markers is not None:
                cp.minMarkers = int(board.min_markers)
            if board.try_refine_markers is not None:
                cp.tryRefineMarkers = bool(board.try_refine_markers)
            charuco_detector.setCharucoParameters(cp)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return _CharucoRuntime(
        cv2=cv2,
        aruco=aruco,
        board=charuco_board,
        dictionary=dictionary,
        detector_params=detector_params,
        aruco_detector=aruco_detector,
        charuco_detector=charuco_detector,
    )


def build_charuco_board(board: CharucoBoardSpec) -> Any:
    """Build an OpenCV ``aruco.CharucoBoard`` from a stable Python dataclass."""
    return _build_charuco_runtime(board).board


def detect_charuco_corners(
    *, image: str | Path | np.ndarray, board: CharucoBoardSpec
) -> CharucoDetections | None:
    """Detect ArUco markers and ChArUco corners in one image."""
    runtime = _build_charuco_runtime(board)
    img_gray = _ensure_gray_u8(image)
    return _detect_view(runtime, img_gray)


def _detect_view(runtime: _CharucoRuntime, img_gray: np.ndarray) -> CharucoDetections | None:
    aruco = runtime.aruco

    if runtime.charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            runtime.charuco_detector.detectBoard(img_gray)
        )
    else:
        if runtime.aruco_detector is not None:
            marker_corners, marker_ids, _rej = runtime.aruco_detector.detectMarkers(img_gray)
        else:  # pragma: no cover
            marker_corners, marker_ids, _rej = aruco.detectMarkers(
                img_gray, runtime.dictionary, parameters=runtime.detector_params
            )

        charuco_corners, charuco_ids = None, None
        if (
            hasattr(aruco, "interpolateCornersCharuco")
            and marker_ids is not None
            and len(marker_ids) > 0
        ):
            ret = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, img_gray, runtime.board
            )
            if ret is not None and len(ret) >= 2:
                if len(ret) == 3:
                    charuco_corners, charuco_ids, _ = ret
                elif len(ret) == 4:  # pragma: no cover
                    _, charuco_corners, charuco_ids, _ = ret

    if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
        return None

    marker_ids_arr = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
    marker_corners_arr = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in marker_corners]
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        charuco_ids_arr = np.zeros((0,), dtype=np.int32)
        charuco_xy = np.zeros((0, 2), dtype=np.float64)
    else:
        charuco_ids_arr = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5

    return CharucoDetections(
        marker_ids=marker_ids_arr,
        marker_corners=marker_corners_arr,
        charuco_ids=charuco_ids_arr,
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return {int(i): xy[k].astype(np.float64) for k, i in enumerate(ids.tolist())}


def _refine_detection_points(
    *,
    runtime: _CharucoRuntime,
    detection: CharucoDetections,
    method2d: RefineMethod,
    tps_lam: float,
    huber_c: float,
    iters: int,
) -> np.ndarray:
    if method2d == "raw":
        return np.asarray(detection.charuco_xy, dtype=np.float64).reshape(-1, 2)
    return np.asarray(
        refine_charuco_corners(
            method=str(method2d),
            board=runtime.board,
            marker_ids=detection.marker_ids,
            marker_corners=detection.marker_corners,
            charuco_ids=detection.charuco_ids,
            charuco_xy=detection.charuco_xy,
            tps_lam=float(tps_lam),
            huber_c=float(huber_c),
            iters=int(iters),
        ),
        dtype=np.float64,
    ).reshape(-1, 2)


def _detect_refined_stereo_pair(
    *,
    runtime: _CharucoRuntime,
    img_left: np.ndarray,
    img_right: np.ndarray,
    method2d: RefineMethod,
    tps_lam: float,
    huber_c: float,
    iters: int,
) -> _RefinedStereoDetections | None:
    det_left = _detect_view(runtime, img_left)
    det_right = _detect_view(runtime, img_right)
    if det_left is None or det_right is None:
        return None

    xy_left = _refine_detection_points(
        runtime=runtime,
        detection=det_left,
        method2d=method2d,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )
    xy_right = _refine_detection_points(
        runtime=runtime,
        detection=det_right,
        method2d=method2d,
        tps_lam=tps_lam,
        huber_c=huber_c,
        iters=iters,
    )
    return _RefinedStereoDetections(
        det_left=det_left,
        det_right=det_right,
        xy_left=xy_left,
        xy_right=xy_right,
        map_left=_dict_from_ids_xy(det_left.charuco_ids, xy_left),
        map_right=_dict_from_ids_xy(det_right.charuco_ids, xy_right),
    )
