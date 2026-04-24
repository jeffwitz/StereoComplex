from __future__ import annotations

import json
from pathlib import Path


def test_public_api_exports() -> None:
    import stereocomplex as sc

    assert hasattr(sc, "CharucoBoardSpec")
    assert hasattr(sc, "StereoImagePair")
    assert hasattr(sc, "StereoOpenCVCalibrationReport")
    assert hasattr(sc, "StereoOpenCVCalibrationResult")
    assert hasattr(sc, "build_charuco_board")
    assert hasattr(sc, "detect_charuco_corners")
    assert hasattr(sc, "fit_opencv_stereo_from_dataset")
    assert hasattr(sc, "fit_opencv_stereo_from_image_dirs")
    assert hasattr(sc, "fit_opencv_stereo_from_image_pairs")
    assert hasattr(sc, "fit_stereo_central_rayfield_from_dataset")
    assert hasattr(sc, "fit_stereo_central_rayfield_from_image_dirs")
    assert hasattr(sc, "fit_stereo_central_rayfield_from_image_pairs")
    assert hasattr(sc, "load_stereo_central_rayfield")
    assert hasattr(sc, "save_stereo_central_rayfield")
    assert hasattr(sc, "StereoCentralRayFieldModel")
    assert hasattr(sc, "refine_charuco_corners")


def test_refine_charuco_corners_accepts_detection_dataclass() -> None:
    import stereocomplex as sc

    scene_dir = Path("dataset/v0_png/train/scene_0000")
    board = sc.CharucoBoardSpec.from_meta(json.loads((scene_dir / "meta.json").read_text(encoding="utf-8")))
    detections = sc.detect_charuco_corners(image=scene_dir / "left" / "000000.png", board=board)
    assert detections is not None
    refined = sc.refine_charuco_corners(
        method="rayfield_tps_robust",
        board=board,
        detections=detections,
    )
    assert refined.shape == detections.charuco_xy.shape
