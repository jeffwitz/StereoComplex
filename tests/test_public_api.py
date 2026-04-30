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
    assert hasattr(sc, "ParallelPlateSyntheticParams")
    assert hasattr(sc, "ZernikeOriginFieldConfig")
    assert hasattr(sc, "ZernikeRayField")
    assert hasattr(sc, "fit_stereo_zernike_origin_field")
    assert hasattr(sc, "run_parallel_plate_origin_field_benchmark")
    assert hasattr(sc, "run_parallel_plate_rendered_image_benchmark")
    assert hasattr(sc, "render_parallel_plate_charuco_images")
    assert hasattr(sc, "detected_observations_from_rendered_parallel_plate")
    assert hasattr(sc, "compare_3d_reconstruction_with_without_origin_field")
    assert hasattr(sc, "oracle_reconstruction_floor_report")


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


def test_experimental_origin_field_api_smoke() -> None:
    import stereocomplex as sc

    assert sc.ParallelPlateSyntheticParams().thickness > 0
    assert sc.ZernikeOriginFieldConfig(image_size=(640, 480)).max_order == 4
    assert callable(sc.ZernikeRayField)
    assert callable(sc.triangulate_two_rays)
    assert callable(sc.oracle_reconstruction_floor_report)
