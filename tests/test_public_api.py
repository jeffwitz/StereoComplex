from __future__ import annotations

import json
from pathlib import Path


def test_public_api_exports() -> None:
    """Verify that Tier 1+2 symbols are importable from stereocomplex."""
    import stereocomplex as sc

    # Tier 1 — user entry points
    assert hasattr(sc, "CharucoBoardSpec")
    assert hasattr(sc, "PhysicalModelSpec")
    assert hasattr(sc, "StereoCentralRayFieldModel")
    assert hasattr(sc, "StereoImagePair")
    assert hasattr(sc, "build_charuco_board")
    assert hasattr(sc, "detect_charuco_corners")
    assert hasattr(sc, "fit_opencv_stereo_from_image_dirs")
    assert hasattr(sc, "fit_stereo_central_rayfield_from_image_dirs")
    assert hasattr(sc, "fit_stereo_zernike_origin_field_from_image_dirs")
    assert hasattr(sc, "load_stereo_central_rayfield")
    assert hasattr(sc, "refine_charuco_corners")
    assert hasattr(sc, "save_stereo_central_rayfield")
    assert hasattr(sc, "select_physical_model_from_rayfield")

    # Tier 2 — result/report dataclasses
    assert hasattr(sc, "OpticalModelSelectionReport")
    assert hasattr(sc, "ParallelPlateFromRayfieldFitResult")
    assert hasattr(sc, "PhysicalModelFitResult")
    assert hasattr(sc, "ReconstructionComparisonReport")
    assert hasattr(sc, "ReconstructionErrorReport")
    assert hasattr(sc, "ReconstructionResult")
    assert hasattr(sc, "StereoCentralRayFieldFitReport")
    assert hasattr(sc, "StereoCentralRayFieldFitResult")
    assert hasattr(sc, "StereoOpenCVCalibrationReport")
    assert hasattr(sc, "StereoOpenCVCalibrationResult")
    assert hasattr(sc, "StereoZernikeOriginFieldFitResult")


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
    from stereocomplex.physics import PhysicalModelSpec
    from stereocomplex.rayfields import ZernikeOriginFieldConfig
    from stereocomplex.physics import (
        CentralPinholeModel,
        CentralBrownConradyModel,
        select_physical_model_from_rayfield,
    )

    assert ZernikeOriginFieldConfig(image_size=(640, 480)).max_order == 4
    assert PhysicalModelSpec("test", CentralPinholeModel, __import__("numpy").zeros(0)).name == "test"
