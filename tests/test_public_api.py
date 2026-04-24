from __future__ import annotations


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
