from stereocomplex.api import (
    # Tier 1 — user entry points
    CharucoBoardSpec,
    PhysicalModelSpec,
    StereoCentralRayFieldModel,
    compare_opencv_stereo_calibration,
    StereoImagePair,
    build_charuco_board,
    detect_charuco_corners,
    fit_opencv_stereo_from_image_dirs,
    fit_stereo_central_rayfield_from_image_dirs,
    fit_stereo_zernike_origin_field_from_image_dirs,
    load_stereo_central_rayfield,
    refine_charuco_corners,
    save_stereo_central_rayfield,
    select_physical_model_from_rayfield,
    # Tier 2 — result/report dataclasses
    OpticalModelSelectionReport,
    ParallelPlateFromRayfieldFitResult,
    PhysicalModelFitResult,
    ReconstructionComparisonReport,
    ReconstructionErrorReport,
    ReconstructionResult,
    StereoCentralRayFieldFitReport,
    StereoCentralRayFieldFitResult,
    StereoOpenCVCalibrationReport,
    StereoOpenCVCalibrationResult,
    StereoZernikeOriginFieldFitResult,
)

__all__ = [
    # Tier 1
    "CharucoBoardSpec",
    "PhysicalModelSpec",
    "StereoCentralRayFieldModel",
    "calibrate_central",
    "calibrate_noncentral",
    "calibrate_opencv",
    "compare_opencv_stereo_calibration",
    "StereoImagePair",
    "build_charuco_board",
    "detect_charuco_corners",
    "fit_opencv_stereo_from_image_dirs",
    "fit_stereo_central_rayfield_from_image_dirs",
    "fit_stereo_zernike_origin_field_from_image_dirs",
    "load_stereo_central_rayfield",
    "refine_charuco_corners",
    "save_stereo_central_rayfield",
    "select_physical_model_from_rayfield",
    # Tier 2
    "OpticalModelSelectionReport",
    "ParallelPlateFromRayfieldFitResult",
    "PhysicalModelFitResult",
    "ReconstructionComparisonReport",
    "ReconstructionErrorReport",
    "ReconstructionResult",
    "StereoCentralRayFieldFitReport",
    "StereoCentralRayFieldFitResult",
    "StereoOpenCVCalibrationReport",
    "StereoOpenCVCalibrationResult",
    "StereoZernikeOriginFieldFitResult",
]

# Short aliases for the most common entry points.
calibrate_opencv = fit_opencv_stereo_from_image_dirs
calibrate_central = fit_stereo_central_rayfield_from_image_dirs
calibrate_noncentral = fit_stereo_zernike_origin_field_from_image_dirs
identify_optics = select_physical_model_from_rayfield
