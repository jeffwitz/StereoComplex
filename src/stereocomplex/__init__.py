from stereocomplex import meta
from stereocomplex.api import (
    # Tier 1 — user entry points
    CharucoBoardSpec,
    PhysicalModelSpec,
    StereoCentralRayFieldModel,
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

# Symbols moved to sub-namespaces. Remain accessible with a DeprecationWarning
# until v1.0 to preserve backward compatibility.
_DEPRECATED_NAMES: dict[str, str] = {
    # Tier 3 → stereocomplex.advanced
    "OracleReconstructionFloorReport": "stereocomplex.advanced",
    "RayfieldComparisonReport": "stereocomplex.advanced",
    "compare_3d_reconstruction_with_without_origin_field": "stereocomplex.advanced",
    "compare_rayfields_on_planes": "stereocomplex.advanced",
    "default_physical_model_specs": "stereocomplex.advanced",
    "fit_opencv_stereo_from_dataset": "stereocomplex.advanced",
    "fit_opencv_stereo_from_image_pairs": "stereocomplex.advanced",
    "fit_parallel_plate_to_zernike_rayfield": "stereocomplex.advanced",
    "fit_physical_model_to_rayfield": "stereocomplex.advanced",
    "fit_stereo_central_rayfield_from_dataset": "stereocomplex.advanced",
    "fit_stereo_central_rayfield_from_image_pairs": "stereocomplex.advanced",
    "fit_stereo_zernike_origin_field": "stereocomplex.advanced",
    "intersect_rays_with_z_plane": "stereocomplex.advanced",
    "oracle_reconstruction_floor_report": "stereocomplex.advanced",
    "reconstruct_points_central_stereo": "stereocomplex.advanced",
    "reconstruct_points_with_origin_fields": "stereocomplex.advanced",
    "reconstruct_points_with_parallel_plate_oracle": "stereocomplex.advanced",
    "reconstruction_error_report": "stereocomplex.advanced",
    "triangulate_two_rays": "stereocomplex.advanced",
    # Tier 4 → stereocomplex.synthetic
    "ParallelPlateImageRenderParams": "stereocomplex.synthetic",
    "ParallelPlateSyntheticParams": "stereocomplex.synthetic",
    "RenderedParallelPlateImageDataset": "stereocomplex.synthetic",
    "SyntheticStereoDataset": "stereocomplex.synthetic",
    "charuco_inner_corners_object_points": "stereocomplex.synthetic",
    "detected_observations_from_rendered_parallel_plate": "stereocomplex.synthetic",
    "generate_parallel_plate_stereo_dataset": "stereocomplex.synthetic",
    "normal_from_tilts": "stereocomplex.synthetic",
    "parallel_plate_ray_from_pixel": "stereocomplex.synthetic",
    "pinhole_ray_from_pixel": "stereocomplex.synthetic",
    "project_point_with_parallel_plate": "stereocomplex.synthetic",
    "render_parallel_plate_charuco_images": "stereocomplex.synthetic",
    # Benchmark symbols → stereocomplex.benchmarks.parallel_plate_origin_field
    # (cannot live in stereocomplex.synthetic due to circular imports)
    "BenchmarkReport": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "RenderedImageBenchmarkReport": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "make_default_parallel_plate_charuco_board": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "make_default_parallel_plate_charuco_dataset": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "make_default_parallel_plate_dataset": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "make_parallel_plate_wide_coverage_dataset": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "run_parallel_plate_origin_field_benchmark": "stereocomplex.benchmarks.parallel_plate_origin_field",
    "run_parallel_plate_rendered_image_benchmark": "stereocomplex.benchmarks.parallel_plate_origin_field",
    # Tier 4 → stereocomplex.physics
    "CentralBrownConradyModel": "stereocomplex.physics",
    "CentralPinholeModel": "stereocomplex.physics",
    "PinholeParallelPlateFitParams": "stereocomplex.physics",
    "PinholeParallelPlateModel": "stereocomplex.physics",
    "PinholeParallelPlateRayField": "stereocomplex.physics",
    "pinhole_parallel_plate_ray_from_pixel": "stereocomplex.physics",
    "rayfield_two_plane_residuals": "stereocomplex.physics",
    # Tier 4 → stereocomplex.rayfields
    "ZernikeOriginField": "stereocomplex.rayfields",
    "ZernikeOriginFieldCoefficients": "stereocomplex.rayfields",
    "ZernikeOriginFieldConfig": "stereocomplex.rayfields",
    "ZernikeRayField": "stereocomplex.rayfields",
    "ZernikeRayFieldCoefficients": "stereocomplex.rayfields",
    # Tier 5 — singular form removed from top-level; use stereocomplex.physics
    "intersect_ray_with_z_plane": "stereocomplex.physics",
}


def __getattr__(name: str):
    import warnings

    if name in _DEPRECATED_NAMES:
        new_home = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"stereocomplex.{name} is moving to {new_home}. "
            f"Direct import from stereocomplex will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(new_home, fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module 'stereocomplex' has no attribute {name!r}")
