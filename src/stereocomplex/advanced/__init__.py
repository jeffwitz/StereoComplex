"""Advanced composition API for expert users building custom pipelines."""

from stereocomplex.api.calibration import (
    fit_opencv_stereo_from_dataset,
    fit_opencv_stereo_from_image_pairs,
    fit_stereo_central_rayfield_from_dataset,
    fit_stereo_central_rayfield_from_image_pairs,
)
from stereocomplex.benchmarks.charuco_observation_simulator import (
    MultiCameraCharucoObservationSet,
    simulate_charuco_observations_from_camera_fields,
)
from stereocomplex.benchmarks.model_selection_oracles import (
    MultiCameraOracle,
    build_pinhole_n_camera_oracle,
)
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_zernike_rayfields_from_multi_camera_observations,
)
from stereocomplex.calibration.fit_zernike_origin_field import fit_stereo_zernike_origin_field
from stereocomplex.metrics.rayfield_metrics import (
    RayfieldComparisonReport,
    compare_rayfields_on_planes,
    intersect_rays_with_z_plane,
)
from stereocomplex.metrics.reconstruction_metrics import (
    OracleReconstructionFloorReport,
    compare_3d_reconstruction_with_without_origin_field,
    oracle_reconstruction_floor_report,
    reconstruct_points_central_stereo,
    reconstruct_points_with_origin_fields,
    reconstruct_points_with_parallel_plate_oracle,
    reconstruction_error_report,
    triangulate_two_rays,
)
from stereocomplex.physics.model_selection import (
    default_physical_model_specs,
    fit_physical_model_to_rayfield,
)
from stereocomplex.physics.parallel_plate_fit import fit_parallel_plate_to_zernike_rayfield

__all__ = [
    "MultiCameraCharucoObservationSet",
    "MultiCameraOracle",
    "OracleReconstructionFloorReport",
    "RayfieldComparisonReport",
    "build_pinhole_n_camera_oracle",
    "compare_3d_reconstruction_with_without_origin_field",
    "compare_rayfields_on_planes",
    "default_physical_model_specs",
    "fit_opencv_stereo_from_dataset",
    "fit_opencv_stereo_from_image_pairs",
    "fit_parallel_plate_to_zernike_rayfield",
    "fit_physical_model_to_rayfield",
    "fit_stereo_central_rayfield_from_dataset",
    "fit_stereo_central_rayfield_from_image_pairs",
    "fit_stereo_zernike_origin_field",
    "fit_zernike_rayfields_from_multi_camera_observations",
    "intersect_rays_with_z_plane",
    "oracle_reconstruction_floor_report",
    "reconstruct_points_central_stereo",
    "reconstruct_points_with_origin_fields",
    "reconstruct_points_with_parallel_plate_oracle",
    "reconstruction_error_report",
    "simulate_charuco_observations_from_camera_fields",
    "triangulate_two_rays",
]
