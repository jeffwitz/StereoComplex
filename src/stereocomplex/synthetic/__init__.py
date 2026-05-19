"""Synthetic dataset generators and image renderers."""

from stereocomplex.synthetic.parallel_plate import (
    ParallelPlateSyntheticParams,
    SyntheticStereoDataset,
    generate_parallel_plate_stereo_dataset,
    normal_from_tilts,
    parallel_plate_ray_from_pixel,
    pinhole_ray_from_pixel,
    project_point_with_parallel_plate,
)
from stereocomplex.synthetic.parallel_plate_images import (
    ParallelPlateImageRenderParams,
    RenderedParallelPlateImageDataset,
    charuco_inner_corners_object_points,
    detected_observations_from_rendered_parallel_plate,
    render_parallel_plate_charuco_images,
)

__all__ = [
    "ParallelPlateImageRenderParams",
    "ParallelPlateSyntheticParams",
    "RenderedParallelPlateImageDataset",
    "SyntheticStereoDataset",
    "charuco_inner_corners_object_points",
    "detected_observations_from_rendered_parallel_plate",
    "generate_parallel_plate_stereo_dataset",
    "normal_from_tilts",
    "parallel_plate_ray_from_pixel",
    "pinhole_ray_from_pixel",
    "project_point_with_parallel_plate",
    "render_parallel_plate_charuco_images",
]
