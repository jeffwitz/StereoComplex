API reference
=============

This page documents the public StereoComplex API, auto-generated from the
docstrings. For conceptual guides and worked examples see the rest of the
documentation; for the tiered overview and the stability contract see
:doc:`PUBLIC_API`.

The most common entry points are also re-exported at the top level for
convenience — the calibration entry point is reachable both as
``stereocomplex.api.calibrate`` and as ``sc.calibrate``. The
``stereocomplex.advanced`` namespace bundles additional composition and
diagnostic helpers, which are re-exports of the symbols documented below.

.. note::

   StereoComplex v0.x is explicitly unstable: public names may still change.
   Backward-compatible stability commitments start at v1.0 — see
   :doc:`PUBLIC_API`.

Physical optical models — ``stereocomplex.physics``
---------------------------------------------------

Physical optical models (pinhole, Brown-Conrady, inclined parallel plate,
compact-mirror-objective) and ray-space model selection.

.. automodule:: stereocomplex.physics
   :members:
   :show-inheritance:

Zernike rayfields — ``stereocomplex.rayfields``
-----------------------------------------------

Zernike origin / ray-field models and the N-camera rayfield container.

.. automodule:: stereocomplex.rayfields
   :members:
   :show-inheritance:

Synthetic datasets — ``stereocomplex.synthetic``
------------------------------------------------

Synthetic dataset generation: the parallel-plate non-central oracle and
rendered ChArUco scenes.

.. automodule:: stereocomplex.synthetic
   :members:
   :show-inheritance:

Calibration, refinement and I/O — ``stereocomplex.api``
-------------------------------------------------------

Calibration entry points, ChArUco detection / refinement, reconstruction and
model I/O. Symbols re-exported from the namespaces above are documented in
their own sections and omitted here to avoid duplication.

.. automodule:: stereocomplex.api
   :members:
   :show-inheritance:
   :exclude-members: CentralBrownConradyModel, CentralPinholeModel, OpticalModelSelectionReport, ParallelPlateFromRayfieldFitResult, ParallelPlateImageRenderParams, ParallelPlateSyntheticParams, PhysicalModelFitResult, PhysicalModelSpec, PinholeParallelPlateFitParams, PinholeParallelPlateModel, PinholeParallelPlateRayField, RenderedParallelPlateImageDataset, SyntheticStereoDataset, ZernikeOriginField, ZernikeOriginFieldCoefficients, ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients, charuco_inner_corners_object_points, default_physical_model_specs, detected_observations_from_rendered_parallel_plate, fit_parallel_plate_to_zernike_rayfield, fit_physical_model_to_rayfield, generate_parallel_plate_stereo_dataset, intersect_ray_with_z_plane, normal_from_tilts, parallel_plate_ray_from_pixel, pinhole_parallel_plate_ray_from_pixel, pinhole_ray_from_pixel, project_point_with_parallel_plate, rayfield_two_plane_residuals, render_parallel_plate_charuco_images, select_physical_model_from_rayfield
