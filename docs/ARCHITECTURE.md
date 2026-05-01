# Architecture

Goal: iterate quickly on **dataset → geometry → metrics**, from practical
OpenCV-compatible ChArUco workflows to experimental central and non-central
ray-based stereo models, without being blocked by OptiX.

## Modules

- `src/stereocomplex/meta.py`: metadata schema + validation (pixel pitch is mandatory, crop/resize/binning).
- `src/stereocomplex/core/geometry.py`: pixel↔sensor (µm) conversions, minimal pinhole model, triangulation.
- `src/stereocomplex/sim/cpu/`: CPU generator (textured board plane + GT correspondences).
- `src/stereocomplex/synthetic/parallel_plate.py`: inclined parallel-plate
  oracle for physically plausible non-central synthetic rayfields.
- `src/stereocomplex/synthetic/parallel_plate_images.py`: rendered ChArUco
  images with vignetting, blur, noise, and detection-front-end stressors for the
  non-central benchmark.
- `src/stereocomplex/eval/`: metrics/evaluations (oracle, ChArUco detection vs GT, compression sweeps).
- `src/stereocomplex/ray3d/`: central 3D ray-field (Zernike) and point↔ray bundle adjustment (mono/stereo).
- `src/stereocomplex/rayfields/zernike_origin_field.py`: non-central Zernike
  origin-field model `O(u,v)` and direction-field extensions for pixel-to-line
  rayfields.
- `src/stereocomplex/calibration/fit_zernike_origin_field.py`: experimental
  non-central origin-field calibration and staged/complete point-to-ray BA.
- `src/stereocomplex/physics/parallel_plate_fit.py`: second-stage physical
  interpretation of a measured rayfield as a compact pinhole + inclined
  parallel-plate model.
- `src/stereocomplex/benchmarks/parallel_plate_origin_field.py`: controlled
  non-central benchmark, oracle floors, rendered-image evaluation, and Ray2D
  front-end comparison.
- `paper/experiments/`: reproducible scripts used to generate tables/figures in the docs/paper.
- `docs/examples/`: end-to-end runnable examples (used in the documentation).
- `examples/notebooks/04_parallel_plate_origin_field.ipynb`: scientific
  inclined-plate oracle and non-central BA walkthrough.
- `examples/notebooks/05_noncentral_calibration_from_images.ipynb`: practical
  non-central calibration path from image folders.

## Model layers

- **Ray2D / planar ray-field**: a homography plus smooth residual field on the
  board plane. This is a 2D observation refinement step, not a 3D camera model.
- **Central 3D ray-field**: each pixel maps to a unit direction and all rays
  share one center.
- **Non-central 3D rayfield**: each pixel maps to a 3D line
  `(O(u,v), d(u,v))`; the origin field is gauge-fixed by the transverse
  condition `O(u,v) · d(u,v) = 0`.

## Non-central paths

- Use `docs/PARALLEL_PLATE_ORIGIN_FIELD.md` / notebook 04 for the controlled
  validation: the inclined plate is only a synthetic oracle; its physical
  parameters are not fitted.
- Use `docs/NONCENTRAL_FROM_IMAGES.md` / notebook 05 for the practical workflow:
  two image folders plus a `CharucoBoardSpec` feed
  `fit_stereo_zernike_origin_field_from_image_dirs(...)`.

## Planned evolution

- `sim/optix/` will replace the data source without changing the dataset format.
- `ml/` will consume dataset v0 and output a latent `z` + a compact decoder.
