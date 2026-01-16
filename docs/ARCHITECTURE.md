# Architecture

Goal: iterate quickly on **dataset → geometry → metrics**, without being blocked by OptiX.

## Modules

- `src/stereocomplex/meta.py`: metadata schema + validation (pixel pitch is mandatory, crop/resize/binning).
- `src/stereocomplex/core/geometry.py`: pixel↔sensor (µm) conversions, minimal pinhole model, triangulation.
- `src/stereocomplex/sim/cpu/`: CPU generator (textured board plane + GT correspondences).
- `src/stereocomplex/eval/`: metrics/evaluations (oracle, ChArUco detection vs GT, compression sweeps).
- `src/stereocomplex/ray3d/`: central 3D ray-field (Zernike) and point↔ray bundle adjustment (mono/stereo).
- `paper/experiments/`: reproducible scripts used to generate tables/figures in the docs/paper.
- `docs/examples/`: end-to-end runnable examples (used in the documentation).

## Planned evolution

- `sim/optix/` will replace the data source without changing the dataset format.
- `ml/` will consume dataset v0 and output a latent `z` + a compact decoder.
