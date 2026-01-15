# Architecture (MVP)

Goal: iterate quickly on **dataset → geometry → metrics**, without being blocked by OptiX.

## Modules

- `src/stereocomplex/meta.py`: metadata schema + validation (pixel pitch is mandatory, crop/resize/binning).
- `src/stereocomplex/core/geometry.py`: pixel↔sensor (µm) conversions, minimal pinhole model, triangulation.
- `src/stereocomplex/sim/cpu/`: CPU generator MVP (textured plane + GT correspondences).
- `src/stereocomplex/eval/`: metrics/evaluations (oracle, ChArUco detection vs GT, compression sweeps).

## Planned evolution

- `sim/optix/` will replace the data source without changing the dataset format.
- `ml/` will consume dataset v0 and output a latent `z` + a compact decoder.
