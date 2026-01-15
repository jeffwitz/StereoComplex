# Start Here

Goal: establish a **reproducible** and **measurable** baseline before moving to OptiX and ML.

## Why this structure

- The specification targets an amortized model (CalibNet) with a compact decoder, but the #1 risk is
  sim→real transfer and numerical conditioning. Therefore we start with:
  1) a stable dataset format,
  2) a canonical geometry convention (µm),
  3) automated metrics.

## What is already implemented

- Dataset v0: `docs/DATASET_SPEC.md` + validator (`validate-dataset`).
- Meta v0 (pixel pitch is mandatory): `src/stereocomplex/meta.py`.
- Minimal geometry core: `src/stereocomplex/core/geometry.py` (pixel↔sensor µm, pinhole, triangulation).
- CPU simulator MVP: `src/stereocomplex/sim/cpu/generate_dataset.py`.
- Oracle evaluation: `src/stereocomplex/eval/oracle.py` (sanity check).
- ChArUco detection evaluation (2D error vs GT): `docs/CHARUCO_IDENTIFICATION.md` + `eval-charuco-detection`.
- Full worked example (raw OpenCV vs ray-field + plots): `docs/RAYFIELD_WORKED_EXAMPLE.md`.

## Next building blocks (recommended order)

1. Add **blur** to the CPU generator (Gaussian, then spatially varying) in physical units via `pitch_um`.
2. Define the compact representation (bases) in `core/model_compact/`.
3. Introduce a first `api/` (calibrate/reconstruct) that does not depend on OptiX.
4. Replace the data source by OptiX without changing dataset v0.

## Useful commands

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/charuco --pattern charuco --blur-fwhm-um 4
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli validate-dataset dataset/charuco
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-oracle dataset/charuco
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/charuco --method rayfield
```
