# Contributing

StereoComplex is an active research prototype; contributions are welcome, especially around:

- documentation clarity and examples,
- robustness and performance of corner refinement / ray-based reconstruction,
- bug reports with minimal repro datasets.

## Development setup

```bash
.venv/bin/python -m pip install -e .[dev,docs]
```

## Tests and docs

```bash
.venv/bin/python -m pytest
make -C docs html
make -C docs latexpdf
```

To run integration/process tests (requires `opencv-contrib-python`):

```bash
.venv/bin/python -m pytest -m integration
```

## Recommended “global” checks (before big changes)

These are not unit tests, but they are cheap and catch most regressions:

```bash
# 1) Validate cached datasets (fast structure + meta consistency).
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/v0_png
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/v0
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/compression_sweep

# 2) Smoke-test one expensive pipeline without regenerating data.
.venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png --split train --scene scene_0000 --max-frames 2 --method rayfield_tps_robust
```

## Public API

Please keep backward compatibility for `stereocomplex` (top-level re-exports) and `stereocomplex.api`.
