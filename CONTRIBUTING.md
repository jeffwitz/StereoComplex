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

## Public API

Please keep backward compatibility for `stereocomplex` (top-level re-exports) and `stereocomplex.api`.

