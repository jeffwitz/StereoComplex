# DEEPCODE.md — StereoComplex (rtk-accelerated)

Token-optimized command reference for this repo. Every command below is an rtk
rewrite of a verbose standard tool. Activation: `rtk init -g`.

## Project at a glance

```
StereoComplex/              # Python 3.10+, GPL-2.0
├── src/stereocomplex/      # Package (api/, core/, ray3d/, eval/, sim/, physics/, cli/)
├── tests/                  # 18 test files, 43 tests (pytest)
├── docs/                   # Sphinx + MyST (make html)
├── paper/                  # Reproducibility experiments
├── examples/notebooks/     # Jupyter walkthroughs
├── dataset/                # Synthetic datasets (clean, blur, compression_sweep)
├── models/                 # Exported ray-field models
├── scripts/                # validate_local.sh, check_docs_nav.py
├── pyproject.toml          # setuptools, ruff config
└── CLAUDE.md               # API audit & reorganization plan (v0.3→v0.4)
```

## Quick reference — common tasks

### Source exploration

```bash
rtk ls src/stereocomplex           # Compact directory tree
rtk ls src/stereocomplex -u        # Ultra-compact (ASCII icons)
rtk find "*.py" src/stereocomplex  # Flat file list
rtk grep "def fit_" src/stereocomplex  # Find all fitting functions
```

### Reading files

```bash
rtk read src/stereocomplex/__init__.py        # Full file
rtk read src/stereocomplex/api/calibration.py -l aggressive  # Signatures only
rtk smart src/stereocomplex/ray3d/central_ba.py  # 2-line heuristic summary
```

### Python tests (pytest)

```bash
rtk test pytest tests/                               # All tests, failures only
rtk test pytest tests/test_physical_model_selection.py  # Single file
rtk test pytest tests/ -k "brown"                    # Filter by keyword
rtk test pytest tests/ -x                            # Stop on first failure
rtk pytest -v                                        # Verbose (when you need all output)
```

### Linting (ruff)

```bash
rtk ruff check src/                  # Lint src, grouped by rule/file
rtk ruff check src/ tests/           # Lint everything
rtk ruff format --check .            # Check formatting
```

### Git operations

```bash
rtk git status                       # Compact status
rtk git diff                         # Condensed diff
rtk git log -n 10                    # One-line commits
rtk git add                          # Stage all -> "ok"
rtk git commit -m "fix: …"          # Commit -> "ok abc1234"
rtk git push                         # Push -> "ok main"
rtk git pull                         # Pull -> "ok 3 files +10 -2"
```

### Build / docs

```bash
make -C docs html                   # Sphinx HTML docs (raw — no rtk wrapper)
rtk err make -C docs html           # Errors only from docs build
rtk err make -C docs latexpdf       # Errors only from PDF build
```

### Dependency management

```bash
rtk pip list                         # Compact Python packages
rtk pip outdated                     # Outdated packages
rtk pip install -e .                 # Editable install (raw — short output anyway)
```

### Synthetic dataset generation (CLI)

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset \
  --out dataset/v1 --scenes 1 --frames-per-scene 8 --width 640 --height 480

.venv/bin/python -m stereocomplex.cli validate-dataset dataset/v1
```

### Run the validation script

```bash
rtk summary bash scripts/validate_local.sh   # Heuristic summary of script output
```

### Log inspection (if any)

```bash
rtk log app.log                     # Deduplicated log lines
```

## Token savings benchmarks (estimated)

| Task | Raw command tokens | rtk tokens | Savings |
|---|---|---|---|
| `find *.py src/` | ~120 | ~15 | ~87% |
| `pytest tests/` (18 tests) | ~400 | ~40 | ~90% |
| `git status` | ~80 | ~12 | ~85% |
| `git log -10` | ~180 | ~20 | ~89% |
| `ruff check src/` | ~60 | ~18 | ~70% |
| `make -C docs html` (errors only) | ~300 | ~25 | ~92% |

## Tips for this repo

- **venv path**: Always prefix Python commands with `.venv/bin/python` or use `.venv/bin/` directly.
- **ruff config**: Line length 100, target py310 — see `pyproject.toml` `[tool.ruff]`.
- **pytest markers**: `integration` marker for end-to-end tests. Use `rtk test pytest tests/ -m "not integration"` to skip them.
- **git hook**: `rtk init -g` was already run — Bash commands are auto-rewritten.
- **43 tests**: Expected after Phase 4 of the CLAUDE.md reorganization plan.
- **`CLAUDE.md`** is the spec for the ongoing API refactoring — 5 phases, one commit each.
