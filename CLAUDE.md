# CLAUDE.md — StereoComplex

Instructions for AI coding agents (Claude Code, Codex) working on this repository.

## 1. What this project is

**StereoComplex** is a Python toolkit for robust stereo calibration and ray-based
3D reconstruction, targeting users who hit the limits of standard OpenCV pinhole
workflows (ChArUco localisation noise, microscope optics, protective glass,
non-central effects, telecentric or CMO objectives).

It exposes three layers — keep them separate when reasoning about code:

| Layer | Meaning | Status |
|---|---|---|
| **Ray2D / planar ray-field** | Homography + smooth residual field on the board plane | stable preprocessing |
| **Central 3D ray-field** | Pixel → 3D direction with a shared camera centre | research prototype |
| **Non-central 3D rayfield** | Pixel → 3D line `(O(u,v), d(u,v))` | research-grade, validated on synthetic + one real CMO case study |

Companion paper (Optics Express submission) lives in `paper/cmo/`. The headline
result is a 26-parameter CMO+SE(3) telecentric model achieving 1.06 px
reprojection on a Pycaso microscope dataset where OpenCV fails (>300 px).

## 2. Current state

- Latest tag: **`v1.0-submission-ready`** (paper submitted to Optics Express).
- `pyproject.toml::version` is still `0.1.0` — known desync, will be bumped at publication.
- 148 tests collected (109 fast + 39 marked `slow`). Default `pytest` excludes `slow`.
- 17.5k LOC `src/`, 4k LOC tests, 70 source files in 12 sub-packages.
- Python ≥ 3.10. Strict mode: `filterwarnings = ["error::DeprecationWarning", ...]`.

## 3. Repository map

```
src/stereocomplex/
├── api/         # Public façade (sc.calibrate_*, sc.identify_optics, ...)
├── physics/     # Brown-Conrady, CMO, parallel plate, model selection
├── ray3d/       # Central BA, stereo BA, virtual rectification
├── rayfields/   # Zernike rayfield models
├── eval/        # ChArUco detection, refinement, prediction methods
├── benchmarks/  # Direct vs rayfield inversion, observation simulators
├── calibration/ # Zernike origin field fits
├── synthetic/   # Parallel plate dataset generation
├── sim/         # CPU simulator, pattern generation, dataset utilities
├── metrics/     # Rayfield + reconstruction metrics
├── viz/         # Figures, plotting primitives, style
├── core/        # Internal: pinhole_fit, image_io
└── cli/         # Command-line entry points (validate-dataset, refine-corners)

tests/           # 30 pytest files, fast by default, opt-in `slow`/`integration`
examples/notebooks/  # 00–09, .py + .ipynb pairs, 00 is the entry point
docs/            # Sphinx docs + Markdown design notes (32 files)
paper/           # cmo/ (Optics Express submission), compression/
dataset/         # Synthetic + real ChArUco datasets
```

**First files to read when starting cold:**
1. `README.md` — high-level user paths
2. `docs/PUBLIC_API.md` — stability contract (critical, see §6)
3. `docs/CONVENTIONS.md` — coordinate frames, pixel-centre convention
4. `docs/ARCHITECTURE.md` — layer separation
5. `src/stereocomplex/__init__.py` — what `sc.X` exposes

## 4. Commands

```bash
# Install (editable)
.venv/bin/python -m pip install -e .[dev,docs]

# Tests — fast subset (default)
.venv/bin/python -m pytest

# Tests — slow + integration
.venv/bin/python -m pytest -m slow
.venv/bin/python -m pytest -m integration

# Lint (ruff config in pyproject.toml: line-length 100, target py310)
ruff check src/
ruff check src/ --fix         # safe auto-fixes
ruff format src/              # only if a formatter pass is explicitly requested

# Docs
make -C docs html
make -C docs latexpdf

# Smoke checks before large changes
.venv/bin/python -m stereocomplex.cli validate-dataset dataset/v0_png
.venv/bin/python -m stereocomplex.cli refine-corners dataset/v0_png \
    --split train --scene scene_0000 --max-frames 2 --method rayfield_tps_robust
```

## 5. Code conventions

- **Python ≥ 3.10**, type hints on all public signatures (`def f(...) -> T:`).
- **Imports**: absolute only (`from stereocomplex.foo import bar`). No relative imports.
- **Docstrings**: required on every public function (no leading `_`).
  Numpydoc-ish style with `Parameters` / `Returns`.
  For algorithmic functions, cite the source paper (DOI or arXiv) in the docstring.
- **Logging, not print**: use the `logging` module. Existing `print()` calls (7
  remaining in `src/`) are legacy and should be migrated when touched.
- **No bare `except:`** — always catch a specific exception.
- **No `TODO`/`FIXME`/`HACK`** in committed code. Either fix it now or open a
  GitHub issue. The repo currently has zero — keep it that way.
- **Line length**: 100 chars (ruff). 360 violations remain in legacy code —
  fix opportunistically, do not bulk-reformat without asking.
- **No semicolon-packed statements** (E702). The pattern `a = x; b = y` on one
  line is an artefact of compact LLM generation; rewrite it on separate lines.

## 6. Public API contract — DO NOT BREAK SILENTLY

Anything in `stereocomplex.api.*` and re-exported through `sc.*` is **public**.
After `v1.0`, public symbols ship with deprecation aliases for at least one
minor version before removal.

Public sub-namespaces: `stereocomplex.{api, advanced, physics, synthetic, rayfields}`.

Internal (may change without notice): `stereocomplex.{core, eval, benchmarks}`,
`paper/`, `docs/examples/`.

Before renaming, removing, or changing the signature of a public function:
1. Open an issue describing the change.
2. Wait for JFW to arbitrate.
3. If approved, ship the change with a `DeprecationWarning`-emitting alias and
   a `CHANGELOG.md` entry.

## 7. Working with JFW

This project uses a triangular workflow:

- **Claude** — diagnostic review, specification (CDC), architectural critique.
- **Codex** — implementation from a written CDC.
- **ChatGPT / DeepSeek** — cross-review when stakes are high.
- **JFW** — arbitrates every decision, signs off on commits.

When you (Claude Code) are dispatched to implement:

- Communicate in **French**. Technical content in French is fine.
- Keep **code, comments, docstrings, commit messages in English**.
- Be **direct, calibrated, honest**. No triumphalism, no padding.
  Say "this works on synthetic, not validated on real data" rather than
  "this achieves excellent performance".
- Acknowledge errors immediately when caught. Do not defend a wrong choice.
- If the user pushes back, do not flip your assessment just to please —
  re-examine and respond honestly.

When unsure about an architectural choice, **stop and ask** rather than
guessing. JFW prefers a short clarifying exchange over a 500-line wrong PR.

## 8. Things to never do

- Push to `main` without a PR.
- Bump `pyproject.toml::version` without a CHANGELOG entry.
- Add a dependency without justification (current core deps: numpy, scipy,
  opencv-contrib-python-headless, pillow — keep the list tight).
- Reformat large swaths of legacy code as a side effect of an unrelated PR.
- Reintroduce `TODO`/`FIXME` markers.
- Suppress a `DeprecationWarning` by silencing it globally; fix the root cause.
- Add `# type: ignore` without a reason in a comment.
- Touch `paper/cmo/manuscript.tex` while the paper is under review (other
  paper directories are fair game).

## 9. Known refactoring targets (priority order)

Diagnosed 2026-05-21. Hot spots ranked by maintenance index (radon):

1. **`eval/charuco_detection.py`** (2133 LOC, MI=0.00).
   - `_eval_one_image` is 478 lines with cyclomatic complexity 126 — the
     single worst function in the repo. Likely a method/refine dispatch that
     accumulated all variants. Refactor into `_PREDICTORS[method]` +
     `_REFINERS[refine]` dispatch tables, then split into
     `eval/predictors/` + `eval/refiners/` sub-packages.
2. **`api/calibration.py`** (1549 LOC, MI=0.00).
   - Three façade functions are 154–235 lines each. Extract a common
     `_run_charuco_to_model_pipeline(detect_cfg, model_factory, ...)` helper.
3. **`physics/cmo_physical.py`** (1388 LOC, MI=4.94).
4. **`physics/cmo.py`** (1252 LOC, MI=2.73).

Lint backlog: 77 ruff errors (53 E702, 9 F401, 8 E741). `ruff check --fix
--unsafe-fixes` clears most. `[tool.ruff.lint]` `select` is currently
unset — only `E`/`F` rules active. Consider expanding to
`["E","F","W","B","UP","SIM","RUF"]` once the backlog is clean.

Docstring coverage: 43.8% on public functions, 26.5% on private. Functions
longer than 80 lines should always have a docstring.

## 10. When stuck

- For algorithmic questions: check `docs/CMO_PHYSICAL_MODEL.md`,
  `docs/CHARUCO_IDENTIFICATION.md`, `docs/DIRECT_VS_RAYFIELD_INVERSION.md`.
- For API behaviour: read the corresponding test file in `tests/` — they are
  the executable specification.
- For "is this a good idea": ask JFW before implementing.
