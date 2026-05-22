# CLAUDE.md — StereoComplex N-camera extension

## Project purpose

StereoComplex is a Python library for non-central stereo camera calibration via
Zernike rayfields. The current production path is still stereo (`left`/`right`),
but the active refactor on `develop` is preparing the codebase for 1-to-N camera
calibration on a shared calibration target.

## Branch policy

- `origin/develop` is the source of truth for current code.
- Treat `develop` as the real implementation branch for large refactors and
  substantial code changes.
- Do not use `main` / `origin/main` as the reference branch for this work unless
  JFW explicitly asks for a release/backport.

## CMO paper status — DONE

26 pages, 31 refs, submitted-ready. No further changes to `paper/cmo/`.

## Current global status

The N-camera migration is not complete, but Phase 1 is now partially implemented
behind compatibility wrappers:

- Existing stereo APIs still work and remain the validated production path.
- New N-camera containers and facades exist for observations, Zernike rayfields,
  CMO telecentric channels, calibration results, and benchmark oracles.
- The new public `calibrate(cameras=...)` API exists, but currently routes only
  the `("left", "right")` case to the existing stereo central rayfield fit.
- Important scope note: the current work validates scaffolding, API shape, data
  containers, and stereo equivalence. It does **not** validate a real N-camera
  calibration algorithm yet.
- A true N-camera bundle adjustment is not implemented yet; Phase 2 is the core
  algorithmic task still ahead.
- Latest full test baseline on `develop`: `119 passed, 39 deselected`.

## Active task: N-camera calibration

### Phase 1 — Refactor camera model to support N channels

Goal: replace hard-coded `left_pixels` / `right_pixels` assumptions with named
camera collections where each camera has its own pixels, intrinsics, Zernike
coefficients, and optional physical model.

Done on `develop`:

- `benchmarks/charuco_observation_simulator.py`
  - Added `MultiCameraCharucoObservationSet`.
  - Added `CharucoObservationSet.to_multi_camera()`.
  - Added `simulate_charuco_observations_from_camera_fields(...)`.
- `benchmarks/rayfield_from_observations.py`
  - `ZernikeFitDiagnostics` exposes `channel_names` and `n_channels`.
  - Added `fit_zernike_rayfields_from_multi_camera_observations(...)`.
  - Current implementation accepts the multi-camera container but only routes
    the `left`/`right` case through the existing stereo solver.
- `rayfields/zernike_origin_field.py`
  - Added `ZernikeRayFieldChannel`.
  - Added `MultiCameraZernikeRayField`.
  - Added `MultiCameraZernikeRayField.from_camera_configs(...)` for independent
    per-channel configs/intrinsics.
- `physics/cmo_physical.py`
  - Added `CMOTelecentricNModel`.
  - Current implementation wraps an existing `CMOTelecentricStereoModel` via
    `CMOTelecentricNModel.from_stereo(...)`.
- `api/calibration.py`
  - Added `CameraSetup`.
  - Added `NCameraCalibrationResult`.
  - Added public `calibrate(...)`.
  - Current implementation routes only `("left", "right")` to
    `fit_stereo_central_rayfield_from_image_dirs(...)`; other topologies raise
    `NotImplementedError`.
- `advanced/__init__.py`
  - Exposes N-camera benchmark/building blocks:
    `MultiCameraCharucoObservationSet`, `MultiCameraOracle`,
    `build_pinhole_n_camera_oracle`,
    `simulate_charuco_observations_from_camera_fields`, and
    `fit_zernike_rayfields_from_multi_camera_observations`.

Remaining Phase 1 work:

- Replace more internal `left`/`right` assumptions with named channel iteration.
- Decide the canonical N-camera observation object for real image calibration,
  not only benchmark/synthetic observations.
- Generalize physical CMO models beyond stereo wrappers.
- Add a stable public result schema for non-stereo calibration outputs.

### Phase 2 — Joint N-rayfield bundle adjustment

Goal: extend the BA objective from two channels to N channels.

Required constraints:

- All cameras share the same board-to-world transform per frame.
- Each camera has independent Zernike origin/direction coefficients.
- Cameras may optionally share a rigid rig transform or fixed relative SE(3).

Current status:

- Not implemented.
- This is the main next technical milestone.
- Existing `fit_zernike_rayfields_from_multi_camera_observations(...)` is a
  compatibility entry point, not a true N-camera optimizer.
- Until this phase is implemented, any N-camera result should be described as
  scaffolding or API validation, not as validated N-camera calibration.

Recommended next implementation step:

- Extract the stereo BA parameter layout in
  `benchmarks/rayfield_from_observations.py` into reusable channel blocks:
  channel observations, per-channel coefficient slices, shared pose vector, and
  residual accumulation over `obs.channel_names`.

### Phase 3 — N-channel model selection

Goal: extend BIC comparison to N channels.

Done on `develop`:

- Added `MultiChannelOpticalModelSelectionReport`.
- Added `aggregate_model_selection_reports(...)`.
- Current aggregation sums BIC by common candidate model name across
  per-channel `OpticalModelSelectionReport` objects.

Remaining work:

- Allow channels to have different candidate families.
- Add shared-rig constraints where model parameters span multiple channels.
- Integrate aggregation into a higher-level N-camera selection workflow.

### Phase 4 — Pinhole × 4 validation

Goal: validate a synthetic 4-camera pinhole rig.

Done on `develop`:

- Added `MultiCameraOracle`.
- Added `build_pinhole_n_camera_oracle(...)`.
- Added `simulate_charuco_observations_from_camera_fields(...)`.

Remaining work:

- Add the actual BA recovery test once Phase 2 exists.
- Validate recovered intrinsics/rayfields/poses against the 4-camera ground
  truth.
- Optionally add a notebook once the code path is stable.

### Phase 5 — N-camera Greenough simulation

Goal: simulate a Greenough binocular plus two additional context cameras.

Current status:

- Not started.

Expected validation:

- Greenough channels should be identified as independent per-channel optical
  centres.
- Context cameras should be identified as pinhole/central channels.

## Recent refactor work already completed on `develop`

In addition to the N-camera scaffolding, `src/stereocomplex/api/calibration.py`
was reduced by extracting helper responsibilities:

- image-pair loading helpers
- refined stereo detection helper
- origin-field image observation collection
- origin-field pinhole seed calibration
- origin-field dataset/pose seeding

This keeps the stereo path validated while making it easier to reuse pieces in
the future N-camera calibration path.

## Validation commands

Use `rtk` for all commands in this repository.

Core validation:

```bash
rtk .venv/bin/python -m pytest
```

Targeted checks used during the refactor:

```bash
rtk .venv/bin/python -m ruff check <changed files>
rtk python3 -m compileall -q <changed files>
rtk .venv/bin/python -m pytest tests/test_public_api.py
rtk .venv/bin/python -m pytest tests/test_rayfield_from_observations.py
rtk .venv/bin/python -m pytest tests/test_charuco_observation_simulator.py
rtk .venv/bin/python -m pytest tests/test_zernike_origin_field.py
rtk .venv/bin/python -m pytest tests/test_cmo_telecentric_model.py
rtk .venv/bin/python -m pytest tests/test_physical_model_selection.py
rtk .venv/bin/python -m pytest tests/test_model_selection_oracles.py
```

## Ruff status (2026-05-22)\n\nBaseline (`E,F` rules): **0 errors**.

| Rule | Description | Fixed | Remaining | Notes |
|---|---|---|---|---|
| E702 | semicolons | 2 | — | |
| E741 | ambiguous var `O` | 3 | — | `# noqa: E741` |
| UP037 | quoted annotations | 36 | — | auto-fix |
| UP006 | PEP585 annotations | 7 | — | auto-fix |
| UP045 | PEP604 optional | 5 | — | auto-fix |
| UP018 | native literals | 1 | — | auto-fix |
| UP035 | deprecated import | 3 | — | |
| B905 | zip without strict | 4 | — | → `strict=True` |
| PERF401 | manual list comp | 2 | — | → list comprehension |
| RUF046 | unnecessary int | 4 | — | `int(round(x))` → `round(x)` |
| B007 | unused loop var | 1 | — | → `_name` |
| B008 | mutable default | 1 | — | → `None` + check |
| RUF059 | unused unpack | 4 | — | → `_name` |
| SIM108 | ternary instead of if | 3 | 2 | 2 in `api/` (skip) |
| PLC0415 | lazy import | 6 | 52 | viz/ done; rest risky |
| E501 | line too long | — | 247 | cosmetic, fix opportunistically |
| PLR2004 | magic values | — | 170 | mostly false positives |
| PLR0913 | too many args | — | 76 | needs dataclass refactoring |
| PLR0915 | too many statements | — | 28 | = §9 refactoring targets |
| PLR0912 | too many branches | — | 17 | complex control flow |
| B023 | loop var in closure | — | 2 | risky, needs careful fix |

## Conventions

- Code, comments, docstrings, and commit messages are in English.
- Communicate with JFW in French.
- Keep backward-compatible stereo APIs unless JFW explicitly asks to break them.
- Do not silently change public API behavior.
- Commit and push each validated task to `origin/develop`.
- Avoid changes under `paper/cmo/` unless explicitly requested.
- All generated/analysis artefacts should be JSON when practical.
- Deep Claude mode: fresh session + this `CLAUDE.md` only.
