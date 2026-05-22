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
- Quality gates (lint + tests): see **Build, lint & test status** below — it is
  authoritative and must be kept up to date.

## Build, lint & test status (2026-05-22)

Authoritative quality snapshot. Update this block whenever a gate moves.

### Lint — ruff

- Enforced gate: `rtk .venv/bin/python -m ruff check src/` → **0 errors**.
- The gate is narrow: `[tool.ruff.lint].select` is unset, so only the default
  `E`/`F` rules gate. A broader run
  (`ruff check src/ --select E,F,W,B,UP,SIM,RUF,PERF,PLR,PLC,C90`) still reports
  **658 errors**.
- A first cleanup batch is fully cleared (0 remaining): E702, E741,
  UP006/018/035/037/045, B905, B007, B008, PERF401, RUF046, RUF059.
- Remaining backlog, by impact:

| Rule | Description | Remaining | Notes |
|---|---|---|---|
| E501 | line too long | 248 | cosmetic, fix opportunistically |
| PLR2004 | magic value comparison | 170 | mostly false positives |
| PLR0913 | too many arguments | 76 | needs dataclass refactors |
| PLC0415 | import outside top-level | 60 | viz/ done; rest are deliberate lazy imports |
| C901 / PLR0915 | complex / too many statements | 29 / 28 | complexity hot-spots |
| PLR0912 | too many branches | 16 | complex control flow |
| RUF005 / RUF001-3 | concat / ambiguous unicode | 9 / 11 | low risk |
| B023 | loop variable bound in closure | 2 | **risky — needs careful fix** |
| SIM108, PLR0911, PLC0414, RUF022 | misc | ~9 | low risk |

- Widening `select` to `["E","F","W","B","UP","SIM","RUF"]` is only safe once the
  `B`/`RUF` rows above are cleared.

### Tests

- Fast: `rtk .venv/bin/python -m pytest` → **120 passed, 39 deselected**.
- Slow: `rtk .venv/bin/python -m pytest -m slow` → **35 passed, 4 FAILED**.

The 4 slow failures are all in `tests/test_cmo_physical_model.py`:

| Test | Failing assertion |
|---|---|
| `test_cmo_physical_oracle_recovery_no_distortion` | `n_parameters == 17` (model gives 19) |
| `test_cmo_physical_oracle_recovery_with_distortion` | parameter-vector slices assume the old 17-param layout |
| `test_cmo_aligned_mode_represents_offset_oracle` | `n_parameters == 19` (model gives 21) |
| `test_select_with_mixed_per_channel_and_stereo_shared_candidates` | `n_parameters == 17` (model gives 19) |

**These are stale test assertions, not a code regression** — they fail identically
on `main`, so they predate the N-camera work:

- `CMOPhysicalStereoModel` now carries a **per-axis SE(3)** (one SE(3) per axis)
  instead of a mutualised transform. This is intentional and validated: a shared
  SE(3) fits the rayfields measurably worse.
- The per-axis SE(3) adds 2 parameters: shared-principal-point case `17 → 19`;
  `share_principal_point=False` case `19 → 21`.
- The 4 tests still encode the old counts and the old parameter-vector slice
  layout (`[4:6]`, `[7:17]`, `[7]`).

Fix: **update the 4 tests** to the new counts and slice indices.
**Do not revert the model** — per-axis SE(3) (19 / 21 parameters) is the chosen,
validated design.

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

## Conventions

- Code, comments, docstrings, and commit messages are in English.
- Communicate with JFW in French.
- Keep backward-compatible stereo APIs unless JFW explicitly asks to break them.
- Do not silently change public API behavior.
- Commit and push each validated task to `origin/develop`.
- Avoid changes under `paper/cmo/` unless explicitly requested.
- All generated/analysis artefacts should be JSON when practical.
- Deep Claude mode: fresh session + this `CLAUDE.md` only.
