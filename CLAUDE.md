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
- Slow: `rtk .venv/bin/python -m pytest -m slow` → **39 passed** (0 failures).

`CMOPhysicalStereoModel` carries a **per-axis SE(3)** (one SE(3) per axis, not a
mutualised transform) — `n_parameters` is **19** with a shared principal point,
**21** with `share_principal_point=False`. This is intentional and validated: a
shared SE(3) fits the rayfields measurably worse.

Gotcha for any recovery test or diagnostic on this model: `f_obj_mm` and
`telecentric_offset_mm` are exactly degenerate — both enter `ray()` only via
`z_pupil = working_distance - f_obj + telecentric_offset`. Only their difference
is identifiable; assert `working_distance` / `b` / `f_tube` and
`f_obj - telecentric_offset`, never `f_obj` alone. The four
`test_cmo_physical_model.py` slow tests were aligned to this in commit `b13d71e`.

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

### Phase 2 — Joint N-rayfield bundle adjustment — CDC

Status: **specification ready, not implemented.** This subsection is the cahier
des charges — implement directly from it. It is the main next technical
milestone; until it ships, any N-camera result is scaffolding, not validated
N-camera calibration.

**Goal.** Generalize the stereo ray-to-point bundle adjustment to `N` named
channels, so `fit_zernike_rayfields_from_multi_camera_observations(...)` becomes
a true N-camera optimizer instead of a `left/right`-only wrapper.

**Grounding — the existing stereo solver.** The reference implementation is
`fit_zernike_rayfield_from_charuco_observations(...)` in
`benchmarks/rayfield_from_observations.py`. Its parameter vector is:

```
x = [ c_left (n_zernike) | c_right (n_zernike) | poses (6 * n_poses) ]
```

- `n_modes  = len(ZernikeOriginFieldConfig(image_size, max_order).modes())`.
- `n_zernike = n_modes * 6` per channel; first `n_modes*3` = origin coeffs
  `(n_modes,3)`, last `n_modes*3` = direction coeffs `(n_modes,3)`.
- `poses`: one shared `(rotvec[3], tvec[3])` block per board frame. Both
  channels read the same pose block — the board-to-reference transform is shared.
- Residual per channel = ray-to-point perpendicular distance
  (`_channel_residuals`); total = `concat` over channels, then per-channel
  origin-Z regularization rows appended (`residuals_reg`).

**Architectural decisions (fixed for Phase 2.0 — do not deviate without JFW).**

1. Extrinsics are absorbed into each channel's rayfield. The world frame is the
   reference channel `channel_names[0]`; every channel's Zernike rayfield is
   expressed in that frame, exactly as the stereo `right` field works today.
   → **No explicit rig SE(3) parameters in Phase 2.0.**
2. An explicit shared rig SE(3) (CDC constraint "optionally share a rigid rig
   transform") is **deferred to Phase 2.1**, a separate task. Do not add it now.
3. One shared board pose per frame, unchanged.
4. Channel order is `obs.channel_names`; reference channel is index 0.

**Generalized parameter layout.**

```
x = [ c_ch0 | c_ch1 | ... | c_ch{N-1} | poses (6 * n_poses) ]
```

Channel block `k` occupies `x[k*n_zernike : (k+1)*n_zernike]` with the same
origin/direction split as stereo; the pose block is unchanged. For `N == 2` and
`channel_names == ("left","right")` this is **byte-identical** to the current
stereo layout — that identity is what makes step 5 below provable.

**Implementation steps (ordered, each with a gate).**

1. Add a layout helper (free function or frozen dataclass) in
   `rayfield_from_observations.py` returning the slice for
   `(channel_index, "origin"|"direction")` and the pose slice. No behavior
   change. Gate: unit test of the slice arithmetic.
2. Write `fit_zernike_rayfields_n_camera(obs: MultiCameraCharucoObservationSet,
   image_size, intrinsics_by_channel, max_order=4, initial_poses_R=None,
   initial_poses_t=None, *, max_nfev=300, origin_reg_weight=1e-3)
   -> tuple[MultiCameraZernikeRayField, ZernikeFitDiagnostics]`. Reuse
   `_precompute`, `_CachedGroup`, `_channel_residuals` unchanged — iterate
   channels instead of hardcoding L/R. Pose init uses the reference channel:
   `estimate_initial_poses_from_central_pinhole` with
   `K_ref = intrinsics_by_channel[channel_names[0]]` and the reference channel's
   pixels — add a reference-channel adapter if that helper does not accept the
   multi-camera container directly.
3. Bounds & regularization: replicate the stereo per-channel bounds (origin Z
   ∈ ±20 mm, direction ∈ ±0.5, pose ∈ x0±0.3) and the origin-Z regularization
   rows, once per channel, in channel order.
4. Make stereo `fit_zernike_rayfield_from_charuco_observations` **delegate** to
   the N-camera function for the 2-channel case — but only after step 5 passes.
   Until then leave the stereo function untouched (no duplicated business logic
   once delegation is proven).
5. **Non-regression gate (mandatory).** Capture a numerical snapshot (recovered
   coeffs, poses, `ray_rms_mm`, `nfev`) of
   `fit_zernike_rayfield_from_charuco_observations` on a fixed seeded oracle, run
   the same case through `fit_zernike_rayfields_n_camera`, and assert
   **max abs diff == 0.0** (identical residual order, x0, and bounds → the
   optimizer must take the identical path). If not exactly 0, the layout/order
   diverged — fix before proceeding.
6. Wire `fit_zernike_rayfields_from_multi_camera_observations(...)`: drop the
   `NotImplementedError`, route every topology through
   `fit_zernike_rayfields_n_camera`. Keep the existing intrinsics validation.
7. Return a `MultiCameraZernikeRayField` (`from_fields`, channel order
   preserved) and a `ZernikeFitDiagnostics` with
   `channel_names = obs.channel_names`.

**Non-regression contract.**

- `fit_zernike_rayfield_from_charuco_observations` public signature unchanged.
- Stereo numerical output bit-exact (step 5 gate == 0.0).
- Fast + slow suites: no new failures beyond the 4 known stale CMO tests.
- `ruff check src/` stays at 0.

**Out of scope / do not touch.**

- `paper/`, the CMO physics (`physics/cmo*`), the central rayfield façade
  `api/calibration.py::fit_stereo_central_rayfield_*`.
- No explicit rig SE(3) (that is Phase 2.1).
- No new public symbols beyond what `advanced/__init__.py` already exports.

**Validation anchors (Phase 4 closes on these).**

- `build_pinhole_n_camera_oracle(channel_names=("cam0","cam1","cam2","cam3"))`
  + noise-free `simulate_charuco_observations_from_camera_fields(...)` →
  `fit_zernike_rayfields_n_camera` converges with `diag.ray_rms_mm < 1e-3` mm
  and recovered Zernike coefficients ≈ 0 (a central pinhole has no
  origin/direction field).
- Recovered board poses within `1e-3` rad / `1e-2` mm of the simulated poses.
- Stereo equivalence: noise-free 2-channel `ray_rms_mm` matches the existing
  stereo solver to machine precision (step 5).

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
