# CLAUDE.md — Close the direct-vs-rayfield experiment (Phase 8)

## Mission

Notebook 08 currently sets up the comparison between direct ChArUco
inversion (pipeline A) and rayfield-mediated inversion (pipeline B), but
does not actually run pipeline A nor measure the rayfield from images in
pipeline B. The infrastructure exists; the experiment does not.

This CDC closes that gap. After this work:

1. Pipeline A runs on at least the CMO oracle, fitting optical
   parameters and board poses jointly to ChArUco corner observations.
2. Pipeline B runs from corner observations through a Zernike rayfield
   fit, then through ray-space model selection — no shortcut to the
   oracle rayfield.
3. The two pipelines are compared head-to-head with three quantitative
   metrics: BIC ranking, parameter-recovery RMS, and Schur-complement
   condition number.
4. The result is reproducible from `python examples/notebooks/08_direct_vs_rayfield_inversion.py`
   and documented in `docs/DIRECT_VS_RAYFIELD_INVERSION.md` with real
   numbers (not placeholders).

The work is split into 5 phases with one commit per phase. Each phase
keeps the existing 102 tests passing and adds new tests of its own.

## Hard constraints

- All 102 existing tests continue to pass after each phase.
- New tests added must pass with `pytest -W error::DeprecationWarning`.
- No new third-party dependencies. numpy, scipy, opencv-contrib-python-headless only.
- The `FAST` flag in notebook 08 must remain a real fast mode (≤ 60 s).
  Full mode (`FAST = False`) may take up to 15 minutes on the CMO oracle
  but must complete deterministically with `seed=42`.
- Document every limitation discovered. If pipeline A diverges on some
  oracles, that is a result, not a failure — capture it.

## Phase 1 — Densify the ChArUco simulator

**Goal:** ensure pipeline A has at least 30 visible corners per frame on
all six oracles, so joint optimization of optical parameters and board
poses is well-conditioned.

### What to change

In `charuco_observation_simulator.py`:

- Add `min_corners_per_frame` kwarg (default 30) to
  `simulate_charuco_observations_from_rayfield`. After projecting all
  corners for a given pose, if fewer than `min_corners_per_frame` are
  visible, the pose is rejected and a new pose is sampled.
- Add `max_pose_attempts` kwarg (default 200) to bound rejection sampling.
- Add `pose_jitter_deg` kwarg (default 5.0) controlling random rotation
  of the board around its normal axis.
- Return a sampling diagnostic dataclass `SamplingDiagnostics` with
  fields: `n_poses_requested`, `n_poses_accepted`, `n_attempts_used`,
  `mean_corners_per_frame`, `min_corners`, `max_corners`. Expose it on
  `CharucoObservationSet.diagnostics`.

### Tests to add

- `test_simulator_meets_min_corners_per_frame_on_cmo_oracle`
- `test_simulator_diagnostics_report_correct_counts`
- `test_simulator_returns_zero_poses_when_min_unsatisfiable`

### Verification

- 102 + 3 = 105 tests pass.
- `python examples/notebooks/08_direct_vs_rayfield_inversion.py` still
  runs in FAST mode without errors.

## Phase 2 — Run pipeline A on the CMO oracle

**Goal:** make `fit_direct_model_from_observations` actually converge on
the CMO oracle for all three candidate models.

Add `estimate_initial_poses_from_central_pinhole` to `direct_inversion.py`
using `cv2.solvePnP`. Refactor `fit_direct_model_from_observations` to
accept optional pose initialization. Add 4 new tests.

### Tests to add

- `test_pose_initialization_close_to_truth_on_pinhole_oracle`
- `test_direct_fit_recovers_brown_on_brown_oracle`
- `test_direct_fit_converges_on_cmo_oracle_with_three_candidates`
- `test_direct_fit_BIC_correctly_orders_brown_oracle`

### Verification

- 105 + 4 = 109 tests pass.
- Notebook pipeline A section executes and prints results.

## Phase 3 — Image-based Zernike rayfield in pipeline B

**Goal:** stop using the oracle rayfield directly. Fit a Zernike rayfield
from ChArUco observations, then use that for ray-space model selection.

Add `fit_zernike_rayfield_from_charuco_observations` to a new module
`stereocomplex.benchmarks.rayfield_from_observations`. Add 3 tests.
Update notebook 08 to use the fitted rayfield.

### Verification

- 109 + 3 = 112 tests pass.
- Notebook 08 still selects `cmo_physical_shared` on CMO oracle.

## Phase 4 — Schur-complement diagnostics

**Goal:** quantify the conditioning advantage of pipeline B over
pipeline A using the Schur-complement formula.

Add `compute_pipeline_diagnostics` to `inverse_problem_diagnostics.py`.
Add 3 tests. Add comparison table to notebook.

### Verification

- 112 + 3 = 115 tests pass.
- Notebook 08 prints the comparison table with real numbers.

## Phase 5 — Documentation and result interpretation

**Goal:** turn the new numbers into a presentable result.

Update `docs/DIRECT_VS_RAYFIELD_INVERSION.md` with real results,
add CHANGELOG entry for v0.5.3, update notebooks README.

### Verification

- 115 tests still pass.
- Notebook 08 FAST mode ≤ 60 s.
- Tag `v0.5.3`.

## Workflow expectations

- One commit per phase, in order.
- Run `pytest -q -W error::DeprecationWarning` after each phase.
- Run notebook 08 in FAST mode after each phase.
- After phase 5, tag `v0.5.3`.

## Out of scope

- Real-image data. This stays synthetic.
- Pipeline A on the exotic Zernike oracle.
- Confidence intervals or bootstrap resampling.
- Six-oracle full sweep (pinhole, Brown, CMO only for now).
