# CLAUDE.md — StereoComplex N-camera extension

## Project purpose

StereoComplex is a Python library for non-central stereo camera calibration via
Zernike rayfields. Currently supports exactly 2 cameras (left/right). The N-camera
extension generalises to 1 to N cameras observing a shared calibration target.

## CMO paper status — DONE

26 pages, 31 refs, submitted-ready. No further changes to paper/cmo/.

## Active task: N-camera calibration

### Phase 1 — Refactor camera model to support N channels

Current state: many functions take `left_pixels, right_pixels` as separate args.
Goal: accept `cameras: list[CameraObs]` where each camera has its own pixels,
Zernike coefficients, and optional physical model.

Key refactors:
- `rayfield_from_observations.py`: `ZernikeFitDiagnostics` → support N cameras
- `zernike_origin_field.py`: `ZernikeRayField` → accept per-camera configs
- `physics/cmo_physical.py`: `CMOTelecentricStereoModel` → `CMOTelecentricNModel`
- API surface: `calibrate(cameras: list[CameraSetup]) -> NCalibrationResult`

### Phase 2 — Joint N-rayfield bundle adjustment

Extend the BA objective from 2 × (origin + direction) to N × (origin + direction).
Shared constraint: all cameras observe the same board poses with potentially
different relative transforms.

Constraints:
- All cameras share the board-to-world transform per frame
- Each camera has independent Zernike coefficients
- Optionally: cameras can share a rigid rig transform (fixed relative SE(3))

### Phase 3 — N-channel model selection

Extend BIC comparison to N channels:
- Each channel can have its own physical model family
- BIC sums across channels
- Shared constraints: rig geometry, baseline pairs

### Phase 4 — Notebook: Pinhole × 4 validation

Validate with a synthetic 4-camera pinhole rig:
- Each camera: pinhole with known calibration
- Rig: known SE(3) transforms between cameras
- Verify BA recovers the ground truth within noise

### Phase 5 — Notebook: N-camera Greenough simulation

Simulate a Greenough binocular with 2 additional context cameras.
Validate that the pipeline correctly identifies:
- Greenough: independent per-channel optical centres
- Context cameras: pinhole models

## Conventions (unchanged)

- `origin/develop` is the source of truth for current code. Treat `develop` as
  the real implementation branch for large refactors and substantial code
  changes. Do not use `main` / `origin/main` as the reference branch for this
  work unless JFW explicitly asks for a release/backport.
- deep-claude: fresh session + CLAUDE.md only
- Flags: --pro --dangerously-skip-permissions --allowedTools "Read,Write,Edit,Bash"
- All artefacts → JSON
- 109 existing tests must pass
- User audits CDC via ChatGPT before execution
- Watcher: persistent wait_telegram.py
- Telegram: send_message to Jeff Witz (dm) for all questions
