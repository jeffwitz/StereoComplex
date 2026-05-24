# Roadmap — StereoComplex user-facing API

## Current state (v0.5.x)

76 fast tests, 113 total. Rayfield measurement, 6-oracle model selection,
physical CMO, direct-vs-rayfield infrastructure, optical diagrams. The
scientific core is solid. The next priority is **user-facing polish**.

## Phase 1 — OpenCV-user façade (~1 day)

**Goal:** someone who knows `cv2.calibrateCamera` can run StereoComplex
in 5 minutes and see a clear benefit.

### 1a. `compare_opencv_stereo_calibration()`

Single entry point that runs OpenCV raw + Ray2D-refined and produces a
comparative report:

```python
report = sc.compare_opencv_stereo_calibration(
    left_dir="left", right_dir="right", board=board,
)
report.print_summary()
```

### 1b. "From OpenCV to StereoComplex" page

Short doc page with an OpenCV→StereoComplex translation table:

| OpenCV | StereoComplex |
|---|---|
| `cv2.aruco.detectMarkers` | `sc.detect_charuco_corners` |
| `cv2.calibrateCamera` | `sc.fit_opencv_stereo_from_image_dirs` |
| raw corners | `method2d="raw"` |
| refined corners | `method2d="rayfield_tps_robust"` |
| `K, dist, R, T` | `result.K_left`, `result.dist_left`, … |

### 1c. Shorter function aliases

Keep the long explicit names. Add short aliases:

```python
sc.calibrate_opencv = sc.fit_opencv_stereo_from_image_dirs
sc.calibrate_central = sc.fit_stereo_central_rayfield_from_image_dirs
sc.calibrate_noncentral = sc.fit_stereo_zernike_origin_field_from_image_dirs
sc.identify_optics = sc.select_physical_model_from_rayfield
```

### 1d. Result export methods

Add to key result dataclasses:

```python
result.summary()          # human-readable one-liner
result.to_dict()          # serialisable
result.save_json(path)    # dump to file
result.to_opencv()        # → (K1, d1, K2, d2, R, T)
```

## Phase 2 — Quality gate (~0.5 day)

**Goal:** the library tells the user whether their calibration is usable
and what to do if it is not.

### 2a. `assess_calibration(result)`

```python
assessment = sc.assess_calibration(result)
# → CalibrationAssessment(status="ok", messages=[...], recommendations=[...])
```

Rules:
- Not enough detected frames → `status="failed"`
- RMS > 0.5 px → `status="warning"`
- Fewer than 4 poses → `status="warning"`
- Non-central model with too few observations for Zernike order → warning
- `status="ok"` only when all health checks pass

### 2b. `result.report.warnings` and `result.report.recommendations`

Extend the existing report dataclasses with structured
`warnings: list[str]` and `recommendations: list[str]` fields.

## Phase 3 — API surface cleanup (~0.5 day)

**Goal:** `import stereocomplex as sc` shows only what a user needs.

### 3a. Tiered public API

Keep in `stereocomplex.__init__`:
- User entry points (Tier 1)
- Result/report dataclasses (Tier 2)
- `load_` / `save_` functions
- `compare_opencv_stereo_calibration` (new)
- `assess_calibration` (new)

Move to sub-namespaces (already partially done):
- `stereocomplex.physics` — physical models, model selection
- `stereocomplex.benchmarks` — oracles, sweep scripts
- `stereocomplex.synthetic` — dataset generation
- `stereocomplex.rayfields` — Zernike models

### 3b. Usage-oriented function index

Add a table to the README:

| I want to… | Function |
|---|---|
| Improve my ChArUco corners before OpenCV | `refine_charuco_corners` |
| Compare OpenCV raw vs refined | `compare_opencv_stereo_calibration` |
| Calibrate like OpenCV but with refined corners | `fit_opencv_stereo_from_image_dirs` |
| Switch to a central ray-based model | `fit_stereo_central_rayfield_from_image_dirs` |
| Test for non-central optics | `fit_stereo_zernike_origin_field_from_image_dirs` |
| Identify CMO / plate / Brown / etc. | `select_physical_model_from_rayfield` |
| Check if my calibration is usable | `assess_calibration` |

## Phase 4 — Notebook walkthrough (~0.5 day)

**Goal:** a notebook that replaces the README quickstart and guides the
user from "I have image folders" to "my optics are identified."

### 4a. `00_getting_started.py`

```text
1. Define board
2. Run OpenCV raw
3. Run OpenCV + Ray2D
4. Compare reports
5. If residuals are structured → try non-central
6. Identify optical model
```

### 4b. Link from README

Replace the current multi-section quickstart with a single call-to-action:
"Run `examples/notebooks/00_getting_started.py` or open
`docs/FROM_OPENCV_TO_STEREOCOMPLEX.md`."

## Phase 5 — Packaging and CI polish (~0.5 day)

- `pip install stereocomplex` works with a single command
- `pytest -m "not slow"` runs in CI in < 60 s
- `python examples/reproduce_docs_results.py --fast` verifies key results
- Tag `v0.6.0` with the user-facing API improvements

## Effort estimate

| Phase | Effort |
|---|---|
| 1 — OpenCV façade | ~1 day |
| 2 — Quality gate | ~0.5 day |
| 3 — API cleanup | ~0.5 day |
| 4 — Walkthrough notebook | ~0.5 day |
| 5 — Packaging / CI | ~0.5 day |
| **Total** | **~3 days** |

## Out of scope for this roadmap

- New optical models
- Real-data validation
- Full direct-vs-rayfield symmetric comparison
- Performance optimisation
- GUI / web interface
