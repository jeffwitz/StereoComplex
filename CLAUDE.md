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
- Current strategic goal: turn `develop` into a publishable, merge-ready
  scientific platform snapshot before starting new research features.
- Do not use `main` / `origin/main` as the reference branch for this work unless
  JFW explicitly asks for a release/backport.

### Colab branch hack — TEMPORARY

`origin/main` is stale (missing most notebooks, including
`00_getting_started.ipynb`), so Colab links to `blob/main/...` 404. Until the
next `develop` → `main` merge, **all Colab references point to `develop`**:

- badge URLs in `docs/TUTORIAL.md` (`blob/develop/...`);
- the `git clone --depth 1 --branch develop ...` line in the setup cell (cell 0)
  of every `examples/notebooks/*.ipynb`;
- `examples/notebooks/add_colab_badges.py`.

**Colab branch hack DEACTIVATED as of v0.7.0-alpha merge.**: switch those
three places back from `develop` to `main` (once `main` actually carries the
notebooks). Until then, share `blob/develop/` Colab URLs.

## CMO paper status

Submission-ready as of commit 15a5d78 (paper-readiness pass). 31 pages, 12 figures, 5 tables, Appendix A. All 18 story-level numbers reproducible (`docs/assets/cmo_paper/AUDIT.md`). Items 7 (non-central baseline on the same dataset) and 8 (external 3-D validation) are acknowledged as explicit future work in §5.2 / §5.5.

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

## Active priority: make `develop` publishable

**Status as of 2026-05-24:** `origin/develop` has become the real project
baseline.  The next cross-agent objective is to transform it from a large
research branch into a readable, defensible, publishable snapshot
(`v0.1.0-alpha` / merge-ready baseline), without opening a new algorithmic
front.

Until JFW explicitly redirects:

- Prioritise consolidation tasks over new science features.
- Do **not** start Phase 2 N-camera BA yet.  Keep the CDC below as the next
  research milestone, but freeze the current validated stereo/rayfield/CMO
  platform first.
- Do **not** merge `develop` into `main` casually.  Prepare the branch so the
  merge is boring: clear release notes, clear stable/experimental boundaries,
  green gates, and known heavy artefacts documented.
- Keep Colab links pointing at `develop` until the actual `develop` -> `main`
  merge happens; then remove the temporary branch hack documented above.

### Publishable snapshot checklist

Work these items in small commits, pushing each validated task to
`origin/develop`:

1. Create `docs/RELEASE_READINESS.md`.
   - Define the current release target (`v0.1.0-alpha` unless JFW changes it).
   - List stable APIs, advanced APIs, experimental APIs, paper-only code, and
     known non-goals.
   - State clearly that N-camera Phase 1 scaffolding exists but true Phase 2 BA
     is not implemented.
2. Classify public API stability.
   - Stable: OpenCV stereo path, Ray2D corner refinement, central stereo
     rayfield, non-central Zernike origin-field fitting, model import/export,
     reconstruction diagnostics.
   - Advanced: physical model selection, CMO physical/telecentric/warped
     models, Schur diagnostics, paper reproduction helpers.
   - Experimental: N-camera facades, multi-camera oracles/simulators, future
     Phase 2 BA, paper-specific optical BA scripts.
3. Align the first-read docs.
   - `README.md`, `docs/START_HERE.md`, `docs/PUBLIC_API.md`, and
     `docs/NOTEBOOKS.md` should tell the same user path:
     images -> OpenCV baseline -> Ray2D refinement -> central rayfield ->
     non-central rayfield -> diagnostics/reconstruction.
   - Do not put CMO/N-camera complexity in the first screen unless clearly
     marked advanced/experimental.
4. Put the real gates in one visible place.
   - Required preflight for code changes:
     `rtk .venv/bin/python -m ruff check src/`
     `rtk .venv/bin/python examples/notebooks/check_docstring_params.py`
     `rtk .venv/bin/python -m pytest`
   - Slow gate before release/merge/tag:
     `rtk .venv/bin/python -m pytest -m slow`
   - If CI exists, wire the docstring guard into CI.  If CI does not exist,
     create a documented local preflight command instead of inventing hidden
     process.
5. Audit repository weight before public release.
   - Produce a tracked-file size report (`git ls-files` based), not a raw
     working-tree `du`.
   - Identify the largest assets and classify them: keep in repo, move to
     release assets, move to Zenodo, or Git LFS candidate.
   - Do not delete paper/figure/data artefacts without an explicit decision;
     document the plan first.
6. Mark experimental surfaces honestly.
   - N-camera APIs must remain honest scaffolding until Phase 2 exists.
   - CMO functions must distinguish 19/21-parameter paraxial fits,
     telecentric 12/14/16-parameter fits, warped variants, and the paper's
     26-parameter CMO + per-arm SE(3) BA.
   - Public docs must not promise unsupported topology, parameter vector size,
     or result fields.

### Snapshot acceptance gates

A `develop` snapshot is considered publishable only when all are true:

- `git status --short --branch` is clean and aligned with `origin/develop`.
- `ruff check src/` passes.
- `check_docstring_params.py` passes over the full `src/stereocomplex` tree,
  including `api/`.
- Fast tests pass.
- Slow tests pass or any failure is explicitly documented as pre-existing and
  accepted by JFW.
- `docs/RELEASE_READINESS.md` states stable/advanced/experimental/paper-only
  boundaries.
- Heavy tracked artefacts are inventoried and their release strategy is known.

## Build, lint & test status (2026-05-24)

Authoritative quality snapshot. Update this block whenever a gate moves.

### Lint — ruff

- Enforced gate: `rtk .venv/bin/python -m ruff check src/` → **0 errors**.
- Docstring contract gate:
  `rtk .venv/bin/python examples/notebooks/check_docstring_params.py` →
  **0 errors across 88 files**.  This now scans `api/`, detects invented and
  missing parameters when a `Parameters` section exists, rejects stale phrases
  such as `"Named tuple"` / `"with x"`, and requires docstrings on public
  `Result` / `Report` dataclasses.
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

- Fast: `rtk .venv/bin/python -m pytest` → **159 passed, 39 deselected**.
- Slow: `rtk .venv/bin/python -m pytest -m slow` → **39 passed,
  159 deselected** in 382.20 s (0 failures).

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

## Docstring coverage — DONE (100 %)

**Status as of 2026-05-24:** 425/425 public functions documented.  The former
`docs/DOCSTRING_TODO.md` checklist has been completed and deleted.

**Perpetual rule:** every function *touched* in a future change must
be brought up to the standard below.  The guard script
`examples/notebooks/check_docstring_params.py` is the pre-commit
barrier — 0 alarme is non-negotiable.

A good docstring here is teaching material, not a type echo. **`physics/cmo_physical.py`
is the reference exemplar** — match the depth of its `from_parameter_vector` and
`n_parameters` docstrings (full parameter layout, indices, shapes, units,
named conventions).

- Lead with **what the function is for and why**, in scientific terms (the
  geometry / optics / estimation problem it solves) — not a restatement of the
  signature.
- A bare **one-line docstring is acceptable only for genuinely trivial
  functions** (simple accessor, property, `rotx`/`roty`/`rotz`). Any function
  with non-trivial arguments or a structured return value **requires**
  numpydoc-ish `Parameters` / `Returns` sections — each entry with its **unit**
  (mm, px, rad), its **shape** if an array, and its **physical meaning**, not
  just its type.
- For any algorithmic function, **cite the source**: paper DOI / arXiv, or the
  governing equation.
- Spell out non-obvious constraints, gauges or degeneracies (e.g. the
  `f_obj` / `telecentric_offset` degeneracy) — the things that surprise a reader.

The stopping criterion is **not** the coverage percentage — it is "a
non-Python-expert scientist can use the function from its docstring alone". A
one-line pass that reaches 100 % but leaves substantive functions shallow does
**not** satisfy this objective.

Rules:

- Every **new** public function ships with such a docstring.
- Every public function **touched** for any reason gets its docstring brought up
  to this standard in the same change.
- Priority order — the hardest-to-read scientific core first: `physics/cmo*.py`,
  `ray3d/`, `benchmarks/rayfield_from_observations.py`,
  `physics/model_selection.py`.
- **Rework needed:** `cmo.py` and `rayfields/zernike_origin_field.py` reached
  100 % count with one-line docstrings — their non-trivial functions must be
  brought to the depth required above.
- Do **not** refactor / split files for this; readability comes from the
  docstrings, not from moving code around.

There is no remaining docstring checklist file.  Future work is a maintenance
rule: any touched public function must keep or reach this depth in the same
change.

Measure progress:

```bash
rtk .venv/bin/python - <<'PY'
import ast, glob
pub = pud = prv = prd = 0
for f in glob.glob("src/**/*.py", recursive=True):
    for n in ast.walk(ast.parse(open(f).read())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(n) is not None
            if n.name.startswith("_"):
                prv += 1; prd += d
            else:
                pub += 1; pud += d
print(f"public  {pud}/{pub} = {100 * pud / pub:.1f}%")
print(f"private {prd}/{prv} = {100 * prd / prv:.1f}%")
PY
```

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

#### Phase 2.0 — Multi-camera virtual scene & view-graph (pre-requisite)

**Status: specification written, not implemented.**

Before a true N-camera BA can be validated, the codebase needs the ability to
generate realistic multi-camera observations where the view-graph is non-trivial:
some cameras see some poses, not all see all.  Stereo (2 cameras, 100 % overlap)
is a degenerate case — the real value of Phase 2 is handling 4+ cameras with
partial overlap.

**1. The view-graph problem.**

Given `C` cameras placed around a scene and `P` board poses sampled in the
scene volume, the **view-graph** is a bipartite graph `G = (Cameras, Poses, E)`
where an edge `(c, p)` exists iff camera `c` sees board pose `p`.

A pose is *visible* from a camera when:
- the board centre is in front of the camera (`Z > 0` in camera frame),
- the board projects inside the image bounds (at least 4 corners visible),
- the board normal faces the camera (angle < 90°),
- the projected ChArUco corners are not too small / too large (pixel extent in
  `[20, 500]` px).

**2. Key properties the generated graph MUST satisfy.**

- **Overlap:** every camera pair intended to share a rig constraint must have at
  least `min_shared_poses` poses visible from both.
- **Connectivity:** the camera-pose graph must be connected (otherwise some
  cameras cannot be extrinsically calibrated).
- **Balanced:** no camera sees all poses while another sees none.
- **Reproducible:** seeded RNG, fixed board/extrinsics → deterministic graph.

**3. Components to implement.**

| Component | Responsibility |
|---|---|
| `VirtualScene` | Data class holding N cameras (intrinsics + SE(3) pose), board geometry, board volume (Z range), image size. |
| `PoseSampler` | Generates random board poses inside the scene volume. Default: uniform in the AABB, random axis-angle orientation. Must accept `rng` for reproducibility. |
| `FrustumChecker` | For a given pose + camera, returns `(visible: bool, n_corners: int, pixel_extent: float)`. Computes the 4 ChArUco board corners, projects them, and checks frustum constraints (see §1 above). |
| `view_graph(scene, n_poses, min_shared, rng)` | Samples poses, builds the bipartite graph, rejects poses that don't satisfy overlap/connectivity constraints. Returns `MultiCameraCharucoObservationSet` (or a new container if needed). |
| `view_graph_summary(graph)` | Diagnostic: prints overlap matrix `C×C`, per-camera coverage histogram, connectivity check. |

**4. Acceptance tests (run on 4-camera pinhole rig, 90° spacing).**

- `test_view_graph_connected`: 20 poses → graph is 1-connected component.
- `test_view_graph_overlap`: every adjacent camera pair shares ≥ 3 poses.
- `test_view_graph_balanced`: min coverage ≥ 0.5 × max coverage.
- `test_view_graph_deterministic`: same seed → identical observations.

**Reference 8-camera 360° rig.**  Cameras placed on a circle of radius
`R = 500 mm` around the scene centre, at 45° intervals, all pointing inward.
With a 60° horizontal FOV per camera, each camera sees ~2 neighbours — the
view-graph is a ring of overlapping pairs plus potential skip-edges.

| Camera | Azimuth | Shares poses with |
|---|---|---|
| cam0 | 0° | cam1, cam2, cam7 |
| cam1 | 45° | cam0, cam2, cam3 |
| ... | ... | neighbours ±1, ±2 |
| cam7 | 315° | cam6, cam0, cam5 |

With 8 cameras × 30 poses, partial overlap, and balanced coverage, this is
the canonical test case for Phase 2.0 — non-trivial view-graph, challenging
connectivity constraints, realistic 360° inspection scenario.

**5. API sketch.**

```python
from stereocomplex.synthetic.virtual_scene import VirtualScene, view_graph

scene = VirtualScene(
    cameras={
        "cam0": ("pinhole", K_4k, T0_world),
        "cam1": ("pinhole", K_4k, T1_world),
        "cam2": ("pinhole", K_4k, T2_world),
        "cam3": ("pinhole", K_4k, T3_world),
    },
    board=CharucoBoardSpec(squares_x=7, squares_y=5, square_size_mm=24),
    z_range_mm=(200, 800),
    image_size=(4000, 3000),
)

obs = view_graph(scene, n_poses=30, min_shared=3, rng=np.random.default_rng(42))
# → MultiCameraCharucoObservationSet ready for Phase 2 BA
```

**6. Out of scope for Phase 2.0.**

- Non-pinhole cameras (plate, CMO) — accept central rayfield `.ray()` interface.
- Image rendering — use the existing `simulate_charuco_observations_from_camera_fields`.
- Realistic occlusions, lighting, depth-of-field.

**7. 3-D ground-truth object & reconstruction validation.**

The ChArUco board is only a *calibration tool*.  A multi-camera rig is
interesting because it can *measure* a 3-D scene.  The virtual bench must
therefore include a known 3-D object so that the full calibration→reconstruction
pipeline can be validated end-to-end:

| Step | What | Validation |
|---|---|---|
| 1. Scene | Place N cameras + known object in a virtual volume | Object mesh or point cloud, known to < 1 µm |
| 2. Calibrate | Observe ChArUco board through `view_graph` | Phase 2.1 BA |
| 3. Match | Dense stereo matching on the object (DIS optical flow or virtual ground-truth correspondences) | Sub-pixel accuracy |
| 4. Reconstruct | Triangulate N-view correspondences with calibrated rayfields | `reconstruct_points_*` |
| 5. Evaluate | Compare reconstructed 3-D points against ground truth | RMS, P95 error in mm |

**Minimum viable 3-D object (Phase 2.0 shipped with one).**

A textured plane with known relief (e.g., a sinusoidal grating, a speckle
pattern printed on a tilted plane, or a step wedge).  Simple enough to
generate analytically, complex enough to test reconstruction accuracy.

**Object specification.**

```python
@dataclass(frozen=True)
class VirtualObject:
    geometry: str           # "plane", "sinusoid", "step_wedge"
    params: dict            # e.g. {"amplitude_mm": 5, "wavelength_mm": 50}
    extent_mm: tuple[float, float]  # (width, height) of the object
    texture: str            # "speckle", "checkerboard", "sinusoid"
    ground_truth_mm: np.ndarray  # (M, 3) reference points with known XYZ
```

The object is fixed in the world frame.  N cameras observe it.  After
calibration, `reconstruct_points_*` is called on matched pixel pairs (or
N-tuples), and the reconstruction error is the Euclidean distance to
`ground_truth_mm`.

**Acceptance test for Phase 2.0 (no BA yet — use oracle rayfields).**

```python
scene = VirtualScene(
    cameras=four_cameras_90deg,
    object=VirtualObject(geometry="sinusoid", ...),
    board=CharucoBoardSpec(...),
)
obs = view_graph(scene, n_poses=30)
# Calibrate with oracle rayfields (perfect, no optimisation needed)
# Match pixels → reconstruct → error < 0.1 mm RMS
```

This gives Phase 2.0 a meaningful success criterion *before* any BA code is
written: « a known 3-D object is reconstructed to within 0.1 mm using oracle
rayfields on N cameras ».

**8. Digital twin validation with Blender (two-step architecture).**

For realistic texture-mapped validation objects (speckle on a cylinder, sphere,
or industrial part), a `numpy` analytical function is insufficient — UV mapping,
occlusions, and perspective projection require a real renderer.

Blender Python (`bpy`) is the recommended tool for this:

**Step A — Scene assembly (one Blender `.blend` file, scripted).**

- A **ChArUco board** as a thick textured plane (one face).
- A **3-D object** (cylinder, sphere, mesh) with a speckle texture applied
  via UV mapping.
- N **cameras** placed around the scene at known SE(3) poses, each with
  known intrinsics (focal length, sensor size).
- Both the board and the object are in the same world frame — no separate
  coordinate systems.

**Step B — Render & export (one render per camera).**

Each camera renders two outputs:

1. **RGB image** — the combined image (board + object), used for ChArUco
   detection and dense stereo matching.
2. **XYZ pass** — per-pixel 3-D coordinates in world frame (Blender's
   `Mist` + `Vector` passes, or a custom compositor node).  This is the
   ground-truth position map used to validate reconstruction accuracy.

**Integration with the view-graph pipeline.**

```python
from stereocomplex.synthetic.blender_scene import BlenderScene

scene = BlenderScene.from_blend("360_rig.blend")
# → loads N cameras (K + SE(3)) and object metadata from the .blend file

obs = view_graph(scene, n_poses=30)
# → calibrates ChArUco board observations through the blender cameras

xyz_maps = scene.load_xyz_maps()
# → (C, H, W, 3) world-frame XYZ per pixel per camera

reconstructed = reconstruct_from_calibrated_rays(obs, xyz_maps)
# → compare against Blender ground truth → RMS error in mm
```

**Blender scripting requirements (to be implemented in Phase 2.0).**

| Component | File | Responsibility |
|---|---|---|
| `blender_scene.py` | `synthetic/blender/` | Load a `.blend` file, extract camera poses (SE(3)), intrinsics (focal, sensor), board/object metadata. |
| `blender_render.py` | `synthetic/blender/` | Script called *from inside Blender* (headless): enable XYZ pass, render all cameras, export PNG + NPY. Not called from the Python library — shipped as a standalone `render_scene.py` that users run with `blender --background --python`. |
| `render_scene.py` | `scripts/` | Standalone Blender Python script: sets up compositor for XYZ pass, loops over cameras, renders, exports. |

**Out of scope for Phase 2.0.**

- Photorealistic materials, HDR lighting, depth-of-field — the goal is
  geometric validation, not aesthetic rendering.
- Mitsuba 3 integration — deferred to Phase 5 (microscope optics).

#### Phase 2.1 — Joint N-rayfield BA
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
- **Every paper figure is versioned**: a standalone Python script generates a
  vector PDF from JSON/NPZ data. The script *is* the source of truth; no
  inline generation, no PNG without a script.
- Keep backward-compatible stereo APIs unless JFW explicitly asks to break them.
- Do not silently change public API behavior.
- Commit and push each validated task to `origin/develop`.
- Avoid changes under `paper/cmo/` unless explicitly requested.
- All generated/analysis artefacts should be JSON when practical.
- Deep Claude mode: fresh session + this `CLAUDE.md` only.

### No orphan figures

**Rule.** Every figure committed under `paper/*/figures/` (or anywhere else
in this repository) must be regenerateable end-to-end from this repo alone,
with no external context, no hidden state, and no manual editing required.
A figure for which no contributor can reproduce *both* its data and its
layout is **orphaned** and forbidden.

**Why.** Reviewers and future contributors will need to update numbers,
fix typos, change RMS after a re-fit, or re-render at a new size. Hunting
for the script that produced a figure — or worse, rebuilding it from
memory — wastes hours and silently drifts the paper from the code.

**How to apply.**

1. Place the editable data — measurement `.npz`, sweep `.json`, labels,
   numbers cited in the figure — under
   `docs/assets/<paper>/figureN_<short_name>/`. Conceptual diagrams
   (no measurement data) still get a JSON of labels/numbers there; the
   layout-only code lives in the script.
2. Place the generation code in `examples/notebooks/generate_fig_<short_name>.py`.
   The script must:
   - read everything it needs from the asset folder above (no hard-coded
     RMS, no figure-specific magic numbers buried in matplotlib calls);
   - write **the format `paper/.../manuscript.tex` actually references**
     (PDF for vector figures, PNG for raster-heavy ones like dense
     point clouds), and also a PNG for docs/preview when the paper
     itself uses PDF — in the same run;
   - be runnable as `rtk .venv/bin/python examples/notebooks/generate_fig_<name>.py`
     with no extra flags.
3. Place a `README.md` in the asset folder that:
   - names the figure it produces and where it is referenced in the paper;
   - lists every editable input file;
   - gives the exact regenerate command.

**Reference example.** `docs/assets/cmo_paper/figure2_pipeline/` +
`examples/notebooks/generate_fig_pipeline.py` (Figure 2 of the CMO
paper). Match that structure for every new or revised figure.

**CMO paper figure compliance tracker.** Update this table when a figure
is brought into compliance, so the next contributor (or a fresh Claude
session) knows what is left.

| # | Figure file | Script | Asset folder | Status |
|---|---|---|---|---|
| 1 | `cmo_physical.pdf` | `generate_fig_cmo_physical.py` | `figure1_cmo_physical/` | **DONE** |
| 2 | `pipeline.pdf` | `generate_fig_pipeline.py` | `figure2_pipeline/` | **DONE** |
| 3 | `BIC_vs_order.pdf` | `generate_fig_bic_vs_order.py` | `figure3_BIC_vs_order/` | **DONE** |
| 4 | `subpupil_3d.pdf` | `generate_fig_subpupil_3d.py` | `figure4_subpupil_3d/` | **DONE** |
| 5 | `dy_profile_comparison.pdf` | `generate_fig_dy_profile.py` | `figure5_dy_profile_comparison/` | **DONE** |
| 6 | `residual_evolution.pdf` | `generate_fig_residual_evolution.py` | `figure6_residual_evolution/` | **DONE** (heavy compute cached as `residual_evolution_data.npz`; `--recompute` forces rerun) |
| 7 | `pareto_gauge_regularization.pdf` | `generate_fig_pareto.py` | `figure7_pareto_gauge_regularization/` | **DONE** |
| 8 | `schur_singular_values.pdf` | `generate_fig_schur_svd.py` | `figure8_schur_singular_values/` | **DONE** |
| 9 | `specimen_reconstruction.pdf` | `generate_fig_specimen_reconstruction.py` | `figure9_specimen_reconstruction/` | **DONE** |
| 10 | `zernike_cmo_rigid_removed.pdf` | `generate_fig_zernike_cmo_rigid_removed.py` | `figure10_zernike_cmo_rigid_removed/` | **DONE** (also fixed a pre-existing dimensional bug: Kabsch SE(3) now applied on 3-D mm points; "dZ after SE(3)" matches the manuscript's quoted 0.06 mm residual) |
| 11 | `specimen_schur_regularized.png` | `generate_fig_specimen_schur_regularized.py` | `figure11_specimen_schur_regularized/` | **DONE** |
| 12 | `bic_bars.pdf` | `generate_fig_bic_bars.py` | `figure12_bic_bars/` | **DONE** |

**All 12 figures DONE** — every paper figure now has a manifest in
`docs/assets/cmo_paper/figureN_<name>/` and a standalone generator in
`examples/notebooks/generate_fig_<name>.py` that emits the PDF the
manuscript references plus a PNG sibling for docs/preview. Heavy
computations (Figure 6) are cached as a sibling `.npz` with a
`--recompute` flag to refresh.

**Pre-flight check before adding a figure.** Before committing a new
PDF/PNG under `paper/*/figures/`, verify in this order:

- the asset folder under `docs/assets/<paper>/figureN_<name>/` exists;
- the script at `examples/notebooks/generate_fig_<name>.py` exists and
  reads only from that asset folder;
- running the script on a fresh checkout reproduces the committed figure
  byte-equivalent (modulo non-deterministic timestamps in the PDF
  metadata).

If any of those three fails, the figure is orphaned — fix the missing
piece before commit.
