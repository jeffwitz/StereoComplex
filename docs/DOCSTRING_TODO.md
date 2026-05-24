# Docstring deepening — TODO for the next pass

**Status as of commit `9f0181b`.** Coverage is 96.7 % public / 23 % private.
The hallucination layer is closed (the guard script
`examples/notebooks/check_docstring_params.py` reports **1 alarm**,
which is a pre-existing case in `direct_inversion.py`). The remaining
work is **depth**: 43 public functions with ≥ 3 arguments still carry
a single-line docstring, against the standard set by CLAUDE.md and
exemplified by `physics/cmo_physical.py::from_parameter_vector` /
`physics/cmo_physical.py::ray`.

Delete this file when the table at the bottom is fully checked.

## Hard rules — non-negotiable

1. **Read the actual signature with `ast` before writing `Parameters\n----`.**
   `examples/notebooks/check_docstring_params.py` is the local pre-commit
   guard. Run it on the file you touched **before every commit**:
   ```bash
   .venv/bin/python examples/notebooks/check_docstring_params.py <file>
   ```
   If it reports any alarm on a function you just edited, you invented a
   parameter name. Fix it before committing. No exceptions.

2. **Commit titles must reflect the exact number of functions touched.**
   Use `git diff --stat` to count. `"deepen 12 docstrings"` for 2
   functions is unacceptable and creates trust debt. Honest forms:
   - `docs: deepen N docstrings — <file>, <file>` (N is the real count)
   - `docs(<area>): deepen <function_name>` (when N == 1)

3. **No copy-paste between functions.** Each docstring is read in
   isolation: it must describe the function's own signature, units, and
   physics — not a sibling's.

4. **One file per commit** for this pass, so a reviewer can audit each
   change quickly. Bundling 5 files makes per-function review painful.

## The depth standard

Match `physics/cmo_physical.py::from_parameter_vector` (39 lines,
Parameters with units/shapes, Returns, full layout explained) and
`physics/cmo_physical.py::ray` (43 lines).

### Template

```python
def fit_cmo_telecentric_model_to_rayfields(
    left_field: ZernikeRayField,
    right_field: ZernikeRayField,
    image_size: tuple[int, int],
    initial_parameters: np.ndarray,
    *,
    pixel_pitch_mm: float = 0.0055,
    z_planes: tuple[float, ...] = (50.0, 80.0),
    grid_shape: tuple[int, int] = (31, 31),
    support_pixels_left: np.ndarray | None = None,
    support_pixels_right: np.ndarray | None = None,
    support_weight: float = 1.0,
    full_grid_weight: float = 1.0,
    robust_loss: str = "huber",
    max_nfev: int = 300,
) -> CMOTelecentricFitResult:
    """Fit a telecentric CMO model to a pair of measured Zernike rayfields.

    The fit minimises a point-to-ray transverse residual on a sampling
    grid in ray space (no pixel reprojection involved). Two Z-planes are
    used to constrain the depth of field of view; the residual at each
    pixel is the perpendicular distance from a sample point on the model
    ray to the rayfield ray, summed over both Z planes and both
    channels. See §3.5 of the CMO paper for the geometric construction.

    Parameters
    ----------
    left_field, right_field : ZernikeRayField
        Measured rayfields for the left and right channels (mm world,
        rayfield gauge ``O · d = 0``).
    image_size : tuple of (int, int)
        ``(W, H)`` of the sensor in pixels.
    initial_parameters : ndarray, shape (14,)
        Initial CMO telecentric parameter vector; see
        :meth:`CMOTelecentricStereoModel.from_parameter_vector` for the
        14-entry layout (`f_obj_mm`, `working_distance_mm`, `b_mm`, …).
    pixel_pitch_mm : float, optional
        Sensor pixel pitch in millimetres. Default `0.0055` (Pycaso).
    z_planes : tuple of float
        Z values (mm) of the two reference planes at which the
        point-to-ray residual is sampled.
    grid_shape : tuple of (int, int)
        ``(n_u, n_v)`` of the sampling pixel grid used to build the
        residual.
    support_pixels_left, support_pixels_right : ndarray, shape (N, 2), optional
        Optional anchor pixels (one per channel) where an extra
        higher-weight residual term is added.
    support_weight, full_grid_weight : float
        Relative weights of the two residual blocks.
    robust_loss : str
        SciPy ``least_squares`` ``loss`` argument. ``"huber"`` is the
        default; ``"linear"`` for a pure L2 fit.
    max_nfev : int
        Optimiser budget (residual evaluations).

    Returns
    -------
    CMOTelecentricFitResult
        Dataclass carrying the optimised 14-parameter vector, the
        residual RMS in ray space (mm) and its pixel-equivalent (px),
        and the underlying :class:`scipy.optimize.OptimizeResult`.

    Notes
    -----
    The optimisation uses a fixed scale ``D_theta`` per parameter
    (millimetres for distances, radians for the convergence half-angle,
    pixels for the principal point) so the Jacobian is well-conditioned.
    Read CLAUDE.md §"Gauge sensitivity" before reading any of the
    optimised parameters as physical measurements.
    """
```

Notable elements:
- units on every numeric parameter (mm, px, rad);
- shapes on every ndarray;
- physics, not just types — what the function does, with a one-line
  cross-reference to where the construction is justified (paper section,
  equation, or another docstring);
- gauge / convention notes when relevant.

## Priority queue (43 functions)

### A — Scientific core (do first, 14 functions)

The functions a working scientist will reach for. Untouched here means
the paper's central claims have shallow API documentation.

| File | Line | Function | Args |
|---|---:|---|---:|
| `core/distortion.py` | 41 | `undistort` | 3 |
| `core/geometry.py` | 40 | `sensor_um_to_pixel` | 3 |
| `core/geometry.py` | 97 | `triangulate_midpoint` | 4 |
| `core/model_compact/zernike.py` | 99 | `eval_real_zernike` | 3 |
| `core/model_compact/zernike.py` | 111 | `pixel_to_unit_disk` | 5 |
| `core/model_compact/zernike.py` | 133 | `zernike_design_matrix` | 6 |
| `metrics/reconstruction_metrics.py` | 133 | `reconstruct_points_central_stereo` | 5 |
| `metrics/reconstruction_metrics.py` | 156 | `reconstruct_points_with_origin_fields` | 5 |
| `metrics/reconstruction_metrics.py` | 294 | `compare_3d_reconstruction_with_without_origin_field` | 3 |
| `physics/cmo_physical.py` | 1395 | `fit_cmo_telecentric_model_to_rayfields` | 9 |
| `physics/cmo_physical.py` | 2053 | `fit_cmo_warped_model_to_rayfields` | 10 |
| `physics/model_selection.py` | 357 | `fit_physical_model_to_rayfield` | 14 |
| `physics/parallel_plate_fit.py` | 110 | `pinhole_parallel_plate_ray_from_pixel` | 4 |
| `physics/parallel_plate_fit.py` | 120 | `intersect_ray_with_z_plane` | 3 |

### B — Instrumentation / benchmarks / synthetic (8 functions)

Tools the working scientist will read to understand what was simulated
or benchmarked.

| File | Line | Function | Args |
|---|---:|---|---:|
| `benchmarks/charuco_observation_simulator.py` | 346 | `simulate_charuco_observations_from_camera_fields` | 13 |
| `benchmarks/parallel_plate_origin_field.py` | 61 | `make_grid_board` | 3 |
| `benchmarks/parallel_plate_origin_field.py` | 419 | `run_parallel_plate_origin_field_benchmark` | 7 |
| `calibration/fit_zernike_origin_field.py` | 264 | `make_fields` | 4 |
| `core/pinhole_fit.py` | 80 | `project_brown_pinhole_with_rvec` | 3 |
| `synthetic/parallel_plate.py` | 122 | `pinhole_ray_from_pixel` | 3 |
| `synthetic/parallel_plate.py` | 174 | `project_point_with_parallel_plate` | 4 |
| `synthetic/parallel_plate.py` | 217 | `generate_parallel_plate_stereo_dataset` | 11 |

### C — CLI / viz / eval (21 functions)

Lower priority — these are user-facing CLIs, plotting helpers, and
detector dispatch code. Their existing one-liners are correct, just not
deep. Touch only after A and B are clean.

(See `examples/notebooks/check_docstring_params.py` output for the full
file:line list — it can be regenerated with the audit script you've
already written.)

## Stopping criterion

Per CLAUDE.md:

> The stopping criterion is **not** the coverage percentage — it is
> "a non-Python-expert scientist can use the function from its
> docstring alone".

For every function on the list above, re-read the docstring you wrote
after a 5-minute break, and ask:
- Could a physicist who has never opened this file know which units to
  pass, which shapes to provide, and what the function returns?
- Do they know which gauge / convention the result lives in?
- If the function fits an optimisation problem, do they know the
  residual being minimised and the parameterisation?

If any answer is "no", the docstring is not done yet.

## Workflow per commit

1. Pick **one file** from the priority list.
2. Open `examples/notebooks/check_docstring_params.py` in another
   terminal: `watch -n 5 '.venv/bin/python examples/notebooks/check_docstring_params.py <file>'`.
3. Read the function signature with the cursor — never from memory.
4. Write the docstring matching the template.
5. Save. The guard reports clean? Commit. Otherwise fix.
6. Commit message: `docs(<short_area>): deepen <function_name>` or
   `docs(<short_area>): deepen N functions in <file>` where N is the
   real count.
