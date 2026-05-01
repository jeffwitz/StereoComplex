# Identify My Optics

This tutorial shows how to use `select_physical_model_from_rayfield` to
determine which physical optical model best explains a measured Zernike
rayfield.

## Mental model

Think of the workflow in two stages:

1. **Measure** — `fit_stereo_zernike_origin_field_from_image_dirs` fits a
   compact Zernike model to your calibration images. The result is a measured
   rayfield: a function `(u, v) → (O(u,v), d(u,v))` that describes how each
   pixel maps to a 3D line.

2. **Interpret** — `select_physical_model_from_rayfield` takes that rayfield
   as a geometric object and asks: *which physical hypothesis most compactly
   explains it?* Each candidate model (pinhole, Brown-Conrady, inclined plate,
   or a custom model you supply) is fitted in ray space, and the winner is
   chosen by the Bayesian Information Criterion (BIC).

The two stages are independent. If the Zernike fit is good, the physical
interpretation is reliable. If the Zernike fit is noisy, the physical
interpretation will reflect that noise.

## Minimum example

```python
from pathlib import Path
import stereocomplex as sc

board = sc.CharucoBoardSpec(
    squares_x=9,
    squares_y=6,
    square_size_mm=20.0,
    marker_size_mm=15.0,
    aruco_dictionary="DICT_4X4_50",
)

# Step 1: measure the rayfield from image folders.
fit = sc.fit_stereo_zernike_origin_field_from_image_dirs(
    left_dir=Path("my_data/left"),
    right_dir=Path("my_data/right"),
    board=board,
    max_order=4,
    method2d="rayfield_tps_robust",
)

# Step 2: identify the physical optics.
report = sc.select_physical_model_from_rayfield(
    target_field=fit.left_field,
    candidate_specs=None,          # default: pinhole, Brown-Conrady, inclined plate
    K=fit.left_field.K,
    image_size=fit.left_field.config.image_size,
)

print(report.best_by_bic)
for row in report.rows():
    print(row)
```

## How to read the report

`report.rows()` returns a list of dicts with these keys:

| Key | Description |
| --- | --- |
| `model` | Model name (`central_pinhole`, `central_brown_conrady`, `pinhole_parallel_plate`). |
| `parameters` | Number of free parameters fitted. |
| `rms_mm` | Overall RMS distance between model and target rayfield at two reference planes, in mm. |
| `support_rms_mm` | RMS on the observed pixel support only. |
| `full_grid_rms_mm` | RMS on a dense full-image grid (extrapolation quality). |
| `bic` | Bayesian Information Criterion (lower is better). |
| `aic` | Akaike Information Criterion (lower is better). |
| `selected_bic` | `True` for the BIC winner. |

`report.best_by_bic` is a string name; `report.best_by_rms` and
`report.best_by_aic` are available for alternative criteria.

**Interpretation guide:**

- A large gap in `rms_mm` between first and second candidate is a strong signal.
- `support_rms_mm` measures fit quality on observed pixels; `full_grid_rms_mm`
  measures extrapolation quality, which is more sensitive to model
  misspecification at the image edges.
- If BIC and AIC disagree, prefer BIC for compact models on typical calibration
  grids (BIC penalises free parameters more strongly).

## Example result on the inclined-plate oracle

On the synthetic inclined-plate benchmark, the output looks like:

| model | parameters | rms_mm | support_rms_mm | full_grid_rms_mm | bic | selected_bic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| central_pinhole | 0 | 2.770 | 2.770 | 3.690 | −3 290 | False |
| central_brown_conrady | 5 | 2.020 | 2.020 | 2.690 | −11 323 | False |
| pinhole_parallel_plate | 3 | 0.003 | 0.003 | 0.044 | −148 648 | **True** |

The plate model wins by a factor of ~13× in BIC and by ~700× in RMS relative
to Brown-Conrady. This is expected: the oracle is an inclined plate, and the
plate model has the right structural degrees of freedom.

## Adding a custom candidate model

Implement the `PhysicalRayFieldModel` protocol — a class with a `ray(u, v)`
method and optionally `from_parameter_vector`, `parameter_dict`,
`n_parameters`. Then wrap it in a `PhysicalModelSpec` and pass a list as
`candidate_specs`:

```python
import numpy as np
from stereocomplex.physics import PhysicalModelSpec

class MyTiltModel:
    """A custom physical candidate: single-axis tilt only."""
    name = "my_tilt"

    def __init__(self, K, tilt_deg=0.0):
        self.K = K
        self.tilt_deg = tilt_deg

    @classmethod
    def from_parameter_vector(cls, x, K, **kwargs):
        return cls(K=K, tilt_deg=float(x[0]))

    def parameter_dict(self):
        return {"tilt_deg": float(self.tilt_deg)}

    def ray(self, u, v):
        # ... your ray computation here
        ...

my_spec = PhysicalModelSpec(
    name="my_tilt",
    model_class=MyTiltModel,
    initial_parameters=np.zeros(1),
    bounds=(np.array([-45.0]), np.array([45.0])),
)

report = sc.select_physical_model_from_rayfield(
    target_field=fit.left_field,
    candidate_specs=[my_spec],     # only test your model
    K=fit.left_field.K,
    image_size=fit.left_field.config.image_size,
)
```

Or extend the default set:

```python
from stereocomplex.advanced import default_physical_model_specs

specs = default_physical_model_specs() + [my_spec]
report = sc.select_physical_model_from_rayfield(
    target_field=fit.left_field,
    candidate_specs=specs,
    K=K,
    image_size=image_size,
)
```

## Pitfalls

**BIC counts residual scalars, not independent pixel observations.**
The residual vector has shape `(N_pixels × 6,)` — 3D errors at two reference
planes. The BIC formula uses `n = N_pixels × 6`. This is consistent across
all candidates, so relative ordering is preserved, but the absolute BIC value
is not the textbook version (which would use `n = N_pixels`).

**Support vs. extrapolation.**
By default, `full_grid_weight=0.25` adds a dense evaluation grid weighted at
25 % of the support. The `full_grid_rms_mm` column measures how well each
model extrapolates beyond the calibration grid; models with wrong structure
(e.g., a central model on a non-central rig) often have larger extrapolation
residuals than support residuals.

**Convergence of the Brown-Conrady fitter.**
The undistortion loop uses 10 fixed-point iterations. For strongly distorted
optics (|k1| > 0.5), convergence may be slow. Increase `max_nfev` or tighten
bounds in the `PhysicalModelSpec` if needed.

**A low RMS does not mean the model is "correct".**
A plate model with 3 parameters can achieve near-zero residuals on a plate
oracle because it has the right structure. But on a real rig with a complex
non-central signature, all models may show residuals, and the BIC winner is
simply the most compact viable explanation — not a ground-truth identification.
