# Reference: physical model classes

The CMO optical family in `physics/cmo_physical.py`.  Every variant is
listed with its parameter‑vector layout, the exact functions that produce
and consume each, and the relationship between the fitted models and the
paper's 26‑parameter CMO + per‑arm SE(3) model.

## Canonical model hierarchy

| Model | Fitting function | `n_parameters` | What it optimises |
|---|---|---|---|
| **CMO paraxial** (19/21p) | `fit_cmo_physical_stereo_model_to_rayfields` | 19 (shared PP) or 21 (aligned PP) | Baseline, WD, f_obj, f_tube, per‑channel sub‑pupil positions, per‑arm SE(3) |
| **CMO telecentric** (10–16p) | `fit_cmo_telecentric_model_to_rayfields` | 10, 12, 14, or 16 depending on flags | Same as paraxial but f_obj → ∞ (telecentric in object space) |
| **CMO warped** | `fit_cmo_warped_model_to_rayfields` | Varies with `warp_level` | Paraxial + per‑channel 2‑D polynomial sensor‑plane warp |
| **CMO 26p + SE(3)** (paper) | `run_cmo_26p_ba` (benchmark) | 26 | 19 paraxial + per‑arm SE(3) parameters — *this is the paper's validated model* |

## Parameter vector layout (19p paraxial)

```
x = [  f_tube_mm | b_mm | WD_mm | f_obj_mm | cx_px | cy_px
     | uL_z_pupil | uL_x_pupil | vL_x_pupil | uL_y_pupil | vL_y_pupil
     | uR_z_pupil | uR_x_pupil | vR_x_pupil | uR_y_pupil | vR_y_pupil
     | sL_x | sL_y | gamma_L_deg ]  ← per‑arm SE(3) (L only; R is
                                      slaved by stereo constraint)
```

- **f_tube_mm**: tube lens focal length (mm)
- **b_mm**: stereo baseline (mm)
- **WD_mm**: working distance from objective to specimen (mm)
- **f_obj_mm**: objective focal length (mm) — degenerate with telecentric_offset
- **cx_px, cy_px**: principal point (px)
- **uL/uR_*_pupil**: per‑channel sub‑pupil positions (mm)
- **sL_x, sL_y, gamma_L_deg**: per‑arm SE(3) alignment for left channel

## Important degeneracies

**f_obj vs telecentric_offset.**  `f_obj_mm` and `telecentric_offset_mm`
enter `ray()` only via `z_pupil = WD - f_obj + telecentric_offset`.
Only their difference is identifiable.  In tests, assert
`f_obj - telecentric_offset`, never `f_obj` alone.

**21p vs 19p.**  The 21‑parameter variant has `cx_left_px, cx_right_px`
instead of a single `cx_px`.  The difference is typically < 2 px on the
Pycaso specimen.

## Paper model (26p)

The paper's `run_cmo_26p_ba` fits a 26‑parameter model that adds
per‑arm SE(3) parameters to each channel (3 rotation + 3 translation per
channel) rather than absorbing the arm alignment into the sub‑pupil
positions.  This is the validated model that achieves 1.06 px reprojection.

Do NOT confuse the 19/21p `fit_cmo_physical_*` functions with the paper's
26p model — they are different optimisation problems with different
outputs.
