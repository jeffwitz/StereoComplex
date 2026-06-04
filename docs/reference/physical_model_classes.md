# Reference: physical model classes

The CMO optical family in `physics/cmo_physical.py`.  Every variant is
listed with its parameter‑vector layout, the exact functions that produce
and consume each, and the relationship between the fitted models and the
paper's 26‑parameter CMO + per‑arm SE(3) model.

## Canonical model hierarchy

| Model | Fitting function | `n_parameters` | What it optimises |
|---|---|---|---|
| **CMO paraxial** (19/21p) | `fit_cmo_physical_stereo_model_to_rayfields` | 19 (shared PP) or 21 (aligned PP) | Baseline, WD, f_obj, f_tube, principal point(s), two global tilts, telecentric offset, per-channel Brown-like direction distortion |
| **CMO telecentric** (10–16p) | `fit_cmo_telecentric_model_to_rayfields` | 10, 12, 14, or 16 depending on flags | Same as paraxial but f_obj → ∞ (telecentric in object space) |
| **CMO warped** | `fit_cmo_warped_model_to_rayfields` | Varies with `warp_level` | Paraxial + per‑channel 2‑D polynomial sensor‑plane warp |
| **CMO 26p + SE(3)** (paper) | `run_cmo_26p_ba` (benchmark) | 26 | 14-parameter telecentric CMO + 12 per-arm SE(3) parameters — *this is the paper's validated model* |

## Parameter vector layout (19p paraxial)

```
x = [ f_obj_mm | WD_mm | b_mm | f_tube_mm | cx_px | cy_px
    | theta_axis_tilt_rad | theta_pitch_rad | telecentric_offset_mm
    | k1_L | k2_L | p1_L | p2_L | k3_L
    | k1_R | k2_R | p1_R | p2_R | k3_R ]
```

- **f_obj_mm**: objective focal length (mm) — degenerate with telecentric_offset
- **WD_mm**: working distance from objective to specimen (mm)
- **b_mm**: stereo baseline (mm)
- **f_tube_mm**: tube lens focal length (mm)
- **cx_px, cy_px**: principal point (px)
- **theta_axis_tilt_rad, theta_pitch_rad**: small global optical-axis tilts
- **telecentric_offset_mm**: axial pupil offset, degenerate with `f_obj_mm`
- **k1/k2/p1/p2/k3**: per-channel effective Brown-like direction distortion

The 21p aligned-principal-point variant inserts
`delta_cx_diff_px, delta_cy_diff_px` after `cy_px`. The left channel receives
`-0.5 * delta`, the right channel receives `+0.5 * delta`, keeping the gauge
centred.

## Important degeneracies

**f_obj vs telecentric_offset.**  `f_obj_mm` and `telecentric_offset_mm`
enter `ray()` only via `z_pupil = WD - f_obj + telecentric_offset`.
Only their difference is identifiable.  In tests, assert
`f_obj - telecentric_offset`, never `f_obj` alone.

**21p vs 19p.**  The 21‑parameter variant keeps a shared mean
`cx_px, cy_px` and adds two relative principal-point offsets.  This is useful
for real sensors with small mechanical offsets; the shared 19p mode is the
cleaner synthetic-oracle and first-pass candidate.

## Paper model (26p)

The paper's `run_cmo_26p_ba` fits a 26‑parameter model that adds
per‑arm SE(3) parameters to the 14‑parameter telecentric CMO skeleton
(3 rotation + 3 translation per channel).  This is the validated model that
achieves 1.06 px reprojection before direct BA and 0.28 px in the
Schur-stabilised free-pose BA.

Do NOT confuse the 19/21p `fit_cmo_physical_*` functions with the paper's
26p model — they are different optimisation problems with different
outputs.
