# Figure 4 (CMO paper) — 3-D sub-pupil reconstruction from the Zernike rayfield

Editable source of truth for Figure 4 of `paper/cmo/manuscript.tex`
(line 252, `subpupil_3d.pdf`).

## What this figure shows

The caption of Figure 4 explicitly states that the sub-pupils
`O_L`, `O_R` are read from the **centre pixel of the Zernike rayfield**
(57 modes, `O=0, d=2`), *not* from the CMO 26-parameter model. The
quoted descriptors in the caption (`b = 24.9 mm`, `WD = 64.7 mm`, half-
angle `θ`) match the `O=0, d=2` entry of
`docs/assets/pycaso_real_data/zernike_order_sweep.json`.

## Files

- `manifest.json` — input checkpoint, fit order (`max_order_d`),
  optimiser budget, intrinsic reference (`fx_reference_px`,
  `principal_point_px`), chief-ray length and viewing angles.
- `zernike_rayfield_canonical.npz` — **cached** rayfield coefficients
  (`left_origin_coeffs`, `left_direction_coeffs`,
  `right_origin_coeffs`, `right_direction_coeffs`, plus per-frame poses
  and the resulting descriptors). Written by the generator the first
  time it runs; re-used on every subsequent invocation.

## Upstream data

- `docs/assets/pycaso_real_data/intermediate_state.npz` — Pycaso
  checkpoint (10 stereo pairs, 165 ChArUco corners, board poses).

## Regenerate

The generator first looks for the cache. If absent (or if
`--recompute` is passed), it re-fits the constrained Zernike rayfield
(`fit_constrained_zernike_rayfield`, `max_order_d=2`, ~15 s) and
writes the cache, then renders the figure.

```bash
# Fast path (uses cache):
rtk .venv/bin/python examples/notebooks/generate_fig_subpupil_3d.py

# Force a refit:
rtk .venv/bin/python examples/notebooks/generate_fig_subpupil_3d.py --recompute
```

Outputs:

- `paper/cmo/figures/subpupil_3d.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 252).
- `paper/cmo/figures/subpupil_3d.png` — docs/preview.

## Why not the CMO 26-parameter model?

The CMO 26p sits on the `f_obj / WD` degeneracy manifold (only their
difference `wd − f_obj + telecentric_offset` is identifiable), so its
individual descriptors are not absolute. The Zernike rayfield is the
gauge-fixed observable used throughout the paper: its centre-pixel
sub-pupils are the geometric reference that the CMO 26p is then fit
to. Reading Figure 4 from the rayfield therefore matches what the paper
text claims and what the bootstrap CIs in
`validation_experiments.json` characterise.
