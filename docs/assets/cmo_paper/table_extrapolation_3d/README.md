# Table `tab:extrapolation_3d` — common-metric (3-D µm) extrapolation comparison

Produces the second extrapolation table of the CMO paper (§ Soloff comparison →
*Extrapolation to Held-Out Poses*): Zernike rayfield vs. Soloff polynomial,
3-D reconstruction RMS in micrometres on held-out extreme poses, under **two
symmetric board-pose references** (CMO-pose and Zernike-pose).

## Referenced in
`paper/cmo/manuscript.tex`, Table `tab:extrapolation_3d` (complements the
native-metric Table `tab:extrapolation`).

## Editable inputs
- `intermediate_state.npz` (optional local copy). If absent, the generator falls
  back to the shared real-data asset
  `docs/assets/pycaso_real_data/intermediate_state.npz`, which carries:
  `obj_pts (165,3)`, `left_pixels/right_pixels (10,165,2)`,
  `opt_R (10,3,3)`, `opt_t (10,3)` (CMO 26-p board poses), `FX` (reference focal).
- Train/test split and `MAX_ORDER` are constants at the top of the generator.

## Output
- `extrapolation_3d.json` — train/test RMS (µm) and degradation ratio for
  Soloff deg.2/3 and the 180-parameter Zernike rayfield, under both references.

## Regenerate
```bash
rtk .venv/bin/python examples/notebooks/generate_table_extrapolation_3d.py
```

## Protocol notes (why pose freezing)
A common 3-D metric requires a **shared** board-pose reference: the absolute
board pose is a gauge freedom neither uncalibrated model fixes alone, and an
unaligned Zernike-vs-CMO comparison inflates the Zernike error to ~500 µm
(almost entirely pose mismatch between the two bundle adjustments).
- **CMO-pose ref.**: Zernike refit with poses frozen to `opt_R/opt_t`
  (coefficients-only LS), so both models share Soloff's implicit geometry.
- **Zernike-pose ref.**: poses self-consistent with the free-fit Zernike field
  recovered by pose-only LS; Soloff refit against that geometry.
A single global rigid (Kabsch) transform, estimated on the training frames,
removes the residual camera-to-world gauge before scoring.
