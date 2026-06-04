# Specimen relief: Soloff vs CMO 26p vs Zernike

Diagnostic supporting the discussion of the CMO–Zernike depth-relief discrepancy
(currently framed as `γz ≈ 0.70` in §4.8 / §5 of the CMO paper). It tests whether
the compact CMO 26p model *under-estimates* the relief or the flexible Zernike
rayfield *over-amplifies* it.

## Produces

`specimen_relief_soloff_comparison.json` — relief std/MAD and the per-point
regression against the Soloff degree-3 reference for each model.

## Inputs (all committed, this folder)

- `intermediate_state.npz` — ChArUco corners + CMO-fit poses (Soloff calibration).
- `specimen_correspondences.npz` — dense DIS correspondences on the coin.
- `specimen_reconstruction_cmo26.npz` — CMO 26p specimen reconstruction.
- `specimen_reconstruction_zernike.npz` — Zernike 57p specimen reconstruction.

## Regenerate

```bash
PYTHONPATH=src rtk .venv/bin/python \
    examples/notebooks/diagnose_specimen_relief_soloff.py
```

## Current result

Soloff is degree-stable (deg 2/3/4 relief std within ~4 %); CMO 26p matches
Soloff to ~1 % (regression slope 0.98, R² 0.98); Zernike is +43 % as a coherent
scalar gain (slope 1.42, R² 0.99 — not noise). I.e. the Zernike rayfield
over-amplifies the dense-specimen relief; the CMO 26p is faithful to Soloff.

## Caveat — not yet metrologically independent

The Soloff here is **re-fitted** on calibration `XYZ = obj_pts @ (opt_R, opt_t)`
taken from the CMO joint fit, and shares the `fx = 25600` length gauge with the
CMO model, so part of the Soloff–CMO agreement is inherited. The decisive,
citable reference is the **native profilometry-validated Pycaso Soloff** applied
to the same specimen. Plug it in to replace the re-fit before using this result
to change the manuscript narrative.
