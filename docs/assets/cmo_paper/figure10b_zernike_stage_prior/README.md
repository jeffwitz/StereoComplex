# Figure after rigid-gauge removal — Zernike stage-metric prior

This figure tests whether the depth-only specimen discrepancy diagnosed after
the rigid-gauge-removal figure can be resolved inside the Zernike rayfield BA
by transferring the translation-stage metrology into the pose model.

## Editable input

- `stage_prior_results.json` — nominal and fitted ten-frame stage ladders,
  prior widths, ray-space RMS values, and specimen relief amplitudes for the
  free, weak, strong, and near-fixed pose hypotheses.
- `near_hard_zernike_rayfield.npz` — left/right O(2)+d(2) coefficients and
  pose diagnostics for the selected near-fixed ladder. Figure 11 uses these
  exact coefficients for its Zernike relief map and depth profile.

The JSON is regenerated from the cached Pycaso calibration corners, the raw
coin images, and the versioned profilometry registration:

```bash
rtk .venv/bin/python examples/notebooks/evaluate_zernike_stage_prior.py
```

## Figure generation

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_zernike_stage_prior.py
```

Outputs:

- `paper/cmo/figures/zernike_stage_prior.pdf` — vector figure used by the paper.
- `paper/cmo/figures/zernike_stage_prior.png` — preview sibling.

The generator contains layout only; all displayed numerical values are read
from `stage_prior_results.json`.
