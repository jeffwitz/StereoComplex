# Figure 7 (CMO paper) — Pareto gauge regularization

Editable source of truth for Figure 7 of `paper/cmo/manuscript.tex`
(line 382, `pareto_gauge_regularization.pdf`).

## Files

- `manifest.json` — path to the gauge-sweep JSON plus the constrained
  reference RMS used as a vertical guide line.

## Upstream data

- `docs/assets/pycaso_real_data/zernike_gauge_regularization_sweep.json`
  — outputs of the Zernike gauge regularization sweep
  (`sweep[i]` = `{sigma_z0, sigma_z1, ray_rms_mm, drift_z0_deg,
  baseline_mm, convergence_angle_deg, ...}`).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_pareto.py
```

Outputs:

- `paper/cmo/figures/pareto_gauge_regularization.pdf` — used by
  `\includegraphics` in `manuscript.tex` (line 382).
- `paper/cmo/figures/pareto_gauge_regularization.png` — docs/preview.

## Re-running upstream

If the sweep needs to be re-run:

```bash
rtk .venv/bin/python examples/notebooks/run_sweep_complete.py
```
