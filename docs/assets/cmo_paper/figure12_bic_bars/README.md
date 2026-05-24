# Figure 12 (CMO paper) — BIC bar chart

Editable source of truth for Figure 12 of `paper/cmo/manuscript.tex`
(line 724, `bic_bars.pdf`).

## Files

- `manifest.json` — BIC JSON path, the 1.5 px reprojection guard,
  display-name remapping for each model, and the operational-panel
  model list with colours.

## Upstream data

- `docs/assets/pycaso_real_data/bic_model_selection.json` — output of
  the ray-space BIC selection (`candidates[]` with
  `{model, bic_ray, parameters}` + `operational_bic.model_26p`).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_bic_bars.py
```

Outputs:

- `paper/cmo/figures/bic_bars.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 724).
- `paper/cmo/figures/bic_bars.png` — docs/preview.
