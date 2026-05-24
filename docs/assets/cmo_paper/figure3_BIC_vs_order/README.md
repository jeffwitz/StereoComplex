# Figure 3 (CMO paper) — BIC vs Zernike order

Editable source of truth for Figure 3 of `paper/cmo/manuscript.tex`
(line 163, `BIC_vs_order.pdf`).

## Files

- `manifest.json` — sweep input, observation-count breakdown
  (`n_obs = n_frames × n_corners × n_channels × n_coords_per_point`),
  selected model label, and figure size.

## Upstream data

- `docs/assets/pycaso_real_data/zernike_order_sweep.json` — list of
  `{O, d, p, rms}` entries (one per Zernike order pair tested).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_bic_vs_order.py
```

Outputs:

- `paper/cmo/figures/BIC_vs_order.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 163).
- `paper/cmo/figures/BIC_vs_order.png` — docs/preview.
