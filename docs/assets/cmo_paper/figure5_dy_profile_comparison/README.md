# Figure 5 (CMO paper) — d_y profile comparison

Editable source of truth for Figure 5 of `paper/cmo/manuscript.tex`
(line 270, `dy_profile_comparison.pdf`).

## Files

- `manifest.json` — path to the profile JSON and figure size.

## Upstream data

- `docs/assets/pycaso_real_data/dy_profile_data.json` — pre-computed
  `d_y(u, v)` profiles along the sensor centre column for the three
  models (Zernike measured / Telecentric CMO / Perspective CMO).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_dy_profile.py
```

Outputs:

- `paper/cmo/figures/dy_profile_comparison.pdf` — used by
  `\includegraphics` in `manuscript.tex` (line 270).
- `paper/cmo/figures/dy_profile_comparison.png` — docs/preview.
