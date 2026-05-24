# Figure 8 (CMO paper) — Schur singular values

Editable source of truth for Figure 8 of `paper/cmo/manuscript.tex`
(line 436, `schur_singular_values.pdf`).

## Files

- `manifest.json` — path to the diagnostic JSON and figure parameters
  (annotated-mode count, figure size).

## Upstream data (consumed by the figure)

- `docs/assets/pycaso_real_data/zernike_conditioning_diagnostic.json` —
  Phase-1 design-matrix SVD (regular grid 41×41, orders 2 and 4) and
  Phase-2 modal decomposition. Produced by
  `examples/notebooks/diagnose_zernike_conditioning.py`.

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_schur_svd.py
```

Outputs:

- `paper/cmo/figures/schur_singular_values.pdf` — used by
  `\includegraphics` in `manuscript.tex` (line 436).
- `paper/cmo/figures/schur_singular_values.png` — docs/preview
  counterpart.

## Re-running upstream

If the diagnostic data needs to be refreshed:

```bash
rtk .venv/bin/python examples/notebooks/diagnose_zernike_conditioning.py
```

The figure regeneration above is then a single command.
