# Figure 8 — Schur complement eigenvalue spectrum (26-parameter optical block)

Plots the normalised eigenvalue spectrum of the Schur complement S_theta of the
Fisher matrix on the 26-parameter CMO optical block, at the rayfield-initialised
Pycaso solution. Strong (observable) modes lie above the weak threshold
lambda_i/lambda_max = 1e-3; the trailing modes collapse after pose
marginalisation. The pose/optics coupling norm c = 0.98 is annotated.

## Referenced in
`paper/cmo/manuscript.tex`, Figure `fig:schur` (§ Schur conditioning diagnostic).

## Editable inputs
- `schur_spectrum.json` — 26 Schur eigenvalues (descending), `coupling_norm`,
  `weak_threshold`, `weak_mode_indices`, `rank_effective`, `theta_labels`.
  Extracted from the Schur BA diagnostic (see the `source` field); that bundle
  lives under the gitignored `docs/assets/pycaso_real_data/schur_ba/`, so the
  values needed by the figure are copied here to keep it reproducible from
  tracked data.
- `manifest.json` — title, pointer to the spectrum JSON, figure size.

## Regenerate
```bash
rtk .venv/bin/python examples/notebooks/generate_fig_schur_svd.py
```
