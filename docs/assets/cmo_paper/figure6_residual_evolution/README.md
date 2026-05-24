# Figure 6 (CMO paper) — residual evolution across model stages

Editable source of truth for Figure 6 of `paper/cmo/manuscript.tex`
(line 329, `residual_evolution.pdf`).

## Files

- `manifest.json` — input checkpoint, grid size, Z planes used for the
  point residual, optimiser budget, and figure size.
- `residual_evolution_data.npz` — **cached** 31×31 direction-error maps
  for the three model stages (Perspective CMO, Telecentric CMO,
  CMO+SE(3)), plus the matching RMS values. Written by
  `examples/notebooks/generate_fig_residual_evolution.py` when it runs
  for the first time, then re-used on every subsequent run.

## Upstream data

- `docs/assets/pycaso_real_data/intermediate_state.npz` — Pycaso checkpoint
  (frames, corners, intrinsics, board poses, initial 26p vector).

## Regenerate

Two-mode generator. The slow path runs the Zernike rayfield BA, the
Telecentric CMO fit, and the CMO+SE(3) fit (~2–3 min), then writes the
`residual_evolution_data.npz` cache. The fast path reads that cache and
re-renders the figure in under a second.

```bash
# Fast (uses cached data if present):
rtk .venv/bin/python examples/notebooks/generate_fig_residual_evolution.py

# Force a full recompute (e.g. after changing the BA pipeline):
rtk .venv/bin/python examples/notebooks/generate_fig_residual_evolution.py --recompute
```

Outputs:

- `paper/cmo/figures/residual_evolution.pdf` — used by
  `\includegraphics` in `manuscript.tex` (line 329).
- `paper/cmo/figures/residual_evolution.png` — docs/preview.
