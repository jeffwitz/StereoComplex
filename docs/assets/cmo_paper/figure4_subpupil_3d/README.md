# Figure 4 (CMO paper) — 3-D sub-pupil reconstruction

Editable source of truth for Figure 4 of `paper/cmo/manuscript.tex`
(line 252, `subpupil_3d.pdf`).

## Files

- `manifest.json` — input checkpoint, sensor pixel pitch, chief-ray
  length used for the 3-D arrows, the 3-D viewing angles, and figure size.

## Upstream data

- `docs/assets/pycaso_real_data/intermediate_state.npz` — Pycaso checkpoint
  (provides `x_26p`, `image_size`, `opt_t` for the working distance estimate).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_subpupil_3d.py
```

Outputs:

- `paper/cmo/figures/subpupil_3d.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 252).
- `paper/cmo/figures/subpupil_3d.png` — docs/preview.
