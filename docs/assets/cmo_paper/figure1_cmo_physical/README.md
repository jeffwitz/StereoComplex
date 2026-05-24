# Figure 1 (CMO paper) — CMO optical layout diagram

Editable source of truth for Figure 1 of `paper/cmo/manuscript.tex`
(line 66, `cmo_physical.pdf`).

## Files

- `manifest.json` — pedagogical geometry parameters
  (`f_obj_mm`, `working_distance_mm`, `b_mm`, `exaggerated` flag) and
  figure size.

## Upstream data

None — purely conceptual diagram drawn from the parameters above by
`stereocomplex.viz.figures.diagram_cmo_physical`. Change a number in
`manifest.json` to redraw the schematic with different geometry.

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_cmo_physical.py
```

Outputs:

- `paper/cmo/figures/cmo_physical.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 66).
- `paper/cmo/figures/cmo_physical.png` — docs/preview.
