# Figure 2 (CMO paper) — pipeline diagram

Editable source of truth for Figure 2 of `paper/cmo/manuscript.tex`
(the two-stage decomposition flowchart).

## Files

- `pipeline.json` — all editable text and numbers (titles, step labels, RMS,
  ΔBIC, input/output captions). Edit this file to change any number or label
  that appears in the figure.

## Regenerate

The generation script reads this JSON and writes both the PDF used by the
paper and a PNG used by the docs site:

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_pipeline.py
```

Outputs:

- `paper/cmo/figures/pipeline.pdf` — used by `\includegraphics` in
  `manuscript.tex` (line 73).
- `paper/cmo/figures/pipeline.png` — used by docs / preview.

## Why split?

The pipeline diagram is conceptual (no measurement data); its "data" are the
labels and the few numbers it cites. Keeping those numbers here (rather than
hard-coded in the script) means a single edit propagates to the figure and
no future contributor has to dig into matplotlib code to fix a stale RMS.

The layout (box geometry, colours, arrow routing) lives in the script — it
is the visual design of the figure.
