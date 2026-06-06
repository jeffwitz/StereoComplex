# Figure — Per-channel SE(3) arm-alignment schematic

Conceptual diagram (no measurement data) illustrating the per-channel SE(3) arm
misalignment of the CMO model: the ideal telecentric skeleton (two off-axis
sub-pupils on the baseline $b$, chief rays converging at the object point $O$ at
working distance $WD$ with full convergence angle $\Theta$) versus the *actual*
orientation of each optical arm after the fitted rigid SE(3) rotation
($\sim2.5^\circ$ left, $\sim3.7^\circ$ right). Visualises the small misalignment
the reviewer asked to see.

- **Produced figure:** `se3_arm_alignment.pdf` (vector, manuscript) +
  `se3_arm_alignment.png` (docs/preview).
- **Referenced in the manuscript:** §3.5.6 "Per-Channel SE(3) Arm Alignment".
- **Generator:** `examples/notebooks/generate_fig_se3_arm_alignment.py`.

## Editable inputs (this folder)

- `se3_schematic.json` — label/number manifest: baseline (24.9 mm), working
  distance (64.7 mm), convergence angle (22.6°), per-arm rotation magnitudes
  (2.5° / 3.7°), the fitted per-axis rotation vectors (deg, dominant axis $x$),
  and the drawing exaggeration factor (×4).

The numbers are the CMO geometry descriptors quoted in the manuscript and the
fitted per-channel SE(3) rotations from
`docs/assets/pycaso_real_data/intermediate_state.npz` (`x_26p`, indices 14–25).

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_se3_arm_alignment.py
```

The script is layout-only: it reads every number from `se3_schematic.json` and
writes the PDF + PNG. The arm tilt is drawn exaggerated (×4) for visibility; the
true magnitudes are annotated and stored in the manifest.
