# Figure 10 (CMO paper) — Zernike vs CMO rigid-gauge removal

Editable source of truth for Figure 10 of `paper/cmo/manuscript.tex`
(line 668, `zernike_cmo_rigid_removed.pdf`).

## Files

- `manifest.json` — paths to the four input data files plus crop and
  percentile parameters.

## Upstream data (consumed by the figure)

- `docs/assets/pycaso_real_data/specimen_correspondences.npz` —
  pixel correspondences and image size.
- `docs/assets/pycaso_real_data/specimen_reconstruction_cmo26.npz` —
  CMO 26p Z map (per pixel).
- `docs/assets/pycaso_real_data/specimen_reconstruction_zernike.npz` —
  Zernike rayfield Z map (per pixel).
- `docs/assets/pycaso_real_data/zernike_cmo_rigid_comparison.json` —
  Kabsch SE(3) alignment of the Zernike onto the CMO surface, plus
  affine-plane R² metrics. Produced by
  `examples/notebooks/11_compare_zernike_cmo_rigid_removed.py`.

## Regenerate

```bash
rtk .venv/bin/python examples/notebooks/generate_fig_zernike_cmo_rigid_removed.py
```

Outputs:

- `paper/cmo/figures/zernike_cmo_rigid_removed.pdf` — used by
  `\includegraphics` in `manuscript.tex` (line 668).
- `paper/cmo/figures/zernike_cmo_rigid_removed.png` — docs/preview
  counterpart.

## Re-running upstream

If a re-fit changes the underlying point clouds or the rigid alignment:

```bash
rtk .venv/bin/python examples/notebooks/10_pycaso_specimen_reconstruction.py
rtk .venv/bin/python examples/notebooks/11_compare_zernike_cmo_rigid_removed.py
```

The figure regeneration above is then a single command.

## Bug fixed in this generator

The previous helper (`convert_figures_to_pdf.py`) applied the Kabsch
SE(3) — fit on 3-D world points in millimetres — to **pixel-coordinate
triples `(j, i, Z)`** instead of the 3-D `(X, Y, Z)` points. That mixed
pixel indices with millimetres in the rotation, producing a `dZ raw`
panel reporting `med ≈ 367 mm` (several times the working distance) —
an obvious dimensional inconsistency.

This generator applies the SE(3) on the actual 3-D mm points stored in
the `.npz` files. The "dZ after SE(3)" panel now reports
`med ≈ 0.053 mm` and the histogram shows `3.82 → 0.0600 mm`, which
matches the value the manuscript already cites
("median 3D residual drops from 3.8 mm to 0.06 mm").

