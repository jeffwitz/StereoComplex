# Validation limits

This is an explanation page.  It does not contain commands you need to run.

## What we have validated

- Stereo (2 cameras): OpenCV path, central rayfield, non‑central Zernike
  origin field → synthetic oracles + 1 real CMO specimen.
- Model selection: BIC correctly identifies the generating model family
  on all 6 oracles.
- Reconstruction: central vs non‑central improvement verified on
  parallel‑plate oracle (0.05 → 0.001 mm RMS).

## What we have NOT validated

- **N‑camera calibration (> 2 cameras).**  The scaffolding exists; the BA
  does not.
- **External 3‑D validation.**  No independent metrology (CMM, LIDAR,
  interferometer) has been used to confirm reconstructed 3‑D coordinates.
- **Real‑world non‑central cameras beyond the Pycaso CMO specimen.**
  One real case study is not a statistical validation.
- **Temporal stability / repeatability.**  No multi‑session repeat tests.
- **Greenough / non‑CMO microscopes.**

## What the paper acknowledges

Item 7 (non‑central baseline on the same dataset) and item 8 (external
3‑D validation) are explicitly listed as future work in §5.2 / §5.5 of
the CMO paper.
