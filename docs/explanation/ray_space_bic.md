# Ray‑space BIC

This is an explanation page.  It does not contain commands you need to run.

StereoComplex performs model selection **in ray space**, not in pixel
space.  This is a deliberate choice:

- **Pixel‑space BIC** compares models by their ability to reproduce
  observed corner coordinates.  It conflates the 2‑D corner detector
  quality with the 3‑D camera model quality.
- **Ray‑space BIC** compares models by their ability to reproduce
  MEASURED RAYS (from the Zernike rayfield).  The rayfield is already
  fitted to the observations; model selection then asks: "which physical
  model best reproduces this measured rayfield?"

The BIC is computed as:

    BIC = n * ln(RMS²) + k * ln(n)

where `n` is the number of ray samples (pixels × z‑planes), `k` is the
number of model parameters, and RMS is the ray‑to‑ray distance in mm.

The model with the LOWEST BIC is preferred.  A difference of > 2 is
considered significant; > 6 is strong evidence.
