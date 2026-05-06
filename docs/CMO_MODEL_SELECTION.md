# CMO rayfield measurement and optical model selection

This page documents the CMO model-selection experiment introduced in notebook
06:

- notebook: `examples/notebooks/06_cmo_model_selection.ipynb`;
- companion script: `examples/notebooks/06_cmo_model_selection.py`;
- generated assets: `docs/assets/cmo_model_selection/`.

The purpose is to demonstrate a ray-space workflow:

```text
ChArUco CMO scene
→ generic Zernike rayfield measurement
→ physical candidate fitting
→ AIC/BIC model selection
```

The central idea is:

```text
Measure the rayfield first; explain the optics second.
```

In French:

```text
Les points 2D servent à mesurer le champ de rayons ; le champ de rayons sert
ensuite à identifier l'optique.
```

## Why this is a separate notebook

Notebook 04 demonstrates the non-central Zernike pipeline on an inclined
parallel-plate oracle. It also introduces ray-space physical model fitting.

Notebook 06 separates the **model-selection** story from the parallel-plate
story. It uses a CMO-like stereo model and asks a different question:

> Given a measured generic rayfield, which physical optical hypothesis explains
> it best?

The measured object is a generic Zernike rayfield

```{math}
\widehat{\mathcal R}_Z(u,v)=
\left(\widehat O(u,v),\widehat d(u,v)\right).
```

The physical candidates are then fitted to this rayfield in ray space. The
Zernike field is not treated as a competing physical model; it is the measured
geometric object.

## ChArUco target policy

The rendered calibration target is a **ChArUco** board. This is deliberate.

Plain checkerboards are acceptable for purely geometric rayfield unit tests, but
they are not the right target for image-based calibration workflows. ChArUco
corners carry IDs, so the target remains identifiable when only part of the
board is visible, when the board is blurred, or when the image contains
vignetting and contrast variation.

In the CMO notebook, the board is kept fully in the image to make the visual
example clean, but the generated target is still ChArUco so the same generator
can be reused later for detection-based experiments.

```{figure} assets/cmo_model_selection/cmo_rendered_pair.png
:alt: Rendered CMO left and right ChArUco images
:width: 95%

Rendered CMO ChArUco pair. The renderer uses the physics model directly:
pixel → CMO ray → plane intersection → ChArUco texture sample.
```

## Shared physics: generation and fitting use the same model

The CMO implementation lives in `stereocomplex.physics`, not in a separate
simulation-only namespace. This is important because the image generator and the
physical fitting candidate must not diverge.

The generator uses:

- `CMOStereoSpec` for the stereo CMO layout;
- `CMOChannelSpec` for each effective channel;
- `CMOChannelRayField` for pixel-to-line ray generation;
- `BrownConrady` for per-channel geometric distortion;
- `PolynomialRayAberration` for common and differential angular aberrations;
- `SensorWarp` and `Vignetting` for image-side effects.

The CMO model-selection candidate, `CMOPolynomialChannelModel`, reuses the same
Brown-Conrady and polynomial ray-aberration primitives. It fits one effective
channel at a time. The mathematical definition of this candidate is centralized
in [Identify My Optics](IDENTIFY_MY_OPTICS.md#candidate-3-cmo-polynomial-stereo-channels).

Because the notebook fits channels independently, the fitted aberration
coefficients represent an effective channel aberration: common CMO aberration
plus left/right differential aberration.

## Generic Zernike rayfield measurement

The notebook first measures the CMO rayfield using a generic Zernike model with
both ray origins and ray directions:

```{math}
\mathcal R_Z(u,v)=
\left(O_Z(u,v),d_Z(u,v)\right).
```

The origin field is represented in the transverse gauge

```{math}
O_Z(u,v)\cdot d_Z(u,v)=0,
```

and the direction field is a smooth Zernike perturbation around the pinhole
direction. This step deliberately does **not** use the CMO parameters. Its role
is to create a measured rayfield that physical models can explain afterwards.

On the generated CMO case, the measured Zernike `O,d` fields approximate the
CMO oracle with:

| Channel | Zernike coefficients | Rayfield RMS | Median | P95 |
|---|---:|---:|---:|---:|
| left | 60 | 0.115 mm | 0.094 mm | 0.171 mm |
| right | 60 | 0.102 mm | 0.085 mm | 0.153 mm |

These numbers should be read as the fidelity of the generic rayfield
measurement before any physical interpretation is applied.

## Ray-space candidate models

The physical candidates fitted in notebook 06 are:

| Candidate | Parameters | What it can represent |
|---|---:|---|
| central pinhole | 0 | one camera center and pinhole directions |
| central Brown-Conrady | 5 | central rays with radial/tangential direction bending |
| pinhole + inclined parallel plate | 3 | a non-central parallel-plate line family |
| CMO polynomial channel | 17 | effective sub-pupil origin, Brown-Conrady, and polynomial ray aberration |

All candidates are fitted to the measured Zernike rayfield using the shared
two-plane ray residual defined in
[Identify My Optics](IDENTIFY_MY_OPTICS.md#ray-space-comparison). The residual
compares line geometry, not raw origins; this avoids gauge artifacts when two
equivalent parameterizations of the same 3D line choose different points on
that line.

## Selection scores

For each physical candidate, StereoComplex reports RMS rayfield error and
information criteria. The exact AIC/BIC convention is documented in
[Identify My Optics](IDENTIFY_MY_OPTICS.md#information-criteria). In short, the
likelihood term uses the scalar two-plane residual components, while the BIC
complexity penalty uses the number of sampled pixels as the independent
observation count.

## Results

The CMO candidate is selected by BIC on both channels.

| Channel | Candidate | Parameters | RMS | Support RMS | Full-grid RMS | BIC | Selected |
|---|---|---:|---:|---:|---:|---:|---|
| left | central pinhole | 0 | 10.234 mm | 10.234 mm | 9.921 mm | 3077.7 | no |
| left | central Brown-Conrady | 5 | 4.709 mm | 4.709 mm | 4.714 mm | 864.8 | no |
| left | pinhole + plate | 3 | 10.369 mm | 10.369 mm | 10.010 mm | 3130.7 | no |
| left | CMO polynomial channel | 17 | 0.042 mm | 0.042 mm | 0.040 mm | -12785.9 | yes |
| right | central pinhole | 0 | 9.430 mm | 9.430 mm | 9.164 mm | 2841.3 | no |
| right | central Brown-Conrady | 5 | 4.623 mm | 4.623 mm | 4.630 mm | 811.3 | no |
| right | pinhole + plate | 3 | 9.459 mm | 9.459 mm | 9.137 mm | 2864.2 | no |
| right | CMO polynomial channel | 17 | 0.038 mm | 0.038 mm | 0.036 mm | -13105.7 | yes |

```{figure} assets/cmo_model_selection/cmo_model_selection_rms.png
:alt: CMO model selection rayfield RMS
:width: 95%

Ray-space RMS after fitting each physical candidate to the measured Zernike
rayfields. The CMO candidate explains the measured field much better than the
central and parallel-plate alternatives.
```

```{figure} assets/cmo_model_selection/cmo_model_selection_bic.png
:alt: CMO model selection BIC
:width: 95%

Delta-BIC relative to the best candidate. The CMO candidate remains selected
even after the BIC penalty for its larger parameter count.
```

## Parameter recovery

The effective CMO origins are recovered accurately:

| Channel | True origin `(x,y)` | Fitted origin `(x,y)` |
|---|---:|---:|
| left | `(-4.000, 0.150)` mm | `(-3.9999, 0.1510)` mm |
| right | `(4.000, -0.100)` mm | `(4.0001, -0.1007)` mm |

The Brown-Conrady coefficients are recovered as an effective channel model, not
as a unique physical decomposition. In this setup the polynomial ray aberration
and Brown terms are partially correlated, so individual Brown coefficients
should not be over-interpreted. The key physically stable result is the
ray-space score and the effective origin recovery.

## Scientific interpretation

The experiment is not meant to prove that every CMO can be identified from one
synthetic image. It validates a more specific methodological point:

1. a generic Zernike `O,d` rayfield can act as a measured geometric object;
2. physical candidates can be fitted afterwards in ray space;
3. central Brown-Conrady improves over a pinhole model but cannot explain the
   non-central effective sub-pupil origin;
4. an inclined parallel plate is non-central but is the wrong physical family
   for this CMO rayfield;
5. the CMO candidate wins despite a larger parameter count because it removes
   the structured ray-space residual.

This supports the intended StereoComplex workflow:

```text
measure a generic rayfield first,
then compare compact physical explanations.
```

## Current limitations

This page describes a controlled synthetic benchmark. The current notebook:

- fits left and right channels independently;
- uses a compact polynomial aberration model, not a full optical design;
- measures the generic rayfield directly from oracle rays, not yet from
  detected image points;
- uses one ChArUco rendered pose only for the visual generator check;
- does not estimate uncertainty or perform train/test splits.

The next natural step is to connect this page to the full image-based pipeline:
render multiple ChArUco poses, detect them, fit the Zernike rayfield by BA, and
then rerun the same physical model-selection step on the measured rayfield.
