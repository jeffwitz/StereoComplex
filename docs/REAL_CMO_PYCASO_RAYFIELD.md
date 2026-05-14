# Real CMO microscope calibration on Pycaso data

## A rayfield-based case study with legacy ChArUco images

> **First real-data validation of the StereoComplex non-central pipeline.**
> This page is the stable, readable reference.  The executable protocol is
> [Notebook 09](../examples/notebooks/09_pycaso_real_data.py).

## Executive summary

This case study demonstrates a complete StereoComplex workflow on **real CMO
stereo microscope calibration images** from the
[Pycaso](https://github.com/LaboratoireMecaniqueLille/Pycaso) open-source project.

The pipeline detects legacy ChArUco corners, completes missing points with a
Hessian-based subpixel procedure, applies a **double Ray2D TPS denoising
pass**, and fits a constrained Zernike rayfield.  From this measured
rayfield we **read physical descriptors directly** (baseline, working
distance, convergence angle), **diagnose structural mismatches** with
idealised models, and **iteratively build a compact physical model** that
captures the dominant CMO geometry.

The final model — a quasi-telecentric CMO skeleton with per-channel SE(3)
arm alignment — achieves **1.13 px reprojection** on this dataset
(P50 = 0.94 px), compared to > 300 px for a standard OpenCV stereo
calibration under the tested configuration.

## Key result: the rayfield as a diagnostic and modelling instrument

This study's central contribution is methodological, not just a performance
number.  The Zernike rayfield closes a **Ray2D → Ray3D → diagnose → fix
→ verify** feedback loop that no classical reprojection-based calibration
can provide:

```text
┌──────────────────────────────────────────────────────────────┐
│  Ray2D: corner detection + Hessian completion + double TPS   │
│                         ↓                                    │
│  Ray3D: Zernike rayfield O(0)+d(2) — the experimental oracle │
│                         ↓                                    │
│  Read descriptors: b, WD, f_obj, θ directly from (O, d)     │
│                         ↓                                    │
│  Propose physical model → residual vs Zernike                │
│                         ↓                                    │
│  Residual is Z0 (global)? → missing global DOF               │
│  Residual is spatial?     → missing field structure          │
│                         ↓                                    │
│  Add DOF → refit → evaluate → iterate                        │
│                         ↓                                    │
│  Final model: 1.13 px reprojection (P50 = 0.94 px)          │
└──────────────────────────────────────────────────────────────┘
```

The 2‑D reprojection error is **blind** to the pose/rayfield gauge — a
fit can absorb corner noise into rayfield distortions without increasing
pixel RMS.  Only the rayfield reveals the problem, and only the rayfield
tells you *which* degree of freedom is missing.

## The dataset

| Property | Value |
|---|---|
| Sensor | 2048 × 2048 px |
| Board | Legacy ChArUco, 16 × 12 squares, 0.3 mm |
| Dictionary | DICT_6X6_250, `setLegacyPattern(True)` |
| Frames | 10 stereo pairs |
| Z range | 2.65 – 3.35 mm (Δ = 0.70 mm) |

The dataset is **not vendored** in the StereoComplex repository.  Clone
[Pycaso](https://github.com/LaboratoireMecaniqueLille/Pycaso) at
`examples/pycaso_data`.

## Pipeline

```text
ChArUco legacy detection (DICT_6X6_250, setLegacyPattern)
       ↓
Hessian corner completion (|det H| + Otsu + barycentre)  →  165/165 corners
       ↓
Ray2D TPS denoising on ArUco markers → predict 165 ChArUco
       ↓
TPS re-denoising on completed 165 corners (λ=3, Huber c=1.5)
       ↓
Constrained Zernike rayfield O(0)+d(2), shared R+XY, per-pose Z
       ↓
Stability test: ΔZ₀ < 0.1° between constrained and full-pose fits
       ↓
Read CMO descriptors from (O, d)
       ↓
Propose physical models → fit → residual analysis → iterate
```

### Detection and preprocessing

OpenCV `CharucoDetector` with `DICT_6X6_250`, `setLegacyPattern(True)`,
and tuned detector parameters.  The Hessian corner completion fills
missing ChArUco corners via $|\det H|$, Otsu thresholding, connected
components, and subpixel barycentre via `cv2.moments`.

The **double TPS pass** is critical for rayfield stability:

1. TPS on ArUco marker corners (homography + thin-plate spline residuals)
   predicts all 165 ChArUco grid corners.
2. A second TPS pass uses the completed 165 corners themselves as control
   points with tighter smoothing (λ = 3, Huber c = 1.5).  This eliminates
   residual detection noise and makes the Zernike rayfield **gauge-stable**
   (Z₀ drift between constrained and full-pose fits drops from 8.5° to
   0.023°).

The double TPS is a denoising regularizer whose validity is confirmed not
by the 2‑D residual alone, but by the disappearance of gauge drift in the
3‑D Zernike fit.

### Error metric

> **The reported residual is not an OpenCV reprojection RMS.**

For each observed pixel, the fitted ray is intersected with the estimated
board plane.  The 3‑D distance to the corresponding board point is
converted to a **local pixel-equivalent residual**:

$$e_{\text{px}} \approx \frac{e_{\text{mm}}}{|t|} f_x.$$

## Step-by-step: from rayfield to physical model

### Step 1 — The Zernike rayfield as observable

The Zernike rayfield $\mathcal{R}(u,v) = (O(u,v), d(u,v))$ maps each
pixel to a 3‑D line.  The model is O(0) + d(2): rigid sub-pupil per
channel (origin order 0), spatially-varying direction correction
(direction order 2), with constrained poses (shared rotation + XY,
per-pose Z).  This gives 57 parameters total.

The fit reaches **0.47 px** local pixel-equivalent RMS.

From the centre-pixel ray $(O, d)$ we can **read physical descriptors
directly** — no model fit required:

| Descriptor | Symbol | How to read it | Value |
|---|---|---|---|
| Stereo baseline | $b$ | $\|O_R - O_L\|$ | **24.9 mm** |
| Sub-pupil depth | $z_p$ | $(|O_{L,z}| + |O_{R,z}|)/2$ | **2.5 mm** |
| Working distance | $WD$ | Mean of pose Z estimates | **64.7 mm** |
| Objective focal length | $f_{\text{obj}}$ | $WD - z_p$ | **62.2 mm** |
| Convergence angle | $\theta$ | $\arccos(d_L \cdot d_R)$ | **22.6°** |

These are **not fitted physical CMO parameters** — they are rayfield
readouts under a constrained Zernike gauge.  They give us a starting
point for building physical models.

### Step 2 — Perspective CMO: the baseline hypothesis

The simplest CMO model assumes each channel is a perspective camera
viewing the object through a decentered sub-pupil.  Rays originate from
the sub-pupil $S_c = (\pm b/2,\; 0,\; WD - f_{\text{obj}})$ and fan out
to the sensor.  This predicts a direction field with a strong linear
gradient:

$$d_y(u,v) \propto (v - c_y)$$

We build this model using the descriptors read in Step 1, optimise its
19 parameters, and compare against the Zernike rayfield.

**What we observe.**  The Zernike $d_y$ field is **nearly constant**
across the field of view (range = 0.079, mean = +0.059), while the
perspective CMO predicts a gradient from −0.116 to +0.116 (range = 0.232).
This is a **3× range difference** — the real system is far more
telecentric than the perspective model.

**Diagnosis.**  The $d_y(u,v)$ field tells us the dominant ray
geometry.  The near-constant $d_y$ is the signature of **object-space
telecentricity**: the chief rays are almost parallel, not diverging from
a point.  No adjustment of principal point, distortion, or pitch can fix
a 3× structural mismatch — we need a different model family.

### Step 3 — Telecentric CMO: matching the observed structure

The rayfield tells us what the model should look like:

- **Origins** $O_c$ are well described by rigid sub-pupils (read from
  $O(u,v)$ at order 0).
- **Directions** $d_c(u,v)$ are nearly constant, with weak affine
  variations — no perspective gradient.

This leads to the **telecentric CMO model**
(`CMOTelecentricStereoModel`):

$$O_c = S_c = (\pm b/2,\; 0,\; WD - f_{\text{obj}})$$

$$d_c(u,v) = \operatorname{normalize}\left(d_{c,0} + s_x \tilde{u}\,
e_x + s_y \tilde{v}\, e_y + \text{cross} + \text{quadratic}\right)$$

where $\tilde{u}, \tilde{v}$ are normalised angular coordinates.

The key difference from the perspective model: **the direction is not
derived from a point projection**.  Instead, $d(u,v)$ is directly
parameterised as an affine function of pixel position, with slopes
$s_x, s_y$ controlling the residual perspective (or telecentricity).

Adding pupil shear ($\rho_x, \rho_y$) — a small affine variation of the
origin transverse to the direction — gives the **14-parameter** variant.

**Result.**  The telecentric model with pupil shear achieves:

| Metric | Perspective CMO | Telecentric L0 |
|---|---|---|
| Ray RMS (two-plane) | 3.48 mm | **0.12 mm** (29× better) |
| Pixel RMS | 86 px | **14.6 px** (5.9× better) |
| Parameters | 19 | 14 |

The 14-parameter telecentric model captures the dominant geometry far
better than the 19-parameter perspective model — fewer parameters, more
physical fidelity, because the model family matches the observed
structure.

### Step 4 — Residual analysis: what is the model still missing?

The telecentric model reaches 0.12 mm ray RMS but plateaus at ~14.6 px
reprojection — far from the Zernike's 0.47 px.  To understand what is
missing, we compute the residual between the telecentric model and the
Zernike oracle on a 41×41 grid:

- **Direction residual:** $\Delta d(u,v) = d_{\text{Zernike}} - d_{\text{CMO}}$
- **Moment residual:** $\Delta m(u,v) = m_{\text{Zernike}} - m_{\text{CMO}}$,
  where $m = O \times d$ (the Plücker moment).

We project these residuals onto Zernike modes up to order 4:

| Mode | Δd (L) | Δd (R) | Δm (L) | Δm (R) | Interpretation |
|---|---:|---:|---:|---:|---|
| $Z_0^0$ (piston) | **97 %** | **96 %** | **98 %** | **98 %** | **Global offset** |
| $Z_1^1$ (tilt) | 2 % | 3 % | 2 % | 2 % | Negligible |
| All $n \ge 2$ | < 0.5 % | < 1 % | < 0.1 % | < 0.1 % | Negligible |

**Both Δd and Δm are dominated by $Z_0^0$ — a constant (piston) mode.**
This is the crucial diagnostic: a $Z_0$-dominated residual is **not** a
spatial field distortion (which would appear in higher Zernike modes).
It is a **global line-bundle offset** — the entire set of rays from one
channel is slightly rotated or translated relative to the Zernike
reference.

**Hypothesis:** The two optical arms (left and right channels) each have
a small rigid misalignment relative to the ideal CMO skeleton — a
rotation and translation of the entire ray bundle.  This could come from
tube lens decentering, camera port tilt, prism misalignment, or zoom
body asymmetry.

### Step 5 — Testing alternative hypotheses

Before committing to the arm-alignment hypothesis, we test two
alternatives that the residual analysis already suggests are unlikely:

**Hypothesis A — Image-space pre-warp.**  Perhaps the pixel-to-angle
mapping needs more degrees of freedom (non-radial distortion).  We add a
polynomial pre-warp $\xi = W(u,v)$ before the direction model.

| Model | Params | Ray RMS | Pixel RMS |
|---|---|---|---|
| Telecentric L0 | 14 | 0.118 mm | 14.6 px |
| + affine warp | 20 | 0.115 mm | **16.0 px** (worse) |
| + quadratic warp | 26 | 0.115 mm | **16.5 px** (worse) |

The warp coefficients stay near-identity and pixel RMS *degrades*.  The
pre-warp is not the missing degree of freedom — consistent with the
$Z_0$ diagnostic (a warp would produce spatial, not global, changes).

**Hypothesis B — Spatially varying origin.**  Perhaps the effective
origin $O(u,v)$ needs to vary across the field.  We fit affine and
quadratic transverse origin fields while keeping the direction fixed.

| Origin model | Ray RMS | vs constant |
|---|---|---|
| O0 (constant) | 0.117 mm | baseline |
| O1 (affine) | 0.107 mm | 8 % reduction |
| O2 (quadratic) | 0.107 mm | no further gain |

A spatially varying origin improves the ray RMS by only 8 % — the
residual is not spatial.  Again, consistent with the $Z_0$ diagnostic.

### Step 6 — SE(3) arm alignment: the breakthrough

The $Z_0$-dominated residual points to a **global** misalignment.  We
add a per-channel rigid transform (SE(3)) to the telecentric model's
Plücker lines:

$$d' = R_c \, d_{\text{tel}}, \qquad
  O' = R_c \, O_{\text{tel}} + t_c$$

where $(R_c, t_c)$ is a small rotation and translation for each channel
(12 additional parameters total).

Fitting jointly (telecentric parameters + arm transforms) against the
Zernike rayfield:

| Metric | Telecentric L0 | **Telecentric + SE(3)** | Zernike ref |
|---|---|---|---|
| Parameters | 14 | **26** | 57 |
| Ray RMS (mm) | 0.118 | **0.0022** | 0.0007 |
| Direction RMS (°) | 0.27 | **0.003** | 0 |
| Moment RMS (mm) | 0.32 | **0.001** | 0 |
| **Pixel RMS (px)** | 14.6 | **1.13** | 0.47 |
| **Pixel P50 (px)** | 13.2 | **0.94** | — |

The fitted arm transforms are physically plausible:

| Channel | Rotation | Translation |
|---|---|---|
| Left | 2.5° | (−0.02, −0.05, 0.05) mm |
| Right | 3.7° | (−0.71, −0.19, −0.74) mm |

The SE(3) arm alignment reduces pixel RMS by **13×** (14.6 → 1.13 px)
and brings the compact CMO model into the **usable calibration range**.
The Zernike rayfield (57 params, 0.47 px) remains the reference for
subpixel work, but the 26-parameter aligned CMO now provides a
physically interpretable model with 1.13 px accuracy — a 2.4× gap that
likely comes from residual field structure not captured by the compact
parameterisation.

### Step 7 — Why the residual analysis was decisive

The sequence of investigations followed directly from the rayfield
diagnostic:

```text
Residual Δd, Δm projected on Zernike modes
       │
       ├── Z0-dominated (97-98%) → GLOBAL misalignment
       │       │
       │       ├── Pre-warp image? → NO (degrades)
       │       ├── Variable origin? → NO (8% gain)
       │       └── SE(3) arm alignment? → YES (13× improvement)
       │
       └── Higher modes dominant → SPATIAL distortion
               └── (Not what we observe)
```

Without the rayfield, we would be guessing.  The 2‑D reprojection error
tells you *that* the model is wrong, but not *how*.  The Zernike
projection of Δd and Δm tells you exactly what kind of degree of freedom
is missing.

## The Ray2D → Ray3D feedback loop

The double TPS pass was essential: before it, the constrained and
full-pose Zernike rayfields differed dramatically (Z₀ drift = 8.5°,
baseline 17 ↔ 28 mm).  This gauge instability would have made any
residual analysis meaningless — we couldn't have distinguished model
error from preprocessing noise.

After double TPS, the gauge ambiguity vanishes (Z₀ drift = 0.023°), and
the Zernike rayfield becomes a **stable experimental oracle**.  This
feedback loop — Ray2D → Ray3D → diagnose → fix Ray2D → verify with
Ray3D — is a general strategy for any stereo calibration pipeline.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| Pycaso dataset can be processed as legacy ChArUco | Detection with `DICT_6X6_250` + `setLegacyPattern(True)` | Supported |
| Hessian completion fills all 165 corners | `\|det H\|` + Otsu + barycentre | Supported |
| Double TPS eliminates the pose/rayfield gauge | Z₀ drift drops from 8.5° to 0.023° | **Key result** |
| Zernike rayfield reaches subpixel calibration | 0.47 px local pixel-equivalent RMS | Supported |
| Physical descriptors are read directly from (O, d) | $b, WD, f_{\text{obj}}, \theta$ without model fit | Diagnostic |
| $d_y(u,v)$ reveals telecentricity | 3× range difference vs perspective model | Diagnostic |
| Residual modal analysis identifies missing DOF | Δd and Δm are 97-98 % $Z_0^0$ (global, not spatial) | **Diagnostic method** |
| SE(3) arm alignment resolves the global residual | Pixel RMS 14.6 → 1.13 px (13× improvement) | **Key result** |
| The rayfield is a general diagnostic instrument | The feedback loop works: observe → diagnose → fix → verify | **General strategy** |

## What this case study does **not** evaluate

- It does **not** validate absolute metrological accuracy on an independent
  3‑D object.
- It does **not** estimate a full uncertainty budget.
- It does **not** prove that the SE(3) arm transforms correspond to specific
  physical misalignments (they are an effective parameterisation).
- It does **not** test generalisation to other microscopes or datasets.

## Limitations

1. **Gauge dependence.**  The Zernike origin $O(u,v)$ is defined up to a
   displacement along the ray direction.  The transverse gauge
   $O(u,v) \cdot d(u,v) = 0$ is enforced.

2. **Constrained poses.**  The shared-rotation + per-pose-Z assumption is
   physically motivated but unverified.

3. **Fixed K.**  The Zernike BA uses a fixed pinhole reference
   ($f_x = 25600$, principal point at image centre).

4. **No independent 3‑D ground truth.**  Residuals are computed on the
   same board points used for calibration.

5. **Single dataset.**  These results are for one specific Pycaso
   microscope and one calibration target.

6. **SE(3) parameters are effective, not absolute.**  The fitted arm
   transforms capture the global line-bundle misalignment, but may absorb
   other global effects (scale errors, principal plane offsets) that are
   not individually identifiable without additional constraints.

## Saved artefacts

```text
docs/assets/pycaso_real_data/
    detection_summary.json                 ← per-frame ChArUco counts
    summary.json                           ← calibration RMS, CMO descriptors
    model_comparison.json                  ← Zernike vs telecentric vs perspective
    zernike_pose_variants.json             ← full Zernike coeffs for both pose models
    zernike_conditioning_diagnostic.json   ← design matrix, modal Δd, sensitivity
    zernike_gauge_regularization_sweep.json ← regularization sweep
    moment_residual_diagnostic.json        ← Δm modal decomposition + O1/O2 fits
    arm_alignment_diagnostic.json          ← SE(3) arm alignment sweep
    aligned_cmo_fit.json                   ← final joint fit (telecentric + SE(3))
    warped_model_comparison.json           ← pre-warp L1/L2 evaluation
    pareto_gauge_regularization.png        ← Pareto frontier plot
```

To regenerate:

```bash
PYTHONPATH=src python examples/notebooks/09_pycaso_real_data.py
```

## See also

- :doc:`IDENTIFY_MY_OPTICS` — how to read physical descriptors from a rayfield
- :doc:`CMO_PHYSICAL_MODEL` — the shared-rig CMO model definition
- :doc:`DIRECT_VS_RAYFIELD_INVERSION` — why measure a rayfield before fitting optics
- :doc:`NOTEBOOKS` — all walkthrough notebooks
- [Notebook 09](../examples/notebooks/09_pycaso_real_data.py) — executable protocol
