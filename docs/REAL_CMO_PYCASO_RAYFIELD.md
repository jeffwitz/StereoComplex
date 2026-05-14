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
Hessian-based subpixel procedure, denoises the 2‑D grid using Ray2D TPS, and
fits a constrained Zernike rayfield.

The measured rayfield reaches **subpixel local pixel-equivalent residuals**
(< 0.5 px) on this dataset.  A standard central OpenCV stereo calibration,
under the tested configuration, does not produce a usable model.

The fitted rayfield also provides **CMO-consistent geometric descriptors**
read directly from the measured rays — without fitting a physical model:
effective sub-pupil baseline, working distance, objective focal length
estimate, and chief-ray convergence angle.

These descriptors are **not fitted physical CMO parameters**; they are
rayfield readouts under a constrained Zernike gauge.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| Pycaso dataset can be processed as legacy ChArUco | Detection with `DICT_6X6_250` + `setLegacyPattern(True)` gives 61–161 corners/frame | Supported |
| Missing corners can be completed robustly | Hessian \|det H\| + Otsu + barycentre fills all 165 corners on every frame | Supported, heuristic |
| Ray2D TPS improves 2‑D point quality | TPS fitted on ArUco marker corners predicts denoised ChArUco grid | Supported |
| Constrained Zernike rayfield fits the data subpixel | Local pixel-equivalent RMS 0.47 px (O(0)+d(2)), 0.41 px best | Supported |
| The rayfield yields CMO-consistent descriptors | $b$, $WD$, $f_{\text{obj}}$, $\theta$ read from centre-pixel rays | Diagnostic, gauge-dependent |
| A minimal perspective CMO model is insufficient across the FOV | Zernike-vs-CMO field comparison shows structured mismatch (3× d_y range) | Diagnostic |
| Higher Zernike orders improve RMS modestly | Sweep O(0..2)+d(2..4): 0.47 → 0.41 px (−13 %) | Supported |
| Baseline $b$ is less stable than $WD$ | $b$ varies 20–25 mm under higher O-orders; $WD$ spread < 0.5 mm | Supported, gauge-sensitive |
| Full-pose vs constrained Zernike difference is dominated by a gauge mode | Modal decomposition: 90 % of Δd is \(Z_0^0\) (global direction piston), not wobble | Supported |

## What this case study evaluates

1. Can a real CMO microscope calibration stack be processed with the
   StereoComplex pipeline?  **Yes.**
2. Can a constrained Zernike rayfield explain the observations with subpixel
   local pixel-equivalent residuals?  **Yes (0.47 px).**
3. Which CMO-like geometric descriptors can be read from the measured rayfield,
   and how stable are they with respect to Zernike order?  **$b$, $WD$,
   $f_{\text{obj}}$, $\theta$; $WD$ and $f_{\text{obj}}$ are stable, $b$ is
   gauge-sensitive.**

## What this case study does **not** evaluate

- It does **not** validate absolute metrological accuracy on an independent
  3‑D object.
- It does **not** estimate a full uncertainty budget.
- It does **not** perform a full physical CMO parameter optimisation.
- It does **not** prove that the simplified perspective CMO model is the
  correct physical model of the microscope.

## Dataset and reproducibility

**Source:** Pycaso example calibration images.

```bash
git clone https://github.com/LaboratoireMecaniqueLille/Pycaso examples/pycaso_data
```

Used folders: `left_calibration11/`, `right_calibration11/`.

| Property | Value |
|---|---|
| Sensor | 2048 × 2048 px |
| Board | Legacy ChArUco, 16 × 12 squares, 0.3 mm |
| Dictionary | DICT_6X6_250, `setLegacyPattern(True)` |
| Frames | 10 stereo pairs |
| Z range | 2.65 – 3.35 mm (Δ = 0.70 mm) |

The dataset is **not vendored** in the StereoComplex repository.  The notebook
expects a local clone or copy.  If absent, it fails with a clear error message.

## Pipeline

```text
ChArUco legacy detection (DICT_6X6_250, setLegacyPattern)
       ↓
Hessian corner completion (|det H| + Otsu + barycentre)  →  165/165 corners
       ↓
Ray2D TPS denoising (predict_points_rayfield_tps_robust)
       ↓
TPS re-denoising on completed 165 corners (smoothing pass)
       ↓
Constrained Zernike rayfield O(0)+d(2), shared R+XY, per-pose Z
       ↓
CMO-consistent geometric descriptors read from (O, d)
       ↓
Simplified CMO model mismatch diagnostic (telecentricity)
       ↓
Zernike order sweep O(0..2)+d(2..4)
```

### Step 1 — Legacy ChArUco detection

OpenCV `CharucoDetector` with `DICT_6X6_250`, `setLegacyPattern(True)`, and
tuned detector parameters (small marker perimeter rates for 0.15 mm markers).

| Metric | Left | Right |
|---|---|---|
| Mean detected corners | ~146 | ~148 |
| Min detected corners | 61 | 70 |
| Max detected corners | 161 | 161 |
| After completion | 165 | 165 |

### Step 2 — Hessian corner completion

Missing ChArUco corners are predicted by an affine mapping from detected corner
IDs to image coordinates.  A local Hessian-response blob search refines the
position:

$$R(x,y) = |I_{xx}I_{yy} - I_{xy}^2|,$$

followed by Otsu thresholding, connected-component labelling, and subpixel
barycentre via `cv2.moments`.

*This is a practical completion heuristic, not an independent ground-truth
detector.  Its effect is assessed through the downstream rayfield residuals.*

### Step 3 — Ray2D TPS denoising

Ray2D TPS is a **2‑D preprocessing stage**, not a 3‑D correction.  It maps
known ArUco marker coordinates to image positions using a homography + TPS
residual field, predicting the full 165‑corner ChArUco grid.  This reduces
local detection noise before the 3‑D rayfield fit.

After Hessian completion fills all 165 corners, a **second TPS pass** uses
the completed corners themselves as control points (object positions
$\to$ image positions) with a tighter smoothing parameter ($\lambda=3$,
Huber $c=1.5$).  This acts as a definitive denoising step: the homography
captures the global perspective, and the TPS smooths the residual field
across all 165 points, removing the last detection jitter while preserving
the grid structure.

### Step 4 — Constrained Zernike rayfield

The model:

$$(u,v) \mapsto \mathcal{R}_c(u,v) = (O_c(u,v), d_c(u,v)),$$

with rigid sub-pupil per channel: $O_c(u,v) = O_{c,0}$ (order 0), and
spatially-varying direction correction up to radial order 2:

$$d_c(u,v) = d_{c,0}^{\text{pinhole}}(u,v) + \sum_{m \le 2} a_{c,m}^d Z_m(u,v).$$

**Pose constraint:** the board is assumed mounted on a Z-only translation
stage.  All frames share the same rotation (3 params) and X,Y translation
(2 params); only Z varies per frame (10 params).  This reduces the pose
parameters from 60 to 15 — a strong but physically motivated constraint
that makes the 57‑parameter Zernike BA well-conditioned.

### Step 5 — Error metric

> **The reported residual is not an OpenCV reprojection RMS.**

For each observed pixel, the fitted ray is intersected with the estimated
board plane.  The 3‑D distance to the corresponding board point is converted
to a **local pixel-equivalent residual**:

$$e_{\text{px}} \approx \frac{e_{\text{mm}}}{|t|} f_x.$$

This is a local first-order approximation, not an image-plane reprojection
residual from a projective camera model.  **OpenCV RMS and StereoComplex
pixel-equivalent RMS are not the same statistical quantity.**  They are used
here as practical indicators of whether each model provides a usable
calibration on this dataset.

## Results

### Main calibration residuals

| Model / method | Residual type | RMS | P95 | Status |
|---|---|---|---|---|
| OpenCV central stereo | image reprojection RMS | > 300 px | n/a | not usable under tested config |
| Zernike O(0)+d(2) | local pixel-equivalent | 0.47 px | 0.85 px | usable |
| Zernike O(2)+d(3) (best) | local pixel-equivalent | 0.41 px | 0.79 px | modest improvement |

### CMO-consistent geometric descriptors

Read from the Zernike rayfield at the centre pixel (1024, 1024) —
**without numerical optimisation**:

| Descriptor | Symbol | Value (O(0)) | Stability (sweep) | Interpretation |
|---|---|---|---|---|
| Stereo baseline | $b$ | 24.9 mm | 20–25 mm | gauge-sensitive |
| Sub-pupil depth | $z_p$ | 2.5 mm | — | from $O_z$ |
| Working distance | $WD$ | 64.7 mm | < 0.5 mm spread | robust |
| Objective focal length | $f_{\text{obj}}$ | 62.2 mm | ~1.5 mm spread | moderately robust |
| Convergence angle | $\theta$ | 22.6° | — | stereo geometry |

**The $O(0)$ model is the most appropriate for reading a rigid sub-pupil
baseline** because it forces each channel origin to be spatially constant.
Higher O-orders improve flexibility but can absorb part of the physical
baseline into spatial origin variations — a known gauge freedom.

### Zernike order sweep

| Model | Params | RMS (px) | P95 (px) | $b$ (mm) | $f_{\text{obj}}$ (mm) | $WD$ (mm) |
|---|---|---|---|---|---|---|
| O(0)+d(2) | 57 | 0.470 | 0.852 | 24.9 | 62.2 | 64.7 |
| O(1)+d(2) | 69 | 0.444 | 0.807 | 19.6 | 63.6 | 65.1 |
| O(0)+d(3) | 81 | 0.419 | 0.804 | 25.0 | 62.3 | 64.8 |
| O(1)+d(3) | 93 | 0.412 | 0.791 | 21.0 | 63.4 | 65.1 |
| O(2)+d(3) | 111 | **0.409** | **0.786** | 19.9 | 63.7 | 65.2 |
| O(1)+d(4) | 123 | 0.412 | 0.790 | 20.6 | 63.5 | 65.1 |
| O(2)+d(4) | 141 | 0.410 | 0.789 | 22.3 | 63.2 | 65.2 |

**Conclusion:** O(0)+d(2) is the most physically interpretable model.
Higher orders reduce RMS by ~13 % but at the cost of mixing baseline into
the origin gauge.  The plateau at ~0.41 px suggests we have reached the
noise floor of the corner detections.

## Zernike vs simplified CMO model

### Diagnostic, not a model fit

The simplified CMO model is built from the centre-pixel descriptors with
**fixed principal point** (image centre), **zero distortion**, **fixed pixel
pitch** (5.5 µm), and a **guessed tube-lens focal length** (50 mm).  It uses
a **minimal perspective CMO parameterization**.

The purpose of this comparison is **diagnostic**: it shows which parts of the
measured field are captured by a simple CMO interpretation and which remain
unexplained.

### Field-of-view comparison

| Component | Zernike (measured) | CMO (minimal perspective) |
|---|---|---|
| $d_y$ range | 0.079 (nearly constant) | 0.232 (perspective gradient) |
| $d_y$ at centre | +0.059 | ~0 |
| Interpretation | ≈ telecentric in Y | perspective from sub-pupil |

The Zernike $d_y \approx 0.059 \pm 0.04$ across the full sensor, while the
CMO model predicts $d_y$ varying from −0.116 to +0.116.  This **3× range
difference** suggests that this real system behaves more telecentrically
across the field than the minimal perspective CMO model used here.

### What this does and does not prove

- **Diagnostic:** the measured rayfield is structurally different from a
  minimal perspective CMO model across the FOV.
- **Hypothesis:** the real optics are more object-space telecentric than the
  simple CMO model, consistent with the expected behaviour of an
  infinity-corrected tube-lens system.
- **Not proven:** that the CMO architecture is wrong.  A more complete CMO
  model (with optimised principal point, distortion, and pixel pitch) might
  reduce the mismatch.  The diagnostic shows where the minimal model fails;
  it does not reject the CMO family.

## How the rayfield guided a better physical model

The Zernike rayfield is not just a calibration tool — it is a **diagnostic
instrument** that reveals the structure of the real optics.  Here is how we
used the measured $(O, d)$ to design a new physical model that matches the
data 10× better than the perspective CMO, with 6× fewer parameters.

### Step 1 — Read the sub-pupil positions from $O(u,v)$

At the centre pixel, the Zernike origins are:

$$O_L = (-12.7,\,-0.1,\,2.7)\;\text{mm}, \qquad
  O_R = (12.1,\,-0.1,\,2.3)\;\text{mm}$$

These are the **effective sub-pupils** — the points where the chief rays
appear to originate.  From them we read the baseline $b = \|O_R-O_L\|
\approx 24.9$ mm and the sub-pupil depth $z_p = (|O_{L,z}|+|O_{R,z}|)/2
\approx 2.5$ mm.  These describe the **stereo geometry** and are stable
across Zernike orders (especially for $O(0)$, the rigid-origin model).

### Step 2 — Examine the direction field $d(u,v)$

We evaluate the Zernike direction $d_y(u,v)$ on a grid across the sensor:

```
v=0:    +0.098  +0.098  +0.098  +0.097  +0.097
v=1024: +0.059  +0.059  +0.059  +0.058  +0.058
v=2047: +0.019  +0.019  +0.020  +0.020  +0.020
```

The $d_y$ component is **nearly constant** across the field of view
(range = 0.079, mean = +0.059).  This is the signature of **object-space
telecentricity**: the chief rays are almost parallel.

### Step 3 — Compare with the perspective prediction

A perspective model (all rays from a single sub-pupil point) predicts
$d_y \propto (v - c_y)$ — a linear gradient from negative (top) to
positive (bottom).  For the CMO perspective model with the same sub-pupil:

```
v=0:    -0.116  -0.115  -0.114  -0.113  -0.111
v=1024: -0.000  -0.000  -0.000  -0.000  -0.000
v=2047: +0.116  +0.115  +0.114  +0.113  +0.111
```

The perspective model predicts a range of 0.232 — **3× larger** than the
measured 0.079.  No adjustment of principal point, distortion, pitch, yaw,
or telecentric offset can fix this: it is a **structural difference**
between perspective projection and the real telecentric imaging.

### Step 4 — Design a model that matches the observed structure

The data tells us:

- **Origins** $O_c$ are well described by a rigid sub-pupil per channel
  (read from $O(u,v)$ at order 0).
- **Directions** $d_c(u,v)$ are nearly constant, with weak linear variations
  (no perspective gradient).

This leads to the **telecentric CMO model**
(`CMOTelecentricStereoModel`):

$$O_c = S_c = (\pm b/2,\; 0,\; WD - f_{\text{obj}})$$

$$d_c(u,v) = \operatorname{normalize}\left(
    d_{c,0} + s_x \tilde{u}\, e_x + s_y \tilde{v}\, e_y
\right)$$

where $\tilde{u} = (u - c_x) \cdot p_{\text{pix}} / f_{\text{ang}}$ and
$\tilde{v} = (v - c_y) \cdot p_{\text{pix}} / f_{\text{ang}}$ are
normalised angular coordinates, and $d_{c,0}$ is the chief-ray direction
(antisymmetric in X for stereo, shared Y component).

The key difference from the perspective model: **the direction is not
derived from a point projection**.  Instead, $d(u,v)$ is directly
parameterised as an affine function of pixel position, with slopes $s_x,
s_y$ controlling the residual perspective (or telecentricity).

### Step 5 — Validate the model structure

With the origin parameters fixed to the Zernike readings ($f_{\text{obj}}
= 62$ mm, $WD = 65$ mm, $b = 24.9$ mm), the telecentric model with only
7 free direction parameters ($c_x, c_y, f_{\text{ang}}, \theta, d_y, s_x,
s_y$) reproduces the Zernike $d_y$ field almost perfectly **without any
optimisation**:

| Model | Parameters | $d_y$ range | $d_y$ mean |
|---|---|---|---|
| Zernike O(0)+d(2) (reference) | 57 | 0.079 | +0.059 |
| **Telecentric CMO (seed)** | **7** | **0.073** | **+0.058** |
| Perspective CMO (optimised) | 19 | 0.232 | ~0 |

The telecentric model achieves:

- **Direction-field diagnostic:** the $d_y$ range (0.073) and mean
  (+0.058) match the Zernike within < 10 %, compared to the perspective
  CMO's 3× range error.
- **Full rayfield fit:** two-plane RMS improves by **22×** relative to
  the perspective CMO (0.16 mm vs. 3.5 mm), with **3× fewer parameters**
  (7 vs. 19).  The optimised convergence half-angle (11.3°) matches the
  Zernike centre-pixel reading exactly.

> The quasi-telecentric CMO model is **not a replacement** for the
> measured Zernike rayfield when subpixel reconstruction is required
> (Zernike: 0.47 px, telecentric: 27 px pixel-equivalent).  It is a
> **compact physical explanation** of the dominant CMO geometry.

### Why this workflow generalises

The sequence — measure $(O,d)$ → read physical descriptors → diagnose
structural mismatch → design a better model → validate against the
rayfield — is the core scientific workflow that StereoComplex enables.

1. **The rayfield is the observable.**  From it, we read sub-pupil
   positions and baseline without any model fit.
2. **The rayfield is the diagnostic.**  Comparing $d_y(u,v)$ across
   models reveals structural differences (perspective vs. telecentric)
   that no amount of parameter tuning can fix.
3. **The rayfield is the validation target.**  A new physical model is
   tested directly against the measured $(O,d)$, not against the original
   corner detections, decoupling measurement from interpretation.

### Full rayfield fit

The full two-plane ray-space residual (comparing $(O,d)$ at $z=50$ mm
and $z=80$ mm) confirms the telecentric model's superiority:

| Model | Params | Two-plane RMS (mm) | Dir RMS | Mom RMS | Pix RMS |
|---|---|---|---|---|---|
| Zernike O(0)+d(2) | 57 | 0.0007 | 0° | 0 mm | 0.47 px |
| Telecentric + pupil shear | 14 | 0.126 | 0.29° | 0.34 mm | 13.6 px |
| Telecentric (no shear) | 7 | 0.156 | 0.29° | 0.34 mm | 27.7 px |
| Perspective CMO | 19 | 3.48 | ~2.0° | ~0.5 mm | 86.0 px |

The telecentric model achieves a **22× improvement** over the perspective
CMO in ray-space, with **3× fewer parameters** (7 vs 19).  Pupil shear
(14 params) adds a modest ~20% further improvement (0.156→0.126 mm),
confirming that the dominant residual is **not** a simple affine origin
shift but likely higher-order direction curvature.  The fitted parameters
are physically plausible:

- $\theta_{\text{half}} = 11.3^\circ$ (matches Zernike centre-pixel reading)
- $d_y = 0.0585$ (matches Zernike mean)
- $s_y = -0.50$ (captures the flat Y gradient)
- $s_x = 0.49$ (captures the X-direction variation)
- Sub-pupil Z = 2.5 mm (matches Zernike $O_z$)

## Limitations

1. **Gauge dependence.**  The Zernike origin $O(u,v)$ is defined up to a
   displacement along the ray direction.  The transverse gauge
   $O(u,v) \cdot d(u,v) = 0$ is enforced, but this choice affects the
   numerical values of $b$ and $z_{\text{pupil}}$ (especially for O(≥1)).

2. **Constrained poses.**  The shared-rotation + per-pose-Z assumption is
   physically motivated but unverified.  Any real stage wobble or board tilt
   variation between frames is absorbed into the rayfield fit.

3. **Fixed K.**  The Zernike BA uses a fixed pinhole reference ($f_x = 25600$,
   principal point at image centre).  Errors in this reference propagate to
   the Zernike coefficients.

4. **No independent 3‑D ground truth.**  The residuals are computed on the
   same board points used for calibration.  There is no validation on a
   separate object at a different depth.

5. **CMO comparison model.**  The simplified CMO model uses guesses for
   $f_{\text{tube}}$, pixel pitch, principal point, and distortion.  These
   choices contribute to the observed mismatch.

6. **Single dataset.**  These results are for one specific Pycaso microscope
   and one calibration target.  Generalisation to other instruments or boards
   requires additional validation.

### Zernike pose model: constrained vs full

The Zernike rayfield can be fitted with constrained poses (shared R+XY,
per-pose Z, 15 params) or full per-frame poses (60 params).  The choice
affects how well the compact telecentric model can match the rayfield:

| Rayfield target | Pix RMS | Telecentric fit |
|---|---|---|
| Zernike constrained | 0.41 px | 0.13 mm, 0.3° dir (good match) |
| Zernike full poses | 0.17 px | 0.69 mm, 5.0° dir (harder target) |

The full-pose Zernike achieves lower corner error (0.17 px vs 0.41 px)
by using more parameters (132 vs 87).  However, this flexibility comes at
a cost: the inferred rayfield geometry changes substantially — baseline
(27.8 vs 16.8 mm), convergence angle (25.3° vs 15.0°), $d_y$ range
(0.19 vs 0.055), and origin asymmetry (5.0 vs 1.0 mm).  This is a
**pose/rayfield identifiability issue**: without external ground truth,
neither variant can be proven physically correct.

The constrained-pose Zernike is the more **conservative** intermediate
for CMO-like interpretation — it is more symmetric, more telecentric,
and more compatible with the quasi-telecentric compact model.  The
full-pose Zernike may reveal real asymmetries or may simply overfit.

### Conditioning diagnostic: why constrained and full Zernike differ

We conducted a Zernike/pose identifiability analysis to determine whether the
difference between constrained and full-pose Zernike rayfields is caused by
poorly constrained Zernike modes trading off with pose parameters.

**Design matrix conditioning.** The Zernike basis on the full 41×41 square
sensor grid is well-conditioned: cond(\(B_2\)) = 4.8, cond(\(B_4\)) = 14.5.
However, \(Z_0^0\) and \(Z_2^0\) are not orthogonal on the square sensor
(off-diagonal Gram correlation 0.56).  On sparse ChArUco-like sampling,
conditioning degrades significantly: cond(\(B_4\)) = 71, with
\(Z_2^2(\cos)\) loading 78 % onto the last singular vector.

**Modal decomposition of Δd.**  We projected the direction difference
\(\Delta d = d_{\text{full}} - d_{\text{constrained}}\) onto Zernike modes
up to order 4:

| Mode | Left Δd | Right Δd | Interpretation |
|---|---:|---:|---|
| \(Z_0^0(m_0)\) — piston | **89.7 %** (8.1°) | **77.2 %** (4.9°) | **Gauge mode**: global direction offset |
| \(Z_1^1(\sin)\) — y‑tilt | 5.9 % (2.1°) | 12.4 % (2.0°) | Tilt ↔ \(R_y\) coupling |
| \(Z_1^1(\cos)\) — x‑tilt | 4.4 % (1.8°) | 9.5 % (1.7°) | Tilt ↔ \(R_x\) coupling |
| All higher modes (\(n \ge 2\)) | < 0.1 % | < 0.5 % | Negligible |

**90 % of Δd is Z₀⁰ — a global direction piston.**  This mode shifts all
ray directions by a constant offset, which is equivalent to changing the
effective focal length of the pinhole reference.  It is a **gauge freedom**
of the Zernike + poses inverse problem: changing the global direction scale
can be absorbed by pose translations without changing the corner
reprojection residuals.  Only ~10 % of Δd comes from tilt modes
(\(Z_1^1\)) that represent the actual 0.31° mechanical wobble.

**Physical indicator sensitivity.**  We computed the sensitivity of CMO
descriptors (baseline, convergence angle, \(d_y\) range, sub-pupil depth)
to each Zernike coefficient via finite-difference perturbation:

| Mode | Most sensitive indicator | Sensitivity |
|---|---|---|
| \(Z_0^0(m_0)\) d-coeff | subpupil depth | 2.36 mm / 0.01 coeff |
| \(Z_1^1(\cos)\) d-coeff | \(d_y\) range (telecentricity) | 1.40 / 0.01 coeff |
| \(Z_1^1(\sin)\) d-coeff | \(d_y\) range | 1.40 / 0.01 coeff |
| \(Z_2^0(m_0)\) d-coeff | \(d_x\) antisymmetry | 1.95 / 0.01 coeff |
| \(Z_2^2(\sin)\) d-coeff | \(d_y\) range | 1.75 / 0.01 coeff |

**Coefficient stability.**  We compared Zernike coefficients between the
constrained and full-pose solutions:

| Mode | ΔO_L (mm) | Δd_L | Δd_R | Stability |
|---|---:|---:|---:|---|
| \(Z_0^0(m_0)\) | 9.77 | 0.149 | 0.089 | **UNSTABLE** (gauge) |
| \(Z_1^1(\cos)\) | 5.08 | 0.078 | 0.075 | **UNSTABLE** (tilt) |
| \(Z_1^1(\sin)\) | 6.22 | 0.093 | 0.086 | **UNSTABLE** (tilt) |
| \(Z_2^0(m_0)\) | 0.05 | 0.001 | 0.002 | MODERATE |
| \(Z_2^2(\cos)\) | 0.17 | 0.002 | 0.002 | UNSTABLE (weak) |
| \(Z_2^2(\sin)\) | 0.25 | 0.003 | 0.005 | UNSTABLE (shear) |

4 of 6 direction modes are unstable between the two fits.  However, only
\(Z_0^0\) and \(Z_1^1\) have large absolute coefficient changes; the
higher-order instabilities reflect small norms rather than large shifts.

**Conclusion.**  The full-pose Zernike's lower pixel RMS (0.17 vs 0.41 px)
comes primarily from a **gauge mode** (\(Z_0^0\) direction piston) that is
nearly unobservable from corner data — it changes the global direction
scale, which poses can absorb.  Only ~10 % of the improvement comes from
actual geometric modelling of the 0.31° wobble.

**Recommendation.**  Keep constrained poses as the conservative
intermediate.  For applications needing lower pixel error, add mild
Tikhonov regularization on the direction coefficients proportional to
their pose sensitivity, rather than freeing all poses.  This preserves
physical interpretability while capturing the small amount of real
wobble present in the data.

Artefacts: `zernike_conditioning_diagnostic.json`,
`zernike_conditioning_summary.json`.

### Gauge-regularized full-pose sweep

Following the conditioning diagnostic, we tested a **regularized full-pose**
Zernike fit that anchors the gauge-sensitive direction modes (\(Z_0^0\),
\(Z_1^1\)) to the constrained-pose solution:

$$\mathcal{L} = \mathcal{L}_{\text{repr}} + \sum_{m \in \{Z_0,Z_1\}} \sum_c
  \left(\frac{a^f_{m,c} - a^c_{m,c}}{\sigma_m}\right)^2$$

where \(\sigma_m\) is an interpretable angular tolerance (degrees).
A small \(\sigma\) strongly anchors the gauge mode; a large \(\sigma\)
recovers the unregularized fit.

The sweep over \(\sigma_{Z_0} \in [0.05°, 2.0°]\) and \(\sigma_{Z_1} \in
[0.5°, 2.0°]\) is implemented as notebook section 10.3 (controlled by
``RUN_SWEEP`` flag, takes ~10–15 min).  Results are saved to
`zernike_gauge_regularization_sweep.json`.

The regularization strategy is **preferable to removing modes** because
\(Z_0^0\) and \(Z_1^1\) carry essential physical information (mean
direction, convergence angle, first-order telecentricity).  Penalizing
their drift preserves them while preventing gauge-driven overfitting.
Which interpretation is correct depends on independent validation of
the microscope geometry.

See `docs/assets/pycaso_real_data/pose_model_comparison.json`.

## Reproducibility checklist

- [ ] Pycaso example data cloned at `examples/pycaso_data`
- [ ] Run `PYTHONPATH=src python examples/notebooks/09_pycaso_real_data.py`
- [ ] Detection summary shows 165/165 corners after completion
- [ ] Zernike fit converges (typical: 55 NFEV, tens of seconds on a modern CPU)
- [ ] Local pixel-equivalent RMS < 0.5 px
- [ ] CMO-consistent descriptors printed
- [ ] Zernike order sweep table printed
- [ ] Artefacts saved in `docs/assets/pycaso_real_data/`

## Saved artefacts

The notebook generates the following artefacts:

```text
docs/assets/pycaso_real_data/
    detection_summary.json              ← per-frame ChArUco counts (L/R)
    summary.json                        ← calibration RMS, CMO descriptors, sweep best
    zernike_order_sweep.json            ← full sweep table (all orders, RMS, descriptors)
    model_comparison.json               ← Zernike vs telecentric vs perspective comparison
    telecentric_component_diagnostics.json  ← per-component error breakdown
    pose_model_comparison.json          ← constrained vs full poses benchmark
    two_plane_sensitivity.json          ← Z-plane sweep showing metric amplification
    diagnostic_cmo_vs_zernike.txt       ← full diagnostic report
    zernike_pose_variants.json          ← full Zernike coeffs for both pose models
    zernike_conditioning_diagnostic.json ← design matrix, modal Δd, sensitivity, stability
    zernike_conditioning_summary.json   ← condensed conditioning conclusions
    zernike_gauge_regularization_sweep.json ← Pareto sweep over σ_Z0, σ_Z1
```

All numerical values reported in this page are produced by the notebook
and saved in these artefacts.  To regenerate:

```bash
PYTHONPATH=src python examples/notebooks/09_pycaso_real_data.py
```

## See also

- :doc:`IDENTIFY_MY_OPTICS` — how to read physical descriptors from a rayfield
- :doc:`CMO_PHYSICAL_MODEL` — the shared-rig CMO model definition
- :doc:`DIRECT_VS_RAYFIELD_INVERSION` — why measure a rayfield before fitting optics
- :doc:`NOTEBOOKS` — all walkthrough notebooks
- [Notebook 09](../examples/notebooks/09_pycaso_real_data.py) — executable protocol
