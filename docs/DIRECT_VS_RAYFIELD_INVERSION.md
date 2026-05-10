# Direct model inversion vs rayfield-mediated inversion

## Why this page exists

StereoComplex follows a two-stage philosophy:

```text
measure a generic rayfield first  →  then explain the optics
```

This page compares that strategy against the more traditional approach
of fitting optical models directly to 2-D corner observations, and
explains **why** the intermediate rayfield step improves model-selection
interpretability.

## The two inverse problems

### Pipeline A — Direct ChArUco inversion

Given ChArUco corner detections $(u_{ik}^\text{obs})$ for poses $i$ and
points $k$, jointly optimise optical parameters $\theta$ and board poses
$\eta_i$:

```{math}
\min_{\theta,\eta}
\sum_{i,k,c}
\left\|
u_{ik,c}^{\text{obs}}
-
\operatorname{Project}_{\mathcal{R}_{\theta,c}}\!\big(T_{\eta_i}X_k\big)
\right\|^2
```

The optical parameters and the board poses are **coupled**: a bias in the
pose can compensate for an error in the optical model, and vice versa.
This coupling makes model selection harder — a wrong optical model can
achieve low pixel RMS by absorbing its error into the fitted poses.

### Pipeline B — Rayfield-mediated inversion

First, estimate the instrument response as a generic pixel-to-line map
$\widehat{\mathcal{R}}(u,v) = (\widehat{O}(u,v), \widehat{d}(u,v))$ from
the same ChArUco observations.  Then, compare physical hypotheses in a
common ray space:

```{math}
\min_\theta
\sum_k
\left\|
\widehat{\mathcal{R}}(u_k,v_k)
-
\mathcal{R}_\theta(u_k,v_k)
\right\|^2_{\text{line}}
```

The two-plane ray residual (see [Identify My Optics](IDENTIFY_MY_OPTICS.md#ray-space-comparison))
compares line geometry, not raw origins, making the comparison
gauge-invariant.  **Board poses do not appear in this second stage** —
they have already been absorbed into the rayfield measurement.

## Why pinhole initialisation works for non-pinhole optics

A natural question: pipeline A is initialised with `cv2.solvePnP`, which
assumes a central pinhole model.  How can this work for a CMO, where rays
pass through off-axis sub-pupils rather than a single camera centre?

The answer is that the pinhole model is **structurally wrong but
geometrically close**.  For the pedagogical CMO oracle:

- Sub-pupil $S_L = (-4, 0, 40)$ mm (offset from the axis by 4 mm, at
  depth $Z_w - f_\text{obj} = 40$ mm behind the main objective).
- Working distance $Z_w = 120$ mm.
- Chief-ray angle from $S_L$ to the axis at the working plane:
  $\arctan(4/80) \approx 2.9°$.
- A pinhole at the origin would see the same working-plane point at
  $\arctan(4/120) \approx 1.9°$.

The angular difference is only ~1°.  `solvePnP` compensates by shifting
the estimated pose by ~1–2 mm in translation and ~2–3° in rotation,
producing an initial pixel reprojection error of ~1–2 px.

The joint nonlinear optimiser then refines **both** the optical parameters
and the poses.  As $f_\text{obj}$, $Z_w$, and $b$ converge toward their
true values, the sub-pupil moves to its correct off-axis position, the
poses self-correct, and the pixel RMS drops from ~2 px to machine zero.

This initialisation robustness is **not guaranteed** for all optical
architectures.  A Scheimpflug system or a strongly decentered relay
would produce larger pinhole-pose errors, potentially causing the joint
optimiser to diverge or converge to a local minimum.  Pipeline B
(rayfield-mediated) mitigates this risk: the Zernike BA still needs a
pose initialisation, but it does not assume any specific optical model.
Pose errors are partially absorbed by the flexible Zernike
parameterisation rather than fighting the optical parameters.

## Nuisance parameters and conditioning

In pipeline A, the joint parameter vector includes both optical
parameters $\theta$ and pose parameters $\eta$.  The information matrix
for $\theta$ after eliminating the nuisance poses is the Schur complement:

```{math}
I_{\theta|\eta}
=
J_\theta^T J_\theta
-
J_\theta^T J_\eta
\left(J_\eta^T J_\eta\right)^{-1}
J_\eta^T J_\theta
```

where $J_\theta$ and $J_\eta$ are the Jacobian blocks of the pixel
residual with respect to optical and pose parameters.  When
$\|J_\theta^T J_\eta\|$ is large, poses and optics are strongly coupled,
and the effective information $I_{\theta|\eta}$ can be much smaller than
the naïve optics-only information $J_\theta^T J_\theta$.

Pipeline B has **no pose parameters** in the second stage, so its
information matrix is simply $J_\theta^T J_\theta$.  The rayfield has
already absorbed the pose degrees of freedom.

The module `stereocomplex.benchmarks.inverse_problem_diagnostics` provides
tools to compute these quantities: Jacobians via finite differences, Schur
complements, condition numbers, and pose-optics correlation matrices.

## Infrastructure

### Shared oracle builders

`stereocomplex.benchmarks.model_selection_oracles` provides six synthetic
stereo rayfield pairs (pinhole, Brown-Conrady, inclined plate, CMO
shared-rig, Greenough, exotic Zernike) as `StereoOracle` dataclasses.
These are shared between the classification matrix (notebook 07) and the
direct-vs-rayfield study (notebook 08).

### ChArUco observation simulator

`stereocomplex.benchmarks.charuco_observation_simulator` generates
synthetic ChArUco corner observations from any stereo rayfield oracle.
For each board pose, 3-D board points are inverse-projected through the
left and right rayfields to produce 2-D pixel coordinates, with optional
Gaussian noise and dropout.

### Inverse point→pixel projection

`stereocomplex.benchmarks.rayfield_projection` solves

```{math}
\min_{u,v} \left\|(I - d\,d^T)(X - O)\right\|^2
```

to find the pixel whose ray passes closest to a 3-D point.  This is used
both by the observation simulator and by the direct inversion pipeline.

### Direct inversion baseline

`stereocomplex.benchmarks.direct_inversion` implements pipeline A: joint
optimisation of optical parameters and board poses by minimising 2-D
reprojection error, using the inverse projection for every model
evaluation.  The interface is `fit_direct_model_from_observations`.

### Conditioning diagnostics

`stereocomplex.benchmarks.inverse_problem_diagnostics` provides
`compute_inverse_problem_diagnostics` to analyse the conditioning of the
direct inverse problem, returning singular values, Schur-complement ranks,
coupling norms, and parameter correlations.

## Current state (v0.5.3)

| Component | Status |
|---|---|
| Oracle builders (6 families) | Done |
| Inverse point→pixel projection | Done |
| ChArUco observation simulator | Done (with rejection sampling) |
| Direct inversion pipeline (A) | Done (cv2.solvePnP init + joint BA) |
| Image-based Zernike rayfield (B) | Done (BA from ChArUco corners) |
| Conditioning diagnostics | Done (Schur complement, pipeline-aware) |
| Notebook 08 (CMO oracle demo) | Wired (A+B), FAST mode bottleneck pending |
| Full 6-oracle sweep (pipeline B) | Done (6/6 correct) |
| Full 6-oracle sweep (pipeline A) | Partial (4/6, CMO fails on init) |

Both pipelines are implemented and tested (113 tests, 0 warnings).
Pipeline A converges on pinhole and Brown oracles with cv2.solvePnP
pose initialisation.  Pipeline B fits a Zernike rayfield from the same
ChArUco observations used by pipeline A.  The CMO oracle remains
challenging in FAST mode due to its narrow field of view (~5-10 corners
per frame even with a dense board) — this is a physical property of the
CMO architecture, not a software defect.

The conditioning diagnostics confirm that pipeline A has higher
(poorer) condition numbers due to optics-pose coupling, while pipeline B
eliminates pose parameters by absorbing them into the rayfield
measurement.  This is the central scientific claim of the study.

## Results — 6-oracle classification matrix

Pipeline B (oracle rayfield → ray-space model selection) correctly
classifies all six oracles.  Reproduced by `python examples/sweep_direct_vs_rayfield.py`
with `seed=42`.

| Oracle | Pipeline B winner | ΔBIC |
|---|---|--:|
| Central pinhole | `central_pinhole` | +54 |
| Central Brown-Conrady | `central_brown_conrady` | +896 |
| Inclined parallel plate | `pinhole_parallel_plate` | +58 328 |
| CMO shared-rig | `cmo_physical_shared` | +62 991 |
| Greenough (Brown × 2) | `central_brown_conrady` | +653 |
| **Uncatalogued Zernike** | **`zernike_compact`** | +1 540 |

Pipeline A (direct fit, all 6 oracles tested with analytic projection
on the expected-winner candidate):

| Oracle | RMS | Converged |
|---|---|:---:|
| Pinhole | 0.04 px | ✓ |
| Brown-Conrady | 0.06 px | ✓ |
| Inclined plate | 0.11 px | ✓ |
| CMO shared-rig | 1.63 px | ✗ |
| Greenough | 1.51 px | ✓ |
| Exotic Zernike | 0.52 px | ✓ |

### Key findings

**1. Pipeline B classifies all 6 oracles correctly** (~30 s total).
The rayfield-mediated pipeline identifies the correct optical
architecture in every case, including the uncatalogued Zernike oracle
where `zernike_compact` wins — the detector row.

**2. Pipeline A converges on 5/6 oracles; CMO fails on pose init.**  The
joint optical+pose optimisation converges on pinhole, Brown-Conrady,
inclined plate, Greenough, and exotic Zernike oracles (0.04–1.51 px).
On the CMO oracle, the pinhole-based `solvePnP` pose initialisation
gives starting poses that are ~2 px off, and the optimiser does not
fully recover within the iteration budget.  *With ground-truth poses*,
pipeline A converges to RMS = 0.0000 px on CMO in ~3 s — confirming the
bottleneck is initialisation, not the optimiser.

**3. Pipeline B is 10–100× faster** (3–12 s per oracle vs 96–712 s
for pipeline A).  Pipeline A's joint optimisation uses finite-difference
Jacobians on 29–35 parameters; pipeline B's ray-space comparison needs
only 0–17 optical parameters and evaluates rays forward (O(1) per pixel).

**4. Poses are eliminated from model selection in pipeline B.** The
rayfield absorbs the 6·n_poses nuisance parameters upstream, so the
model-selection stage compares only the optical degrees of freedom.
Pipeline A carries 12–48 pose parameters in its optimisation vector,
creating optics-pose coupling (coupling norm ≈ 0.32–0.69 on CMO).

### The rayfield is a geometric intermediate variable

Pipeline B separates *measurement* (Zernike fit from corners) from
*interpretation* (physical model comparison).  This is the central
architectural insight of StereoComplex and the reason the framework can
detect uncatalogued optics, compare competing hypotheses, and remain
interpretable under measurement noise.

### When to use which pipeline

| Criterion | Pipeline A (direct) | Pipeline B (rayfield) |
|---|---|---|
| Speed (model comparison) | Fast with analytic projection (~s) | Fast (~s per candidate) |
| Speed (single known model) | OK with good initialisation | Requires Zernike fit first |
| Model selection interpretability | Poses + optics coupled | Poses absent from comparison |
| Detection of uncatalogued optics | Possible but fragile | Built-in via Zernike fallback |
| Parameter recovery for known model | Maximum-likelihood (efficient) | Two-stage (Zernike → physical) |

**Recommendation:** use pipeline B when comparing competing optical
hypotheses or diagnosing unknown instruments.  Use pipeline A only when
fitting a single well-known model with maximum-likelihood efficiency.

## Recommended workflow

- Use **pipeline A** (direct fit) when the optical model is known and
  simple, and when an efficient maximum-likelihood estimate is desired.
- Use **pipeline B** (rayfield-mediated) when you want to **compare**
  competing physical hypotheses, **detect** structural mismatch, or
  **diagnose** whether the optics fall outside the known catalogue.
- The two pipelines are complementary, not competing.  The rayfield is a
  diagnostic instrument, not a replacement for direct fitting.

## Limitations

- The direct pipeline requires at least 20–30 visible corners per frame
  for stable joint optimisation.  The CMO oracle has a narrow field of
  view (~9°) requiring a dense board (1 mm squares) for sufficient coverage.
- The direct pipeline's generic inverse projection is impractically slow
  (~180 `least_squares` calls per evaluation).  Analytic `project_point`
  methods on physical models (currently implemented for CMO) make it
  competitive.  Other candidates (Brown, plate) still need analytic
  projection methods.
- Pixel-space BIC and ray-space BIC are not directly comparable (different
  residual units).  Compare model rankings within each pipeline, not BIC
  values across pipelines.
- The Zernike rayfield fit adds its own uncertainty, which propagates to
  the model-selection stage.  The sweep uses oracle rayfields to
  isolate the model-selection comparison; notebook 08 demonstrates the
  full ChArUco → Zernike → selection loop on the CMO oracle.

## See also

- [Identify My Optics](IDENTIFY_MY_OPTICS.md) — the model catalogue and interpretation guide
- [CMO Model Selection](CMO_MODEL_SELECTION.md) — the full 6-oracle classification matrix
- [Physical CMO Model](CMO_PHYSICAL_MODEL.md) — the CMO shared-rig model definition
- Notebook 07 (`examples/notebooks/07_model_selection_matrix.py`) — classification matrix
- Notebook 08 (`examples/notebooks/08_direct_vs_rayfield_inversion.py`) — direct vs rayfield comparison
