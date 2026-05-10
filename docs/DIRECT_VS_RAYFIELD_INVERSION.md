# Rayfield mediation: separating measurement from optical interpretation

## Why this page exists

StereoComplex follows a two-stage philosophy:

```text
measure a generic rayfield first  →  then explain the optics
```

This page is **not** a claim that rayfield-mediated inversion universally
replaces direct ChArUco fitting.  The two strategies answer different
questions:

- **Direct fitting** is the natural route when the optical family is
  already known: choose a model, jointly optimise optical parameters and
  board poses, and obtain a maximum-likelihood estimate in pixel space.
- **Rayfield mediation** addresses a different problem: before fitting
  one known model, we often need to decide *which* optical family is
  plausible.  For that purpose, it is useful to first estimate a generic
  rayfield $\widehat{\mathcal{R}}(u,v) = (\widehat{O}, \widehat{d})$,
  then compare physical hypotheses in a common ray-space.

The rayfield is therefore an **intermediate observable**: it separates
measurement from interpretation.

This page explains the two inverse problems, documents the current
experimental evidence, and provides practical guidance on when to use
each strategy.

## The two inverse problems

### Direct fitting — an estimator for a known model

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
pose can compensate for an error in the optical model.  Direct fitting is
the right tool when the model family is known and a statistically
efficient estimate is desired.

### Rayfield mediation — a diagnostic layer for model selection

First, the instrument response is measured as a generic pixel-to-line map
$\widehat{\mathcal{R}}(u,v) = (\widehat{O}(u,v), \widehat{d}(u,v))$ from
the same ChArUco observations (via Zernike bundle adjustment).  Then,
physical hypotheses are compared in a common ray space:

```{math}
\min_\theta
\sum_k
\left\|
\widehat{\mathcal{R}}(u_k,v_k)
-
\mathcal{R}_\theta(u_k,v_k)
\right\|^2_{\text{line}}
```

**Board poses do not appear in this second stage** — they have already
been absorbed into the rayfield measurement.  This is the central
architectural insight: model comparison happens after nuisance pose
parameters have been removed.

## Nuisance parameters and conditioning

In direct fitting, the joint parameter vector includes both optical
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

When $\|J_\theta^T J_\eta\|$ is large, poses and optics are strongly coupled.
Pipeline B has **no pose parameters** in the second stage, so its
information matrix is simply $J_\theta^T J_\theta$ — the rayfield has
already absorbed the pose degrees of freedom.

Measured on the CMO oracle: pipeline A has coupling norm ≈ 0.32–0.69
between optics and poses; pipeline B has coupling norm = 0 (0 pose
parameters in the model-selection stage).

## What the current experiments show

Three experiments at different levels of completeness:

| Experiment | What it demonstrates | Result |
|---|---|---|
| **Notebook 08 FAST** | Full ChArUco → Zernike → selection loop on CMO | CMO correctly identified (~56 s) |
| **6-oracle ray-space sweep** | Model selection from oracle rayfields | 6/6 correct (pinhole through exotic) |
| **Direct expected-model fit** | Direct fitting of the correct model on each oracle | 6/6 converge when well-initialised |

### Ray-space model selection (6-oracle sweep)

Pipeline B correctly classifies all six oracle families from their
rayfields (~30 s total):

| Oracle | Winner | ΔBIC |
|---|---|--:|
| Central pinhole | `central_pinhole` | +54 |
| Central Brown-Conrady | `central_brown_conrady` | +896 |
| Inclined parallel plate | `pinhole_parallel_plate` | +58 328 |
| CMO shared-rig | `cmo_physical_shared` | +62 991 |
| Greenough (Brown × 2) | `central_brown_conrady` | +653 |
| **Uncatalogued Zernike** | **`zernike_compact`** | +1 540 |

This sweep uses oracle rayfields to isolate the model-selection stage.
The full ChArUco → Zernike → selection loop is demonstrated on the CMO
oracle in notebook 08.

### Direct fitting baseline (6-oracle sweep)

Pipeline A fits the expected-winner candidate directly to noisy ChArUco
observations (0.05 px noise).  All six oracles converge when the model is
well-initialised:

| Oracle | RMS | Notes |
|---|---|:---|
| Pinhole | 0.04 px | ✓ solvePnP init |
| Brown-Conrady | 0.06 px | ✓ solvePnP init |
| Inclined plate | 0.11 px | ✓ solvePnP init |
| CMO shared-rig | 0.05 px | ✓ with truth poses (solvePnP pinhole init fails) |
| Greenough | 1.51 px | ✓ solvePnP init |
| Exotic Zernike | 0.52 px | ✓ solvePnP init |

CMO is the only oracle where solvePnP initialisation fails — the pinhole
model is ~2 px off for CMO optics.  With ground-truth poses, CMO
converges to RMS = 0.047 px in 2 s.

### End-to-end ChArUco → Zernike → selection (notebook 08)

Notebook 08 demonstrates the full pipeline B on a CMO oracle: ChArUco
corner simulation → Zernike bundle adjustment from observations →
ray-space model selection.  CMO is correctly identified.  The Zernike BA
is the computational bottleneck (~50 s for one oracle), which is why the
6-oracle sweep above uses oracle rayfields.

## When to use which strategy

| Criterion | Direct fitting | Rayfield mediation |
|---|---|---|
| Model family is known | Best choice (efficient estimate) | Possible but two-stage |
| Model family is unknown | Requires multiple fits, fragile | Best choice (diagnostic layer) |
| Comparing CMO vs plate vs Brown | Coupled with poses | Natural ray-space comparison |
| Detecting non-catalogued optics | Not natural | Built-in via Zernike fallback |
| Maximum-likelihood efficiency | Best if model is correct | Two-stage (Zernike → physical) |

**Recommendation:** use direct fitting when the optical family is known
and a statistically efficient estimate is desired.  Use rayfield
mediation to diagnose which family is plausible, compare competing
hypotheses, or detect uncatalogued optics.

## Infrastructure

The modules in `stereocomplex.benchmarks` support both strategies:

- `model_selection_oracles` — six synthetic stereo rayfield pairs
- `charuco_observation_simulator` — synthetic ChArUco corners from rayfields
- `rayfield_projection` — inverse point-to-pixel projection (generic + analytic)
- `direct_inversion` — joint optical+pose optimisation (pipeline A)
- `rayfield_from_observations` — Zernike BA from ChArUco corners (pipeline B, stage 1)
- `inverse_problem_diagnostics` — Schur complement, conditioning, coupling norms

Analytic `project_point` methods exist on `CentralPinholeModel`,
`CentralBrownConradyModel`, and `CMOPhysicalChannelModel`, making direct
fitting fast (~2–10 s per oracle).

## Limitations

- The Zernike BA from observations is the computational bottleneck in
  pipeline B (~1–2 min per oracle).  The 6-oracle sweep uses oracle
  rayfields to isolate the model-selection comparison.
- Pixel-space BIC and ray-space BIC are not directly comparable (different
  residual units).  Compare model rankings within each pipeline, not BIC
  values across pipelines.
- The direct fitting baseline fits only the expected-winner candidate
  (not a full multi-candidate model selection).  A symmetric comparison
  where both pipelines perform full model selection on the same data is
  future work.
- The CMO oracle has a narrow field of view (~9°), requiring a dense
  board (1 mm squares) for sufficient corner coverage.

## See also

- [Identify My Optics](IDENTIFY_MY_OPTICS.md) — the model catalogue and interpretation guide
- [CMO Model Selection](CMO_MODEL_SELECTION.md) — the full 6-oracle classification matrix
- [Physical CMO Model](CMO_PHYSICAL_MODEL.md) — the CMO shared-rig model definition
- Notebook 07 (`examples/notebooks/07_model_selection_matrix.py`) — classification matrix
- Notebook 08 (`examples/notebooks/08_direct_vs_rayfield_inversion.py`) — end-to-end CMO demo
