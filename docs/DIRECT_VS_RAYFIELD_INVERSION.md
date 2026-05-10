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

## Current state (v0.5.2)

| Component | Status |
|---|---|
| Oracle builders (6 families) | Done |
| Inverse point→pixel projection | Done |
| ChArUco observation simulator | Done |
| Direct inversion pipeline | Implemented, needs pose-initialisation tuning |
| Conditioning diagnostics | Done |
| Notebook 08 (CMO oracle demo) | Done (rayfield pipeline) |
| Full 6-oracle direct-vs-rayfield sweep | Pending (simulator coverage + direct-fit tuning) |

The rayfield-mediated pipeline (pipeline B) is fully operational and
correctly identifies the CMO physical model on a CMO oracle.  The direct
pipeline (pipeline A) is implemented but requires further work on the
observation simulator to produce sufficient corner coverage for robust
joint optimisation.

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
  for stable joint optimisation.  The current observation simulator
  produces sparse coverage for some oracles; this is a known issue.
- The conditioning diagnostics use finite-difference Jacobians, which are
  slow for large problems.  An analytic or automatic-differentiation
  approach would scale better.
- The rayfield fit from ChArUco observations (Zernike BA) is not yet
  integrated into notebook 08 — the notebook currently uses the oracle
  rayfield directly.  Closing the loop with a full image-based Zernike
  BA is the next natural step.
- Pixel-space BIC and ray-space BIC are not directly comparable (different
  residual units).  Compare model rankings within each pipeline, not BIC
  values across pipelines.

## See also

- [Identify My Optics](IDENTIFY_MY_OPTICS.md) — the model catalogue and interpretation guide
- [CMO Model Selection](CMO_MODEL_SELECTION.md) — the full 6-oracle classification matrix
- [Physical CMO Model](CMO_PHYSICAL_MODEL.md) — the CMO shared-rig model definition
- Notebook 07 (`examples/notebooks/07_model_selection_matrix.py`) — classification matrix
- Notebook 08 (`examples/notebooks/08_direct_vs_rayfield_inversion.py`) — direct vs rayfield comparison
