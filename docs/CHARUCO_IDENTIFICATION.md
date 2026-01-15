# ChArUco: 2D identification strategy (baseline)

Goal: obtain ChArUco 2D corner positions that are as stable as possible (sub-pixel) in order to quantify the impact of blur, compression, and aberrations, and to prepare calibration/reconstruction stages.

The project deliberately separates:

- a **geometric prior** (planar board + ArUco/ChArUco correspondences);
- an **image observation** (blur, compression, contrast, etc.);
- methods that rely on a **parametric model** (pinhole + distortion) or a **non-parametric model** (smoothed field).

## Error measurement

On synthetic datasets, the error is computed against the ground truth stored in `gt_charuco_corners.npz`:

- matching by `corner_id` (stable ID);
- per-view metrics (left/right): RMS, p50, p95, max, bias dx/dy.

Command:

```bash
PYTHONPATH=src .venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/v0 --method <METHOD>
```

## Pixel-center convention (important)

The project uses a “pixel centers at integer coordinates” convention (see `docs/CONVENTIONS.md`).
OpenCV often reports corners in a convention shifted by 0.5 px; the evaluation code compensates for that shift for `--method charuco`.

## Available methods (CLI `--method`)

### 1) `charuco` (direct OpenCV)

- OpenCV ArUco pipeline → ChArUco interpolation (inner chessboard corners).
- Pro: simple, no camera model assumption.
- Limitation: accuracy is often limited (sensitivity to blur/compression + conventions + internal heuristics).

### 2) `homography` (2nd-pass planar geometry)

- Detects ArUco corners, estimates a global homography (RANSAC), then projects all ChArUco corners.
- Works well when the image is well explained by a “simple” planar projective mapping.
- Limitation: degrades in the presence of out-of-model distortions (e.g. strong radial distortion).

### 3) `pnp` (2nd-pass parametric K + distortion)

- Uses `meta.json` (pitch/crop/resize + `f_um`) to build `K` and distortion coefficients, then:
  - runs `solvePnPRansac` on ArUco 3D→2D corners,
  - uses `projectPoints` to predict ChArUco corners.
- Pro: robust when the optics can be modeled as pinhole + (Brown) distortion.
- Limitation: not applicable / biased for non-pinhole systems (e.g. non-central microscope/CMO models).

**Important note (focal length)**

In the current synthetic dataset, `f_um` is known because it is generated and stored in `meta.json` (`sim_params.f_um`).
Therefore, method `pnp` uses it as a **known** parameter to isolate the “point identification” effect.

In real data, `f_um` (and more generally `K` and distortion) are not known a priori:

- either they are estimated by a classical multi-view calibration (e.g. Zhang) before running `pnp`,
- or they are part of an auto-calibration problem (latent variables to estimate),
- or one avoids the pinhole assumption and uses a non-parametric method (e.g. `rayfield`).

### 4) `rayfield` (2nd-pass non-parametric “smoothed field” on the board plane)

Goal: replace a pinhole model by a weaker assumption: the mapping from the board plane to the image is **low-frequency**.

Implementation (plane-only):

- global homography `H` (RANSAC) as a stable baseline;
- residual field `r(x,y)` estimated on a grid (bilinear), regularized with a smoothing term (Laplacian) and robust to outliers (Huber);
- prediction: `u(x,y) = H(x,y) + r(x,y)`.

Pros:

- does not depend on a pinhole optical model;
- captures slow variations (complex aberrations) while remaining stable.

Limitation:

- this “ray-field” is **restricted to the plane** (a 2D warp); for a full 3D per-pixel ray field, calibration across multiple poses/planes is required.

### 5) `kfield` (a “local K” field approximated by smoothed affines)

This method was an intermediate step: the idea is to replace a global `K` by a spatially varying field, under a low-frequency assumption.

Note: in the current code, `kfield` does **not** interpolate a pinhole matrix \(K\) in the strict sense. Instead, it builds
a smoothed field of local **affine** (first-order) models obtained by **linearizing** the plane→image mapping.

#### Linearization (Jacobian)

Consider an unknown (potentially complex) mapping between the board plane and the image:

- `u = u(x,y)`
- `v = v(x,y)`

Around a reference point \((x_q, y_q)\), we can write a first-order Taylor expansion:

- `u(x,y) ≈ u_q + (∂u/∂x)_q · (x-x_q) + (∂u/∂y)_q · (y-y_q)`
- `v(x,y) ≈ v_q + (∂v/∂x)_q · (x-x_q) + (∂v/∂y)_q · (y-y_q)`

The local **Jacobian** (the linear part) is:

```
J(x_q,y_q) = [[∂u/∂x, ∂u/∂y],
             [∂v/∂x, ∂v/∂y]]  (évalué en (x_q,y_q))
```

The `kfield` idea is to estimate this local Jacobian (and offset) from the ArUco correspondences available in the image,
then smooth/interpolate it to obtain a low-frequency approximation.

#### Construction (what the code does)

- choose an anchor grid in board coordinates \((x,y)\);
- at each anchor, fit a local affine model by weighted least squares (nearest ArUco neighbors):
  ```{math}
  u(x,y)=a_0 + a_1 x + a_2 y,\quad v(x,y)=b_0 + b_1 x + b_2 y
  ```
  where `a1,a2,b1,b2` estimate the local Jacobian \((\partial u/\partial x, \partial u/\partial y, \partial v/\partial x, \partial v/\partial y)\).
- smooth each parameter \((a_0,a_1,a_2,b_0,b_1,b_2)\) on the grid (Gaussian);
- for a query point \((x,y)\), bilinearly interpolate these parameters and apply the affine mapping.

Why this is not sufficient:

- a local affine model does not capture projective effects (and even less distortion) over the full board;
- directly interpolating a matrix \(K\) is not “geometrically stable” (constraints on \(f_x,f_y\), etc.).

In practice, `rayfield` (homography + smoothed residual field) matches the “low-frequency” intuition while remaining numerically stable.

## Assumptions per method (what it “assumes”)

Summary of dependencies (as of the current code):

- `charuco`: does not require `K`/distortion, but depends on OpenCV heuristics.
- `homography`: does not require `K`/distortion; assumes a global homography explains the board image well.
- `tps`: does not require `K`/distortion; assumes a smooth 2D warp (thin-plate spline) and can extrapolate unstably if underconstrained.
- `pnp`: **requires** an optical model (pinhole + distortion) and its parameters (or a prior step estimating them).
- `rayfield`: does not require `K`/distortion; assumes a low-frequency planar warp and uses only correspondences (Aruco) + regularization.
- `rayfield_tps`: a `rayfield` variant where the residual is reconstructed by regularized TPS (instead of a bilinear grid + Laplacian).

## Photometric refinements (CLI `--refine`)

Refinements based on structure tensor/gradients exist (`tensor`, `lines`, `lsq`, `noble`), but on the current datasets they often moved corners toward a photometric optimum that does not match the GT geometric center.
They should be considered as ablations/experiments rather than the recommended method.

## Current recommendation

- If the optics are well approximated by pinhole + distortion: prefer `pnp`.
- If the optics are complex/non-central: prefer `rayfield` (low-frequency assumption) and increase regularization if needed.

## Paper comparison (reproducible script)

The manuscript includes an automatically generated table (methods vs errors). To regenerate it:

```bash
PYTHONPATH=src .venv/bin/python paper/experiments/compare_charuco_methods.py dataset/v0_png --splits train
bash paper/build_pdflatex.sh
```

## Worked example (raw OpenCV vs ray-field + plots)

See `docs/RAYFIELD_WORKED_EXAMPLE.md` (includes a detailed explanation of why a global homography + a smoothed residual field can correct part of the aberrations/distortions on the board plane).
