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
.venv/bin/python -m stereocomplex.cli eval-charuco-detection dataset/v0 --method <METHOD>
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

### 4b) `rayfield_tps` and `rayfield_tps_robust` (recommended in this repo)

The current default used in the examples/paper is `rayfield_tps_robust`:

- base homography `H`,
- TPS residual field,
- robustification by IRLS (Huber).

Compared to the grid backend, TPS is usually more stable when the residual field is only observed sparsely (AruCo corners).

### 4c) `hessian_barycentre` — subpixel refinement via the Hessian structure tensor

When ChArUco corners are missing or poorly localised (e.g. at extreme Z
positions, low contrast, or strong blur), a **photometric refinement** can
recover them using the local image structure.  StereoComplex implements the
**Pycaso method**: predict missing corners with Ray2D TPS, then refine with
the Hessian determinant + Otsu + barycentre.

#### Mathematics

The Hessian matrix of the image $I(x,y)$ is:

$$H = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}$$

where $I_{xx} = \partial^2 I / \partial x^2$, $I_{xy} = \partial^2 I / \partial x \partial y$,
$I_{yy} = \partial^2 I / \partial y^2$, computed via Sobel derivatives on a
Gaussian-blurred image ($\sigma = 9$ px).

The **corner response** is the absolute value of the Hessian determinant:

$$R(x,y) = |\det H| = |I_{xx} I_{yy} - I_{xy}^2|$$

This responds strongly to saddle-like image structures (corners) and is
invariant to the orientation of the corner.  Unlike the Harris measure
($\det H - \kappa \cdot \text{trace}^2(H)$), this simpler invariant is
used in Pycaso because the Otsu threshold isolates corner-like blobs
without needing to tune a sensitivity parameter $\kappa$.

#### Otsu thresholding

The response $R(x,y)$ is normalised to $[0, 255]$ and binarised with
Otsu's method, which automatically selects the threshold that minimises
intra-class variance:

$$\text{mask}(x,y) = \begin{cases} 1 & \text{if } R(x,y) \ge \tau_{\text{Otsu}} \\ 0 & \text{otherwise} \end{cases}$$

This yields a binary image where corner-like regions are white blobs.

#### Subpixel barycentre

For each missing corner, the algorithm:

1. **Predicts** the pixel position using **Ray2D TPS** fitted to the
   detected ArUco marker corners (or an affine fallback if markers are
   unavailable).  TPS handles non-linear lens distortion better than a
   pure affine or homography.

2. **Searches** a square window centred at the predicted position in the
   binary mask.

3. **Labels** connected components in the window (`cv2.connectedComponentsWithStats`).

4. **Selects** the central blob (closest to the predicted position), or
   the largest blob if `prefer_largest=True` (the original Pycaso behaviour).

5. **Computes the subpixel barycentre** via image moments:

   $$x_{\text{sub}} = x_0 + \frac{M_{10}}{M_{00}}, \qquad
     y_{\text{sub}} = y_0 + \frac{M_{01}}{M_{00}}$$

   where $M_{pq} = \sum_{(x,y) \in \text{blob}} x^p y^q \cdot \text{mask}(x,y)$
   and $(x_0, y_0)$ is the window origin.  `cv2.moments` computes these
   exactly on the binary component.

6. **Refines** a second time: the window is re-centred at the first
   barycentre, and steps 2–5 are repeated (two-pass refinement).

#### Integration with Ray2D TPS

The initial prediction uses Ray2D TPS rather than a simple affine mapping.
This is important for optics with strong distortion (e.g. microscopes,
endoscopes, laparoscopes) where an affine model cannot accurately predict
corner positions across the full field of view.  The TPS warp is fitted to
the ArUco marker corners (4 corners per detected marker, known 3‑D
positions) and maps any board coordinate to the image.

When marker detection fails (fewer than 4 markers), the algorithm falls
back to an affine fit on the detected ChArUco corners.

#### When to use

- **Missing corners:** the primary use case — fills undetected ChArUco
  corners (typically 0–50 missing out of 165 on Pycaso data at extreme Z).
- **Low-contrast targets:** when OpenCV corner refinement fails.
- **After Ray2D TPS:** TPS provides the initial prediction; Hessian
  provides the photometric refinement.
- **Not a replacement for OpenCV:** detected OpenCV corners are kept
  (they are already subpixel).  Only missing corners are completed.

#### Limitations

- The Hessian response is computed at a fixed scale ($\sigma = 9$ px).
  Corners at very different scales may be missed.
- Otsu thresholding is global; local contrast variations can cause
  false positives/negatives.
- The method is **heuristic** — it is validated by the downstream
  rayfield residuals, not by independent ground truth.
- For corners near image borders, the search window may be truncated,
  producing `NaN` (filled by affine fallback).

### 5) `kfield` (a “local K” field approximated by smoothed affines)

This method was an intermediate step: the idea is to replace a global `K` by a spatially varying field, under a low-frequency assumption.

Note: in the current code, `kfield` does **not** interpolate a pinhole matrix $K$ in the strict sense. Instead, it builds
a smoothed field of local **affine** (first-order) models obtained by **linearizing** the plane→image mapping.

#### Linearization (Jacobian)

Consider an unknown (potentially complex) mapping between the board plane and the image:

- `u = u(x,y)`
- `v = v(x,y)`

Around a reference point $(x_q, y_q)$, we can write a first-order Taylor expansion:

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

- choose an anchor grid in board coordinates $(x,y)$;
- at each anchor, fit a local affine model by weighted least squares (nearest ArUco neighbors):
  ```{math}
  u(x,y)=a_0 + a_1 x + a_2 y,\quad v(x,y)=b_0 + b_1 x + b_2 y
  ```
  where `a1,a2,b1,b2` estimate the local Jacobian $(\partial u/\partial x, \partial u/\partial y, \partial v/\partial x, \partial v/\partial y)$.
- smooth each parameter $(a_0,a_1,a_2,b_0,b_1,b_2)$ on the grid (Gaussian);
- for a query point $(x,y)$, bilinearly interpolate these parameters and apply the affine mapping.

Why this is not sufficient:

- a local affine model does not capture projective effects (and even less distortion) over the full board;
- directly interpolating a matrix $K$ is not “geometrically stable” (constraints on $f_x,f_y$, etc.).

In practice, `rayfield` (homography + smoothed residual field) matches the “low-frequency” intuition while remaining numerically stable.

## Assumptions per method (what it “assumes”)

Summary of dependencies (as of the current code):

- `charuco`: does not require `K`/distortion, but depends on OpenCV heuristics.
- `homography`: does not require `K`/distortion; assumes a global homography explains the board image well.
- `tps`: does not require `K`/distortion; assumes a smooth 2D warp (thin-plate spline) and can extrapolate unstably if underconstrained.
- `pnp`: **requires** an optical model (pinhole + distortion) and its parameters (or a prior step estimating them).
- `rayfield`: does not require `K`/distortion; assumes a low-frequency planar warp and uses only correspondences (Aruco) + regularization.
- `rayfield_tps`: a `rayfield` variant where the residual is reconstructed by regularized TPS (instead of a bilinear grid + Laplacian).
- `rayfield_tps_robust`: TPS residual + robust loss (recommended default).
- `hessian_barycentre`: does not require `K`/distortion; predicts missing corners via Ray2D TPS (or affine fallback), refines with $|\det H|$ + Otsu + subpixel blob barycentre. Heuristic; validated by downstream rayfield residuals.

## Photometric refinements (CLI `--refine`)

Refinements based on structure tensor/gradients exist (`tensor`, `lines`, `lsq`, `noble`), but on the current datasets they often moved corners toward a photometric optimum that does not match the GT geometric center.
They should be considered as ablations/experiments rather than the recommended method.

## Current recommendation

- If the optics are well approximated by pinhole + distortion: prefer `pnp`.
- If the optics are complex/non-central: prefer `rayfield_tps_robust` (low-frequency assumption) and increase regularization if needed.
- **If corners are missing** (incomplete ChArUco detection at some poses):
  use the `hessian_barycentre` method after Ray2D TPS prediction.  This is
  the workflow demonstrated in Notebook 09 on real Pycaso CMO data.

## Paper comparison (reproducible script)

The manuscript includes an automatically generated table (methods vs errors). To regenerate it:

```bash
.venv/bin/python paper/experiments/compare_charuco_methods.py dataset/v0_png --splits train
bash paper/build_pdflatex.sh
```

## Worked example (raw OpenCV vs ray-field + plots)

See `docs/RAYFIELD_WORKED_EXAMPLE.md` (includes a detailed explanation of why a global homography + a smoothed residual field can correct part of the aberrations/distortions on the board plane).
