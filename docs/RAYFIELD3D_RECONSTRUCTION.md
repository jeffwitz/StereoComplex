# 3D ray-field (central) on GT data

This page introduces a first “ray-based 3D” building block targeting optical systems **more complex than a global pinhole model** (e.g., CMO), while deliberately starting validation on **synthetic pinhole** data to provide a clear “oracle” reference.

In this chapter, we start from **GT correspondences** (already perfect) and compare the 3D models below. The ChArUco *denoising* pipeline (2D ray-field) and its impact on OpenCV stereo calibration are fully described in the stereo documentation: [Stereo 3D reconstruction (OpenCV)](STEREO_RECONSTRUCTION.md).

- an “oracle” reconstruction using **pinhole + Brown distortion** (exact synthesis parameters),
- a reconstruction using a **central 3D ray-field** represented with a **Zernike basis**.

## Conventions and data

We use dataset v0 (see `DATASET_SPEC.md`). For a given scene, we load `gt_points.npz` (or `gt_charuco_corners.npz`):

- $P_i = (X_i, Y_i, Z_i)^\top$: 3D coordinates (mm) in the left-camera frame,
- $p^L_i = (u^L_i, v^L_i)^\top$ and $p^R_i = (u^R_i, v^R_i)^\top$: distorted pixel projections (left/right),
- baseline $B$ (mm) from `meta.json` → `sim_params.baseline_mm`.

We adopt the synthetic convention: the left camera center is $C_L=(0,0,0)^\top$ and the right camera center expressed in the left frame is:

```{math}
C_R = (B,0,0)^\top.
```

## Central 3D ray-field model

A pixel $p=(u,v)$ defines a 3D **ray**:

```{math}
\ell_p(t) = C + t\,\hat d(u,v), \quad t\ge 0
```

where:

- $C$ is a **constant** origin (central model),
- $\hat d(u,v)\in\mathbb{R}^3$ is a unit direction.

We parametrize direction via “normalized” coordinates:

```{math}
\tilde d(u,v) = \begin{bmatrix} x(u,v) \\ y(u,v) \\ 1 \end{bmatrix},
\qquad
\hat d(u,v)=\frac{\tilde d(u,v)}{\lVert\tilde d(u,v)\rVert}.
```

The learning problem is therefore to estimate the two scalar fields $x(u,v)$ and $y(u,v)$.

## Zernike basis (unit disk)

We map the image plane to the unit disk using:

```{math}
\tilde u = \frac{u-u_0}{R},\qquad \tilde v = \frac{v-v_0}{R},
```

where $u_0,v_0$ are the image center coordinates and $R$ is a radius covering the full image (circumscribed circle).

We use real Zernike polynomials $Z_k(\rho,\theta)$ (defined for $\rho\in[0,1]$) and approximate:

```{math}
x(u,v) = \sum_{k=1}^{K} a_k Z_k(\tilde u,\tilde v),
\qquad
y(u,v) = \sum_{k=1}^{K} b_k Z_k(\tilde u,\tilde v).
```

In the implementation (`CentralRayFieldZernike`), $K$ is set by the maximum radial order `nmax` (modes up to $n\le n_{\max}$).

## GT fit (ridge / Tikhonov regression)

With GT data, each 3D point $P_i$ lies on the ray defined by its pixel. In normalized coordinates:

```{math}
x_i = \frac{X_i}{Z_i},\qquad y_i = \frac{Y_i}{Z_i}.
```

We build a design matrix $A\in\mathbb{R}^{N\times K}$ with $A_{ik}=Z_k(\tilde u_i,\tilde v_i)$, and estimate coefficients with **ridge** regression (also called $L^2$ regularization or **Tikhonov**):

```{math}
\hat a = \arg\min_a\ \lVert Aa-x\rVert^2 + \lambda\lVert a\rVert^2,
\qquad
\hat b = \arg\min_b\ \lVert Ab-y\rVert^2 + \lambda\lVert b\rVert^2.
```

This is an **MVP** (*minimum viable prototype*): it yields a compact ray-field without non-linear optimization.

## Triangulation and metrics

For a pair $(p^L_i, p^R_i)$ we obtain two rays:

```{math}
\ell^L_i(t)= C_L + t\,\hat d_L(p^L_i),
\qquad
\ell^R_i(s)= C_R + s\,\hat d_R(p^R_i).
```

We reconstruct $\hat P_i$ using midpoint triangulation (midpoint of the common perpendicular segment). We report:

- **3D error**: $e_i = \lVert \hat P_i - P_i\rVert$ (mm),
- **skew-ray distance**: $d^{\mathrm{skew}}_i = \mathrm{dist}(\ell^L_i,\ell^R_i)$ (mm), i.e., the length of the common perpendicular segment.

## Pinhole “oracle” baseline (reference)

On a synthetic pinhole dataset, we know the exact parameters:

- focal length $f$ (via `sim_params.f_um`),
- Brown distortion (via `sim_params.distortion_left/right`),
- pixel pitch (via `meta.json`).

We can therefore map a distorted pixel to an undistorted ray as:

1. pixel $(u,v)$ → sensor coordinates $(x_{\mu m},y_{\mu m})$,
2. distorted normalization: $x_d=x_{\mu m}/f_{\mu m}$, $y_d=y_{\mu m}/f_{\mu m}$,
3. Brown inversion: $(x,y)=\mathrm{undistort}(x_d,y_d)$,
4. direction: $\hat d = \mathrm{normalize}([x,y,1])$.

This is not a “fit”: it is an **oracle** (expected lower bound on this dataset).

## Full GT example and comparison

Command:

```bash
.venv/bin/python paper/experiments/compare_pinhole_vs_rayfield3d_gt.py \
  --scene dataset/v0_png/train/scene_0000 \
  --gt gt_points.npz \
  --nmax 12 \
  --lam 1e-3
```

### Metrics (mm, px, %)

We report:

- triangulation error in mm: $e_i = \lVert \hat P_i - P_i\rVert$,
- relative error (order-of-magnitude): $100\,e_i / \bar Z$ (%) where $\bar Z$ is mean depth,
- reprojection error in pixels (left/right), by reprojecting $\hat P_i$ through the **GT Brown pinhole model** and comparing to GT $(u,v)$.

Outputs (summary, order-of-magnitude):

```{table} 3D comparison on GT (pinhole oracle vs central 3D ray-field).
:name: tab-rayfield3d-gt-summary

| 3D method | Triangulation RMS (mm) | Triangulation RMS (% depth) | Reproj RMS L/R (px) | Skew RMS (mm) |
|---|---:|---:|---:|---:|
| Pinhole oracle (GT params) | $\approx 1\times 10^{-4}$ | $\approx 1\times 10^{-5}$ | $\approx 5\times 10^{-6}$ | $\approx 1\times 10^{-5}$ |
| Central 3D ray-field (Zernike) | $\approx 3.2\times 10^{-1}$ | $\approx 2.4\times 10^{-2}$ | $\approx 4\times 10^{-2}$ | $\approx 2.8\times 10^{-2}$ |
```

Quick reading:

- On **pinhole** data, the pinhole oracle is nearly perfect (as expected).
- The central 3D ray-field is a compact approximation: its performance depends strongly on `nmax` (capacity) and $\lambda$ (smoothness). It mainly serves as a stepping stone towards future “complex optics” models.

## Code references

- Zernike basis (real modes + design matrix): `src/stereocomplex/core/model_compact/zernike.py`
- Central model `CentralRayFieldZernike`: `src/stereocomplex/core/model_compact/central_rayfield.py`
- Pinhole-oracle vs 3D ray-field GT comparison: `paper/experiments/compare_pinhole_vs_rayfield3d_gt.py`

## From images: detection + 2D ray-field, then reconstruction

This section connects the 3D ray-field chapter to the 2D identification pipeline:

1. OpenCV ChArUco detection on images (measured pixels),
2. center correction using the **2D ray-field** (`rayfield_tps_robust`),
3. 3D reconstruction by triangulation, with two 3D methods:
   - **pinhole oracle**: rays by Brown inversion using GT synthesis parameters,
   - **central 3D ray-field**: fit Zernike on $(u,v)\leftrightarrow P$ (GT) then triangulate.

### Script

```bash
.venv/bin/python paper/experiments/compare_3d_from_images_rayfield2d.py \
  dataset/v0_png \
  --split train --scene scene_0000 \
  --tps-lam 10 --tps-huber 3 --tps-iters 3 \
  --nmax 12 --lam3d 1e-3
```

The script writes a JSON metrics file (default: `paper/tables/3d_from_images_rayfield2d.json`) and prints the same content to `stdout`.

### Results (example)

The table below illustrates a run on `scene_0000` (5 frames). On these synthetic images, the 2D ray-field correction substantially reduces 2D pixel error, and triangulation improves mechanically for both 3D reconstructions.

```{table} 3D reconstruction from images (OpenCV raw vs 2D ray-field), with two 3D methods.
:name: tab-rayfield3d-from-images

| 2D method | 2D RMS L/R (px) | Pinhole oracle: 3D RMS (mm) | 3D ray-field: 3D RMS (mm) |
|---|---:|---:|---:|
| OpenCV raw | $\approx 0.38 / 0.36$ | $\approx 3.82$ | $\approx 3.82$ |
| 2D ray-field (`rayfield_tps_robust`) | $\approx 0.23 / 0.14$ | $\approx 1.28$ | $\approx 1.33$ |
```

Note: the “3D ray-field” used here is a central prototype (constant origin) and the fit uses GT 3D correspondences to start from a clean baseline. The longer-term goal is to replace this GT-assisted fit by a full ray-based calibration (multi-poses, non-central optics, etc.).

## Ray-based calibration (no GT 3D): point↔ray bundle adjustment

This section replaces the “GT-assisted 3D fit” by a full calibration from:

- multi-pose board correspondences $(u,v)\leftrightarrow (X,Y,0)$,
- a compact central ray-field $d(u,v)$ (Zernike),
- per-frame board poses $(R_i,t_i)$.

### Geometric residual

For an observation $(u_{ij},v_{ij})$ of board point $P_j$ in image $i$:

- camera-frame point: $P^{\mathrm{cam}}_{ij}=R_i P_j + t_i$,
- unit direction: $\hat d_{ij}=\hat d(u_{ij},v_{ij})$.

The ray is $\ell_{ij}(t)=C+t\hat d_{ij}$ (here $C$ is constant, and we fix $C=(0,0,0)^\top$).

We minimize the point↔ray distance using the vector residual:

```{math}
r_{ij} = (I - \hat d_{ij}\hat d_{ij}^\top)\,P^{\mathrm{cam}}_{ij}.
```

This residual is minimized with a robust loss (Huber) and an $L^2$ regularization on Zernike coefficients.

### Joint optimization (stereo)

In the stereo version, we optimize **simultaneously**:

- Zernike coefficients of $d_L(u,v)$ and $d_R(u,v)$,
- a single rigid pose of the rig $(R_{RL},t_{RL})$ such that $P_R = R_{RL}P_L+t_{RL}$,
- board poses per image in the left-camera frame $(R_i,t_i)$.

We solve with `scipy.optimize.least_squares` (robust Gauss-Newton/LM) using Huber loss and $L^2$ coefficient regularization.

### Script (images → 2D ray-field → 3D ray-field BA → stereo)

```bash
.venv/bin/python paper/experiments/calibrate_central_rayfield3d_from_images.py \
  dataset/v0_png \
  --split train --scene scene_0000 \
  --max-frames 5 \
  --method2d rayfield_tps_robust \
  --tps-lam 10 --tps-huber 3 --tps-iters 3 \
  --nmax 8 --lam-coeff 1e-3 --outer-iters 3 --fscale-mm 1.0
```

Output: JSON (default: `paper/tables/rayfield3d_ba_from_images.json`) with:

- estimated baseline (mm + px-equivalent at mean depth),
- 3D errors (mm and % depth),
- reprojection errors (px), and skew-ray distances (mm),
- optimization diagnostics (cost per iteration),
- an `opencv_pinhole_calib` section: OpenCV pinhole calibration (intrinsics + distortion + stereo rig) on the **same 2D points**.

### Results (example)

On `scene_0000` (5 frames), the “pinhole oracle” remains a lower bound (pinhole + GT Brown). The central 3D ray-field BA (Zernike, central model) is calibrated **without solvePnP** and **without a known** $K$: initial board poses are obtained from homographies (Zhang-style) only as an *initialization*, and the solver then directly optimizes the point↔ray cost (robust Gauss-Newton via SciPy).

```{table} Central ray-based calibration from images: comparison to the pinhole oracle (example).
:name: tab-rayfield3d-ba-example

| 3D method (same 2D points) | Baseline abs. err. (mm) | Baseline abs. err. (px) | 3D RMS (mm) | Reproj RMS L/R (px) |
|---|---:|---:|---:|---:|
| Pinhole oracle (GT params) | $0$ | $0$ | $\approx 1.28$ | $\approx 0.20 / 0.15$ |
| OpenCV pinhole calibrated (images, non-GT) | $\approx 0.32$ | $\approx 0.29$ | $\approx 14.48$ | $\approx 3.02 / 2.77$ |
| 3D ray-field BA (central, Zernike) | $\approx 0.21$ | $\approx 0.19$ | $\approx 1.55$ | $\approx 1.36 / 1.33$ |
```

Note: for the “3D ray-field BA” row, the 3D RMS and reprojections are computed **after** a fixed-origin similarity alignment (rotation + scale, no translation) between the reconstruction and the GT reference. Without this step, errors “in the GT frame” become arbitrarily large because the point↔ray cost does not, by itself, fix the global frame choice (gauge).

### Discussion: (i) baseline, (ii) reprojection, (iii) triangulation

The table highlights three important points:

1. **Baseline is now better with the ray-field.** Here, ray-based calibration yields a smaller baseline error than OpenCV pinhole calibration (mm and px-equivalent). This is consistent with the fact that the ray-based optimization is constrained by a single rig $(R_{RL},t_{RL})$ and a geometric point↔ray cost over all observations, which limits the “intrinsics ↔ distortion ↔ extrinsics” compensations typical of pinhole calibration on planar targets.

2. **Baseline: norm vs direction.** A small error on $\lVert C_R\rVert$ does not guarantee a perfect direction. In this example, both methods produce a slightly off-axis baseline (non-zero $y,z$ components), so the script also reports the angle to the $x$ axis and the off-axis norm (see the script JSON).

   For example on `scene_0000`: the angle is about $3.38^\circ$ (ray-field) versus $2.62^\circ$ (OpenCV), despite a smaller norm error on the ray-field side. This illustrates why both baseline norm and baseline direction matter.

3. **Why “non-GT pinhole” can have a decent baseline but poor reprojection/3D vs GT.** OpenCV minimizes its own image error, but the errors reported here are measured **against the GT model** (synthetic pinhole + Brown). A pinhole calibration can thus be self-consistent (low `mono_rms_*`) while still far from the GT parameters (high GT reprojection error), especially due to identifiability couplings on planar targets.

Practical takeaway:

- in robotics (rectification, dense stereo), baseline accuracy and epipolar coherence often dominate matching success;
- in metrology (stereo-DIC), ray-based calibration can stabilize stereo geometry when a global pinhole becomes only an approximation.

### Discussion: why “3D ray-field BA” needs an “aligned” comparison

The point↔ray cost

```{math}
r_{ij}=(I-\hat d_{ij}\hat d_{ij}^\top)\,P^{\mathrm{cam}}_{ij}
```

is invariant to a global Euclidean transformation of the camera frame (rotation) and, to some extent, to a scale factor coupled to depth (limited identifiability on planar targets). In other words: **the calibration is defined up to a gauge**, while GT enforces an absolute reference frame (left camera, $x$ axis aligned with the baseline, etc.). To avoid conflating “bad geometry” with “different frame”, we report:

- an “aligned” 3D RMS (rotation + scale),
- an “aligned” reprojection RMS (GT projection after alignment).

These metrics reflect practical reconstruction interest (coherence and stability), while baseline (mm and px-equivalent) remains a directly interpretable stereo-vision quantity.

## Post-hoc pinhole identification from the ray-field 3D reconstruction

To assess whether a ray-based 3D reconstruction can *help* identify a conventional pinhole model, the script also performs a post-hoc pinhole fit:

- input: reconstructed 3D points (left-camera frame) from the ray-field 3D model, and the corresponding observed pixels,
- model: Brown pinhole $(K, d)$ per camera,
- solver: `scipy.optimize.least_squares` (Huber loss), with an additional per-camera global rotation (gauge correction).

This produces a new JSON block:

- `pinhole_from_rayfield3d` (estimated $K,d$ + reprojection RMS on the same correspondences),
- `pinhole_vs_gt` (relative parameter errors vs GT for synthetic datasets).

On the same example (`scene_0000`, 5 frames), the distortion-field error relative to GT decreases compared to the direct OpenCV pinhole calibration:

```{list-table} Pinhole parameter identification vs GT (example; lower is better).
:name: tab-rayfield3d-posthoc-pinhole
:header-rows: 1

* - Method
  - dist err L (% of GT)
  - dist err R (% of GT)
  - fx err L (%)
  - fx err R (%)
* - OpenCV pinhole calib (images → pinhole)
  - 18.81
  - 19.26
  - 0.94
  - 0.44
* - Pinhole from ray-field 3D (images → ray-field 3D → pinhole)
  - 13.55
  - 13.43
  - 0.75
  - 0.85
```

Commentary (Tab. {numref}`tab-rayfield3d-posthoc-pinhole`): the distortion-field error (measured in pixel space) is reduced by about **28%** on the left camera (18.81 → 13.55) and about **30%** on the right camera (19.26 → 13.43) compared to the direct OpenCV pinhole calibration. This suggests that, even on synthetic pinhole data, reconstructing a geometrically consistent 3D first (ray-based) can improve the *identification* of a conventional pinhole + Brown model when compared to fitting pinhole parameters directly from noisy 2D detections.

Notes:

- The “dist err (% of GT)” is computed in pixel space via distortion-displacement vectors on sampled circles (see `pinhole_vs_gt.*.distortion_displacement_vs_gt` in the script JSON).
- The post-hoc reprojection RMS reported under `pinhole_from_rayfield3d.reprojection_error_*` is a **self-consistency** metric on the same correspondences used to reconstruct the 3D points, and should not be interpreted as a standalone accuracy guarantee.

## Usage after identification (robotics / stereo-DIC)

This section clarifies what is required and what it costs to use a ray-field model **once calibrated**, i.e., “after identification” of 2D correspondences (ChArUco, dense stereo matching, optical flow, DIC correlation, etc.).

### Minimal inputs/outputs

To reconstruct a 3D field from a stereo pair (left/right), one needs:

- **Stereo model (calibration)**:
  - rig $(R_{RL},t_{RL})$,
  - left ray-field $d_L(u,v)$ and right ray-field $d_R(u,v)$ (Zernike coefficients, central model).
- **2D correspondences**:
  - either pairs $(u_L,v_L)\leftrightarrow(u_R,v_R)$,
  - or a disparity map $d(u,v)$ on a rectified image (classic robotics case).

Outputs:

- a point cloud (or field) $\hat P$ in mm in the left-camera frame,
- optionally a per-point quality metric (skew-ray distance).

### Per-point computation and algorithmic cost

For each correspondence:

1. **Pixel → ray** (left and right):
   - pinhole: normalization + (un)distortion + normalization → $\hat d$,
   - ray-field: evaluate $x(u,v),y(u,v)$ (Zernike), then $\hat d=\mathrm{normalize}([x,y,1])$.
2. **Triangulation** (least-squares intersection):
   - midpoint of the common perpendicular segment (a few vector operations).

In terms of complexity for $N$ correspondences:

- pinhole: $\mathcal{O}(N)$ (small constant cost),
- Zernike ray-field: $\mathcal{O}(N\,K)$ if evaluating $K$ modes explicitly (e.g., $K=45$ for `nmax=8`), plus $\mathcal{O}(N)$ for triangulation.

In practice, **runtime can be brought to the same order as pinhole** by precomputing a ray-direction map:

- precompute once: $d(u,v)$ for all image pixels (amortized cost),
- real-time: lookup $d$ + triangulation → $\mathcal{O}(N)$.

This precompute can store a $(H\times W\times 3)$ `float32` array (a few MiB), which is typically acceptable in robotics.

### Real-time pipeline (robotics)

For dense stereo in robotics (depth map), a realistic pipeline is:

1. precompute $d_L(u,v)$ and $d_R(u,v)$ (ray directions) over the image grid,
2. compute correspondences (stereo matching):
   - either by rectifying to a virtual (pinhole) camera then using standard disparity,
   - or directly on non-rectified images using a more general matcher,
3. triangulate point-by-point and output depth / point cloud.

If rectifying to a virtual camera, the additional step compared to pinhole is building remap tables once, then running `cv2.remap` (real-time, optimized).

### Two-time pipeline (stereo-DIC)

With two stereo pairs (reference + deformed), a 3D displacement field can be obtained by:

1. identifying stereo correspondences at $t_0$ and $t_1$ (or tracking points between $t_0\to t_1$),
2. triangulating $\hat P(t_0)$ and $\hat P(t_1)$ with the same stereo model,
3. computing $\Delta \hat P = \hat P(t_1)-\hat P(t_0)$.

Again, the ray-field overhead relative to pinhole is concentrated in pixel→ray evaluation; with a precomputed map, reconstruction remains compatible with high frame rates.

### Model size (“parameter complexity”)

A central Zernike ray-field remains compact:

- per camera: $2K$ coefficients (for $x$ and $y$), e.g. $2\times 45=90$ scalars at `nmax=8`,
- stereo: +6 rig parameters $(R_{RL},t_{RL})$.

This is on the same order of magnitude as a pinhole model (focal length, principal point, distortion), but the representation is more flexible (it does not enforce a particular polynomial distortion form).

### Code references

- Central stereo point↔ray BA: `src/stereocomplex/ray3d/central_stereo_ba.py`
- Experimental driver (images → BA): `paper/experiments/calibrate_central_rayfield3d_from_images.py`
