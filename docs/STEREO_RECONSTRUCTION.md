# Stereo 3D reconstruction (OpenCV) and the impact of the ray-field

Goal: show how the **2D** improvement (ChArUco corner localization) translates into an improvement of **stereo calibration** and **3D triangulation** with “classic” OpenCV tools.

This page is intentionally separate from `docs/RAYFIELD_WORKED_EXAMPLE.md`: it focuses on the “traditional calibration + 3D reconstruction” pipeline.

## Why is this surprising on a *pinhole* dataset?

Even if images are generated from a pinhole model (with Brown distortion), the 2D measurements are not “perfect”:

- blur (including edge blur), texture interpolation, sensor noise,
- optional compression/quantization depending on the dataset,
- ArUco/ChArUco detection outliers.

In this regime, OpenCV calibration is often limited by **2D localization quality** (more than by the projection model itself).
The ray-field acts as a **geometric denoiser** on the board plane: OpenCV is fed with more coherent 2D observations.

## Evaluated pipeline

For each frame (left/right pair):

1) Detect ArUco markers (marker corners).
2) Build two variants of ChArUco corners passed to OpenCV:
   - `raw`: OpenCV “raw” ChArUco corners,
   - `rayfield_tps_robust`: corners predicted by `H + TPS (λ) + IRLS (Huber)`.
3) Monocular calibration: `cv2.calibrateCamera` (left, then right).
4) Stereo calibration: `cv2.stereoCalibrate` with fixed intrinsics, estimating a **single** $(R,T)$ over **all** selected pairs.
5) Triangulation: `cv2.triangulatePoints` after `cv2.undistortPoints`.
6) Compare to the dataset 3D ground truth (`XYZ_world_mm` in `gt_charuco_corners.npz`).

### Used views (important)

The stereo rig $(R,T)$ is not estimated from a single pair: OpenCV minimizes the error over a **list of views** (a view = a left/right pair with enough corners).
The exported JSON contains:

- `n_views_left`, `n_views_right`: number of monocular views used by `calibrateCamera`,
- `n_views_stereo`: number of stereo views used by `stereoCalibrate`,
- `view_stats.*.frame_ids`: which `frame_id` actually contributed,
- `view_stats.*.n_corners`: number-of-corners statistics per view (mean/p50/p95/min/max).

## “Baseline error” in pixels (disparity-equivalent)

The baseline is in mm, but its error can be expressed as an equivalent disparity error (px) at depth $Z$:

```{math}
:label: eq-baseline-px
\Delta d\;(\mathrm{px}) \approx \frac{f_x\;(\mathrm{px})\;\Delta B\;(\mathrm{mm})}{Z\;(\mathrm{mm})}.
```

In the results below, we report a summary of $|\Delta d|$ over GT points (RMS/P95), which provides a more intuitive “image-domain” unit.

## Reproducible script

The script:

- compares `raw` vs `rayfield_tps_robust`,
- produces calibration and triangulation metrics,
- also compares against the GT baseline (if present in `meta.json`).

Command:

```bash
PYTHONPATH=src .venv/bin/python paper/experiments/compare_opencv_calibration_rayfield.py dataset/v0_png \
  --split train --scene scene_0000 \
  --out docs/assets/stereo_reconstruction_example/scene_0000_calib.json
```

Output: `docs/assets/stereo_reconstruction_example/scene_0000_calib.json`.

## Results (example)

Extract (scene_0000, split `train`):

```{list-table} Stereo calibration and triangulation summary (scene_0000, train).
:name: tab-stereo-calib-example
:header-rows: 1

* - 2D method
  - Mono RMS L (px)
  - Mono RMS R (px)
  - Stereo RMS (px)
  - Baseline $\Delta B$ (mm)
  - Baseline $|\Delta d|$ RMS (px)
  - Triangulation RMS (mm)
* - raw
  - 0.306
  - 0.302
  - 0.381
  - 0.439
  - 0.424
  - 8.986
* - rayfield\_tps\_robust
  - 0.079
  - 0.061
  - 0.163
  - -0.212
  - 0.205
  - 7.161
```

The main result is Tab. {numref}`tab-stereo-calib-example`: the 2D method only changes the quality of the 2D points provided to OpenCV, and we then observe its impact on stereo calibration and 3D reconstruction.

### Intrinsics and distortion vs GT (%)

On synthetic data, we can also compare the estimated “physical” parameters (focal length and distortion) to ground truth. The script exports:

- `mono.percent_vs_gt.left.K.fx` / `fy`: relative error (%) on $f_x, f_y$,
- relative errors (%) for each coefficient $k_1,k_2,p_1,p_2,k_3$,
- `mono.distortion_displacement_vs_gt.*`: distortion-field comparison in pixels (more robust/interpretable than comparing coefficients directly).

In the example below, the RMS GT distortion displacement is $\approx 1.404\,\mathrm{px}$ (left) and $\approx 0.947\,\mathrm{px}$ (right) on the sampled circles.

```{list-table} Relative errors (%) on focal length and distortion field (scene_0000, train).
:name: tab-mono-percent-example
:header-rows: 1

* - 2D method
  - fx L (%)
  - fy L (%)
  - dist L err (%)
  - dist L err RMS (px)
  - fx R (%)
  - fy R (%)
  - dist R err (%)
  - dist R err RMS (px)
* - raw
  - 0.062
  - 0.010
  - 14.6
  - 0.205
  - 1.672
  - 1.544
  - 15.7
  - 0.149
* - rayfield\_tps\_robust
  - 0.251
  - 0.320
  - 22.6
  - 0.317
  - 0.688
  - 0.690
  - 16.9
  - 0.160
```

Note: these percentages must be interpreted carefully, because OpenCV can trade off “intrinsics vs distortion” while keeping a low reprojection RMS. For reconstruction, Tab. {numref}`tab-stereo-calib-example` (RMS + baseline in px) remains the most direct indicator.

### Rectification: epipolar stability (vertical disparity)

To quantify the impact on a dense-stereo pipeline, the script also computes **post-rectification** metrics from the estimated model $(K_L,d_L,K_R,d_R,R,T)$:

- `vertical_disparity_measured_px`: $|y_L^{rect}-y_R^{rect}|$ on detected points,
- `vertical_disparity_gt_px`: same on GT points (same estimated rectification, hence “model error”),
- `disparity_error_measured_px`: rectified disparity error $|(x_L^{rect}-x_R^{rect})-(x_{L,GT}^{rect}-x_{R,GT}^{rect})|$.

```{list-table} Rectification metrics (scene_0000, train).
:name: tab-rectification-example
:header-rows: 1

* - 2D method
  - |Δy| RMS (px)
  - |Δy| GT RMS (px)
  - |Δd| RMS (px)
  - ray skew RMS (mm)
* - raw
  - 0.379
  - 0.244
  - 0.369
  - 0.400
* - rayfield\_tps\_robust
  - 0.218
  - 0.195
  - 0.138
  - 0.250
```

Tab. {numref}`tab-rectification-example` makes the key trade-off explicit: even if some intrinsics/distortion parameters can drift, the **epipolar coherence** (vertical disparity and disparity error) improves significantly — which is critical for stereo algorithms that assume **row-wise** correspondences.

### Discussion: epipolar stability vs “parameter truth”

With planar targets and a limited number of poses, OpenCV optimization is known to exhibit couplings between:

- intrinsics ($f_x,f_y,c_x,c_y$),
- distortion (e.g., Brown $k_1,k_2,p_1,p_2,k_3$),
- stereo relative pose ($R,T$).

The ray-field only changes the 2D observations, and can therefore shift the optimum towards a solution with more stable **epipolar geometry** (Tab. {numref}`tab-rectification-example`), without necessarily matching the GT Brown model coefficient-by-coefficient.

For reconstruction, the rectified stereo equation

```{math}
:label: eq-stereo-depth
Z = \frac{f_x\,B}{d}
```

shows that a relative error on $f_x$ (or $B$) mainly yields a global scale error on $Z$, whereas rectification errors (vertical disparity) and disparity errors $d$ directly affect matching quality and 3D noise.

## Theory: from baseline to ray intersection

In metric stereo vision (robotics, dense stereo) as well as in metrology (stereo-DIC), it is tempting to think that 3D accuracy depends only on 2D matching quality. In practice, accuracy also depends — and often primarily — on how well the **geometric model** makes the two optical rays associated with corresponding pixels **nearly intersect** in 3D.

### 1) Two 3D rays associated with a 2D correspondence

Let a 2D correspondence be $\mathbf u_L=(u_L,v_L)$ in the left image and $\mathbf u_R=(u_R,v_R)$ in the right image.
Define homogeneous coordinates $\tilde{\mathbf u}=(u,v,1)^\top$ and normalized coordinates:

```{math}
:label: eq-normalized-coords
\mathbf x_L \sim \mathbf K_L^{-1}\tilde{\mathbf u}_L,\qquad
\mathbf x_R \sim \mathbf K_R^{-1}\tilde{\mathbf u}_R.
```

In the left-camera frame, a ray can be written as a line:

```{math}
:label: eq-rays
\mathcal D_L(\lambda)=\mathbf C_L+\lambda\,\mathbf d_L,\qquad
\mathcal D_R(\mu)=\mathbf C_R+\mu\,\mathbf d_R,
```

where $\mathbf C_L=(0,0,0)^\top$, $\mathbf d_L$ is the normalized $\mathbf x_L$, and $\mathbf d_R$ is $\mathbf x_R$ expressed in the left frame.
With OpenCV’s `stereoCalibrate` convention ($\mathbf X_R=\mathbf R\,\mathbf X_L+\mathbf T$), the right-camera center in the left frame is:

```{math}
:label: eq-right-center
\mathbf C_R = -\mathbf R^\top \mathbf T.
```

### 2) The practical case: skew lines

In a perfect world, $\mathcal D_L$ and $\mathcal D_R$ intersect exactly at the 3D point $\mathbf X$.
In practice (imperfect calibration, residual 2D noise), the two lines are not intersecting: they are **skew**.

Triangulation algorithms (e.g., `cv2.triangulatePoints`) then choose a best-fit point $\hat{\mathbf X}$, typically by minimizing a reprojection criterion or by finding the point closest to both lines.
A useful geometric quantity is the **minimum distance between the two lines**, which directly measures “how much the rays miss each other”.
For $\mathbf C_L=\mathbf 0$, this distance (per point) can be written:

```{math}
:label: eq-skew-distance
d_{\mathrm{skew}} = \frac{\left|(\mathbf C_R)\cdot(\mathbf d_L\times \mathbf d_R)\right|}{\lVert \mathbf d_L\times \mathbf d_R\rVert}.
```

The script exports this metric in mm: `stereo.ray_skew_distance_mm` (RMS/P95/max). It does not replace a GT error, but it explains *why* a calibration may yield unstable triangulation even if 2D correspondences look plausible.

### 3) Epipolar constraint and the role of the baseline

The ideal condition for a pair $(\mathbf x_L,\mathbf x_R)$ to correspond to a common 3D point under a model $(\mathbf R,\mathbf T)$ is the **epipolar constraint**:

```{math}
:label: eq-epipolar
\mathbf x_R^\top \mathbf E\,\mathbf x_L = 0,
\qquad \mathbf E = [\mathbf T]_{\times}\mathbf R,
```

where $[\mathbf T]_{\times}$ is the skew-symmetric matrix associated with the cross product. For $\mathbf T=(t_x,t_y,t_z)^\top$:

```{math}
:label: eq-cross-matrix
[\mathbf T]_{\times} =
\begin{bmatrix}
0 & -t_z & t_y\\
t_z & 0 & -t_x\\
-t_y & t_x & 0
\end{bmatrix},
\qquad
[\mathbf T]_{\times}\,\mathbf a = \mathbf T \times \mathbf a.
```

An error on the baseline (or rotation) makes $\mathbf E$ inconsistent: observed pairs $(\mathbf x_L,\mathbf x_R)$ no longer satisfy the constraint, which manifests as more skew rays (higher $d_{\mathrm{skew}}$) and less reliable rectification (higher $|\Delta y|$ and disparity errors; see Tab. {numref}`tab-rectification-example`).

### 4) Why this matters for robotics and stereo-DIC

- **Robotics / dense stereo**: rectification assumes nearly horizontal correspondences. Reducing $|\Delta y|$ and the post-rectification disparity error facilitates “row-wise” matching and reduces depth noise.
- **Metrology / stereo-DIC**: even if rectification is sometimes avoided (to limit interpolation), reconstruction still relies on triangulation with $(K,d,R,T)$. Stabilizing epipolar geometry reduces ray inconsistency, hence 3D bias/noise induced by calibration.

### Metric definitions (table columns)

- **2D method**: how 2D corners $(u,v)$ are produced before being passed to OpenCV.
  - `raw`: OpenCV raw ChArUco corners.
  - `rayfield_tps_robust`: corners predicted by `H + TPS (λ) + IRLS (Huber)` from ArUco corners.
- **Mono RMS L (px)**: reprojection RMS (px) returned by `cv2.calibrateCamera` on the left camera using corners from the given method.
- **Mono RMS R (px)**: same for the right camera.
- **Stereo RMS (px)**: reprojection RMS (px) returned by `cv2.stereoCalibrate` (fixed intrinsics), using corners from the given method on left/right pairs.
- **Baseline $\Delta B$ (mm)**: error on the norm of the estimated translation,

  ```{math}
  :label: eq-baseline-mm
  \Delta B = \lVert \mathbf T\rVert - B_{\mathrm{GT}}.
  ```

- **Baseline $|\Delta d|$ RMS (px)**: converts baseline error into an “equivalent disparity” error (px) at GT depths,

  ```{math}
  :label: eq-baseline-px-abs
  |\Delta d| = \left|\frac{f_x\,\Delta B}{Z}\right|,
  ```

  then summarizes $|\Delta d|$ over GT points (RMS/P95/max). This is the most intuitive unit for reconstruction impact.
- **Triangulation RMS (mm)**: RMS 3D error $\lVert \hat{\mathbf X}-\mathbf X_{\mathrm{GT}}\rVert$ (mm) after `cv2.triangulatePoints` (on undistorted points), summarized over all triangulated corners.

To make this value interpretable, the script also exports:

- **depth\_mm**: depth distribution $Z$ (mm) of used GT points (P05/P50/P95),
- **triangulation\_error\_rel\_z\_percent**: relative error $100\,\lVert \hat{\mathbf X}-\mathbf X_{\mathrm{GT}}\rVert/Z$ (RMS/P95/max).

Thus, a “RMS = 7.4 mm” can be read as “$\approx 0.55\%$ at $Z \approx 1.3\,\mathrm{m}$” for this scene.

### Interpreting triangulation (mm) vs working distance

Absolute errors (mm) depend strongly on working distance: for a constant disparity error $\sigma_d$ (px), the classic stereo approximation yields:

```{math}
:label: eq-depth-error
\sigma_Z \approx \frac{Z^2}{f_x\,B}\,\sigma_d
\quad\Longrightarrow\quad
\frac{\sigma_Z}{Z} \approx \frac{Z}{f_x\,B}\,\sigma_d \approx \frac{\sigma_d}{d},
```

where $Z$ is depth, $B$ the baseline, $f_x$ the focal length (px), $d$ the disparity (px).
We therefore also report a relative metric (% of $Z$), which enables comparisons across scenes with different distances.

On the example of Tab. {numref}`tab-stereo-calib-example`:

```{list-table} Depth and normalized 3D error (same example).
:name: tab-stereo-triang-interpret
:header-rows: 1

* - 2D method
  - Depth P50 (mm)
  - Depth [P05, P95] (mm)
  - Triang RMS (%Z)
  - Triang P95 (%Z)
* - raw
  - 1539
  - [909, 1612]
  - 0.615
  - 1.057
* - rayfield\_tps\_robust
  - 1539
  - [909, 1612]
  - 0.518
  - 0.589
```

Tab. {numref}`tab-stereo-triang-interpret` shows that, despite “visually large” mm errors, the relative error is on the order of $10^{-2}$ (percent), and that the ray-field improvement is consistent with the strong drop in baseline error in pixel units.

Note: on synthetic datasets v0, `gt_charuco_corners.npz` provides `XYZ_world_mm`. The script assumes that this 3D is consistent with the triangulation convention (left-camera frame), which holds for `dataset/v0_png/train/scene_0000` (verified by reprojection).

Reading guide:

- The drop in reprojection RMS (mono + stereo) indicates that OpenCV absorbs much less localization error.
- The baseline error in pixels (equivalent disparity) drops significantly, showing that stereo geometry (scale) becomes much more stable.
- 3D triangulation improves as well, but it also depends on the quality of estimated intrinsics/distortion and on pose geometry.

