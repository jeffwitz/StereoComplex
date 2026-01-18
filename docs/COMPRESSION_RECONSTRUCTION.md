# Image compression and 3D reconstruction

This page studies how image compression impacts:

1. **ChArUco-based OpenCV stereo calibration**, and
2. the **ray-based 3D reconstruction** pipeline (*central 3D ray-field + point↔ray BA*).

The main goal is practical: determine whether a ray-based reconstruction remains usable when images are **strongly compressed** (for storage, telemetry, or embedded/robotics pipelines).

## Experimental design

We reuse the exact same scene stored with different codecs/qualities via `stereocomplex sweep-compression`.

We compare three pipelines:

1. **OpenCV pinhole (raw)**: calibrate from raw OpenCV ChArUco detections.
2. **OpenCV pinhole (+ 2D ray-field)**: refine corners with `rayfield_tps_robust`, then calibrate with OpenCV.
3. **3D ray-field BA (+ 2D ray-field)**: refine corners with `rayfield_tps_robust`, then calibrate a central 3D ray-field via point↔ray bundle adjustment and triangulate.

### Metrics

- **Stereo baseline error in pixels**: reported as `baseline_abs_error_px_at_mean_depth`.
  This is a disparity-equivalent measure and is the most interpretable unit for stereo matching quality.
- **Stereo RMS reprojection error (px)**: OpenCV-only metric (low is better).
- **Triangulation RMS (% mean depth)**:
  - For OpenCV pinhole: direct RMS error vs GT (absolute scale is defined).
  - For 3D ray-field: we report a **similarity-aligned** 3D error, because the ray-based model has a weak gauge (global similarity drift) in this prototype.

## WebP quality sweep

Command (uses cached intermediate JSON files; does not regenerate datasets):

```bash
.venv/bin/python paper/experiments/sweep_compression_3d_methods.py \
  --root dataset/compression_sweep \
  --png png_lossless \
  --webp webp_q70,webp_q80,webp_q90,webp_q95 \
  --split train --scene scene_0000 \
  --out paper/tables/compression_compare/sweep_webp_quality.json \
  --plots-out docs/assets/compression_sweep
```

### Baseline error (px @ mean depth)

This figure is the key robotics/stereo-DIC indicator: it directly relates to vertical/horizontal consistency after rectification and to the stability of epipolar geometry.

![](assets/compression_sweep/baseline_abs_error_px_at_mean_depth.png)

### Stereo RMS reprojection (OpenCV)

This is the OpenCV objective. It can decrease even when some physical parameters drift, because the optimizer can trade off intrinsics/distortion/pose to reduce pixel residuals.

![](assets/compression_sweep/stereo_rms_px.png)

### Triangulation error (RMS, % mean depth)

This plot illustrates 3D robustness under compression. For the ray-field pipeline, the error is reported after similarity alignment (see Metrics).

![](assets/compression_sweep/tri_rms_rel_depth_percent.png)

## Discussion: why can compression sometimes “help”?

It is not intuitive, but it is plausible to observe that **moderate lossy compression can improve some metrics** in a ChArUco-based pipeline, for at least three reasons:

1. **Implicit low-pass filtering**: codecs often suppress high-frequency components. When the limiting factor is corner localization jitter caused by aliasing/noise/over-sharpened edges, mild smoothing can reduce bias and outliers.
2. **Detection non-linearities and selection effects**: the set of detected markers/corners can change across codecs (some frames may fail, some points may be rejected). This changes the calibration problem itself and can shift the solution (sometimes for the better, sometimes for the worse).
3. **Model mismatch compensation**: OpenCV calibration may reach a different local optimum depending on the outlier pattern. A change in compression can change the outlier pattern, which can make the pose/baseline estimation appear “better” even if the image is objectively worse.

For these reasons, compression quality sweeps are essential before concluding about robustness.

In contrast, the ray-field approach aims to reduce sensitivity to these effects by:

- using a geometric prior on the board plane (2D ray-field),
- and using a ray-based representation for 3D (instead of forcing a global pinhole model).

