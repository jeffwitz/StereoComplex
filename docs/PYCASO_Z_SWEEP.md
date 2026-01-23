# Pycaso-style depth sweep vs 3D ray-field

The Pycaso workflow is reimplemented here end-to-end so you can reproduce the **Soloff polynomial calibration** and compare it directly to the 3D ray-field reconstruction in the same synthetic setting.

## Soloff direct vs Soloff LM (LM identification)

Pycaso originally distinguishes the **direct polynomial** (Eq. 2.1 of the paper) from the **Soloff polynomial** (Eq. 2.4/2.5). The direct variant fits three polynomials that map `(uL,vL,uR,vR) → (X,Y,Z)` and solves in one shot via `lstsq`. The Soloff variant fits the reverse polynomials `S(X,Y,Z) → (C,R)` and then inverts them numerically.

Our implementation of the Soloff pathway (`PycasoSoloffStereoModel`) goes one step further: after fitting the `S` polynomials we run a **Levenberg–Marquardt correction** initialized with the degree‑1 affine approximation. We refer to this as **“Soloff LM”**. The LM step improves stability when the polynomial basis becomes large and the geometry becomes over‑parameterized, even though both variants end up with similar RMS Z on this sweep (see the results table below).

## Dataset: board slides in Z only

We use the CPU renderer to produce `dataset/pycaso_z_sweep/train/scene_0000/` with the following properties:

* Two camerassharing the same sensor grid (pitch 4.88 µm) and focal length 5937.56 µm.
* `--z-only-mode`: the board’s X/Y translation and tilts are fixed while Z follows the schedule 2.96 mm … 3.04 mm repeated to cover 45 frames.
* Aberrations: Brown distortion (strength 0.8), Gaussian blur FWHM 6 µm, additive noise STD 0.01.
* Texture interpolation is `lanczos4` and the output is PNG so we isolate the effect of compression-free aberrations.

Generate the dataset with:

```bash
.venv/bin/python -m stereocomplex.cli generate-cpu-dataset --out dataset/pycaso_z_sweep \
    --frames-per-scene 45 --width 800 --height 600 --pattern charuco --tex-interp lanczos4 \
    --distort brown --distort-strength 0.8 --blur-fwhm-um 6.0 --noise-std 0.01 \
    --pitch-um 4.8799945 --f-um 5937.5567 --baseline-mm 0.5 \
    --tz-schedule-mm 2.96,2.97,2.98,2.99,3.00,3.01,3.02,3.03,3.04 \
    --z-only-mode
```

Repeat the nine-depth schedule above five times if you need all 45 frames. The metadata file flags `"sim_params": {"z_only_mode": true}` so you can validate that only the Z coordinate changed.

Inspect the renders in `dataset/pycaso_z_sweep/train/scene_0000/{left,right}/`. The planner also writes `gt_charuco_corners.npz` containing the GT 3D locations for every detection.

## Benchmark protocol

We reuse `paper/experiments/sweep_z_compare_pycaso.py`. For each frame:

1. Detect the ChArUco corners in left/right (OpenCV), shift to the dataset’s pixel-corner convention, and backmap to the GT board points using `corner_id`.
2. Train:
   * `SoloffPolynomialModel` (direct polynomial from `(uL,vL,uR,vR)` to `(X,Y,Z)`),
   * `PycasoSoloffStereoModel` (Soloff polynomials `XYZ→pixels` + Levenberg–Marquardt inversion).
3. Run OpenCV pinhole calibration/triangulation on the same detections.
4. Fit the central 3D ray-field coefficients with `fit_central_stereo_rayfield_coeffs_fixed` using the **known rig and ground-truth per-frame poses** (synthetic control).

Execute the benchmark with:

```bash
.venv/bin/python paper/experiments/sweep_z_compare_pycaso.py dataset/pycaso_z_sweep \
    --use-detections \
    --out-json validation/z_sweep/pycaso_z_sweep_metrics.json \
    --out-plot validation/z_sweep/pycaso_z_sweep_rms_z.png
```

The script writes per-method RMS statistics, skew, and depth-binned RMSE to the JSON file, and the PNG overlays each method’s RMS Z error as a function of the true depth.

## Impact of `rayfield_tps_robust` pre-refinement

Before running the benchmark, you can refine the detections with `rayfield_tps_robust`:

```bash
.venv/bin/python -m stereocomplex.cli refine-corners dataset/pycaso_z_sweep \
    --split train --scene scene_0000 \
    --method rayfield_tps_robust --tps-lam 10 --tps-huber 3 --tps-iters 3 \
    --out-json validation/z_sweep/pycaso_z_sweep_refined.json \
    --out-npz validation/z_sweep/pycaso_z_sweep_refined.npz
```

Then rerun the sweep with `--use-refined`:

```bash
.venv/bin/python paper/experiments/sweep_z_compare_pycaso.py dataset/pycaso_z_sweep \
    --use-refined --use-detections \
    --out-json validation/z_sweep/pycaso_z_sweep_metrics_refined.json \
    --out-plot validation/z_sweep/pycaso_z_sweep_rms_z_refined.png
```

| Method | Raw RMS Z | Refined RMS Z | Ray skew (refined P95) |
| --- | --- | --- | --- |
| Pycaso direct polynomial | 0.00431 mm | 0.00147 mm | – |
| Pycaso Soloff LM | 0.00437 mm | 0.00155 mm | – |
| Central 3D ray-field (fixed poses) | 0.00448 mm | 0.00156 mm | 0.00139 mm |

Applying `rayfield_tps_robust` before calibration reduces the depth RMS by about a factor of three across all methods, because the TPS-based refinement removes local distortions and compression-like aliasing from the detected corners. This also helps the ray-field bundle adjustment even though its rig and poses stay fixed: less noisy 2D correspondences mean tighter ray skew and more precise triangulation.

## Results: Pycaso vs 3D ray-field

The table below uses the `validation/z_sweep/pycaso_z_sweep_metrics_refined.json` output (depth range ≈ 2.96 mm … 3.04 mm, mean ≈ 3.00 mm) generated *after* feeding the detections through `rayfield_tps_robust`. It highlights the refined RMS along Z (absolute, relative, xyz breakdown) plus the 95 % ray skew for the ray-field.

| Method | RMS Z (mm) | RMS Z (% of depth ≈ 3 mm) | RMS XYZ (mm) | Ray skew (P95, mm) |
| --- | --- | --- | --- | --- |
| OpenCV pinhole (refined detections, GT prior) | 0.00719 | 0.24 % | 0.00249 / 0.00244 / 0.00719 | – |
| Pycaso direct polynomial (refined) | 0.00147 | 0.049 % | 0.00021 / 0.00013 / 0.00147 | – |
| Pycaso Soloff + LM (refined) | 0.00155 | 0.052 % | 0.00019 / 0.00013 / 0.00155 | – |
| Central 3D ray-field (fixed poses, refined) | 0.00156 | 0.052 % | 0.00022 / 0.00014 / 0.00156 | 0.00045 |

Despite the GT prior, the OpenCV baseline still converges to an order of magnitude worse depth error than the Pycaso/ray-field reconstructions, which all cluster at ≈0.05 % RMS Z. The ray-field remains on par with Pycaso while also providing a geometric interpretation via ray skew and triangulation without relying on a pinhole `K`.

## Compression stress test (lossy JPEG/WebP)

The same scripts can also stress-test Pycaso vs the 3D ray-field under severe lossy compression on a **different benchmark**: `dataset/compression_sweep/*` (mean depth ≈ 1.2 m, varying poses).
This is **not** the millimetric Z-only sweep above: it is a planar calibration scene designed to probe **compression robustness** of corner identification and stereo geometry.

We reran `sweep_z_compare_pycaso.py` with `--use-refined` on `dataset/compression_sweep/{png_lossless,webp_q70,jpeg_q80}` (TPS refinement enabled). The polynomial mappings can become numerically unstable in this pose-sweep setting (even on lossless PNG), while the ray-field reconstruction stays near ~1–2 mm RMS Z. This stress test therefore illustrates robustness of the ray-based geometry under photometric degradation rather than “best-case Pycaso” performance.

| Dataset | Pycaso direct RMS Z (mm) | Pycaso Soloff+LM RMS Z (mm) | Central 3D ray-field RMS Z (mm) |
| --- | --- | --- | --- |
| PNG lossless | 18.12 | 162.65 | 1.31 |
| WebP q70 | 11.60 | 162.79 | 1.13 |
| JPEG q80 | 31.15 | 161.90 | 1.23 |

The JSON outputs for these runs live in `validation/compression/webp_q70_pycaso_metrics.json` and `validation/compression/jpeg_q80_pycaso_metrics.json`, with plots under `validation/compression/webp_q70_pycaso_rms_z.png` / `.../jpeg_q80_pycaso_rms_z.png`. This benchmark is therefore the natural “Pycaso × compression” companion to the Z-sweep section above.

## Pycaso example images (real): OpenCV ChArUco configuration

The Pycaso repository also ships real images under:

- `/home/jeff/Code/Pycaso/Exemple/Images_example/left_calibration*`
- `/home/jeff/Code/Pycaso/Exemple/Images_example/right_calibration*`

The filename (e.g. `3.0.png`) encodes the nominal stage depth in millimeters.

### Board parameters

From the Pycaso code (`pycaso/pattern.py` and `pycaso/data_library.py`):

- ArUco dictionary: `DICT_6X6_250` for detection (Pycaso generates markers from `DICT_6X6_1000`, but IDs are `0..95`, so `DICT_6X6_250` is equivalent).
- `squares_x=16`, `squares_y=12`
- `square_size_mm=0.3`, `marker_size_mm=0.15`

### OpenCV ≥ 4.7 (CharucoDetector) note

The Pycaso board places marker IDs on the opposite square parity compared to OpenCV's default `CharucoBoard` generator. With OpenCV's `CharucoDetector`, this means `checkMarkers` must be disabled to interpolate ChArUco corners:

```python
import cv2.aruco as aruco

det = aruco.CharucoDetector(board)
cp = det.getCharucoParameters()
cp.checkMarkers = False
det.setCharucoParameters(cp)
```

The exact settings used in this repo (including `adaptiveThreshWinSizeMax=300`) are collected in `validation/pycaso_images_example_opencv.json`.

## Artefacts

- Dataset images and JSON/frames: `dataset/pycaso_z_sweep/train/scene_0000/`
- Benchmark output: `validation/z_sweep/pycaso_z_sweep_metrics.json` and `validation/z_sweep/pycaso_z_sweep_rms_z.png`
- If you want to compare different aberrations, regenerate the dataset with new `--tz-schedule-mm` values (still with `--z-only-mode`) and rerun the script.
