# Robustness sweep (board size, focal length, aberrations)

This page answers a practical question: *are the observed gains from the 2D ray-field denoiser stable when we change the physical scale of the board, the focal length, and the image degradations?*

We run a controlled synthetic sweep where we vary:

- **board physical size** (small / medium / large),
- **focal length** computed to keep a comparable **framing** (board occupies a similar image fraction),
- **aberration level** (blur + geometric distortion + noise).

The evaluation pipeline is the same as in the stereo documentation: we calibrate OpenCV mono + stereo on `raw` vs `rayfield_tps_robust` corners, then report stereo/3D metrics against GT.

## Sweep design

We use a fixed image size (640×480) and fix the pixel pitch to `pitch_um=3.45` for comparability. For each board size, we set a nominal working distance `Z` and choose a focal length `f` such that the board width occupies a constant fraction of the sensor width.

Focal selection (pinhole geometry):

```{math}
f = \frac{2\,\alpha\,s\,Z}{W_{\mathrm{board}}},
```

where:

- $\alpha\in(0,1)$ is the desired half-width framing fraction,
- $s$ is the sensor half-width (mm) derived from `pitch_um` and image width,
- $Z$ is the working distance (mm),
- $W_{\mathrm{board}}$ is the board width (mm).

Aberration levels are defined as:

- `low`: no distortion, no blur, low noise,
- `medium`: Brown distortion + moderate blur + moderate noise,
- `high`: stronger Brown distortion + stronger blur + higher noise.

## Results summary (18 cases)

We ran 3 board sizes × 3 aberration levels × 2 random seeds = 18 cases and aggregated the results.

Overall improvement ratios (raw / ray-field), across the 18 cases:

- Mono RMS (left): p50 ≈ **5.74×** (p05 ≈ 4.36×, p95 ≈ 7.94×)
- Mono RMS (right): p50 ≈ **5.40×** (p05 ≈ 4.14×, p95 ≈ 7.09×)
- Stereo RMS: p50 ≈ **4.70×** (p05 ≈ 2.78×, p95 ≈ 6.21×)
- Triangulation RMS (mm): p50 ≈ **1.98×** (p05 ≈ 1.21×, p95 ≈ 3.48×)

Baseline error in pixels ($|\Delta d|$) also improves on average but is more variable: some cases have a very small baseline error already with `raw`, so the ratio can be <1 without contradicting the strong reduction in 2D RMS.

Per-scenario means (2 seeds each):

```{list-table} Robustness sweep summary (mean over 2 seeds).
:name: tab-robustness-sweep-summary
:header-rows: 1

* - Scenario
  - Mono RMS L (raw→rf) (px)
  - Stereo RMS (raw→rf) (px)
  - Triang RMS (raw→rf) (mm)
* - small_low
  - 0.274 → 0.057
  - 0.314 → 0.134
  - 5.18 → 2.44
* - small_medium
  - 0.281 → 0.051
  - 0.297 → 0.083
  - 2.94 → 1.13
* - small_high
  - 0.276 → 0.055
  - 0.285 → 0.079
  - 3.58 → 1.51
* - medium_low
  - 0.388 → 0.058
  - 0.391 → 0.067
  - 10.37 → 5.34
* - medium_medium
  - 0.395 → 0.056
  - 0.380 → 0.063
  - 7.42 → 4.04
* - medium_high
  - 0.408 → 0.055
  - 0.382 → 0.064
  - 7.33 → 2.66
* - large_low
  - 0.351 → 0.075
  - 0.361 → 0.092
  - 34.18 → 21.72
* - large_medium
  - 0.365 → 0.064
  - 0.367 → 0.072
  - 35.33 → 20.52
* - large_high
  - 0.373 → 0.065
  - 0.372 → 0.076
  - 34.22 → 18.96
```

## Practical conclusion

On these synthetic datasets, the 2D ray-field denoiser produces a large and consistent reduction of the 2D measurement noise. This propagates mechanically to OpenCV stereo calibration and improves 3D triangulation across:

- very different **physical scales** (tens of mm to ~1 m boards),
- different **focals** (set to keep comparable framing),
- increasing **degradations** (blur, distortion, noise).

## Pinhole identification from ray-field 3D reconstruction (status)

The “post-hoc pinhole identification” results (ray-field 3D → pinhole) are documented in the 3D ray-field page (Tab. {numref}`tab-rayfield3d-posthoc-pinhole` in `RAYFIELD3D_RECONSTRUCTION.md`). Extending this part to the full sweep is implemented (see the script below) but is significantly more expensive than the OpenCV-only sweep.

## Reproduce

Run the sweep (writes per-case JSON + `summary.json`):

```bash
.venv/bin/python paper/experiments/sweep_robustness_board_focal_aberrations.py --seeds 0,1 --frames 16
```

Results are stored in:

- `paper/tables/robustness_sweep/*.json` (per-case reports),
- `paper/tables/robustness_sweep/summary.json` (aggregated).

Optional (slow): also run the ray-field 3D BA + post-hoc pinhole identification for each case:

```bash
.venv/bin/python paper/experiments/sweep_robustness_board_focal_aberrations.py --seeds 0,1 --frames 16 --run-rayfield3d
```

Code: `paper/experiments/sweep_robustness_board_focal_aberrations.py`.

To force recomputing everything (overwriting existing reports), pass `--rerun`.
