# Tutorial 4 — First non‑central rayfield (10 minutes)

Tutorial 3 gave each pixel its own ray *direction* but kept a single camera
centre.  Now add per‑pixel ray **origins** — the defining feature of a
non‑central camera.  This is what you need when a protective glass, inclined
window, or microscope optics breaks the single‑viewpoint assumption.

We use a **synthetic oracle** so you can validate the result against known
ground truth.

**Prerequisite:** Tutorial 3 completed.

## Step 1 — Generate a synthetic parallel‑plate dataset

```python
import numpy as np
from stereocomplex.synthetic import generate_parallel_plate_stereo_dataset
from stereocomplex.synthetic import ParallelPlateSyntheticParams, make_transform
from stereocomplex import CharucoBoardSpec

board = CharucoBoardSpec(squares_x=5, squares_y=7, square_size_mm=20, marker_size_mm=15)
obj = board.object_points.astype(np.float64)

K = np.array([[3200, 0, 320], [0, 3200, 240], [0, 0, 1]], dtype=np.float64)

# 8 random board poses
rng = np.random.default_rng(42)
board_poses = [make_transform(rvec, tvec) for rvec, tvec in zip(
    rng.uniform(-0.3, 0.3, (8, 3)),
    rng.uniform(-30, 30, (8, 3))
)]

plate_left = ParallelPlateSyntheticParams(
    eta=1.5, thickness=16.0, alpha_deg=13.0, beta_deg=5.0, d1=70.0
)
plate_right = ParallelPlateSyntheticParams(
    eta=1.5, thickness=14.0, alpha_deg=10.0, beta_deg=7.0, d1=75.0
)

dataset = generate_parallel_plate_stereo_dataset(
    object_points=obj, board_poses=board_poses,
    K_left=K, K_right=K,
    T_left_world=np.eye(4), T_right_world=np.eye(4),
    plate_left=plate_left, plate_right=plate_right,
    image_size=(640, 480), noise_std_px=0.05,
)
```

## Step 2 — Fit a Zernike origin field

```python
from stereocomplex.calibration import fit_stereo_zernike_origin_field

fit_result = fit_stereo_zernike_origin_field(
    observations=dataset,
    K_left=K, K_right=K,
    T_right_left_initial=np.eye(4),
    max_order=4,
)
```

## Step 3 — Reconstruct and compare against ground truth

```python
from stereocomplex.metrics import reconstruct_points_with_origin_fields, compare_3d_reconstruction_with_without_origin_field

comparison = compare_3d_reconstruction_with_without_origin_field(
    dataset=dataset,
    central_model_result=rayfield_result,      # from Tutorial 3
    origin_field_result=fit_result,
)

print(f"Central RMS:     {comparison.central.rms_mm:.4f} mm")
print(f"Non‑central RMS: {comparison.noncentral.rms_mm:.4f} mm")
```

The non‑central reconstruction should be measurably better — the parallel
plate shifts ray origins, and the central model cannot capture that.

## Where to go next

- [Explanation: Central vs non‑central](../explanation/central_vs_noncentral) — when the pinhole assumption fails.
- [Explanation: Gauge choices](../explanation/gauge_choices) — how origin fields are regularised.
- Notebook `04_parallel_plate_origin_field` in `examples/notebooks/` — the full interactive version.
