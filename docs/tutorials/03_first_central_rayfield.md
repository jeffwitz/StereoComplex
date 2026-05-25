# Tutorial 3 — Your first central rayfield (10 minutes)

Tutorials 1–2 used the classical OpenCV pinhole model.  Now replace it
with a **central rayfield**: every pixel gets its own 3-D ray direction
(but still a single camera centre).  This is the first step toward
handling non‑pinhole optics.

**Prerequisite:** Tutorial 2 completed (you have `result_ray2d`).

## Step 1 — Fit the rayfield

```python
from stereocomplex.api import fit_stereo_central_rayfield_from_dataset

rayfield_result = fit_stereo_central_rayfield_from_dataset(
    dataset_root="dataset/v0_png",
    split="train",
    scene="scene_0000",
    board=board,
    max_frames=20,
    method2d="rayfield_tps_robust",
    nmax=6,              # Zernike radial order
)
```

## Step 2 — Triangulate a point

```python
# Take one stereo pair of corners
uv_left  = rayfield_result.observations.left_pixels[0]   # (N,2) → first row
uv_right = rayfield_result.observations.right_pixels[0]

from stereocomplex.metrics import reconstruct_points_central_stereo

rec = reconstruct_points_central_stereo(
    left_pixels=uv_left,
    right_pixels=uv_right,
    model=rayfield_result.model,
)
print(f"3-D point: {rec.points_3d_mm}")    # in millimetres
print(f"Ray gap:   {rec.ray_gap_mm:.4f}")   # should be < 0.01 mm
```

## Step 3 — What changed

| Tutorial | Model | Rays per pixel |
|---|---|---|
| 1–2 | OpenCV pinhole | 1 matrix × pixel → direction |
| 3 | Central rayfield | 1 Zernike polynomial × pixel → direction |

The rayfield captures residual distortion that the 5‑parameter Brown model
misses, while still keeping a single camera centre.

## Where to go next

- [Tutorial 4 — First non‑central rayfield](04_first_noncentral_rayfield) — add per‑pixel ray origins.
- [Explanation: Central vs non-central](../explanation/central_vs_noncentral) — when the pinhole assumption fails.
