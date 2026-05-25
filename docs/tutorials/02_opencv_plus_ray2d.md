# Tutorial 2 — Improve with Ray2D (5 minutes)

In Tutorial 1 you got a standard OpenCV calibration.  Now re‑run the
same images with **Ray2D corner refinement** — a sub‑pixel correction that
reduces reprojection error without changing the camera model.

**Prerequisite:** Tutorial 1 completed (you have `board` defined).

## Step 1 — Refine corners

```python
from stereocomplex.api import fit_opencv_stereo_from_dataset

result_ray2d = fit_opencv_stereo_from_dataset(
    dataset_root="dataset/v0_png",
    split="train",
    scene="scene_0000",
    board=board,
    max_frames=20,
    method2d="rayfield_tps_robust",   # ← the only change from Tutorial 1
)
```

## Step 2 — Compare

```python
from stereocomplex.api import assess_calibration

print("OpenCV raw:", assess_calibration(result).rms_px)
print("Ray2D:     ", assess_calibration(result_ray2d).rms_px)
```

You should see a visible improvement (typically 0.3 → 0.15 px on this
dataset).

## Step 3 — What Ray2D does (in one sentence)

Ray2D fits a smooth 2‑D warp on the board plane, correcting localisation
errors introduced by blur, compression, or imperfect corner detectors.
It is NOT a 3‑D camera model — it's a 2‑D preprocessing step applied
*before* the standard OpenCV stereo calibration.

## Where to go next

- [Tutorial 3 — Your first central rayfield](03_first_central_rayfield) — move to a 3‑D ray‑based model.
- [Explanation: Ray2D vs 3D](../explanation/ray2d_vs_3d) — why 2‑D refinement is not a camera model.
