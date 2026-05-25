# Assess calibration quality

Use `assess_calibration` to get a quick diagnostic of any calibration result.

```python
from stereocomplex.api import assess_calibration

report = assess_calibration(result)
print(report)
# → ok / warning / failed
# → per‑channel RMS, skew, point‑to‑ray distance, N points, N frames
```

The report includes:

| Field | Unit | What it tells you |
|---|---|---|
| `verdict` | — | `ok`, `warning`, or `failed` |
| `rms_px` | px | Mean reprojection error per channel |
| `skew_deg` | ° | Skew angle (should be ≈ 0) |
| `point_to_ray_rms_mm` | mm | Ray‑to‑point distance after triangulation |
| `n_points_total` | — | Total corners used in calibration |
| `n_frames` | — | Number of frames used |

If the verdict is `warning` or `failed`, see [Fix my calibration](fix_my_calibration).
